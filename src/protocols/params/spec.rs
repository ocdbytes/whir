use core::marker::PhantomData;

use ordered_float::OrderedFloat;

use crate::{bits::Bits, engines::EngineId};

/// Phantom-typed newtype — `Tagged<T, A>` and `Tagged<T, B>` are distinct types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tagged<T, Tag>(T, PhantomData<Tag>);

impl<T: Copy, Tag> Tagged<T, Tag> {
    pub const fn new(v: T) -> Self {
        Self(v, PhantomData)
    }

    pub const fn get(self) -> T {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct SecuritySpec {
    pub mode: Mode,
    pub target_security_bits: u32,
    /// Per-slot PoW budget — every grinding slot may close at most this many
    /// bits of gap to `target_security_bits`. Not a cumulative budget across
    /// slots; `check_pow_bits` enforces it per-slot. `None` ⇒ `Some(0)`.
    pub max_pow_bits: Option<u32>,
    pub hash_id: EngineId,
}

impl SecuritySpec {
    pub fn protocol_security_target_bits(&self) -> Bits {
        let pow = self.max_pow_bits.unwrap_or(0);
        Bits::new(f64::from(self.target_security_bits.saturating_sub(pow)))
    }
}

/// Per-round folding strategy. `at_round(i)` returns the factor for round `i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldingFactor {
    /// Same folding factor across all rounds.
    Constant(usize),
    /// `at_round(0) = initial`; `at_round(i) = rest` for `i ≥ 1`.
    ConstantFromSecondRound { initial: usize, rest: usize },
}

impl FoldingFactor {
    pub const fn at_round(&self, round: usize) -> usize {
        match self {
            Self::Constant(f) => *f,
            Self::ConstantFromSecondRound { initial, rest } => {
                if round == 0 {
                    *initial
                } else {
                    *rest
                }
            }
        }
    }

    /// Smallest factor across rounds.
    pub const fn min(&self) -> usize {
        match self {
            Self::Constant(f) => *f,
            Self::ConstantFromSecondRound { initial, rest } => {
                if *initial < *rest {
                    *initial
                } else {
                    *rest
                }
            }
        }
    }
}

/// Proof-size / prover-time / soundness-margin tradeoffs.
#[derive(Debug, Clone)]
pub struct TuningSpec {
    pub vector_size: usize,
    pub starting_log_inv_rate: u32,
    pub folding_factor: FoldingFactor,
}

/// Per-round context handed to a sub-protocol builder.
#[derive(Debug, Clone)]
pub struct RoundContext {
    pub vector_size: usize,
    pub log_inv_rate: u32,
    pub folding_factor: u32,
}

/// Both variants run in the Johnson regime — Construction 9.7's OOD-query
/// requirement makes unique-decoding incompatible with code-switch, so it is
/// not representable here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Standard,
    ZeroKnowledge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OodSampleBudgetTag {}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskCodeMessageLenTag {}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogInvRateTag {}

/// OOD-sample budget (Lemma 9.9 / bounds doc §5.2).
pub type OodSampleBudget = Tagged<usize, OodSampleBudgetTag>;

impl Tagged<usize, OodSampleBudgetTag> {
    /// Sentinel for "no OOD samples". Used by sub-protocols that don't
    /// require an OOD challenge round (e.g. Standard mode, basecase).
    pub const ZERO: Self = Self::new(0);
}

/// C_zk message length (Theorem 9.6: `ℓ_zk ≥ source mask length`).
pub type MaskCodeMessageLen = Tagged<usize, MaskCodeMessageLenTag>;

/// `rate = 2^-log_inv_rate`.
pub type LogInvRate = Tagged<u32, LogInvRateTag>;

/// Reed–Solomon list-decoding ball size `|Λ(C, δ)|`. Wraps `OrderedFloat<f64>`
/// so it can be stored alongside the `Tagged` integer newtypes without losing
/// `Eq`/`Hash`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ListSize(OrderedFloat<f64>);

impl ListSize {
    pub const fn new(v: f64) -> Self {
        Self(OrderedFloat(v))
    }

    pub const fn get(self) -> f64 {
        self.0 .0
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::hash;

    /// Fixture target. 100 is chosen so the expected `target − pow` values in
    /// the tests below are round numbers (80, 40, 0) for readability.
    const TARGET_BITS: u32 = 100;

    fn spec(max_pow_bits: Option<u32>) -> SecuritySpec {
        SecuritySpec {
            mode: Mode::ZeroKnowledge,
            target_security_bits: TARGET_BITS,
            max_pow_bits,
            hash_id: hash::BLAKE3,
        }
    }

    #[test]
    fn none_means_no_pow_credit() {
        assert_eq!(
            spec(None).protocol_security_target_bits(),
            Bits::new(f64::from(TARGET_BITS)),
        );
    }

    #[test]
    fn some_zero_matches_none() {
        assert_eq!(
            spec(Some(0)).protocol_security_target_bits(),
            spec(None).protocol_security_target_bits(),
        );
    }

    #[test]
    fn pow_credit_shifts_analytic_floor() {
        // Two below-target PoW budgets: `target − pow` shifts down 1:1.
        assert_eq!(
            spec(Some(20)).protocol_security_target_bits(),
            Bits::new(80.0),
        );
        assert_eq!(
            spec(Some(60)).protocol_security_target_bits(),
            Bits::new(40.0),
        );
    }

    #[test]
    fn pow_exceeding_target_saturates_to_zero() {
        // `pow > target` saturates rather than going negative.
        let pow_over_target = TARGET_BITS + 100;
        assert_eq!(
            spec(Some(pow_over_target)).protocol_security_target_bits(),
            Bits::new(0.0),
        );
    }
}
