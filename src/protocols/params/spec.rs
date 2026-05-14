use core::marker::PhantomData;

use crate::{algebra::embedding::Embedding, engines::EngineId};

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
pub struct SecuritySpec<M: Embedding> {
    pub mode: Mode,
    pub target_security_bits: u32,
    pub max_pow_bits: Option<u32>,
    pub hash_id: EngineId,
    pub _embedding: PhantomData<M>,
}

impl<M: Embedding> SecuritySpec<M> {
    pub fn protocol_security_target_bits(&self) -> f64 {
        let pow = self.max_pow_bits.unwrap_or(0);
        f64::from(self.target_security_bits.saturating_sub(pow))
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

    /// Smallest factor across rounds; used by `TuningSpec` validation.
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
    pub round_index: usize,
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

/// Bound 2 OOD-sample budget.
pub type OodSampleBudget = Tagged<usize, OodSampleBudgetTag>;

/// C_zk message length (Theorem 9.6: `ℓ_zk ≥ source mask length`).
pub type MaskCodeMessageLen = Tagged<usize, MaskCodeMessageLenTag>;

/// `rate = 2^-log_inv_rate`.
pub type LogInvRate = Tagged<u32, LogInvRateTag>;

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::{
        algebra::{embedding::Identity, fields::Field64},
        hash,
    };

    fn spec(max_pow_bits: Option<u32>) -> SecuritySpec<Identity<Field64>> {
        SecuritySpec {
            mode: Mode::ZeroKnowledge,
            target_security_bits: 100,
            max_pow_bits,
            hash_id: hash::BLAKE3,
            _embedding: PhantomData,
        }
    }

    #[test]
    fn none_means_no_pow_credit() {
        assert_eq!(spec(None).protocol_security_target_bits(), 100.0);
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
        assert_eq!(spec(Some(20)).protocol_security_target_bits(), 80.0);
        assert_eq!(spec(Some(60)).protocol_security_target_bits(), 40.0);
    }

    #[test]
    fn pow_exceeding_target_saturates_to_zero() {
        assert_eq!(spec(Some(200)).protocol_security_target_bits(), 0.0);
    }
}
