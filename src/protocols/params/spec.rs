use core::marker::PhantomData;

use crate::{algebra::embedding::Embedding, engines::EngineId};

/// Phantom-typed primitive — `Tagged<T, A>` and `Tagged<T, B>` are distinct types.
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

/// Security-target spec — *what* security the user wants. Tuning knobs live
/// in [`TuningSpec`].
#[derive(Debug, Clone)]
pub struct SecuritySpec<M: Embedding> {
    pub mode: Mode,
    pub target_security_bits: u32,
    // TODO: cross-protocol PoW pass; until then, set this to `None` or `Some(0)`
    // to avoid silently surrendering `max_pow_bits` of security.
    pub max_pow_bits: Option<u32>,
    pub hash_id: EngineId,
    pub _embedding: PhantomData<M>,
}

/// Tuning knobs — proof-size / prover-time / soundness-margin tradeoffs.
#[derive(Debug, Clone)]
pub struct TuningSpec {
    /// Witness vector size (input polynomial coefficient count).
    pub vector_size: usize,
    /// Starting log inverse rate for the initial RS code.
    pub starting_log_inv_rate: u32,
    /// Folding factor for the first (initial) sumcheck round.
    pub initial_folding_factor: usize,
    /// Folding factor for subsequent sumcheck rounds.
    pub folding_factor: usize,
}

/// Per-round context for bound calculations.
#[derive(Debug, Clone)]
pub struct RoundContext {
    pub round_index: usize,
    pub vector_size: usize,
    pub log_inv_rate: u32,
    pub folding_factor: u32,
    // Reserved for the orchestrator's combination-error sizing; unused by
    // current solvers.
    pub prev_round_in_domain_samples: usize,
    pub prev_round_query_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Regime is selectable.
    Standard { unique_decoding: bool },
    /// Always Johnson regime — Construction 9.7 needs OOD queries.
    ZeroKnowledge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OodSampleBudgetTag {}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskCodeMessageLenTag {}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogInvRateTag {}

/// `t_ood` — Bound 2's OOD-sample budget (produced by code-switch).
pub type OodSampleBudget = Tagged<usize, OodSampleBudgetTag>;

/// `ℓ_zk` — C_zk message length (Theorem 9.6: ℓ_zk ≥ source mask length).
pub type MaskCodeMessageLen = Tagged<usize, MaskCodeMessageLenTag>;

/// `rate = 2^-log_inv_rate`.
pub type LogInvRate = Tagged<u32, LogInvRateTag>;

impl Mode {
    pub const fn unique_decoding(&self) -> bool {
        matches!(
            self,
            Self::Standard {
                unique_decoding: true
            }
        )
    }
}

impl<M: Embedding> SecuritySpec<M> {
    /// Security bits the non-PoW parameters must deliver alone; the remaining
    /// `max_pow_bits` are closed by PoW grinding.
    ///
    /// **Until the cross-protocol PoW pass lands**, solvers emit no PoW —
    /// so subtracting `max_pow_bits` would silently under-target security.
    /// This function therefore asserts `max_pow_bits` is zero. Re-enable the
    /// subtraction when PoW grinding is wired in.
    pub fn protocol_security_target_bits(&self) -> f64 {
        assert!(
            self.max_pow_bits.unwrap_or(0) == 0,
            "max_pow_bits must be None or Some(0) until cross-protocol PoW grinding lands; \
             setting it nonzero now would silently surrender that many bits of security",
        );
        f64::from(self.target_security_bits)
    }
}
