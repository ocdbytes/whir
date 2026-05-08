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

/// Protocol-wide security spec.
#[derive(Debug, Clone)]
pub struct SecuritySpec<M: Embedding> {
    pub mode: Mode,
    pub target_security_bits: u32,
    pub vector_size: usize,
    pub starting_log_inv_rate: u32,
    pub initial_folding_factor: usize,
    pub folding_factor: usize,
    pub max_pow_bits: Option<u32>,
    pub hash_id: EngineId,
    pub _embedding: PhantomData<M>,
}

/// Per-round context for bound calculations.
#[derive(Debug, Clone)]
pub struct RoundContext {
    pub round_index: usize,
    pub vector_size: usize,
    pub log_inv_rate: u32,
    pub folding_factor: u32,
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
