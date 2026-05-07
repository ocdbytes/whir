use core::marker::PhantomData;

use crate::{algebra::embedding::Embedding, engines::EngineId};

/// Security spec definition for the protocol
pub struct SecuritySpec<M: Embedding> {
    /// Protocol Mode of operation
    pub mode: Mode,
    /// Target security bits
    pub target_security_bits: u32,
    /// Use the unique-decoding regime (`true`) instead of the Johnson regime.
    /// ZK mode requires Johnson — Construction 9.7 / Bound 2 needs OOD queries,
    /// and `num_ood_samples` returns 0 in unique-decoding.
    pub unique_decoding: bool,
    /// Size of the input witness / vector
    pub vector_size: usize,
    /// Starting log inverse rate for RS code
    pub starting_log_inv_rate: u32,
    /// Initial Folding factor for the first round of sumcheck
    pub initial_folding_factor: usize,
    /// Folding factor for subsequent round of sumcheck
    pub folding_factor: usize,
    /// POW bits
    pub max_pow_bits: Option<u32>,
    /// Hash Engine
    pub hash_id: EngineId,
    pub _embedding: PhantomData<M>,
}

/// Per round context struct for calculating the bounds
pub struct RoundContext {
    /// Round index
    pub round_index: usize,
    /// Vector size for the particular round
    pub vector_size: usize,
    /// rate for the RS encoding for the round vector
    pub log_inv_rate: u32,
    /// Forlding factor for sumcheck
    pub folding_factor: u32,
    /// Previous round's in domain samples count
    pub prev_round_in_domain_samples: usize,
    /// To keep track of the errors of all the rounds
    pub prev_round_query_error: f64,
}

pub enum Mode {
    Standard,
    ZeroKnowledge,
}
