//! Shared test fixtures for `params/` solvers.

use std::{marker::PhantomData, ops::RangeInclusive};

use proptest::prelude::*;

use crate::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field64, Field64_2},
    },
    hash,
    protocols::params::{
        irs_commit as params_irs,
        plan::RoundModeParams,
        spec::{LogInvRate, MaskCodeMessageLen, Mode, RoundContext, SecuritySpec},
    },
};

pub type TestField = Field64;
pub type TestEmbedding = Identity<TestField>;

/// Extension field used by non-identity smoke tests.
pub type TestExtensionField = Field64_2;

/// Non-identity embedding: `Source = Field64`, `Target = Field64_2`.
pub type TestNonIdentityEmbedding = Basefield<TestExtensionField>;

/// Build a deterministic `SecuritySpec` for the given embedding and mode.
/// Useful for one-shot smoke / negative tests.
pub fn deterministic_spec<M: Embedding>(mode: Mode) -> SecuritySpec<M> {
    SecuritySpec {
        mode,
        target_security_bits: 80,
        max_pow_bits: None,
        hash_id: hash::BLAKE3,
        _embedding: PhantomData,
    }
}

/// `SecuritySpec` strategy with `max_pow_bits ∈ {None, Some(0)}` (PoW deferred).
pub fn arb_spec(
    mode: Mode,
    target_range: RangeInclusive<u32>,
) -> impl Strategy<Value = SecuritySpec<TestEmbedding>> {
    (target_range, prop_oneof![Just(None), Just(Some(0u32))]).prop_map(move |(target, max_pow)| {
        SecuritySpec {
            mode,
            target_security_bits: target,
            max_pow_bits: max_pow,
            hash_id: hash::BLAKE3,
            _embedding: PhantomData,
        }
    })
}

pub fn arb_zk_spec(
    target_range: RangeInclusive<u32>,
) -> impl Strategy<Value = SecuritySpec<TestEmbedding>> {
    arb_spec(Mode::ZeroKnowledge, target_range)
}

pub fn arb_standard_johnson_spec(
    target_range: RangeInclusive<u32>,
) -> impl Strategy<Value = SecuritySpec<TestEmbedding>> {
    arb_spec(
        Mode::Standard {
            unique_decoding: false,
        },
        target_range,
    )
}

pub fn arb_round_ctx() -> impl Strategy<Value = RoundContext> {
    (0usize..=3, 4u32..=8, 1u32..=4, 1u32..=3).prop_map(
        |(round_index, log_size, log_inv_rate, folding_factor)| RoundContext {
            round_index,
            vector_size: 1usize << log_size,
            log_inv_rate,
            folding_factor,
            prev_round_in_domain_samples: 0,
            prev_round_query_error: 0.0,
        },
    )
}

/// Minimal `RoundModeParams` matching `spec.mode`:
/// - `Mode::Standard` → `RoundModeParams::Standard`.
/// - `Mode::ZeroKnowledge` → `ZeroKnowledge { c_zk, l_zk }` with ℓ_zk = 2 and
///   C_zk at rate 1/2.
pub fn build_minimal_round_mode(
    spec: &SecuritySpec<TestEmbedding>,
) -> RoundModeParams<TestEmbedding> {
    if !matches!(spec.mode, Mode::ZeroKnowledge) {
        return RoundModeParams::Standard;
    }
    let l_zk = MaskCodeMessageLen::new(2);
    let c_zk = params_irs::solve_mask_code(spec, l_zk, 0, LogInvRate::new(1), 2);
    RoundModeParams::ZeroKnowledge { c_zk, l_zk }
}
