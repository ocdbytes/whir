//! Shared test fixtures.

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
        plan::MaskOracleInfo,
        spec::{LogInvRate, MaskCodeMessageLen, Mode, RoundContext, SecuritySpec},
    },
};

pub type TestField = Field64;
pub type TestEmbedding = Identity<TestField>;
pub type TestExtensionField = Field64_2;
/// `Source = Field64, Target = Field64_2`.
pub type TestNonIdentityEmbedding = Basefield<TestExtensionField>;

pub fn deterministic_spec<M: Embedding>(mode: Mode) -> SecuritySpec<M> {
    SecuritySpec {
        mode,
        target_security_bits: 80,
        max_pow_bits: None,
        hash_id: hash::BLAKE3,
        _embedding: PhantomData,
    }
}

/// `max_pow_bits` ∈ `{None, Some(0..=16)}`; bounded so the analytic floor
/// stays positive for the lowest test targets and the PoW gap stays under the
/// 60-bit cap.
pub fn arb_spec(
    mode: Mode,
    target_range: RangeInclusive<u32>,
) -> impl Strategy<Value = SecuritySpec<TestEmbedding>> {
    let pow_strategy = prop_oneof![Just(None), (0u32..=16).prop_map(Some)];
    (target_range, pow_strategy).prop_map(move |(target, max_pow)| SecuritySpec {
        mode,
        target_security_bits: target,
        max_pow_bits: max_pow,
        hash_id: hash::BLAKE3,
        _embedding: PhantomData,
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
        },
    )
}

/// `None` in Standard; `Some(ℓ_zk=2, c_zk rate 1/2)` in ZK.
pub fn build_minimal_mask_oracle(spec: &SecuritySpec<TestEmbedding>) -> Option<MaskOracleInfo> {
    if !matches!(spec.mode, Mode::ZeroKnowledge) {
        return None;
    }
    let l_zk = MaskCodeMessageLen::new(2);
    let c_zk = params_irs::solve_mask_code(spec, l_zk, 0, LogInvRate::new(1), 2);
    Some(MaskOracleInfo {
        c_zk_list_size: c_zk.list_size(),
        l_zk,
    })
}
