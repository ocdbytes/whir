//! Shared test fixtures.

use std::ops::RangeInclusive;

use proptest::prelude::*;

use crate::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field64, Field64_2},
    },
    bits::Bits,
    hash,
    protocols::{
        irs_commit::Config as IrsConfig,
        params::{
            build_round::{solve_t_ood, OodMode},
            irs_commit as irs_params,
            protocol_config::MaskOracleInfo,
            spec::{
                DecodingRegime, ListSize, LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget,
                PowBudget, RoundContext, SecuritySpec, ZkSpec,
            },
        },
        proof_of_work::Config as PowConfig,
    },
};

pub type TestField = Field64;
pub type TestEmbedding = Identity<TestField>;
pub type TestExtensionField = Field64_2;
/// `Source = Field64, Target = Field64_2`.
pub type TestNonIdentityEmbedding = Basefield<TestExtensionField>;

pub const TEST_TARGET_RANGE: RangeInclusive<u32> = 30..=50;

pub const FIXTURE_TARGET_BITS: u32 = 80;

pub const EPS: f64 = 1e-9;

pub const FIXTURE_POW_BUDGET_BITS: u32 = 60;

pub fn deterministic_spec(mode: Mode) -> SecuritySpec {
    SecuritySpec {
        mode,
        decoding_regime: DecodingRegime::Johnson,
        target_security_bits: FIXTURE_TARGET_BITS,
        pow_budget: PowBudget::per_slot(FIXTURE_POW_BUDGET_BITS),
        hash_id: hash::BLAKE3,
    }
}

fn arb_decoding_regime() -> impl Strategy<Value = DecodingRegime> {
    prop_oneof![
        Just(DecodingRegime::Johnson),
        Just(DecodingRegime::Unique),
        Just(DecodingRegime::Capacity),
    ]
}

pub fn arb_spec(
    mode: Mode,
    target_range: RangeInclusive<u32>,
) -> impl Strategy<Value = SecuritySpec> {
    (target_range, arb_decoding_regime()).prop_map(move |(target, decoding_regime)| SecuritySpec {
        mode,
        decoding_regime,
        target_security_bits: target,
        pow_budget: PowBudget::per_slot(FIXTURE_POW_BUDGET_BITS),
        hash_id: hash::BLAKE3,
    })
}

pub fn arb_zk_spec(target_range: RangeInclusive<u32>) -> impl Strategy<Value = SecuritySpec> {
    arb_spec(Mode::ZeroKnowledge, target_range)
}

pub fn arb_standard_spec(target_range: RangeInclusive<u32>) -> impl Strategy<Value = SecuritySpec> {
    arb_spec(Mode::Standard, target_range)
}

pub fn arb_round_ctx() -> impl Strategy<Value = RoundContext> {
    (4u32..=8, 1u32..=4, 1u32..=3).prop_map(|(log_size, log_inv_rate, folding_factor)| {
        RoundContext {
            vector_size: 1usize << log_size,
            log_inv_rate,
            folding_factor,
        }
    })
}

/// `None` in Standard; `Some(ℓ_zk=2, c_zk rate 1/2)` in ZK.
pub fn build_minimal_mask_oracle(spec: &SecuritySpec) -> Option<MaskOracleInfo> {
    let zk_spec = ZkSpec::try_new(spec)?;
    let l_zk = MaskCodeMessageLen::new(2);
    let c_zk: IrsConfig<TestEmbedding> =
        irs_params::solve_mask_code(zk_spec, l_zk, 0, LogInvRate::new(1), 2);
    Some(MaskOracleInfo {
        c_zk_list_size: ListSize::new(c_zk.list_size()),
        l_zk,
    })
}

/// `analytic_error_bits + pow.difficulty() ≥ target_security_bits`.
pub fn assert_pow_closes_gap(spec: &SecuritySpec, analytic: Bits, pow: &PowConfig) {
    let error = f64::from(analytic);
    let pow_bits = f64::from(pow.difficulty());
    let target = f64::from(spec.target_security_bits);
    assert!(
        error + pow_bits >= target - 1e-3,
        "error {error} + pow {pow_bits} < target {target}",
    );
}

/// `|got − expected| < EPS`.
pub fn assert_close(got: f64, expected: f64) {
    assert!(
        (got - expected).abs() < EPS,
        "got {got} vs expected {expected}",
    );
}

/// C_zk fixture for `mask_proximity` tests.
pub fn build_test_c_zk(
    spec: &SecuritySpec,
    l_zk: usize,
    log_inv_rate: u32,
    num_masks: usize,
) -> IrsConfig<TestEmbedding> {
    let zk_spec = ZkSpec::try_new(spec).expect("build_test_c_zk requires a ZK spec");
    irs_params::solve_mask_code(
        zk_spec,
        MaskCodeMessageLen::new(l_zk),
        0,
        LogInvRate::new(log_inv_rate),
        2 * num_masks,
    )
}

/// Builds a self-consistent `(source, target, t_ood)` triplet matching the
/// per-round shape that `code_switch::solve` expects.
pub fn build_round_io<M: Embedding + Default>(
    spec: &SecuritySpec,
    log_inv_rate: u32,
    folding_factor: u32,
    num_vars: u32,
    c_zk_log_inv_rate: Option<u32>,
) -> (IrsConfig<M>, IrsConfig<Identity<M::Target>>, usize) {
    let source_ctx = RoundContext {
        vector_size: 1usize << num_vars,
        log_inv_rate,
        folding_factor,
    };
    let target_log_inv_rate = log_inv_rate + folding_factor - 1;
    let target_log_degree = f64::from(num_vars - folding_factor);
    let target_list_size = spec
        .decoding_regime
        .list_size_estimate(target_log_degree, f64::from(target_log_inv_rate));
    let ood_mode = c_zk_log_inv_rate.map_or(OodMode::Standard, |rate| OodMode::ZeroKnowledge {
        c_zk_log_inv_rate: f64::from(rate),
    });
    let (source, t_ood) = solve_t_ood::<M>(spec, &source_ctx, target_list_size, ood_mode, 0)
        .expect("solve_t_ood diverged in test fixture");

    let target_ctx = RoundContext {
        vector_size: source.message_length(),
        log_inv_rate: target_log_inv_rate,
        folding_factor,
    };
    let target = irs_params::solve(spec, &target_ctx, OodSampleBudget::new(t_ood));
    (source, target, t_ood)
}
