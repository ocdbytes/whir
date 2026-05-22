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
            derive::solve_t_ood,
            irs_commit as irs_solver,
            protocol_config::MaskOracleInfo,
            regime::johnson_list_size,
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

/// `target_security_bits` range used by every solver-level proptest.
/// Upper bound keeps `target − analytic_error ≤ 60`, matching the cap in
/// `proof_of_work::threshold`. Lower bound keeps the analytic floor away from 0.
pub const TEST_TARGET_RANGE: RangeInclusive<u32> = 30..=50;

/// Default `target_security_bits` for `deterministic_spec` fixtures.
/// 80 leaves enough analytic headroom on `Field64` (~64-bit) that every
/// sub-protocol solver has a closable gap to target.
pub const FIXTURE_TARGET_BITS: u32 = 80;

/// Tolerance for `(got - expected).abs() < EPS` checks on formula-reconstruction
/// tests. `1e-9` is well above the `f64` rounding noise on log/sum expressions
/// used in the analytic-error formulas.
pub const EPS: f64 = 1e-9;

/// Matches `proof_of_work::MAX_DIFFICULTY` so per-slot budget checks in
/// `grind_to_at` never bite. Tests exercising budget enforcement build their
/// own specs.
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

pub fn arb_spec(
    mode: Mode,
    target_range: RangeInclusive<u32>,
) -> impl Strategy<Value = SecuritySpec> {
    target_range.prop_map(move |target| SecuritySpec {
        mode,
        decoding_regime: DecodingRegime::Johnson,
        target_security_bits: target,
        pow_budget: PowBudget::per_slot(FIXTURE_POW_BUDGET_BITS),
        hash_id: hash::BLAKE3,
    })
}

pub fn arb_zk_spec(target_range: RangeInclusive<u32>) -> impl Strategy<Value = SecuritySpec> {
    arb_spec(Mode::ZeroKnowledge, target_range)
}

pub fn arb_standard_johnson_spec(
    target_range: RangeInclusive<u32>,
) -> impl Strategy<Value = SecuritySpec> {
    arb_spec(Mode::Standard, target_range)
}

/// `log_size ∈ 4..=8` (vector_size 16..256) leaves room for ≥ 2·folding_factor
/// post-folding while capping proptest time.
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
        irs_solver::solve_mask_code(zk_spec, l_zk, 0, LogInvRate::new(1), 2);
    Some(MaskOracleInfo {
        c_zk_list_size: ListSize::new(c_zk.list_size()),
        l_zk,
    })
}

/// Shared check used by every sub-protocol's `pow_closes_gap_to_target*` test:
/// `analytic_error_bits + pow.difficulty() ≥ target_security_bits` (the `1e-3`
/// tolerance absorbs `proof_of_work::threshold`'s ceil quantization).
pub fn assert_pow_closes_gap(spec: &SecuritySpec, analytic: Bits, pow: &PowConfig) {
    let error = f64::from(analytic);
    let pow_bits = f64::from(pow.difficulty());
    let target = f64::from(spec.target_security_bits);
    assert!(
        error + pow_bits >= target - 1e-3,
        "error {error} + pow {pow_bits} < target {target}",
    );
}

/// `|got − expected| < EPS` with a uniform error message. Shared by every
/// `analytic_error_*_formula` test.
pub fn assert_close(got: f64, expected: f64) {
    assert!(
        (got - expected).abs() < EPS,
        "got {got} vs expected {expected}",
    );
}

/// C_zk fixture used by every `mask_proximity` test: source mask length 0,
/// `num_vectors = 2 · num_masks` (Construction 7.2 originals + fresh pairs).
pub fn build_test_c_zk(
    spec: &SecuritySpec,
    l_zk: usize,
    log_inv_rate: u32,
    num_masks: usize,
) -> IrsConfig<TestEmbedding> {
    let zk_spec = ZkSpec::try_new(spec).expect("build_test_c_zk requires a ZK spec");
    irs_solver::solve_mask_code(
        zk_spec,
        MaskCodeMessageLen::new(l_zk),
        0,
        LogInvRate::new(log_inv_rate),
        2 * num_masks,
    )
}

/// Builds a self-consistent `(source, target, t_ood)` triplet matching the
/// per-round shape that `code_switch::solve` expects.
///
/// `t_ood` is solved against the rate-only `johnson_list_size(target_log_inv_rate)`,
/// mirroring `derive::solve_t_ood`. Using `target.list_size()` here instead
/// would couple `t_ood` to the target's effective rate (which itself depends
/// on `t_ood` via the mask), producing a non-monotone oscillation once the
/// mask is tight (Lemma 9.5 part ii) rather than pow2-padded.
pub fn build_round_io<M: Embedding + Default>(
    spec: &SecuritySpec,
    log_inv_rate: u32,
    folding_factor: u32,
    num_vars: u32,
    c_zk_list_size: Option<f64>,
) -> (IrsConfig<M>, IrsConfig<Identity<M::Target>>, usize) {
    let source_ctx = RoundContext {
        vector_size: 1usize << num_vars,
        log_inv_rate,
        folding_factor,
    };
    let target_log_inv_rate = log_inv_rate + folding_factor - 1;
    let target_list_size = johnson_list_size(f64::from(target_log_inv_rate));
    let (source, t_ood) = solve_t_ood::<M>(spec, &source_ctx, target_list_size, c_zk_list_size, 0)
        .expect("solve_t_ood diverged in test fixture");

    let target_ctx = RoundContext {
        vector_size: source.message_length(),
        log_inv_rate: target_log_inv_rate,
        folding_factor,
    };
    let target = irs_solver::solve(spec, &target_ctx, OodSampleBudget::new(t_ood));
    (source, target, t_ood)
}
