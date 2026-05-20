//! Sumcheck parameter selection. ZK mode adds a degree-2 mask per round
//! (Lemma 6.4, p.38). PoW closes the gap between target and analytic error.

use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        irs_commit::Config as IrsConfig,
        params::{
            protocol_config::MaskOracleInfo,
            spec::{RoundContext, SecuritySpec},
        },
        proof_of_work::Config as PowConfig,
        sumcheck::{self, Config as SumcheckConfig},
    },
};

/// `mask_oracle` is `Some` iff ZK; only C_zk's list size + ℓ_zk are read here.
pub fn solve<M: Embedding>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
    source_irs: &IrsConfig<M>,
    mask_oracle: Option<MaskOracleInfo>,
) -> SumcheckConfig<M::Target> {
    let num_rounds = num_sumcheck_rounds(ctx);
    let round_pow = PowConfig::grind_to(
        Bits::new(f64::from(spec.target_security_bits)),
        analytic_error_bits(source_irs, mask_oracle),
        spec.hash_id,
    );
    let mode = match mask_oracle {
        None => sumcheck::SumcheckMode::Standard,
        Some(_) => sumcheck::SumcheckMode::ZeroKnowledge {
            mask_length: zk_mask_length(),
        },
    };
    SumcheckConfig::new(ctx.vector_size, round_pow, num_rounds, mode)
}

/// Per-sumcheck-round soundness in bits: `min(ε_mca, poly_identity_term)`.
///
/// - Standard (degree-2): `log|F| − log|Λ(C)| − 1`.
/// - ZK (Lemma 6.5, p.40): `log|F| − log|Λ(C)| − log|Λ(C_zk)| − log ℓ_zk`.
pub fn analytic_error_bits<M: Embedding>(
    source_irs: &IrsConfig<M>,
    mask_oracle: Option<MaskOracleInfo>,
) -> Bits {
    let field_bits = M::Target::field_size_bits();
    let log_list_size = source_irs.list_size().log2();
    let prox_gaps = source_irs.rbr_soundness_fold_prox_gaps();

    let poly_id = mask_oracle.map_or(field_bits - log_list_size - 1.0, |info| {
        let log_list_size_c_zk = info.c_zk_list_size.log2();
        #[allow(clippy::cast_precision_loss)]
        let log_l_zk = (info.l_zk.get() as f64).log2();
        field_bits - log_list_size - log_list_size_c_zk - log_l_zk
    });

    Bits::new(prox_gaps.min(poly_id).max(0.0))
}

/// Number of degree-2 round-polynomial masks sumcheck contributes to C_zk
/// per round (Lemma 6.4): one per sumcheck round.
pub const fn masks_required(ctx: &RoundContext) -> usize {
    num_sumcheck_rounds(ctx)
}

const fn num_sumcheck_rounds(ctx: &RoundContext) -> usize {
    ctx.folding_factor as usize
}

/// Construction 6.3 step 4(a) sends `h_j ∈ F^{<max{2, ℓ_zk}}[X]`. WHIR's round
/// polynomial is degree-2, so 3 coefficients suffice; `ℓ_zk = 3` is the
/// smallest value that masks it (Lemma 6.4 requires only `ℓ_zk ≥ 2`).
const fn zk_mask_length() -> usize {
    3
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::{
        irs_commit as irs_solver,
        spec::{MaskCodeMessageLen, Mode, OodSampleBudget},
        test_utils::{
            arb_round_ctx, arb_standard_johnson_spec, arb_zk_spec, assert_close,
            assert_pow_closes_gap, build_minimal_mask_oracle, deterministic_spec, TestEmbedding,
            TestField, TestNonIdentityEmbedding, EPS, TEST_TARGET_RANGE,
        },
    };

    /// Mask-oracle fixture used by the formula tests + the ZK smoke test.
    /// Both values are pow2 so `log2` is exact (no f64 drift in expected-vs-got).
    const FIXTURE_C_ZK_LIST_SIZE: f64 = 4.0;
    const FIXTURE_L_ZK: usize = 8;

    fn build_source_irs(spec: &SecuritySpec, ctx: &RoundContext) -> IrsConfig<TestEmbedding> {
        irs_solver::solve(spec, ctx, OodSampleBudget::new(0))
    }

    /// Smallest pow2 shape that still produces a non-degenerate IRS.
    const FIXTURE_LOG_VECTOR_SIZE: u32 = 4;
    const FIXTURE_LOG_INV_RATE: u32 = 1;
    const FIXTURE_FOLDING_FACTOR: u32 = 2;

    fn fixture_ctx() -> RoundContext {
        RoundContext {
            vector_size: 1 << FIXTURE_LOG_VECTOR_SIZE,
            log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FIXTURE_FOLDING_FACTOR,
        }
    }

    /// Lemma 6.4: ZK round polynomial has 3 coefficients.
    #[test]
    fn zk_mode_has_three_mask_coefficients() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = fixture_ctx();
        let source_irs = build_source_irs(&spec, &ctx);
        let mask_oracle = build_minimal_mask_oracle(&spec);
        let config = solve(&spec, &ctx, &source_irs, mask_oracle);
        match config.mode {
            sumcheck::SumcheckMode::ZeroKnowledge { mask_length } => {
                assert_eq!(mask_length, 3);
            }
            sumcheck::SumcheckMode::Standard => panic!("expected ZK"),
        }
    }

    /// Standard branch: `min(prox_gaps, log|F| − log|Λ(C)| − 1).max(0)`.
    #[test]
    fn analytic_error_standard_formula() {
        let spec = deterministic_spec(Mode::Standard);
        let ctx = fixture_ctx();
        let irs = build_source_irs(&spec, &ctx);

        let got = f64::from(analytic_error_bits::<TestEmbedding>(&irs, None));

        let field_bits = TestField::field_size_bits();
        let log_list = irs.list_size().log2();
        let prox = irs.rbr_soundness_fold_prox_gaps();
        let expected = prox.min(field_bits - log_list - 1.0).max(0.0);

        assert_close(got, expected);
    }

    /// ZK branch (Lemma 6.5): `min(prox_gaps, log|F| − log|Λ(C)| − log|Λ(C_zk)| − log ℓ_zk).max(0)`.
    #[test]
    fn analytic_error_zk_formula() {
        let log_c_zk_list = FIXTURE_C_ZK_LIST_SIZE.log2();
        let log_l_zk = (FIXTURE_L_ZK as f64).log2();

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = fixture_ctx();
        let irs = build_source_irs(&spec, &ctx);
        let info = MaskOracleInfo {
            c_zk_list_size: FIXTURE_C_ZK_LIST_SIZE,
            l_zk: MaskCodeMessageLen::new(FIXTURE_L_ZK),
        };

        let got = f64::from(analytic_error_bits::<TestEmbedding>(&irs, Some(info)));

        let field_bits = TestField::field_size_bits();
        let log_list = irs.list_size().log2();
        let prox = irs.rbr_soundness_fold_prox_gaps();
        let expected = prox
            .min(field_bits - log_list - log_c_zk_list - log_l_zk)
            .max(0.0);

        assert_close(got, expected);
    }

    /// Oracle large enough to drive `poly_id` strongly negative → clamped to 0.
    #[test]
    fn analytic_error_clamps_to_zero() {
        // `log2(c_zk_list_size) + log2(l_zk) > field_bits` on `Field64`.
        const OVERSIZED_LOG_C_ZK_LIST: i32 = 60;
        const OVERSIZED_LOG_L_ZK: u32 = 30;

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = fixture_ctx();
        let irs = build_source_irs(&spec, &ctx);
        let huge = MaskOracleInfo {
            c_zk_list_size: 2_f64.powi(OVERSIZED_LOG_C_ZK_LIST),
            l_zk: MaskCodeMessageLen::new(1 << OVERSIZED_LOG_L_ZK),
        };
        let bits = f64::from(analytic_error_bits::<TestEmbedding>(&irs, Some(huge)));
        assert_eq!(bits, 0.0);
    }

    proptest! {
        #[test]
        fn standard_mode_propagates(
            spec in arb_standard_johnson_spec(TEST_TARGET_RANGE),
            ctx in arb_round_ctx(),
        ) {
            let source_irs = build_source_irs(&spec, &ctx);
            let mask_oracle = build_minimal_mask_oracle(&spec);
            let config = solve(&spec, &ctx, &source_irs, mask_oracle);
            prop_assert!(matches!(config.mode, sumcheck::SumcheckMode::Standard));
        }

        #[test]
        fn num_rounds_matches_folding_factor(
            spec in prop_oneof![
                arb_standard_johnson_spec(TEST_TARGET_RANGE),
                arb_zk_spec(TEST_TARGET_RANGE),
            ],
            ctx in arb_round_ctx(),
        ) {
            let source_irs = build_source_irs(&spec, &ctx);
            let mask_oracle = build_minimal_mask_oracle(&spec);
            let config = solve(&spec, &ctx, &source_irs, mask_oracle);
            prop_assert_eq!(config.num_rounds, ctx.folding_factor as usize);
        }

        /// ZK subtracts two non-negative log terms beyond Standard, so the ZK
        /// error term cannot exceed the Standard one for any source IRS.
        #[test]
        fn zk_error_le_standard_error(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            ctx in arb_round_ctx(),
        ) {
            let irs = build_source_irs(&spec, &ctx);
            let mo = build_minimal_mask_oracle(&spec);
            let zk = f64::from(analytic_error_bits::<TestEmbedding>(&irs, mo));
            let standard = f64::from(analytic_error_bits::<TestEmbedding>(&irs, None));
            prop_assert!(zk <= standard + EPS, "zk {} > standard {}", zk, standard);
        }

        /// `analytic_error + pow ≥ target`.
        #[test]
        fn round_pow_closes_gap_to_target(
            spec in prop_oneof![
                arb_standard_johnson_spec(TEST_TARGET_RANGE),
                arb_zk_spec(TEST_TARGET_RANGE),
            ],
            ctx in arb_round_ctx(),
        ) {
            let source_irs = build_source_irs(&spec, &ctx);
            let mask_oracle = build_minimal_mask_oracle(&spec);
            let error = analytic_error_bits(&source_irs, mask_oracle);
            let config = solve(&spec, &ctx, &source_irs, mask_oracle);
            assert_pow_closes_gap(&spec, error, &config.round_pow);
        }
    }

    /// Smoke test: `M::Source ≠ M::Target`, ZK mode.
    #[test]
    fn solve_works_with_basefield_embedding_zk() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = fixture_ctx();
        let source_irs: IrsConfig<TestNonIdentityEmbedding> =
            irs_solver::solve(&spec, &ctx, OodSampleBudget::new(0));
        let info = MaskOracleInfo {
            c_zk_list_size: FIXTURE_C_ZK_LIST_SIZE,
            l_zk: MaskCodeMessageLen::new(FIXTURE_L_ZK),
        };
        let config = solve(&spec, &ctx, &source_irs, Some(info));
        assert!(matches!(
            config.mode,
            sumcheck::SumcheckMode::ZeroKnowledge { .. }
        ));
    }
}
