//! Sumcheck parameter selection. ZK mode adds a degree-2 mask per round
//! (Lemma 6.4, p.38). PoW closes the gap between target and analytic error.

use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        irs_commit,
        params::{
            plan::MaskOracleInfo,
            spec::{RoundContext, SecuritySpec},
        },
        proof_of_work, sumcheck,
    },
};

/// `mask_oracle` is `Some` iff ZK; only C_zk's list size + ℓ_zk are read here.
pub fn solve<M: Embedding>(
    spec: &SecuritySpec<M>,
    ctx: &RoundContext,
    source_irs: &irs_commit::Config<M>,
    mask_oracle: Option<MaskOracleInfo>,
) -> sumcheck::Config<M::Target> {
    let num_rounds = num_sumcheck_rounds(ctx);
    let round_pow = proof_of_work::Config::grind_to(
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
    sumcheck::Config::new(ctx.vector_size, round_pow, num_rounds, mode)
}

/// Per-sumcheck-round soundness in bits: `min(ε_mca, poly_identity_term)`.
///
/// - Standard (degree-2): `log|F| − log|Λ(C)| − 1`.
/// - ZK (Lemma 6.5, p.40): `log|F| − log|Λ(C)| − log|Λ(C_zk)| − log ℓ_zk`.
pub fn analytic_error_bits<M: Embedding>(
    source_irs: &irs_commit::Config<M>,
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

pub const fn masks_required(is_zk: bool, ctx: &RoundContext) -> usize {
    if is_zk {
        num_sumcheck_rounds(ctx)
    } else {
        0
    }
}

const fn num_sumcheck_rounds(ctx: &RoundContext) -> usize {
    ctx.folding_factor as usize
}

/// Lemma 6.4, p.38: 3 coefficients suffice for a degree-2 round polynomial.
const fn zk_mask_length() -> usize {
    3
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::{
        irs_commit as params_irs,
        spec::OodSampleBudget,
        test_utils::{
            arb_round_ctx, arb_standard_johnson_spec, arb_zk_spec, build_minimal_mask_oracle,
            TestEmbedding,
        },
    };

    // Keeps `target - error ≤ 60`, the upper bound `proof_of_work::threshold` enforces.
    const TEST_TARGET_RANGE: std::ops::RangeInclusive<u32> = 30..=50;

    fn build_source_irs(
        spec: &SecuritySpec<TestEmbedding>,
        ctx: &RoundContext,
    ) -> irs_commit::Config<TestEmbedding> {
        params_irs::solve(spec, ctx, OodSampleBudget::new(0))
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

        /// Lemma 6.4: ZK round polynomial mask_length = 3.
        #[test]
        fn zk_mode_has_three_mask_coefficients(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            ctx in arb_round_ctx(),
        ) {
            let source_irs = build_source_irs(&spec, &ctx);
            let mask_oracle = build_minimal_mask_oracle(&spec);
            let config = solve(&spec, &ctx, &source_irs, mask_oracle);
            match config.mode {
                sumcheck::SumcheckMode::ZeroKnowledge { mask_length } => {
                    prop_assert_eq!(mask_length, 3);
                }
                sumcheck::SumcheckMode::Standard => prop_assert!(false, "expected ZK"),
            }
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

        #[test]
        fn masks_required_matches_mode(
            spec in prop_oneof![
                arb_standard_johnson_spec(TEST_TARGET_RANGE),
                arb_zk_spec(TEST_TARGET_RANGE),
            ],
            ctx in arb_round_ctx(),
        ) {
            let mask_oracle = build_minimal_mask_oracle(&spec);
            let required = masks_required(mask_oracle.is_some(), &ctx);
            let expected = if mask_oracle.is_some() { ctx.folding_factor as usize } else { 0 };
            prop_assert_eq!(required, expected);
        }

        #[test]
        fn solve_assembles_without_panic(
            spec in prop_oneof![
                arb_standard_johnson_spec(TEST_TARGET_RANGE),
                arb_zk_spec(TEST_TARGET_RANGE),
            ],
            ctx in arb_round_ctx(),
        ) {
            let source_irs = build_source_irs(&spec, &ctx);
            let mask_oracle = build_minimal_mask_oracle(&spec);
            let config = solve(&spec, &ctx, &source_irs, mask_oracle);
            prop_assert_eq!(config.initial_size, ctx.vector_size);
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
            let config = solve(&spec, &ctx, &source_irs, mask_oracle);
            let error = f64::from(analytic_error_bits(&source_irs, mask_oracle));
            let pow_bits = f64::from(config.round_pow.difficulty());
            // Tolerance for `proof_of_work::threshold`'s ceil quantization.
            prop_assert!(
                error + pow_bits >= f64::from(spec.target_security_bits) - 1e-3,
                "error {} + pow {} < target {}",
                error, pow_bits, spec.target_security_bits,
            );
        }
    }
}
