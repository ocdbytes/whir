//! Code-switching IOR (Construction 9.7, p.55) builder + Lemma 9.9 OOD bound.
//! The `t_ood` / `ℓ_zk` fixed-points live in the planner.

use std::num::NonZeroUsize;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    bits::Bits,
    protocols::{
        code_switch::{self, Config as CodeSwitchConfig},
        irs_commit::Config as IrsConfig,
        params::{protocol_config::MaskOracleInfo, spec::SecuritySpec},
        proof_of_work::Config as PowConfig,
    },
};

/// `mask_oracle.l_zk` must have been used to size C_zk (planner's job).
///
/// PoW closes the Lemma 9.9 OOD gap to `spec.target_security_bits`. `t_ood ≥ 1`
/// is required: enforced by [`analytic_error_bits`] and
/// [`code_switch::Config::new`] (Construction 9.7 needs OOD queries).
pub fn solve<M: Embedding>(
    spec: &SecuritySpec,
    source: IrsConfig<M>,
    target: IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    mask_oracle: Option<MaskOracleInfo>,
) -> CodeSwitchConfig<M> {
    let mode = mask_oracle.map_or(code_switch::CodeSwitchMode::Standard, |info| {
        let l_zk = info.l_zk.get();
        assert!(
            l_zk >= source.mask_length() + t_ood,
            "ℓ_zk ({l_zk}) < r + t_ood ({} + {}) — violates Bound 3",
            source.mask_length(),
            t_ood,
        );
        code_switch::CodeSwitchMode::ZeroKnowledge {
            message_mask_length: NonZeroUsize::new(l_zk).expect("ℓ_zk > 0"),
        }
    });

    let target_bits = Bits::new(f64::from(spec.target_security_bits));
    let analytic = analytic_error_bits(&source, &target, t_ood, mask_oracle);
    let pow = PowConfig::grind_to(target_bits, analytic, spec.hash_id);

    CodeSwitchConfig::new(source, target, t_ood, mode, pow)
}

/// Per-round code-switch soundness in bits: `min(ood_term, combination_term)`.
///
/// - OOD (Lemma 9.9, term 1):  `t_ood · (log|F| − log(deg − 1))  −  log(L choose 2)`
/// - Combination (Bound 1, γ-RLC): `log|F| − log(count) − log L`
///
/// `L = |Λ(target)| · |Λ(C_zk)|` (mask-oracle absent ⇒ `|Λ(C_zk)| = 1`).
/// `deg = ℓ + r + t_ood` in ZK, `ℓ` in Standard.
/// `count = t_ood + t · ι` (OOD samples + in-domain ι-interleaved source queries).
///
/// `t_ood ≥ 1` per [`code_switch::Config::new`].
pub fn analytic_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    mask_oracle: Option<MaskOracleInfo>,
) -> Bits {
    assert!(t_ood > 0, "code-switch requires t_ood ≥ 1");

    let field_bits = M::Target::field_size_bits();
    let combined_list = target.list_size() * mask_oracle.map_or(1.0, |info| info.c_zk_list_size);
    // OOD polynomial is over witness `[f; r_C; s]` of length `ℓ + ℓ_zk` (ZK) or
    // `ℓ` (Standard). The `s`-tail is sampled at full length `ℓ_zk − r` (not
    // just `t_ood`), so degree must use the realized `ℓ_zk`, not `r + t_ood`.
    let degree = mask_oracle.map_or_else(
        || source.message_length(),
        |info| source.message_length() + info.l_zk.get(),
    );
    #[allow(clippy::cast_precision_loss)]
    let t_ood_f = t_ood as f64;

    // OOD term — Lemma 9.9, term 1.
    #[allow(clippy::cast_precision_loss)]
    let log_degree_minus_1 = ((degree - 1) as f64).log2();
    let log_l_choose_2 = (combined_list * (combined_list - 1.0) / 2.0).log2();
    let ood_term = t_ood_f * (field_bits - log_degree_minus_1) - log_l_choose_2;

    // Combination term — Bound 1 (γ-RLC): `t_ood` OOD samples plus the
    // in-domain batch of `t · ι` source columns, all RLC'd into one target
    // codeword.
    #[allow(clippy::cast_precision_loss)]
    let log_count = ((t_ood + source.in_domain_samples * source.interleaving_depth) as f64).log2();
    let combination_term = field_bits - log_count - combined_list.log2();

    Bits::new(ood_term.min(combination_term).max(0.0))
}

/// Number of `(r ‖ s)` mask polynomials code-switch contributes to C_zk per
/// round. Mirrors [`super::sumcheck::masks_required`].
pub const fn masks_required() -> usize {
    1
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::{
        derive::{compute_l_zk, compute_t_ood},
        irs_commit as irs_solver,
        spec::{LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget, RoundContext, SecuritySpec},
        test_utils::{
            arb_standard_johnson_spec as utils_standard_spec, arb_zk_spec as utils_zk_spec,
            assert_close, assert_pow_closes_gap, build_round_io, deterministic_spec, TestEmbedding,
            TestExtensionField, TestField, TestNonIdentityEmbedding, TEST_TARGET_RANGE,
        },
    };

    type M = TestEmbedding;

    fn arb_zk_spec() -> impl Strategy<Value = SecuritySpec> {
        utils_zk_spec(TEST_TARGET_RANGE)
    }

    fn arb_standard_johnson_spec() -> impl Strategy<Value = SecuritySpec> {
        utils_standard_spec(TEST_TARGET_RANGE)
    }

    const NUM_VARS_HEADROOM: u32 = 4;

    /// `(log_inv_rate, folding_factor, num_vars)`. `num_vars ≥ 2 · folding_factor`
    /// keeps target IRS valid.
    fn arb_dims() -> impl Strategy<Value = (u32, u32, u32)> {
        (1u32..=3, 1u32..=2).prop_flat_map(|(log_inv_rate, folding_factor)| {
            let min_num_vars = 2 * folding_factor;
            (
                Just(log_inv_rate),
                Just(folding_factor),
                min_num_vars..=(min_num_vars + NUM_VARS_HEADROOM),
            )
        })
    }

    const FORMULA_LOG_INV_RATE: u32 = 1;
    const FORMULA_FOLDING_FACTOR: u32 = 2;
    const FORMULA_NUM_VARS: u32 = 6;

    /// Standard OOD bound (Lemma 9.9 first term, no mask):
    ///   `min(t_ood · (log|F| − log(ℓ−1)) − log(L choose 2), combination)`.
    /// `L = target.list_size()`, `combination = log|F| − log(t_ood + t·ι) − log L`.
    #[test]
    fn analytic_error_standard_formula() {
        let spec: SecuritySpec = deterministic_spec(Mode::Standard);
        let (source, target, t_ood) = build_round_io::<M>(
            &spec,
            FORMULA_LOG_INV_RATE,
            FORMULA_FOLDING_FACTOR,
            FORMULA_NUM_VARS,
            None,
        );
        let got = f64::from(analytic_error_bits(&source, &target, t_ood, None));

        let field_bits = <TestField as FieldWithSize>::field_size_bits();
        let target_list = target.list_size();
        let degree = source.message_length();
        let log_deg_m1 = ((degree - 1) as f64).log2();
        let l_choose_2 = target_list * (target_list - 1.0) / 2.0;
        let ood = (t_ood as f64) * (field_bits - log_deg_m1) - l_choose_2.log2();
        let count = t_ood + source.in_domain_samples * source.interleaving_depth;
        let comb = field_bits - (count as f64).log2() - target_list.log2();
        let expected = ood.min(comb).max(0.0);

        assert_close(got, expected);
    }

    /// ZK OOD bound: combined list `L = target × c_zk`, masked degree `ℓ + r + t_ood`,
    /// combination term also subtracts `log|Λ(C_zk)|`.
    #[test]
    fn analytic_error_zk_formula() {
        // Both mask-oracle values are pow2 so `log2` is exact (avoids
        // floating-point drift in the expected-vs-got comparison).
        const C_ZK_LIST_SIZE: f64 = 4.0; // log2 = 2
        const L_ZK_USIZE: usize = 8; // log2 = 3

        let spec: SecuritySpec = deterministic_spec(Mode::ZeroKnowledge);
        let mask_oracle = MaskOracleInfo {
            c_zk_list_size: C_ZK_LIST_SIZE,
            l_zk: MaskCodeMessageLen::new(L_ZK_USIZE),
        };
        let (source, target, t_ood) = build_round_io::<M>(
            &spec,
            FORMULA_LOG_INV_RATE,
            FORMULA_FOLDING_FACTOR,
            FORMULA_NUM_VARS,
            Some(C_ZK_LIST_SIZE),
        );
        let got = f64::from(analytic_error_bits(
            &source,
            &target,
            t_ood,
            Some(mask_oracle),
        ));

        let field_bits = <TestField as FieldWithSize>::field_size_bits();
        let target_list = target.list_size();
        let combined_list = target_list * C_ZK_LIST_SIZE;
        let degree = source.message_length() + L_ZK_USIZE;
        let log_deg_m1 = ((degree - 1) as f64).log2();
        let l_choose_2 = combined_list * (combined_list - 1.0) / 2.0;
        let ood = (t_ood as f64) * (field_bits - log_deg_m1) - l_choose_2.log2();
        let count = t_ood + source.in_domain_samples * source.interleaving_depth;
        let comb = field_bits - (count as f64).log2() - target_list.log2() - C_ZK_LIST_SIZE.log2();
        let expected = ood.min(comb).max(0.0);

        assert_close(got, expected);
    }

    proptest! {
        #[test]
        fn solve_standard_assembles(
            spec in arb_standard_johnson_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (source, target, t_ood) =
                build_round_io::<M>(&spec, log_inv_rate, folding_factor, num_vars, None);
            let config = solve(&spec, source, target, t_ood, None);
            prop_assert!(matches!(config.mode, code_switch::CodeSwitchMode::Standard));
            prop_assert!(config.out_domain_samples >= 1);
        }

        /// ZK: `ℓ_zk = next_power_of_two(r + t_ood)`.
        #[test]
        fn solve_zk_mask_equals_padded_r_plus_t_ood(
            spec in arb_zk_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            // Break the t_ood ↔ c_zk.list_size cycle with a placeholder C_zk.
            let placeholder_source_ctx = RoundContext {
                vector_size: 1usize << num_vars,
                log_inv_rate,
                folding_factor,
            };
            let placeholder_source = irs_solver::solve::<M>(
                &spec,
                &placeholder_source_ctx,
                OodSampleBudget::new(0),
            );
            let c_zk_placeholder = irs_solver::solve_mask_code::<M>(
                &spec,
                compute_l_zk(&placeholder_source, 1),
                placeholder_source.mask_length(),
                LogInvRate::new(log_inv_rate),
                2,
            );
            let (source, target, t_ood) = build_round_io::<M>(
                &spec, log_inv_rate, folding_factor, num_vars, Some(c_zk_placeholder.list_size()),
            );
            let r = source.mask_length();
            let l_zk = compute_l_zk(&source, t_ood);
            let c_zk = irs_solver::solve_mask_code::<M>(
                &spec,
                l_zk,
                r,
                LogInvRate::new(log_inv_rate),
                2,
            );
            let recomputed_t_ood =
                compute_t_ood(&spec, &source, target.list_size(), Some(c_zk.list_size()));
            prop_assert_eq!(t_ood, recomputed_t_ood, "placeholder ⇒ final C_zk fixed-point");
            let mask_oracle = MaskOracleInfo {
                c_zk_list_size: c_zk.list_size(),
                l_zk,
            };
            let config = solve(&spec, source, target, t_ood, Some(mask_oracle));
            prop_assert_eq!(config.message_mask_length(), (r + t_ood).next_power_of_two());
        }

        /// `analytic_error + pow ≥ target` (Lemma 9.9 OOD term).
        #[test]
        fn pow_closes_gap_to_target_standard(
            spec in arb_standard_johnson_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (source, target, t_ood) =
                build_round_io::<M>(&spec, log_inv_rate, folding_factor, num_vars, None);
            let error = analytic_error_bits(&source, &target, t_ood, None);
            let config = solve(&spec, source, target, t_ood, None);
            assert_pow_closes_gap(&spec, error, &config.pow);
        }
    }

    /// Shared shape for the `M::Source ≠ M::Target` smoke tests.
    /// `target_ctx` mirrors the planner's per-round chaining.
    fn non_identity_smoke_ctxs() -> (RoundContext, RoundContext) {
        const SOURCE_VECTOR_SIZE: usize = 64;
        const SOURCE_LOG_INV_RATE: u32 = 1;
        const FOLDING_FACTOR: u32 = 2;

        let source_ctx = RoundContext {
            vector_size: SOURCE_VECTOR_SIZE,
            log_inv_rate: SOURCE_LOG_INV_RATE,
            folding_factor: FOLDING_FACTOR,
        };
        let target_ctx = RoundContext {
            vector_size: source_ctx.vector_size / (1 << source_ctx.folding_factor),
            log_inv_rate: source_ctx.log_inv_rate + source_ctx.folding_factor - 1,
            folding_factor: source_ctx.folding_factor,
        };
        (source_ctx, target_ctx)
    }

    /// Smoke test: `M::Source ≠ M::Target`, Standard mode.
    #[test]
    fn solve_works_with_basefield_embedding_standard() {
        let spec: SecuritySpec = deterministic_spec(Mode::Standard);
        let (source_ctx, target_ctx) = non_identity_smoke_ctxs();

        let source = irs_solver::solve::<TestNonIdentityEmbedding>(
            &spec,
            &source_ctx,
            OodSampleBudget::new(0),
        );
        // Standard target: codeword_length is t_ood-independent (mask = 0).
        let target = irs_solver::solve::<Identity<TestExtensionField>>(
            &spec,
            &target_ctx,
            OodSampleBudget::new(0),
        );
        let t_ood = compute_t_ood(&spec, &source, target.list_size(), None);

        let config = solve(&spec, source, target, t_ood, None);
        assert!(matches!(config.mode, code_switch::CodeSwitchMode::Standard));
    }

    /// Placeholder mask-oracle list size for the smoke test. Pow2 keeps
    /// `log2` exact and matches `analytic_error_zk_formula`'s fixture.
    const SMOKE_C_ZK_LIST_SIZE: f64 = 4.0;
    /// Cap on the smoke-test `t_ood ↔ (source, target)` fixed-point. Matches the
    /// loop bound used in `build_round_io`; in practice converges in 1–3 iters.
    const SMOKE_FIXED_POINT_MAX_ITER: usize = 8;

    /// Smoke test: `M::Source ≠ M::Target`, ZK mode.
    #[test]
    fn solve_works_with_basefield_embedding_zk() {
        let spec: SecuritySpec = deterministic_spec(Mode::ZeroKnowledge);
        let (source_ctx, target_ctx) = non_identity_smoke_ctxs();

        let mut t_ood = 0;
        let mut source = irs_solver::solve::<TestNonIdentityEmbedding>(
            &spec,
            &source_ctx,
            OodSampleBudget::new(0),
        );
        let mut target = irs_solver::solve::<Identity<TestExtensionField>>(
            &spec,
            &target_ctx,
            OodSampleBudget::new(0),
        );
        for _ in 0..SMOKE_FIXED_POINT_MAX_ITER {
            let new_t_ood = compute_t_ood(
                &spec,
                &source,
                target.list_size(),
                Some(SMOKE_C_ZK_LIST_SIZE),
            );
            if new_t_ood == t_ood {
                break;
            }
            t_ood = new_t_ood;
            source = irs_solver::solve(&spec, &source_ctx, OodSampleBudget::new(t_ood));
            target = irs_solver::solve(&spec, &target_ctx, OodSampleBudget::new(t_ood));
        }

        let mask_oracle = MaskOracleInfo {
            c_zk_list_size: SMOKE_C_ZK_LIST_SIZE,
            l_zk: MaskCodeMessageLen::new((source.mask_length() + t_ood).next_power_of_two()),
        };
        let config = solve(&spec, source, target, t_ood, Some(mask_oracle));
        assert!(matches!(
            config.mode,
            code_switch::CodeSwitchMode::ZeroKnowledge { .. }
        ));
    }
}
