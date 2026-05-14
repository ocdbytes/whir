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
        code_switch,
        irs_commit::Config as IrsConfig,
        params::{plan::MaskOracleInfo, spec::SecuritySpec},
        proof_of_work,
    },
};

/// `mask_oracle.l_zk` must have been used to size C_zk (planner's job).
///
/// PoW closes the Lemma 9.9 OOD gap to `spec.target_security_bits`. When
/// `t_ood == 0` no OOD challenge is drawn, so no grinding is required.
pub fn solve<M: Embedding>(
    spec: &SecuritySpec<M>,
    source: IrsConfig<M>,
    target: IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    mask_oracle: Option<MaskOracleInfo>,
) -> code_switch::Config<M> {
    let mode = mask_oracle.map_or(code_switch::Mode::Standard, |info| {
        let l_zk = info.l_zk.get();
        assert!(
            l_zk >= source.mask_length() + t_ood,
            "ℓ_zk ({l_zk}) < r + t_ood ({} + {}) — violates Bound 3",
            source.mask_length(),
            t_ood,
        );
        code_switch::Mode::ZeroKnowledge {
            message_mask_length: NonZeroUsize::new(l_zk).expect("ℓ_zk > 0"),
        }
    });

    let target_bits = Bits::new(f64::from(spec.target_security_bits));
    let analytic = analytic_error_bits(&source, &target, t_ood, mask_oracle);
    let pow = proof_of_work::Config::grind_to(target_bits, analytic, spec.hash_id);

    code_switch::Config::new(source, target, t_ood, mode, pow)
}

/// Dominant soundness gap that PoW must close: `min(OOD term, combination term)`.
///
/// - OOD (Lemma 9.9, term 1): `t_ood · (log|F| − log(degree − 1)) − log(L choose 2)`,
///   with `L = target × c_zk` (ZK) or `target` (Standard), and
///   `degree = ℓ + r + t_ood` (ZK) or `ℓ` (Standard).
/// - Combination (Bound 1, γ-RLC): `log|F| − log(t_ood + t·ι) − log|Λ(target)| − [log|Λ(C_zk)|]`.
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
    let target_list = target.list_size();
    let combined_list = mask_oracle.map_or(target_list, |info| target_list * info.c_zk_list_size);
    let degree = mask_oracle.map_or_else(
        || source.message_length(),
        |_| source.masked_message_length() + t_ood,
    );

    #[allow(clippy::cast_precision_loss)]
    let log_degree_minus_1 = ((degree - 1) as f64).log2();
    let l_choose_2 = combined_list * (combined_list - 1.0) / 2.0;
    #[allow(clippy::cast_precision_loss)]
    let ood_term = (t_ood as f64) * (field_bits - log_degree_minus_1) - l_choose_2.log2();

    // Combination term: counts OOD samples plus the in-domain batch
    // (t source queries, each contributing one column of the ι-interleaved
    // source codeword to the geometric_challenge RLC).
    let count = t_ood + source.in_domain_samples * source.interleaving_depth;
    #[allow(clippy::cast_precision_loss)]
    let log_count = (count as f64).log2();
    let log_target_list = target_list.log2();
    let log_c_zk_list = mask_oracle.map_or(0.0, |info| info.c_zk_list_size.log2());
    let combination_term = field_bits - log_count - log_target_list - log_c_zk_list;

    Bits::new(ood_term.min(combination_term).max(0.0))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::{
        irs_commit as params_irs,
        planner::{compute_l_zk, compute_t_ood},
        spec::{LogInvRate, Mode, OodSampleBudget, RoundContext, SecuritySpec},
        test_utils::{
            arb_standard_johnson_spec as utils_standard_spec, arb_zk_spec as utils_zk_spec,
            deterministic_spec, TestEmbedding, TestExtensionField, TestNonIdentityEmbedding,
        },
    };

    type M = TestEmbedding;

    // Keeps `target − error ≤ 60`, the cap `proof_of_work::threshold` enforces.
    // On Field64 the γ-RLC combination term sits at ~0 bits in ZK and ~30 bits
    // in Standard, so the gap to target must stay under 60.
    const TEST_TARGET_RANGE: std::ops::RangeInclusive<u32> = 30..=50;

    fn arb_zk_spec() -> impl Strategy<Value = SecuritySpec<M>> {
        utils_zk_spec(TEST_TARGET_RANGE)
    }

    fn arb_standard_johnson_spec() -> impl Strategy<Value = SecuritySpec<M>> {
        utils_standard_spec(TEST_TARGET_RANGE)
    }

    /// Iterates target until `codeword_length` stabilizes — its realized rate
    /// depends on `mask_length`, which depends on `t_ood`.
    fn build_inputs(
        spec: &SecuritySpec<M>,
        log_inv_rate: u32,
        folding_factor: u32,
        num_vars: u32,
        c_zk_list_size: Option<f64>,
    ) -> (IrsConfig<M>, IrsConfig<M>, usize) {
        let source_ctx = RoundContext {
            round_index: 0,
            vector_size: 1usize << num_vars,
            log_inv_rate,
            folding_factor,
        };
        let source = params_irs::solve(spec, &source_ctx, OodSampleBudget::new(0));

        let target_ctx = RoundContext {
            round_index: 1,
            vector_size: source.message_length(),
            log_inv_rate: log_inv_rate + folding_factor - 1,
            folding_factor,
        };

        let mut target = params_irs::solve(spec, &target_ctx, OodSampleBudget::new(0));
        for _ in 0..8 {
            let t_ood = compute_t_ood(spec, &source, target.list_size(), c_zk_list_size);
            let new_target = params_irs::solve(spec, &target_ctx, OodSampleBudget::new(t_ood));
            if new_target.codeword_length == target.codeword_length {
                return (source, new_target, t_ood);
            }
            target = new_target;
        }
        panic!("target IRS did not stabilize");
    }

    /// `num_vars ≥ 2 * folding_factor` keeps target IRS valid.
    fn arb_dims() -> impl Strategy<Value = (u32, u32, u32)> {
        (1u32..=3, 1u32..=2).prop_flat_map(|(log_inv_rate, folding_factor)| {
            let min_num_vars = 2 * folding_factor;
            (
                Just(log_inv_rate),
                Just(folding_factor),
                min_num_vars..=(min_num_vars + 4),
            )
        })
    }

    proptest! {
        #[test]
        fn solve_standard_assembles(
            spec in arb_standard_johnson_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (source, target, t_ood) =
                build_inputs(&spec, log_inv_rate, folding_factor, num_vars, None);
            let config = solve(&spec, source, target, t_ood, None);
            prop_assert!(matches!(config.mode, code_switch::Mode::Standard));
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
                round_index: 0,
                vector_size: 1usize << num_vars,
                log_inv_rate,
                folding_factor,
            };
            let placeholder_source = params_irs::solve(
                &spec,
                &placeholder_source_ctx,
                OodSampleBudget::new(0),
            );
            let c_zk_placeholder = params_irs::solve_mask_code(
                &spec,
                compute_l_zk(&placeholder_source, 1),
                placeholder_source.mask_length(),
                LogInvRate::new(log_inv_rate),
                2,
            );
            let (source, target, t_ood) = build_inputs(
                &spec, log_inv_rate, folding_factor, num_vars, Some(c_zk_placeholder.list_size()),
            );
            let r = source.mask_length();
            let l_zk = compute_l_zk(&source, t_ood);
            let c_zk = params_irs::solve_mask_code(
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

        #[test]
        fn compute_t_ood_converges(
            spec in arb_zk_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (_source, _target, t_ood) =
                build_inputs(&spec, log_inv_rate, folding_factor, num_vars, None);
            prop_assert!(t_ood >= 1);
        }

        /// `analytic_error + pow ≥ target` (Lemma 9.9 OOD term).
        #[test]
        fn pow_closes_gap_to_target_standard(
            spec in arb_standard_johnson_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (source, target, t_ood) =
                build_inputs(&spec, log_inv_rate, folding_factor, num_vars, None);
            let config = solve(&spec, source.clone(), target.clone(), t_ood, None);
            let error = f64::from(analytic_error_bits(&source, &target, t_ood, None));
            let pow_bits = f64::from(config.pow.difficulty());
            prop_assert!(
                error + pow_bits >= f64::from(spec.target_security_bits) - 1e-3,
                "error {} + pow {} < target {}",
                error, pow_bits, spec.target_security_bits,
            );
        }
    }

    /// Shared shape so Standard and ZK smoke tests differ only in mode.
    fn non_identity_smoke_ctxs() -> (RoundContext, RoundContext) {
        let source_ctx = RoundContext {
            round_index: 0,
            vector_size: 64,
            log_inv_rate: 1,
            folding_factor: 2,
        };
        let target_ctx = RoundContext {
            round_index: 1,
            vector_size: source_ctx.vector_size / (1 << source_ctx.folding_factor),
            log_inv_rate: source_ctx.log_inv_rate + source_ctx.folding_factor - 1,
            folding_factor: source_ctx.folding_factor,
        };
        (source_ctx, target_ctx)
    }

    /// Smoke test: `M::Source ≠ M::Target`, Standard mode.
    #[test]
    fn solve_works_with_basefield_embedding_standard() {
        let spec_source: SecuritySpec<TestNonIdentityEmbedding> =
            deterministic_spec(Mode::Standard {
                unique_decoding: false,
            });
        let spec_target: SecuritySpec<Identity<TestExtensionField>> =
            deterministic_spec(Mode::Standard {
                unique_decoding: false,
            });
        let (source_ctx, target_ctx) = non_identity_smoke_ctxs();

        let source = params_irs::solve(&spec_source, &source_ctx, OodSampleBudget::new(0));
        // Standard target: codeword_length is t_ood-independent (mask = 0).
        let target = params_irs::solve(&spec_target, &target_ctx, OodSampleBudget::new(0));
        let t_ood = compute_t_ood(&spec_source, &source, target.list_size(), None);

        let config = solve(&spec_source, source, target, t_ood, None);
        assert!(matches!(config.mode, code_switch::Mode::Standard));
    }

    /// Smoke test: `M::Source ≠ M::Target`, ZK mode with shared C_zk.
    #[test]
    fn solve_works_with_basefield_embedding_zk() {
        let spec_source: SecuritySpec<TestNonIdentityEmbedding> =
            deterministic_spec(Mode::ZeroKnowledge);
        let spec_target: SecuritySpec<Identity<TestExtensionField>> =
            deterministic_spec(Mode::ZeroKnowledge);
        let (source_ctx, target_ctx) = non_identity_smoke_ctxs();

        let source = params_irs::solve(&spec_source, &source_ctx, OodSampleBudget::new(0));
        // Placeholder ℓ_zk to bootstrap c_zk.list_size.
        let c_zk_placeholder = params_irs::solve_mask_code(
            &spec_target,
            compute_l_zk(&source, 1),
            source.mask_length(),
            LogInvRate::new(1),
            2,
        );
        let c_zk_list_size = c_zk_placeholder.list_size();
        let target_placeholder =
            params_irs::solve(&spec_target, &target_ctx, OodSampleBudget::new(0));
        let t_ood = compute_t_ood(
            &spec_source,
            &source,
            target_placeholder.list_size(),
            Some(c_zk_list_size),
        );
        let target = params_irs::solve(&spec_target, &target_ctx, OodSampleBudget::new(t_ood));
        let t_ood_check = compute_t_ood(
            &spec_source,
            &source,
            target.list_size(),
            Some(c_zk_list_size),
        );
        assert_eq!(t_ood, t_ood_check, "fixed-point in one iteration");

        let l_zk = compute_l_zk(&source, t_ood);
        let c_zk = params_irs::solve_mask_code(
            &spec_target,
            l_zk,
            source.mask_length(),
            LogInvRate::new(1),
            2,
        );
        let mask_oracle = MaskOracleInfo {
            c_zk_list_size: c_zk.list_size(),
            l_zk,
        };
        let config = solve(&spec_source, source, target, t_ood, Some(mask_oracle));
        assert!(matches!(
            config.mode,
            code_switch::Mode::ZeroKnowledge { .. }
        ));
    }
}
