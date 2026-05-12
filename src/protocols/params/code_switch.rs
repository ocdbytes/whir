//! Parameter selection for the code-switching IOR (Construction 9.7, p.55).
//!
//! Computes `t_ood` (Bound 2 / Lemma 9.9 first error term) and sizes the ZK
//! mask oracle `ℓ_zk` per Theorem 9.6 + Lemma 9.5.

use std::num::NonZeroUsize;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        code_switch,
        irs_commit::{self, Config as IrsConfig},
        params::{
            plan::RoundModeParams,
            spec::{MaskCodeMessageLen, Mode, SecuritySpec},
        },
    },
};

/// Assemble the [`code_switch::Config`] from precomputed `t_ood` and a
/// mode-typed `zk` context.
///
/// In ZK mode, `zk` carries the `l_zk` produced by [`compute_l_zk`]; the
/// orchestrator must have used the same value for `irs_commit::solve_mask_code`
/// so both consumers see the same mask-oracle length.
pub fn solve<M: Embedding>(
    source: IrsConfig<M>,
    target: IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    zk: &RoundModeParams<M>,
) -> code_switch::Config<M> {
    let mode = match zk {
        RoundModeParams::Standard => code_switch::Mode::Standard,
        RoundModeParams::ZeroKnowledge { l_zk, .. } => {
            let l_zk = l_zk.get();
            assert!(
                l_zk >= source.mask_length() + t_ood,
                "ℓ_zk ({l_zk}) < r + t_ood ({} + {}) — violates Bound 3",
                source.mask_length(),
                t_ood,
            );
            code_switch::Mode::ZeroKnowledge {
                message_mask_length: NonZeroUsize::new(l_zk).expect("ℓ_zk > 0"),
            }
        }
    };
    code_switch::Config::new(source, target, t_ood, mode)
}

/// `ℓ_zk = next_power_of_two(r + t_ood)` — shared by code-switch and C_zk.
///
/// Bound 3 / Lemma 9.3 requires `ℓ_zk ≥ r + t_ood`; pow2 padding lets the
/// same value drive `irs_commit::solve_mask_code`'s NTT-order assertion.
pub const fn compute_l_zk<M: Embedding>(
    source: &irs_commit::Config<M>,
    t_ood: usize,
) -> MaskCodeMessageLen {
    MaskCodeMessageLen::new((source.mask_length() + t_ood).next_power_of_two())
}

/// `t_ood` from Bound 2 / Lemma 9.9 first error term.
///
/// Solves `(|Λ(C')| · |Λ(C_zk)|)² / 2 · ((ℓ + ℓ_zk - 1) / |F|)^{t_ood} ≤
/// 2^{-security}`. In ZK mode `ℓ_zk = r + t_ood` is mutually dependent with
/// `t_ood`; iterate to the fixed point.
pub fn compute_t_ood<M: Embedding>(
    spec: &SecuritySpec<M>,
    source: &IrsConfig<M>,
    target_list_size: f64,
    c_zk_list_size: Option<f64>,
) -> usize {
    const MAX_ITER: usize = 32;

    let security_target = spec.protocol_security_target_bits();
    let field_bits = M::Target::field_size_bits();
    let unique_decoding = spec.mode.unique_decoding();
    let combined_list_size = target_list_size * c_zk_list_size.unwrap_or(1.0);
    let message_length = source.message_length();
    let source_mask_length = source.mask_length();

    let solve_for_degree = |degree: usize| {
        irs_commit::num_ood_samples(
            unique_decoding,
            security_target,
            field_bits,
            combined_list_size,
            degree,
        )
    };

    if !matches!(spec.mode, Mode::ZeroKnowledge) {
        return solve_for_degree(message_length);
    }

    // ZK: t_ood = f(ℓ + r + t_ood); iterate.
    let mut t_ood = 0;
    for _ in 0..MAX_ITER {
        let new_t_ood = solve_for_degree(message_length + source_mask_length + t_ood);
        if new_t_ood == t_ood {
            return t_ood;
        }
        t_ood = new_t_ood;
    }
    panic!("compute_t_ood did not converge in {MAX_ITER} iterations");
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::{
        irs_commit as params_irs,
        plan::RoundModeParams,
        spec::{LogInvRate, OodSampleBudget, RoundContext},
        test_utils::{
            arb_standard_johnson_spec as utils_standard_spec, arb_zk_spec as utils_zk_spec,
            deterministic_standard_spec, TestEmbedding,
        },
    };

    type M = TestEmbedding;

    fn arb_zk_spec() -> impl Strategy<Value = SecuritySpec<M>> {
        utils_zk_spec(80..=128)
    }

    fn arb_standard_johnson_spec() -> impl Strategy<Value = SecuritySpec<M>> {
        utils_standard_spec(80..=128)
    }

    /// Orchestrator-style: build source + target IRS and the matching `t_ood`.
    /// Iterates target until its `codeword_length` stabilizes (target's realized
    /// rate depends on its mask budget, which depends on `t_ood`).
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
            prev_round_in_domain_samples: 0,
            prev_round_query_error: 0.0,
        };
        let source = params_irs::solve(spec, &source_ctx, OodSampleBudget::new(0));

        let target_ctx = RoundContext {
            round_index: 1,
            vector_size: source.message_length(),
            log_inv_rate: log_inv_rate + folding_factor - 1,
            folding_factor,
            prev_round_in_domain_samples: source.in_domain_samples,
            prev_round_query_error: 0.0,
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

    /// `num_vars ≥ 2 * folding_factor` so target's `vector_size` stays divisible
    /// by target's `interleaving_depth = 1 << folding_factor`.
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
        /// Standard mode: `Config::new` assertions pass, `t_ood ≥ 1` in Johnson.
        #[test]
        fn solve_standard_assembles(
            spec in arb_standard_johnson_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (source, target, t_ood) =
                build_inputs(&spec, log_inv_rate, folding_factor, num_vars, None);
            let config = solve(source, target, t_ood, &RoundModeParams::Standard);
            prop_assert!(matches!(config.mode, code_switch::Mode::Standard));
            prop_assert!(config.out_domain_samples >= 1);
        }

        /// ZK mode: `ℓ_zk = next_power_of_two(r + t_ood)` shared with C_zk.
        #[test]
        fn solve_zk_mask_equals_padded_r_plus_t_ood(
            spec in arb_zk_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            // Bootstrap C_zk with a placeholder t_ood to break the
            // t_ood ↔ c_zk.list_size circular dependency.
            let placeholder_source_ctx = RoundContext {
                round_index: 0,
                vector_size: 1usize << num_vars,
                log_inv_rate,
                folding_factor,
                prev_round_in_domain_samples: 0,
                prev_round_query_error: 0.0,
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
            // Fixed-point check: t_ood was computed with the placeholder
            // C_zk's list_size; the final C_zk's list_size must agree.
            let recomputed_t_ood =
                compute_t_ood(&spec, &source, target.list_size(), Some(c_zk.list_size()));
            prop_assert_eq!(
                t_ood, recomputed_t_ood,
                "t_ood computed with placeholder C_zk must equal t_ood with final C_zk",
            );
            let zk = RoundModeParams::ZeroKnowledge { c_zk, l_zk };
            let config = solve(source, target, t_ood, &zk);
            prop_assert_eq!(config.message_mask_length(), (r + t_ood).next_power_of_two());
        }

        /// `compute_t_ood` converges and returns `t_ood ≥ 1` in Johnson regime.
        #[test]
        fn compute_t_ood_converges(
            spec in arb_zk_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (_source, _target, t_ood) =
                build_inputs(&spec, log_inv_rate, folding_factor, num_vars, None);
            prop_assert!(t_ood >= 1);
        }
    }

    /// Smoke test: the generics compile and `solve` works end-to-end with a
    /// non-identity embedding (`M::Source ≠ M::Target`).
    #[test]
    fn solve_works_with_basefield_embedding() {
        use crate::algebra::{embedding::Basefield, fields::Field64_2};
        type NonIdM = Basefield<Field64_2>;

        let spec_source: SecuritySpec<NonIdM> = deterministic_standard_spec();
        let spec_target: SecuritySpec<Identity<Field64_2>> = deterministic_standard_spec();

        let source_ctx = RoundContext {
            round_index: 0,
            vector_size: 16,
            log_inv_rate: 1,
            folding_factor: 2,
            prev_round_in_domain_samples: 0,
            prev_round_query_error: 0.0,
        };
        let source = params_irs::solve(&spec_source, &source_ctx, OodSampleBudget::new(0));

        let target_ctx = RoundContext {
            round_index: 1,
            vector_size: source.message_length(),
            log_inv_rate: source_ctx.log_inv_rate + source_ctx.folding_factor - 1,
            folding_factor: source_ctx.folding_factor,
            prev_round_in_domain_samples: source.in_domain_samples,
            prev_round_query_error: 0.0,
        };

        let mut target = params_irs::solve(&spec_target, &target_ctx, OodSampleBudget::new(0));
        let mut t_ood = compute_t_ood(&spec_source, &source, target.list_size(), None);
        for _ in 0..8 {
            let new_target =
                params_irs::solve(&spec_target, &target_ctx, OodSampleBudget::new(t_ood));
            if new_target.codeword_length == target.codeword_length {
                target = new_target;
                break;
            }
            target = new_target;
            t_ood = compute_t_ood(&spec_source, &source, target.list_size(), None);
        }

        let config = solve(source, target, t_ood, &RoundModeParams::Standard);
        assert!(matches!(config.mode, code_switch::Mode::Standard));
    }
}
