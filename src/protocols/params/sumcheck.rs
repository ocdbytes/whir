//! Parameter selection for the per-round sumcheck protocol.
//!
//! Produces a [`sumcheck::Config`] from a `RoundContext` and the ZK context.
//! ZK mode adds a degree-2 masking polynomial per round (Lemma 6.4, p.38).

use crate::{
    algebra::embedding::Embedding,
    protocols::{
        params::{plan::RoundModeParams, spec::RoundContext},
        proof_of_work, sumcheck,
    },
};

/// Solve sumcheck parameters for one round.
pub fn solve<M: Embedding>(
    ctx: &RoundContext,
    zk: &RoundModeParams<M>,
) -> sumcheck::Config<M::Target> {
    let num_rounds = num_sumcheck_rounds(ctx);
    let mode = match zk {
        RoundModeParams::Standard => sumcheck::SumcheckMode::Standard,
        RoundModeParams::ZeroKnowledge { .. } => sumcheck::SumcheckMode::ZeroKnowledge {
            mask_length: zk_mask_length(),
        },
    };
    sumcheck::Config::new(
        ctx.vector_size,
        proof_of_work::Config::none(),
        num_rounds,
        mode,
    )
}

/// Number of mask polynomials required for one round of sumcheck.
pub const fn masks_required<M: Embedding>(zk: &RoundModeParams<M>, ctx: &RoundContext) -> usize {
    if zk.is_zk() {
        num_sumcheck_rounds(ctx)
    } else {
        0
    }
}

const fn num_sumcheck_rounds(ctx: &RoundContext) -> usize {
    ctx.folding_factor as usize
}

/// 3 coefficients suffice to mask the degree-2 sumcheck round polynomial —
/// Lemma 6.4, p.38.
const fn zk_mask_length() -> usize {
    3
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::test_utils::{
        arb_round_ctx, arb_standard_johnson_spec, arb_zk_spec, build_minimal_round_mode,
    };

    proptest! {
        /// Standard spec produces `SumcheckMode::Standard`.
        #[test]
        fn standard_mode_propagates(
            spec in arb_standard_johnson_spec(80..=128),
            ctx in arb_round_ctx(),
        ) {
            let zk = build_minimal_round_mode(&spec);
            let config = solve(&ctx, &zk);
            prop_assert!(matches!(config.mode, sumcheck::SumcheckMode::Standard));
        }

        /// ZK spec produces `SumcheckMode::ZeroKnowledge { mask_length: 3 }` — Lemma 6.4.
        #[test]
        fn zk_mode_has_three_mask_coefficients(
            spec in arb_zk_spec(80..=128),
            ctx in arb_round_ctx(),
        ) {
            let zk = build_minimal_round_mode(&spec);
            let config = solve(&ctx, &zk);
            match config.mode {
                sumcheck::SumcheckMode::ZeroKnowledge { mask_length } => {
                    prop_assert_eq!(mask_length, 3);
                }
                sumcheck::SumcheckMode::Standard => prop_assert!(false, "expected ZK"),
            }
        }

        /// `num_rounds = ctx.folding_factor`.
        #[test]
        fn num_rounds_matches_folding_factor(
            spec in prop_oneof![
                arb_standard_johnson_spec(80..=128),
                arb_zk_spec(80..=128),
            ],
            ctx in arb_round_ctx(),
        ) {
            let zk = build_minimal_round_mode(&spec);
            let config = solve(&ctx, &zk);
            prop_assert_eq!(config.num_rounds, ctx.folding_factor as usize);
        }

        /// `masks_required` = 0 in Standard, = `ctx.folding_factor` in ZK.
        #[test]
        fn masks_required_matches_mode(
            spec in prop_oneof![
                arb_standard_johnson_spec(80..=128),
                arb_zk_spec(80..=128),
            ],
            ctx in arb_round_ctx(),
        ) {
            let zk = build_minimal_round_mode(&spec);
            let required = masks_required(&zk, &ctx);
            let expected = if zk.is_zk() { ctx.folding_factor as usize } else { 0 };
            prop_assert_eq!(required, expected);
        }

        /// Smoke test: `solve` doesn't panic on assembly.
        #[test]
        fn solve_assembles_without_panic(
            spec in prop_oneof![
                arb_standard_johnson_spec(80..=128),
                arb_zk_spec(80..=128),
            ],
            ctx in arb_round_ctx(),
        ) {
            let zk = build_minimal_round_mode(&spec);
            let config = solve(&ctx, &zk);
            prop_assert_eq!(config.initial_size, ctx.vector_size);
        }
    }
}
