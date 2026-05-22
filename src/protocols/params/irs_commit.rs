//! IRS-commit parameter selection.
//!
//! ZK mask is sized per Lemma 9.5 (paper p.53) at the tight bound
//! `in-domain + OOD` queries. Codeword NTT-smoothness is enforced inside
//! [`IrsConfig::new`] on `codeword_length`, not by inflating the mask.

use crate::{
    algebra::embedding::Embedding,
    protocols::{
        irs_commit::{num_in_domain_queries, Config as IrsConfig, IrsMode},
        params::{
            bounds::rate,
            spec::{
                LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget, RoundContext, SecuritySpec,
                ZkSpec,
            },
        },
    },
};

pub fn solve<M: Embedding + Default>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
    out_domain_samples: OodSampleBudget,
) -> IrsConfig<M> {
    let security_target = f64::from(spec.protocol_security_target_bits());
    let rate = rate(f64::from(ctx.log_inv_rate));
    let interleaving_depth = 1_usize << ctx.folding_factor;

    let mode = match spec.mode {
        Mode::Standard => IrsMode::Standard,
        Mode::ZeroKnowledge => {
            // Lemma 9.5 (part ii): r-query perfect-ZK encoding requires
            // `r ≥ in-domain + OOD`. Use the tight bound; do not pow2-pad here.
            let mask_length = num_in_domain_queries(spec.decoding_regime, security_target, rate)
                .checked_add(out_domain_samples.get())
                .expect("usize overflow");
            IrsMode::ZeroKnowledge { mask_length }
        }
    };

    IrsConfig::new(
        security_target,
        spec.decoding_regime,
        spec.hash_id,
        1, // one vector committed per round
        ctx.vector_size,
        interleaving_depth,
        rate,
        mode,
    )
}

/// Shared C_zk IRS config for mask polynomials.
///
/// - `l_zk`: message length, must be a power of 2.
/// - `source_mask_length`: `r` from Theorem 9.6.
/// - `num_vectors`: `2 * num_masks` (Construction 7.2: originals + fresh).
pub fn solve_mask_code<M: Embedding + Default>(
    spec: ZkSpec<'_>,
    l_zk: MaskCodeMessageLen,
    source_mask_length: usize,
    log_inv_rate: LogInvRate,
    num_vectors: usize,
) -> IrsConfig<M> {
    let l_zk = l_zk.get();
    assert!(
        l_zk >= source_mask_length,
        "Theorem 9.6: ℓ_zk ({l_zk}) ≥ source mask length ({source_mask_length})",
    );
    assert!(l_zk.is_power_of_two(), "ℓ_zk ({l_zk}) must be a power of 2");
    assert!(
        num_vectors.is_multiple_of(2),
        "num_vectors ({num_vectors}) must be even (mask-proximity original/fresh pairs)",
    );

    let security_target = f64::from(spec.protocol_security_target_bits());
    let rate = rate(f64::from(log_inv_rate.get()));

    IrsConfig::new(
        security_target,
        spec.decoding_regime,
        spec.hash_id,
        num_vectors,
        l_zk,
        1,
        rate,
        IrsMode::Standard,
    )
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::test_utils::{
        arb_round_ctx, arb_spec, arb_zk_spec, deterministic_spec, TestEmbedding,
        TestNonIdentityEmbedding,
    };

    type M = TestEmbedding;

    #[test]
    fn zk_spec_rejects_standard_mode() {
        let spec: SecuritySpec = deterministic_spec(Mode::Standard);
        assert!(ZkSpec::try_new(&spec).is_none());
    }

    #[test]
    #[should_panic(expected = "must be a power of 2")]
    fn solve_mask_code_rejects_non_pow2_l_zk() {
        let spec: SecuritySpec = deterministic_spec(Mode::ZeroKnowledge);
        let zk_spec = ZkSpec::try_new(&spec).unwrap();
        let _ = solve_mask_code::<M>(
            zk_spec,
            MaskCodeMessageLen::new(3),
            0,
            LogInvRate::new(1),
            2,
        );
    }

    #[test]
    #[should_panic(expected = "Theorem 9.6")]
    fn solve_mask_code_rejects_l_zk_below_source_mask_length() {
        let spec: SecuritySpec = deterministic_spec(Mode::ZeroKnowledge);
        let zk_spec = ZkSpec::try_new(&spec).unwrap();
        let _ = solve_mask_code::<M>(
            zk_spec,
            MaskCodeMessageLen::new(2),
            4,
            LogInvRate::new(1),
            2,
        );
    }

    #[test]
    #[should_panic(expected = "must be even")]
    fn solve_mask_code_rejects_odd_num_vectors() {
        let spec: SecuritySpec = deterministic_spec(Mode::ZeroKnowledge);
        let zk_spec = ZkSpec::try_new(&spec).unwrap();
        let _ = solve_mask_code::<M>(
            zk_spec,
            MaskCodeMessageLen::new(2),
            0,
            LogInvRate::new(1),
            3,
        );
    }

    /// `irs_commit::solve` doesn't grind PoW, so this range can sit higher than
    /// the shared `TEST_TARGET_RANGE` (which is capped at 50 to keep the PoW
    /// gap below the 60-bit threshold). 80..=128 covers production-realistic
    /// target sizes.
    const IRS_TARGET_RANGE: std::ops::RangeInclusive<u32> = 80..=128;

    fn arb_zk_spec_default() -> impl Strategy<Value = SecuritySpec> {
        arb_zk_spec(IRS_TARGET_RANGE)
    }

    fn arb_standard_spec() -> impl Strategy<Value = SecuritySpec> {
        arb_spec(Mode::Standard, IRS_TARGET_RANGE)
    }

    proptest! {
        /// Lemma 9.5 (part ii): mask covers all revealed evaluations.
        #[test]
        fn zk_mask_covers_lemma_9_5(
            spec in arb_zk_spec_default(),
            ctx in arb_round_ctx(),
            out_domain in 0usize..16,
        ) {
            let config = solve::<M>(&spec, &ctx, OodSampleBudget::new(out_domain));
            prop_assert!(
                config.mask_length() >= config.in_domain_samples + out_domain,
                "mask {} < in_domain {} + out_domain {}",
                config.mask_length(), config.in_domain_samples, out_domain,
            );
        }

        #[test]
        fn standard_has_no_mask(
            spec in arb_standard_spec(),
            ctx in arb_round_ctx(),
            out_domain in 0usize..8,
        ) {
            let config = solve::<M>(&spec, &ctx, OodSampleBudget::new(out_domain));
            prop_assert_eq!(config.mask_length(), 0);
        }
    }

    /// Smoke-test fixture: 64-element vector folded by 2 at rate 1/2 — small
    /// but produces a non-degenerate IRS for the non-identity embedding.
    const SMOKE_VECTOR_SIZE: usize = 64;
    const SMOKE_LOG_INV_RATE: u32 = 1;
    const SMOKE_FOLDING_FACTOR: u32 = 2;
    /// Arbitrary > 0 so the ZK mask sizing exercises the OOD path.
    const SMOKE_OOD_BUDGET: usize = 2;

    /// Smoke test: `M::Source ≠ M::Target`, ZK path. Mask sizing depends only
    /// on the target field (via `field_size_bits`), but the generic embedding
    /// still flows through the Config and must compile + execute.
    #[test]
    fn solve_works_with_basefield_embedding_zk() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = RoundContext {
            vector_size: SMOKE_VECTOR_SIZE,
            log_inv_rate: SMOKE_LOG_INV_RATE,
            folding_factor: SMOKE_FOLDING_FACTOR,
        };
        let config: IrsConfig<TestNonIdentityEmbedding> =
            solve(&spec, &ctx, OodSampleBudget::new(SMOKE_OOD_BUDGET));
        assert!(config.mask_length() > 0);
    }
}
