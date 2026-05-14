//! IRS-commit parameter selection. ZK mask sized per Lemma 9.5, padded so
//! `message + mask` is a pow2 (NTT-valid codeword length).

use std::num::NonZeroUsize;

use crate::{
    algebra::embedding::Embedding,
    protocols::{
        irs_commit::{self, num_in_domain_queries, IrsMode},
        params::spec::{
            LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget, RoundContext, SecuritySpec,
        },
    },
};

pub fn solve<M: Embedding + Default>(
    spec: &SecuritySpec<M>,
    ctx: &RoundContext,
    out_domain_samples: OodSampleBudget,
) -> irs_commit::Config<M> {
    let security_target = spec.protocol_security_target_bits();
    let rate = 2_f64.powf(-f64::from(ctx.log_inv_rate));
    let interleaving_depth = 1_usize << ctx.folding_factor;
    let unique_decoding = spec.mode.unique_decoding();
    let message_length = ctx.vector_size / interleaving_depth;

    let mode = match spec.mode {
        Mode::Standard { .. } => IrsMode::Standard,
        Mode::ZeroKnowledge => {
            let min_mask = num_in_domain_queries(unique_decoding, security_target, rate)
                .checked_add(out_domain_samples.get())
                .expect("usize overflow");
            // Lemma 9.5 is `≥`, so pow2 padding is safe.
            let mask_length = message_length
                .checked_add(min_mask.get())
                .expect("usize overflow")
                .next_power_of_two()
                .checked_sub(message_length)
                .and_then(NonZeroUsize::new)
                .expect("mask_length non-zero in ZK");
            IrsMode::ZeroKnowledge { mask_length }
        }
    };

    irs_commit::Config::new(
        security_target,
        unique_decoding,
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
    spec: &SecuritySpec<M>,
    l_zk: MaskCodeMessageLen,
    source_mask_length: usize,
    log_inv_rate: LogInvRate,
    num_vectors: usize,
) -> irs_commit::Config<M> {
    let l_zk = l_zk.get();
    assert!(
        matches!(spec.mode, Mode::ZeroKnowledge),
        "C_zk only exists in ZK mode"
    );
    assert!(
        l_zk >= source_mask_length,
        "Theorem 9.6: ℓ_zk ({l_zk}) ≥ source mask length ({source_mask_length})",
    );
    assert!(l_zk.is_power_of_two(), "ℓ_zk ({l_zk}) must be a power of 2");
    assert!(
        num_vectors.is_multiple_of(2),
        "num_vectors ({num_vectors}) must be even (mask-proximity original/fresh pairs)",
    );

    let security_target = spec.protocol_security_target_bits();
    let rate = 2_f64.powf(-f64::from(log_inv_rate.get()));

    irs_commit::Config::new(
        security_target,
        false, // ZK ⇒ Johnson regime
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
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use proptest::prelude::*;

    use super::*;
    use crate::{
        algebra::random_vector,
        protocols::params::test_utils::{
            arb_round_ctx, arb_spec, arb_zk_spec, deterministic_spec, TestEmbedding,
        },
        transcript::{DomainSeparator, ProverState, VerifierState},
    };

    type M = TestEmbedding;
    type F = <M as Embedding>::Source;

    #[test]
    #[should_panic(expected = "C_zk only exists in ZK mode")]
    fn solve_mask_code_rejects_standard_spec() {
        let spec: SecuritySpec<M> = deterministic_spec(Mode::Standard {
            unique_decoding: false,
        });
        let _ = solve_mask_code(&spec, MaskCodeMessageLen::new(2), 0, LogInvRate::new(1), 2);
    }

    #[test]
    #[should_panic(expected = "must be a power of 2")]
    fn solve_mask_code_rejects_non_pow2_l_zk() {
        let spec: SecuritySpec<M> = deterministic_spec(Mode::ZeroKnowledge);
        let _ = solve_mask_code(&spec, MaskCodeMessageLen::new(3), 0, LogInvRate::new(1), 2);
    }

    #[test]
    #[should_panic(expected = "Theorem 9.6")]
    fn solve_mask_code_rejects_l_zk_below_source_mask_length() {
        let spec: SecuritySpec<M> = deterministic_spec(Mode::ZeroKnowledge);
        let _ = solve_mask_code(&spec, MaskCodeMessageLen::new(2), 4, LogInvRate::new(1), 2);
    }

    #[test]
    #[should_panic(expected = "must be even")]
    fn solve_mask_code_rejects_odd_num_vectors() {
        let spec: SecuritySpec<M> = deterministic_spec(Mode::ZeroKnowledge);
        let _ = solve_mask_code(&spec, MaskCodeMessageLen::new(2), 0, LogInvRate::new(1), 3);
    }

    fn arb_zk_spec_default() -> impl Strategy<Value = SecuritySpec<M>> {
        arb_zk_spec(80..=128)
    }

    /// Varies `unique_decoding` to exercise both regimes.
    fn arb_standard_spec() -> impl Strategy<Value = SecuritySpec<M>> {
        any::<bool>()
            .prop_flat_map(|unique_decoding| arb_spec(Mode::Standard { unique_decoding }, 80..=128))
    }

    fn commit_open_verify(config: &irs_commit::Config<M>, seed: u64) -> irs_commit::Witness<F> {
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&seed);
        let mut rng = StdRng::seed_from_u64(seed);
        let vector = random_vector::<F>(&mut rng, config.vector_size);

        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit(&mut prover_state, &[&vector]);
        let _ = config.open(&mut prover_state, &[&witness]);
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        let _ = config.verify(&mut verifier_state, &[&commitment]).unwrap();
        verifier_state.check_eof().unwrap();
        witness
    }

    proptest! {
        /// Lemma 9.5: mask covers all revealed evaluations.
        #[test]
        fn zk_mask_covers_lemma_9_5(
            spec in arb_zk_spec_default(),
            ctx in arb_round_ctx(),
            out_domain in 0usize..16,
        ) {
            let config = solve(&spec, &ctx, OodSampleBudget::new(out_domain));
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
            let config = solve(&spec, &ctx, OodSampleBudget::new(out_domain));
            prop_assert_eq!(config.mask_length(), 0);
        }

        #[test]
        fn zk_round_trips(
            spec in arb_zk_spec_default(),
            ctx in arb_round_ctx(),
            out_domain in 0usize..8,
            seed: u64,
        ) {
            let config = solve(&spec, &ctx, OodSampleBudget::new(out_domain));
            prop_assert!(config.mask_length() > 0);
            let witness = commit_open_verify(&config, seed);
            prop_assert_eq!(witness.masks.len(), config.mask_length() * config.num_messages());
        }

        #[test]
        fn standard_round_trips(
            spec in arb_standard_spec(),
            ctx in arb_round_ctx(),
            seed: u64,
        ) {
            let config = solve(&spec, &ctx, OodSampleBudget::new(0));
            prop_assert_eq!(config.mask_length(), 0);
            let witness = commit_open_verify(&config, seed);
            prop_assert!(witness.masks.is_empty());
        }
    }
}
