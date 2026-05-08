//! Parameter selection for the IRS commit protocol.

use std::iter;

use crate::{
    algebra::{embedding::Embedding, ntt},
    protocols::{
        irs_commit::{self, num_in_domain_queries, IrsMode},
        params::spec::{
            LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget, RoundContext, SecuritySpec,
        },
    },
};

/// Solve per-round IRS-commit parameters. ZK mask sized per Lemma 9.5.
pub fn solve<M: Embedding + Default>(
    spec: &SecuritySpec<M>,
    ctx: &RoundContext,
    out_domain: OodSampleBudget,
) -> irs_commit::Config<M> {
    let security_target = f64::from(spec.protocol_security_target_bits());
    let raw_rate = 2_f64.powf(-f64::from(ctx.log_inv_rate));
    let interleaving_depth = 1_usize << ctx.folding_factor;
    let unique_decoding = spec.mode.unique_decoding();

    let mode = match spec.mode {
        Mode::Standard { .. } => IrsMode::Standard,
        Mode::ZeroKnowledge => {
            // Lemma 9.5 ZK budget: every revealed evaluation counts.
            let in_domain = num_in_domain_queries(unique_decoding, security_target, raw_rate);
            let mask_length = in_domain
                .checked_add(out_domain.get())
                .expect("usize overflow in mask_length");
            IrsMode::ZeroKnowledge { mask_length }
        }
    };

    let mask_length_value = match &mode {
        IrsMode::Standard => 0,
        IrsMode::ZeroKnowledge { mask_length } => mask_length.get(),
    };
    let masked_message_length = ctx.vector_size / interleaving_depth + mask_length_value;
    let rate = snap_rate::<M>(masked_message_length, raw_rate);

    irs_commit::Config::new(
        security_target,
        unique_decoding,
        spec.hash_id,
        // Orchestrator commits one vector per round.
        1,
        ctx.vector_size,
        interleaving_depth,
        rate,
        mode,
    )
}

/// Solve the shared C_zk IRS config for committing mask polynomials.
///
/// - `l_zk` — message length (Theorem 9.6: ℓ_zk ≥ `source_mask_length`).
/// - `source_mask_length` — `r`, the source IRS mask length.
/// - `log_inv_rate` — C_zk rate.
/// - `num_vectors` — total masks per commit; must equal `2 * num_masks` to be
///   consumable by `mask_proximity::Config::new` (original/fresh pairs).
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
        "Theorem 9.6: ℓ_zk ({l_zk}) must be ≥ source mask length ({source_mask_length})",
    );
    assert!(
        num_vectors.is_multiple_of(2),
        "num_vectors ({num_vectors}) must be even — mask-proximity expects 2 · num_masks (original + fresh)",
    );

    let security_target = f64::from(spec.protocol_security_target_bits());
    let raw_rate = 2_f64.powf(-f64::from(log_inv_rate.get()));
    // C_zk has interleaving_depth = 1 and IrsMode::Standard, so masked_message_length = l_zk.
    let rate = snap_rate::<M>(l_zk, raw_rate);

    irs_commit::Config::new(
        security_target,
        // ZK ⇒ Johnson regime.
        false,
        spec.hash_id,
        num_vectors,
        l_zk,
        1,
        rate,
        IrsMode::Standard,
    )
}

/// Snap `rate` so `Config::new`'s codeword sizing lands on a valid power-of-two
/// NTT order. Returns a rate `≤ raw_rate`.
fn snap_rate<M: Embedding>(masked_message_length: usize, raw_rate: f64) -> f64 {
    #[allow(clippy::cast_sign_loss)]
    let desired = (masked_message_length as f64 / raw_rate).ceil() as usize;
    let codeword_length = iter::successors(ntt::next_order::<M::Source>(desired), |&n| {
        ntt::next_order::<M::Source>(n + 1)
    })
    .find(|n| n.is_power_of_two())
    .expect("no valid power-of-two NTT order ≥ desired codeword length");
    masked_message_length as f64 / codeword_length as f64
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use proptest::prelude::*;

    use super::*;
    use crate::{
        algebra::{embedding::Identity, fields::Field64, random_vector},
        hash,
        transcript::{DomainSeparator, ProverState, VerifierState},
    };

    type F = Field64;
    type M = Identity<F>;

    fn arb_spec_with(mode: impl Strategy<Value = Mode>) -> impl Strategy<Value = SecuritySpec<M>> {
        (mode, 80u32..=128, 1u32..=4, prop::option::of(0u32..=20)).prop_map(
            |(mode, target_security_bits, starting_log_inv_rate, max_pow_bits)| SecuritySpec {
                mode,
                target_security_bits,
                vector_size: 1 << 8,
                starting_log_inv_rate,
                initial_folding_factor: 4,
                folding_factor: 4,
                max_pow_bits,
                hash_id: hash::BLAKE3,
                _embedding: PhantomData,
            },
        )
    }

    fn arb_zk_spec() -> impl Strategy<Value = SecuritySpec<M>> {
        arb_spec_with(Just(Mode::ZeroKnowledge))
    }

    fn arb_standard_spec() -> impl Strategy<Value = SecuritySpec<M>> {
        arb_spec_with(any::<bool>().prop_map(|unique_decoding| Mode::Standard { unique_decoding }))
    }

    fn arb_any_spec() -> impl Strategy<Value = SecuritySpec<M>> {
        prop_oneof![arb_zk_spec(), arb_standard_spec()]
    }

    fn arb_ctx() -> impl Strategy<Value = RoundContext> {
        (4u32..=8, 1u32..=4, 1u32..=3).prop_map(|(log_size, log_inv_rate, folding_factor)| {
            RoundContext {
                round_index: 0,
                vector_size: 1_usize << log_size,
                log_inv_rate,
                folding_factor,
                prev_round_in_domain_samples: 0,
                prev_round_query_error: 0.0,
            }
        })
    }

    proptest! {
        /// Lemma 9.5: ZK mask covers all revealed evaluations.
        #[test]
        fn zk_mask_covers_lemma_9_5(
            spec in arb_zk_spec(),
            ctx in arb_ctx(),
            out_domain in 0usize..16,
        ) {
            let config = solve(&spec, &ctx, OodSampleBudget::new(out_domain));
            prop_assert!(
                config.mask_length() >= config.in_domain_samples + out_domain,
                "mask {} < in_domain {} + out_domain {}",
                config.mask_length(), config.in_domain_samples, out_domain,
            );
        }

        /// Standard mode produces no IRS randomness.
        #[test]
        fn standard_has_no_mask(spec in arb_standard_spec(), ctx in arb_ctx()) {
            let config = solve(&spec, &ctx, OodSampleBudget::new(0));
            prop_assert_eq!(config.mask_length(), 0);
        }

        /// Round-trip: solve → commit → verify with the produced config.
        #[test]
        fn solve_round_trips_through_irs_commit(
            spec in arb_any_spec(),
            ctx in arb_ctx(),
            out_domain in 0usize..8,
            seed: u64,
        ) {
            let config = solve(&spec, &ctx, OodSampleBudget::new(out_domain));

            let ds = DomainSeparator::protocol(&config)
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
        }
    }
}
