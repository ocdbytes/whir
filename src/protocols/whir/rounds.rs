//! Shared round execution logic for the WHIR protocol.
//!
//! These free functions implement the round body for rounds 1+ and the final
//! round. Both the base WHIR prover/verifier and the zkWHIR 2.0 prover/verifier
//! call these functions to avoid duplicating the round loop.

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};

use super::RoundConfig;
use crate::{
    algebra::{
        dot,
        embedding::Identity,
        linear_form::{Evaluate, UnivariateEvaluation},
        MultilinearPoint,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, irs_commit, proof_of_work, sumcheck},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationResult, VerifierState,
    },
    utils::zip_strict,
    verify,
};

/// Result of a single verifier round (rounds 1+).
pub struct VerifyRoundResult<F: FftField> {
    pub commitment: irs_commit::Commitment<F>,
    pub in_domain: irs_commit::Evaluations<F>,
    pub stir_rlc_coeffs: Vec<F>,
    pub stir_challenges: Vec<UnivariateEvaluation<F>>,
    pub folding_randomness: MultilinearPoint<F>,
}

/// Single prover round body for rounds 1+ of the WHIR protocol.
///
/// Commits the current vector, opens the previous round's witness, accumulates
/// STIR constraints (OOD + in-domain), and runs the round's sumcheck.
#[allow(clippy::too_many_arguments)]
pub fn prove_round<F, H, R>(
    round_config: &RoundConfig<F>,
    prev_round_config: &RoundConfig<F>,
    prover_state: &mut ProverState<H, R>,
    vector: &mut Vec<F>,
    covector: &mut Vec<F>,
    the_sum: &mut F,
    prev_witness: &irs_commit::Witness<F, F>,
    folding_randomness: &MultilinearPoint<F>,
) -> (
    irs_commit::Witness<F, F>,
    irs_commit::Evaluations<F>,
    MultilinearPoint<F>,
)
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    u8: Decoding<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let new_witness = round_config
        .irs_committer
        .commit(prover_state, &[vector.as_slice()]);
    round_config.pow.prove(prover_state);

    let in_domain = prev_round_config
        .irs_committer
        .open(prover_state, &[prev_witness]);

    let stir_challenges: Vec<_> = new_witness
        .out_of_domain()
        .evaluators(round_config.initial_size())
        .chain(in_domain.evaluators(round_config.initial_size()))
        .collect();
    let stir_evaluations: Vec<F> = new_witness
        .out_of_domain()
        .values(&[F::ONE])
        .chain(in_domain.values(&folding_randomness.eq_weights()))
        .collect();

    let stir_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, stir_challenges.len());
    UnivariateEvaluation::accumulate_many(&stir_challenges, covector, &stir_rlc_coeffs);
    *the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);
    debug_assert_eq!(dot(vector, covector), *the_sum);

    let new_folding = round_config
        .sumcheck
        .prove(prover_state, vector, covector, the_sum);
    debug_assert_eq!(dot(vector, covector), *the_sum);

    (new_witness, in_domain, new_folding)
}

/// Final prover round.
///
/// Sends the (small) final folded vector directly, runs PoW, opens the last
/// commitment, and runs the final sumcheck.
#[allow(clippy::too_many_arguments)]
pub fn prove_final_round<F, H, R>(
    final_sumcheck: &sumcheck::Config<F>,
    final_pow: &proof_of_work::Config,
    last_round_config: &RoundConfig<F>,
    prover_state: &mut ProverState<H, R>,
    vector: &mut Vec<F>,
    covector: &mut Vec<F>,
    the_sum: &mut F,
    prev_witness: &irs_commit::Witness<F, F>,
) -> (irs_commit::Evaluations<F>, MultilinearPoint<F>)
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    u8: Decoding<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert_eq!(vector.len(), final_sumcheck.initial_size);
    for coeff in vector.iter() {
        prover_state.prover_message(coeff);
    }

    final_pow.prove(prover_state);

    let in_domain = last_round_config
        .irs_committer
        .open(prover_state, &[prev_witness]);

    let final_folding = final_sumcheck.prove(prover_state, vector, covector, the_sum);

    (in_domain, final_folding)
}

/// Single verifier round body for rounds 1+.
///
/// Receives commitment, verifies PoW, opens the previous round's commitment,
/// accumulates STIR constraints, and runs the round's sumcheck.
pub fn verify_round<F, H>(
    round_config: &RoundConfig<F>,
    prev_round_config: &RoundConfig<F>,
    verifier_state: &mut VerifierState<'_, H>,
    the_sum: &mut F,
    prev_commitment: &irs_commit::Commitment<F>,
    folding_randomness: &MultilinearPoint<F>,
) -> VerificationResult<VerifyRoundResult<F>>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let commitment = round_config
        .irs_committer
        .receive_commitment(verifier_state)?;
    round_config.pow.verify(verifier_state)?;

    let in_domain = prev_round_config
        .irs_committer
        .verify(verifier_state, &[prev_commitment])?;

    let stir_challenges: Vec<UnivariateEvaluation<F>> = commitment
        .out_of_domain()
        .evaluators(round_config.initial_size())
        .chain(in_domain.evaluators(round_config.initial_size()))
        .collect();
    let stir_evaluations: Vec<F> = commitment
        .out_of_domain()
        .values(&[F::ONE])
        .chain(in_domain.values(&folding_randomness.eq_weights()))
        .collect();

    let stir_rlc_coeffs: Vec<F> = geometric_challenge(verifier_state, stir_challenges.len());
    *the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

    let folding_randomness = round_config.sumcheck.verify(verifier_state, the_sum)?;

    Ok(VerifyRoundResult {
        commitment,
        in_domain,
        stir_rlc_coeffs,
        stir_challenges,
        folding_randomness,
    })
}

/// Final verifier round.
///
/// Receives the final vector, verifies PoW, opens the last commitment, checks
/// in-domain evaluations directly, and runs the final sumcheck.
///
/// Returns `(final_vector, in_domain, final_folding_randomness)`.
pub fn verify_final_round<F, H>(
    final_sumcheck: &sumcheck::Config<F>,
    final_pow: &proof_of_work::Config,
    last_round_config: &RoundConfig<F>,
    verifier_state: &mut VerifierState<'_, H>,
    the_sum: &mut F,
    prev_commitment: &irs_commit::Commitment<F>,
    folding_randomness: &MultilinearPoint<F>,
) -> VerificationResult<(Vec<F>, irs_commit::Evaluations<F>, MultilinearPoint<F>)>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let final_vector: Vec<F> = verifier_state.prover_messages_vec(final_sumcheck.initial_size)?;

    final_pow.verify(verifier_state)?;

    let in_domain = last_round_config
        .irs_committer
        .verify(verifier_state, &[prev_commitment])?;

    for (weight, eval) in zip_strict(
        in_domain.evaluators(final_vector.len()),
        in_domain.values(&folding_randomness.eq_weights()),
    ) {
        verify!(weight.evaluate(&Identity::<F>::new(), &final_vector) == eval);
    }

    let final_folding = final_sumcheck.verify(verifier_state, the_sum)?;

    Ok((final_vector, in_domain, final_folding))
}
