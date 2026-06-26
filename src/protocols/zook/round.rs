//! The atomic per-round operation shared by every Zook flow.
//!
//! Both `prove_whir_round` and `verify_whir_round` are thin orchestrators
//! around the multi-source code-switch and mask paths. Calling them with a
//! single source (`t = 1`, `theta = [F::ONE]`) produces transcript-byte-
//! identical output to the legacy single-source path because the original
//! `code_switch::prove` / `verify_for_implicit` and `bind_code_switch_mask`
//! were themselves wrappers around the multi-source primitives.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use zeroize::Zeroize;

use crate::{
    algebra::{dot, embedding::Identity},
    hash::Hash,
    protocols::{
        code_switch::{self, CovectorUpdateParams},
        irs_commit::{Commitment as IrsCommitment, Witness as IrsWitness},
        params::config::RoundConfig,
        zook::{
            block::{ProverBlock, VerifierBlock},
            prover::RoundMaskOracle,
            slot_weights,
            verifier::RoundMaskOracleCheck,
        },
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationResult, VerifierState,
    },
};

#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, name = "zook::prove_whir_round", fields(msg_len = round.code_switch().source().message_length(), t = block.witnesses.len())))]
pub(crate) fn prove_whir_round<F, H, R>(
    round: &RoundConfig<Identity<F>>,
    block: ProverBlock<F>,
    ps: &mut ProverState<H, R>,
) -> ProverBlock<F>
where
    F: Field + Default + Zeroize + Codec<[H::U]>,
    Standard: Distribution<F>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let ProverBlock {
        mut message,
        mut covector,
        mut sum,
        witnesses,
        theta,
    } = block;

    debug_assert_eq!(witnesses.len(), theta.len());
    debug_assert!(!witnesses.is_empty());
    debug_assert_eq!(
        dot(&message, &covector),
        sum,
        "prove_whir_round entry: dot(message, covector) must equal sum"
    );

    let msg_len = round.code_switch().source().message_length();

    let mut masker = RoundMaskOracle::begin(round, ps);

    let opening = round.sumcheck().prove(
        ps,
        &mut message,
        &mut covector,
        &mut sum,
        masker.sumcheck_blinding(),
    );

    let witness_refs: Vec<&IrsWitness<F>> = witnesses.iter().collect();
    masker.bind_code_switch_mask_multi_source(&witness_refs, &theta, &opening, &mut sum, ps);

    debug_assert_eq!(
        dot(&message, &covector),
        sum,
        "prove_whir_round post-reconcile: dot(message, covector) must equal sum"
    );

    covector.resize(msg_len + masker.covector_extension(), F::ZERO);

    let slot_weights = slot_weights::build(&theta, &opening.round_challenges);
    let cs_witness = round.code_switch().prove_virtual(
        ps,
        message,
        &witness_refs,
        &slot_weights,
        code_switch::Claim {
            covector: &mut covector,
            sum: &mut sum,
        },
        masker.code_switch_blinding(),
    );

    masker.finish(
        &opening.round_challenges,
        &covector[msg_len..],
        &mut sum,
        ps,
    );

    covector.truncate(cs_witness.message.len());

    debug_assert_eq!(
        dot(&cs_witness.message, &covector),
        sum,
        "prove_whir_round exit: dot(message, covector) must equal sum"
    );

    ProverBlock::single_source(cs_witness.message, covector, sum, cs_witness.target_witness)
}

/// Verifier-side output of one round: the data needed by the caller to
/// accumulate implicit constraints and track per-round scale factors.
pub(crate) struct VerifyRoundOutput<F: Field> {
    pub(crate) round_challenges: Vec<F>,
    pub(crate) update_params: CovectorUpdateParams<F>,
}

#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, name = "zook::verify_whir_round", fields(msg_len = round.code_switch().source().message_length(), t = block.commitments.len())))]
pub(crate) fn verify_whir_round<F, H>(
    round: &RoundConfig<Identity<F>>,
    block: VerifierBlock<F>,
    vs: &mut VerifierState<H>,
) -> VerificationResult<(VerifierBlock<F>, VerifyRoundOutput<F>)>
where
    F: Field + Default + Codec<[H::U]>,
    Standard: Distribution<F>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let VerifierBlock {
        mut sum,
        commitments,
        theta,
    } = block;

    debug_assert_eq!(commitments.len(), theta.len());
    debug_assert!(!commitments.is_empty());

    let msg_len = round.code_switch().source().message_length();

    let mut masker = RoundMaskOracleCheck::begin(round, vs)?;
    let opening = round.sumcheck().verify(vs, &mut sum)?;
    masker.receive_cs_mask_and_reconcile(&opening, vs, &mut sum)?;

    let slot_weights = slot_weights::build(&theta, &opening.round_challenges);
    let commitment_refs: Vec<&IrsCommitment> = commitments.iter().collect();
    let (target_commitment, update_params) = round.code_switch().verify_virtual_for_implicit(
        vs,
        &mut sum,
        &commitment_refs,
        &slot_weights,
    )?;

    masker.verify_and_discharge(
        &opening.round_challenges,
        msg_len,
        &update_params,
        vs,
        &mut sum,
    )?;

    let next = VerifierBlock::single_source(sum, target_commitment);
    let out = VerifyRoundOutput {
        round_challenges: opening.round_challenges,
        update_params,
    };
    Ok((next, out))
}
