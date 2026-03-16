//! Base Case Linear Opening Protocol
//!
//! It is ZK but it is not Succinct.
//!
//! <https://eprint.iacr.org/2026/391.pdf> § 7.

use ark_ff::FftField;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};
use spongefish::{Decoding, VerificationResult};

use crate::{
    algebra::{dot, embedding::Identity, linear_form::Evaluate, multilinear_extend},
    hash::Hash,
    protocols::{irs_commit, sumcheck},
    transcript::{
        codecs::U64, Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerifierMessage,
        VerifierState,
    },
    utils::zip_strict,
    verify,
};

pub struct Config<F: FftField> {
    pub commit: irs_commit::Config<Identity<F>>,
    pub sumcheck: sumcheck::Config<F>,
}

impl<F: FftField> Config<F> {
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vector: Vec<F>,
        vector_witness: &irs_commit::Witness<F>,
        mut covector: Vec<F>,
        sum: F,
    ) -> (Vec<F>, F)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
        Standard: Distribution<F>,
    {
        debug_assert_eq!(dot(&vector, &covector), sum);

        // Create masking vectors.
        let mask = (0..vector.len())
            .map(|_| prover_state.rng().gen())
            .collect::<Vec<F>>();

        // Commit to the masking vectors.
        let mask_witness = self.commit.commit(prover_state, &[&mask]);

        // Compute and send linear form of mask (μ' in paper).
        let mask_sum = dot(&mask, &covector);
        prover_state.prover_message(&mask_sum);

        // RLC the mask with the vector
        let mask_rlc = prover_state.verifier_message::<F>();
        let mut masked_vector = zip_strict(vector.iter(), mask.iter())
            .map(|(v, m)| *v + mask_rlc * *m)
            .collect::<Vec<F>>();

        // Send masked vector in full.
        for v in masked_vector.iter() {
            prover_state.prover_message(v);
        }

        // Send combined IRS randomness. (r^* in paper)
        // TODO: Implement IRS randomness.

        // Open the commitment and mask simultaneously.
        let _ = self
            .commit
            .open(prover_state, &[&vector_witness, &mask_witness]);

        // Run sumcheck to reduce linear form claim
        let mut masked_sum = sum + mask_rlc * mask_sum;
        let point = self.sumcheck.prove(
            prover_state,
            &mut masked_vector,
            &mut covector,
            &mut masked_sum,
        );

        // Compute implied MLE of the linear form
        // f*(r) · l(r) = sum  =>  l(r) = sum / f*(r)
        let masked_mle = multilinear_extend(&masked_vector, &point.0);
        let linear_mle = sum / masked_mle;

        // TODO: Isn't this just covector[0]?
        assert_eq!(linear_mle, covector[0]);

        // Return evaluation point and value of the covector.
        (point.0, linear_mle)
    }

    pub fn verify<H, R>(
        &self,
        verifier_state: &mut VerifierState<H>,
        vector_commitment: irs_commit::Commitment<F>,
        sum: F,
    ) -> VerificationResult<(Vec<F>, F)>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let mask_commitment = self.commit.receive_commitment(verifier_state)?;
        let mask_sum: F = verifier_state.prover_message()?;
        let mask_rlc: F = verifier_state.verifier_message();
        let masked_vector: Vec<F> = verifier_state.prover_messages_vec(self.commit.vector_size)?;
        // TODO: Implement IRS randomness.

        // Open the commitment and mask simultaneously.
        let evals = self
            .commit
            .verify(verifier_state, &[&vector_commitment, &mask_commitment])?;

        // Spot check evaluations.
        for (point, value) in zip_strict(
            evals.evaluators(self.commit.vector_size),
            evals.values(&[F::ONE, mask_rlc]),
        ) {
            verify!(point.evaluate(&Identity::new(), &masked_vector) == value);
        }

        // Sumcheck on masked inner product
        let mut masked_sum = sum + mask_rlc * mask_sum;
        let point = self.sumcheck.verify(verifier_state, &mut masked_sum)?;

        // Compute implied MLE of the linear form
        // f*(r) · l(r) = sum  =>  l(r) = sum / f*(r)
        let masked_mle = multilinear_extend(&masked_vector, &point.0);
        let linear_mle = sum / masked_mle;

        Ok((point.0, linear_mle))
    }
}
