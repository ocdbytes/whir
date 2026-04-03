//! Code-switching IOR: R_{C, sl} → R_{C', sl'}
//!
//! Reduces a proximity claim about oracle f (source code C) to a proximity
//! claim about oracle g (target code C'). Non-ZK variant.
//!
//! Paper: Construction 9.7 (p.55), Theorem 9.6 (p.54), Lemma 9.9 (p.57)

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{
        dot,
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
        linear_form::UnivariateEvaluation,
        mixed_dot, mixed_univariate_evaluate, univariate_evaluate,
    },
    bits::Bits,
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, irs_commit, proof_of_work},
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Typed,
};

/// Prover output from the code-switch interaction phase.
/// Paper: Equation 9 (p.55)
#[derive(Clone, Debug)]
pub struct Witness<F: Field, G: Field = F> {
    /// Target oracle commitment data. G_target = F since target uses Identity<F>.
    pub target_witness: irs_commit::Witness<F>,
    /// The message f, taken by ownership from the caller.
    pub message: Vec<F>,
    /// Source oracle evaluations at in-domain query points (in M::Source = G).
    pub source_evaluations: irs_commit::Evaluations<G>,
}

/// Verifier output from the code-switch decision phase.
/// Paper: Equation 9 (p.55) — output (x', y')
#[derive(Clone, Debug)]
pub struct CodeSwitchClaim<F: Field, G: Field = F> {
    /// μ' — batched target value. Decision phase formula (p.55).
    pub mu_prime: F,
    /// ν_1 = batching_coeffs[0]. Caller scales original sl by this.
    pub original_sl_coeff: F,
    /// Target oracle commitment (g).
    pub target_commitment: irs_commit::Commitment<F>,
    /// RLC coefficients for constraint weights (batching_coeffs[1..]).
    pub constraint_rlc_coeffs: Vec<F>,
    /// OOD constraint weights — UnivariateEvaluation at each OOD point (M::Target).
    pub ood_constraint_weights: Vec<UnivariateEvaluation<F>>,
    /// In-domain constraint weights — UnivariateEvaluation at each source eval point (M::Source).
    pub in_domain_constraint_weights: Vec<UnivariateEvaluation<G>>,
    /// Source oracle evaluation data from in-domain queries.
    pub source_evaluations: irs_commit::Evaluations<G>,
}

/// Code-switching IOR config. Non-ZK variant (n=0, no C_zk masks).
///
/// TODO [ZK]: Add mask oracle, private zero-evaders, mask terms in sl'.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<M: Embedding> {
    pub embedding: Typed<M>,
    /// Source code C.
    pub source: irs_commit::Config<M>,
    /// Target code C'. Message length ℓ' = ℓ (same message).
    pub target: irs_commit::Config<Identity<M::Target>>,
    /// OOD samples tying source and target to the same message.
    pub cs_ood_samples: usize,
    // In-domain queries reuse source.in_domain_samples (same formula).
    // Reintroduce cs_in_domain_queries if security budgets diverge or mask_length > 0.
    /// Proof-of-work config (set by parent orchestrator).
    pub pow: proof_of_work::Config,
}

/// Squeeze batching randomness from the transcript.
/// Both prove() and verify() must call this at the same transcript position
/// to stay in sync. Returns (ν_1, batching_coeffs) where ν_1 scales the
/// original sl and batching_coeffs[i] scales the i-th constraint weight.
fn batching_challenge<T, F>(transcript: &mut T, count: usize) -> (F, Vec<F>)
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    let mut coeffs = geometric_challenge(transcript, count);
    let original_sl_coeff = if coeffs.is_empty() {
        F::ONE
    } else {
        coeffs.remove(0)
    };
    (original_sl_coeff, coeffs)
}

impl<M: Embedding> Config<M> {
    pub fn new(
        security_target: f64,
        unique_decoding: bool,
        hash_id: crate::engines::EngineId,
        source_config: irs_commit::Config<M>,
        target_log_inv_rate: usize,
        target_interleaving_depth: usize,
    ) -> Self
    where
        M: Default,
        M::Target: Default,
    {
        let source_message_length = source_config.message_length();

        // Target: ℓ' = ℓ, m' = ⌈ℓ'/ρ'⌉
        let target_message_length = source_message_length;
        let target_rate = 0.5_f64.powf(target_log_inv_rate as f64);
        let target_vector_size = target_message_length * target_interleaving_depth;
        let target_config = irs_commit::Config::<Identity<M::Target>>::new(
            security_target,
            unique_decoding,
            hash_id,
            1,
            target_vector_size,
            target_interleaving_depth,
            target_rate,
        );

        // OOD: solve |Λ(C',δ')|²/2 · ((ℓ-1)/|F|)^{t_ood} ≤ 2^{-λ}
        // Target list size, source message length (Lemma 9.9 Error 1, p.57)
        let cs_ood_samples = irs_commit::num_ood_samples(
            unique_decoding,
            security_target,
            M::Target::field_size_bits(),
            target_config.list_size(),
            source_message_length,
        );

        Self {
            embedding: Typed::<M>::default(),
            source: source_config,
            target: target_config,
            cs_ood_samples,
            pow: proof_of_work::Config {
                hash_id,
                threshold: proof_of_work::threshold(Bits::new(0.0)),
            },
        }
    }

    /// Prover interaction phase — Construction 9.7 Steps 1-4 (p.55).
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        message: Vec<M::Target>,
        source_randomness: &[M::Source],
        source_witness: &irs_commit::Witness<M::Source, M::Target>,
    ) -> Witness<M::Target, M::Source>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Step 1: g := Enc_{C'}(f, r')
        let target_witness = self.target.commit(prover_state, &[&message]);

        // Step 2: receive OOD challenge ρ_ood
        let ood_points: Vec<M::Target> = prover_state.verifier_message_vec(self.cs_ood_samples);

        // Step 3: send OOD answers y_i = f̂(ρ_i) + ρ_i^ℓ · r̂(ρ_i)
        let msg_len = message.len();
        for &point in &ood_points {
            let f_eval = univariate_evaluate(&message, point);
            let r_eval =
                mixed_univariate_evaluate(self.source.embedding(), source_randomness, point);
            let shift = point.pow([msg_len as u64]);
            prover_state.prover_message(&(f_eval + shift * r_eval));
            // TODO [ZK]: add ρ^{ℓ+r} · ŝ(ρ) term for mask oracle
        }

        // Step 4: open source oracle at in-domain queries
        let source_evaluations = self.source.open(prover_state, &[source_witness]);

        // Step 5: batching randomness (must match verify's batching_challenge call)
        let _batching = batching_challenge::<_, M::Target>(
            prover_state,
            1 + ood_points.len() + source_evaluations.matrix.len(),
        );

        Witness {
            target_witness,
            message,
            source_evaluations,
        }
    }

    /// Verifier decision phase — Construction 9.7 Steps 1-4 + Decision (p.55).
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        mu: M::Target,
        source_commitment: &irs_commit::Commitment<M::Target>,
    ) -> VerificationResult<CodeSwitchClaim<M::Target, M::Source>>
    where
        H: DuplexSpongeInterface,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Step 1: receive target commitment
        let target_commitment = self.target.receive_commitment(verifier_state)?;

        // Step 2-3: OOD challenge + answers
        let ood_points: Vec<M::Target> = verifier_state.verifier_message_vec(self.cs_ood_samples);
        let ood_answers: Vec<M::Target> =
            verifier_state.prover_messages_vec(self.cs_ood_samples)?;

        // Step 4: verify source oracle openings
        let source_evaluations = self.source.verify(verifier_state, &[source_commitment])?;

        // Step 5: batching coefficients (shared helper ensures transcript sync with prover)
        let t_ood = ood_points.len();
        let t_times_iota = source_evaluations.matrix.len();
        let (original_sl_coeff, constraint_rlc_coeffs) =
            batching_challenge(verifier_state, 1 + t_ood + t_times_iota);

        // Step 6: μ' = ν_1·μ + Σ ν_{1+i}·y_i + ΣΣ ν_{...}·f(x_i)_l
        let mu_prime = original_sl_coeff * mu
            + dot(&constraint_rlc_coeffs[..t_ood], &ood_answers)
            + mixed_dot(
                self.source.embedding(),
                &constraint_rlc_coeffs[t_ood..],
                &source_evaluations.matrix,
            );

        // Step 7: constraint weights for sl'
        let source_msg_len = self.source.message_length();

        let ood_constraint_weights: Vec<UnivariateEvaluation<M::Target>> = ood_points
            .iter()
            .map(|&point| UnivariateEvaluation::new(point, source_msg_len))
            .collect();

        let in_domain_constraint_weights: Vec<UnivariateEvaluation<M::Source>> =
            source_evaluations.evaluators(source_msg_len).collect();

        Ok(CodeSwitchClaim {
            mu_prime,
            original_sl_coeff,
            target_commitment,
            constraint_rlc_coeffs,
            ood_constraint_weights,
            in_domain_constraint_weights,
            source_evaluations,
        })
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng,
    };
    use proptest::{proptest, sample::select};

    use super::*;
    use crate::{
        algebra::{embedding::Identity, fields, linear_form::Evaluate, ntt, random_vector},
        transcript::{codecs::U64, DomainSeparator},
    };

    /// Run full code switch (prove + verify) and check μ' completeness.
    /// Uses ι=1 (num_vectors=1, interleaving_depth=1) so weight count matches RLC count.
    fn test_completeness<F: Field + FieldWithSize + Codec<[u8]> + 'static>(
        seed: u64,
        source_vector_size: usize,
        source_log_inv_rate: usize,
        target_log_inv_rate: usize,
    ) where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        let security_target = 32.0;
        let unique_decoding = false;
        #[allow(clippy::cast_possible_wrap)]
        let source_rate = 0.5_f64.powi(source_log_inv_rate as i32);

        let source_config = irs_commit::Config::<Identity<F>>::new(
            security_target,
            unique_decoding,
            crate::hash::BLAKE3,
            1,
            source_vector_size,
            1,
            source_rate,
        );
        let cs_config = Config::<Identity<F>>::new(
            security_target,
            unique_decoding,
            crate::hash::BLAKE3,
            source_config.clone(),
            target_log_inv_rate,
            1,
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let message: Vec<F> = random_vector(&mut rng, source_config.message_length());
        let mu: F = rng.gen();

        let instance = U64(seed);
        let ds = DomainSeparator::protocol(&cs_config)
            .session(&String::from("code_switch_proptest"))
            .instance(&instance);

        // Prover
        let mut prover_state = ProverState::new_std(&ds);
        let source_witness = source_config.commit(&mut prover_state, &[&message]);
        let cs_witness = cs_config.prove(
            &mut prover_state,
            message.clone(),
            &source_witness.masks,
            &source_witness,
        );
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let source_commitment = source_config
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let claim = cs_config
            .verify(&mut verifier_state, mu, &source_commitment)
            .unwrap();
        verifier_state.check_eof().unwrap();

        // Witness carries the message
        assert_eq!(cs_witness.message, message);

        // μ' completeness: Σ rlc[i] · weight[i].evaluate(f) == μ' - ν_1·μ
        let embedding = Identity::<F>::new();
        let t_ood = claim.ood_constraint_weights.len();

        let weights_on_f: F = claim
            .ood_constraint_weights
            .iter()
            .enumerate()
            .map(|(i, w)| claim.constraint_rlc_coeffs[i] * w.evaluate(&embedding, &message))
            .chain(
                claim
                    .in_domain_constraint_weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| {
                        claim.constraint_rlc_coeffs[t_ood + j] * w.evaluate(&embedding, &message)
                    }),
            )
            .sum();

        assert_eq!(
            weights_on_f,
            claim.mu_prime - claim.original_sl_coeff * mu,
            "Constraint weights evaluated on f should equal μ' - ν_1·μ"
        );
        assert_eq!(
            claim.constraint_rlc_coeffs.len(),
            claim.ood_constraint_weights.len() + claim.in_domain_constraint_weights.len(),
        );
    }

    fn test<F: Field + FieldWithSize + Codec<[u8]> + 'static>()
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        // Valid sizes: NTT-friendly AND power-of-two (challenge_indices requires pow2 codeword)
        let valid_sizes: Vec<usize> = (3..=8)
            .map(|k| 1_usize << k) // 8, 16, 32, 64, 128, 256
            .filter(|&n| ntt::next_order::<F>(n) == Some(n))
            .collect();
        assert!(!valid_sizes.is_empty(), "No valid NTT sizes for field");

        let size = select(valid_sizes);
        let log_inv_rates = select(vec![1_usize, 2, 3]); // rates 1/2, 1/4, 1/8

        proptest!(|(
            seed: u64,
            source_size in size,
            src_lir in log_inv_rates.clone(),
            tgt_lir in log_inv_rates,
        )| {
            test_completeness::<F>(seed, source_size, src_lir, tgt_lir);
        });
    }

    #[test]
    fn test_field64() {
        test::<fields::Field64>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_2() {
        test::<fields::Field64_2>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_3() {
        test::<fields::Field64_3>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        test::<fields::Field128>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field192() {
        test::<fields::Field192>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        test::<fields::Field256>();
    }
}
