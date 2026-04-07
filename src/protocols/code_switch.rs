//! Code-switching IOR: R_{C, C_zk, sl} → R_{C', C_zk, sl'}
//!
//! Reduces a proximity claim about oracle f (source code C) to a proximity
//! claim about oracle g (target code C'). Supports optional ZK via mask oracle.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        dot,
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
        linear_form::UnivariateEvaluation,
        mixed_dot, mixed_univariate_evaluate, random_vector, univariate_evaluate,
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

/// Prover output. Paper: Equation 9 (p.55).
#[derive(Clone, Debug)]
pub struct Witness<F: Field, G: Field = F> {
    pub target_witness: irs_commit::Witness<F>,
    pub message: Vec<F>,
    pub source_evaluations: irs_commit::Evaluations<G>,
    pub mask_witness: Option<irs_commit::Witness<F>>,
    pub mask_message: Option<Vec<F>>,
}

/// OOD and in-domain constraint weights for a single oracle (message or mask).
#[derive(Clone, Debug)]
pub struct ConstraintWeights<F: Field, G: Field = F> {
    pub ood: Vec<UnivariateEvaluation<F>>,
    pub in_domain: Vec<UnivariateEvaluation<G>>,
}

/// Verifier output. Paper: Equation 9 (p.55).
#[derive(Clone, Debug)]
pub struct CodeSwitchClaim<F: Field, G: Field = F> {
    pub mu_prime: F,
    pub original_sl_coeff: F,
    pub target_commitment: irs_commit::Commitment<F>,
    pub mask_commitment: Option<irs_commit::Commitment<F>>,
    pub constraint_rlc_coeffs: Vec<F>,
    /// Message weights: ze_ood^{←,i} and G_C^#[x,·]
    pub source_weights: ConstraintWeights<F, G>,
    /// Mask weights: ze_ood^{→,i} and G_C^s[x,·]. None if non-ZK.
    pub mask_weights: Option<ConstraintWeights<F, G>>,
    pub source_evaluations: irs_commit::Evaluations<G>,
}

/// Code-switching IOR config with optional ZK.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<M: Embedding> {
    pub embedding: Typed<M>,
    pub source: irs_commit::Config<M>,
    pub target: irs_commit::Config<Identity<M::Target>>,
    pub cs_ood_samples: usize,
    pub pow: proof_of_work::Config,
    pub mask: Option<MaskConfig<M>>,
}

/// ZK mask code config (C_zk).
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MaskConfig<M: Embedding> {
    pub commit: irs_commit::Config<Identity<M::Target>>,
    pub num_masks: usize,
}

/// Shared transcript operation for batching randomness.
fn batching_challenge<T, F>(transcript: &mut T, count: usize) -> (F, Vec<F>)
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    let coeffs = geometric_challenge(transcript, count);
    match coeffs.split_first() {
        Some((&first, rest)) => (first, rest.to_vec()),
        None => (F::ONE, Vec::new()),
    }
}

impl<M: Embedding> Config<M> {
    pub fn new(
        security_target: f64,
        unique_decoding: bool,
        hash_id: crate::engines::EngineId,
        source_config: irs_commit::Config<M>,
        target_log_inv_rate: usize,
        target_interleaving_depth: usize,
        masked: bool,
    ) -> Self
    where
        M: Default,
        M::Target: Default,
    {
        let source_message_length = source_config.message_length();

        let target_rate = 0.5_f64.powf(target_log_inv_rate as f64);
        let target_vector_size = source_message_length * target_interleaving_depth;
        let mut target_config = irs_commit::Config::<Identity<M::Target>>::new(
            security_target,
            unique_decoding,
            hash_id,
            1,
            target_vector_size,
            target_interleaving_depth,
            target_rate,
        );

        let mask = if masked {
            let source_randomness_len = source_config.mask_length * source_config.num_messages();
            let mask_msg_len = source_randomness_len.max(1);
            let mut mask_commit = irs_commit::Config::<Identity<M::Target>>::new(
                security_target,
                unique_decoding,
                hash_id,
                1,
                mask_msg_len,
                1,
                source_config.rate(),
            );
            // ZK encoding: mask_length ≥ queries for ζ = 0 (Prop 3.19, p.30)
            target_config.mask_length = target_config.out_domain_samples;
            mask_commit.mask_length = mask_commit.out_domain_samples;
            Some(MaskConfig {
                commit: mask_commit,
                num_masks: 1,
            })
        } else {
            None
        };

        // OOD: Lemma 9.9 Error 1
        let (ood_list_size, ood_degree) = mask.as_ref().map_or_else(
            || (target_config.list_size(), source_message_length),
            |m| {
                (
                    target_config.list_size() * m.commit.list_size(),
                    source_message_length + m.commit.message_length(),
                )
            },
        );
        let cs_ood_samples = irs_commit::num_ood_samples(
            unique_decoding,
            security_target,
            M::Target::field_size_bits(),
            ood_list_size,
            ood_degree,
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
            mask,
        }
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
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
        debug_assert_eq!(message.len(), self.source.message_length());
        debug_assert_eq!(
            source_randomness.len(),
            self.source.mask_length * self.source.num_messages()
        );

        // Step 1a: g := Enc_{C'}(f, r')
        let target_witness = self.target.commit(prover_state, &[&message]);

        // Step 1b: s := Enc_{C_zk}((r, s_leftover), r'')
        let (mask_message, mask_witness) = self.mask.as_ref().map_or((None, None), |mask_config| {
            let mask_msg_len = mask_config.commit.message_length();
            let embedding = self.source.embedding();
            let r_embedded: Vec<M::Target> = source_randomness
                .iter()
                .map(|&x| embedding.map(x))
                .collect();
            let r_len = r_embedded.len();
            let mut mask_msg = Vec::with_capacity(mask_msg_len);
            mask_msg.extend_from_slice(&r_embedded);
            let s_leftover: Vec<M::Target> =
                random_vector(prover_state.rng(), mask_msg_len - r_len);
            mask_msg.extend_from_slice(&s_leftover);
            let witness = mask_config.commit.commit(prover_state, &[&mask_msg]);
            (Some(mask_msg), Some(witness))
        });

        // Step 2: OOD challenge
        let ood_points: Vec<M::Target> = prover_state.verifier_message_vec(self.cs_ood_samples);

        // Step 3: OOD answers
        let msg_len = message.len();
        for &point in &ood_points {
            let f_eval = univariate_evaluate(&message, point);
            let randomness_eval = mask_message.as_ref().map_or_else(
                || mixed_univariate_evaluate(self.source.embedding(), source_randomness, point),
                |mask_msg| univariate_evaluate(mask_msg, point),
            );
            let shift = point.pow([msg_len as u64]);
            prover_state.prover_message(&(f_eval + shift * randomness_eval));
        }

        // Step 4: open source oracle
        let source_evaluations = self.source.open(prover_state, &[source_witness]);

        // Step 5: batching randomness (sync with verify)
        let _batching = batching_challenge::<_, M::Target>(
            prover_state,
            1 + ood_points.len() + source_evaluations.matrix.len(),
        );

        Witness {
            target_witness,
            message,
            source_evaluations,
            mask_witness,
            mask_message,
        }
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
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
        debug_assert!(
            self.mask.is_some() == (self.target.mask_length > 0),
            "mask config and target mask_length must agree"
        );

        // Step 1
        let target_commitment = self.target.receive_commitment(verifier_state)?;
        let mask_commitment = self
            .mask
            .as_ref()
            .map(|m| m.commit.receive_commitment(verifier_state))
            .transpose()?;

        // Step 2-3: OOD
        let ood_points: Vec<M::Target> = verifier_state.verifier_message_vec(self.cs_ood_samples);
        let ood_answers: Vec<M::Target> =
            verifier_state.prover_messages_vec(self.cs_ood_samples)?;

        // Step 4: source opening
        let source_evaluations = self.source.verify(verifier_state, &[source_commitment])?;

        // Step 5: batching
        let t_ood = ood_points.len();
        let t_times_iota = source_evaluations.matrix.len();
        let (original_sl_coeff, constraint_rlc_coeffs) =
            batching_challenge(verifier_state, 1 + t_ood + t_times_iota);

        // Step 6: μ'
        let mu_prime = original_sl_coeff * mu
            + dot(&constraint_rlc_coeffs[..t_ood], &ood_answers)
            + mixed_dot(
                self.source.embedding(),
                &constraint_rlc_coeffs[t_ood..],
                &source_evaluations.matrix,
            );

        // Step 7: constraint weights
        let source_msg_len = self.source.message_length();

        let source_weights = ConstraintWeights {
            ood: ood_points
                .iter()
                .map(|&point| UnivariateEvaluation::new(point, source_msg_len))
                .collect(),
            in_domain: source_evaluations.evaluators(source_msg_len).collect(),
        };

        let mask_weights = self.mask.as_ref().map(|mask_config| {
            let mask_msg_len = mask_config.commit.message_length();
            ConstraintWeights {
                ood: ood_points
                    .iter()
                    .map(|&point| UnivariateEvaluation::new(point, mask_msg_len))
                    .collect(),
                in_domain: source_evaluations.evaluators(mask_msg_len).collect(),
            }
        });

        Ok(CodeSwitchClaim {
            mu_prime,
            original_sl_coeff,
            target_commitment,
            mask_commitment,
            constraint_rlc_coeffs,
            source_weights,
            mask_weights,
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

        let mut source_config = irs_commit::Config::<Identity<F>>::new(
            security_target,
            unique_decoding,
            crate::hash::BLAKE3,
            1,
            source_vector_size,
            1,
            source_rate,
        );
        // Source needs mask_length > 0 for ZK (randomness r to hide in mask oracle).
        // In production, set by the parent protocol. Here we use 1 — the minimum
        // nontrivial value that exercises the ZK path without NTT sizing issues.
        source_config.mask_length = 1;
        let cs_config = Config::<Identity<F>>::new(
            security_target,
            unique_decoding,
            crate::hash::BLAKE3,
            source_config.clone(),
            target_log_inv_rate,
            1,
            true, // ZK enbled
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

        assert_eq!(cs_witness.message, message);

        check_mu_prime_completeness(
            &claim,
            &message,
            mu,
            cs_witness.mask_message.as_deref(),
            source_witness.masks.len(),
        );
    }

    /// Verify μ' = ν_1·μ + ⟨f, message_weights⟩ + ⟨mask_msg, mask_weights⟩
    fn check_mu_prime_completeness<F: Field>(
        claim: &CodeSwitchClaim<F>,
        message: &[F],
        mu: F,
        mask_msg: Option<&[F]>,
        r_len: usize,
    ) {
        let embedding = Identity::<F>::new();
        let t_ood = claim.source_weights.ood.len();

        // Message part: Σ rlc[i] · weight[i].evaluate(f)
        let weights_on_f: F = claim
            .source_weights
            .ood
            .iter()
            .enumerate()
            .map(|(i, w)| claim.constraint_rlc_coeffs[i] * w.evaluate(&embedding, message))
            .chain(
                claim
                    .source_weights
                    .in_domain
                    .iter()
                    .enumerate()
                    .map(|(j, w)| {
                        claim.constraint_rlc_coeffs[t_ood + j] * w.evaluate(&embedding, message)
                    }),
            )
            .sum();

        // Mask part (ZK only): Σ rlc[i] · mask_weight[i].evaluate(mask_msg)
        let weights_on_mask: F =
            mask_msg
                .zip(claim.mask_weights.as_ref())
                .map_or(F::ZERO, |(mask_msg, mask_w)| {
                    let msg_len = message.len();
                    let ood_sum: F = mask_w
                        .ood
                        .iter()
                        .enumerate()
                        .map(|(i, w)| {
                            let shift = w.point.pow([msg_len as u64]);
                            claim.constraint_rlc_coeffs[i]
                                * shift
                                * w.evaluate(&embedding, mask_msg)
                        })
                        .sum();
                    let in_domain_sum: F = mask_w
                        .in_domain
                        .iter()
                        .enumerate()
                        .map(|(j, w)| {
                            let shift = w.point.pow([msg_len as u64]);
                            let r_eval = if r_len > 0 {
                                w.evaluate(&embedding, &mask_msg[..r_len])
                            } else {
                                F::ZERO
                            };
                            claim.constraint_rlc_coeffs[t_ood + j] * shift * r_eval
                        })
                        .sum();
                    ood_sum + in_domain_sum
                });

        assert_eq!(
            weights_on_f + weights_on_mask,
            claim.mu_prime - claim.original_sl_coeff * mu,
            "Message + mask weights should equal μ' - ν_1·μ"
        );
        assert_eq!(
            claim.constraint_rlc_coeffs.len(),
            claim.source_weights.ood.len() + claim.source_weights.in_domain.len(),
        );
    }

    fn test<F: Field + FieldWithSize + Codec<[u8]> + 'static>()
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        let valid_sizes: Vec<usize> = (3..=8)
            .map(|k| 1_usize << k)
            .filter(|&n| ntt::next_order::<F>(n) == Some(n))
            .collect();
        assert!(!valid_sizes.is_empty(), "No valid NTT sizes for field");

        let size = select(valid_sizes);
        let log_inv_rates = select(vec![1_usize, 2, 3]);

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
