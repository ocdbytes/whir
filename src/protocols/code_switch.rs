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
        lift,
        linear_form::UnivariateEvaluation,
        mixed_dot, random_vector, univariate_evaluate,
    },
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge,
        irs_commit::{
            num_ood_samples, Commitment as IrsCommitment, Config as IrsConfig,
            Evaluations as IrsEvaluations, Witness as IrsWitness,
        },
    },
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Typed,
};

/// Prover output from the code-switch.
#[derive(Clone, Debug)]
pub struct Witness<F: Field, G: Field = F> {
    pub target_witness: IrsWitness<F>,
    pub message: Vec<F>,
    pub source_evaluations: IrsEvaluations<G>,
    pub mask_witness: Option<IrsWitness<F>>,
    pub mask_message: Option<Vec<F>>,
}

/// OOD and in-domain constraint weights for a single oracle.
#[derive(Clone, Debug)]
pub struct ConstraintWeights<F: Field, G: Field = F> {
    pub ood: Vec<UnivariateEvaluation<F>>,
    pub in_domain: Vec<UnivariateEvaluation<G>>,
}

/// ZK mask oracle claim. OOD weights evaluate full mask; in-domain weights
/// evaluate only `mask_msg[..source_randomness_len]` (paper: ⟨(r,s), (G_C^s[x,·], 0)⟩).
#[derive(Clone, Debug)]
pub struct MaskClaimInfo<F: Field, G: Field = F> {
    pub commitment: IrsCommitment<F>,
    pub ood_rlc_coeffs: Vec<F>,
    pub in_domain_rlc_coeffs: Vec<F>,
    pub weights: ConstraintWeights<F, G>,
    pub source_randomness_len: usize,
}

impl<F: Field, G: Field> MaskClaimInfo<F, G> {
    /// Weighted mask contribution. In-domain points are lifted G → F via embedding.
    pub fn evaluate(
        &self,
        embedding: &impl Embedding<Source = G, Target = F>,
        mask_msg: &[F],
    ) -> F {
        // OOD: ⟨(r,s), ze^{→,i}_ood(ρ)⟩ = ρ^ℓ · (r,s)^(ρ)
        let ood_sum: F = self
            .ood_rlc_coeffs
            .iter()
            .zip(&self.weights.ood)
            .map(|(&c, w)| c * univariate_evaluate(mask_msg, w.point))
            .sum();

        // In-domain: ⟨(r,s), (G^s_C[x,·], 0)⟩ = φ(x)^ℓ · r̂(φ(x))
        let r_slice = &mask_msg[..self.source_randomness_len];
        let in_domain_sum: F = self
            .in_domain_rlc_coeffs
            .iter()
            .zip(&self.weights.in_domain)
            .map(|(&c, w)| c * univariate_evaluate(r_slice, embedding.map(w.point)))
            .sum();

        ood_sum + in_domain_sum
    }
}

/// Verifier output. Paper: Equation 9 (p.55).
#[derive(Clone, Debug)]
pub struct CodeSwitchClaim<F: Field, G: Field = F> {
    pub mu_prime: F,
    pub original_sl_coeff: F,
    pub target_commitment: IrsCommitment<F>,
    pub ood_rlc_coeffs: Vec<F>,
    pub in_domain_rlc_coeffs: Vec<F>,
    pub source_weights: ConstraintWeights<F, G>,
    pub mask_info: Option<MaskClaimInfo<F, G>>,
    pub source_evaluations: IrsEvaluations<G>,
}

/// Next stage's query budgets for ZK encoding (Prop 3.19).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub struct ZkQueryBudget {
    /// Queries the next stage makes to the target oracle g.
    pub target: usize,
    /// Queries the next stage makes to the mask oracle s.
    pub mask: usize,
}

/// Code-switching IOR config with optional ZK.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<M: Embedding> {
    pub embedding: Typed<M>,
    pub source: IrsConfig<M>,
    pub target: IrsConfig<Identity<M::Target>>,
    pub cs_ood_samples: usize,
    pub mask_commit: Option<IrsConfig<Identity<M::Target>>>,
}

/// Split geometric challenge into (ν_1, rest). Requires count ≥ 1.
fn batching_challenge<T, F>(transcript: &mut T, count: usize) -> (F, Vec<F>)
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    assert!(count > 0, "batching requires at least one coefficient");
    let mut coeffs = geometric_challenge(transcript, count);
    let first = coeffs[0];
    let rest = coeffs.split_off(1);
    (first, rest)
}

impl<M: Embedding> Config<M> {
    /// `zk`: `None` for non-ZK, `Some(budget)` for ZK with next stage's query budgets.
    pub fn new(
        security_target: f64,
        unique_decoding: bool,
        hash_id: crate::engines::EngineId,
        source_config: IrsConfig<M>,
        target_log_inv_rate: usize,
        zk: Option<ZkQueryBudget>,
    ) -> Self
    where
        M: Default,
        M::Target: Default,
    {
        assert!(target_log_inv_rate > 0);

        let source_message_length = source_config.message_length();
        let target_rate = 0.5_f64.powf(target_log_inv_rate as f64);

        let mut target_config = IrsConfig::<Identity<M::Target>>::new(
            security_target,
            unique_decoding,
            hash_id,
            1,
            source_message_length,
            1,
            target_rate,
        );

        let mask_commit = zk.map(|budget| {
            assert!(budget.target > 0, "ZK requires nonzero target query budget");
            assert!(budget.mask > 0, "ZK requires nonzero mask query budget");
            target_config.mask_length = budget.target;
            target_config.reparameterise_security(security_target, unique_decoding);

            let source_randomness_len = source_config.mask_length * source_config.num_messages();
            assert!(
                source_randomness_len > 0,
                "ZK code-switch requires source_config.mask_length > 0"
            );
            // TODO : move this config out as this will be a shared C_zk config
            // across the protocol
            let mut mask_config = IrsConfig::<Identity<M::Target>>::new(
                security_target,
                unique_decoding,
                hash_id,
                1,
                source_randomness_len,
                1,
                source_config.rate(),
            );
            mask_config.mask_length = budget.mask;
            mask_config.reparameterise_security(security_target, unique_decoding);
            mask_config
        });

        let mut config = Self {
            embedding: Typed::<M>::default(),
            source: source_config,
            target: target_config,
            cs_ood_samples: 0,
            mask_commit,
        };
        let (list_size, degree) = config.list_size_and_degree();
        config.cs_ood_samples = num_ood_samples(
            unique_decoding,
            security_target,
            M::Target::field_size_bits(),
            list_size,
            degree,
        );
        config
    }

    /// RBR soundness of the OOD round in bits (Lemma 9.9 Error 1).
    pub fn rbr_ood(&self) -> f64 {
        if self.cs_ood_samples == 0 {
            return f64::INFINITY;
        }
        let (list_size, degree) = self.list_size_and_degree();
        let l_choose_2 = list_size * (list_size - 1.) / 2.;
        let log_per_sample = M::Target::field_size_bits() - ((degree - 1) as f64).log2();
        -l_choose_2.log2() + self.cs_ood_samples as f64 * log_per_sample
    }

    /// RBR soundness of the in-domain round in bits (Lemma 9.9 Error 2).
    pub fn rbr_in_domain(&self) -> f64 {
        self.source.rbr_queries()
    }

    /// RBR soundness of the batching round in bits (Lemma 9.9 Error 3).
    /// `n` = number of input mask oracles from previous stages.
    /// Total mask oracles = n + 1 (input + new s), interleaved as C_zk^{≡n+1}.
    pub fn rbr_batching(&self, n: usize) -> f64 {
        let field_bits = M::Target::field_size_bits();
        let target_log_list = self.target.list_size().log2();

        // Mask interleaving: |Λ(C_zk^{≡n+1}, δ_zk)| ≤ (n+1) · |Λ(C_zk, δ_zk)| (Lemma 3.15)
        let mask_log_list = self.mask_commit.as_ref().map_or(0., |m| {
            let k = (n + 1) as f64;
            (k * m.list_size()).log2()
        });

        field_bits - target_log_list - mask_log_list
    }

    /// Combined list-size bound and polynomial degree for OOD sampling.
    fn list_size_and_degree(&self) -> (f64, usize) {
        self.mask_commit.as_ref().map_or_else(
            || (self.target.list_size(), self.source.message_length()),
            |m| {
                (
                    self.target.list_size() * m.list_size(),
                    self.source.message_length() + m.message_length(),
                )
            },
        )
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        message: Vec<M::Target>,
        source_randomness: &[M::Source],
        source_witness: &IrsWitness<M::Source, M::Target>,
    ) -> Witness<M::Target, M::Source>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(message.len(), self.source.message_length());
        assert_eq!(
            source_randomness.len(),
            self.source.mask_length * self.source.num_messages()
        );
        assert!(
            self.mask_commit.is_some() == (self.target.mask_length > 0),
            "mask config and target mask_length must agree"
        );

        // Step 1a: g := Enc_{C'}(f, r')
        let target_witness = self.target.commit(prover_state, &[&message]);

        // Step 1b: s := Enc_{C_zk}((r || padding), r'')
        let (mask_message, mask_witness) =
            self.mask_commit
                .as_ref()
                .map_or((None, None), |mask_config| {
                    let mask_msg_len = mask_config.message_length();
                    let r_embedded = lift(self.source.embedding(), source_randomness);
                    let embedded_randomness_len = r_embedded.len();
                    let mut mask_msg = Vec::with_capacity(mask_msg_len);
                    mask_msg.extend_from_slice(&r_embedded);
                    let random_padding: Vec<M::Target> =
                        random_vector(prover_state.rng(), mask_msg_len - embedded_randomness_len);
                    mask_msg.extend_from_slice(&random_padding);
                    let witness = mask_config.commit(prover_state, &[&mask_msg]);
                    (Some(mask_msg), Some(witness))
                });

        // Step 2-3: OOD challenge + answers
        let ood_points: Vec<M::Target> = prover_state.verifier_message_vec(self.cs_ood_samples);
        let msg_len = message.len();
        for &point in &ood_points {
            let f_eval = univariate_evaluate(&message, point);
            if source_randomness.is_empty() {
                // Non-ZK: codeword encodes only f, so y_i = f̂(ρ)
                prover_state.prover_message(&f_eval);
            } else {
                // ZK: codeword encodes h(X) = f̂(X) + X^ℓ · r̂(X), so y_i = h(ρ)
                let mask_msg = mask_message
                    .as_ref()
                    .expect("ZK code-switch requires mask_message");
                let randomness_eval = univariate_evaluate(mask_msg, point);
                let shift = point.pow([msg_len as u64]);
                prover_state.prover_message(&(f_eval + shift * randomness_eval));
            }
        }

        // Step 4: in-domain queries
        let source_evaluations = self.source.open(prover_state, &[source_witness]);

        // Step 5: batching (advance transcript to stay in sync with verify)
        batching_challenge::<_, M::Target>(
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
        source_commitment: &IrsCommitment<M::Target>,
    ) -> VerificationResult<CodeSwitchClaim<M::Target, M::Source>>
    where
        H: DuplexSpongeInterface,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert!(
            self.mask_commit.is_some() == (self.target.mask_length > 0),
            "mask config and target mask_length must agree"
        );

        // Step 1: commitments
        let target_commitment = self.target.receive_commitment(verifier_state)?;
        let mask_commitment = self
            .mask_commit
            .as_ref()
            .map(|mask_cfg| mask_cfg.receive_commitment(verifier_state))
            .transpose()?;

        // Step 2-3: OOD
        let ood_points: Vec<M::Target> = verifier_state.verifier_message_vec(self.cs_ood_samples);
        let ood_answers: Vec<M::Target> =
            verifier_state.prover_messages_vec(self.cs_ood_samples)?;

        // Step 4: source opening
        let source_evaluations = self.source.verify(verifier_state, &[source_commitment])?;

        // Step 5: batching
        let num_ood = ood_points.len();
        let num_in_domain = source_evaluations.matrix.len();
        let (original_sl_coeff, all_rlc_coeffs) =
            batching_challenge(verifier_state, 1 + num_ood + num_in_domain);
        let (ood_rlc_coeffs, in_domain_rlc_coeffs) = all_rlc_coeffs.split_at(num_ood);

        // Step 6: μ'
        let mu_prime = original_sl_coeff * mu
            + dot(ood_rlc_coeffs, &ood_answers)
            + mixed_dot(
                self.source.embedding(),
                in_domain_rlc_coeffs,
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

        // Mask weights with shift ρ^ℓ baked into RLC coefficients.
        // In-domain shift lifted G → F via embedding (ring homomorphism).
        let mask_info = self.mask_commit.as_ref().map(|mask_config| {
            let source_randomness_len = self.source.mask_length * self.source.num_messages();
            let weights = ConstraintWeights {
                ood: ood_points
                    .iter()
                    .map(|&point| UnivariateEvaluation::new(point, mask_config.message_length()))
                    .collect(),
                in_domain: source_evaluations
                    .evaluators(source_randomness_len)
                    .collect(),
            };
            let embedding = self.source.embedding();
            let mask_ood_rlc_coeffs: Vec<M::Target> = ood_rlc_coeffs
                .iter()
                .zip(&weights.ood)
                .map(|(&c, w)| c * w.point.pow([source_msg_len as u64]))
                .collect();
            let mask_in_domain_rlc_coeffs: Vec<M::Target> = in_domain_rlc_coeffs
                .iter()
                .zip(&weights.in_domain)
                .map(|(&c, w)| c * embedding.map(w.point.pow([source_msg_len as u64])))
                .collect();

            MaskClaimInfo {
                commitment: mask_commitment.expect("mask commitment must exist when masked"),
                ood_rlc_coeffs: mask_ood_rlc_coeffs,
                in_domain_rlc_coeffs: mask_in_domain_rlc_coeffs,
                weights,
                source_randomness_len,
            }
        });

        Ok(CodeSwitchClaim {
            mu_prime,
            original_sl_coeff,
            target_commitment,
            ood_rlc_coeffs: ood_rlc_coeffs.to_vec(),
            in_domain_rlc_coeffs: in_domain_rlc_coeffs.to_vec(),
            source_weights,
            mask_info,
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

    /// Σ rlc[i] * weight[i].evaluate(vector)
    fn weighted_eval<F: Field>(rlc: &[F], weights: &[UnivariateEvaluation<F>], v: &[F]) -> F {
        let e = Identity::<F>::new();
        rlc.iter()
            .zip(weights)
            .map(|(&c, w)| c * w.evaluate(&e, v))
            .sum()
    }

    struct TestResult<F: Field> {
        claim: CodeSwitchClaim<F>,
        message: Vec<F>,
        target_mu: F,
        mask_message: Option<Vec<F>>,
    }

    fn run_code_switch<F: Field + FieldWithSize + Codec<[u8]> + 'static>(
        seed: u64,
        size: usize,
        src_lir: usize, // source log inverse rate
        tgt_lir: usize, // target log inverse rate
        masked: bool,
    ) -> TestResult<F>
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        #[allow(clippy::cast_possible_wrap)]
        let rate = 0.5_f64.powi(src_lir as i32);
        let mut source_config =
            IrsConfig::<Identity<F>>::new(32.0, false, crate::hash::BLAKE3, 1, size, 1, rate);
        if masked {
            source_config.mask_length = 1;
        }

        let zk = if masked {
            Some(ZkQueryBudget { target: 1, mask: 1 })
        } else {
            None
        };
        let cs_config = Config::<Identity<F>>::new(
            32.0,
            false,
            crate::hash::BLAKE3,
            source_config.clone(),
            tgt_lir,
            zk,
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let message: Vec<F> = random_vector(&mut rng, source_config.message_length());
        let target_mu: F = rng.gen();
        let instance = U64(seed);
        let domain_separator = DomainSeparator::protocol(&cs_config)
            .session(&"cs_test")
            .instance(&instance);

        let mut prover_state = ProverState::new_std(&domain_separator);
        let source_witness = source_config.commit(&mut prover_state, &[&message]);
        let cs_witness = cs_config.prove(
            &mut prover_state,
            message.clone(),
            &source_witness.masks,
            &source_witness,
        );

        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&domain_separator, &proof);
        let source_commitment = source_config
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let claim = cs_config
            .verify(&mut verifier_state, target_mu, &source_commitment)
            .unwrap();
        verifier_state.check_eof().unwrap();
        assert_eq!(cs_witness.message, message);

        TestResult {
            claim,
            message,
            target_mu,
            mask_message: cs_witness.mask_message,
        }
    }

    /// Completeness check (p.56): μ' = ν_1·μ + ⟨f, source_weights⟩ + ⟨mask_msg, mask_weights⟩
    fn check_completeness<F: Field>(result: &TestResult<F>, masked: bool) {
        let claim = &result.claim;

        // ⟨f, ze_ood^{←,i}⟩ + ⟨f, G_C^#[x,·]⟩ (message part of μ' decomposition)
        let msg_sum = weighted_eval(
            &claim.ood_rlc_coeffs,
            &claim.source_weights.ood,
            &result.message,
        ) + weighted_eval(
            &claim.in_domain_rlc_coeffs,
            &claim.source_weights.in_domain,
            &result.message,
        );

        // ⟨(r,s), ze_ood^{→,i}⟩ + ⟨(r,s), (G_C^s[x,·], 0)⟩ (mask part, shift pre-baked)
        let mask_sum = result
            .mask_message
            .as_deref()
            .zip(claim.mask_info.as_ref())
            .map_or(F::ZERO, |(mm, info)| {
                info.evaluate(&Identity::<F>::new(), mm)
            });

        // Decision phase formula (p.55): μ' = ν_1·μ + message_contribution + mask_contribution
        assert_eq!(
            msg_sum + mask_sum,
            claim.mu_prime - claim.original_sl_coeff * result.target_mu
        );
        // Coefficient counts match weight counts
        assert_eq!(claim.ood_rlc_coeffs.len(), claim.source_weights.ood.len());
        assert_eq!(
            claim.in_domain_rlc_coeffs.len(),
            claim.source_weights.in_domain.len()
        );
        // ZK structure: mask_info present iff ZK mode
        assert_eq!(claim.mask_info.is_some(), masked);
        if let Some(ref info) = claim.mask_info {
            assert_eq!(info.ood_rlc_coeffs.len(), info.weights.ood.len());
            assert_eq!(
                info.in_domain_rlc_coeffs.len(),
                info.weights.in_domain.len()
            );
        }
    }

    /// Powers of 2 in [8, 256] that are valid NTT domain sizes for field F.
    fn valid_sizes<F: 'static>() -> Vec<usize> {
        (3..=8)
            .map(|k| 1_usize << k)
            .filter(|&n| ntt::next_order::<F>(n) == Some(n))
            .collect()
    }

    fn proptest_completeness<F: Field + FieldWithSize + Codec<[u8]> + 'static>()
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        let rates = select(vec![1_usize, 2, 3]);
        proptest!(|(seed: u64, sz in select(valid_sizes::<F>()), s in rates.clone(), t in rates, m: bool)| {
            check_completeness(&run_code_switch::<F>(seed, sz, s, t, m), m);
        });
    }

    #[test]
    fn completeness_field64() {
        proptest_completeness::<fields::Field64>();
    }
    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn completeness_field64_2() {
        proptest_completeness::<fields::Field64_2>();
    }
    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn completeness_field64_3() {
        proptest_completeness::<fields::Field64_3>();
    }
    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn completeness_field128() {
        proptest_completeness::<fields::Field128>();
    }
    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn completeness_field192() {
        proptest_completeness::<fields::Field192>();
    }
    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn completeness_field256() {
        proptest_completeness::<fields::Field256>();
    }
}
