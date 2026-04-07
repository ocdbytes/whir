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
        mixed_dot, mixed_univariate_evaluate, random_vector, univariate_evaluate,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, irs_commit},
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Typed,
};

/// Prover output from the code-switch.
#[derive(Clone, Debug)]
pub struct Witness<F: Field, G: Field = F> {
    pub target_witness: irs_commit::Witness<F>,
    pub message: Vec<F>,
    pub source_evaluations: irs_commit::Evaluations<G>,
    pub mask_witness: Option<irs_commit::Witness<F>>,
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
    pub commitment: irs_commit::Commitment<F>,
    /// RLC coefficients with shift ρ^ℓ baked in.
    pub rlc_coeffs: Vec<F>,
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
        let num_ood = self.weights.ood.len();
        let ood_sum: F = self.rlc_coeffs[..num_ood]
            .iter()
            .zip(&self.weights.ood)
            .map(|(&c, w)| c * univariate_evaluate(mask_msg, w.point))
            .sum();
        let r_slice = &mask_msg[..self.source_randomness_len];
        let in_domain_sum: F = self.rlc_coeffs[num_ood..]
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
    pub target_commitment: irs_commit::Commitment<F>,
    pub constraint_rlc_coeffs: Vec<F>,
    pub source_weights: ConstraintWeights<F, G>,
    pub mask_info: Option<MaskClaimInfo<F, G>>,
    pub source_evaluations: irs_commit::Evaluations<G>,
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
    pub source: irs_commit::Config<M>,
    pub target: irs_commit::Config<Identity<M::Target>>,
    pub cs_ood_samples: usize,
    pub mask_commit: Option<irs_commit::Config<Identity<M::Target>>>,
}

/// Split geometric challenge into (ν_1, rest). Requires count ≥ 1.
fn batching_challenge<T, F>(transcript: &mut T, count: usize) -> (F, Vec<F>)
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    debug_assert!(count > 0, "batching requires at least one coefficient");
    let coeffs = geometric_challenge(transcript, count);
    let (&first, rest) = coeffs
        .split_first()
        .expect("count > 0 guarantees non-empty coefficients");
    (first, rest.to_vec())
}

impl<M: Embedding> Config<M> {
    /// `zk`: `None` for non-ZK, `Some(budget)` for ZK with next stage's query budgets.
    pub fn new(
        security_target: f64,
        unique_decoding: bool,
        hash_id: crate::engines::EngineId,
        source_config: irs_commit::Config<M>,
        target_log_inv_rate: usize,
        target_interleaving_depth: usize,
        zk: Option<ZkQueryBudget>,
    ) -> Self
    where
        M: Default,
        M::Target: Default,
    {
        assert!(target_log_inv_rate > 0);
        assert!(target_interleaving_depth > 0);

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

        let mask_commit = zk.map(|budget| {
            assert!(budget.target > 0, "ZK requires nonzero target query budget");
            assert!(budget.mask > 0, "ZK requires nonzero mask query budget");
            target_config.mask_length = budget.target;

            let r_len = source_config.mask_length * source_config.num_messages();
            assert!(
                r_len > 0,
                "ZK code-switch requires source_config.mask_length > 0"
            );
            let mut mc = irs_commit::Config::<Identity<M::Target>>::new(
                security_target,
                unique_decoding,
                hash_id,
                1,
                r_len,
                1,
                source_config.rate(),
            );
            mc.mask_length = budget.mask;
            mc
        });

        let (list_size, degree) = mask_commit.as_ref().map_or_else(
            || (target_config.list_size(), source_message_length),
            |m| {
                (
                    target_config.list_size() * m.list_size(),
                    source_message_length + m.message_length(),
                )
            },
        );
        let cs_ood_samples = irs_commit::num_ood_samples(
            unique_decoding,
            security_target,
            M::Target::field_size_bits(),
            list_size,
            degree,
        );

        Self {
            embedding: Typed::<M>::default(),
            source: source_config,
            target: target_config,
            cs_ood_samples,
            mask_commit,
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
                    let r_len = r_embedded.len();
                    let mut mask_msg = Vec::with_capacity(mask_msg_len);
                    mask_msg.extend_from_slice(&r_embedded);
                    let random_padding: Vec<M::Target> =
                        random_vector(prover_state.rng(), mask_msg_len - r_len);
                    mask_msg.extend_from_slice(&random_padding);
                    let witness = mask_config.commit(prover_state, &[&mask_msg]);
                    (Some(mask_msg), Some(witness))
                });

        // Step 2-3: OOD challenge + answers (y_i = f̂(ρ) + ρ^ℓ · r̂(ρ))
        let ood_points: Vec<M::Target> = prover_state.verifier_message_vec(self.cs_ood_samples);
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
        source_commitment: &irs_commit::Commitment<M::Target>,
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
            .map(|m| m.receive_commitment(verifier_state))
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
        let (original_sl_coeff, constraint_rlc_coeffs) =
            batching_challenge(verifier_state, 1 + num_ood + num_in_domain);

        // Step 6: μ'
        let mu_prime = original_sl_coeff * mu
            + dot(&constraint_rlc_coeffs[..num_ood], &ood_answers)
            + mixed_dot(
                self.source.embedding(),
                &constraint_rlc_coeffs[num_ood..],
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
            let r_len = self.source.mask_length * self.source.num_messages();
            let weights = ConstraintWeights {
                ood: ood_points
                    .iter()
                    .map(|&point| UnivariateEvaluation::new(point, mask_config.message_length()))
                    .collect(),
                in_domain: source_evaluations.evaluators(r_len).collect(),
            };
            let embedding = self.source.embedding();
            let mut rlc_coeffs = Vec::with_capacity(num_ood + weights.in_domain.len());
            for (i, w) in weights.ood.iter().enumerate() {
                let shift = w.point.pow([source_msg_len as u64]);
                rlc_coeffs.push(constraint_rlc_coeffs[i] * shift);
            }
            for (j, w) in weights.in_domain.iter().enumerate() {
                let shift = embedding.map(w.point.pow([source_msg_len as u64]));
                rlc_coeffs.push(constraint_rlc_coeffs[num_ood + j] * shift);
            }

            MaskClaimInfo {
                commitment: mask_commitment.expect("mask commitment must exist when masked"),
                rlc_coeffs,
                weights,
                source_randomness_len: r_len,
            }
        });

        Ok(CodeSwitchClaim {
            mu_prime,
            original_sl_coeff,
            target_commitment,
            constraint_rlc_coeffs,
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
        mu: F,
        mask_message: Option<Vec<F>>,
    }

    fn run_code_switch<F: Field + FieldWithSize + Codec<[u8]> + 'static>(
        seed: u64,
        size: usize,
        src_lir: usize,
        tgt_lir: usize,
        masked: bool,
    ) -> TestResult<F>
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        #[allow(clippy::cast_possible_wrap)]
        let rate = 0.5_f64.powi(src_lir as i32);
        let mut src = irs_commit::Config::<Identity<F>>::new(
            32.0,
            false,
            crate::hash::BLAKE3,
            1,
            size,
            1,
            rate,
        );
        if masked {
            src.mask_length = 1;
        }

        let zk = if masked {
            Some(ZkQueryBudget { target: 1, mask: 1 })
        } else {
            None
        };
        let cfg = Config::<Identity<F>>::new(
            32.0,
            false,
            crate::hash::BLAKE3,
            src.clone(),
            tgt_lir,
            1,
            zk,
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let msg: Vec<F> = random_vector(&mut rng, src.message_length());
        let mu: F = rng.gen();
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(&cfg)
            .session(&"cs_test")
            .instance(&instance);

        let mut ps = ProverState::new_std(&ds);
        let sw = src.commit(&mut ps, &[&msg]);
        let cw = cfg.prove(&mut ps, msg.clone(), &sw.masks, &sw);

        let proof = ps.proof();
        let mut vs = VerifierState::new_std(&ds, &proof);
        let sc = src.receive_commitment(&mut vs).unwrap();
        let claim = cfg.verify(&mut vs, mu, &sc).unwrap();
        vs.check_eof().unwrap();
        assert_eq!(cw.message, msg);

        TestResult {
            claim,
            message: msg,
            mu,
            mask_message: cw.mask_message,
        }
    }

    /// Completeness: μ' = ν_1·μ + ⟨f, source_weights⟩ + ⟨mask_msg, mask_weights⟩
    /// Completeness check (p.56): μ' = ν_1·μ + ⟨f, source_weights⟩ + ⟨mask_msg, mask_weights⟩
    fn check_completeness<F: Field>(r: &TestResult<F>, masked: bool) {
        let c = &r.claim;
        let t = c.source_weights.ood.len();

        // ⟨f, ze_ood^{←,i}⟩ + ⟨f, G_C^#[x,·]⟩ (message part of μ' decomposition)
        let msg_sum = weighted_eval(
            &c.constraint_rlc_coeffs[..t],
            &c.source_weights.ood,
            &r.message,
        ) + weighted_eval(
            &c.constraint_rlc_coeffs[t..],
            &c.source_weights.in_domain,
            &r.message,
        );

        // ⟨(r,s), ze_ood^{→,i}⟩ + ⟨(r,s), (G_C^s[x,·], 0)⟩ (mask part, shift pre-baked)
        let mask_sum = r
            .mask_message
            .as_deref()
            .zip(c.mask_info.as_ref())
            .map_or(F::ZERO, |(mm, info)| {
                info.evaluate(&Identity::<F>::new(), mm)
            });

        // Decision phase formula (p.55): μ' = ν_1·μ + message_contribution + mask_contribution
        assert_eq!(msg_sum + mask_sum, c.mu_prime - c.original_sl_coeff * r.mu);
        // Thm 9.6 (p.54): ze has 1 + t_ood + t·ι coefficients
        assert_eq!(
            c.constraint_rlc_coeffs.len(),
            t + c.source_weights.in_domain.len()
        );
        // ZK structure: mask_info present iff ZK mode
        assert_eq!(c.mask_info.is_some(), masked);
        if let Some(ref info) = c.mask_info {
            // Mask RLC count must cover all mask weights (OOD + in-domain)
            assert_eq!(
                info.rlc_coeffs.len(),
                info.weights.ood.len() + info.weights.in_domain.len()
            );
        }
    }

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
