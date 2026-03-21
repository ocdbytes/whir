/// zkWHIR 2.0 — Zero-Knowledge WHIR with poly-logarithmic overhead.
///
/// Implements the "Alternative Randomness Sampling" variant (paper pp. 16-24)
/// which samples only ν = ⌊μ/ℓ⌋ + 1 blinding polynomials (instead of μ + 1),
/// reducing proof size to (ν + 1) · q(δ) field elements.
///
/// Two WHIR instances run as sub-protocols:
///   1. `blinded_polynomial`: over the μ-variate masked witness f̂ = f + msk(Φ₀)
///   2. `blinding_polynomial`: over (ℓ+1)-variate committed vectors M and ĝ₁..ĝ_ν
///
/// Protocol phases (see prover.rs / verifier.rs for per-step details):
///   Step 1: Commitment — sample msk, ĝ₀..ĝ_ν; commit [[f̂]], [[M]], [[ĝᵢ]]
///   Step 2: Blinding claims — V samples β; P builds g(x̄) = Σ βⁱ·ĝᵢ(Φᵢ(x̄)), sends G
///   Step 3: Combination — V samples ρ ≠ 0; P forms f_zk = ρ·f + g
///   Step 4: Initial sumcheck on f_zk; P sends [[H]] = fold_k(f_zk, r̄)
///   Step 5: Virtual OOD/STIR queries + remaining WHIR rounds
///   Step 6: Γ consistency check — verify [[f̂]] openings match [[H]]
///   Step 7: Batched blinding proof via second WHIR instance
use ark_ff::FftField;
use serde::{Deserialize, Serialize};

use crate::algebra::embedding::Embedding;

mod committer;
mod prover;
mod utils;
mod verifier;

pub use self::{committer::Witness, verifier::Commitments};
use crate::{
    algebra::embedding::Identity,
    parameters::ProtocolParameters,
    protocols::{irs_commit, whir},
};

#[allow(clippy::trait_duplication_in_bounds)]
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(bound = "")]
pub struct Config<F: FftField> {
    /// First WHIR instance: proves claims about f_zk = ρ·f + g over 2^μ evaluations.
    pub blinded_polynomial: whir::Config<Identity<F>>,
    /// Second WHIR instance: batched proof of blinding polynomial evaluations
    /// over 2^(ℓ+1) evaluations with n + ν committed vectors.
    pub blinding_polynomial: whir::Config<Identity<F>>,
}

impl<F: FftField> Config<F> {
    pub fn new(params: &ProtocolParameters, num_variables_main: usize) -> Self {
        let blinded_config: whir::Config<Identity<F>> =
            whir::Config::new(1 << num_variables_main, params);
        let witness_sec = params.security_level.saturating_sub(params.pow_bits) as f64;
        let blinding_sec = params.security_level as f64;

        // T(δ) for the witness instance: the polynomial size sent in the final WHIR round.
        let witness_t_delta = blinded_config.final_sumcheck.initial_size;

        let mut witness_leak = InstanceLeak::new(
            params,
            witness_sec,
            &blinded_config.initial_committer,
            num_variables_main,
        );
        witness_leak.t_delta = witness_t_delta;
        // For the blinding instance, T(δ) is approximated as q(δ) since its config
        // depends on ℓ, which we're computing here (conservative upper bound).
        let blinding_leak = InstanceLeak::new(
            params,
            blinding_sec,
            &blinded_config.initial_committer,
            num_variables_main,
        );

        let q_ub = query_upper_bound(&witness_leak, &blinding_leak);

        // ell = smallest integer such that 2^ell > q_ub
        let ell = (usize::BITS - q_ub.leading_zeros()) as usize;
        assert!(
            ell + 1 < num_variables_main,
            "blinding variables ell+1={} must be < mu={num_variables_main}",
            ell + 1
        );
        debug_assert!(
            (1usize << ell) > q_ub,
            "2^ell ({}) must exceed q_ub ({q_ub})",
            1usize << ell
        );

        // nu = ⌊mu/ell⌋ — number of blinding polynomials (alternative sampling, page 16)
        let nu = num_variables_main / ell;
        let blinding_params = ProtocolParameters {
            batch_size: params.batch_size + nu,
            ..*params
        };

        Self {
            blinded_polynomial: blinded_config,
            blinding_polynomial: whir::Config::new(1 << (ell + 1), &blinding_params),
        }
    }
}

/// Maximum degree of the sumcheck round polynomial.
const MAX_SUMCHECK_DEGREE: usize = 3;

/// Per-instance leakage parameters for a single WHIR execution.
///
/// See paper Section "Query Complexity Computation" (page 5):
/// `leak(δ, k, μ, d) := k · [q(δ) + stir(δ)] + T(δ) + ood(δ) + (d+1) · μ`
struct InstanceLeak {
    /// Folding factor `k = 2^s` for the first round.
    k: usize,
    /// Number of sumcheck rounds (= number of witness variables).
    mu: usize,
    /// Max degree of sumcheck round polynomial.
    d: usize,
    /// `q(δ)`: query complexity.
    q_delta: usize,
    /// `stir(δ)`: STIR queries during the first folding round.
    stir_delta: usize,
    /// `ood(δ)`: out-of-domain queries.
    ood_delta: usize,
    /// `T(δ)`: number of raw coefficients sent during the last sumcheck round.
    /// Invariant: `T(δ) ≥ q(δ)`.
    t_delta: usize,
}

impl InstanceLeak {
    /// Construct leak parameters for one WHIR instance.
    ///
    /// `security_target` is `λ - pow_bits` for the witness side
    /// or full `λ` for the blinding side.
    fn new<M>(
        params: &ProtocolParameters,
        security_target: f64,
        irs_config: &irs_commit::Config<M>,
        num_variables: usize,
    ) -> Self
    where
        M: Embedding,
        M::Source: FftField,
        M::Target: FftField,
    {
        #[allow(clippy::cast_possible_wrap)]
        let rate = 0.5_f64.powi(params.starting_log_inv_rate as i32);
        let (q, stir) = Self::query_counts(
            params.unique_decoding,
            security_target,
            rate,
            params.initial_folding_factor,
        );

        Self {
            k: 1 << params.initial_folding_factor,
            mu: num_variables,
            d: MAX_SUMCHECK_DEGREE,
            q_delta: q,
            stir_delta: stir,
            ood_delta: irs_config.out_domain_samples,
            t_delta: q, // conservative default; overridden for witness instance in Config::new
        }
    }

    /// `leak(δ, k, μ, d) := k · [q(δ) + stir(δ)] + T(δ) + ood(δ) + (d+1) · μ`
    const fn leak(&self) -> usize {
        self.k * (self.q_delta + self.stir_delta)
            + self.t_delta
            + self.ood_delta
            + (self.d + 1) * self.mu
    }

    /// Compute `q(δ)` and `stir(δ)` for a given security target and rate.
    ///
    /// - `q(δ) = ⌈λ / log₂(1/(1-δ))⌉`
    /// - `stir(δ) ≈ ⌈λ / (s + log₂(1/(1-δ)))⌉`
    #[allow(clippy::cast_sign_loss)]
    fn query_counts(
        unique_decoding: bool,
        security_target: f64,
        rate: f64,
        folding_factor: usize,
    ) -> (usize, usize) {
        let q = irs_commit::num_in_domain_queries(unique_decoding, security_target, rate);
        let s = folding_factor as f64;
        let per_sample = if unique_decoding {
            f64::midpoint(1., rate)
        } else {
            let johnson_slack = rate.sqrt() / 20.;
            rate.sqrt() + johnson_slack
        };
        let stir = (security_target / (s + (-per_sample.log2()))).ceil() as usize;
        (q, stir)
    }
}

/// Compute `q_ub` — the total leakage upper bound across both WHIR instances.
///
/// `q_ub ≤ leak(δ₁, k₁, μ, d) + leak(δ₂, k₂, μ, 3)`
const fn query_upper_bound(witness: &InstanceLeak, blinding: &InstanceLeak) -> usize {
    witness.leak() + blinding.leak()
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, Field};

    use super::Config;
    use crate::{
        algebra::{
            fields::Field64,
            linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension},
            MultilinearPoint,
        },
        hash,
        parameters::ProtocolParameters,
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    type F = Field64;

    const TEST_NUM_VARIABLES: usize = 12;
    const TEST_NUM_COEFFS: usize = 1 << TEST_NUM_VARIABLES;

    fn make_test_config() -> Config<F> {
        let whir_params = ProtocolParameters {
            unique_decoding: false,
            security_level: 16,
            pow_bits: 0,
            initial_folding_factor: 2,
            folding_factor: 2,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        let mut config = Config::new(&whir_params, TEST_NUM_VARIABLES);
        config.blinded_polynomial.disable_pow();
        config.blinding_polynomial.disable_pow();
        config
    }

    /// Materialize linear forms into Covectors for the prover.
    fn to_prove_forms(
        forms: &[Box<dyn LinearForm<F>>],
        size: usize,
    ) -> Vec<Box<dyn LinearForm<F>>> {
        forms
            .iter()
            .map(|f| {
                let mut cv = vec![F::ZERO; size];
                f.accumulate(&mut cv, F::ONE);
                Box::new(Covector::new(cv)) as Box<dyn LinearForm<F>>
            })
            .collect()
    }

    /// Helper: run a full prove → verify cycle for zkWHIR 2.0.
    /// `vectors` is a list of witness polynomial evaluation tables.
    /// `evaluations` is row-major: `evaluations[j * n + i]` = ⟨wⱼ, fᵢ⟩.
    #[allow(clippy::needless_pass_by_value)]
    fn prove_and_verify(
        config: &Config<F>,
        vectors: Vec<Vec<F>>,
        forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: &[F],
    ) {
        let prove_forms = to_prove_forms(forms.as_slice(), vectors[0].len());

        let ds = DomainSeparator::protocol(config)
            .session(&format!("zk2-pv {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let poly_refs: Vec<&[F]> = vectors.iter().map(|v| v.as_slice()).collect();
        let witness = config.commit(&mut prover_state, &poly_refs);
        config.prove(
            &mut prover_state,
            vectors,
            witness,
            prove_forms,
            evaluations,
        );

        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let commitments = config
            .receive_commitments(&mut verifier_state)
            .expect("receive_commitments failed");

        let weight_refs: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();

        config
            .verify(&mut verifier_state, &weight_refs, evaluations, &commitments)
            .expect("verification failed");
    }

    #[test]
    fn test_zk2_prove_verify_single_point() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector = vec![F::ONE; TEST_NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let form = MultilinearExtension { point: point.0 };
        let evaluation = form.evaluate(config.blinded_polynomial.embedding(), &vector);

        prove_and_verify(&config, vec![vector], vec![Box::new(form)], &[evaluation]);
    }

    #[test]
    fn test_zk2_prove_verify_multiple_points() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();

        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };

        let embedding = config.blinded_polynomial.embedding();
        let eval0 = f0.evaluate(embedding, &vector);
        let eval1 = f1.evaluate(embedding, &vector);

        prove_and_verify(
            &config,
            vec![vector],
            vec![Box::new(f0), Box::new(f1)],
            &[eval0, eval1],
        );
    }

    #[test]
    fn test_zk2_prove_verify_with_covector() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector = vec![F::ONE; TEST_NUM_COEFFS];

        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let mle_form = MultilinearExtension { point: point.0 };
        let embedding = config.blinded_polynomial.embedding();
        let mle_eval = mle_form.evaluate(embedding, &vector);

        let cov = Covector::new((0..TEST_NUM_COEFFS).map(|i| F::from(i as u64)).collect());
        let cov_eval = cov.evaluate(embedding, &vector);

        prove_and_verify(
            &config,
            vec![vector],
            vec![Box::new(mle_form), Box::new(cov)],
            &[mle_eval, cov_eval],
        );
    }

    fn make_test_config_batch(batch_size: usize) -> Config<F> {
        let whir_params = ProtocolParameters {
            unique_decoding: false,
            security_level: 16,
            pow_bits: 0,
            initial_folding_factor: 2,
            folding_factor: 2,
            starting_log_inv_rate: 1,
            batch_size,
            hash_id: hash::SHA2,
        };
        let mut config = Config::new(&whir_params, TEST_NUM_VARIABLES);
        config.blinded_polynomial.disable_pow();
        config.blinding_polynomial.disable_pow();
        config
    }

    #[test]
    fn test_zk2_prove_verify_multi_vector() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config_batch(2);

        let v0: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let v1: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 * 3 + 7))
            .collect();

        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };

        let embedding = config.blinded_polynomial.embedding();
        // evaluations[j * n + i] = ⟨wⱼ, fᵢ⟩
        // 1 form, 2 vectors: evaluations = [⟨w₀, f₀⟩, ⟨w₀, f₁⟩]
        let eval_0_0 = f0.evaluate(embedding, &v0);
        let eval_0_1 = f0.evaluate(embedding, &v1);

        prove_and_verify(
            &config,
            vec![v0, v1],
            vec![Box::new(f0)],
            &[eval_0_0, eval_0_1],
        );
    }

    #[test]
    fn test_zk2_prove_verify_multi_vector_multi_form() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config_batch(2);

        let v0: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let v1: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 * 3 + 7))
            .collect();

        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };

        let embedding = config.blinded_polynomial.embedding();
        // Row-major: evaluations[j * n + i] = ⟨wⱼ, fᵢ⟩
        // 2 forms × 2 vectors = 4 evaluations
        let eval_0_0 = f0.evaluate(embedding, &v0);
        let eval_0_1 = f0.evaluate(embedding, &v1);
        let eval_1_0 = f1.evaluate(embedding, &v0);
        let eval_1_1 = f1.evaluate(embedding, &v1);

        prove_and_verify(
            &config,
            vec![v0, v1],
            vec![Box::new(f0), Box::new(f1)],
            &[eval_0_0, eval_0_1, eval_1_0, eval_1_1],
        );
    }
}
