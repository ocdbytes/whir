use ark_ff::FftField;
use serde::{Deserialize, Serialize};

use crate::algebra::embedding::Embedding;

mod commiter;
mod prover;
mod utils;
mod verifier;

pub use self::{commiter::Witness, verifier::Commitments};
use crate::{
    algebra::embedding::Identity,
    parameters::ProtocolParameters,
    protocols::{irs_commit, whir},
};

#[allow(clippy::trait_duplication_in_bounds)]
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(bound = "")]
pub struct Config<F: FftField> {
    pub blinded_polynomial: whir::Config<Identity<F>>,
    pub blinding_polynomial: whir::Config<Identity<F>>,
}

impl<F: FftField> Config<F> {
    pub fn new(params: &ProtocolParameters, num_variables_main: usize) -> Self {
        let blinded_config: whir::Config<Identity<F>> =
            whir::Config::new(1 << num_variables_main, params);
        let witness_sec = params.security_level.saturating_sub(params.pow_bits) as f64;
        let blinding_sec = params.security_level as f64;

        let irs_commit_config = blinded_config.initial_committer;

        let witness_leak =
            InstanceLeak::new(params, witness_sec, &irs_commit_config, num_variables_main);
        let blinding_leak =
            InstanceLeak::new(params, blinding_sec, &irs_commit_config, num_variables_main);

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
            blinded_polynomial: whir::Config::new(1 << num_variables_main, params),
            blinding_polynomial: whir::Config::new(1 << (ell + 1), &blinding_params),
        }
    }
}

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
            d: 3,
            q_delta: q,
            stir_delta: stir,
            ood_delta: irs_config.out_domain_samples,
            // TODO: use q(delta_last_round) instead of q(delta).
            t_delta: q,
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
    fn to_prove_forms(forms: &[Box<dyn LinearForm<F>>], size: usize) -> Vec<Box<dyn LinearForm<F>>> {
        forms
            .iter()
            .map(|f| {
                let mut cv = vec![F::ZERO; size];
                f.accumulate(&mut cv, F::ONE);
                Box::new(Covector { vector: cv }) as Box<dyn LinearForm<F>>
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
            &witness,
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

    /// Test that the RS-fold decomposition holds:
    /// fold(r̄, f_zk)(z) = ρ · fold(r̄, f̂)(z) + m̃_RS(z) + Σ βⁱ · g̃_i_RS(z)
    ///
    /// This validates the `compute_rs_fold_blinding_coeffs` helper by checking
    /// that the RS-fold of f_zk decomposes correctly into f̂ and blinding parts.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_rs_fold_decomposition() {
        use crate::{
            algebra::univariate_evaluate,
            protocols::{
                geometric_challenge::geometric_challenge,
                whir_zk_2::utils::{compute_rs_fold_blinding_coeffs, phi_i_bits},
            },
            transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierMessage},
        };

        let config = make_test_config();
        let mu = TEST_NUM_VARIABLES;
        let ell = config.blinding_polynomial.initial_num_variables() - 1;
        let rem = mu % ell;
        let num_g_polys = config.blinding_polynomial.initial_committer.num_vectors;
        let size = 1usize << mu;

        // Set up a transcript to get deterministic randomness
        let ds = DomainSeparator::protocol(&config)
            .session(&format!("rs-fold-test {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        // Commit (generates masking_polys, g_polys, f_hat_polys)
        let vector = vec![F::ONE; size];
        let witness = config.commit(&mut prover_state, &[vector.as_slice()]);

        // Step 1-2: get β, build g_poly
        let beta: F = prover_state.verifier_message();
        let mut beta_powers = Vec::with_capacity(num_g_polys);
        let mut bp = F::ONE;
        for _ in 0..num_g_polys {
            beta_powers.push(bp);
            bp *= beta;
        }

        let mut g_poly = vec![F::ZERO; size];
        for (b, g_val) in g_poly.iter_mut().enumerate().take(size) {
            for (i, &bp) in beta_powers.iter().enumerate().take(num_g_polys) {
                let idx = phi_i_bits(b, i, mu, ell, rem);
                *g_val += bp * witness.g_polys[i][idx];
            }
        }

        // G claims (read by verifier)
        let linear_forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(MultilinearExtension {
            point: MultilinearPoint::rand(&mut ark_std::test_rng(), mu).0,
        })];
        let g_claims: Vec<F> = linear_forms
            .iter()
            .map(|w| {
                let mut covector = vec![F::ZERO; size];
                w.accumulate(&mut covector, F::ONE);
                covector
                    .iter()
                    .zip(g_poly.iter())
                    .map(|(&a, &b)| a * b)
                    .sum()
            })
            .collect();
        for g_claim in &g_claims {
            prover_state.prover_message(g_claim);
        }

        // Step 3: get ρ, build f_zk
        let rho: F = prover_state.verifier_message();

        let f_zk: Vec<F> = vector
            .iter()
            .zip(g_poly.iter())
            .map(|(&f, &g)| rho * f + g)
            .collect();

        // Run initial sumcheck to get r̄
        let constraint_rlc_coeffs: Vec<F> =
            geometric_challenge(&mut prover_state, linear_forms.len());
        let mut f_zk_copy = f_zk.clone();
        let mut covector = vec![F::ZERO; size];
        for (&coeff, lf) in constraint_rlc_coeffs.iter().zip(linear_forms.iter()) {
            lf.accumulate(&mut covector, coeff);
        }
        // Compute combined claims for the_sum
        let evaluations: Vec<F> = linear_forms
            .iter()
            .map(|w| {
                let mut cv = vec![F::ZERO; size];
                w.accumulate(&mut cv, F::ONE);
                cv.iter()
                    .zip(vector.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<F>()
            })
            .collect();
        let combined_claims: Vec<F> = evaluations
            .iter()
            .zip(g_claims.iter())
            .map(|(&e, &g)| rho * e + g)
            .collect();
        let mut the_sum: F = constraint_rlc_coeffs
            .iter()
            .zip(combined_claims.iter())
            .map(|(&c, &v)| c * v)
            .sum();
        let folding_randomness = config.blinded_polynomial.initial_sumcheck.prove(
            &mut prover_state,
            &mut f_zk_copy,
            &mut covector,
            &mut the_sum,
        );
        let r_bar = folding_randomness.0;
        let s = r_bar.len();
        let big_m = 1usize << (mu - s);
        let k = 1usize << s;

        // Compute RS-fold coefficients of f_zk directly
        let eq_weights = MultilinearPoint(r_bar.clone()).eq_weights();
        let mut f_zk_fold_coeffs = vec![F::ZERO; big_m];
        for (j, &eq_w) in eq_weights.iter().enumerate().take(k) {
            for (m, coeff) in f_zk_fold_coeffs.iter_mut().enumerate().take(big_m) {
                *coeff += eq_w * f_zk[j * big_m + m];
            }
        }

        // Compute RS-fold coefficients of f̂ (single polynomial, use f_hat_polys[0])
        let mut f_hat_fold_coeffs = vec![F::ZERO; big_m];
        for (j, &eq_w) in eq_weights.iter().enumerate().take(k) {
            for (m, coeff) in f_hat_fold_coeffs.iter_mut().enumerate().take(big_m) {
                *coeff += eq_w * witness.f_hat_polys[0][j * big_m + m];
            }
        }

        // Compute RS-fold blinding coefficients using the helper
        // Single polynomial: alpha_coeffs = [ONE]
        let (m_coeffs_all, g_i_coeffs) = compute_rs_fold_blinding_coeffs(
            &r_bar,
            &witness.g_polys,
            &witness.masking_polys,
            &[F::ONE],
            rho,
            mu,
            ell,
            rem,
        );

        // Test at several z values: f_zk_fold(z) == ρ·f̂_fold(z) + m̃_RS(z) + Σ βⁱ·g̃ᵢ_RS(z)
        let test_points: Vec<F> = (1u64..=10).map(|i| F::from(i * 7)).collect();
        for z in test_points {
            let lhs = univariate_evaluate(&f_zk_fold_coeffs, z);

            let f_hat_term = rho * univariate_evaluate(&f_hat_fold_coeffs, z);
            let m_term = univariate_evaluate(&m_coeffs_all[0], z);
            let g_terms: F = g_i_coeffs
                .iter()
                .enumerate()
                .map(|(i, coeffs)| beta_powers[i + 1] * univariate_evaluate(coeffs, z))
                .sum();
            let rhs = f_hat_term + m_term + g_terms;

            assert_eq!(
                lhs, rhs,
                "RS-fold decomposition failed at z={z:?}: lhs={lhs:?}, rhs={rhs:?}"
            );
        }
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

        prove_and_verify(&config, vec![vector], vec![Box::new(f0), Box::new(f1)], &[eval0, eval1]);
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

        let cov = Covector {
            vector: (0..TEST_NUM_COEFFS).map(|i| F::from(i as u64)).collect(),
        };
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
