use ark_ff::FftField;
use serde::{Deserialize, Serialize};

use crate::algebra::embedding::Embedding;

mod commiter;
mod prover;
mod verifier;

pub use self::{commiter::Witness, verifier::Commitments};
use crate::{
    algebra::embedding::Identity,
    parameters::ProtocolParameters,
    protocols::{irs_commit, whir},
};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(bound = "F: FftField")]
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
            batch_size: nu + 1,
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
    fn leak(&self) -> usize {
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
fn query_upper_bound(witness: &InstanceLeak, blinding: &InstanceLeak) -> usize {
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
    fn to_prove_forms(forms: &[Box<dyn LinearForm<F>>]) -> Vec<Box<dyn LinearForm<F>>> {
        forms
            .iter()
            .map(|f| {
                let mut cv = vec![F::ZERO; TEST_NUM_COEFFS];
                f.accumulate(&mut cv, F::ONE);
                Box::new(Covector { vector: cv }) as Box<dyn LinearForm<F>>
            })
            .collect()
    }

    /// Helper: run a full prove → verify cycle for zkWHIR 2.0.
    fn prove_and_verify(vector: Vec<F>, forms: Vec<Box<dyn LinearForm<F>>>, evaluations: Vec<F>) {
        let config = make_test_config();
        let prove_forms = to_prove_forms(&forms);

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("zk2-pv {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let witness = config.commit(&mut prover_state, &vector);
        config.prove(
            &mut prover_state,
            vector,
            witness,
            prove_forms,
            evaluations.clone(),
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
            .verify(&mut verifier_state, &weight_refs, &evaluations, commitments)
            .expect("verification failed");
    }

    #[test]
    fn test_zk2_prove_verify_single_point() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector = vec![F::ONE; TEST_NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let form = MultilinearExtension {
            point: point.0.clone(),
        };
        let evaluation = form.evaluate(config.blinded_polynomial.embedding(), &vector);

        prove_and_verify(vector, vec![Box::new(form)], vec![evaluation]);
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

        prove_and_verify(vector, vec![Box::new(f0), Box::new(f1)], vec![eval0, eval1]);
    }

    #[test]
    fn test_zk2_prove_verify_with_covector() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector = vec![F::ONE; TEST_NUM_COEFFS];

        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let mle_form = MultilinearExtension {
            point: point.0.clone(),
        };
        let embedding = config.blinded_polynomial.embedding();
        let mle_eval = mle_form.evaluate(embedding, &vector);

        let cov = Covector {
            vector: (0..TEST_NUM_COEFFS).map(|i| F::from(i as u64)).collect(),
        };
        let cov_eval = cov.evaluate(embedding, &vector);

        prove_and_verify(
            vector,
            vec![Box::new(mle_form), Box::new(cov)],
            vec![mle_eval, cov_eval],
        );
    }
}
