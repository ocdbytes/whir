use std::sync::Arc;

use ark_ff::FftField;
use ark_std::{
    rand::{CryptoRng, RngCore},
    UniformRand,
};

use crate::{
    algebra::poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    protocols::irs_commit,
    whir::config::WhirConfig,
};

#[derive(Clone)]
pub struct ZkParams {
    /// ℓ: Number of variables for helper polynomials
    /// Chosen such that 2^ℓ > conservative query upper bound
    pub ell: usize,

    /// μ: Number of variables in the witness polynomial
    pub mu: usize,
}

impl ZkParams {
    /// Compute ell and mu from WHIR parameters.
    pub fn from_whir_params<F: FftField>(whir_params: &WhirConfig<F>) -> Self {
        let mu = whir_params.mv_parameters.num_variables;
        let k = 1 << whir_params.folding_factor.at_round(0);
        let q1 = whir_params
            .round_configs
            .first()
            .map_or(whir_params.final_queries, |r| r.num_queries);

        let q_ub = 2 * k * q1 + 4 * mu + 10;
        let ell = (q_ub as f64).log2().ceil() as usize;
        assert!(
            ell < mu,
            "ZK requires ℓ < μ (ℓ={ell}, μ={mu}). \
             Increase num_variables or lower security_level/queries. \
             (q_ub={q_ub}, k={k}, q1={q1})"
        );
        Self { ell, mu }
    }
}

/// Sampling random polynomials before the witness polynomial
#[derive(Clone)]
pub struct ZkPreprocessingPolynomials<F: FftField> {
    pub msk: CoefficientList<F>,

    pub g0_hat: CoefficientList<F>,

    pub m_poly: CoefficientList<F>,

    pub g_hats: Vec<CoefficientList<F>>,

    pub params: ZkParams,
}

impl<F: FftField> ZkPreprocessingPolynomials<F> {
    pub fn sample<R: RngCore + CryptoRng>(rng: &mut R, params: ZkParams) -> Self {
        let poly_size = 1 << params.ell;
        let m_poly_size = 1 << (params.ell + 1);

        // Sample all preprocessing polynomials from the BASE FIELD, then lift to extension.
        // This is required because these polynomials are committed via base-field IRS commitment,
        // and the conversion back to base field (to_base_prime_field_elements().next()) must be
        // lossless.
        let msk_coeffs: Vec<F> = (0..poly_size)
            .map(|_| F::from_base_prime_field(F::BasePrimeField::rand(rng)))
            .collect();
        let msk = CoefficientList::new(msk_coeffs.clone());

        let g0_coeffs: Vec<F> = (0..poly_size)
            .map(|_| F::from_base_prime_field(F::BasePrimeField::rand(rng)))
            .collect();
        let g0 = CoefficientList::new(g0_coeffs.clone());

        let mut m_coeffs = vec![F::ZERO; m_poly_size];
        for (i, (g0_c, &msk_c)) in g0_coeffs.iter().zip(msk_coeffs.iter()).enumerate() {
            m_coeffs[2 * i] = *g0_c;
            m_coeffs[2 * i + 1] = msk_c;
        }
        let m_poly = CoefficientList::new(m_coeffs);

        let g_hats = (0..params.mu)
            .map(|_| {
                let coeffs = (0..poly_size)
                    .map(|_| F::from_base_prime_field(F::BasePrimeField::rand(rng)))
                    .collect();
                CoefficientList::new(coeffs)
            })
            .collect();

        Self {
            msk,
            g0_hat: g0,
            m_poly,
            g_hats,
            params,
        }
    }

    /// Extend msk to μ variables by padding with zeros
    pub fn extend_msk(&self) -> CoefficientList<F> {
        let target_size = 1 << self.params.mu;
        let mut coeffs = self.msk.coeffs().to_vec();
        coeffs.resize(target_size, F::ZERO);
        CoefficientList::new(coeffs)
    }

    /// Evaluate M(pow(γ), -ρ) = ĝ₀(pow(γ)) - ρ·msk(pow(γ))
    /// For virtual oracle identity
    pub fn evaluate_m_at(&self, gamma: F, rho: F) -> F {
        let pow_gamma = MultilinearPoint::expand_from_univariate(gamma, self.params.ell);

        let g0_eval = self.g0_hat.evaluate(&pow_gamma);
        let msk_eval = self.msk.evaluate(&pow_gamma);

        g0_eval - rho * msk_eval
    }

    /// Evaluate all helper polynomials at pow(γ)
    /// Returns: [ĝ₁(pow(γ)), ..., ĝμ(pow(γ))]
    pub fn evaluate_g_hats_at(&self, gamma: F) -> Vec<F> {
        let pow_gamma = MultilinearPoint::expand_from_univariate(gamma, self.params.ell);
        self.g_hats
            .iter()
            .map(|g_hat| g_hat.evaluate(&pow_gamma))
            .collect()
    }

    /// Batch-evaluate all helper polynomials at multiple gamma points using
    /// univariate Horner evaluation. This is much faster than per-point
    /// multilinear evaluation because:
    /// - Each polynomial is converted to dense univariate form ONCE
    /// - All k×q gamma points are evaluated in a single pass per polynomial
    /// - No `expand_from_univariate` overhead per point
    ///
    /// Returns a Vec of `HelperEvaluations` (one per gamma point), in the same
    /// order as the input gammas.
    pub fn batch_evaluate_helpers(&self, gammas: &[F], rho: F) -> Vec<HelperEvaluations<F>> {
        // Batch-evaluate msk and g₀ at all gamma points.
        // evaluate_at_univariate(γ) == evaluate(expand_from_univariate(γ, ℓ))
        // because the univariate ↔ multilinear bijection is built into the coefficient ordering.
        let msk_evals = self.msk.evaluate_at_univariate(gammas);
        let g0_evals = self.g0_hat.evaluate_at_univariate(gammas);

        // m(γ, ρ) = g₀(pow(γ)) - ρ·msk(pow(γ))
        let m_evals: Vec<F> = g0_evals
            .iter()
            .zip(&msk_evals)
            .map(|(&g0, &msk)| g0 - rho * msk)
            .collect();

        // Batch-evaluate each ĝⱼ at all gamma points (parallelized across polynomials)
        #[cfg(feature = "parallel")]
        let g_hat_evals_per_j: Vec<Vec<F>> = {
            use rayon::prelude::*;
            self.g_hats
                .par_iter()
                .map(|g_hat| g_hat.evaluate_at_univariate(gammas))
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let g_hat_evals_per_j: Vec<Vec<F>> = self
            .g_hats
            .iter()
            .map(|g_hat| g_hat.evaluate_at_univariate(gammas))
            .collect();

        // Transpose: from per-polynomial → per-point
        gammas
            .iter()
            .enumerate()
            .map(|(i, &gamma)| {
                let g_hat_evals = g_hat_evals_per_j.iter().map(|evals| evals[i]).collect();
                HelperEvaluations {
                    gamma,
                    m_eval: m_evals[i],
                    g_hat_evals,
                }
            })
            .collect()
    }

    /// Compute the full blinding value g(γ) at a query point
    ///
    /// g(γ) = g₀(γ) + Σᵢ₌₁^μ βⁱ·γ^(2^(i-1))·gᵢ(γ)
    ///
    /// where gᵢ(γ) = ĝᵢ(pow(γ))
    pub fn evaluate_full_blinding_g(&self, gamma: F, beta: F) -> F {
        let pow_gamma = MultilinearPoint::expand_from_univariate(gamma, self.params.ell);

        let mut result = self.g0_hat.evaluate(&pow_gamma);

        let mut beta_pow = beta;
        let mut gamma_pow = gamma;

        for (_i, g_hat_i) in self.g_hats.iter().enumerate() {
            let g_i_eval = g_hat_i.evaluate(&pow_gamma);
            result += beta_pow * gamma_pow * g_i_eval;
            beta_pow *= beta;
            gamma_pow = gamma_pow.square();
        }

        result
    }
}

/// ZK Witness: contains commitment witnesses for all ZK components
#[derive(Clone)]
pub struct ZkWitness<F: FftField> {
    /// Witness for [[f̂]] = [[f + msk]] in main WHIR
    pub f_hat_witness: irs_commit::Witness<F::BasePrimeField, F>,

    /// Single batch witness for all helper polynomials [[M, ĝ₁, ..., ĝμ]]
    /// committed via helper_config with batch_size = μ+1
    pub helper_witness: irs_commit::Witness<F::BasePrimeField, F>,

    /// Reference to preprocessing data (Arc-shared to avoid deep clone)
    pub preprocessing: Arc<ZkPreprocessingPolynomials<F>>,

    /// Base-field representation of M polynomial (for helper WHIR prove)
    pub m_poly_base: CoefficientList<F::BasePrimeField>,

    /// Base-field representations of embedded ĝⱼ polynomials (for helper WHIR prove)
    /// Each ĝⱼ is embedded from ℓ-variate to (ℓ+1)-variate
    pub g_hats_embedded_base: Vec<CoefficientList<F::BasePrimeField>>,
}

/// ZK Commitment: public commitment data (verifier side)
#[derive(Clone)]
pub struct ZkCommitment<F: FftField> {
    /// Commitment to [[f̂]]
    pub f_hat_commitment: irs_commit::Commitment<F>,

    /// Single batch commitment for all helper polynomials [[M, ĝ₁, ..., ĝμ]]
    pub helper_commitment: irs_commit::Commitment<F>,
}

/// Helper evaluations at a single query point γ
#[derive(Clone, Debug)]
pub struct HelperEvaluations<F> {
    /// The query point γ
    pub gamma: F,

    /// m(γ,ρ) = M(pow(γ), -ρ)
    pub m_eval: F,

    /// [ĝ₁(pow(γ)), ..., ĝμ(pow(γ))]
    pub g_hat_evals: Vec<F>,
}

impl<F: FftField> HelperEvaluations<F> {
    /// Compute the helper polynomial value h(γ) (without the ρ·f̂ term).
    ///
    /// h(γ) = m(γ,ρ) + Σᵢ βⁱ·γ^(2^(i-1))·ĝᵢ(pow(γ))
    pub fn compute_h_value(&self, beta: F) -> F {
        let mut value = self.m_eval;

        let mut beta_power = beta;
        let mut gamma_power = self.gamma;

        for (i, &g_hat_eval) in self.g_hat_evals.iter().enumerate() {
            value += beta_power * gamma_power * g_hat_eval;

            beta_power *= beta;
            if i < self.g_hat_evals.len() - 1 {
                gamma_power = gamma_power.square();
            }
        }

        value
    }

    /// Compute the full virtual oracle value L(γ) = ρ·f̂(γ) + h(γ)
    ///
    /// L(γ) = ρ·f̂(γ) + m(γ,ρ) + Σᵢ βⁱ·γ^(2^(i-1))·ĝᵢ(pow(γ))
    ///      = ρ·(f + msk)(γ) + (ĝ₀ - ρ·msk)(pow(γ)) + blinding_terms
    ///      = ρ·f(γ) + g(γ)
    pub fn compute_virtual_value(&self, f_hat_val: F, rho: F, beta: F) -> F {
        rho * f_hat_val + self.compute_h_value(beta)
    }
}
