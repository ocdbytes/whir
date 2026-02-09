#![allow(type_alias_bounds)] // We need the bound to reference F::BasePrimeField.

use std::sync::Arc;

use ark_ff::{FftField, Field};
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{embedding::Embedding, poly_utils::coeffs::CoefficientList},
    hash::Hash,
    protocols::irs_commit,
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
    },
    whir::{
        config::WhirConfig,
        zk::{ZkPreprocessingPolynomials, ZkWitness},
    },
};

pub type Witness<F: FftField> = irs_commit::Witness<F::BasePrimeField, F>;
pub type Commitment<F: Field> = irs_commit::Commitment<F>;

impl<F: FftField> WhirConfig<F> {
    /// Commit to one or more polynomials in coefficient form.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = polynomials.first().unwrap().num_coeffs())))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let poly_refs = polynomials
            .iter()
            .map(|poly| poly.coeffs())
            .collect::<Vec<_>>();
        self.initial_committer
            .commit(prover_state, poly_refs.as_slice())
    }

    pub fn receive_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.initial_committer.receive_commitment(verifier_state)
    }

    #[allow(clippy::too_many_lines)]
    pub fn commit_zk<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomial: &CoefficientList<F::BasePrimeField>,
        helper_config: &WhirConfig<F>,
        preprocessing: Arc<ZkPreprocessingPolynomials<F>>,
    ) -> ZkWitness<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let msk_extended = preprocessing.extend_msk();

        // 1. Compute f̂ = f + msk (parallelized for large polynomials)
        let embedding = self.embedding();
        let f_coeffs = polynomial.coeffs();
        let msk_coeffs = msk_extended.coeffs();
        #[cfg(feature = "parallel")]
        let f_hat_coeffs: Vec<F> = {
            use rayon::prelude::*;
            f_coeffs
                .par_iter()
                .zip(msk_coeffs.par_iter())
                .map(|(&f_c, &msk_c)| embedding.map(f_c) + msk_c)
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let f_hat_coeffs: Vec<F> = f_coeffs
            .iter()
            .zip(msk_coeffs.iter())
            .map(|(&f_c, &msk_c)| embedding.map(f_c) + msk_c)
            .collect();
        let f_hat = CoefficientList::new(f_hat_coeffs);

        // 2. Commit [[f̂]] using main WHIR config
        //    Base field conversion is parallelized
        #[cfg(feature = "parallel")]
        let f_hat_base_field_coeffs: Vec<F::BasePrimeField> = {
            use rayon::prelude::*;
            f_hat
                .coeffs()
                .par_iter()
                .map(|c| {
                    c.to_base_prime_field_elements()
                        .next()
                        .expect("coefficient should be in base field")
                })
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let f_hat_base_field_coeffs: Vec<F::BasePrimeField> = f_hat
            .coeffs()
            .iter()
            .map(|&c| {
                c.to_base_prime_field_elements()
                    .next()
                    .expect("coefficient should be in base field")
            })
            .collect();
        let f_hat_base_field_polynomial = CoefficientList::new(f_hat_base_field_coeffs);
        let f_hat_witness = self.commit(prover_state, &[&f_hat_base_field_polynomial]);

        // 3. Prepare all helper polynomials in base field for batch commitment
        //    Order: [M, ĝ₁_embedded, ..., ĝμ_embedded]
        #[cfg(feature = "parallel")]
        let m_base_field_coeffs: Vec<F::BasePrimeField> = {
            use rayon::prelude::*;
            preprocessing
                .m_poly
                .coeffs()
                .par_iter()
                .map(|c| {
                    c.to_base_prime_field_elements()
                        .next()
                        .expect("coefficient should be in base field")
                })
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let m_base_field_coeffs: Vec<F::BasePrimeField> = preprocessing
            .m_poly
            .coeffs()
            .iter()
            .map(|&c| {
                c.to_base_prime_field_elements()
                    .next()
                    .expect("coefficient should be in base field")
            })
            .collect();
        let m_base_field_polynomial = CoefficientList::new(m_base_field_coeffs);

        // Parallelize the embedding + base field conversion of all ĝⱼ polynomials
        #[cfg(feature = "parallel")]
        let g_hats_embedded_base: Vec<CoefficientList<F::BasePrimeField>> = {
            use rayon::prelude::*;
            preprocessing
                .g_hats
                .par_iter()
                .map(|g_hat| {
                    let embedded = Self::embed_to_larger(g_hat, preprocessing.params.ell + 1);
                    let coeffs: Vec<F::BasePrimeField> = embedded
                        .coeffs()
                        .iter()
                        .map(|c| {
                            c.to_base_prime_field_elements()
                                .next()
                                .expect("coefficient should be in base field")
                        })
                        .collect();
                    CoefficientList::new(coeffs)
                })
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let g_hats_embedded_base: Vec<CoefficientList<F::BasePrimeField>> = preprocessing
            .g_hats
            .iter()
            .map(|g_hat| {
                let embedded = Self::embed_to_larger(g_hat, preprocessing.params.ell + 1);
                let coeffs: Vec<F::BasePrimeField> = embedded
                    .coeffs()
                    .iter()
                    .map(|&c| {
                        c.to_base_prime_field_elements()
                            .next()
                            .expect("coefficient should be in base field")
                    })
                    .collect();
                CoefficientList::new(coeffs)
            })
            .collect();

        // 4. Batch-commit all μ+1 helper polynomials in ONE IRS commit
        //    (helper_config has batch_size = μ+1, so one Merkle tree for all)
        let mut all_helper_polys: Vec<&CoefficientList<F::BasePrimeField>> =
            Vec::with_capacity(1 + preprocessing.g_hats.len());
        all_helper_polys.push(&m_base_field_polynomial);
        for g in &g_hats_embedded_base {
            all_helper_polys.push(g);
        }
        let helper_witness = helper_config.commit(prover_state, &all_helper_polys);

        ZkWitness {
            f_hat_witness,
            helper_witness,
            preprocessing,
            m_poly_base: m_base_field_polynomial,
            g_hats_embedded_base,
        }
    }

    /// Embed ℓ-variate polynomial into n-variate (n > ℓ)
    /// by treating extra variables as having zero contribution
    fn embed_to_larger(poly: &CoefficientList<F>, n: usize) -> CoefficientList<F> {
        let ell = poly.num_variables();
        assert!(n >= ell);

        let factor = 1 << (n - ell);
        let new_size = 1 << n;
        let mut coeffs = vec![F::ZERO; new_size];

        for (i, &c) in poly.coeffs().iter().enumerate() {
            // Coefficient at index i in ℓ-variate
            // maps to index i * factor in n-variate
            coeffs[i * factor] = c;
        }

        CoefficientList::new(coeffs)
    }
}
