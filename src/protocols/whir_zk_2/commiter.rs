use ark_ff::FftField;
use spongefish::{Codec, DuplexSpongeInterface};

use super::Config;
use crate::{
    hash::Hash,
    protocols::irs_commit,
    transcript::{ProverMessage, ProverState},
};

pub struct Witness<F: FftField> {
    pub blinded_witness: irs_commit::Witness<F, F>,
    pub blinding_witness: irs_commit::Witness<F, F>,
    pub f_hat_poly: Vec<F>,
    pub masking_poly: Vec<F>,
    pub g_polys: Vec<Vec<F>>,
}

impl<F: FftField> Config<F> {
    // TODO : extend it for multiple polynomials
    pub fn commit<H, R>(&self, prover_state: &mut ProverState<H, R>, polynomial: &[F]) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Generate a random masking polynomial
        let masking_poly_size = self.blinding_polynomial.initial_num_variables();
        let half_size = 1usize << (masking_poly_size - 1);

        let masking_poly = (0..half_size)
            .map(|_| F::rand(prover_state.rng()))
            .collect::<Vec<_>>();

        // Compute the masked polynomial f̂ = f + msk(Φ_0).
        // The masking polynomial depends on the first ℓ variables (Φ_0 projection).
        // In big-endian index convention: msk index = b >> (μ - ℓ).
        let mu = polynomial.len().trailing_zeros() as usize;
        let ell = masking_poly_size - 1;
        let shift = mu - ell;
        let f_hat_poly = polynomial
            .iter()
            .enumerate()
            .map(|(b, &f)| f + masking_poly[b >> shift])
            .collect::<Vec<_>>();
        // Commit to the masked polynomial
        let blinded_witness = self.blinded_polynomial.commit(prover_state, &[&f_hat_poly]);

        // Sample v + 1 polynomials
        let num_g_polys = self.blinding_polynomial.initial_committer.num_vectors;
        let mut g_polys = Vec::with_capacity(num_g_polys);
        for _ in 0..num_g_polys {
            g_polys.push(
                (0..half_size)
                    .map(|_| F::rand(prover_state.rng()))
                    .collect::<Vec<_>>(),
            );
        }
        // interleave all g_polys[1..v] with 0s in between
        let embedded_g_polys: Vec<Vec<F>> = g_polys[1..]
            .iter()
            .map(|g| g.iter().flat_map(|&c| [c, F::ZERO]).collect())
            .collect();

        // calculate m_poly = g_0_poly + masking_poly
        let m_poly: Vec<F> = g_polys[0]
            .iter()
            .zip(&masking_poly)
            .flat_map(|(&g, &m)| [g, m])
            .collect();

        let mut blinding_vectors: Vec<&[F]> = Vec::with_capacity(num_g_polys);
        blinding_vectors.push(&m_poly);
        for g in &embedded_g_polys {
            blinding_vectors.push(g);
        }
        // commit to the blinding polynomials
        let blinding_witness = self
            .blinding_polynomial
            .commit(prover_state, &blinding_vectors);

        Witness {
            blinded_witness,
            blinding_witness,
            f_hat_poly,
            masking_poly,
            g_polys,
        }
    }
}
