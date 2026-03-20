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
    pub f_hat_polys: Vec<Vec<F>>,
    pub masking_polys: Vec<Vec<F>>,
    pub g_polys: Vec<Vec<F>>,
}

impl<F: FftField> Config<F> {
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&[F]],
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let n = polynomials.len();
        assert!(n >= 1, "must have at least one polynomial");
        let mu = polynomials[0].len().trailing_zeros() as usize;
        let masking_poly_size = self.blinding_polynomial.initial_num_variables();
        let half_size = 1usize << (masking_poly_size - 1);
        let ell = masking_poly_size - 1;
        let shift = mu - ell;

        // Generate n random masking polynomials and n masked polynomials f̂ᵢ = fᵢ + mskᵢ(Φ₀)
        let mut masking_polys = Vec::with_capacity(n);
        let mut f_hat_polys = Vec::with_capacity(n);
        for poly in polynomials {
            let masking_poly: Vec<F> = (0..half_size)
                .map(|_| F::rand(prover_state.rng()))
                .collect();
            let f_hat_poly: Vec<F> = poly
                .iter()
                .enumerate()
                .map(|(b, &f)| f + masking_poly[b >> shift])
                .collect();
            masking_polys.push(masking_poly);
            f_hat_polys.push(f_hat_poly);
        }

        // Commit to all n masked polynomials
        let f_hat_refs: Vec<&[F]> = f_hat_polys.iter().map(|p| p.as_slice()).collect();
        let blinded_witness = self.blinded_polynomial.commit(prover_state, &f_hat_refs);

        // Sample ν + 1 blinding polynomials ĝ₀..ĝ_ν
        let num_blinding_vecs = self.blinding_polynomial.initial_committer.num_vectors;
        let nu = num_blinding_vecs - n;
        let num_g_polys = nu + 1;
        let mut g_polys = Vec::with_capacity(num_g_polys);
        for _ in 0..num_g_polys {
            g_polys.push(
                (0..half_size)
                    .map(|_| F::rand(prover_state.rng()))
                    .collect::<Vec<_>>(),
            );
        }

        // Build n m_poly vectors: m_poly_i = [g₀[k], mskᵢ[k]] interleaved
        let m_polys: Vec<Vec<F>> = masking_polys
            .iter()
            .map(|msk| {
                g_polys[0]
                    .iter()
                    .zip(msk.iter())
                    .flat_map(|(&g, &m)| [g, m])
                    .collect()
            })
            .collect();

        // Build ν embedded g vectors: emb_g_j = [g_j[k], 0] interleaved, j = 1..ν
        let embedded_g_polys: Vec<Vec<F>> = g_polys[1..]
            .iter()
            .map(|g| g.iter().flat_map(|&c| [c, F::ZERO]).collect())
            .collect();

        // Assemble blinding vectors: [m_poly_0, ..., m_poly_{n-1}, emb_g_1, ..., emb_g_ν]
        let mut blinding_vectors: Vec<&[F]> = Vec::with_capacity(num_blinding_vecs);
        for m in &m_polys {
            blinding_vectors.push(m);
        }
        for g in &embedded_g_polys {
            blinding_vectors.push(g);
        }

        let blinding_witness = self
            .blinding_polynomial
            .commit(prover_state, &blinding_vectors);

        Witness {
            blinded_witness,
            blinding_witness,
            f_hat_polys,
            masking_polys,
            g_polys,
        }
    }
}
