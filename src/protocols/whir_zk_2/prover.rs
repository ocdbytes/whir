use std::borrow::Cow;

use ark_ff::FftField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    utils::{build_beq_tables, build_fold_args, compute_rs_fold_blinding_coeffs, phi_i_bits},
    Config,
};
#[cfg(feature = "parallel")]
use crate::utils::workload_size;
use crate::{
    algebra::{
        dot,
        embedding::Identity,
        geometric_sequence,
        linear_form::{Covector, Evaluate, LinearForm, UnivariateEvaluation},
        multilinear_extend, univariate_evaluate,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, whir, whir_zk_2::commiter::Witness},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
};

impl<F: FftField> Config<F> {
    /// zkWHIR 2.0 prover using Alternative Randomness Sampling (paper pages 16-24).
    ///
    /// Supports batching of multiple witness polynomials via RLC.
    /// `vectors[i]` is the i-th witness polynomial evaluation table.
    /// `evaluations` is row-major: `evaluations[j * n + i]` = ⟨wⱼ, fᵢ⟩.
    #[allow(clippy::too_many_lines)]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: Vec<Vec<F>>,
        witness: &Witness<F>,
        linear_forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: &[F],
    ) where
        H: DuplexSpongeInterface<U = u8>,
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // =====================================================================
        // Protocol parameters
        // =====================================================================
        let num_vectors = vectors.len();
        let num_forms = linear_forms.len();
        assert_eq!(evaluations.len(), num_forms * num_vectors);

        let mu = vectors[0].len().trailing_zeros() as usize;
        let ell = self.blinding_polynomial.initial_num_variables() - 1;
        let rem = mu % ell;
        let num_blinding_vecs = self.blinding_polynomial.initial_committer.num_vectors;
        let nu = num_blinding_vecs - num_vectors;
        let num_g_polys = nu + 1;
        let size = 1usize << mu;

        // =====================================================================
        // Steps 1-2: Build blinding polynomial g and send blinding claims G
        // =====================================================================
        let beta: F = prover_state.verifier_message();
        let beta_powers = geometric_sequence(beta, num_g_polys);

        let compute_g = |b: usize| -> F {
            let mut sum = F::ZERO;
            for (i, &bp) in beta_powers.iter().enumerate() {
                let idx = phi_i_bits(b, i, mu, ell, rem);
                sum += bp * witness.g_polys[i][idx];
            }
            sum
        };

        #[cfg(feature = "parallel")]
        let g_poly: Vec<F> = if size > workload_size::<F>() {
            (0..size).into_par_iter().map(compute_g).collect()
        } else {
            (0..size).map(compute_g).collect()
        };

        #[cfg(not(feature = "parallel"))]
        let g_poly: Vec<F> = (0..size).map(compute_g).collect();

        // G_j = ⟨w_j, g⟩ for each linear form (g is shared across all witnesses)
        let mut buf = vec![F::ZERO; size];
        let g_claims: Vec<F> = linear_forms
            .iter()
            .map(|w| {
                buf.fill(F::ZERO);
                w.accumulate(&mut buf, F::ONE);
                dot(&buf, &g_poly)
            })
            .collect();
        drop(buf);

        for g_claim in &g_claims {
            prover_state.prover_message(g_claim);
        }

        // =====================================================================
        // Step 2.5: Sample α for multi-vector RLC
        // =====================================================================
        let alpha_coeffs: Vec<F> = geometric_challenge(prover_state, num_vectors);

        // =====================================================================
        // Step 3: Form combined witness f_zk = ρ·f_combined + g
        // =====================================================================
        let rho: F = prover_state.verifier_message();
        assert_ne!(rho, F::ZERO, "rho should not be zero");

        // f_combined = Σ αⁱ fᵢ, then f_zk = ρ·f_combined + g
        let mut f_zk = {
            let mut iter = vectors.into_iter();
            let mut combined = iter.next().unwrap();
            // alpha_coeffs[0] = ONE, so combined starts as vectors[0]
            for (vec_i, &alpha) in iter.zip(alpha_coeffs[1..].iter()) {
                for (f, v) in combined.iter_mut().zip(vec_i.iter()) {
                    *f += alpha * *v;
                }
            }
            combined
        };

        #[cfg(feature = "parallel")]
        if f_zk.len() > workload_size::<F>() {
            f_zk.par_iter_mut()
                .zip(g_poly.par_iter())
                .for_each(|(f, &g)| *f = rho * *f + g);
        } else {
            for (f, &g) in f_zk.iter_mut().zip(g_poly.iter()) {
                *f = rho * *f + g;
            }
        }

        #[cfg(not(feature = "parallel"))]
        for (f, &g) in f_zk.iter_mut().zip(g_poly.iter()) {
            *f = rho * *f + g;
        }
        drop(g_poly);

        // combined_eval_j = dot(α, evaluations[j*n..(j+1)*n])
        let combined_claims: Vec<F> = (0..num_forms)
            .map(|j| {
                let row = &evaluations[j * num_vectors..(j + 1) * num_vectors];
                let combined_eval: F = alpha_coeffs.iter().zip(row).map(|(&a, &e)| a * e).sum();
                rho * combined_eval + g_claims[j]
            })
            .collect();

        // =====================================================================
        // Step 4: Initial sumcheck on f_zk
        // =====================================================================
        let constraint_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, linear_forms.len());
        let mut covector = vec![F::ZERO; size];
        for (coeff, lf) in constraint_rlc_coeffs.iter().zip(linear_forms.iter()) {
            lf.accumulate(&mut covector, *coeff);
        }
        drop(linear_forms);

        let mut the_sum: F = constraint_rlc_coeffs
            .iter()
            .zip(combined_claims.iter())
            .map(|(&c, &eval)| c * eval)
            .sum();

        let folding_randomness = self.blinded_polynomial.initial_sumcheck.prove(
            prover_state,
            &mut f_zk,
            &mut covector,
            &mut the_sum,
        );

        // =====================================================================
        // Step 5: Virtual OOD and STIR Queries (first WHIR round)
        // =====================================================================
        let round_config = &self.blinded_polynomial.round_configs[0];
        let folded_f_zk_commitment = round_config.irs_committer.commit(prover_state, &[&f_zk]);
        round_config.pow.prove(prover_state);
        let in_domain = self
            .blinded_polynomial
            .initial_committer
            .open(prover_state, &[&witness.blinded_witness]);

        let r_bar = folding_randomness.0;
        let (m_coeffs_all, g_i_coeffs) = compute_rs_fold_blinding_coeffs(
            &r_bar,
            &witness.g_polys,
            &witness.masking_polys,
            &alpha_coeffs,
            rho,
            mu,
            ell,
            rem,
        );

        // Precompute combined f̂ for OOD MLE evaluations
        let f_hat_combined: Vec<F> = (0..size)
            .map(|k| {
                alpha_coeffs
                    .iter()
                    .zip(witness.f_hat_polys.iter())
                    .map(|(&a, p)| a * p[k])
                    .sum()
            })
            .collect();

        let z_points = folded_f_zk_commitment.out_of_domain().points.clone();

        // Λ accumulators for Step 7
        let mut lambda_z_points: Vec<F> = Vec::new();

        // --- OOD responses ---
        for z in z_points {
            let fold_point = build_fold_args(&r_bar, z, mu);
            let ood_f_hat = multilinear_extend(&f_hat_combined, &fold_point);
            prover_state.prover_message(&ood_f_hat);

            // Send n m_evals
            for m_coeffs in &m_coeffs_all {
                let m_eval = univariate_evaluate(m_coeffs, z);
                prover_state.prover_message(&m_eval);
            }
            // Send ν g_evals
            for g_coeffs in &g_i_coeffs {
                let g_eval = univariate_evaluate(g_coeffs, z);
                prover_state.prover_message(&g_eval);
            }
            lambda_z_points.push(z);
        }

        // --- STIR responses ---
        for &z in &in_domain.points {
            // Send n m_evals
            for m_coeffs in &m_coeffs_all {
                let m_eval = univariate_evaluate(m_coeffs, z);
                prover_state.prover_message(&m_eval);
            }
            // Send ν g_evals
            for g_coeffs in &g_i_coeffs {
                let g_eval = univariate_evaluate(g_coeffs, z);
                prover_state.prover_message(&g_eval);
            }
            lambda_z_points.push(z);
        }

        // --- STIR constraint accumulation ---
        let stir_challenges: Vec<UnivariateEvaluation<F>> = folded_f_zk_commitment
            .out_of_domain()
            .evaluators(round_config.initial_size())
            .chain(in_domain.evaluators(round_config.initial_size()))
            .collect();

        let one_weight = [F::ONE];
        let ood_evals = folded_f_zk_commitment.out_of_domain().values(&one_weight);
        let num_ood = folded_f_zk_commitment.out_of_domain().points.len();
        let embedding = Identity::new();

        let in_domain_evals: Vec<F> = stir_challenges[num_ood..]
            .iter()
            .map(|challenge| challenge.evaluate(&embedding, &f_zk))
            .collect();

        let stir_evaluations: Vec<F> = ood_evals.chain(in_domain_evals).collect();

        let stir_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, stir_challenges.len());
        UnivariateEvaluation::accumulate_many(&stir_challenges, &mut covector, &stir_rlc_coeffs);
        the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

        debug_assert_eq!(
            dot(&f_zk, &covector),
            the_sum,
            "invariant broken after STIR accumulation"
        );

        // Round 0 sumcheck
        let mut folding_randomness =
            round_config
                .sumcheck
                .prove(prover_state, &mut f_zk, &mut covector, &mut the_sum);

        // =====================================================================
        // Step 5 (continued): Remaining standard WHIR rounds
        // =====================================================================
        let mut prev_witness = folded_f_zk_commitment;
        let mut gamma_points: Vec<F> = Vec::new();

        for round_index in 1..self.blinded_polynomial.round_configs.len() {
            let rc = &self.blinded_polynomial.round_configs[round_index];
            let prev_rc = &self.blinded_polynomial.round_configs[round_index - 1];
            let (new_witness, in_domain, new_folding) = whir::rounds::prove_round(
                rc,
                prev_rc,
                prover_state,
                &mut f_zk,
                &mut covector,
                &mut the_sum,
                &prev_witness,
                &folding_randomness,
            );
            if round_index == 1 {
                gamma_points.clone_from(&in_domain.points);
            }
            folding_randomness = new_folding;
            prev_witness = new_witness;
        }

        // =====================================================================
        // Final round of the first WHIR instance
        // =====================================================================
        let last_rc = self.blinded_polynomial.round_configs.last().unwrap();
        let (final_in_domain, _final_folding) = whir::rounds::prove_final_round(
            &self.blinded_polynomial.final_sumcheck,
            &self.blinded_polynomial.final_pow,
            last_rc,
            prover_state,
            &mut f_zk,
            &mut covector,
            &mut the_sum,
            &prev_witness,
        );
        if self.blinded_polynomial.round_configs.len() == 1 {
            gamma_points = final_in_domain.points;
        }

        // =====================================================================
        // Step 6: Verifier Consistency Check at Γ points
        // =====================================================================
        let n0 = self.blinded_polynomial.initial_committer.codeword_length;
        let n1 = self.blinded_polynomial.round_configs[0]
            .irs_committer
            .codeword_length;
        let stride = n0 / n1;
        let gen_h = self.blinded_polynomial.round_configs[0]
            .irs_committer
            .generator();

        let mut gamma_f_hat_indices = vec![0usize; gamma_points.len()];
        let mut remaining = gamma_points.len();
        let mut found = vec![false; gamma_points.len()];
        let mut g = F::ONE;
        for i in 0..n1 {
            if remaining == 0 {
                break;
            }
            for (j, &gamma) in gamma_points.iter().enumerate() {
                if !found[j] && g == gamma {
                    gamma_f_hat_indices[j] = i * stride;
                    found[j] = true;
                    remaining -= 1;
                }
            }
            g *= gen_h;
        }
        assert_eq!(remaining, 0, "some gamma points not found in Ω₁ domain");

        let _gamma_f_hat_evals = self.blinded_polynomial.initial_committer.open_at_indices(
            prover_state,
            &[&witness.blinded_witness],
            &gamma_f_hat_indices,
        );

        for gamma in gamma_points {
            // Send n m_evals
            for m_coeffs in &m_coeffs_all {
                let m_eval = univariate_evaluate(m_coeffs, gamma);
                prover_state.prover_message(&m_eval);
            }
            // Send ν g_evals
            for g_coeffs in &g_i_coeffs {
                let g_eval = univariate_evaluate(g_coeffs, gamma);
                prover_state.prover_message(&g_eval);
            }
            lambda_z_points.push(gamma);
        }

        // =====================================================================
        // Step 7: Batched Proof on Blinding Polynomials
        // =====================================================================
        drop(m_coeffs_all);
        drop(g_i_coeffs);
        drop(f_hat_combined);

        let tau: F = prover_state.verifier_message();

        let half_size = 1usize << ell;
        let full_size = 1usize << (ell + 1);

        // beq_tables has num_g_polys = ν+1 entries (one per Φ projection)
        let beq_tables =
            build_beq_tables(&lambda_z_points, &r_bar, tau, mu, ell, rem, num_g_polys);

        // Build weight covectors: n + ν total
        let mut weight_covectors: Vec<Vec<F>> = Vec::with_capacity(num_blinding_vecs);

        // w_0: first M-polynomial weight (includes g₀ and msk₀)
        //   w_0[2k]   = beq_0[k]       (g₀ coefficient)
        //   w_0[2k+1] = (-ρ)·beq_0[k]  (msk₀ coefficient)
        {
            let mut w0 = vec![F::ZERO; full_size];
            let neg_rho = -rho;
            for k in 0..half_size {
                w0[2 * k] = beq_tables[0][k];
                w0[2 * k + 1] = neg_rho * beq_tables[0][k];
            }
            weight_covectors.push(w0);
        }

        // w_i (1 ≤ i < n): additional M-polynomial weights (masking only, no g₀)
        //   w_i[2k]   = 0
        //   w_i[2k+1] = (-ρ·αⁱ)·beq_0[k]
        for i in 1..num_vectors {
            let mut wi = vec![F::ZERO; full_size];
            let scale = -rho * alpha_coeffs[i];
            for k in 0..half_size {
                wi[2 * k + 1] = scale * beq_tables[0][k];
            }
            weight_covectors.push(wi);
        }

        // w_{n+j-1} (1 ≤ j ≤ ν): ĝ_j weights
        //   w[2k]   = beq_j[k]
        //   w[2k+1] = 0
        for beq_table in beq_tables.iter().take(num_g_polys).skip(1) {
            let mut wj = vec![F::ZERO; full_size];
            for k in 0..half_size {
                wj[2 * k] = beq_table[k];
            }
            weight_covectors.push(wj);
        }

        // Reconstruct blinding vectors (same layout as commiter.rs)
        let mut blinding_vecs: Vec<Vec<F>> = Vec::with_capacity(num_blinding_vecs);

        // n m_poly vectors: [g₀[k], mskᵢ[k]] interleaved
        for msk in &witness.masking_polys {
            let m_poly: Vec<F> = witness.g_polys[0]
                .iter()
                .zip(msk.iter())
                .flat_map(|(&g, &m)| [g, m])
                .collect();
            blinding_vecs.push(m_poly);
        }

        // ν emb_g vectors: [g_j[k], 0] interleaved
        for g in &witness.g_polys[1..] {
            let emb: Vec<F> = g.iter().flat_map(|&c| [c, F::ZERO]).collect();
            blinding_vecs.push(emb);
        }

        // Compute eval matrix E[i][j] = ⟨w_i, v_j⟩ (row-major, num_blinding_vecs²)
        let mut eval_matrix: Vec<F> = Vec::with_capacity(num_blinding_vecs * num_blinding_vecs);
        for w in &weight_covectors {
            for v in &blinding_vecs {
                let eval: F = w.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
                eval_matrix.push(eval);
            }
        }

        for eval in &eval_matrix {
            prover_state.prover_message(eval);
        }

        let blinding_forms: Vec<Box<dyn LinearForm<F>>> = weight_covectors
            .into_iter()
            .map(|cv| Box::new(Covector::new(cv)) as Box<dyn LinearForm<F>>)
            .collect();

        let blinding_vector_cows: Vec<Cow<'_, [F]>> =
            blinding_vecs.into_iter().map(Cow::Owned).collect();
        let _ = self.blinding_polynomial.prove(
            prover_state,
            blinding_vector_cows,
            vec![Cow::Borrowed(&witness.blinding_witness)],
            blinding_forms,
            Cow::Owned(eval_matrix),
        );
    }
}
