use ark_ff::FftField;

use super::{
    utils::build_beq_tables,
    Config,
};
use crate::{
    algebra::{
        dot,
        embedding::Identity,
        geometric_sequence, tensor_product,
        linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension, UnivariateEvaluation},
        MultilinearPoint,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, whir},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, VerificationResult,
        VerifierMessage, VerifierState,
    },
    utils::zip_strict,
    verify,
};

pub struct Commitments<F: FftField> {
    pub blinded_commitment: whir::Commitment<F>,
    pub blinding_commitment: whir::Commitment<F>,
}

impl<F: FftField> Config<F> {
    /// Receive the two commitments (blinded polynomial and blinding polynomial)
    /// from the transcript. This mirrors the commit phase in the prover.
    pub fn receive_commitments<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
    ) -> VerificationResult<Commitments<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let blinded_commitment = self.blinded_polynomial.receive_commitment(verifier_state)?;
        let blinding_commitment = self
            .blinding_polynomial
            .receive_commitment(verifier_state)?;
        Ok(Commitments {
            blinded_commitment,
            blinding_commitment,
        })
    }

    /// zkWHIR 2.0 verifier with multi-polynomial batching support.
    ///
    /// `evaluations` is row-major: `evaluations[j * n + i]` = ⟨wⱼ, fᵢ⟩.
    #[allow(clippy::too_many_lines)]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        weights: &[&dyn LinearForm<F>],
        evaluations: &[F],
        commitments: &Commitments<F>,
    ) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // =====================================================================
        // Protocol parameters
        // =====================================================================
        let mu = self.blinded_polynomial.initial_num_variables();
        let ell = self.blinding_polynomial.initial_num_variables() - 1;
        let rem = mu % ell;
        let num_vectors = self.blinded_polynomial.initial_committer.num_vectors;
        let num_blinding_vecs = self.blinding_polynomial.initial_committer.num_vectors;
        let nu = num_blinding_vecs - num_vectors;
        let num_g_polys = nu + 1;
        let num_forms = weights.len();
        assert_eq!(evaluations.len(), num_forms * num_vectors);

        // =====================================================================
        // Steps 1-2: Sample β, receive blinding claims G
        // =====================================================================
        let beta: F = verifier_state.verifier_message();
        let beta_powers = geometric_sequence(beta, num_g_polys);

        let g_claims: Vec<F> = verifier_state.prover_messages_vec(num_forms)?;

        // =====================================================================
        // Step 2.5: Sample α for multi-vector RLC
        // =====================================================================
        let alpha_coeffs: Vec<F> = geometric_challenge(verifier_state, num_vectors);

        // =====================================================================
        // Step 3: Sample ρ, compute combined claims
        // =====================================================================
        let rho: F = verifier_state.verifier_message();
        verify!(rho != F::ZERO);

        let combined_claims: Vec<F> = (0..num_forms)
            .map(|j| {
                let row = &evaluations[j * num_vectors..(j + 1) * num_vectors];
                let combined_eval: F =
                    alpha_coeffs.iter().zip(row).map(|(&a, &e)| a * e).sum();
                rho * combined_eval + g_claims[j]
            })
            .collect();

        // =====================================================================
        // Step 4: Initial sumcheck
        // =====================================================================
        let constraint_rlc_coeffs: Vec<F> = geometric_challenge(verifier_state, num_forms);
        let mut the_sum: F = constraint_rlc_coeffs
            .iter()
            .zip(combined_claims.iter())
            .map(|(&c, &v)| c * v)
            .sum();

        let folding_randomness = self
            .blinded_polynomial
            .initial_sumcheck
            .verify(verifier_state, &mut the_sum)?;

        let r_bar = folding_randomness.0.clone();
        let mut round_folding_randomness = vec![folding_randomness.clone()];

        // =====================================================================
        // Step 5: First WHIR round — virtual OOD and STIR queries
        // =====================================================================
        let round_config = &self.blinded_polynomial.round_configs[0];
        let commitment_h = round_config
            .irs_committer
            .receive_commitment(verifier_state)?;
        round_config.pow.verify(verifier_state)?;
        let in_domain = self
            .blinded_polynomial
            .initial_committer
            .verify(verifier_state, &[&commitments.blinded_commitment])?;

        // Λ accumulators for Step 7.
        // lambda_m_evals[point][i] = claim for i-th M-polynomial at that point.
        // lambda_g_evals[point][j] = claim for (j+1)-th g-polynomial at that point.
        let mut lambda_m_evals: Vec<Vec<F>> = Vec::new();
        let mut lambda_g_evals: Vec<Vec<F>> = Vec::new();
        let mut lambda_z_points: Vec<F> = Vec::new();

        // --- 5d: OOD responses ---
        let ood_points = commitment_h.out_of_domain().points.clone();
        let one_weight = [F::ONE];

        for &z in &ood_points {
            let _ood_f_hat: F = verifier_state.prover_message()?;
            // Read n m_evals + ν g_evals
            let m_evals: Vec<F> = verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = verifier_state.prover_messages_vec(nu)?;

            lambda_z_points.push(z);
            lambda_m_evals.push(m_evals);
            lambda_g_evals.push(g_evals);
        }

        // --- 5e: In-domain blinding claims ---
        let mut in_domain_m_evals = Vec::new();
        let mut in_domain_g_evals = Vec::new();
        for &z in &in_domain.points {
            let m_evals: Vec<F> = verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = verifier_state.prover_messages_vec(nu)?;

            lambda_z_points.push(z);
            lambda_m_evals.push(m_evals.clone());
            in_domain_m_evals.push(m_evals);
            lambda_g_evals.push(g_evals.clone());
            in_domain_g_evals.push(g_evals);
        }

        // --- 5e': Reconstruct f_zk evaluations at in-domain points ---
        // fold(r̄, [[f̂_combined]])(z) uses tensor_product(α, eq_weights)
        let folding_rand_eq_weights = folding_randomness.eq_weights();
        let batching_weights = tensor_product(&alpha_coeffs, &folding_rand_eq_weights);
        let f_zk_in_domain_evals: Vec<F> = in_domain
            .values(&batching_weights)
            .zip(in_domain_m_evals.iter().zip(in_domain_g_evals.iter()))
            .map(|(f_hat_combined_fold, (m_evals, g_evals))| {
                // m_combined = Σ m_evals[i] (they already include α scaling from RS-fold)
                let m_combined: F = m_evals.iter().copied().sum();
                let g_sum: F = g_evals
                    .iter()
                    .enumerate()
                    .map(|(i, &g)| beta_powers[i + 1] * g)
                    .sum();
                rho * f_hat_combined_fold + m_combined + g_sum
            })
            .collect();

        // --- 5f: STIR constraint accumulation ---
        let stir_challenges: Vec<UnivariateEvaluation<F>> = commitment_h
            .out_of_domain()
            .evaluators(round_config.initial_size())
            .chain(in_domain.evaluators(round_config.initial_size()))
            .collect();

        let stir_evaluations: Vec<F> = commitment_h
            .out_of_domain()
            .values(&one_weight)
            .chain(f_zk_in_domain_evals)
            .collect();

        let stir_rlc_coeffs: Vec<F> = geometric_challenge(verifier_state, stir_challenges.len());
        the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

        let mut round_constraints: Vec<(Vec<F>, Vec<UnivariateEvaluation<F>>)> =
            vec![(stir_rlc_coeffs, stir_challenges)];

        // --- 5g: Round 0 sumcheck ---
        let mut folding_randomness = round_config.sumcheck.verify(verifier_state, &mut the_sum)?;
        round_folding_randomness.push(folding_randomness.clone());

        // =====================================================================
        // Step 5 (continued): Remaining standard WHIR rounds
        // =====================================================================
        let mut prev_commitment = commitment_h;
        let mut gamma_points: Vec<F> = Vec::new();
        let mut gamma_h_values: Vec<F> = Vec::new();

        for round_index in 1..self.blinded_polynomial.round_configs.len() {
            let rc = &self.blinded_polynomial.round_configs[round_index];
            let prev_rc = &self.blinded_polynomial.round_configs[round_index - 1];
            let result = whir::rounds::verify_round(
                rc,
                prev_rc,
                verifier_state,
                &mut the_sum,
                &prev_commitment,
                &folding_randomness,
            )?;
            if round_index == 1 {
                gamma_points.clone_from(&result.in_domain.points);
                let msg_len = prev_rc.irs_committer.message_length();
                let d = prev_rc.irs_committer.interleaving_depth;
                gamma_h_values = result
                    .in_domain
                    .points
                    .iter()
                    .zip(result.in_domain.rows())
                    .map(|(&gamma, row)| {
                        let gamma_step = gamma.pow([msg_len as u64]);
                        let mut gamma_pow = F::ONE;
                        let mut val = F::ZERO;
                        for &r in row.iter().take(d) {
                            val += gamma_pow * r;
                            gamma_pow *= gamma_step;
                        }
                        val
                    })
                    .collect();
            }
            round_constraints
                .push((result.stir_rlc_coeffs, result.stir_challenges));
            folding_randomness = result.folding_randomness.clone();
            round_folding_randomness.push(result.folding_randomness);
            prev_commitment = result.commitment;
        }

        // =====================================================================
        // Final round of blinded polynomial WHIR
        // =====================================================================
        let last_rc = self.blinded_polynomial.round_configs.last().unwrap();
        let (final_vector, final_in_domain, final_sumcheck_randomness) =
            whir::rounds::verify_final_round(
                &self.blinded_polynomial.final_sumcheck,
                &self.blinded_polynomial.final_pow,
                last_rc,
                verifier_state,
                &mut the_sum,
                &prev_commitment,
                &folding_randomness,
            )?;

        if self.blinded_polynomial.round_configs.len() == 1 {
            gamma_points.clone_from(&final_in_domain.points);
            let msg_len = last_rc.irs_committer.message_length();
            let d = last_rc.irs_committer.interleaving_depth;
            gamma_h_values = final_in_domain
                .points
                .iter()
                .zip(final_in_domain.rows())
                .map(|(&gamma, row)| {
                    let gamma_step = gamma.pow([msg_len as u64]);
                    let mut gamma_pow = F::ONE;
                    let mut val = F::ZERO;
                    for &r in row.iter().take(d) {
                        val += gamma_pow * r;
                        gamma_pow *= gamma_step;
                    }
                    val
                })
                .collect();
        }

        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // =====================================================================
        // Verify linear form RLC from the sumcheck chain
        // =====================================================================
        let evaluation_point: Vec<F> = round_folding_randomness
            .into_iter()
            .flat_map(|p| p.0.into_iter())
            .collect();

        let poly_eval = MultilinearExtension::new(final_sumcheck_randomness.0)
            .evaluate(&Identity::new(), &final_vector);
        let mut linear_form_rlc = the_sum / poly_eval;

        for (idx, (rlc_coeffs, stir_weights)) in round_constraints.into_iter().enumerate() {
            let num_variables = self.blinded_polynomial.round_configs[idx].initial_num_variables();
            let start = evaluation_point.len().saturating_sub(num_variables);
            for (coeff, weight) in zip_strict(rlc_coeffs, stir_weights) {
                linear_form_rlc -= coeff * weight.mle_evaluate(&evaluation_point[start..]);
            }
        }

        let expected_rlc: F = constraint_rlc_coeffs
            .iter()
            .zip(weights.iter())
            .map(|(&c, w)| c * w.mle_evaluate(&evaluation_point))
            .sum();
        verify!(expected_rlc == linear_form_rlc);

        // =====================================================================
        // Step 6: Γ consistency — verify [[f̂]] opening and check decomposition
        // =====================================================================
        let n0 = self.blinded_polynomial.initial_committer.codeword_length;
        let n1 = self.blinded_polynomial.round_configs[0]
            .irs_committer
            .codeword_length;
        let stride = n0 / n1;
        let gen_h = self.blinded_polynomial.round_configs[0]
            .irs_committer
            .generator();
        let gamma_f_hat_indices: Vec<usize> = gamma_points
            .iter()
            .map(|&gamma| {
                let mut g = F::ONE;
                for i in 0..n1 {
                    if g == gamma {
                        return i * stride;
                    }
                    g *= gen_h;
                }
                panic!("gamma not found in Ω₁ domain");
            })
            .collect();

        let gamma_f_hat_evals = self
            .blinded_polynomial
            .initial_committer
            .verify_at_indices(
                verifier_state,
                &[&commitments.blinded_commitment],
                &gamma_f_hat_indices,
            )?;

        // fold(r̄, [[f̂_combined]])(γ) uses tensor_product(α, eq_weights)
        let initial_eq_weights = MultilinearPoint(r_bar.clone()).eq_weights();
        let gamma_batching_weights = tensor_product(&alpha_coeffs, &initial_eq_weights);
        let f_hat_fold_at_gamma: Vec<F> =
            gamma_f_hat_evals.values(&gamma_batching_weights).collect();

        // Read blinding claims and check decomposition
        for (idx, &gamma) in gamma_points.iter().enumerate() {
            let m_evals: Vec<F> = verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = verifier_state.prover_messages_vec(nu)?;

            // m_combined = Σ m_evals[i]
            let m_combined: F = m_evals.iter().copied().sum();
            let g_sum: F = g_evals
                .iter()
                .enumerate()
                .map(|(i, &g)| beta_powers[i + 1] * g)
                .sum();
            let l_gamma = rho * f_hat_fold_at_gamma[idx] + m_combined + g_sum;
            verify!(l_gamma == gamma_h_values[idx]);

            lambda_z_points.push(gamma);
            lambda_m_evals.push(m_evals);
            lambda_g_evals.push(g_evals);
        }

        // =====================================================================
        // Step 7: Batched blinding proof
        // =====================================================================
        let tau: F = verifier_state.verifier_message();

        let half_size = 1usize << ell;
        let full_size = 1usize << (ell + 1);

        let beq_tables =
            build_beq_tables(&lambda_z_points, &r_bar, tau, mu, ell, rem, num_g_polys);

        // Build weight covectors: n + ν total
        let mut weight_covectors: Vec<Vec<F>> = Vec::with_capacity(num_blinding_vecs);

        // w_0: includes g₀ and msk₀
        {
            let mut w0 = vec![F::ZERO; full_size];
            let neg_rho = -rho;
            for k in 0..half_size {
                w0[2 * k] = beq_tables[0][k];
                w0[2 * k + 1] = neg_rho * beq_tables[0][k];
            }
            weight_covectors.push(w0);
        }

        // w_i (1 ≤ i < n): masking only, no g₀
        for i in 1..num_vectors {
            let mut wi = vec![F::ZERO; full_size];
            let scale = -rho * alpha_coeffs[i];
            for k in 0..half_size {
                wi[2 * k + 1] = scale * beq_tables[0][k];
            }
            weight_covectors.push(wi);
        }

        // w_{n+j-1} (1 ≤ j ≤ ν): ĝ_j weights
        for beq_table in beq_tables.iter().take(num_g_polys).skip(1) {
            let mut wj = vec![F::ZERO; full_size];
            for k in 0..half_size {
                wj[2 * k] = beq_table[k];
            }
            weight_covectors.push(wj);
        }

        // Read eval_matrix from transcript
        let eval_matrix: Vec<F> =
            verifier_state.prover_messages_vec(num_blinding_vecs * num_blinding_vecs)?;

        // Verify diagonal: E[i][i] = Σ_p τ^{p+1} · claim_i_p
        let num_lambda = lambda_z_points.len();
        for i in 0..num_blinding_vecs {
            let mut expected = F::ZERO;
            let mut tp = tau;
            for p in 0..num_lambda {
                let claim = if i < num_vectors {
                    lambda_m_evals[p][i]
                } else {
                    lambda_g_evals[p][i - num_vectors]
                };
                expected += tp * claim;
                tp *= tau;
            }
            verify!(eval_matrix[i * num_blinding_vecs + i] == expected);
        }

        // Package weight covectors as LinearForm trait objects
        let blinding_forms: Vec<Box<dyn LinearForm<F>>> = weight_covectors
            .into_iter()
            .map(|cv| Box::new(Covector::new(cv)) as Box<dyn LinearForm<F>>)
            .collect();

        // Run blinding WHIR verifier
        self.blinding_polynomial
            .verify(
                verifier_state,
                &[&commitments.blinding_commitment],
                &eval_matrix,
            )?
            .verify(
                blinding_forms
                    .iter()
                    .map(|l| l.as_ref() as &dyn LinearForm<F>),
            )?;

        VerificationResult::Ok(())
    }
}
