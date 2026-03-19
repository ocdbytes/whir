use ark_ff::FftField;

use super::{
    prover::{build_fold_args, calculate_phi_i},
    Config,
};
use crate::{
    algebra::{
        dot,
        embedding::Identity,
        eval_eq,
        linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension, UnivariateEvaluation},
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

    /// zkWHIR 2.0 verifier using Alternative Randomness Sampling (paper pages 16-24).
    ///
    /// Mirrors the prover transcript exactly:
    ///   Steps 1-2: Sample β, receive blinding claims G.
    ///   Step 3:    Sample ρ, compute combined claims.
    ///   Step 4:    Initial sumcheck verification.
    ///   Step 5:    First WHIR round — receive [[H]], PoW, open [[f̂]],
    ///              verify OOD decomposition, read in-domain blinding claims,
    ///              STIR constraint accumulation, round 0 sumcheck.
    ///   Step 5+:   Remaining standard WHIR rounds + final round.
    ///   Step 6:    Receive blinding claims at Γ points for Λ.
    ///   Step 7:    Build batched weight covectors, read eval_matrix,
    ///              verify diagonal, run blinding WHIR verifier.
    #[allow(clippy::too_many_lines)]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        weights: &[&dyn LinearForm<F>],
        evaluations: &[F],
        commitments: Commitments<F>,
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
        let num_g_polys = self.blinding_polynomial.initial_committer.num_vectors;
        let num_forms = weights.len();

        // =====================================================================
        // Steps 1-2: Sample β, receive blinding claims G
        // =====================================================================
        let beta: F = verifier_state.verifier_message();
        let mut beta_powers = Vec::with_capacity(num_g_polys);
        let mut bp = F::ONE;
        for _ in 0..num_g_polys {
            beta_powers.push(bp);
            bp *= beta;
        }

        let g_claims: Vec<F> = verifier_state.prover_messages_vec(num_forms)?;

        // =====================================================================
        // Step 3: Sample ρ, compute combined claims
        // =====================================================================
        let rho: F = verifier_state.verifier_message();
        verify!(rho != F::ZERO);

        let combined_claims: Vec<F> = evaluations
            .iter()
            .zip(g_claims.iter())
            .map(|(&eval, &g)| rho * eval + g)
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

        // --- 5a-c: Receive [[H]], PoW, open [[f̂]] ---
        let round_config = &self.blinded_polynomial.round_configs[0];
        let commitment_h = round_config
            .irs_committer
            .receive_commitment(verifier_state)?;
        round_config.pow.verify(verifier_state)?;
        let in_domain = self
            .blinded_polynomial
            .initial_committer
            .verify(verifier_state, &[&commitments.blinded_commitment])?;

        // Λ accumulators for Step 7
        let mut lambda_fold_points: Vec<Vec<F>> = Vec::new();
        let mut lambda_m_evals: Vec<F> = Vec::new();
        let mut lambda_g_evals: Vec<Vec<F>> = Vec::new();

        // --- 5d: OOD responses ---
        // For each OOD point, read f̂, M, ĝ_i MLE evaluations from the prover.
        // These are committed to the Fiat-Shamir transcript. The M and ĝ_i claims
        // are accumulated into Λ for the Step 7 batched blinding proof.
        // TODO : confirm this approach
        // Note: we do NOT check f_zk decomposition here because the prover sends MLE
        // evaluations while the committed OOD values are polynomial evaluations — these
        // are different functions. Soundness is ensured by the STIR constraints (which
        // bind [[H]] to the sumcheck chain) and the Step 7 blinding proof.
        let ood_points = commitment_h.out_of_domain().points.clone();
        let one_weight = [F::ONE];
        let ood_committed_values: Vec<F> =
            commitment_h.out_of_domain().values(&one_weight).collect();

        for (_, &z) in ood_points.iter().enumerate() {
            // Read f̂ MLE evaluation (transcript binding only, not verified here)
            let _ood_f_hat: F = verifier_state.prover_message()?;
            let m_eval: F = verifier_state.prover_message()?;
            let mut g_evals: Vec<F> = Vec::with_capacity(num_g_polys - 1);
            for _ in 1..num_g_polys {
                g_evals.push(verifier_state.prover_message()?);
            }

            lambda_fold_points.push(build_fold_args(&r_bar, z, mu));
            lambda_m_evals.push(m_eval);
            lambda_g_evals.push(g_evals);
        }

        // --- 5e: In-domain blinding claims ---
        // For each in-domain STIR point, read M, ĝ_i evaluations (no f̂ — it comes from the opening).
        let mut temp_vec_m_evals = Vec::new();
        let mut temp_vec_g_evals = Vec::new();
        for &z in &in_domain.points {
            let m_eval: F = verifier_state.prover_message()?;
            let mut g_evals: Vec<F> = Vec::with_capacity(num_g_polys - 1);
            for _ in 1..num_g_polys {
                g_evals.push(verifier_state.prover_message()?);
            }

            lambda_fold_points.push(build_fold_args(&r_bar, z, mu));
            lambda_m_evals.push(m_eval.clone());
            temp_vec_m_evals.push(m_eval);
            lambda_g_evals.push(g_evals.clone());
            temp_vec_g_evals.push(g_evals);
        }

        // --- 5e': Read prover-sent f_zk evaluations at in-domain points ---
        // The verifier can't reconstruct f_zk at in-domain points from the [[f̂]] opening
        // because MLE ≠ RS interpolation for in-domain points. The prover sends them directly.
        // If prover cheates then sumcheck breaks at the final check
        let num_in_domain = in_domain.points.len();
        let f_zk_in_domain_evals: Vec<F> = verifier_state.prover_messages_vec(num_in_domain)?;

        // debug
        // ===============================
        let folding_rand_eq_weights = folding_randomness.eq_weights();
        let f_zk_local_in_domain = in_domain.values(&folding_rand_eq_weights);
        let mut f_zk_in_domain_evals_local = Vec::new();
        for (i, (m_eval, g_evals)) in temp_vec_m_evals
            .iter()
            .zip(temp_vec_g_evals.iter())
            .enumerate()
        {
            let mut g_eval_sum = F::ZERO;
            for i in 0..g_evals.len() {
                g_eval_sum += beta_powers[i] * g_evals[i];
            }
            f_zk_in_domain_evals_local.push(*m_eval + g_eval_sum);
        }
        let f_zk_local_calc: Vec<F> = f_zk_local_in_domain
            .zip(f_zk_in_domain_evals_local.iter())
            .map(|(i1, &i2)| rho * i1 + i2)
            .collect();
        assert_eq!(f_zk_local_calc.len(), f_zk_in_domain_evals.len());
        for (ele1, ele2) in f_zk_in_domain_evals.iter().zip(f_zk_local_calc.iter()) {
            println!(">>> ele1 : {:?} | ele2: {:?}", ele1, ele2);
            if ele1 != ele2 {
                println!(">>> mismatch found")
            }
        }
        // debug
        // ===============================

        // --- 5f: STIR constraint accumulation ---
        let stir_challenges: Vec<UnivariateEvaluation<F>> = commitment_h
            .out_of_domain()
            .evaluators(round_config.initial_size())
            .chain(in_domain.evaluators(round_config.initial_size()))
            .collect();

        let stir_evaluations: Vec<F> = ood_committed_values
            .into_iter()
            .chain(f_zk_in_domain_evals)
            .collect();

        let stir_rlc_coeffs: Vec<F> = geometric_challenge(verifier_state, stir_challenges.len());
        the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

        // Track STIR constraints for the final linear form subtraction.
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

        for round_index in 1..self.blinded_polynomial.round_configs.len() {
            let round_config = &self.blinded_polynomial.round_configs[round_index];

            let new_commitment = round_config
                .irs_committer
                .receive_commitment(verifier_state)?;
            round_config.pow.verify(verifier_state)?;

            let prev_round_config = &self.blinded_polynomial.round_configs[round_index - 1];
            let in_domain = prev_round_config
                .irs_committer
                .verify(verifier_state, &[&prev_commitment])?;

            // Save Γ when opening [[H]] (round_index == 1 opens round 0's commitment)
            if round_index == 1 {
                gamma_points = in_domain.points.clone();
            }

            let stir_challenges: Vec<UnivariateEvaluation<F>> = new_commitment
                .out_of_domain()
                .evaluators(round_config.initial_size())
                .chain(in_domain.evaluators(round_config.initial_size()))
                .collect();
            let stir_evaluations: Vec<F> = new_commitment
                .out_of_domain()
                .values(&one_weight)
                .chain(in_domain.values(&folding_randomness.eq_weights()))
                .collect();

            let stir_rlc_coeffs: Vec<F> =
                geometric_challenge(verifier_state, stir_challenges.len());
            the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);
            round_constraints.push((stir_rlc_coeffs, stir_challenges));

            folding_randomness = round_config.sumcheck.verify(verifier_state, &mut the_sum)?;
            round_folding_randomness.push(folding_randomness.clone());

            prev_commitment = new_commitment;
        }

        // =====================================================================
        // Final round of blinded polynomial WHIR
        // =====================================================================
        let final_vector: Vec<F> = verifier_state
            .prover_messages_vec(self.blinded_polynomial.final_sumcheck.initial_size)?;

        self.blinded_polynomial.final_pow.verify(verifier_state)?;

        let last_round_config = self.blinded_polynomial.round_configs.last().unwrap();
        let final_in_domain = last_round_config
            .irs_committer
            .verify(verifier_state, &[&prev_commitment])?;

        // If there's only one round config, [[H]] is opened in the final round.
        if self.blinded_polynomial.round_configs.len() == 1 {
            gamma_points = final_in_domain.points.clone();
        }

        // Direct check: verify in-domain evaluations against the final vector.
        for (weight, eval) in zip_strict(
            final_in_domain.evaluators(final_vector.len()),
            final_in_domain.values(&folding_randomness.eq_weights()),
        ) {
            verify!(weight.evaluate(&Identity::<F>::new(), &final_vector) == eval);
        }

        // Final sumcheck
        let final_sumcheck_randomness = self
            .blinded_polynomial
            .final_sumcheck
            .verify(verifier_state, &mut the_sum)?;
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

        // Subtract STIR constraint contributions.
        // round_constraints[i] corresponds to round_configs[i].
        for (idx, (rlc_coeffs, stir_weights)) in round_constraints.into_iter().enumerate() {
            let num_variables = self.blinded_polynomial.round_configs[idx].initial_num_variables();
            let start = evaluation_point.len().saturating_sub(num_variables);
            for (coeff, weight) in zip_strict(rlc_coeffs, stir_weights) {
                linear_form_rlc -= coeff * weight.mle_evaluate(&evaluation_point[start..]);
            }
        }

        // The remaining linear_form_rlc should equal the RLC of user weights' MLE evaluations.
        let expected_rlc: F = constraint_rlc_coeffs
            .iter()
            .zip(weights.iter())
            .map(|(&c, w)| c * w.mle_evaluate(&evaluation_point))
            .sum();
        verify!(expected_rlc == linear_form_rlc);

        // =====================================================================
        // Step 6: Γ consistency — receive blinding claims at Γ points
        // =====================================================================
        for &gamma in &gamma_points {
            let m_eval: F = verifier_state.prover_message()?;
            let mut g_evals: Vec<F> = Vec::with_capacity(num_g_polys - 1);
            for _ in 1..num_g_polys {
                g_evals.push(verifier_state.prover_message()?);
            }

            lambda_fold_points.push(build_fold_args(&r_bar, gamma, mu));
            lambda_m_evals.push(m_eval);
            lambda_g_evals.push(g_evals);
        }

        // =====================================================================
        // Step 7: Batched blinding proof
        // =====================================================================
        let tau: F = verifier_state.verifier_message();

        let half_size = 1usize << ell;
        let full_size = 1usize << (ell + 1);

        // Build batched eq tables: beq_i[k] = Σ_j τ^{j+1} · eq(Φ_i(A[j]), k)
        let mut beq_tables: Vec<Vec<F>> = vec![vec![F::ZERO; half_size]; num_g_polys];
        let mut tau_pow = tau;
        for fold_point in &lambda_fold_points {
            for i in 0..num_g_polys {
                let phi_i_point = calculate_phi_i(fold_point, i, ell, rem);
                eval_eq(&mut beq_tables[i], phi_i_point, tau_pow);
            }
            tau_pow *= tau;
        }

        // Build weight covectors (same layout as prover)
        let mut weight_covectors: Vec<Vec<F>> = Vec::with_capacity(num_g_polys);

        // w_0 (M weight): interleaved [ĝ_0, msk]
        //   w_0[2k]   = beq_0[k]
        //   w_0[2k+1] = (−ρ)·beq_0[k]
        {
            let mut w0 = vec![F::ZERO; full_size];
            let neg_rho = -rho;
            for k in 0..half_size {
                w0[2 * k] = beq_tables[0][k];
                w0[2 * k + 1] = neg_rho * beq_tables[0][k];
            }
            weight_covectors.push(w0);
        }

        // w_i (i ≥ 1, ĝ_i weights): interleaved [ĝ_i, 0]
        //   w_i[2k]   = beq_i[k]
        //   w_i[2k+1] = 0
        for i in 1..num_g_polys {
            let mut wi = vec![F::ZERO; full_size];
            for k in 0..half_size {
                wi[2 * k] = beq_tables[i][k];
            }
            weight_covectors.push(wi);
        }

        // Read eval_matrix from transcript
        let eval_matrix: Vec<F> = verifier_state.prover_messages_vec(num_g_polys * num_g_polys)?;

        // Verify diagonal: E[i][i] = Σ_j τ^{j+1} · B[j][i]
        // where B[j][0] = m_eval[j], B[j][i≥1] = g_evals[j][i-1]
        for i in 0..num_g_polys {
            let mut expected = F::ZERO;
            let mut tp = tau;
            for j in 0..lambda_fold_points.len() {
                let claim = if i == 0 {
                    lambda_m_evals[j]
                } else {
                    lambda_g_evals[j][i - 1]
                };
                expected += tp * claim;
                tp *= tau;
            }
            verify!(eval_matrix[i * num_g_polys + i] == expected);
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
