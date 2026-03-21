use ark_ff::FftField;
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{
    utils::{
        build_beq_tables, build_weight_covectors, compute_eq_weights, gamma_to_f_hat_indices,
        LambdaAccumulator, ProtocolDims,
    },
    Config,
};
use crate::{
    algebra::{
        dot,
        embedding::Identity,
        geometric_sequence,
        linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension, UnivariateEvaluation},
        tensor_product, MultilinearPoint,
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

/// Commitments for blinded and blinding polynomials
pub struct Commitments<F: FftField> {
    pub blinded_commitment: whir::Commitment<F>,
    pub blinding_commitment: whir::Commitment<F>,
}

/// Intermediate result from verifying the blinded polynomial (Steps 2-6).
///
/// Carries the values needed by [`Config::verify_blinding_polynomial`] (Step 7).
#[must_use]
struct BlindedVerifyResult<F> {
    lambda: LambdaAccumulator<F>,
    eq_weights: Vec<F>,
    rho: F,
    alpha_coeffs: Vec<F>,
    dims: ProtocolDims,
}

/// Result of Steps 2-4 (blinding claims, batching, combined claims, initial sumcheck).
#[must_use]
struct VerifyPrepareResult<F> {
    beta_powers: Vec<F>,
    constraint_rlc_coeffs: Vec<F>,
    the_sum: F,
    rho: F,
    alpha_coeffs: Vec<F>,
    folding_randomness: MultilinearPoint<F>,
    eq_weights: Vec<F>,
    batching_weights: Vec<F>,
}

/// Result of Step 5 (OOD/STIR queries, remaining WHIR rounds, linear form RLC check).
#[must_use]
struct VerifyOodStirResult<F> {
    lambda: LambdaAccumulator<F>,
    gamma_points: Vec<F>,
    gamma_h_values: Vec<F>,
}

/// Context for verifying the blinded polynomial (Steps 2-6).
///
/// Bundles the constant context shared across protocol steps so that each step
/// method only needs its step-specific arguments.
struct BlindedVerifyCtx<'a, 'vs, F: FftField, H: DuplexSpongeInterface> {
    config: &'a Config<F>,
    verifier_state: &'a mut VerifierState<'vs, H>,
    commitments: &'a Commitments<F>,
    dims: ProtocolDims,
}

impl<F, H> BlindedVerifyCtx<'_, '_, F, H>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    /// Steps 2-4: Blinding claims, multi-polynomial batching, combined claims, initial sumcheck.
    fn prepare_and_sumcheck(
        &mut self,
        weights: &[&dyn LinearForm<F>],
        evaluations: &[F],
    ) -> VerificationResult<VerifyPrepareResult<F>> {
        let num_vectors = self.dims.num_vectors;
        let num_g_polys = self.dims.num_g_polys();
        let num_forms = weights.len();
        assert_eq!(evaluations.len(), num_forms * num_vectors);

        // =====================================================================
        // Step 2: Blinding Polynomial Claim Generation (paper p. 19)
        //
        // V → P: β ←$ F_q
        // P → V: G_j = ⟨w_j, g⟩ where g(x̄) = Σ βⁱ·ĝᵢ(Φᵢ(x̄))
        // =====================================================================
        let beta: F = self.verifier_state.verifier_message();
        let beta_powers = geometric_sequence(beta, num_g_polys);
        let g_claims: Vec<F> = self.verifier_state.prover_messages_vec(num_forms)?;

        // =====================================================================
        // Step 2.5: Multi-polynomial batching (extension to paper protocol)
        //
        // V → P: α ←$ F_q (for n > 1; when n = 1, α = [1])
        // =====================================================================
        let alpha_coeffs: Vec<F> = geometric_challenge(self.verifier_state, num_vectors);

        // =====================================================================
        // Step 3: Preparation for WHIR Sumcheck Rounds (paper p. 20)
        //
        // V → P: ρ ←$ F_q \ {0}
        // V computes combined_claim_j = ρ · Σᵢ αⁱ·eval[j,i] + G_j
        // =====================================================================
        let rho: F = self.verifier_state.verifier_message();
        verify!(rho != F::ZERO);

        let combined_claims: Vec<F> = (0..num_forms)
            .map(|j| {
                let row = &evaluations[j * num_vectors..(j + 1) * num_vectors];
                let combined_eval: F = alpha_coeffs.iter().zip(row).map(|(&a, &e)| a * e).sum();
                rho * combined_eval + g_claims[j]
            })
            .collect();

        // =====================================================================
        // Step 4: WHIR Initial Round — sumcheck on f_zk (paper p. 20)
        //
        // P ↔ V: s-round sumcheck yielding folding randomness r̄
        // =====================================================================
        let constraint_rlc_coeffs: Vec<F> = geometric_challenge(self.verifier_state, num_forms);
        let mut the_sum: F = constraint_rlc_coeffs
            .iter()
            .zip(combined_claims.iter())
            .map(|(&c, &v)| c * v)
            .sum();

        let folding_randomness = self
            .config
            .blinded_polynomial
            .initial_sumcheck
            .verify(self.verifier_state, &mut the_sum)?;

        let eq_weights = compute_eq_weights(&folding_randomness.0);
        let batching_weights = tensor_product(&alpha_coeffs, &eq_weights);

        Ok(VerifyPrepareResult {
            beta_powers,
            constraint_rlc_coeffs,
            the_sum,
            rho,
            alpha_coeffs,
            folding_randomness,
            eq_weights,
            batching_weights,
        })
    }

    /// Step 5: Virtual OOD and STIR queries, remaining WHIR rounds, linear form RLC check.
    #[allow(clippy::too_many_lines)]
    fn ood_stir_and_rounds(
        &mut self,
        prepare: &mut VerifyPrepareResult<F>,
        weights: &[&dyn LinearForm<F>],
    ) -> VerificationResult<VerifyOodStirResult<F>> {
        let nu = self.dims.nu;
        let num_vectors = self.dims.num_vectors;

        // =====================================================================
        // Step 5: Virtual OOD and STIR Queries (paper pp. 21-22)
        //
        // V receives [[H]], then queries OOD/STIR points.
        // For each point z: reads ood_f̂, n m_evals, ν g_evals.
        // V reconstructs: f_zk(z) = ρ·fold(r̄, [[f̂]])(z) + Σ m_evals + Σ βⁱ·g_evals
        // All claims accumulate into Λ for Step 7.
        // =====================================================================
        let round_config = &self.config.blinded_polynomial.round_configs[0];
        let commitment_h = round_config
            .irs_committer
            .receive_commitment(self.verifier_state)?;
        round_config.pow.verify(self.verifier_state)?;
        let in_domain = self
            .config
            .blinded_polynomial
            .initial_committer
            .verify(self.verifier_state, &[&self.commitments.blinded_commitment])?;

        let mut lambda = LambdaAccumulator::new();

        // --- 5d: OOD responses (paper p. 21) ---
        let one_weight = [F::ONE];
        // Collect is necessary: the iterator borrows commitment_h, which is reused below.
        #[allow(clippy::needless_collect)]
        let ood_h_evals: Vec<F> = commitment_h.out_of_domain().values(&one_weight).collect();

        for &z in &commitment_h.out_of_domain().points {
            // Consumed for Fiat-Shamir transcript binding. The prover sends
            // MLE(f̂_combined, fold_args(r̄, z)), which differs from the univariate
            // fold evaluation fold(r̄, f̂)(z). The decomposition is verified at
            // in-domain and Γ points (Steps 5e' and 6) where the verifier can
            // reconstruct the univariate fold from committed values.
            let _ood_f_hat: F = self.verifier_state.prover_message()?;
            let m_evals: Vec<F> = self.verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = self.verifier_state.prover_messages_vec(nu)?;
            lambda.push(z, m_evals, g_evals);
        }

        // --- 5e: In-domain blinding claims ---
        // Collected separately for STIR reconstruction, then moved into Λ.
        let mut in_domain_m_evals = Vec::with_capacity(in_domain.points.len());
        let mut in_domain_g_evals = Vec::with_capacity(in_domain.points.len());
        for &_z in &in_domain.points {
            let m_evals: Vec<F> = self.verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = self.verifier_state.prover_messages_vec(nu)?;
            in_domain_m_evals.push(m_evals);
            in_domain_g_evals.push(g_evals);
        }

        // --- 5e': Reconstruct virtual STIR response (paper p. 21) ---
        // stir_{ρ·f+g}(z_j) = ρ·fold(r̄, [[f̂]])(z_j) + stir_{M_ρ}(z_j) + Σ βⁱ·stir_{ĝᵢ}(z_j)
        // fold(r̄, [[f̂_combined]])(z) uses tensor_product(α, eq_weights) for batching.
        let f_zk_in_domain_evals: Vec<F> = in_domain
            .values(&prepare.batching_weights)
            .zip(in_domain_m_evals.iter().zip(in_domain_g_evals.iter()))
            .map(|(f_hat_combined_fold, (m_evals, g_evals))| {
                let m_combined: F = m_evals.iter().copied().sum();
                let g_sum: F = g_evals
                    .iter()
                    .enumerate()
                    .map(|(i, &g)| prepare.beta_powers[i + 1] * g)
                    .sum();
                prepare.rho * f_hat_combined_fold + m_combined + g_sum
            })
            .collect();

        // Move in-domain claims into Λ (after reconstruction is done)
        for ((z, m), g) in in_domain
            .points
            .iter()
            .zip(in_domain_m_evals)
            .zip(in_domain_g_evals)
        {
            lambda.push(*z, m, g);
        }

        // --- 5f: STIR constraint accumulation ---
        let stir_challenges: Vec<UnivariateEvaluation<F>> = commitment_h
            .out_of_domain()
            .evaluators(round_config.initial_size())
            .chain(in_domain.evaluators(round_config.initial_size()))
            .collect();

        let stir_evaluations: Vec<F> = ood_h_evals
            .into_iter()
            .chain(f_zk_in_domain_evals)
            .collect();

        let stir_rlc_coeffs: Vec<F> =
            geometric_challenge(self.verifier_state, stir_challenges.len());
        prepare.the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

        let mut round_constraints: Vec<(Vec<F>, Vec<UnivariateEvaluation<F>>)> =
            vec![(stir_rlc_coeffs, stir_challenges)];

        // --- 5g: Round 0 sumcheck ---
        let folding_randomness = round_config
            .sumcheck
            .verify(self.verifier_state, &mut prepare.the_sum)?;
        let mut round_folding_randomness = vec![
            prepare.folding_randomness.clone(),
            folding_randomness.clone(),
        ];

        // =====================================================================
        // Step 5 (continued): Remaining standard WHIR rounds (paper p. 22)
        // =====================================================================
        let remaining = whir::rounds::verify_remaining_rounds(
            &self.config.blinded_polynomial.round_configs,
            &whir::rounds::FinalRoundConfig {
                sumcheck: &self.config.blinded_polynomial.final_sumcheck,
                pow: &self.config.blinded_polynomial.final_pow,
            },
            self.verifier_state,
            &mut prepare.the_sum,
            &commitment_h,
            &folding_randomness,
        )?;

        // Extract gamma points and compute gamma_h_values from first in-domain opening
        let gamma_points = remaining.first_in_domain.points.clone();
        let msg_len = round_config.irs_committer.message_length();
        let interleaving_depth = round_config.irs_committer.interleaving_depth;
        let gamma_h_values: Vec<F> = remaining
            .first_in_domain
            .points
            .iter()
            .zip(remaining.first_in_domain.rows())
            .map(|(&gamma, row)| {
                let gamma_step = gamma.pow([msg_len as u64]);
                let mut gamma_pow = F::ONE;
                let mut val = F::ZERO;
                for &r in row.iter().take(interleaving_depth) {
                    val += gamma_pow * r;
                    gamma_pow *= gamma_step;
                }
                val
            })
            .collect();

        round_constraints.extend(remaining.round_constraints);
        round_folding_randomness.extend(remaining.round_folding_randomness);

        // =====================================================================
        // Verify linear form RLC from the sumcheck chain
        // =====================================================================
        let evaluation_point: Vec<F> = round_folding_randomness
            .into_iter()
            .flat_map(|p| p.0.into_iter())
            .collect();

        let poly_eval = MultilinearExtension::new(remaining.final_sumcheck_randomness.0)
            .evaluate(&Identity::new(), &remaining.final_vector);
        let mut linear_form_rlc = prepare.the_sum / poly_eval;

        for (idx, (rlc_coeffs, stir_weights)) in round_constraints.into_iter().enumerate() {
            let num_variables =
                self.config.blinded_polynomial.round_configs[idx].initial_num_variables();
            let start = evaluation_point.len().saturating_sub(num_variables);
            for (coeff, weight) in zip_strict(rlc_coeffs, stir_weights) {
                linear_form_rlc -= coeff * weight.mle_evaluate(&evaluation_point[start..]);
            }
        }

        let expected_rlc: F = prepare
            .constraint_rlc_coeffs
            .iter()
            .zip(weights.iter())
            .map(|(&c, w)| c * w.mle_evaluate(&evaluation_point))
            .sum();
        verify!(expected_rlc == linear_form_rlc);

        Ok(VerifyOodStirResult {
            lambda,
            gamma_points,
            gamma_h_values,
        })
    }

    /// Step 6: Verifier Consistency Check — Γ point decomposition (paper p. 22).
    fn gamma_check(
        &mut self,
        prepare: VerifyPrepareResult<F>,
        mut ood: VerifyOodStirResult<F>,
    ) -> VerificationResult<BlindedVerifyResult<F>> {
        let num_vectors = self.dims.num_vectors;
        let nu = self.dims.nu;

        // =====================================================================
        // Step 6: Verifier Consistency Check (paper p. 22)
        //
        // V computes Γ = {γ₁..γ_q} ⊆ Ω₁ (FRI query indices).
        // V locally folds: F̂_r̄(γ) = fold_k(r̄, [[f̂]])[γ]   [Eq. (4)]
        // V checks: [[L]](γ) = ρ·F̂_r̄(γ) + m̃ + Σ βⁱ·g̃ᵢ == [[H]](γ)
        // =====================================================================
        let gamma_f_hat_indices = gamma_to_f_hat_indices(&ood.gamma_points, self.config);

        let gamma_f_hat_evals = self
            .config
            .blinded_polynomial
            .initial_committer
            .verify_at_indices(
                self.verifier_state,
                &[&self.commitments.blinded_commitment],
                &gamma_f_hat_indices,
            )?;

        // fold(r̄, [[f̂_combined]])(γ) uses tensor_product(α, eq_weights)
        let f_hat_fold_at_gamma: Vec<F> = gamma_f_hat_evals
            .values(&prepare.batching_weights)
            .collect();

        // Read blinding claims and check decomposition
        for (idx, &gamma) in ood.gamma_points.iter().enumerate() {
            let m_evals: Vec<F> = self.verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = self.verifier_state.prover_messages_vec(nu)?;

            let m_combined: F = m_evals.iter().copied().sum();
            let g_sum: F = g_evals
                .iter()
                .enumerate()
                .map(|(i, &g)| prepare.beta_powers[i + 1] * g)
                .sum();
            let l_gamma = prepare.rho * f_hat_fold_at_gamma[idx] + m_combined + g_sum;
            verify!(l_gamma == ood.gamma_h_values[idx]);

            ood.lambda.push(gamma, m_evals, g_evals);
        }

        Ok(BlindedVerifyResult {
            lambda: ood.lambda,
            eq_weights: prepare.eq_weights,
            rho: prepare.rho,
            alpha_coeffs: prepare.alpha_coeffs,
            dims: self.dims,
        })
    }
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

    /// Steps 2-6: Verify the blinded polynomial instance.
    ///
    /// Reads blinding claims (Step 2), reconstructs combined claims (Steps 2.5-3),
    /// runs the initial sumcheck (Step 4), verifies OOD/STIR/remaining rounds (Step 5),
    /// verifies the linear form RLC, and checks Γ consistency (Step 6).
    fn verify_blinded_polynomial<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        weights: &[&dyn LinearForm<F>],
        evaluations: &[F],
        commitments: &Commitments<F>,
        protocol_dims: ProtocolDims,
    ) -> VerificationResult<BlindedVerifyResult<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let mut ctx = BlindedVerifyCtx {
            config: self,
            verifier_state,
            commitments,
            dims: protocol_dims,
        };
        let mut prepare = ctx.prepare_and_sumcheck(weights, evaluations)?;
        let ood = ctx.ood_stir_and_rounds(&mut prepare, weights)?;
        ctx.gamma_check(prepare, ood)
    }

    /// Step 7: Batched Proof on Blinding Polynomials (paper pp. 23-24).
    ///
    /// V → P: τ ←$ F_q (batching randomness)
    /// Both sides build beq tables and weight covectors wᵢ [Eq. (5)].
    /// V reads E[i][j] = ⟨wᵢ, vⱼ⟩ and checks diagonal against Λ claims:
    ///   E[i][i] = Σ_p τ^{p+1} · B[p][i]
    /// Then run second WHIR instance to verify batch opening.
    #[allow(clippy::needless_pass_by_value)]
    fn verify_blinding_polynomial<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        commitments: &Commitments<F>,
        blinded: BlindedVerifyResult<F>,
    ) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let dims = blinded.dims;
        let tau: F = verifier_state.verifier_message();

        let beq_tables = build_beq_tables(&blinded.lambda.z_points, &blinded.eq_weights, tau, dims);

        let weight_covectors =
            build_weight_covectors(&beq_tables, blinded.rho, &blinded.alpha_coeffs, dims);

        // Read eval_matrix from transcript
        let eval_matrix: Vec<F> =
            verifier_state.prover_messages_vec(dims.num_blinding_vecs * dims.num_blinding_vecs)?;

        // Verify diagonal: E[i][i] = Σ_p τ^{p+1} · claim_i_p
        let num_lambda = blinded.lambda.len();
        for i in 0..dims.num_blinding_vecs {
            let mut expected = F::ZERO;
            let mut tau_power = tau;
            for lambda_idx in 0..num_lambda {
                let claim = if i < dims.num_vectors {
                    blinded.lambda.m_evals[lambda_idx][i]
                } else {
                    blinded.lambda.g_evals[lambda_idx][i - dims.num_vectors]
                };
                expected += tau_power * claim;
                tau_power *= tau;
            }
            verify!(eval_matrix[i * dims.num_blinding_vecs + i] == expected);
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

        Ok(())
    }

    /// zkWHIR 2.0 verifier — Alternative Randomness Sampling (paper pp. 16-24).
    ///
    /// Executes Steps 2-7 of the Σ-IOP protocol (paper pp. 19-24).
    /// `evaluations` is row-major: `evaluations[j * n + i]` = ⟨wⱼ, fᵢ⟩.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
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
        let protocol_dims = ProtocolDims::new(self, commitments.blinded_commitment.num_vectors());
        let blinded = self.verify_blinded_polynomial(
            verifier_state,
            weights,
            evaluations,
            commitments,
            protocol_dims,
        )?;
        self.verify_blinding_polynomial(verifier_state, commitments, blinded)
    }
}
