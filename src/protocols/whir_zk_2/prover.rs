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
    protocols::{geometric_challenge::geometric_challenge, whir_zk_2::commiter::Witness},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
};
#[cfg(feature = "tracing")]
use tracing::instrument;

// TODO: find some way to remove the reimplementation of rounds here.
// This does not look clean and reimplements the same logic which is inside whir.
//
// TODO: extend the prove for multiple vectors as an input
impl<F: FftField> Config<F> {
    /// zkWHIR 2.0 prover using Alternative Randomness Sampling (paper pages 16-24).
    ///
    /// Protocol overview:
    ///   Step 1-2: Build blinding polynomial g and send blinding claims G.
    ///   Step 3:   Form combined witness f_zk = ρ·f + g.
    ///   Step 4:   Initial sumcheck on f_zk.
    ///   Step 5:   Virtual OOD/STIR queries — commit [[H]], open [[f̂]],
    ///             send blinding evaluations, accumulate STIR constraints.
    ///   Step 5+:  Remaining standard WHIR rounds and final round on f_zk.
    ///   Step 6:   Consistency check — send blinding evaluations at Γ points.
    ///   Step 7:   Batched proof — run second WHIR on blinding polynomials
    ///             to prove all claimed evaluations from Steps 5-6.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<'a, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vector: Vec<F>,
        witness: &'a Witness<F>,
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

        let mu = vector.len().trailing_zeros() as usize; // number of witness variables
        let ell = self.blinding_polynomial.initial_num_variables() - 1; // blinding vars per ĝ_i
        let rem = mu % ell; // remainder for Φ_i partition
        let num_g_polys = self.blinding_polynomial.initial_committer.num_vectors; // ν + 1
        let size = 1usize << mu; // 2^μ — witness evaluation table size

        // =====================================================================
        // Steps 1-2: Build blinding polynomial g and send blinding claims G
        // =====================================================================
        // V → P: sample β for combining blinding polynomials.
        // P: construct g(x̄) = Σ_{i=0}^{ν} β^i · ĝ_i(Φ_i(x̄)) over the hypercube.
        // P → V: send G_j = Σ_{b̄} w_j(g(b̄), b̄) for each linear form w_j.

        let beta: F = prover_state.verifier_message();

        // Precompute β powers: [1, β, β², ..., β^ν]
        let beta_powers = geometric_sequence(beta, num_g_polys);

        // Evaluate g on the full 2^μ hypercube via table lookups into ĝ_i.
        // g[b] = Σ_i β^i · ĝ_i[Φ_i(b)]  (no polynomial arithmetic — pure index extraction)
        let compute_g = |b: usize| -> F {
            let mut sum = F::ZERO;
            for i in 0..num_g_polys {
                let idx = phi_i_bits(b, i, mu, ell, rem);
                sum += beta_powers[i] * witness.g_polys[i][idx];
            }
            sum
        };

        #[cfg(feature = "tracing")]
        let _span_g_poly = tracing::info_span!("zk2_build_g_poly", size).entered();

        #[cfg(feature = "parallel")]
        let g_poly: Vec<F> = if size > workload_size::<F>() {
            (0..size).into_par_iter().map(compute_g).collect()
        } else {
            (0..size).map(compute_g).collect()
        };

        #[cfg(not(feature = "parallel"))]
        let g_poly: Vec<F> = (0..size).map(compute_g).collect();

        #[cfg(feature = "tracing")]
        drop(_span_g_poly);

        #[cfg(feature = "tracing")]
        let _span_g_claims = tracing::info_span!("zk2_g_claims").entered();

        // Compute blinding claims G_j = ⟨w_j, g⟩ for each linear form w_j.
        let mut buf = vec![F::ZERO; size];
        let g_claims: Vec<F> = linear_forms
            .iter()
            .map(|w| {
                buf.fill(F::ZERO);
                w.accumulate(&mut buf, F::ONE);
                dot(&buf, &g_poly)
            })
            .collect();

        #[cfg(feature = "tracing")]
        drop(_span_g_claims);

        drop(buf); // free 2^μ scratch buffer, no longer needed

        // P → V: send G claims
        for g_claim in &g_claims {
            prover_state.prover_message(g_claim);
        }

        // =====================================================================
        // Step 3: Form combined witness f_zk = ρ·f + g
        // =====================================================================
        // V → P: sample non-zero ρ.
        // P: update witness to f_zk(x̄) = ρ·f(x̄) + g(x̄).
        // P: update evaluation claims to ρ·F_j + G_j for each linear form j.

        let rho: F = prover_state.verifier_message();
        assert_ne!(rho, F::ZERO, "rho should not be zero");

        #[cfg(feature = "tracing")]
        let _span_blend = tracing::info_span!("zk2_blend_f_zk", size).entered();

        let mut f_zk = vector; // take ownership, no new allocation
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
        drop(g_poly); // free 2^μ immediately

        #[cfg(feature = "tracing")]
        drop(_span_blend);

        let combined_claims: Vec<F> = evaluations
            .iter()
            .zip(g_claims.iter())
            .map(|(&eval, &g)| rho * eval + g)
            .collect();

        // =====================================================================
        // Step 4: Initial sumcheck on f_zk
        // =====================================================================
        // RLC the linear forms into a single covector and scalar claim.
        // Run s rounds of sumcheck, producing folding randomness r̄ = (r_0, ..., r_{s-1}).

        #[cfg(feature = "tracing")]
        let _span_covector = tracing::info_span!("zk2_initial_covector_setup", size).entered();

        let constraint_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, linear_forms.len());
        let mut covector = vec![F::ZERO; size];
        for (coeff, lf) in constraint_rlc_coeffs.iter().zip(linear_forms.iter()) {
            lf.accumulate(&mut covector, *coeff);
        }
        drop(linear_forms); // free linear form objects, no longer needed

        let mut the_sum: F = constraint_rlc_coeffs
            .iter()
            .zip(combined_claims.iter())
            .map(|(&c, &eval)| c * eval)
            .sum();

        #[cfg(feature = "tracing")]
        drop(_span_covector);

        let folding_randomness = self.blinded_polynomial.initial_sumcheck.prove(
            prover_state,
            &mut f_zk,
            &mut covector,
            &mut the_sum,
        );

        // =====================================================================
        // Step 5: Virtual OOD and STIR Queries (first WHIR round)
        // =====================================================================
        // P → V: commit [[H]] := fold_k(ρ·f + g, r̄) on Ω_1.
        // P: PoW for the round.
        // V → P: open [[f̂]] at STIR query points (in-domain samples from Ω_0).
        // P → V: for each OOD/STIR point z, send f̂, M, ĝ_i evaluations.
        //
        // Accumulate all evaluation points into Λ for the batched blinding proof (Step 7).

        let round_config = &self.blinded_polynomial.round_configs[0];
        let folded_f_zk_commitment = round_config.irs_committer.commit(prover_state, &[&f_zk]);
        round_config.pow.prove(prover_state);
        let in_domain = self
            .blinded_polynomial
            .initial_committer
            .open(prover_state, &[&witness.blinded_witness]);

        #[cfg(feature = "tracing")]
        let _span_rs_fold = tracing::info_span!("zk2_rs_fold_blinding_coeffs").entered();

        let r_bar = folding_randomness.0.clone();
        let (m_coeffs, g_i_coeffs) = compute_rs_fold_blinding_coeffs(
            &r_bar,
            &witness.g_polys,
            &witness.masking_poly,
            rho,
            mu,
            ell,
            rem,
        );

        #[cfg(feature = "tracing")]
        drop(_span_rs_fold);

        let z_points = folded_f_zk_commitment.out_of_domain().points.clone();

        // Λ accumulator: fold_args points and blinding evaluations for Step 7.
        //   lambda_fold_points[j] = fold_args(r̄, z_j)          (μ-variate point)
        //   lambda_m_evals[j]     = M(Φ_0(A[j]), −ρ)           (masking eval)
        //   lambda_g_evals[j]     = [ĝ_1(Φ_1(A[j])), ..., ĝ_ν(Φ_ν(A[j]))]
        let mut lambda_fold_points: Vec<Vec<F>> = Vec::new();
        let mut lambda_m_evals: Vec<F> = Vec::new();
        let mut lambda_g_evals: Vec<Vec<F>> = Vec::new();
        let mut lambda_z_points: Vec<F> = Vec::new();

        #[cfg(feature = "tracing")]
        let _span_ood_stir =
            tracing::info_span!("zk2_ood_stir_responses", num_ood = z_points.len(), num_stir = in_domain.points.len()).entered();

        // --- OOD responses ---
        // For each OOD point z₀: send f̂(fold_args(r̄, z₀)), M, and ĝ_i evaluations.
        // The verifier reconstructs ood_{ρ·f+g} = ρ·ood_f̂ + ood_M + Σ β^i·ood_{ĝ_i}.
        for z in z_points {
            let fold_point = build_fold_args(&r_bar, z, mu);
            let ood_f_hat = multilinear_extend(&witness.f_hat_poly, &fold_point);
            prover_state.prover_message(&ood_f_hat);

            let m_eval = univariate_evaluate(&m_coeffs, z);
            prover_state.prover_message(&m_eval);
            let g_evals = g_i_coeffs
                .iter()
                .map(|g| univariate_evaluate(&g, z))
                .collect();
            for g_eval in &g_evals {
                prover_state.prover_message(g_eval);
            }
            lambda_fold_points.push(fold_point);
            lambda_z_points.push(z);
            lambda_m_evals.push(m_eval);
            lambda_g_evals.push(g_evals);
        }

        // --- STIR responses ---
        // For each in-domain point z_j: send M and ĝ_i evaluations.
        // The prover does NOT send f̂ here — the verifier folds [[f̂]] locally.
        for &z in &in_domain.points {
            let fold_point = build_fold_args(&r_bar, z, mu);

            let m_eval = univariate_evaluate(&m_coeffs, z);
            prover_state.prover_message(&m_eval);
            let g_evals = g_i_coeffs
                .iter()
                .map(|g| univariate_evaluate(&g, z))
                .collect();
            for g_eval in &g_evals {
                prover_state.prover_message(g_eval);
            }
            lambda_fold_points.push(fold_point);
            lambda_z_points.push(z);
            lambda_m_evals.push(m_eval);
            lambda_g_evals.push(g_evals);
        }

        #[cfg(feature = "tracing")]
        drop(_span_ood_stir);

        // --- STIR constraint accumulation ---
        // Collect UnivariateEvaluation constraints from OOD and in-domain points.
        // Evaluate f_zk at each STIR point via RS interpolation.
        // RLC all STIR constraints into the covector and the_sum.

        #[cfg(feature = "tracing")]
        let _span_stir_accum = tracing::info_span!("zk2_stir_accumulation").entered();

        let stir_challenges: Vec<UnivariateEvaluation<F>> = folded_f_zk_commitment
            .out_of_domain()
            .evaluators(round_config.initial_size())
            .chain(in_domain.evaluators(round_config.initial_size()))
            .collect();

        // STIR evaluations: compute directly from f_zk via RS interpolation.
        // For OOD points, the commitment already gives the correct value.
        // For in-domain points, we must evaluate directly on the folded f_zk vector
        // (not via MLE decomposition, which disagrees with RS interpolation).
        let one_weight = [F::ONE];
        let ood_evals = folded_f_zk_commitment.out_of_domain().values(&one_weight);
        let num_ood = folded_f_zk_commitment.out_of_domain().points.len();
        let embedding = Identity::new();

        let in_domain_evals: Vec<F> = stir_challenges[num_ood..]
            .iter()
            .map(|challenge| challenge.evaluate(&embedding, &f_zk))
            .collect();

        let stir_evaluations: Vec<F> = ood_evals.chain(in_domain_evals.into_iter()).collect();

        let stir_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, stir_challenges.len());
        UnivariateEvaluation::accumulate_many(&stir_challenges, &mut covector, &stir_rlc_coeffs);
        the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

        #[cfg(feature = "tracing")]
        drop(_span_stir_accum);

        // DEBUG: verify invariant after STIR accumulation
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
        // These rounds follow the base WHIR pattern: commit → PoW → open prev →
        // STIR accumulate → sumcheck. No blinding adjustments needed here.
        // We save Γ (FRI query points from opening [[H]]) for Step 6.

        let mut prev_witness = folded_f_zk_commitment;
        let mut gamma_points: Vec<F> = Vec::new();

        #[cfg(feature = "tracing")]
        let _span_whir_rounds = tracing::info_span!(
            "zk2_whir_rounds",
            num_rounds = self.blinded_polynomial.round_configs.len() - 1
        )
        .entered();

        for round_index in 1..self.blinded_polynomial.round_configs.len() {
            #[cfg(feature = "tracing")]
            let _span_round =
                tracing::info_span!("zk2_whir_round", round_index).entered();

            let round_config = &self.blinded_polynomial.round_configs[round_index];

            let new_witness = round_config.irs_committer.commit(prover_state, &[&f_zk]);
            round_config.pow.prove(prover_state);

            let prev_round_config = &self.blinded_polynomial.round_configs[round_index - 1];
            let in_domain = prev_round_config
                .irs_committer
                .open(prover_state, &[&prev_witness]);

            // Save Γ when opening [[H]] (round_index == 1 opens round 0's commitment)
            if round_index == 1 {
                gamma_points = in_domain.points.clone();
            }

            #[cfg(feature = "tracing")]
            let _span_round_stir =
                tracing::info_span!("zk2_round_stir_accumulation", round_index).entered();

            let stir_challenges: Vec<UnivariateEvaluation<F>> = new_witness
                .out_of_domain()
                .evaluators(round_config.initial_size())
                .chain(in_domain.evaluators(round_config.initial_size()))
                .collect();
            let stir_evaluations: Vec<F> = new_witness
                .out_of_domain()
                .values(&[F::ONE])
                .chain(in_domain.values(&folding_randomness.eq_weights()))
                .collect();
            let stir_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, stir_challenges.len());
            UnivariateEvaluation::accumulate_many(
                &stir_challenges,
                &mut covector,
                &stir_rlc_coeffs,
            );
            the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

            #[cfg(feature = "tracing")]
            drop(_span_round_stir);

            folding_randomness =
                round_config
                    .sumcheck
                    .prove(prover_state, &mut f_zk, &mut covector, &mut the_sum);

            prev_witness = new_witness;
        }

        #[cfg(feature = "tracing")]
        drop(_span_whir_rounds);

        // =====================================================================
        // Final round of the first WHIR instance
        // =====================================================================
        // Send the (small) final folded vector directly. Open the last commitment.
        // If there's only one round, [[H]] is opened here — save Γ in that case.

        #[cfg(feature = "tracing")]
        let _span_final = tracing::info_span!("zk2_final_round").entered();

        assert_eq!(
            f_zk.len(),
            self.blinded_polynomial.final_sumcheck.initial_size
        );
        for coeff in &f_zk {
            prover_state.prover_message(coeff);
        }

        self.blinded_polynomial.final_pow.prove(prover_state);

        let last_round_config = self.blinded_polynomial.round_configs.last().unwrap();
        let final_in_domain = last_round_config
            .irs_committer
            .open(prover_state, &[&prev_witness]);

        if self.blinded_polynomial.round_configs.len() == 1 {
            gamma_points = final_in_domain.points.clone();
        }

        let _final_folding_randomness = self.blinded_polynomial.final_sumcheck.prove(
            prover_state,
            &mut f_zk,
            &mut covector,
            &mut the_sum,
        );

        #[cfg(feature = "tracing")]
        drop(_span_final);

        // =====================================================================
        // Step 6: Verifier Consistency Check
        // =====================================================================
        // For each γ ∈ Γ (FRI query points from opening [[H]]):
        //   P computes m̃(r̄, γ, ρ) = M(Φ_0(fold_args(r̄, γ)), −ρ)
        //             g̃_i(r̄, γ)   = ĝ_i(Φ_i(fold_args(r̄, γ)))
        //   P → V: send {m̃, g̃_1, ..., g̃_ν} for each γ.
        //
        // The verifier uses these to reconstruct [[L]](γ) = ρ·F̂_r̄(γ) + m̃ + Σ β^i·g̃_i
        // and checks [[L]](γ) == [[H]](γ) for all γ ∈ Γ.
        //
        // All evaluation points and values are accumulated into Λ for Step 7.

        #[cfg(feature = "tracing")]
        let _span_gamma = tracing::info_span!("zk2_gamma_consistency", num_gamma = gamma_points.len()).entered();

        // Map Γ points (from Ω₁) to [[f̂]] codeword indices (in Ω₀).
        // gen₁ = gen₀^stride where stride = N₀/N₁, so Ω₁ index i → Ω₀ index i·stride.
        let n0 = self.blinded_polynomial.initial_committer.codeword_length;
        let n1 = self.blinded_polynomial.round_configs[0]
            .irs_committer
            .codeword_length;
        let stride = n0 / n1;
        let gen_h = self.blinded_polynomial.round_configs[0]
            .irs_committer
            .generator();

        #[cfg(feature = "tracing")]
        let _span_gamma_search = tracing::info_span!("zk2_gamma_index_search", n1).entered();

        // Single pass through Ω₁ instead of per-gamma linear scan
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

        #[cfg(feature = "tracing")]
        drop(_span_gamma_search);

        // Open [[f̂]] at Γ-mapped indices (goes on transcript before blinding claims)
        let _gamma_f_hat_evals = self.blinded_polynomial.initial_committer.open_at_indices(
            prover_state,
            &[&witness.blinded_witness],
            &gamma_f_hat_indices,
        );

        for gamma in gamma_points {
            let fold_point = build_fold_args(&r_bar, gamma, mu);
            let m_eval = univariate_evaluate(&m_coeffs, gamma);
            prover_state.prover_message(&m_eval);
            let g_evals = g_i_coeffs
                .iter()
                .map(|g| univariate_evaluate(&g, gamma))
                .collect();
            for g_eval in &g_evals {
                prover_state.prover_message(g_eval);
            }
            lambda_fold_points.push(fold_point);
            lambda_z_points.push(gamma);
            lambda_m_evals.push(m_eval);
            lambda_g_evals.push(g_evals);
        }

        #[cfg(feature = "tracing")]
        drop(_span_gamma);

        // =====================================================================
        // Step 7: Batched Proof on Blinding Polynomials
        // =====================================================================
        // Prove that all M and ĝ_i evaluation claims from Steps 5-6 are correct
        // by running a second WHIR instance on the committed blinding polynomials.
        //
        // The committed blinding vectors (from commiter.rs) are (ℓ+1)-variate:
        //   v_0 = m_poly  = [ĝ_0[0], msk[0], ĝ_0[1], msk[1], ...]
        //   v_i = emb_g_i = [ĝ_i[0], 0,      ĝ_i[1], 0,      ...]   (i ≥ 1)
        //
        // For each polynomial i, we construct a weight covector w_i such that
        // ⟨w_i, v_i⟩ = F_i = Σ_j τ^{j+1} · B[j][i], where B[j][i] is the
        // claimed evaluation at Λ point j for polynomial i.
        //
        // The base WHIR prove() handles vector RLC and constraint RLC internally.

        // m_coeffs and g_i_coeffs are no longer needed after Step 6
        drop(m_coeffs);
        drop(g_i_coeffs);

        // V → P: sample τ for Λ-point batching
        let tau: F = prover_state.verifier_message();

        let half_size = 1usize << ell; // 2^ℓ — size of each ĝ_i table
        let full_size = 1usize << (ell + 1); // 2^{ℓ+1} — size of committed vectors

        #[cfg(feature = "tracing")]
        let _span_beq = tracing::info_span!("zk2_beq_tables", num_lambda = lambda_z_points.len(), half_size).entered();

        let beq_tables =
            build_beq_tables(&lambda_z_points, &r_bar, tau, mu, ell, rem, num_g_polys);
        drop(lambda_fold_points); // no longer needed after beq_tables are built

        #[cfg(feature = "tracing")]
        drop(_span_beq);

        #[cfg(feature = "tracing")]
        let _span_weights = tracing::info_span!("zk2_weight_covectors", full_size, num_g_polys).entered();

        // Expand ℓ-variate beq tables to (ℓ+1)-variate weight covectors.
        // The lowest bit (t) accounts for the interleaved committed layout.
        let mut weight_covectors: Vec<Vec<F>> = Vec::with_capacity(num_g_polys);

        // w_0 (M weight): the interleaving [ĝ_0, msk] means index 2k → ĝ_0[k], 2k+1 → msk[k].
        // M(Φ_0, −ρ) = ĝ_0(Φ_0) + (−ρ)·msk(Φ_0), so:
        //   w_0[2k]   = 1·beq_0[k]      (coefficient of ĝ_0[k])
        //   w_0[2k+1] = (−ρ)·beq_0[k]   (coefficient of msk[k])
        {
            let mut w0 = vec![F::ZERO; full_size];
            let neg_rho = -rho;
            for k in 0..half_size {
                w0[2 * k] = beq_tables[0][k];
                w0[2 * k + 1] = neg_rho * beq_tables[0][k];
            }
            weight_covectors.push(w0);
        }

        // w_i (i ≥ 1, ĝ_i weights): the interleaving [ĝ_i, 0] means only even indices matter.
        //   w_i[2k]   = beq_i[k]
        //   w_i[2k+1] = 0  (multiplied by zero in emb_g_i anyway)
        for i in 1..num_g_polys {
            let mut wi = vec![F::ZERO; full_size];
            for k in 0..half_size {
                wi[2 * k] = beq_tables[i][k];
            }
            weight_covectors.push(wi);
        }

        #[cfg(feature = "tracing")]
        drop(_span_weights);

        #[cfg(feature = "tracing")]
        let _span_blinding_vecs = tracing::info_span!("zk2_blinding_vecs", full_size, num_g_polys).entered();

        // Reconstruct blinding vectors (same interleaved layout as commiter.rs).
        let mut blinding_vecs: Vec<Vec<F>> = Vec::with_capacity(num_g_polys);

        let m_poly_vec: Vec<F> = witness.g_polys[0]
            .iter()
            .zip(witness.masking_poly.iter())
            .flat_map(|(&g, &m)| [g, m])
            .collect();
        blinding_vecs.push(m_poly_vec);

        for g in &witness.g_polys[1..] {
            let emb: Vec<F> = g.iter().flat_map(|&c| [c, F::ZERO]).collect();
            blinding_vecs.push(emb);
        }

        #[cfg(feature = "tracing")]
        drop(_span_blinding_vecs);

        #[cfg(feature = "tracing")]
        let _span_eval_matrix = tracing::info_span!("zk2_eval_matrix", num_g_polys, full_size).entered();

        // Compute full evaluation matrix E[i][j] = ⟨w_i, v_j⟩ (row-major).
        // Diagonal E[i][i] = F_i (the batched claim for polynomial i).
        // Off-diagonal entries are cross-evaluations required by the base WHIR
        // prover which applies every linear form to every vector.
        let mut eval_matrix: Vec<F> = Vec::with_capacity(num_g_polys * num_g_polys);
        for w in &weight_covectors {
            for v in &blinding_vecs {
                let eval: F = w.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
                eval_matrix.push(eval);
            }
        }

        #[cfg(feature = "tracing")]
        drop(_span_eval_matrix);

        // Send eval_matrix to transcript so the verifier can reconstruct it.
        // The verifier can compute diagonal entries from Λ claims but not off-diagonal.
        for eval in &eval_matrix {
            prover_state.prover_message(eval);
        }

        // Package weight covectors as LinearForm trait objects.
        let blinding_forms: Vec<Box<dyn LinearForm<F>>> = weight_covectors
            .into_iter()
            .map(|cv| Box::new(Covector::new(cv)) as Box<dyn LinearForm<F>>)
            .collect();

        // Run the second WHIR instance on the blinding polynomials.
        // The returned FinalClaim is discarded — it is not needed by the outer protocol.
        #[cfg(feature = "tracing")]
        let _span_inner_prove = tracing::info_span!("zk2_inner_blinding_prove").entered();

        let blinding_vector_cows: Vec<Cow<'_, [F]>> =
            blinding_vecs.into_iter().map(Cow::Owned).collect();
        let _ = self.blinding_polynomial.prove(
            prover_state,
            blinding_vector_cows,
            vec![Cow::Borrowed(&witness.blinding_witness)],
            blinding_forms,
            Cow::Owned(eval_matrix),
        );

        #[cfg(feature = "tracing")]
        drop(_span_inner_prove);
    }
}
