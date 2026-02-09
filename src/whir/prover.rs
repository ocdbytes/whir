use ark_ff::{FftField, Field, PrimeField};
use ark_poly::EvaluationDomain;
use ark_std::rand::{CryptoRng, RngCore};
use zerocopy::IntoBytes;
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{
    committer::Witness,
    config::WhirConfig,
    statement::{Statement, Weights},
};
use crate::{
    algebra::{
        domain::Domain,
        embedding::{self, Embedding},
        poly_utils::{
            coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
        },
        tensor_product,
    },
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge,
        irs_commit, matrix_commit,
        sumcheck::{self, SumcheckSingle},
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, Encoding, NargSerialize,
        ProverMessage, ProverState, VerifierMessage,
    },
    type_info::Type,
    utils::{expand_randomness, zip_strict},
    whir::{
        config::RoundConfig,
        utils::{get_challenge_stir_queries, sample_ood_points},
        zk::{HelperEvaluations, ZkWitness},
    },
};

/// Encode a field element directly into a byte buffer without heap allocation.
///
/// Produces the same byte representation as spongefish's `Encoding<[u8]>` for
/// arkworks field elements: little-endian limb bytes, truncated to the
/// canonical byte width of the base prime field.
#[inline]
fn encode_field_element_into<F: Field>(f: &F, dst: &mut Vec<u8>) {
    let base_field_size = (F::BasePrimeField::MODULUS_BIT_SIZE.div_ceil(8)) as usize;
    for base_element in f.to_base_prime_field_elements() {
        let bigint = base_element.into_bigint();
        let limbs: &[u64] = bigint.as_ref();
        let all_bytes = limbs.as_bytes();
        dst.extend_from_slice(&all_bytes[..base_field_size]);
    }
}

impl<F: FftField> WhirConfig<F> {
    #[allow(clippy::too_many_lines)] // TODO
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&CoefficientList<F::BasePrimeField>],
        witnesses: &[&Witness<F>],
        statements: &[&Statement<F>],
    ) -> (MultilinearPoint<F>, Vec<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
        Vec<u8>: Encoding<[H::U]> + NargSerialize,
    {
        assert_eq!(polynomials.len(), statements.len());
        assert_eq!(
            polynomials.len(),
            witnesses.len() * self.initial_committer.num_polynomials
        );
        if polynomials.is_empty() {
            // TODO: Implement something sensible.
            unimplemented!("Empty case not implemented");
        }
        let num_variables = self.initial_committer.polynomial_size.trailing_zeros() as usize;

        // Validate statements
        for (polynomial, statement) in polynomials.iter().zip(statements) {
            assert_eq!(polynomial.num_variables(), num_variables);
            assert_eq!(statement.num_variables(), num_variables);

            #[cfg(debug_assertions)]
            {
                // In debug mode, verify all statment.
                // TODO: Add a `mixed_verify` function that takes an embedding into account.
                let polynomial = polynomial.lift(self.embedding());
                assert!(statement.verify(&polynomial));
            }
        }

        // Step 1: Commit to the full N×M constraint evaluation matrix BEFORE sampling γ.
        //
        // Security: The prover must commit to ALL cross-term evaluations (P_i(w_j) for all i,j)
        // before learning the batching randomness γ. This prevents the prover from adaptively
        // choosing P_i(w_j) values after seeing γ, which would allow breaking soundness by
        // constructing polynomials that satisfy P_batched = Σ γ^i·P_i at constraint points
        // but differ elsewhere.

        // Collect all constraint weights from OOD samples and statements
        let mut all_constraint_weights = Vec::new();
        all_constraint_weights.extend(witnesses.iter().flat_map(|w| {
            w.out_of_domain()
                .weights(&embedding::Identity::<F>::new(), num_variables)
        }));
        all_constraint_weights.extend(
            statements
                .iter()
                .flat_map(|s| s.constraints.iter().map(|c| c.weights.clone())),
        );
        let num_constraints = all_constraint_weights.len();

        // Complete evaluations of EVERY polynomial at EVERY constraint point
        // This creates an N×M matrix where N = #polynomials, M = #constraints
        // We commit this matrix to the script.
        let mut constraint_evals_matrix: Vec<Option<F>> =
            vec![None; polynomials.len() * num_constraints];
        // OODs Points
        let mut constraint_offset = 0;
        let mut polynomial_offset = 0;
        for witness in witnesses {
            for row in witness.out_of_domain().rows() {
                let mut index = polynomial_offset + constraint_offset;
                for value in row {
                    assert!(constraint_evals_matrix[index].is_none());
                    constraint_evals_matrix[index] = Some(*value);
                    index += num_constraints;
                }
                constraint_offset += 1;
            }
            polynomial_offset += witness.out_of_domain().num_columns() * num_constraints;
        }
        // Statement
        let mut index = witnesses
            .iter()
            .map(|w| w.out_of_domain().num_points())
            .sum::<usize>();
        for statement in statements {
            for constraint in &statement.constraints {
                assert!(constraint_evals_matrix[index].is_none());
                constraint_evals_matrix[index] = Some(constraint.sum);
                index += 1; // To next column
            }
            index += num_constraints; // To next row, same column
        }
        // Completion
        for (polynomial, row) in zip_strict(
            polynomials,
            constraint_evals_matrix.chunks_exact_mut(all_constraint_weights.len()),
        ) {
            let mut lifted = None;
            for (weights, cell) in zip_strict(&all_constraint_weights, row) {
                if cell.is_none() {
                    // TODO: Avoid lifting by evaluating directly through embedding.
                    let lifted = lifted.get_or_insert_with(|| polynomial.lift(self.embedding()));
                    let eval = weights.evaluate(lifted);
                    prover_state.prover_message(&eval);
                    *cell = Some(eval);
                }
            }
        }
        let constraint_evals_matrix: Vec<F> = constraint_evals_matrix
            .into_iter()
            .map(|e| e.unwrap())
            .collect();

        // Step 2: Sample batching randomness γ (cryptographically bound to committed matrix)
        let batching_weights: Vec<F> = geometric_challenge(prover_state, polynomials.len());

        // Step 3: Materialize the batched polynomial P_batched = P₀ + γ·P₁ + γ²·P₂ + ...
        // This also lifts the polynomial to the extension field.
        assert_eq!(batching_weights[0], F::ONE);
        let mut batched_coeffs = polynomials
            .first()
            .unwrap()
            .coeffs()
            .iter()
            .map(|c| self.embedding().map(*c))
            .collect::<Vec<_>>();
        for (weight, polynomial) in zip_strict(&batching_weights, polynomials).skip(1) {
            for (acc, src) in batched_coeffs.iter_mut().zip(polynomial.coeffs()) {
                *acc += self.embedding().mixed_mul(*weight, *src);
            }
        }
        let batched_poly = CoefficientList::new(batched_coeffs);

        // Step 4: Build combined statement using RLC of the committed evaluation matrix
        // For each constraint j: combined_eval[j] = Σᵢ γⁱ·eval[i][j]
        let mut combined_statement = Statement::new(num_variables);

        for (constraint_idx, weights) in all_constraint_weights.into_iter().enumerate() {
            let mut combined_eval = F::ZERO;
            for (weight, poly_evals) in batching_weights
                .iter()
                .zip(constraint_evals_matrix.chunks_exact(num_constraints))
            {
                combined_eval += *weight * poly_evals[constraint_idx];
            }
            combined_statement.add_constraint(weights, combined_eval);
        }

        // Run initial sumcheck on batched polynomial with combined statement
        let mut sumcheck_prover = None;
        let folding_randomness = if self.initial_statement {
            let combination_randomness_gen = prover_state.verifier_message();
            let sumcheck_config = sumcheck::Config {
                // TODO: Make part of parameters
                field: Type::<F>::new(),
                initial_size: batched_poly.num_coeffs(),
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: self.starting_folding_pow
                    };
                    self.folding_factor.at_round(0)
                ],
            };
            let mut sumcheck = SumcheckSingle::new(
                batched_poly.clone(),
                &combined_statement,
                combination_randomness_gen,
            );
            let folding_randomness = sumcheck_config.prove(prover_state, &mut sumcheck);
            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            let mut folding_randomness = vec![F::ZERO; self.folding_factor.at_round(0)];
            for randomness in &mut folding_randomness {
                *randomness = prover_state.verifier_message();
            }
            self.starting_folding_pow.prove(prover_state);
            MultilinearPoint(folding_randomness)
        };

        let mut randomness_vec = Vec::with_capacity(self.mv_parameters.num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(self.mv_parameters.num_variables, F::ZERO);

        let mut round_state = RoundState {
            domain: self.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            coefficients: batched_poly,
            prev_commitment: RoundWitness::Initial {
                witnesses,
                batching_weights,
            },
            randomness_vec,
            statement: combined_statement,
        };

        // Execute standard WHIR rounds on the batched polynomial
        for _round in 0..=self.n_rounds() {
            self.round(prover_state, &mut round_state);
        }

        // Hints for deferred constraints
        let constraint_eval =
            MultilinearPoint(round_state.randomness_vec.iter().copied().rev().collect());
        let deferred = round_state
            .statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect();

        prover_state.prover_hint_ark(&deferred);

        (constraint_eval, deferred)
    }

    pub fn prove_zk<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomial: &CoefficientList<F::BasePrimeField>,
        witness: &ZkWitness<F>,
        helper_config: &WhirConfig<F>,
        statement: &Statement<F>,
    ) -> (MultilinearPoint<F>, Vec<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
        Vec<u8>: Encoding<[H::U]> + NargSerialize,
    {
        // Step 1: Sample beta from verifier
        let beta: F = prover_state.verifier_message();

        // Step 2: construct g blinding polynomial
        // g(X) = g₀(X) + Σᵢ βⁱ · X^(2^(i-1)) · gᵢ(X)
        //
        // We build g into a mutable Vec, evaluate it at constraint points,
        // then transform it into P = ρ·f + g IN-PLACE to avoid a second
        // 2^μ-element allocation. This halves peak memory usage.
        let mu = witness.preprocessing.params.mu;
        let poly_size = 1 << mu;
        let mut coeffs = vec![F::ZERO; poly_size];
        // Copy g₀ coefficients
        let g0_coeffs = witness.preprocessing.g0_hat.coeffs();
        coeffs[..g0_coeffs.len()].copy_from_slice(g0_coeffs);

        // Terms 1..μ: β^i · X_(i-1) · gᵢ(X)
        //   where gᵢ(X) = ĝᵢ(pow(X))  (embedded ℓ-variate → μ-variate)
        //   and X_(i-1) is the (i-1)-th variable
        //
        // Each term writes to coeffs[shift..shift + 2^ℓ] where shift = 2^(i-1).
        // We add each term's contribution to the target slice using parallel iteration.
        let mut beta_power = beta;
        for i in 1..=mu {
            let variable_index = i - 1;
            let shift = 1 << variable_index;
            let g_hat_coeffs = witness.preprocessing.g_hats[i - 1].coeffs();
            let bp = beta_power;
            let target = &mut coeffs[shift..shift + g_hat_coeffs.len()];
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if target.len() >= 1024 {
                    target
                        .par_iter_mut()
                        .zip(g_hat_coeffs.par_iter())
                        .for_each(|(c, &g_c)| {
                            *c += bp * g_c;
                        });
                } else {
                    for (c, &g_c) in target.iter_mut().zip(g_hat_coeffs) {
                        *c += bp * g_c;
                    }
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (c, &g_c) in target.iter_mut().zip(g_hat_coeffs) {
                    *c += bp * g_c;
                }
            }
            beta_power *= beta;
        }

        // Step 3: evaluate g at constraint points (g is in `coeffs` right now)
        let g_as_poly = CoefficientList::new(coeffs);
        let g_evals: Vec<F> = statement
            .constraints
            .iter()
            .map(|constraint| {
                let eval = constraint.weights.evaluate(&g_as_poly);
                prover_state.prover_message(&eval);
                eval
            })
            .collect();

        // Step 4: Sample rho from verifier
        let rho: F = prover_state.verifier_message();

        // Step 5: Transform g → P = ρ·f + g IN-PLACE (no new allocation)
        let mut coeffs = g_as_poly.into_coeffs();
        {
            let embedding = self.embedding();
            let f_coeffs = polynomial.coeffs();
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                coeffs[..f_coeffs.len()]
                    .par_iter_mut()
                    .zip(f_coeffs.par_iter())
                    .for_each(|(c, &f_coeff)| {
                        *c += rho * embedding.map(f_coeff);
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (c, &f_coeff) in coeffs.iter_mut().zip(f_coeffs.iter()) {
                    *c += rho * embedding.map(f_coeff);
                }
            }
        }
        let p_poly = CoefficientList::new(coeffs);

        // Step 6: Build modified statement
        let mut modified_statement = Statement::new(statement.num_variables());
        for (original_constraint, g_eval) in statement.constraints.iter().zip(g_evals) {
            let new_sum = rho * original_constraint.sum + g_eval;
            modified_statement.add_constraint(original_constraint.weights.clone(), new_sum);
        }

        // Step 7: Run initial sumcheck on P with modified statement
        let mut sumcheck_prover = None;
        let folding_randomness = if self.initial_statement {
            let combination_randomness_gen = prover_state.verifier_message();
            let sumcheck_config = sumcheck::Config {
                field: Type::<F>::new(),
                initial_size: p_poly.num_coeffs(),
                rounds: vec![
                    sumcheck::RoundConfig {
                        pow: self.starting_folding_pow
                    };
                    self.folding_factor.at_round(0)
                ],
            };
            let mut sumcheck = SumcheckSingle::new(
                p_poly.clone(),
                &modified_statement,
                combination_randomness_gen,
            );
            let folding_randomness = sumcheck_config.prove(prover_state, &mut sumcheck);
            sumcheck_prover = Some(sumcheck);
            folding_randomness
        } else {
            let mut folding_randomness = vec![F::ZERO; self.folding_factor.at_round(0)];
            for randomness in &mut folding_randomness {
                *randomness = prover_state.verifier_message();
            }
            self.starting_folding_pow.prove(prover_state);
            MultilinearPoint(folding_randomness)
        };

        let num_variables = self.mv_parameters.num_variables;
        let mut randomness_vec = Vec::with_capacity(num_variables);
        randomness_vec.extend(folding_randomness.0.iter().rev().copied());
        randomness_vec.resize(num_variables, F::ZERO);

        // Step 8: Build round state
        let mut round_state = RoundState {
            domain: self.starting_domain.clone(),
            round: 0,
            sumcheck_prover,
            folding_randomness,
            coefficients: p_poly,
            prev_commitment: RoundWitness::InitialZk {
                witnesses: &[&witness.f_hat_witness],
                zk_witness: witness,
                helper_config,
                rho,
            },
            statement: modified_statement,
            randomness_vec,
        };

        // Step 9: Execute standard WHIR rounds
        for _round in 0..=self.n_rounds() {
            self.round(prover_state, &mut round_state);
        }

        // Step 10: Provide deferred hints
        let constraint_eval =
            MultilinearPoint(round_state.randomness_vec.iter().copied().rev().collect());
        let deferred = round_state
            .statement
            .constraints
            .iter()
            .filter(|constraint| constraint.defer_evaluation)
            .map(|constraint| constraint.weights.compute(&constraint_eval))
            .collect();

        prover_state.prover_hint_ark(&deferred);

        (constraint_eval, deferred)
    }

    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = round_state.coefficients.num_coeffs())))]
    fn round<H, R>(&self, prover_state: &mut ProverState<H, R>, round_state: &mut RoundState<F>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
        Vec<u8>: Encoding<[H::U]> + NargSerialize,
    {
        // Fold the coefficients

        let folded_coefficients = round_state
            .coefficients
            .fold(&round_state.folding_randomness);

        let num_variables =
            self.mv_parameters.num_variables - self.folding_factor.total_number(round_state.round);
        // num_variables should match the folded_coefficients here.
        assert_eq!(num_variables, folded_coefficients.num_variables());

        // Base case
        if round_state.round == self.n_rounds() {
            return self.final_round(prover_state, round_state, &folded_coefficients);
        }

        let round_params = &self.round_configs[round_state.round];

        // Compute the folding factors for later use
        let folding_factor = self.folding_factor.at_round(round_state.round);
        let folding_factor_next = self.folding_factor.at_round(round_state.round + 1);

        // Fold the coefficients, and compute fft of polynomial (and commit)
        let new_domain = round_state.domain.scale(2); // TODO: Why 2?
        let expansion = new_domain.size() / folded_coefficients.num_coeffs();
        let evals = self.reed_solomon.interleaved_encode(
            folded_coefficients.coeffs(),
            expansion,
            folding_factor_next,
        );

        // Commit to the matrix of evaluations
        let matrix_witness = round_params.matrix_committer.commit(prover_state, &evals);

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points(
            prover_state,
            round_params.ood_samples,
            num_variables,
            |point| folded_coefficients.evaluate(point),
        );

        // PoW
        round_params.pow.prove(prover_state);

        // Open the previous round's commitment, producing in-domain evaluations.
        // We prepend these with the OOD points and answers to produce a new
        // set of constraints for the next round.
        let (stir_challenges, stir_evaluations) = match &round_state.prev_commitment {
            RoundWitness::Initial {
                witnesses,
                batching_weights,
            } => {
                let in_domain = self.initial_committer.open(prover_state, witnesses);

                // Convert oods_points
                let mut points = ood_points
                    .iter()
                    .copied()
                    .map(|a| MultilinearPoint::expand_from_univariate(a, num_variables))
                    .collect::<Vec<_>>();
                let mut evals = ood_answers;

                // Convert RS evaluation point to multivariate over extension
                let weights = tensor_product(
                    batching_weights,
                    &round_state.folding_randomness.coeff_weights(true),
                );
                points.extend(in_domain.points(self.embedding(), num_variables));
                evals.extend(in_domain.values(self.embedding(), &weights));

                (points, evals)
            }
            RoundWitness::InitialZk {
                witnesses,
                zk_witness,
                helper_config,
                rho,
            } => {
                let in_domain = self.initial_committer.open(prover_state, witnesses);

                // 4. For each query point, compute helper evaluations at ALL k coset elements.
                // The IRS opening reveals k sub-polynomial evaluations per query. The virtual
                // oracle L = ρ·f̂ + h must be verified at ALL k coset positions so the verifier
                // can locally reconstruct fold_k(L, r̄)(α).
                let k = self.initial_committer.interleaving_depth;
                let num_rows = self.initial_committer.num_rows();
                let omega: F::BasePrimeField = self.initial_committer.original_domain_generator();
                let zeta = omega.pow([num_rows as u64]); // coset generator (primitive k-th root of unity)

                // Collect all k×q gamma values for batch evaluation
                let gammas: Vec<F> = in_domain
                    .indices
                    .iter()
                    .flat_map(|&idx| {
                        let coset_offset = omega.pow([idx as u64]);
                        (0..k).map(move |j| {
                            let gamma_base = coset_offset * zeta.pow([j as u64]);
                            self.embedding().map(gamma_base)
                        })
                    })
                    .collect();

                // Batch-evaluate all helper polynomials at all gamma points at once
                let helper_evals = zk_witness
                    .preprocessing
                    .batch_evaluate_helpers(&gammas, *rho);

                // 5. Prove helper evaluations via helper WHIR (covers k·q evaluation points)
                self.prover_helper_evaluations(
                    prover_state,
                    helper_config,
                    zk_witness,
                    &helper_evals,
                    *rho,
                );

                // 6. Compute virtual oracle values by evaluating fold_k(P, r̄)(α)
                // The prover does NOT send these to the verifier — the verifier computes
                // them locally from the IRS opening + verified helper evaluations.
                let virtual_values: Vec<F> = {
                    let embedding = self.embedding();
                    #[cfg(feature = "parallel")]
                    {
                        use rayon::prelude::*;
                        in_domain
                            .points
                            .par_iter()
                            .map(|&alpha_base| {
                                let alpha: F = embedding.map(alpha_base);
                                let point =
                                    MultilinearPoint::expand_from_univariate(alpha, num_variables);
                                folded_coefficients.evaluate(&point)
                            })
                            .collect()
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        in_domain
                            .points
                            .iter()
                            .map(|&alpha_base| {
                                let alpha: F = embedding.map(alpha_base);
                                let point =
                                    MultilinearPoint::expand_from_univariate(alpha, num_variables);
                                folded_coefficients.evaluate(&point)
                            })
                            .collect()
                    }
                };

                // 7. Build constraints using virtual values
                let mut points = ood_points
                    .iter()
                    .copied()
                    .map(|p| MultilinearPoint::expand_from_univariate(p, num_variables))
                    .collect::<Vec<_>>();
                let mut evals = ood_answers;

                points.extend(in_domain.points(self.embedding(), num_variables));
                evals.extend(virtual_values);

                (points, evals)
            }
            RoundWitness::Round {
                prev_matrix,
                prev_matrix_committer,
                prev_matrix_witness,
            } => {
                // STIR Queries
                let (stir_challenges, stir_challenges_indexes) = self.compute_stir_queries(
                    prover_state,
                    round_state,
                    num_variables,
                    round_params,
                    ood_points,
                );

                let fold_size = 1 << folding_factor;
                let leaf_size = fold_size;
                let answers: Vec<F> = stir_challenges_indexes
                    .iter()
                    .flat_map(|i| {
                        prev_matrix[i * leaf_size..(i + 1) * leaf_size]
                            .iter()
                            .copied()
                    })
                    .collect();

                assert_eq!(
                    answers.len(),
                    prev_matrix_committer.num_cols * stir_challenges_indexes.len()
                );
                prover_state.prover_hint_ark(&answers);
                prev_matrix_committer.open(
                    prover_state,
                    prev_matrix_witness,
                    &stir_challenges_indexes,
                );

                let mut stir_evaluations = ood_answers;
                stir_evaluations.extend(answers.chunks(fold_size).map(|answers| {
                    CoefficientList::new(answers.to_vec()).evaluate(&round_state.folding_randomness)
                }));
                (stir_challenges, stir_evaluations)
            }
        };

        // Randomness for combination
        let combination_randomness_gen = prover_state.verifier_message();
        let combination_randomness =
            expand_randomness(combination_randomness_gen, stir_challenges.len());

        #[allow(clippy::map_unwrap_or)]
        let mut sumcheck_prover = round_state
            .sumcheck_prover
            .take()
            .map(|mut sumcheck_prover| {
                sumcheck_prover.add_new_equality(
                    &stir_challenges,
                    &stir_evaluations,
                    &combination_randomness,
                );
                sumcheck_prover
            })
            .unwrap_or_else(|| {
                let mut statement = Statement::new(folded_coefficients.num_variables());

                for (point, eval) in stir_challenges.into_iter().zip(stir_evaluations) {
                    let weights = Weights::evaluation(point);
                    statement.add_constraint(weights, eval);
                }
                SumcheckSingle::new(
                    folded_coefficients.clone(),
                    &statement,
                    combination_randomness[1],
                )
            });

        let sumcheck_config = sumcheck::Config {
            field: Type::<F>::new(),
            initial_size: 1 << sumcheck_prover.num_variables(),
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: round_params.folding_pow,
                };
                folding_factor_next
            ],
        };
        let folding_randomness = sumcheck_config.prove(prover_state, &mut sumcheck_prover);

        let start_idx = self.folding_factor.total_number(round_state.round);
        let dst_randomness =
            &mut round_state.randomness_vec[start_idx..][..folding_randomness.0.len()];

        for (dst, src) in dst_randomness
            .iter_mut()
            .zip(folding_randomness.0.iter().rev())
        {
            *dst = *src;
        }

        // Update round state
        round_state.round += 1;
        round_state.domain = new_domain;
        round_state.sumcheck_prover = Some(sumcheck_prover);
        round_state.folding_randomness = folding_randomness;
        round_state.coefficients = folded_coefficients;
        round_state.prev_commitment = RoundWitness::Round {
            prev_matrix: evals,
            prev_matrix_committer: round_params.matrix_committer.clone(),
            prev_matrix_witness: matrix_witness,
        };
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = folded_coefficients.num_coeffs())))]
    fn final_round<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        round_state: &mut RoundState<F>,
        folded_coefficients: &CoefficientList<F>,
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
        Vec<u8>: Encoding<[H::U]> + NargSerialize,
    {
        // Directly send coefficients of the polynomial to the verifier.
        for coeff in folded_coefficients.coeffs() {
            prover_state.prover_message(coeff);
        }

        // Precompute the folding factors for later use
        let folding_factor = self.folding_factor.at_round(round_state.round);

        // PoW
        self.final_pow.prove(prover_state);

        match &round_state.prev_commitment {
            RoundWitness::Initial { witnesses, .. } => {
                for witness in *witnesses {
                    let in_domain = self.initial_committer.open(prover_state, &[witness]);

                    // The verifier will directly test these on the final polynomial.
                    // It has the final polynomial in full, so it needs no furhter help
                    // from us.
                    drop(in_domain);
                }
            }
            RoundWitness::InitialZk {
                witnesses,
                zk_witness,
                helper_config,
                rho,
                ..
            } => {
                // In the final round with ZK, we open f̂'s commitment AND must also
                // compute + prove helper evaluations. The verifier locally reconstructs
                // fold_k(L, r̄)(α) from the IRS opening + verified helper evaluations.
                let k = self.initial_committer.interleaving_depth;
                let num_rows = self.initial_committer.num_rows();
                let omega: F::BasePrimeField = self.initial_committer.original_domain_generator();
                let zeta = omega.pow([num_rows as u64]);

                for witness in *witnesses {
                    let in_domain = self.initial_committer.open(prover_state, &[witness]);

                    // Collect all k×q gamma values for batch evaluation
                    let gammas: Vec<F> = in_domain
                        .indices
                        .iter()
                        .flat_map(|&idx| {
                            let coset_offset = omega.pow([idx as u64]);
                            (0..k).map(move |j| {
                                let gamma_base = coset_offset * zeta.pow([j as u64]);
                                self.embedding().map(gamma_base)
                            })
                        })
                        .collect();

                    // Batch-evaluate all helper polynomials at all gamma points at once
                    let helper_evals = zk_witness
                        .preprocessing
                        .batch_evaluate_helpers(&gammas, *rho);

                    // Prove helper evaluations via helper WHIR proof (k·q points)
                    self.prover_helper_evaluations(
                        prover_state,
                        helper_config,
                        zk_witness,
                        &helper_evals,
                        *rho,
                    );
                    // No virtual_value messages — verifier computes them locally.
                }
            }
            RoundWitness::Round {
                prev_matrix,
                prev_matrix_committer,
                prev_matrix_witness,
            } => {
                // Final verifier queries and answers. The indices are over the
                // *folded* domain.
                let final_challenge_indexes = get_challenge_stir_queries(
                    prover_state,
                    // The size of the *original* domain before folding
                    round_state.domain.size(),
                    // The folding factor we used to fold the previous polynomial
                    folding_factor,
                    self.final_queries,
                );

                // Every query requires opening these many in the previous Merkle tree
                let fold_size = 1 << folding_factor;
                let answers = final_challenge_indexes
                    .iter()
                    .flat_map(|i| {
                        prev_matrix[i * fold_size..(i + 1) * fold_size]
                            .iter()
                            .copied()
                    })
                    .collect::<Vec<F>>();

                assert_eq!(
                    answers.len(),
                    prev_matrix_committer.num_cols * final_challenge_indexes.len()
                );
                prover_state.prover_hint_ark(&answers);
                prev_matrix_committer.open(
                    prover_state,
                    prev_matrix_witness,
                    &final_challenge_indexes,
                );
            }
        }

        // Final sumcheck
        let mut final_folding_sumcheck = round_state.sumcheck_prover.clone().unwrap_or_else(|| {
            SumcheckSingle::new(folded_coefficients.clone(), &round_state.statement, F::ONE)
        });

        let sumcheck_config = sumcheck::Config {
            field: Type::<F>::new(),
            initial_size: 1 << final_folding_sumcheck.num_variables(),
            rounds: vec![
                sumcheck::RoundConfig {
                    pow: self.final_folding_pow
                };
                self.final_sumcheck_rounds
            ],
        };

        let final_folding_randomness =
            sumcheck_config.prove(prover_state, &mut final_folding_sumcheck);

        let start_idx = self.folding_factor.total_number(round_state.round);
        let rand_dst = &mut round_state.randomness_vec
            [start_idx..start_idx + final_folding_randomness.0.len()];

        for (dst, src) in rand_dst
            .iter_mut()
            .zip(final_folding_randomness.0.iter().rev())
        {
            *dst = *src;
        }
    }

    fn prover_helper_evaluations<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        helper_config: &WhirConfig<F>,
        zk_witness: &ZkWitness<F>,
        helper_evals: &[HelperEvaluations<F>],
        rho: F,
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
        Vec<u8>: Encoding<[H::U]> + NargSerialize,
    {
        let preprocessing = &zk_witness.preprocessing;

        // 1. Send all claimed helper evaluations as a single batched transcript message.
        //
        //    Instead of calling prover_message() per element (~6 allocs each due to
        //    spongefish's double-encode), we pre-encode all evaluations into a single
        //    byte buffer using zero-copy field serialization, then send the buffer as
        //    one transcript operation (~3 allocs total).
        //
        //    Transcript compatibility: the byte representation is identical to sending
        //    individual elements, and DuplexSponge::absorb is associative, so the
        //    verifier can still read elements one-by-one.
        let mu = preprocessing.params.mu;
        let evals_per_point = 1 + mu; // m_eval + mu g_hat_evals
        let total_evals = helper_evals.len() * evals_per_point;
        let base_field_size = (F::BasePrimeField::MODULUS_BIT_SIZE.div_ceil(8)) as usize;
        let elem_bytes = base_field_size * F::extension_degree() as usize;
        let mut encoded = Vec::with_capacity(total_evals * elem_bytes);
        for helper_eval in helper_evals.iter() {
            encode_field_element_into(&helper_eval.m_eval, &mut encoded);
            for g_hat_eval in &helper_eval.g_hat_evals {
                encode_field_element_into(g_hat_eval, &mut encoded);
            }
        }
        prover_state.prover_messages_bytes::<F>(total_evals, encoded);

        // 2. Sample τ₂ for combining query points.
        //    NOTE: We do NOT sample τ₁ here. The batching of M, ĝ₁, ..., ĝμ
        //    is handled by helper_config.prove() internally via geometric_challenge.
        //    This ensures the proof is bound to the EXISTING commitments [[M]], [[ĝⱼ]]
        //    from commit_zk, rather than a freshly re-committed polynomial S.
        let tau2: F = prover_state.verifier_message();

        // 3. Compute per-polynomial claims using τ₂
        let ell = preprocessing.params.ell;
        let (m_claim, g_hat_claims) = self.compute_per_polynomial_claims(helper_evals, tau2);

        // 4. Construct the shared beq weight function for the helper WHIR sumcheck:
        //    beq(z,t) = eq(-ρ, t) · [Σᵢ τ₂ⁱ · eq(pow(γᵢ), z)]
        let beq_weights = self.construct_batched_eq_weights(helper_evals, rho, tau2, ell);

        // 5. Create per-polynomial statements.
        //    Each polynomial gets the same beq weight function but its own claimed sum.
        //    M's claim:    Σᵢ τ₂ⁱ · m(γᵢ, ρ)
        //    ĝⱼ's claim:  Σᵢ τ₂ⁱ · ĝⱼ(pow(γᵢ))
        let mut m_statement = Statement::new(ell + 1);
        m_statement.add_constraint(beq_weights.clone(), m_claim);

        let g_hat_statements: Vec<Statement<F>> = g_hat_claims
            .iter()
            .map(|&claim| {
                let mut stmt = Statement::new(ell + 1);
                stmt.add_constraint(beq_weights.clone(), claim);
                stmt
            })
            .collect();

        // 6. Collect all polynomials (base-field) and the single batch witness.
        //    Order: [M, ĝ₁, ..., ĝμ]
        //    The prove() function will:
        //      a) Compute and commit the cross-term evaluation matrix
        //      b) Sample batching weight γ (which plays the role of τ₁)
        //      c) Form the batched polynomial S = M + γ·ĝ₁ + ... + γ^μ·ĝμ
        //      d) Run sumcheck + FRI on S against the combined statement
        //    This binds the proof to the EXISTING batch commitment from commit_zk.
        let mut all_polynomials: Vec<&CoefficientList<F::BasePrimeField>> =
            Vec::with_capacity(1 + preprocessing.params.mu);
        all_polynomials.push(&zk_witness.m_poly_base);
        for g_hat_base in &zk_witness.g_hats_embedded_base {
            all_polynomials.push(g_hat_base);
        }

        // Single batch witness (helper_config.batch_size = μ+1, so one witness = all polys)
        let all_witnesses: Vec<&irs_commit::Witness<F::BasePrimeField, F>> =
            vec![&zk_witness.helper_witness];

        let mut all_statements: Vec<&Statement<F>> =
            Vec::with_capacity(1 + preprocessing.params.mu);
        all_statements.push(&m_statement);
        for stmt in &g_hat_statements {
            all_statements.push(stmt);
        }

        // Run helper WHIR prove with existing batch commitment (no re-commitment!)
        helper_config.prove(
            prover_state,
            &all_polynomials,
            &all_witnesses,
            &all_statements,
        );
    }

    fn compute_stir_queries<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        round_state: &RoundState<F>,
        num_variables: usize,
        round_params: &RoundConfig<F>,
        ood_points: Vec<F>,
    ) -> (Vec<MultilinearPoint<F>>, Vec<usize>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
    {
        let stir_challenges_indexes = get_challenge_stir_queries(
            prover_state,
            round_state.domain.size(),
            self.folding_factor.at_round(round_state.round),
            round_params.num_queries,
        );

        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state
            .domain
            .backing_domain
            .element(1 << self.folding_factor.at_round(round_state.round));
        let stir_challenges = ood_points
            .into_iter()
            .chain(
                stir_challenges_indexes
                    .iter()
                    .map(|i| domain_scaled_gen.pow([*i as u64])),
            )
            .map(|univariate| MultilinearPoint::expand_from_univariate(univariate, num_variables))
            .collect();

        (stir_challenges, stir_challenges_indexes)
    }

    /// Compute per-polynomial claims for the helper WHIR batch proof.
    ///
    /// Returns (m_claim, g_hat_claims) where:
    /// - m_claim = Σᵢ τ₂ⁱ · m(γᵢ, ρ)
    /// - g_hat_claims[j] = Σᵢ τ₂ⁱ · ĝⱼ(pow(γᵢ))
    ///
    /// These are the individual inner-product claims for each polynomial
    /// against the shared beq weight function. The batching across polynomials
    /// (the τ₁ weighting) is handled internally by helper_config.prove().
    pub(crate) fn compute_per_polynomial_claims(
        &self,
        helper_evals: &[HelperEvaluations<F>],
        tau2: F,
    ) -> (F, Vec<F>) {
        let num_g_hats = helper_evals.first().map_or(0, |h| h.g_hat_evals.len());

        let mut m_claim = F::ZERO;
        let mut g_hat_claims = vec![F::ZERO; num_g_hats];
        let mut tau2_power = F::ONE;

        for helper in helper_evals {
            // M's contribution: τ₂ⁱ · m(γᵢ, ρ)
            m_claim += tau2_power * helper.m_eval;

            // Each ĝⱼ's contribution: τ₂ⁱ · ĝⱼ(pow(γᵢ))
            for (j, &g_eval) in helper.g_hat_evals.iter().enumerate() {
                g_hat_claims[j] += tau2_power * g_eval;
            }

            tau2_power *= tau2;
        }

        (m_claim, g_hat_claims)
    }

    /// Construct the weight function for the helper WHIR sumcheck:
    ///
    ///   w(z, t) = eq(-ρ, t) · [Σᵢ τ₂ⁱ · eq(pow(γᵢ), z)]
    ///
    /// The τ₁ batching of ĝⱼ polynomials is handled by the polynomial S(z,t),
    /// not by the weight function.
    ///
    /// Parameters:
    /// - `helper_evals`: Contains the query points γᵢ
    /// - `rho`: The masking challenge ρ (we use -ρ in the eq polynomial)
    /// - `tau2`: Batching randomness for combining query points (γ₁, γ₂, ...)
    /// - `ell`: Number of variables ℓ for the z-space
    ///
    /// Returns: A `Weights::Linear` on (ℓ+1) variables
    pub(crate) fn construct_batched_eq_weights(
        &self,
        helper_evals: &[HelperEvaluations<F>],
        rho: F,
        tau2: F,
        ell: usize,
    ) -> Weights<F> {
        let neg_rho = -rho;
        let z_size = 1 << ell;
        let weight_size = 1 << (ell + 1);

        // ── Butterfly expansion of eq weights ──
        //
        // For each γᵢ, we need eq(pow(γᵢ), z) for ALL z ∈ {0,1}^ℓ.
        // The butterfly expansion computes ALL 2^ℓ eq values for a single γᵢ
        // in O(2^ℓ) using the tensor-product structure of eq:
        //   eq(c, z) = ∏ⱼ (cⱼzⱼ + (1-cⱼ)(1-zⱼ))
        //
        // Start with [1], then for each coordinate cⱼ of pow(γᵢ),
        // double the array: entry at z with bit j = 0 gets factor (1-cⱼ),
        //                    entry at z with bit j = 1 gets factor cⱼ.
        // Total: O(2^ℓ) per γᵢ, giving O(k×q × 2^ℓ) overall — a factor ℓ faster.

        // Accumulate Σᵢ τ₂ⁱ · eq(pow(γᵢ), z) for all z ∈ {0,1}^ℓ
        //
        // Each γᵢ's butterfly expansion is independent. We compute them in parallel
        // (when the parallel feature is enabled), then reduce into a single batched_eq vector.
        let tau2_powers: Vec<F> = {
            let mut powers = Vec::with_capacity(helper_evals.len());
            let mut p = F::ONE;
            for _ in 0..helper_evals.len() {
                powers.push(p);
                p *= tau2;
            }
            powers
        };

        // Butterfly expansion: compute τ₂ⁱ · eq(pow(γᵢ), z) for all z per γᵢ,
        // then reduce into a single batched_eq vector.
        //
        // Uses parallel tree reduction when available: each thread computes a
        // partial sum of a subset of gamma points, then partial sums are merged.
        // This avoids allocating k*q separate Vec<F> before reducing.
        // Max ℓ supported for stack-allocated power buffer.
        // ℓ is typically 8-10 and always < 64 by protocol constraints.
        const MAX_ELL: usize = 64;
        assert!(
            ell <= MAX_ELL,
            "ℓ={ell} exceeds stack buffer size {MAX_ELL}"
        );

        let compute_weighted_eq = |(helper, &tau2_pow): (&HelperEvaluations<F>, &F)| -> Vec<F> {
            // Compute γ powers on the STACK to avoid heap allocation per gamma point.
            // expand_from_univariate(γ, ℓ) computes [γ, γ², γ⁴, ..., γ^(2^(ℓ-1))]
            // then reverses. We compute forward and iterate in reverse for big-endian order.
            let mut powers_buf = [F::ZERO; MAX_ELL];
            let mut cur = helper.gamma;
            for p in powers_buf[..ell].iter_mut() {
                *p = cur;
                cur *= cur;
            }

            let mut eq_vals = Vec::with_capacity(z_size);
            eq_vals.push(F::ONE);
            // Process in FORWARD (big-endian) order: powers[ℓ-1], ..., powers[0]
            for &ci in powers_buf[..ell].iter().rev() {
                let len = eq_vals.len();
                let one_minus_ci = F::ONE - ci;
                eq_vals.resize(2 * len, F::ZERO);
                for j in (0..len).rev() {
                    eq_vals[2 * j + 1] = eq_vals[j] * ci;
                    eq_vals[2 * j] = eq_vals[j] * one_minus_ci;
                }
            }
            // Scale by τ₂ⁱ
            for v in eq_vals.iter_mut() {
                *v *= tau2_pow;
            }
            eq_vals
        };

        // Parallel tree reduction: compute and sum in one pass, avoiding
        // a separate allocation + reduction step.
        #[cfg(feature = "parallel")]
        let batched_eq: Vec<F> = {
            use rayon::prelude::*;
            helper_evals
                .par_iter()
                .zip(tau2_powers.par_iter())
                .fold(
                    || vec![F::ZERO; z_size],
                    |mut acc, pair| {
                        let eq_vals = compute_weighted_eq(pair);
                        for (a, v) in acc.iter_mut().zip(eq_vals) {
                            *a += v;
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![F::ZERO; z_size],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b) {
                            *ai += bi;
                        }
                        a
                    },
                )
        };
        #[cfg(not(feature = "parallel"))]
        let batched_eq: Vec<F> = {
            let mut batched = vec![F::ZERO; z_size];
            for pair in helper_evals.iter().zip(tau2_powers.iter()) {
                let eq_vals = compute_weighted_eq(pair);
                for (a, v) in batched.iter_mut().zip(eq_vals) {
                    *a += v;
                }
            }
            batched
        };

        // ── Build weight evaluations on {0,1}^(ℓ+1) ──
        // Index layout: index = z_idx * 2 + t_bit (t is the last variable, LSB)
        //
        // w(z, t) = eq(-ρ, t) × batched_eq[z]
        // eq(-ρ, 0) = 1 + ρ,  eq(-ρ, 1) = -ρ
        let eq_neg_rho_at_0 = F::ONE - neg_rho; // = 1 + ρ
        let eq_neg_rho_at_1 = neg_rho; // = -ρ

        let mut weight_evals = vec![F::ZERO; weight_size];
        for (z_idx, &beq_z) in batched_eq.iter().enumerate() {
            weight_evals[z_idx * 2] = eq_neg_rho_at_0 * beq_z; // t = 0
            weight_evals[z_idx * 2 + 1] = eq_neg_rho_at_1 * beq_z; // t = 1
        }

        Weights::linear(EvaluationsList::new(weight_evals))
    }
}

pub(crate) enum RoundWitness<'a, F: FftField> {
    Initial {
        witnesses: &'a [&'a irs_commit::Witness<F::BasePrimeField, F>],
        batching_weights: Vec<F>,
    },
    InitialZk {
        witnesses: &'a [&'a irs_commit::Witness<F::BasePrimeField, F>],
        zk_witness: &'a ZkWitness<F>,
        helper_config: &'a WhirConfig<F>,
        rho: F,
    },
    Round {
        prev_matrix: Vec<F>,
        prev_matrix_committer: matrix_commit::Config<F>,
        prev_matrix_witness: matrix_commit::Witness,
    },
}

/// Represents the prover state during a single round of the WHIR protocol.
///
/// Each WHIR round folds the polynomial, commits to the new evaluations,
/// responds to verifier queries, and updates internal randomness for the next step.
/// This struct tracks all data needed to perform that round, and passes it forward
/// across recursive iterations.
pub(crate) struct RoundState<'a, F>
where
    F: FftField,
{
    /// Index of the current WHIR round (0-based).
    ///
    /// Increases after each folding iteration.
    pub(crate) round: usize,

    /// Domain over which the current polynomial is evaluated.
    ///
    /// Grows with each round due to NTT expansion.
    pub(crate) domain: Domain<F>,

    /// Optional sumcheck prover used to enforce constraints.
    ///
    /// Present in rounds with non-empty constraint systems.
    pub(crate) sumcheck_prover: Option<SumcheckSingle<F>>,

    /// Folding randomness sampled by the verifier.
    ///
    /// Used to reduce the number of variables in the polynomial.
    pub(crate) folding_randomness: MultilinearPoint<F>,

    /// Current polynomial in coefficient form.
    ///
    /// Folded and evaluated to produce new commitments and Merkle trees.
    pub(crate) coefficients: CoefficientList<F>,

    /// Matrix commitment to the polynomial evaluations from the previous round.
    ///
    /// Used to prove query openings from the folded function.
    pub(crate) prev_commitment: RoundWitness<'a, F>,

    /// Flat list of evaluations corresponding to `prev_merkle` leaves.
    ///
    /// Each folded function is evaluated on a domain and split into leaves.

    /// Accumulator for all folding randomness across rounds.
    ///
    /// Ordered with the most recent round’s randomness at the front.
    pub(crate) randomness_vec: Vec<F>,

    /// Constraint system being enforced in this round.
    ///
    /// May be updated during recursion as queries are folded and batched.
    pub(crate) statement: Statement<F>,
}
