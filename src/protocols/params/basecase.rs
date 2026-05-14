//! Basecase (Construction 7.2, p.43) parameter selection + γ-combination bound.

use ark_ff::Field;

use crate::{
    algebra::{embedding::Identity, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        basecase,
        irs_commit::Config as IrsConfig,
        params::{
            irs_commit as irs_solver,
            spec::{Mode as SpecMode, OodSampleBudget, RoundContext, SecuritySpec},
            sumcheck as sumcheck_solver,
        },
        proof_of_work, sumcheck,
    },
};

/// PoW closes the Theorem 7.1 γ-slot gap to `spec.target_security_bits`; no
/// γ challenge in Standard mode ⇒ `Config::none()`.
pub fn solve<F: Field>(
    spec: &SecuritySpec<Identity<F>>,
    vector_size: usize,
    log_inv_rate: u32,
) -> basecase::Config<F> {
    assert!(vector_size > 0, "basecase requires vector_size ≥ 1");

    let ctx = RoundContext {
        round_index: 0,
        vector_size,
        log_inv_rate,
        folding_factor: 0,
    };
    let commit = irs_solver::solve(spec, &ctx, OodSampleBudget::new(0));

    let target_bits = Bits::new(f64::from(spec.target_security_bits));
    let sumcheck_pow = proof_of_work::Config::grind_to(
        target_bits,
        sumcheck_solver::analytic_error_bits(&commit, None),
        spec.hash_id,
    );
    let sumcheck = sumcheck::Config::new(
        vector_size,
        sumcheck_pow,
        vector_size.next_power_of_two().trailing_zeros() as usize,
        sumcheck::SumcheckMode::Standard,
    );

    let mode = match spec.mode {
        SpecMode::Standard { .. } => basecase::Mode::Standard,
        SpecMode::ZeroKnowledge => basecase::Mode::ZeroKnowledge,
    };

    let pow = match mode {
        basecase::Mode::Standard => proof_of_work::Config::none(),
        basecase::Mode::ZeroKnowledge => {
            proof_of_work::Config::grind_to(target_bits, analytic_error_bits(&commit), spec.hash_id)
        }
    };

    basecase::Config {
        commit,
        sumcheck,
        mode,
        pow,
    }
}

/// γ-combination soundness (Theorem 7.1, n=0): `log|F| − log|Λ(C^≡2, δ)|`.
pub fn analytic_error_bits<F: Field>(commit: &IrsConfig<Identity<F>>) -> Bits {
    let field_bits = F::field_size_bits();
    let log_list = commit.list_size().log2();
    Bits::new((field_bits - log_list).max(0.0))
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use proptest::prelude::*;

    use super::*;
    use crate::{
        algebra::{dot, multilinear_extend, random_vector},
        protocols::params::test_utils::{
            arb_standard_johnson_spec, arb_zk_spec, deterministic_spec, TestEmbedding,
        },
        transcript::{codecs::U64, DomainSeparator, ProverState, VerifierState},
    };

    // Keeps `target − error ≤ 60`, the cap `proof_of_work::threshold` enforces.
    const TEST_TARGET_RANGE: std::ops::RangeInclusive<u32> = 30..=50;

    fn arb_dims() -> impl Strategy<Value = (u32, u32)> {
        (1u32..=4, 1u32..=3)
    }

    proptest! {
        #[test]
        fn solve_standard_assembles(
            spec in arb_standard_johnson_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve(&spec, 1usize << log_size, log_inv_rate);
            prop_assert!(matches!(config.mode, basecase::Mode::Standard));
            prop_assert_eq!(config.commit.interleaving_depth, 1);
            prop_assert_eq!(config.commit.num_vectors, 1);
            prop_assert_eq!(config.commit.vector_size, config.sumcheck.initial_size);
        }

        #[test]
        fn solve_zk_assembles(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve(&spec, 1usize << log_size, log_inv_rate);
            prop_assert!(matches!(config.mode, basecase::Mode::ZeroKnowledge));
            prop_assert!(config.commit.mask_length() > 0);
        }

        #[test]
        fn pow_closes_gap_to_target_zk(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve(&spec, 1usize << log_size, log_inv_rate);
            let error = f64::from(analytic_error_bits(&config.commit));
            let pow_bits = f64::from(config.pow.difficulty());
            prop_assert!(
                error + pow_bits >= f64::from(spec.target_security_bits) - 1e-3,
                "error {} + pow {} < target {}",
                error, pow_bits, spec.target_security_bits,
            );
        }

        #[test]
        fn standard_mode_has_no_pow(
            spec in arb_standard_johnson_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve(&spec, 1usize << log_size, log_inv_rate);
            prop_assert_eq!(config.pow, proof_of_work::Config::none());
        }
    }

    fn round_trip(seed: u64, vector_size: usize, zk: bool) {
        let spec: SecuritySpec<TestEmbedding> = deterministic_spec(if zk {
            SpecMode::ZeroKnowledge
        } else {
            SpecMode::Standard {
                unique_decoding: false,
            }
        });
        let spec = SecuritySpec {
            target_security_bits: 40,
            ..spec
        };
        let config = solve(&spec, vector_size, 1);

        let mut rng = StdRng::seed_from_u64(seed);
        let vector = random_vector::<<TestEmbedding as crate::algebra::embedding::Embedding>::Source>(
            &mut rng,
            vector_size,
        );
        let covector = random_vector(&mut rng, vector_size);
        let sum = dot(&vector, &covector);

        let instance = U64(seed);
        let ds = DomainSeparator::protocol(&config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);

        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit.commit(&mut prover_state, &[&vector]);
        let prover_result = config.prove(
            &mut prover_state,
            vector.clone(),
            &witness,
            covector.clone(),
            sum,
        );
        assert_eq!(
            multilinear_extend(&covector, &prover_result.evaluation_points),
            prover_result.linear_form_evaluation,
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config
            .commit
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let verifier_result = config
            .verify(&mut verifier_state, &commitment, sum)
            .unwrap();
        verifier_state.check_eof().unwrap();
        assert_eq!(
            verifier_result.linear_form_evaluation,
            prover_result.linear_form_evaluation,
        );
    }

    #[test]
    fn round_trip_standard() {
        round_trip(0x5EED_5EED, 8, false);
    }

    #[test]
    fn round_trip_zk() {
        round_trip(0x5EED_5EED, 8, true);
    }
}
