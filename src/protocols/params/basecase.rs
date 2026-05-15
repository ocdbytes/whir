//! Basecase (Construction 7.2, p.43) parameter selection + γ-combination bound.

use ark_ff::Field;

use crate::{
    algebra::{embedding::Identity, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        basecase::{self, Config as BasecaseConfig},
        irs_commit::Config as IrsConfig,
        params::{
            irs_commit as irs_solver,
            spec::{Mode as SpecMode, OodSampleBudget, RoundContext, SecuritySpec},
            sumcheck as sumcheck_solver,
        },
        proof_of_work::Config as PowConfig,
        sumcheck::{self, Config as SumcheckConfig},
    },
};

/// PoW closes the Theorem 7.1 γ-slot gap to `spec.target_security_bits`; no
/// γ challenge in Standard mode ⇒ `Config::none()`.
pub fn solve<F: Field>(
    spec: &SecuritySpec,
    vector_size: usize,
    log_inv_rate: u32,
) -> BasecaseConfig<F> {
    assert!(vector_size > 0, "basecase requires vector_size ≥ 1");

    let ctx = RoundContext {
        round_index: 0,
        vector_size,
        log_inv_rate,
        folding_factor: 0,
    };
    let commit = irs_solver::solve(spec, &ctx, OodSampleBudget::new(0));

    let target_bits = Bits::new(f64::from(spec.target_security_bits));
    let sumcheck_pow = PowConfig::grind_to(
        target_bits,
        sumcheck_solver::analytic_error_bits(&commit, None),
        spec.hash_id,
    );
    let sumcheck = SumcheckConfig::new(
        vector_size,
        sumcheck_pow,
        vector_size.next_power_of_two().trailing_zeros() as usize,
        sumcheck::SumcheckMode::Standard,
    );

    let mode = match spec.mode {
        SpecMode::Standard => basecase::Mode::Standard,
        SpecMode::ZeroKnowledge => basecase::Mode::ZeroKnowledge,
    };

    let pow = match mode {
        basecase::Mode::Standard => PowConfig::none(),
        basecase::Mode::ZeroKnowledge => {
            PowConfig::grind_to(target_bits, analytic_error_bits(&commit), spec.hash_id)
        }
    };

    BasecaseConfig {
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
#[allow(clippy::float_cmp)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::test_utils::{
        arb_standard_johnson_spec, arb_zk_spec, assert_pow_closes_gap, deterministic_spec,
        TestField, TEST_TARGET_RANGE,
    };

    fn arb_dims() -> impl Strategy<Value = (u32, u32)> {
        (1u32..=4, 1u32..=3)
    }

    /// γ-combination soundness (Theorem 7.1, n=0): `log|F| − log|Λ(C^≡2, δ)|`.
    /// Builds the commit directly via the IRS solver to bypass `solve`'s PoW
    /// grind (which would assert against the cap for default test targets).
    #[test]
    fn analytic_error_formula() {
        use crate::protocols::params::{
            irs_commit as irs_solver,
            spec::{Mode, OodSampleBudget, RoundContext},
        };

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = RoundContext {
            round_index: 0,
            vector_size: 16,
            log_inv_rate: 2,
            folding_factor: 0,
        };
        let commit: IrsConfig<Identity<TestField>> =
            irs_solver::solve(&spec, &ctx, OodSampleBudget::new(0));

        let got = f64::from(analytic_error_bits(&commit));
        let field_bits = TestField::field_size_bits();
        let log_list = commit.list_size().log2();
        let expected = (field_bits - log_list).max(0.0);

        assert!(
            (got - expected).abs() < 1e-9,
            "got {got} vs expected {expected}",
        );
    }

    proptest! {
        #[test]
        fn solve_standard_assembles(
            spec in arb_standard_johnson_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate);
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
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate);
            prop_assert!(matches!(config.mode, basecase::Mode::ZeroKnowledge));
            prop_assert!(config.commit.mask_length() > 0);
        }

        #[test]
        fn pow_closes_gap_to_target_zk(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate);
            assert_pow_closes_gap(&spec, analytic_error_bits(&config.commit), &config.pow);
        }

        #[test]
        fn standard_mode_has_no_pow(
            spec in arb_standard_johnson_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate);
            prop_assert_eq!(config.pow, PowConfig::none());
        }
    }
}
