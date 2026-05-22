//! Basecase (Construction 7.2, p.43) parameter selection + γ-combination bound.

use ark_ff::Field;

use crate::{
    algebra::{embedding::Identity, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        basecase::{self, Config as BasecaseConfig},
        irs_commit::Config as IrsConfig,
        params::{
            error::{grind_to_at, DeriveError, Pow},
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
) -> Result<BasecaseConfig<F>, DeriveError> {
    assert!(vector_size > 0, "basecase requires vector_size ≥ 1");

    let ctx = RoundContext {
        vector_size,
        log_inv_rate,
        folding_factor: 0,
    };
    let commit = irs_solver::solve(spec, &ctx, OodSampleBudget::ZERO);

    let sumcheck_pow = grind_to_at(
        spec,
        sumcheck_solver::analytic_error_bits(&commit, None),
        Pow::BasecaseSumcheck,
    )?;
    let sumcheck = SumcheckConfig::new(
        vector_size,
        sumcheck_pow,
        vector_size.next_power_of_two().trailing_zeros() as usize,
        sumcheck::SumcheckMode::Standard,
    );

    let mode = match spec.mode {
        SpecMode::Standard => basecase::BasecaseMode::Standard,
        SpecMode::ZeroKnowledge => basecase::BasecaseMode::ZeroKnowledge,
    };

    let pow = match mode {
        basecase::BasecaseMode::Standard => PowConfig::none(),
        basecase::BasecaseMode::ZeroKnowledge => grind_to_at(
            spec,
            analytic_error_bits(&commit),
            Pow::BasecaseGammaCombination,
        )?,
    };

    Ok(BasecaseConfig::new(commit, sumcheck, mode, pow))
}

/// γ-combination soundness (Lemma 7.4 combination-randomness slot, paper p.45).
/// At `n = 0` the `C_zk` factors vanish; `ε_mca(C, δ)` does not.
pub fn analytic_error_bits<F: Field>(commit: &IrsConfig<Identity<F>>) -> Bits {
    let field_bits = F::field_size_bits();
    let log_list = commit.list_size().log2();
    let prox_gaps = commit.rbr_soundness_fold_prox_gaps();
    let poly_id = field_bits - log_list;
    Bits::new(prox_gaps.min(poly_id).max(0.0))
}

impl<F: Field> BasecaseConfig<F> {
    /// Analytic soundness bits (excluding PoW): `min(sumcheck round error, γ-slot error)`.
    /// The γ-slot only contributes in ZK mode; Standard collapses to the
    /// sumcheck term.
    pub fn analytic_bits(&self) -> Bits {
        let sumcheck_term = f64::from(sumcheck_solver::analytic_error_bits(&self.commit, None));
        let min_bits = match self.mode {
            basecase::BasecaseMode::Standard => sumcheck_term,
            basecase::BasecaseMode::ZeroKnowledge => {
                sumcheck_term.min(f64::from(analytic_error_bits(&self.commit)))
            }
        };
        Bits::new(min_bits.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::test_utils::{
        arb_standard_johnson_spec, arb_zk_spec, assert_close, assert_pow_closes_gap,
        deterministic_spec, TestField, TEST_TARGET_RANGE,
    };

    /// `vector_size = 16` (2^4) and `log_inv_rate = 2` give a small but
    /// non-degenerate basecase IRS. `folding_factor = 0` is the basecase
    /// invariant (no folding, message_length = vector_size).
    const FIXTURE_VECTOR_SIZE: usize = 16;
    const FIXTURE_LOG_INV_RATE: u32 = 2;

    fn arb_dims() -> impl Strategy<Value = (u32, u32)> {
        (1u32..=4, 1u32..=3)
    }

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
            vector_size: FIXTURE_VECTOR_SIZE,
            log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: 0,
        };
        let commit: IrsConfig<Identity<TestField>> =
            irs_solver::solve(&spec, &ctx, OodSampleBudget::ZERO);

        let got = f64::from(analytic_error_bits(&commit));
        let field_bits = TestField::field_size_bits();
        let log_list = commit.list_size().log2();
        let prox_gaps = commit.rbr_soundness_fold_prox_gaps();
        let poly_id = field_bits - log_list;
        let expected = prox_gaps.min(poly_id).max(0.0);

        assert_close(got, expected);
    }

    /// At `log_inv_rate = 1` on `Field64`, `ε_mca` is below the poly-identity
    /// term — pins the `min` to the prox-gaps arm rather than `poly_id`.
    #[test]
    fn analytic_error_uses_eps_mca_when_limiting() {
        use crate::protocols::params::{
            irs_commit as irs_solver,
            spec::{Mode, OodSampleBudget, RoundContext},
        };

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = RoundContext {
            vector_size: FIXTURE_VECTOR_SIZE,
            log_inv_rate: 1,
            folding_factor: 0,
        };
        let commit: IrsConfig<Identity<TestField>> =
            irs_solver::solve(&spec, &ctx, OodSampleBudget::ZERO);

        let field_bits = TestField::field_size_bits();
        let log_list = commit.list_size().log2();
        let prox_gaps = commit.rbr_soundness_fold_prox_gaps();
        let poly_id = field_bits - log_list;
        assert!(
            prox_gaps < poly_id,
            "fixture wants prox_gaps to bind: prox_gaps {prox_gaps} ≥ poly_id {poly_id}",
        );

        let got = f64::from(analytic_error_bits(&commit));
        assert_close(got, prox_gaps.max(0.0));
    }

    proptest! {
        #[test]
        fn solve_standard_assembles(
            spec in arb_standard_johnson_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate).unwrap();
            prop_assert!(matches!(config.mode, basecase::BasecaseMode::Standard));
            prop_assert_eq!(config.commit.interleaving_depth, 1);
            prop_assert_eq!(config.commit.num_vectors, 1);
            prop_assert_eq!(config.commit.vector_size, config.sumcheck.initial_size);
        }

        #[test]
        fn solve_zk_assembles(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate).unwrap();
            prop_assert!(matches!(config.mode, basecase::BasecaseMode::ZeroKnowledge));
            prop_assert!(config.commit.mask_length() > 0);
        }

        #[test]
        fn pow_closes_gap_to_target_zk(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate).unwrap();
            assert_pow_closes_gap(&spec, analytic_error_bits(&config.commit), &config.pow);
        }

        #[test]
        fn standard_mode_has_no_pow(
            spec in arb_standard_johnson_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate).unwrap();
            prop_assert_eq!(config.pow, PowConfig::none());
        }
    }
}
