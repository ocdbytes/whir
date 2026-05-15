//! Mask-proximity (Construction 7.2) builder + Lemma 7.4 γ-combination bound.
//! ZK-only.

use ark_ff::Field;

use crate::{
    algebra::{embedding::Identity, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        irs_commit::Config as IrsConfig, mask_proximity::Config as MaskProximityConfig,
        params::spec::SecuritySpec, proof_of_work::Config as PowConfig,
    },
};

/// `c_zk.num_vectors` must equal `2 * num_masks` (originals + fresh).
/// PoW closes the Lemma 7.4 γ-combination gap to `spec.target_security_bits`.
pub fn solve<F: Field>(
    spec: &SecuritySpec,
    c_zk: IrsConfig<Identity<F>>,
    num_masks: usize,
) -> MaskProximityConfig<F> {
    let target_bits = Bits::new(f64::from(spec.target_security_bits));
    let analytic = analytic_error_bits(&c_zk, num_masks);
    let pow = PowConfig::grind_to(target_bits, analytic, spec.hash_id);
    MaskProximityConfig::new(c_zk, num_masks, pow)
}

/// γ-combination soundness (Lemma 7.4):
/// `log|F| − log(num_masks · (deg − 1))`, with `deg = c_zk.masked_message_length()`.
pub fn analytic_error_bits<F: Field>(c_zk: &IrsConfig<Identity<F>>, num_masks: usize) -> Bits {
    let field_bits = F::field_size_bits();
    let deg = c_zk.masked_message_length();
    if deg <= 1 || num_masks == 0 {
        return Bits::new(field_bits.max(0.0));
    }
    #[allow(clippy::cast_precision_loss)]
    let log_combined = ((num_masks * (deg - 1)) as f64).log2();
    Bits::new((field_bits - log_combined).max(0.0))
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{
        algebra::fields::Field64,
        hash,
        protocols::{
            irs_commit::IrsMode,
            params::{
                irs_commit as irs_solver,
                spec::{LogInvRate, MaskCodeMessageLen, Mode},
                test_utils::{
                    arb_zk_spec, assert_pow_closes_gap, deterministic_spec, TestEmbedding,
                    TEST_TARGET_RANGE,
                },
            },
        },
    };

    /// γ-combination (Lemma 7.4): `log|F| − log(num_masks · (deg − 1))`,
    /// `deg = c_zk.masked_message_length()`. With `num_masks = 0` or `deg ≤ 1`
    /// the bound saturates to `field_bits`.
    #[test]
    fn analytic_error_formula() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let num_masks = 3_usize;
        let c_zk = irs_solver::solve_mask_code::<TestEmbedding>(
            &spec,
            MaskCodeMessageLen::new(8),
            0,
            LogInvRate::new(1),
            2 * num_masks,
        );

        let got = f64::from(analytic_error_bits(&c_zk, num_masks));

        let field_bits = <Field64 as FieldWithSize>::field_size_bits();
        let deg = c_zk.masked_message_length();
        let log_combined = ((num_masks * (deg - 1)) as f64).log2();
        let expected = (field_bits - log_combined).max(0.0);

        assert!(
            (got - expected).abs() < 1e-9,
            "got {got} vs expected {expected}",
        );
    }

    /// Degenerate inputs (`num_masks == 0` or `deg ≤ 1`) saturate to `field_bits`.
    #[test]
    fn analytic_error_saturates_when_no_masks() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let c_zk = irs_solver::solve_mask_code::<TestEmbedding>(
            &spec,
            MaskCodeMessageLen::new(2),
            0,
            LogInvRate::new(1),
            2,
        );
        let bits = f64::from(analytic_error_bits(&c_zk, 0));
        let field_bits = <Field64 as FieldWithSize>::field_size_bits();
        assert_eq!(bits, field_bits.max(0.0));
    }

    proptest! {
        #[test]
        fn solve_assembles(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            log_inv_rate in 1u32..=3,
            num_masks in 1usize..=8,
            l_zk_log in 1u32..=5,
        ) {
            let l_zk = MaskCodeMessageLen::new(1usize << l_zk_log);
            let c_zk = irs_solver::solve_mask_code::<TestEmbedding>(
                &spec,
                l_zk,
                0,
                LogInvRate::new(log_inv_rate),
                2 * num_masks,
            );
            let config = solve(&spec, c_zk, num_masks);
            prop_assert_eq!(config.num_masks, num_masks);
            prop_assert_eq!(config.c_zk_commit.num_vectors, 2 * num_masks);
            prop_assert_eq!(config.c_zk_commit.interleaving_depth, 1);
        }

        /// `analytic_error + pow ≥ target` (Lemma 7.4 γ-combination).
        #[test]
        fn pow_closes_gap_to_target(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            log_inv_rate in 1u32..=3,
            num_masks in 1usize..=8,
            l_zk_log in 1u32..=5,
        ) {
            let l_zk = MaskCodeMessageLen::new(1usize << l_zk_log);
            let c_zk = irs_solver::solve_mask_code::<TestEmbedding>(
                &spec,
                l_zk,
                0,
                LogInvRate::new(log_inv_rate),
                2 * num_masks,
            );
            let analytic = analytic_error_bits(&c_zk, num_masks);
            let config = solve(&spec, c_zk, num_masks);
            assert_pow_closes_gap(&spec, analytic, &config.pow);
        }
    }

    #[test]
    #[should_panic(expected = "c_zk.num_vectors must be 2 * num_masks")]
    fn solve_rejects_mismatched_num_vectors() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let c_zk = irs_solver::solve_mask_code::<TestEmbedding>(
            &spec,
            MaskCodeMessageLen::new(2),
            0,
            LogInvRate::new(1),
            4,
        );
        let _ = solve(&spec, c_zk, 3);
    }

    #[test]
    #[should_panic(expected = "interleaving_depth = 1")]
    fn solve_rejects_non_unit_interleaving() {
        // All values except `NON_UNIT_INTERLEAVING_DEPTH` are chosen to satisfy
        // `Config::new`'s divisibility/pow2 constraints.
        const SECURITY_TARGET_BITS: f64 = 80.0;
        const UNIQUE_DECODING: bool = false;
        const NUM_VECTORS: usize = 2;
        const VECTOR_SIZE: usize = 8;
        const NON_UNIT_INTERLEAVING_DEPTH: usize = 2;
        const RATE: f64 = 0.5;
        const NUM_MASKS: usize = 1;

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let c_zk = IrsConfig::<Identity<Field64>>::new(
            SECURITY_TARGET_BITS,
            UNIQUE_DECODING,
            hash::BLAKE3,
            NUM_VECTORS,
            VECTOR_SIZE,
            NON_UNIT_INTERLEAVING_DEPTH,
            RATE,
            IrsMode::Standard,
        );
        let _ = solve(&spec, c_zk, NUM_MASKS);
    }
}
