//! Mask-proximity (Construction 7.2) builder + Lemma 7.4 γ-combination bound.
//! ZK-only.

use ark_ff::Field;

use crate::{
    algebra::{embedding::Identity, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        irs_commit::Config as IrsConfig, mask_proximity, params::spec::SecuritySpec, proof_of_work,
    },
};

/// `c_zk.num_vectors` must equal `2 * num_masks` (originals + fresh).
/// PoW closes the Lemma 7.4 γ-combination gap to `spec.target_security_bits`.
pub fn solve<F: Field>(
    spec: &SecuritySpec<Identity<F>>,
    c_zk: IrsConfig<Identity<F>>,
    num_masks: usize,
) -> mask_proximity::Config<F> {
    let target_bits = Bits::new(f64::from(spec.target_security_bits));
    let analytic = analytic_error_bits(&c_zk, num_masks);
    let pow = proof_of_work::Config::grind_to(target_bits, analytic, spec.hash_id);
    mask_proximity::Config::new(c_zk, num_masks, pow)
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
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{
        algebra::fields::Field64,
        hash,
        protocols::{
            irs_commit::IrsMode,
            params::{
                irs_commit as params_irs,
                spec::{LogInvRate, MaskCodeMessageLen, Mode},
                test_utils::{arb_zk_spec, deterministic_spec, TestEmbedding},
            },
        },
    };

    // Keeps `target − error ≤ 60`, the cap `proof_of_work::threshold` enforces.
    const TEST_TARGET_RANGE: std::ops::RangeInclusive<u32> = 30..=50;

    proptest! {
        #[test]
        fn solve_assembles(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            log_inv_rate in 1u32..=3,
            num_masks in 1usize..=8,
            l_zk_log in 1u32..=5,
        ) {
            let l_zk = MaskCodeMessageLen::new(1usize << l_zk_log);
            let c_zk = params_irs::solve_mask_code(
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
            let c_zk = params_irs::solve_mask_code(
                &spec,
                l_zk,
                0,
                LogInvRate::new(log_inv_rate),
                2 * num_masks,
            );
            let analytic = f64::from(analytic_error_bits(&c_zk, num_masks));
            let config = solve(&spec, c_zk, num_masks);
            let pow_bits = f64::from(config.pow.difficulty());
            prop_assert!(
                analytic + pow_bits >= f64::from(spec.target_security_bits) - 1e-3,
                "analytic {} + pow {} < target {}",
                analytic, pow_bits, spec.target_security_bits,
            );
        }
    }

    #[test]
    #[should_panic(expected = "c_zk.num_vectors must be 2 * num_masks")]
    fn solve_rejects_mismatched_num_vectors() {
        let spec = deterministic_spec::<TestEmbedding>(Mode::ZeroKnowledge);
        let c_zk = params_irs::solve_mask_code(
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
        let spec = deterministic_spec::<TestEmbedding>(Mode::ZeroKnowledge);
        let c_zk = crate::protocols::irs_commit::Config::<Identity<Field64>>::new(
            80.0,
            false,
            hash::BLAKE3,
            2,
            8,
            2, // interleaving_depth ≠ 1 — triggers the panic
            0.5,
            IrsMode::Standard,
        );
        let _ = solve(&spec, c_zk, 1);
    }
}
