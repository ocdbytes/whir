//! Parameter selection for the mask-proximity protocol (Construction 7.2).
//!
//! Mask-proximity spot-checks each committed mask oracle against C_zk via
//! γ-combination (Lemma 7.4). ZK-only — Standard mode never invokes it.

use ark_ff::Field;

use crate::{
    algebra::embedding::Identity,
    protocols::{irs_commit, mask_proximity},
};

/// Assemble a [`mask_proximity::Config`] from the shared C_zk IRS config and
/// the number of mask polynomials the protocol commits to.
///
/// `c_zk` must be sized with `num_vectors == 2 * num_masks` (Construction 7.2
/// commits originals and their fresh mask-of-masks side by side in the shared
/// tree). The orchestrator obtains `c_zk` via `irs_commit::solve_mask_code`
/// with that same `num_vectors`.
pub fn solve<F: Field>(
    c_zk: irs_commit::Config<Identity<F>>,
    num_masks: usize,
) -> mask_proximity::Config<F> {
    mask_proximity::Config::new(c_zk, num_masks)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::{
        irs_commit as params_irs,
        spec::{LogInvRate, MaskCodeMessageLen},
        test_utils::arb_zk_spec,
    };

    proptest! {
        /// `solve` produces a Config satisfying `mask_proximity::Config::new`'s
        /// invariants (`num_vectors == 2 * num_masks`, `interleaving_depth == 1`).
        #[test]
        fn solve_assembles(
            spec in arb_zk_spec(80..=128),
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
            let config = solve(c_zk, num_masks);
            prop_assert_eq!(config.num_masks, num_masks);
            prop_assert_eq!(config.c_zk_commit.num_vectors, 2 * num_masks);
            prop_assert_eq!(config.c_zk_commit.interleaving_depth, 1);
        }
    }
}
