//! The atomic round state shared by single-track, pre-merge, and post-merge batched flows.
//!
//! Each flow reduces to a sequence of `prove_whir_round` / `verify_whir_round`
//! calls on a `Block` that carries the round's `(message, covector, sum)` plus
//! the active source(s) being opened. What differs across flows is how the
//! initial `Block` is built (single witness vs. γ-RLC of a bundle) and how
//! many active sources it carries (1 or `t` after a selector merge).

use ark_ff::Field;

use crate::protocols::irs_commit::{Commitment as IrsCommitment, Witness as IrsWitness};

/// Prover-side round state.
///
/// `witnesses` holds the active source IRS witnesses; in single-track and
/// pre-merge rounds it has length 1 with `theta = [F::ONE]`. After a selector
/// merge it has length `t` with `theta` from the merge opening.
pub(crate) struct ProverBlock<F: Field> {
    pub(crate) message: Vec<F>,
    pub(crate) covector: Vec<F>,
    pub(crate) sum: F,
    pub(crate) witnesses: Vec<IrsWitness<F>>,
    pub(crate) theta: Vec<F>,
}

impl<F: Field> ProverBlock<F> {
    pub(crate) fn single_source(
        message: Vec<F>,
        covector: Vec<F>,
        sum: F,
        witness: IrsWitness<F>,
    ) -> Self {
        Self {
            message,
            covector,
            sum,
            witnesses: vec![witness],
            theta: vec![F::ONE],
        }
    }
}

/// Verifier-side round state.
///
/// Mirror of [`ProverBlock`] on the receive side: holds the active source IRS
/// commitments and the post-sumcheck running `sum`. The verifier never
/// materialises `(message, covector)` — they're folded into the implicit
/// constraint accumulator owned by the caller.
pub(crate) struct VerifierBlock<F: Field> {
    pub(crate) sum: F,
    pub(crate) commitments: Vec<IrsCommitment>,
    pub(crate) theta: Vec<F>,
}

impl<F: Field> VerifierBlock<F> {
    pub(crate) fn single_source(sum: F, commitment: IrsCommitment) -> Self {
        Self {
            sum,
            commitments: vec![commitment],
            theta: vec![F::ONE],
        }
    }
}
