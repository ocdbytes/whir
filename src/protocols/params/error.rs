//! Errors raised by [`super::derive::ProtocolConfig::derive`] and the
//! sub-protocol solvers.
//!
//! Two layers: [`super::super::proof_of_work::PowError`] for grinding-cap
//! failures, [`DeriveError`] for everything `derive()` can surface. The latter
//! wraps the former via [`DeriveError::PowUngrindable::source`] so callers can
//! walk the `std::error::Error::source()` chain.

use thiserror::Error;

use crate::{bits::Bits, protocols::proof_of_work::PowError};

/// Coordinate of a PoW slot in the derived protocol. Two axes: where the
/// slot lives (basecase vs. a numbered round) and which sub-protocol owns it.
/// Only valid combinations are representable.
///
/// `Error` is derived for the Display propagation it provides in
/// [`DeriveError`]'s `#[error("...")]` attributes; this type isn't itself a
/// failure (`source()` is always `None`).
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum PowSlot {
    #[error("basecase {0}")]
    Basecase(BasecaseSlot),
    #[error("round {index} {kind}")]
    Round { index: usize, kind: RoundSlot },
}

/// Sub-protocols whose PoW lives in the basecase. `GammaCombination` is the
/// Lemma 7.4 γ-RLC slot, present only in ZK mode.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum BasecaseSlot {
    #[error("γ-combination")]
    GammaCombination,
    #[error("sumcheck")]
    Sumcheck,
}

/// Sub-protocols whose PoW lives in a per-round shape.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum RoundSlot {
    #[error("sumcheck")]
    Sumcheck,
    #[error("code-switch")]
    CodeSwitch,
    #[error("mask-proximity")]
    MaskProximity,
}

/// Failure modes for [`super::derive::ProtocolConfig::derive`] and the
/// sub-protocol solvers it calls.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DeriveError {
    /// `compute_t_ood` failed to reach a fixed point in `MAX_ITER` iterations.
    /// Indicates a pathological spec/tuning combo; should not happen under
    /// realistic security targets on supported fields.
    #[error("t_ood fixed-point did not converge for round {round_index}")]
    TOodFixedPointDidNotConverge { round_index: usize },

    /// The ZK per-round `t_ood ↔ source.mask_length()` loop failed to reach a
    /// fixed point. Same caveat as `TOodFixedPointDidNotConverge`.
    #[error("ZK per-round fixed-point did not converge for round {round_index}")]
    PerRoundFixedPointDidNotConverge { round_index: usize },

    /// A PoW slot cannot be ground at the chosen analytic floor — the spec is
    /// too tight for any single grind slot to close the gap.
    #[error("{slot} cannot be ground: {source}")]
    PowUngrindable {
        slot: PowSlot,
        #[source]
        source: PowError,
    },

    /// A PoW slot fits the grind cap but exceeds the per-slot budget set by
    /// [`super::spec::SecuritySpec::pow_budget`].
    #[error("{slot} requires {required} bits, exceeds spec.pow_budget = {max}")]
    PowBudgetExceeded {
        slot: PowSlot,
        required: Bits,
        max: Bits,
    },

    /// Computed codeword length exceeds the NTT engine's supported order.
    #[error("codeword length {length} exceeds the NTT engine's supported order")]
    CodewordExceedsNtt { length: usize },
}

/// Lift `Result<T, PowError>` into `Result<T, DeriveError>` by attaching a
/// [`PowSlot`] label. Lets call sites stay single-line — no manual
/// `.map_err(|e| DeriveError::PowUngrindable { slot, source: e })` boilerplate.
pub(crate) trait PowResultExt<T> {
    fn at_slot(self, slot: PowSlot) -> Result<T, DeriveError>;
}

impl<T> PowResultExt<T> for Result<T, PowError> {
    fn at_slot(self, slot: PowSlot) -> Result<T, DeriveError> {
        self.map_err(|source| DeriveError::PowUngrindable { slot, source })
    }
}
