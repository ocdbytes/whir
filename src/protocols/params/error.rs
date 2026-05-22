//! Errors raised by [`super::derive::ProtocolConfig::derive`] and the
//! sub-protocol solvers.
//!
//! Two layers: [`super::super::proof_of_work::PowError`] for grinding-cap
//! failures, [`DeriveError`] for everything `derive()` can surface. The latter
//! wraps the former via [`DeriveError::PowUngrindable::source`] so callers can
//! walk the `std::error::Error::source()` chain.

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::{bits::Bits, protocols::proof_of_work::PowError};

/// Identifies a single PoW grind in the derived protocol — basecase
/// sub-protocol or a per-round sub-protocol at a specific round index. Used
/// to label grinding-cap and budget failures.
///
/// Flat by design: each variant is one valid (where, sub-protocol) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pow {
    /// Basecase γ-RLC grind (Lemma 7.4) — ZK mode only.
    BasecaseGammaCombination,
    /// Basecase sumcheck grind.
    BasecaseSumcheck,
    /// Per-round sumcheck grind at `index`.
    RoundSumcheck { index: usize },
    /// Per-round code-switch grind at `index`.
    RoundCodeSwitch { index: usize },
    /// Per-round mask-proximity grind at `index` — ZK mode only.
    RoundMaskProximity { index: usize },
}

impl Display for Pow {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::BasecaseGammaCombination => f.write_str("basecase γ-combination"),
            Self::BasecaseSumcheck => f.write_str("basecase sumcheck"),
            Self::RoundSumcheck { index } => write!(f, "round {index} sumcheck"),
            Self::RoundCodeSwitch { index } => write!(f, "round {index} code-switch"),
            Self::RoundMaskProximity { index } => write!(f, "round {index} mask-proximity"),
        }
    }
}

/// Origin side of a [`DeriveError::RoundChainBroken`]: either a numbered round
/// or the pre-round `tuning` shape (for plans with no rounds at all).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainSource {
    Tuning,
    Round(usize),
}

impl Display for ChainSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tuning => f.write_str("tuning"),
            Self::Round(i) => write!(f, "round {i}"),
        }
    }
}

/// Destination side of a [`DeriveError::RoundChainBroken`]: the next round in
/// sequence, or the basecase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainTarget {
    NextRound(usize),
    Basecase,
}

impl Display for ChainTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NextRound(i) => write!(f, "round {i}"),
            Self::Basecase => f.write_str("basecase"),
        }
    }
}

/// Which fixed-point loop failed to converge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedPointLoop {
    /// `compute_t_ood`'s scalar iteration.
    TOod,
    /// `build_zk_round_data`'s outer `t_ood ↔ source.mask_length()` iteration.
    ZkRound,
}

impl Display for FixedPointLoop {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::TOod => f.write_str("t_ood"),
            Self::ZkRound => f.write_str("ZK per-round"),
        }
    }
}

/// Failure modes for [`super::derive::ProtocolConfig::derive`] and the
/// sub-protocol solvers it calls.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DeriveError {
    /// A fixed-point loop ran out of iterations. Indicates a pathological
    /// spec/tuning combo; should not happen under realistic security targets
    /// on supported fields.
    #[error("{loop_kind} fixed-point did not converge for round {round_index}")]
    FixedPointDidNotConverge {
        round_index: usize,
        loop_kind: FixedPointLoop,
    },

    /// A PoW grind cannot close the analytic-to-target gap — the spec is too
    /// tight for any single grind to reach `target_security_bits`.
    #[error("{pow} cannot be ground: {source}")]
    PowUngrindable {
        pow: Pow,
        #[source]
        source: PowError,
    },

    /// A PoW grind fits the grind cap but exceeds the per-slot budget set by
    /// [`super::spec::SecuritySpec::pow_budget`].
    #[error("{pow} requires {required} bits, exceeds spec.pow_budget = {max}")]
    PowBudgetExceeded { pow: Pow, required: Bits, max: Bits },

    /// Computed codeword length exceeds the NTT engine's supported order.
    #[error("codeword length {length} exceeds the NTT engine's supported order")]
    CodewordExceedsNtt { length: usize },

    /// Cross-round (or round → basecase) shape chain broken: the next
    /// component's source `vector_size` does not match the previous
    /// component's target `vector_size`. Surfaced by
    /// [`super::protocol_config::ProtocolConfig::validate_round_chaining`].
    #[error("chain broken: {from} → {to} expected vector_size {expected}, found {found}")]
    RoundChainBroken {
        from: ChainSource,
        to: ChainTarget,
        expected: usize,
        found: usize,
    },
}

/// Lift `Result<T, PowError>` into `Result<T, DeriveError>` by attaching a
/// [`Pow`] label. Lets call sites stay single-line — no manual
/// `.map_err(|e| DeriveError::PowUngrindable { pow, source: e })` boilerplate.
pub(crate) trait PowResultExt<T> {
    fn at(self, pow: Pow) -> Result<T, DeriveError>;
}

impl<T> PowResultExt<T> for Result<T, PowError> {
    fn at(self, pow: Pow) -> Result<T, DeriveError> {
        self.map_err(|source| DeriveError::PowUngrindable { pow, source })
    }
}
