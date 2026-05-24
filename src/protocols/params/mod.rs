//! Parameter selection for HVZK-WHIR.
//!
//! Soundness and ZK bound derivations (referred to in submodule comments as
//! "the bounds doc, §N") live at
//! <https://hackmd.io/@1q1q-TiuQN6fAkxaN41u-Q/ryBoT_UA-e>.
//!
//! `derive` is the public entry point; the sub-protocol solvers (`basecase`,
//! `code_switch`, `irs_commit`, `mask_proximity`, `sumcheck`) are crate-local
//! and reached only via `derive`. Output and spec types are re-exported below.

pub(crate) mod basecase;
pub(crate) mod bounds;
pub(crate) mod code_switch;
pub mod derive;
pub mod error;
pub(crate) mod irs_commit;
pub(crate) mod mask_proximity;
pub mod protocol_config;
pub(crate) mod regime;
pub mod spec;
pub(crate) mod sumcheck;

#[cfg(test)]
pub(crate) mod test_utils;

pub use error::{ChainSource, ChainTarget, DeriveError, Pow};
pub use protocol_config::{
    MaskOracleConfig, MaskOracleInfo, ProtocolConfig, RoundConfig, RoundMode,
};
pub use spec::{
    DecodingRegime, FoldingFactor, ListSize, LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget,
    PowBudget, RoundContext, SecuritySpec, TuningSpec, ZkSpec,
};

/// Solver-input mode for the per-round sumcheck and code-switch builders.
///
/// Both sub-protocols branch on the same Standard vs. ZK distinction with the
/// same `MaskOracleInfo` payload, so a shared vocabulary keeps call sites
/// uniform.
///
/// Distinct from [`Mode`] (the spec-level policy enum, which carries no
/// payload) and from sub-protocol *output* modes
/// (`sumcheck::SumcheckMode`, `code_switch::CodeSwitchMode`) whose payloads
/// describe the configured round rather than its solver input.
#[derive(Clone, Copy)]
pub enum SolveMode {
    Standard,
    ZeroKnowledge { mask_oracle: MaskOracleInfo },
}
