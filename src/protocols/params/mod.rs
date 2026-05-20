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
pub(crate) mod irs_commit;
pub(crate) mod mask_proximity;
pub mod protocol_config;
pub mod spec;
pub(crate) mod sumcheck;

#[cfg(test)]
pub(crate) mod test_utils;

pub use protocol_config::{
    MaskOracleConfig, MaskOracleInfo, ProtocolConfig, RoundConfig, RoundMode,
};
pub use spec::{
    FoldingFactor, ListSize, LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget, RoundContext,
    SecuritySpec, TuningSpec,
};
