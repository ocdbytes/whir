//! Parameter selection for HVZK-WHIR.
//!
//! Soundness and ZK bound derivations (referred to in submodule comments as
//! "the bounds doc, §N") live at
//! <https://hackmd.io/@1q1q-TiuQN6fAkxaN41u-Q/ryBoT_UA-e>.

pub(crate) mod basecase;
pub(crate) mod bounds;
pub(crate) mod branch;
pub(crate) mod build_round;
pub(crate) mod code_switch;
pub mod derive;
pub mod error;
pub(crate) mod irs_commit;
pub(crate) mod layout;
pub(crate) mod mask_proximity;
pub mod protocol_config;
pub(crate) mod regime;
pub mod spec;
pub(crate) mod sumcheck;

#[cfg(test)]
pub(crate) mod test_utils;

pub use branch::{Branch, SolveMode};
pub use error::{ChainSource, ChainTarget, DeriveError, Pow};
pub use protocol_config::{
    MaskOracleConfig, MaskOracleInfo, ProtocolConfig, RoundConfig, RoundMode,
};
pub use spec::{
    DecodingRegime, FoldingFactor, ListSize, LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget,
    PowBudget, RoundContext, SecuritySpec, TuningSpec, ZkSpec,
};
