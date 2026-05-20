//! Parameter selection for HVZK-WHIR.
//!
//! Soundness and ZK bound derivations (referred to in submodule comments as
//! "the bounds doc, §N") live at
//! <https://hackmd.io/@1q1q-TiuQN6fAkxaN41u-Q/ryBoT_UA-e>.

pub mod basecase;
pub(crate) mod bounds;
pub mod code_switch;
pub mod derive;
pub mod irs_commit;
pub mod mask_proximity;
pub mod protocol_config;
pub mod spec;
pub mod sumcheck;

#[cfg(test)]
pub(crate) mod test_utils;
