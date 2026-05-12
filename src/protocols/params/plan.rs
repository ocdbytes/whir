//! Derived parameter plan for the Construction 9.7 ZK protocol.
//!
//! Built by the orchestrator from a [`SecuritySpec`] + [`TuningSpec`]; owns
//! the cross-protocol resolved values (source/target IRS, C_zk, t_ood, ℓ_zk,
//! per-round sub-protocol configs) so downstream code doesn't coordinate them.

use crate::{
    algebra::embedding::{Embedding, Identity},
    protocols::{
        code_switch, irs_commit,
        params::spec::{MaskCodeMessageLen, SecuritySpec, TuningSpec},
        sumcheck,
    },
};

/// Full derived parameter plan for one protocol run.
#[derive(Clone, Debug)]
pub struct ParameterPlan<M: Embedding> {
    pub security: SecuritySpec<M>,
    pub tuning: TuningSpec,
    pub rounds: Vec<RoundParams<M>>,
}

/// Parameters for a single round (sumcheck + code-switch).
#[derive(Clone, Debug)]
pub struct RoundParams<M: Embedding> {
    pub round_index: usize,
    pub source_irs: irs_commit::Config<M>,
    pub target_irs: irs_commit::Config<Identity<M::Target>>,
    pub sumcheck: sumcheck::Config<M::Target>,
    pub code_switch: code_switch::Config<M>,
    pub zk: RoundModeParams<M>,
}

#[derive(Clone, Debug)]
pub enum RoundModeParams<M: Embedding> {
    Standard,
    ZeroKnowledge {
        c_zk: irs_commit::Config<Identity<M::Target>>,
        l_zk: MaskCodeMessageLen,
    },
}

impl<M: Embedding> RoundModeParams<M> {
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::ZeroKnowledge { .. })
    }
}
