//! Output shape of the planner.
//!
//! C_zk and ℓ_zk are protocol-global (one shared Merkle tree across all
//! rounds) and live in [`SharedPlan`]; per-round sumcheck + code-switch live
//! in [`RoundPlan`]. Source/target IRS configs are accessed via
//! `round.code_switch` — not duplicated at the round level.

use ark_ff::Field;

use crate::{
    algebra::embedding::{Embedding, Identity},
    bits::Bits,
    protocols::{
        basecase, code_switch, irs_commit, mask_proximity,
        params::{
            basecase as basecase_solver,
            bounds::SoundnessBounded,
            code_switch as code_switch_solver, mask_proximity as mask_proximity_solver,
            spec::{MaskCodeMessageLen, OodSampleBudget, SecuritySpec, TuningSpec},
            sumcheck as sumcheck_solver,
        },
        sumcheck,
    },
};

#[derive(Clone, Debug)]
pub struct ParameterPlan<M: Embedding> {
    pub security: SecuritySpec<M>,
    pub tuning: TuningSpec,
    pub shared: SharedPlan<M::Target>,
    pub rounds: Vec<RoundPlan<M>>,
    pub basecase: basecase::Config<M::Target>,
}

impl<M: Embedding> SoundnessBounded for ParameterPlan<M> {
    fn analytic_bits(&self) -> Bits {
        let mut min_bits = f64::INFINITY;
        for round in &self.rounds {
            min_bits = min_bits.min(f64::from(round.analytic_bits()));
        }
        if let Some(mo) = &self.shared.mask_oracle {
            min_bits = min_bits.min(f64::from(mo.analytic_bits()));
        }
        // Basecase sumcheck per-round bound applies in both modes; the γ-slot
        // only contributes in ZK.
        min_bits = min_bits.min(f64::from(sumcheck_solver::analytic_error_bits(
            &self.basecase.commit,
            None,
        )));
        if matches!(self.basecase.mode, basecase::Mode::ZeroKnowledge) {
            min_bits = min_bits.min(f64::from(basecase_solver::analytic_error_bits(
                &self.basecase.commit,
            )));
        }
        if min_bits.is_infinite() {
            return Bits::new(f64::from(self.security.target_security_bits));
        }
        Bits::new(min_bits.max(0.0))
    }
}

#[derive(Clone, Debug)]
pub struct RoundPlan<M: Embedding> {
    pub round_index: usize,
    pub sumcheck: sumcheck::Config<M::Target>,
    pub code_switch: code_switch::Config<M>,
    pub mode: RoundMode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoundMode {
    Standard,
    ZeroKnowledge {
        /// Bound 2 / Lemma 9.9.
        t_ood: OodSampleBudget,
        /// Cached view of the shared mask oracle (denormalized from
        /// [`MaskOraclePlan`]) so each round is self-contained for soundness.
        mask_oracle: MaskOracleInfo,
    },
}

impl RoundMode {
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::ZeroKnowledge { .. })
    }

    pub const fn mask_oracle(&self) -> Option<MaskOracleInfo> {
        match self {
            Self::Standard => None,
            Self::ZeroKnowledge { mask_oracle, .. } => Some(*mask_oracle),
        }
    }
}

impl<M: Embedding> SoundnessBounded for RoundPlan<M> {
    fn analytic_bits(&self) -> Bits {
        let source = &self.code_switch.source;
        let target = &self.code_switch.target;
        let mask_oracle = self.mode.mask_oracle();

        let sumcheck_term = sumcheck_solver::analytic_error_bits(source, mask_oracle);
        let code_switch_term = code_switch_solver::analytic_error_bits(
            source,
            target,
            self.code_switch.out_domain_samples,
            mask_oracle,
        );

        if f64::from(code_switch_term) < f64::from(sumcheck_term) {
            code_switch_term
        } else {
            sumcheck_term
        }
    }
}

#[derive(Clone, Debug)]
pub struct SharedPlan<F: Field> {
    /// `Some` iff `Mode::ZeroKnowledge`.
    pub mask_oracle: Option<MaskOraclePlan<F>>,
}

/// One C_zk codeword + one shared Merkle tree + one mask-proximity check,
/// covering every mask committed across all rounds.
#[derive(Clone, Debug)]
pub struct MaskOraclePlan<F: Field> {
    /// `num_vectors = 2 * total_masks` (Construction 7.2: originals + fresh).
    pub c_zk: irs_commit::Config<Identity<F>>,
    /// Dominates every round's `r + t_ood` (Lemma 9.3).
    pub l_zk: MaskCodeMessageLen,
    pub mask_proximity: mask_proximity::Config<F>,
}

/// Slim mask-oracle view (C_zk's list size + ℓ_zk) for builders that don't
/// need the full config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskOracleInfo {
    pub c_zk_list_size: f64,
    pub l_zk: MaskCodeMessageLen,
}

impl<F: Field> MaskOraclePlan<F> {
    pub fn info(&self) -> MaskOracleInfo {
        MaskOracleInfo {
            c_zk_list_size: self.c_zk.list_size(),
            l_zk: self.l_zk,
        }
    }
}

impl<F: Field> SoundnessBounded for MaskOraclePlan<F> {
    fn analytic_bits(&self) -> Bits {
        mask_proximity_solver::analytic_error_bits(
            &self.mask_proximity.c_zk_commit,
            self.mask_proximity.num_masks,
        )
    }
}
