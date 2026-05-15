//! Output of [`super::derive`]: the assembled per-round and basecase configs.
//!
//! Each ZK round owns its mask oracle: a per-round C_zk codeword (sized for
//! `2·(k+1)` columns — `k` sumcheck masks + 1 code-switch `(r ‖ s)` mask, all
//! doubled by Construction 7.2's originals + fresh pairs) plus a per-round
//! mask-proximity check. Standard rounds carry no mask oracle.

use ark_ff::Field;

use crate::{
    algebra::embedding::{Embedding, Identity},
    bits::Bits,
    protocols::{
        basecase::{self, Config as BasecaseConfig},
        code_switch::Config as CodeSwitchConfig,
        irs_commit::Config as IrsConfig,
        mask_proximity::Config as MaskProximityConfig,
        params::{
            basecase as basecase_solver,
            bounds::SoundnessBounded,
            code_switch as code_switch_solver, mask_proximity as mask_proximity_solver,
            spec::{MaskCodeMessageLen, OodSampleBudget, SecuritySpec, TuningSpec},
            sumcheck as sumcheck_solver,
        },
        proof_of_work::Config as PowConfig,
        sumcheck::Config as SumcheckConfig,
    },
};

#[derive(Clone, Debug)]
pub struct ProtocolConfig<M: Embedding> {
    pub security: SecuritySpec,
    pub tuning: TuningSpec,
    pub rounds: Vec<RoundConfig<M>>,
    pub basecase: BasecaseConfig<M::Target>,
}

impl<M: Embedding> ProtocolConfig<M> {
    /// Returns `true` if every PoW slot's difficulty fits within
    /// `security.max_pow_bits`. Cheap pre-flight check that fails before the
    /// 60-bit cap assertion inside `proof_of_work::threshold`.
    pub fn check_pow_bits(&self) -> bool {
        let max = Bits::new(f64::from(self.security.max_pow_bits.unwrap_or(0)));
        let within = |pow: &PowConfig| pow.difficulty() <= max;
        if !self.rounds.iter().all(|r| {
            within(&r.sumcheck.round_pow)
                && within(&r.code_switch.pow)
                && r.mask_oracle
                    .as_ref()
                    .is_none_or(|mo| within(&mo.mask_proximity.pow))
        }) {
            return false;
        }
        within(&self.basecase.sumcheck.round_pow) && within(&self.basecase.pow)
    }
}

impl<M: Embedding> SoundnessBounded for ProtocolConfig<M> {
    fn analytic_bits(&self) -> Bits {
        let mut min_bits = f64::INFINITY;
        for round in &self.rounds {
            min_bits = min_bits.min(f64::from(round.analytic_bits()));
            if let Some(mo) = &round.mask_oracle {
                min_bits = min_bits.min(f64::from(mo.analytic_bits()));
            }
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
pub struct RoundConfig<M: Embedding> {
    pub round_index: usize,
    pub sumcheck: SumcheckConfig<M::Target>,
    pub code_switch: CodeSwitchConfig<M>,
    pub mode: RoundMode,
    /// `Some` iff this is a ZK round. Sized for this round's `k + 1` masks
    /// (k sumcheck + 1 code-switch).
    pub mask_oracle: Option<MaskOracleConfig<M::Target>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoundMode {
    Standard,
    ZeroKnowledge {
        /// Bound 2 / Lemma 9.9.
        t_ood: OodSampleBudget,
        /// Slim view of this round's [`MaskOracleConfig`] (C_zk's list size +
        /// ℓ_zk) — denormalized so soundness routines can read it without
        /// chasing through `mask_oracle`.
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

impl<M: Embedding> SoundnessBounded for RoundConfig<M> {
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

/// One round's mask oracle: a C_zk codeword + ℓ_zk + mask-proximity check
/// covering `k + 1` masks (sumcheck + code-switch) for this round.
#[derive(Clone, Debug)]
pub struct MaskOracleConfig<F: Field> {
    /// `num_vectors = 2 · (k + 1)` (Construction 7.2: originals + fresh).
    pub c_zk: IrsConfig<Identity<F>>,
    /// `next_pow2(r + t_ood)` for this round (Lemma 9.3).
    pub l_zk: MaskCodeMessageLen,
    pub mask_proximity: MaskProximityConfig<F>,
}

/// Slim mask-oracle view (C_zk's list size + ℓ_zk).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskOracleInfo {
    pub c_zk_list_size: f64,
    pub l_zk: MaskCodeMessageLen,
}

impl<F: Field> MaskOracleConfig<F> {
    pub fn info(&self) -> MaskOracleInfo {
        MaskOracleInfo {
            c_zk_list_size: self.c_zk.list_size(),
            l_zk: self.l_zk,
        }
    }
}

impl<F: Field> SoundnessBounded for MaskOracleConfig<F> {
    fn analytic_bits(&self) -> Bits {
        mask_proximity_solver::analytic_error_bits(
            &self.mask_proximity.c_zk_commit,
            self.mask_proximity.num_masks,
        )
    }
}
