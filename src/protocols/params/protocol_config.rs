//! Output of [`super::derive`]: the assembled per-round and basecase configs.

use ark_ff::Field;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    bits::Bits,
    protocols::{
        basecase::Config as BasecaseConfig,
        code_switch::Config as CodeSwitchConfig,
        irs_commit::Config as IrsConfig,
        mask_proximity::Config as MaskProximityConfig,
        params::{
            bounds::usize_to_f64,
            code_switch as code_switch_params,
            error::{ChainSource, ChainTarget, DeriveError, Pow},
            spec::{ListSize, MaskCodeMessageLen, OodSampleBudget, SecuritySpec, TuningSpec},
            sumcheck as sumcheck_params,
        },
        proof_of_work::Config as PowConfig,
        sumcheck::Config as SumcheckConfig,
    },
};

#[derive(Clone, Debug)]
pub struct ProtocolConfig<M: Embedding> {
    security: SecuritySpec,
    tuning: TuningSpec,
    rounds: Vec<RoundConfig<M>>,
    basecase: BasecaseConfig<M::Target>,
}

impl<M: Embedding> ProtocolConfig<M> {
    pub(crate) const fn new(
        security: SecuritySpec,
        tuning: TuningSpec,
        rounds: Vec<RoundConfig<M>>,
        basecase: BasecaseConfig<M::Target>,
    ) -> Self {
        Self {
            security,
            tuning,
            rounds,
            basecase,
        }
    }

    pub const fn security(&self) -> &SecuritySpec {
        &self.security
    }

    pub const fn tuning(&self) -> &TuningSpec {
        &self.tuning
    }

    pub fn rounds(&self) -> &[RoundConfig<M>] {
        &self.rounds
    }

    pub const fn basecase(&self) -> &BasecaseConfig<M::Target> {
        &self.basecase
    }

    /// `true` if every PoW slot's difficulty fits within `security.pow_budget`.
    pub fn check_pow_bits(&self) -> bool {
        self.validate_pow_budget().is_ok()
    }

    /// Returns `true` if every post-construction invariant holds.
    pub fn check_all_invariants(&self) -> bool {
        self.validate().is_ok()
    }

    /// Run every post-construction invariant check.
    pub fn validate(&self) -> Result<(), DeriveError> {
        self.validate_pow_budget()?;
        self.validate_round_chaining()?;
        Ok(())
    }

    /// PoW slot difficulty ≤ `security.pow_budget` for every slot.
    pub fn validate_pow_budget(&self) -> Result<(), DeriveError> {
        let max = Bits::new(f64::from(self.security.pow_budget.bits()));
        let check = |pow: Pow, cfg: &PowConfig| -> Result<(), DeriveError> {
            let required = cfg.difficulty();
            if required > max {
                Err(DeriveError::PowBudgetExceeded { pow, required, max })
            } else {
                Ok(())
            }
        };
        for r in &self.rounds {
            check(
                Pow::RoundSumcheck {
                    index: r.round_index,
                },
                &r.sumcheck.round_pow,
            )?;
            check(
                Pow::RoundCodeSwitch {
                    index: r.round_index,
                },
                &r.code_switch.pow,
            )?;
            if let Some(mo) = r.mask_oracle() {
                check(
                    Pow::RoundMaskProximity {
                        index: r.round_index,
                    },
                    &mo.mask_proximity.pow,
                )?;
            }
        }
        check(Pow::BasecaseSumcheck, &self.basecase.sumcheck.round_pow)?;
        check(Pow::BasecaseGammaCombination, &self.basecase.pow)?;
        Ok(())
    }

    /// Cross-round shape chaining:
    /// - adjacent rounds: `round[i+1].source.vector_size == round[i].target.vector_size`
    /// - last round → basecase: `basecase.commit.vector_size == last.target.vector_size`
    /// - no rounds: `basecase.commit.vector_size == tuning.vector_size`
    pub fn validate_round_chaining(&self) -> Result<(), DeriveError> {
        for window in self.rounds.windows(2) {
            let prev = &window[0];
            let next = &window[1];
            let expected = prev.code_switch.target.vector_size;
            let found = next.code_switch.source.vector_size;
            if expected != found {
                return Err(DeriveError::RoundChainBroken {
                    from: ChainSource::Round(prev.round_index),
                    to: ChainTarget::NextRound(next.round_index),
                    expected,
                    found,
                });
            }
        }

        let basecase_vector_size = self.basecase.commit.vector_size;
        let expected = self.rounds.last().map_or(self.tuning.vector_size, |last| {
            last.code_switch.target.vector_size
        });
        if expected != basecase_vector_size {
            let from = self
                .rounds
                .last()
                .map_or(ChainSource::Tuning, |r| ChainSource::Round(r.round_index));
            return Err(DeriveError::RoundChainBroken {
                from,
                to: ChainTarget::Basecase,
                expected,
                found: basecase_vector_size,
            });
        }

        Ok(())
    }

    /// HVZK privacy error in bits, summed across ZK rounds:
    /// `−log Σ_r (t_ood_r² + t_ood_r) / (2|F|)` (bounds doc, §5.3 + §5.7).
    pub fn privacy_error_bits(&self) -> Bits {
        let field_bits = <M::Target as FieldWithSize>::field_size_bits();
        let mut total_error = 0.0_f64;
        for r in &self.rounds {
            if let RoundMode::ZeroKnowledge { t_ood, .. } = &r.mode {
                let t = usize_to_f64(t_ood.get());
                let log_err = f64::midpoint(t * t, t).log2() - field_bits;
                total_error += 2_f64.powf(log_err);
            }
        }
        if total_error == 0.0 {
            return Bits::new(f64::from(self.security.target_security_bits));
        }
        Bits::new((-total_error.log2()).max(0.0))
    }
}

impl<M: Embedding> ProtocolConfig<M> {
    /// Analytic soundness bits (excluding PoW).
    pub fn analytic_bits(&self) -> Bits {
        let mut min_bits = f64::from(self.basecase.analytic_bits());
        for round in &self.rounds {
            min_bits = min_bits.min(f64::from(round.analytic_bits()));
        }
        Bits::new(min_bits.max(0.0))
    }
}

#[cfg(test)]
impl<M: Embedding> ProtocolConfig<M> {
    pub(crate) const fn override_basecase_pow_for_test(&mut self, pow: PowConfig) {
        self.basecase.pow = pow;
    }

    pub(crate) fn truncate_rounds_for_test(&mut self, len: usize) {
        self.rounds.truncate(len);
    }

    pub(crate) fn corrupt_round_target_vector_size_for_test(
        &mut self,
        round_idx: usize,
        new_size: usize,
    ) {
        self.rounds[round_idx].code_switch.target.vector_size = new_size;
    }
}

#[derive(Clone, Debug)]
pub struct RoundConfig<M: Embedding> {
    round_index: usize,
    sumcheck: SumcheckConfig<M::Target>,
    code_switch: CodeSwitchConfig<M>,
    mode: RoundMode<M>,
}

impl<M: Embedding> RoundConfig<M> {
    pub(crate) const fn new(
        round_index: usize,
        sumcheck: SumcheckConfig<M::Target>,
        code_switch: CodeSwitchConfig<M>,
        mode: RoundMode<M>,
    ) -> Self {
        Self {
            round_index,
            sumcheck,
            code_switch,
            mode,
        }
    }

    pub const fn round_index(&self) -> usize {
        self.round_index
    }

    pub const fn sumcheck(&self) -> &SumcheckConfig<M::Target> {
        &self.sumcheck
    }

    pub const fn code_switch(&self) -> &CodeSwitchConfig<M> {
        &self.code_switch
    }

    pub const fn mode(&self) -> &RoundMode<M> {
        &self.mode
    }

    /// Borrow the round's mask oracle if this is a ZK round.
    pub fn mask_oracle(&self) -> Option<&MaskOracleConfig<M::Target>> {
        match &self.mode {
            RoundMode::Standard => None,
            RoundMode::ZeroKnowledge { mask_oracle, .. } => Some(mask_oracle.as_ref()),
        }
    }

    /// Slim mask-oracle view derived from `mask_oracle()`.
    pub fn mask_oracle_info(&self) -> Option<MaskOracleInfo> {
        self.mask_oracle().map(MaskOracleConfig::info)
    }
}

/// Standard vs. ZK round.
///
/// The ZK payload is boxed so the enum stays small.
#[derive(Clone, Debug)]
pub enum RoundMode<M: Embedding> {
    Standard,
    ZeroKnowledge {
        /// Lemma 9.9 OOD-sample budget (bounds doc §5.2).
        t_ood: OodSampleBudget,
        /// Per-round mask oracle.
        mask_oracle: Box<MaskOracleConfig<M::Target>>,
    },
}

impl<M: Embedding> RoundMode<M> {
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::ZeroKnowledge { .. })
    }
}

impl<M: Embedding> RoundConfig<M> {
    /// Round-level analytic floor: the smallest of `sumcheck`, `code_switch`,
    /// and (when present) the per-round mask-oracle proximity check.
    pub fn analytic_bits(&self) -> Bits {
        let source = &self.code_switch.source;
        let target = &self.code_switch.target;
        let mask_info = self.mask_oracle_info();

        let sumcheck_term = f64::from(sumcheck_params::analytic_error_bits(source, mask_info));
        let code_switch_term = f64::from(code_switch_params::analytic_error_bits(
            source,
            target,
            self.code_switch.out_domain_samples,
            mask_info,
        ));
        let mask_oracle_term = self
            .mask_oracle()
            .map_or(f64::INFINITY, |mo| f64::from(mo.analytic_bits()));

        Bits::new(
            sumcheck_term
                .min(code_switch_term)
                .min(mask_oracle_term)
                .max(0.0),
        )
    }
}

/// One round's mask oracle: a C_zk codeword + ℓ_zk + mask-proximity check.
#[derive(Clone, Debug)]
pub struct MaskOracleConfig<F: Field> {
    c_zk: IrsConfig<Identity<F>>,
    /// `next_pow2(r + t_ood)` (Theorem 9.6 + Lemma 9.3).
    l_zk: MaskCodeMessageLen,
    mask_proximity: MaskProximityConfig<F>,
}

impl<F: Field> MaskOracleConfig<F> {
    pub(crate) const fn new(
        c_zk: IrsConfig<Identity<F>>,
        l_zk: MaskCodeMessageLen,
        mask_proximity: MaskProximityConfig<F>,
    ) -> Self {
        Self {
            c_zk,
            l_zk,
            mask_proximity,
        }
    }

    pub const fn c_zk(&self) -> &IrsConfig<Identity<F>> {
        &self.c_zk
    }

    pub const fn l_zk(&self) -> MaskCodeMessageLen {
        self.l_zk
    }

    pub const fn mask_proximity(&self) -> &MaskProximityConfig<F> {
        &self.mask_proximity
    }

    pub fn info(&self) -> MaskOracleInfo {
        MaskOracleInfo {
            c_zk_list_size: ListSize::new(self.c_zk.list_size()),
            l_zk: self.l_zk,
        }
    }
}

/// Slim mask-oracle view (C_zk's list size + ℓ_zk).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaskOracleInfo {
    pub c_zk_list_size: ListSize,
    pub l_zk: MaskCodeMessageLen,
}

impl<F: Field> MaskOracleConfig<F> {
    /// Analytic soundness bits (excluding PoW) for this round's mask oracle.
    pub fn analytic_bits(&self) -> Bits {
        self.mask_proximity.analytic_bits()
    }
}
