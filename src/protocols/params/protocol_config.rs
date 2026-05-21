//! Output of [`super::derive`]: the assembled per-round and basecase configs.
//!
//! Each ZK round owns its mask oracle: a per-round C_zk codeword (sized for
//! `2·(k+1)` columns — `k` sumcheck masks + 1 code-switch `(r ‖ s)` mask, all
//! doubled by Construction 7.2's originals + fresh pairs) plus a per-round
//! mask-proximity check. Standard rounds carry no mask oracle.

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
            bounds::{usize_to_f64, SoundnessBounded},
            code_switch as code_switch_solver,
            error::{BasecaseSlot, DeriveError, PowSlot, RoundSlot},
            spec::{ListSize, MaskCodeMessageLen, OodSampleBudget, SecuritySpec, TuningSpec},
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
    /// `security.max_pow_bits`. Boolean predicate kept for callers that want
    /// to re-check after manual inspection; [`Self::validate_pow_budget`] is
    /// the typed version used internally by [`super::derive::ProtocolConfig::derive`].
    pub fn check_pow_bits(&self) -> bool {
        self.validate_pow_budget().is_ok()
    }

    /// Same check as [`Self::check_pow_bits`] but returns the specific slot
    /// and required-vs-max difficulties on failure. Auto-invoked by
    /// `derive()`; callers don't normally need to call this directly.
    pub fn validate_pow_budget(&self) -> Result<(), DeriveError> {
        let max = Bits::new(f64::from(self.security.max_pow_bits.unwrap_or(0)));
        let check = |slot: PowSlot, pow: &PowConfig| -> Result<(), DeriveError> {
            let required = pow.difficulty();
            if required > max {
                Err(DeriveError::PowBudgetExceeded {
                    slot,
                    required,
                    max,
                })
            } else {
                Ok(())
            }
        };
        for r in &self.rounds {
            check(
                PowSlot::Round {
                    index: r.round_index,
                    kind: RoundSlot::Sumcheck,
                },
                &r.sumcheck.round_pow,
            )?;
            check(
                PowSlot::Round {
                    index: r.round_index,
                    kind: RoundSlot::CodeSwitch,
                },
                &r.code_switch.pow,
            )?;
            if let Some(mo) = &r.mask_oracle {
                check(
                    PowSlot::Round {
                        index: r.round_index,
                        kind: RoundSlot::MaskProximity,
                    },
                    &mo.mask_proximity.pow,
                )?;
            }
        }
        check(
            PowSlot::Basecase(BasecaseSlot::Sumcheck),
            &self.basecase.sumcheck.round_pow,
        )?;
        check(
            PowSlot::Basecase(BasecaseSlot::GammaCombination),
            &self.basecase.pow,
        )?;
        Ok(())
    }

    /// HVZK privacy error in bits, summed across ZK rounds:
    /// `−log Σ_r (t_ood_r² + t_ood_r) / (2|F|)` (bounds doc, §5.3 + §5.7).
    /// Standard-mode plans return `target_security_bits` as a sentinel —
    /// HVZK isn't claimed when there are no ZK rounds.
    pub fn privacy_error_bits(&self) -> Bits {
        let field_bits = <M::Target as FieldWithSize>::field_size_bits();
        let mut total_error = 0.0_f64;
        for r in &self.rounds {
            if let RoundMode::ZeroKnowledge { t_ood, .. } = r.mode {
                let t = usize_to_f64(t_ood.get());
                // ζ_ze ≤ (t_ood² + t_ood) / (2|F|). Compute in log space to
                // stay numerically stable for large field_bits.
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

impl<M: Embedding> SoundnessBounded for ProtocolConfig<M> {
    fn analytic_bits(&self) -> Bits {
        let mut min_bits = f64::from(self.basecase.analytic_bits());
        for round in &self.rounds {
            min_bits = min_bits.min(f64::from(round.analytic_bits()));
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoundMode {
    Standard,
    ZeroKnowledge {
        /// Lemma 9.9 OOD-sample budget (bounds doc §5.2).
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
    /// Round-level analytic floor: the smallest of `sumcheck`, `code_switch`,
    /// and (when present) the per-round mask-oracle proximity check. Folding
    /// the mask-oracle term in here keeps `ProtocolConfig::analytic_bits`
    /// a pure `min` over rounds + basecase.
    fn analytic_bits(&self) -> Bits {
        let source = &self.code_switch.source;
        let target = &self.code_switch.target;
        let mask_oracle = self.mode.mask_oracle();

        let sumcheck_term = f64::from(sumcheck_solver::analytic_error_bits(source, mask_oracle));
        let code_switch_term = f64::from(code_switch_solver::analytic_error_bits(
            source,
            target,
            self.code_switch.out_domain_samples,
            mask_oracle,
        ));
        let mask_oracle_term = self
            .mask_oracle
            .as_ref()
            .map_or(f64::INFINITY, |mo| f64::from(mo.analytic_bits()));

        Bits::new(
            sumcheck_term
                .min(code_switch_term)
                .min(mask_oracle_term)
                .max(0.0),
        )
    }
}

/// One round's mask oracle: a C_zk codeword + ℓ_zk + mask-proximity check
/// covering `k + 1` masks (sumcheck + code-switch) for this round.
#[derive(Clone, Debug)]
pub struct MaskOracleConfig<F: Field> {
    /// `num_vectors = 2 · (k + 1)` (Construction 7.2: originals + fresh).
    pub c_zk: IrsConfig<Identity<F>>,
    /// `next_pow2(r + t_ood)` for this round: Theorem 9.6 witness layout
    /// (`0^{ℓ_zk − r}` padding) + Lemma 9.3 `(ℓ_zk − r, 0)`-privacy precondition.
    pub l_zk: MaskCodeMessageLen,
    pub mask_proximity: MaskProximityConfig<F>,
}

/// Slim mask-oracle view (C_zk's list size + ℓ_zk).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaskOracleInfo {
    pub c_zk_list_size: ListSize,
    pub l_zk: MaskCodeMessageLen,
}

impl<F: Field> MaskOracleConfig<F> {
    pub fn info(&self) -> MaskOracleInfo {
        MaskOracleInfo {
            c_zk_list_size: ListSize::new(self.c_zk.list_size()),
            l_zk: self.l_zk,
        }
    }
}

impl<F: Field> SoundnessBounded for MaskOracleConfig<F> {
    fn analytic_bits(&self) -> Bits {
        self.mask_proximity.analytic_bits()
    }
}
