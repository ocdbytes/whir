//! Derives a [`ProtocolConfig`] from a spec + tuning.
//!
//! All cross-protocol coordination lives here: per-round `t_ood ↔ r` and
//! `ℓ_zk ↔ c_zk` fixed-points, plus the per-round mask oracle (C_zk +
//! mask-proximity sized for `k + 1` masks).

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        irs_commit::{self, Config as IrsConfig},
        params::{
            basecase as basecase_solver, code_switch as code_switch_solver,
            error::{DeriveError, FixedPointLoop, Pow},
            irs_commit as irs_solver, mask_proximity as mask_proximity_solver,
            protocol_config::{MaskOracleConfig, ProtocolConfig, RoundConfig, RoundMode},
            regime::list_size_estimate,
            spec::{
                LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget, RoundContext, SecuritySpec,
                TuningSpec, ZkSpec,
            },
            sumcheck as sumcheck_solver,
        },
    },
};

/// Paranoia guard on `solve_t_ood` — convergence proof on the function itself.
const T_OOD_MAX_ITER: usize = 32;

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// In ZK each round owns its mask oracle; the `ℓ_zk ↔ c_zk ↔ t_ood`
    /// fixed-point runs independently per round.
    ///
    /// Fails with [`DeriveError`] when the spec/tuning combination is
    /// infeasible: a PoW slot exceeds the grind cap, a fixed point diverges,
    /// or any slot exceeds `spec.pow_budget` (post-derivation validation).
    pub fn derive(spec: SecuritySpec, tuning: TuningSpec) -> Result<Self, DeriveError> {
        let RoundLayout {
            shapes,
            basecase_vector_size,
            basecase_log_inv_rate,
        } = round_layout(&tuning);

        let rounds: Vec<RoundConfig<M>> = match spec.mode {
            Mode::Standard => shapes
                .iter()
                .map(|shape| build_round_config::<M>(&spec, shape))
                .collect::<Result<_, _>>()?,
            Mode::ZeroKnowledge => {
                let zk_spec = ZkSpec::try_new(&spec).expect("matched Mode::ZeroKnowledge above");
                let c_zk_log_inv_rate = LogInvRate::new(tuning.starting_log_inv_rate);
                shapes
                    .iter()
                    .map(|shape| build_zk_round_config::<M>(zk_spec, shape, c_zk_log_inv_rate))
                    .collect::<Result<_, _>>()?
            }
        };

        let basecase = basecase_solver::solve(&spec, basecase_vector_size, basecase_log_inv_rate)?;

        let plan = Self::new(spec, tuning, rounds, basecase);
        plan.validate()?;
        Ok(plan)
    }
}

/// `target_folding_factor` is the next round's source folding — uniform
/// `tuning.folding_factor` — so `target_r → source_{r+1}` has matching
/// interleaving.
#[derive(Debug, Clone, Copy)]
struct RoundShape {
    round_index: usize,
    source_vector_size: usize,
    source_log_inv_rate: u32,
    source_folding_factor: u32,
    target_folding_factor: u32,
}

struct RoundLayout {
    shapes: Vec<RoundShape>,
    basecase_vector_size: usize,
    basecase_log_inv_rate: u32,
}

/// Stops when there's no room for both a valid source and a valid target IRS.
fn round_layout(tuning: &TuningSpec) -> RoundLayout {
    assert!(tuning.vector_size.is_power_of_two());
    assert!(tuning.folding_factor.min() >= 1);

    let mut num_vars = tuning.vector_size.trailing_zeros() as usize;
    let mut log_inv_rate = tuning.starting_log_inv_rate;
    let mut shapes = Vec::new();

    loop {
        let round = shapes.len();
        let source_folding = tuning.folding_factor.at_round(round);
        let target_folding = tuning.folding_factor.at_round(round + 1);
        if num_vars < source_folding + target_folding {
            break;
        }
        shapes.push(RoundShape {
            round_index: round,
            source_vector_size: 1usize << num_vars,
            source_log_inv_rate: log_inv_rate,
            source_folding_factor: source_folding as u32,
            target_folding_factor: target_folding as u32,
        });
        num_vars -= source_folding;
        log_inv_rate += (source_folding as u32).saturating_sub(1);
    }

    RoundLayout {
        shapes,
        basecase_vector_size: 1usize << num_vars,
        basecase_log_inv_rate: log_inv_rate,
    }
}

const fn round_context(shape: &RoundShape) -> RoundContext {
    RoundContext {
        vector_size: shape.source_vector_size,
        log_inv_rate: shape.source_log_inv_rate,
        folding_factor: shape.source_folding_factor,
    }
}

fn target_context<M: Embedding>(shape: &RoundShape, source: &IrsConfig<M>) -> RoundContext {
    RoundContext {
        vector_size: source.message_length(),
        log_inv_rate: shape.source_log_inv_rate + shape.source_folding_factor.saturating_sub(1),
        folding_factor: shape.target_folding_factor,
    }
}

/// Per-round ZK builder. C_zk holds `2 · (k + 1)` columns (Construction 7.2
/// originals + fresh): `k` sumcheck masks (Lemma 6.4) + one `(r ‖ s)`
/// code-switch mask (Construction 9.7). `ℓ_zk = next_pow2(r + t_ood)` from
/// Theorem 9.6's witness layout + Lemma 9.3's `r ≥ t` privacy precondition;
/// `t_ood` solves Lemma 9.9 term 1.
fn build_zk_round_config<M: Embedding + Default>(
    zk_spec: ZkSpec<'_>,
    shape: &RoundShape,
    c_zk_log_inv_rate: LogInvRate,
) -> Result<RoundConfig<M>, DeriveError> {
    let spec = zk_spec.as_inner();
    let ctx = round_context(shape);
    let num_masks = sumcheck_solver::masks_required(&ctx) + code_switch_solver::masks_required();
    let c_zk_log_inv_rate_f = f64::from(c_zk_log_inv_rate.get());

    let src_ctx = round_context(shape);
    let target_log_inv_rate =
        f64::from(shape.source_log_inv_rate + shape.source_folding_factor.saturating_sub(1));
    // Target encodes one polynomial of length `source.message_length()` =
    // `source_vector_size / 2^source_folding_factor`.
    let target_log_degree =
        f64::from(shape.source_vector_size.trailing_zeros() - shape.source_folding_factor);
    let target_list_size =
        list_size_estimate(spec.decoding_regime, target_log_degree, target_log_inv_rate);

    let (source, t_ood) = solve_t_ood::<M>(
        spec,
        &src_ctx,
        target_list_size,
        Some(c_zk_log_inv_rate_f),
        shape.round_index,
    )?;
    let target: IrsConfig<Identity<M::Target>> = irs_solver::solve(
        spec,
        &target_context(shape, &source),
        OodSampleBudget::new(t_ood),
    );

    let l_zk = compute_l_zk(&source, t_ood);
    let c_zk: IrsConfig<Identity<M::Target>> = irs_solver::solve_mask_code(
        zk_spec,
        l_zk,
        source.mask_length(),
        c_zk_log_inv_rate,
        2 * num_masks,
    );
    let c_zk_list_size_estimate = list_size_estimate(
        spec.decoding_regime,
        (l_zk.get() as f64).log2(),
        c_zk_log_inv_rate_f,
    );
    debug_assert!(
        (c_zk.list_size() - c_zk_list_size_estimate).abs()
            < 1e-9 * c_zk_list_size_estimate.max(1.0),
        "c_zk.list_size() {} drifted from planner estimate {} — \
         see `list_size_estimate` for the invariant",
        c_zk.list_size(),
        c_zk_list_size_estimate,
    );
    let mask_proximity =
        mask_proximity_solver::solve(spec, c_zk.clone(), num_masks, shape.round_index)?;
    let mask_oracle = MaskOracleConfig::new(c_zk, l_zk, mask_proximity);
    let info = mask_oracle.info();

    let sumcheck = sumcheck_solver::solve_zk(
        spec,
        &ctx,
        &source,
        info,
        Pow::RoundSumcheck {
            index: shape.round_index,
        },
    )?;
    let code_switch =
        code_switch_solver::solve_zk(spec, source, target, t_ood, info, shape.round_index)?;
    Ok(RoundConfig::new(
        shape.round_index,
        sumcheck,
        code_switch,
        RoundMode::ZeroKnowledge {
            t_ood: OodSampleBudget::new(t_ood),
            mask_oracle: Box::new(mask_oracle),
        },
    ))
}

fn build_round_config<M: Embedding + Default>(
    spec: &SecuritySpec,
    shape: &RoundShape,
) -> Result<RoundConfig<M>, DeriveError> {
    let src_ctx = round_context(shape);
    let target_log_inv_rate =
        f64::from(shape.source_log_inv_rate + shape.source_folding_factor.saturating_sub(1));
    let target_log_degree =
        f64::from(shape.source_vector_size.trailing_zeros() - shape.source_folding_factor);
    let target_list_size =
        list_size_estimate(spec.decoding_regime, target_log_degree, target_log_inv_rate);

    let (source, t_ood) =
        solve_t_ood::<M>(spec, &src_ctx, target_list_size, None, shape.round_index)?;
    let target: IrsConfig<Identity<M::Target>> =
        irs_solver::solve(spec, &target_context(shape, &source), OodSampleBudget::ZERO);

    let sumcheck = sumcheck_solver::solve_standard(
        spec,
        &src_ctx,
        &source,
        Pow::RoundSumcheck {
            index: shape.round_index,
        },
    )?;
    let code_switch =
        code_switch_solver::solve_standard(spec, source, target, t_ood, shape.round_index)?;
    Ok(RoundConfig::new(
        shape.round_index,
        sumcheck,
        code_switch,
        RoundMode::Standard,
    ))
}

/// `ℓ_zk = next_pow2(r + t_ood)`: Theorem 9.6 witness layout `0^{ℓ_zk − r}`
/// combined with Lemma 9.3's `r ≥ t` privacy precondition.
pub(super) const fn compute_l_zk<M: Embedding>(
    source: &IrsConfig<M>,
    t_ood: usize,
) -> MaskCodeMessageLen {
    MaskCodeMessageLen::new((source.mask_length() + t_ood).next_power_of_two())
}

/// One application of the Lemma 9.9 OOD step. ZK: `degree = ℓ + ℓ_zk(t_ood)`
/// with `ℓ_zk = next_pow2(source.mask_length() + t_ood)`. Standard:
/// `degree = ℓ`.
///
/// Floored at `1`: Lemma 9.9's OOD term vanishes under `Unique` decoding
/// (`|Λ| = 1`), but Construction 9.7's code-switch still needs at least one
/// OOD point to bind the witness polynomial.
pub(super) fn compute_t_ood<M: Embedding>(
    spec: &SecuritySpec,
    source: &IrsConfig<M>,
    target_list_size: f64,
    c_zk_list_size: Option<f64>,
    t_ood: usize,
) -> usize {
    let security_target = f64::from(spec.protocol_security_target_bits());
    let field_bits = M::Target::field_size_bits();
    let combined_list_size = target_list_size * c_zk_list_size.unwrap_or(1.0);
    let message_length = source.message_length();

    let degree = if c_zk_list_size.is_some() {
        let l_zk = (source.mask_length() + t_ood).next_power_of_two();
        message_length + l_zk
    } else {
        message_length
    };

    let soundness_t_ood = irs_commit::num_ood_samples(
        spec.decoding_regime,
        security_target,
        field_bits,
        combined_list_size,
        degree,
    );
    soundness_t_ood.max(1)
}

/// Solves the per-round `t_ood` fixed-point and the source IRS together.
///
/// Convergence: `Φ(t) = num_ood_samples(ℓ + next_pow2(in_domain + 2·t))` is
/// monotone non-decreasing on ℕ (`in_domain` depends only on the requested
/// rate; `next_pow2` and `num_ood_samples` are monotone; under `Capacity` the
/// `c_zk_list_size(t)` factor is monotone too) and bounded above, so Kleene
/// iteration from `t = 0` converges to the least fixed point in finitely many
/// steps. Standard mode (`c_zk_log_inv_rate = None`) has `Φ` constant in
/// `t`, so one application suffices.
pub(super) fn solve_t_ood<M: Embedding + Default>(
    spec: &SecuritySpec,
    src_ctx: &RoundContext,
    target_list_size: f64,
    c_zk_log_inv_rate: Option<f64>,
    round_index: usize,
) -> Result<(IrsConfig<M>, usize), DeriveError> {
    let mut source: IrsConfig<M> = irs_solver::solve(spec, src_ctx, OodSampleBudget::ZERO);

    let Some(c_zk_log_inv_rate) = c_zk_log_inv_rate else {
        let t_ood = compute_t_ood(spec, &source, target_list_size, None, 0);
        return Ok((source, t_ood));
    };

    let mut t_ood = 0;
    for _ in 0..T_OOD_MAX_ITER {
        // Under `Capacity`, c_zk's list size depends on its message length
        // ℓ_zk(t), so recompute per iteration. Under `Johnson`/`Unique` the
        // result is t-independent — the recomputation is a no-op.
        let l_zk = (source.mask_length() + t_ood).next_power_of_two();
        let c_zk_list_size = list_size_estimate(
            spec.decoding_regime,
            (l_zk as f64).log2(),
            c_zk_log_inv_rate,
        );
        let new_t_ood = compute_t_ood(spec, &source, target_list_size, Some(c_zk_list_size), t_ood);
        if new_t_ood == t_ood {
            return Ok((source, t_ood));
        }
        t_ood = new_t_ood;
        source = irs_solver::solve(spec, src_ctx, OodSampleBudget::new(t_ood));
    }
    Err(DeriveError::FixedPointDidNotConverge {
        round_index,
        loop_kind: FixedPointLoop::TOod,
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{
        hash,
        protocols::params::{
            spec::{DecodingRegime, FoldingFactor, PowBudget},
            test_utils::{assert_close, assert_pow_closes_gap, TestEmbedding},
        },
    };

    /// Varied tuning space for proptests. Exercises both `FoldingFactor`
    /// variants. Bounds keep PoW under the 60-bit cap and the IRS solver
    /// inside Field64's reachable range.
    fn arb_tuning() -> impl Strategy<Value = TuningSpec> {
        let folding = prop_oneof![
            (1usize..=3).prop_map(FoldingFactor::Constant),
            (1usize..=3, 1usize..=3).prop_map(|(initial, rest)| {
                FoldingFactor::ConstantFromSecondRound { initial, rest }
            }),
        ];
        (4u32..=8, 1u32..=3, folding).prop_map(|(log_size, log_inv_rate, folding_factor)| {
            TuningSpec {
                vector_size: 1usize << log_size,
                starting_log_inv_rate: log_inv_rate,
                folding_factor,
            }
        })
    }

    /// `tuning_with` uses `FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR)` so
    /// each round folds by 2. With `target_folding == source_folding == 2`,
    /// `round_layout` keeps a round only while `num_vars ≥ 4`.
    const FIXTURE_FOLDING_FACTOR: usize = 2;
    const FIXTURE_LOG_INV_RATE: u32 = 1;

    /// `log_vector_size` chosen to be below `2 · FIXTURE_FOLDING_FACTOR`, so
    /// `round_layout` exits before adding any round → basecase-only plan.
    const LOG_VECTOR_SIZE_NO_ROUNDS: u32 = 3;
    /// Large enough to produce multiple rounds under
    /// `FIXTURE_FOLDING_FACTOR`-uniform folding; used by every multi-round test.
    const LOG_VECTOR_SIZE_MULTI_ROUND: u32 = 8;

    /// Folding pair used by tests that need round-to-round folding variation
    /// (rate stepping, target→source chaining). The two values must differ
    /// from each other so the variation across rounds is observable.
    const VARIED_INITIAL_FOLDING: usize = 3;
    const VARIED_STEADY_FOLDING: usize = 2;

    fn tuning_with(vector_size: usize) -> TuningSpec {
        TuningSpec {
            vector_size,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
        }
    }

    /// Planner-level tests build full `ProtocolConfig`s, so we use a lower target
    /// than `test_utils::FIXTURE_TARGET_BITS` (= 80). Keeps PoW below the 60-bit
    /// cap when every sub-protocol grinds individually. 40 leaves
    /// `target − analytic_error ≤ 60` on `Field64`.
    const PLAN_FIXTURE_TARGET_BITS: u32 = 40;

    fn test_spec(mode: Mode) -> SecuritySpec {
        SecuritySpec {
            mode,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            // Allow up to the grind cap; derive() auto-validates the budget
            // and would reject configs that need any PoW under `Forbidden`.
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        }
    }

    /// `> 1` so the first round's rate is distinct from the boundary.
    const RATE_STEPPING_STARTING_LOG_INV_RATE: u32 = 2;
    /// Pairwise `windows(2)` chaining check needs ≥ 2 rounds.
    const MIN_ROUNDS_FOR_CHAINING_TEST: usize = 2;

    /// Each round's source rate steps up by `source_folding - 1`. The basecase
    /// inherits the rate after the final round. Uses varied folding so the
    /// per-round step is non-uniform (initial step = 2, steady step = 1).
    #[test]
    fn round_layout_rate_steps_up_by_folding_minus_one() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: RATE_STEPPING_STARTING_LOG_INV_RATE,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: VARIED_INITIAL_FOLDING,
                rest: VARIED_STEADY_FOLDING,
            },
        };
        let layout = round_layout(&tuning);

        let mut expected_log_inv_rate = RATE_STEPPING_STARTING_LOG_INV_RATE;
        for shape in &layout.shapes {
            assert_eq!(shape.source_log_inv_rate, expected_log_inv_rate);
            expected_log_inv_rate += shape.source_folding_factor.saturating_sub(1);
        }
        assert_eq!(layout.basecase_log_inv_rate, expected_log_inv_rate);
    }

    /// Cross-round chaining: round `i`'s target folding factor must match
    /// round `i+1`'s source folding factor (the doc-comment on `RoundShape`
    /// codifies this). Varied folding makes the check non-vacuous.
    #[test]
    fn round_layout_chains_target_to_next_source_folding() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: VARIED_INITIAL_FOLDING,
                rest: VARIED_STEADY_FOLDING,
            },
        };
        let layout = round_layout(&tuning);
        assert!(
            layout.shapes.len() >= MIN_ROUNDS_FOR_CHAINING_TEST,
            "need ≥ {MIN_ROUNDS_FOR_CHAINING_TEST} rounds to test chaining",
        );
        for window in layout.shapes.windows(2) {
            assert_eq!(
                window[0].target_folding_factor,
                window[1].source_folding_factor
            );
        }
    }

    /// Basecase consumes whatever `num_vars` the round loop left behind:
    /// `basecase_vector_size = 2^(initial_num_vars - sum(source_folding_factor))`.
    #[test]
    fn round_layout_basecase_size_consumes_remaining_num_vars() {
        let tuning = tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND);
        let layout = round_layout(&tuning);
        let consumed: u32 = layout.shapes.iter().map(|s| s.source_folding_factor).sum();
        let initial_num_vars = tuning.vector_size.trailing_zeros();
        let remaining = initial_num_vars - consumed;
        assert_eq!(layout.basecase_vector_size, 1usize << remaining);
    }

    /// Loop exits when `num_vars < source_folding + target_folding`. Below the
    /// `2 · FIXTURE_FOLDING_FACTOR` threshold, no round is admitted and the
    /// basecase carries the whole vector at the starting rate.
    #[test]
    fn round_layout_stops_when_no_room_for_source_plus_target() {
        let vector_size = 1usize << LOG_VECTOR_SIZE_NO_ROUNDS;
        let tuning = tuning_with(vector_size);
        let layout = round_layout(&tuning);
        assert!(layout.shapes.is_empty());
        assert_eq!(layout.basecase_vector_size, vector_size);
        assert_eq!(layout.basecase_log_inv_rate, FIXTURE_LOG_INV_RATE);
    }

    #[test]
    fn derive_standard_with_no_rounds_uses_basecase_only() {
        let spec = test_spec(Mode::Standard);
        let vector_size = 1usize << LOG_VECTOR_SIZE_NO_ROUNDS;
        let plan = ProtocolConfig::<TestEmbedding>::derive(spec, tuning_with(vector_size)).unwrap();
        assert!(plan.rounds().is_empty());
        assert_eq!(plan.basecase().commit.vector_size, vector_size);
    }

    /// ZK with zero WHIR rounds = ZK basecase only. Per-round mask oracles are
    /// absent (there are no rounds); the basecase γ-slot PoW carries soundness.
    #[test]
    fn derive_zk_with_no_rounds_uses_zk_basecase_only() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_NO_ROUNDS),
        )
        .unwrap();
        assert!(plan.rounds().is_empty());
        assert!(matches!(
            plan.basecase().mode,
            crate::protocols::basecase::BasecaseMode::ZeroKnowledge
        ));
    }

    /// Lemma 9.9 fixed-point: every ZK round needs at least one OOD challenge.
    #[test]
    fn compute_t_ood_nonzero_in_zk() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        for r in plan.rounds() {
            let RoundMode::ZeroKnowledge { t_ood, .. } = r.mode() else {
                panic!("expected ZK round")
            };
            assert!(t_ood.get() >= 1);
        }
    }

    #[test]
    fn analytic_bits_finite_and_positive_standard() {
        let spec = test_spec(Mode::Standard);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let bits: f64 = plan.analytic_bits().into();
        assert!(bits.is_finite() && bits > 0.0, "bits = {bits}");
        let min_round = plan
            .rounds()
            .iter()
            .map(|r| f64::from(r.analytic_bits()))
            .fold(f64::INFINITY, f64::min);
        let expected = min_round.min(f64::from(plan.basecase().analytic_bits()));
        assert_close(bits, expected);
    }

    #[test]
    fn analytic_bits_includes_mask_oracle_in_zk() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let plan_bits: f64 = plan.analytic_bits().into();
        let mo_floor = plan
            .rounds()
            .iter()
            .filter_map(|r| r.mask_oracle().map(|mo| f64::from(mo.analytic_bits())))
            .fold(f64::INFINITY, f64::min);
        assert!(
            mo_floor.is_finite(),
            "ZK plan must contribute mask-oracle bits"
        );
        let min_round = plan
            .rounds()
            .iter()
            .map(|r| f64::from(r.analytic_bits()))
            .fold(f64::INFINITY, f64::min);
        let expected = mo_floor
            .min(min_round)
            .min(f64::from(plan.basecase().analytic_bits()));
        assert_close(plan_bits, expected);
    }

    #[test]
    fn derive_plans_basecase() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(matches!(
            plan.basecase().mode,
            crate::protocols::basecase::BasecaseMode::ZeroKnowledge
        ));
        assert_eq!(plan.basecase().commit.interleaving_depth, 1);
        // Sumcheck folds basecase to size 1.
        assert_eq!(plan.basecase().sumcheck.final_size(), 1);
    }

    /// Matches `proof_of_work::threshold`'s 60-bit cap.
    const LOOSE_POW_BUDGET_BITS: u32 = 60;
    /// Sits between a moderate budget (30) and the grind cap (60) — used by
    /// `check_pow_bits_detects_over_budget_slot` to inject a slot that fits
    /// the cap but exceeds the test's `pow_budget`.
    const OVER_BUDGET_INJECTED_BITS: f64 = 50.0;

    /// Bounds doc §5.3 + §5.7: HVZK privacy error in bits matches the closed
    /// form `−log Σ_r (t_ood_r² + t_ood_r) / (2|F|)` over ZK rounds.
    #[test]
    fn privacy_error_bits_matches_bound_3_sum() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let field_bits = <crate::algebra::fields::Field64 as FieldWithSize>::field_size_bits();
        let mut expected_total = 0.0_f64;
        for r in plan.rounds() {
            let RoundMode::ZeroKnowledge { t_ood, .. } = r.mode() else {
                panic!("expected ZK round");
            };
            let t = t_ood.get() as f64;
            expected_total += 2_f64.powf(f64::midpoint(t * t, t).log2() - field_bits);
        }
        let expected_bits = -expected_total.log2();
        let got = f64::from(plan.privacy_error_bits());
        assert_close(got, expected_bits);
    }

    /// Standard-mode plans have no HVZK claim — `privacy_error_bits` returns
    /// the spec's `target_security_bits` as a sentinel.
    #[test]
    fn privacy_error_bits_standard_returns_target_sentinel() {
        let spec = test_spec(Mode::Standard);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert_close(
            f64::from(plan.privacy_error_bits()),
            f64::from(PLAN_FIXTURE_TARGET_BITS),
        );
    }

    /// Derived plans must satisfy their own `pow_budget`.
    #[test]
    fn check_pow_bits_passes_on_derived_plan() {
        let spec = SecuritySpec {
            mode: Mode::ZeroKnowledge,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(plan.check_pow_bits());
    }

    /// Hand-injected over-budget PoW slot fails `check_pow_bits()`.
    ///
    /// Derive with a moderately tight budget (passes auto-validation because
    /// the natural slot pow stays well below it), then mutate the basecase
    /// pow to a value above that budget but still within the grind cap, and
    /// verify the boolean check trips.
    #[test]
    fn check_pow_bits_detects_over_budget_slot() {
        use crate::{bits::Bits, protocols::proof_of_work::Config as PowConfig};
        const MODERATE_POW_BUDGET_BITS: u32 = 30;
        let spec = SecuritySpec {
            mode: Mode::ZeroKnowledge,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(MODERATE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        };
        let mut plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        plan.override_basecase_pow_for_test(PowConfig::from_difficulty(Bits::new(
            OVER_BUDGET_INJECTED_BITS,
        )));
        assert!(!plan.check_pow_bits());
    }

    /// `validate_round_chaining` trips when the basecase no longer chains
    /// to the (new) last round after the tail is dropped. Multi-round plan
    /// is required so dropping the last leaves at least one round behind.
    #[test]
    fn validate_round_chaining_detects_basecase_mismatch() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let mut plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let n = plan.rounds().len();
        assert!(n >= 2, "need ≥ 2 rounds to break the chain by truncation");
        assert!(plan.check_all_invariants(), "fresh plan must validate");

        plan.truncate_rounds_for_test(n - 1);
        let err = plan
            .validate_round_chaining()
            .expect_err("truncated tail breaks basecase chaining");
        assert!(
            matches!(
                err,
                DeriveError::RoundChainBroken {
                    to: crate::protocols::params::error::ChainTarget::Basecase,
                    ..
                }
            ),
            "got {err:?}",
        );
        assert!(!plan.check_all_invariants());
    }

    /// `derive()` reports `PowUngrindable` when the spec demands a per-slot
    /// difficulty above the grind cap. `target_security_bits = 200` against
    /// `analytic ≈ 64` on `Field64` gives `required ≈ 136` ≫ 60.
    #[test]
    fn derive_reports_pow_ungrindable() {
        const UNREACHABLE_TARGET_BITS: u32 = 200;
        let spec = SecuritySpec {
            mode: Mode::Standard,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: UNREACHABLE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        };
        let err = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .expect_err("target above grind cap must fail");
        assert!(
            matches!(err, DeriveError::PowUngrindable { .. }),
            "got {err:?}",
        );
    }

    /// `derive()` reports `PowBudgetExceeded` when a slot's required PoW
    /// fits the grind cap but exceeds `spec.pow_budget`. `target = 40`
    /// with `pow_budget = PerSlot { bits: 5 }` forces this on `Field64`.
    #[test]
    fn derive_reports_pow_budget_exceeded() {
        const TIGHT_MAX_POW: u32 = 5;
        let spec = SecuritySpec {
            mode: Mode::ZeroKnowledge,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(TIGHT_MAX_POW),
            hash_id: hash::BLAKE3,
        };
        let err = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .expect_err("tight pow_budget must trip auto-validation");
        assert!(
            matches!(err, DeriveError::PowBudgetExceeded { .. }),
            "got {err:?}",
        );
    }

    /// Unique decoding threads through to the basecase IRS in Standard mode.
    /// Uses a basecase-only tuning so the regime is unambiguous (no rate
    /// stepping across rounds).
    #[test]
    fn derive_threads_unique_decoding_standard() {
        let spec = SecuritySpec {
            mode: Mode::Standard,
            decoding_regime: DecodingRegime::Unique,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_NO_ROUNDS),
        )
        .unwrap();
        assert!(plan.rounds().is_empty());
        assert!(plan.basecase().commit.unique_decoding());
    }

    /// Same threading check under ZK mode (basecase-only fixture).
    #[test]
    fn derive_threads_unique_decoding_zk() {
        let spec = SecuritySpec {
            mode: Mode::ZeroKnowledge,
            decoding_regime: DecodingRegime::Unique,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_NO_ROUNDS),
        )
        .unwrap();
        assert!(plan.rounds().is_empty());
        assert!(plan.basecase().commit.unique_decoding());
    }

    /// Multi-round derivation under Unique: every round's IRS carries the
    /// Unique regime and every code-switch slot satisfies the Construction
    /// 9.7 `t_ood ≥ 1` floor.
    #[test]
    fn derive_multi_round_unique_decoding_succeeds() {
        let spec = SecuritySpec {
            mode: Mode::Standard,
            decoding_regime: DecodingRegime::Unique,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(!plan.rounds().is_empty(), "expected multi-round plan");
        for r in plan.rounds() {
            let cs = r.code_switch();
            assert!(cs.source.unique_decoding());
            assert!(cs.target.unique_decoding());
            assert!(cs.out_domain_samples >= 1, "Construction 9.7 floor");
        }
        assert!(plan.basecase().commit.unique_decoding());
    }

    /// ZK + Unique multi-round: per-round mask oracle still assembled, C_zk
    /// built under Unique, code-switch carries `t_ood ≥ 1` per floor.
    #[test]
    fn derive_multi_round_unique_decoding_zk_succeeds() {
        let spec = SecuritySpec {
            mode: Mode::ZeroKnowledge,
            decoding_regime: DecodingRegime::Unique,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(!plan.rounds().is_empty(), "expected multi-round plan");
        for r in plan.rounds() {
            let mo = r.mask_oracle().expect("ZK round must own a mask oracle");
            assert!(mo.c_zk().unique_decoding());
            assert!(r.code_switch().source.unique_decoding());
            assert!(r.code_switch().out_domain_samples >= 1);
        }
        assert!(plan.basecase().commit.unique_decoding());
    }

    /// Multi-round Capacity (Standard): IRS configs carry the Capacity regime
    /// and the `c_zk_list_size(t)` fixed-point resolves inside `solve_t_ood`.
    #[test]
    fn derive_multi_round_capacity_decoding_succeeds() {
        let spec = SecuritySpec {
            mode: Mode::Standard,
            decoding_regime: DecodingRegime::Capacity,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(!plan.rounds().is_empty(), "expected multi-round plan");
        for r in plan.rounds() {
            assert!(r.code_switch().out_domain_samples >= 1);
        }
    }

    /// ZK + Capacity multi-round: exercises the degree-dependent c_zk list
    /// size inside the t_ood fixed-point.
    #[test]
    fn derive_multi_round_capacity_decoding_zk_succeeds() {
        let spec = SecuritySpec {
            mode: Mode::ZeroKnowledge,
            decoding_regime: DecodingRegime::Capacity,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(!plan.rounds().is_empty(), "expected multi-round plan");
        for r in plan.rounds() {
            r.mask_oracle().expect("ZK round must own a mask oracle");
            assert!(r.code_switch().out_domain_samples >= 1);
        }
    }

    /// `analytic_error + pow ≥ target` for every PoW slot in the plan.
    fn assert_plan_meets_target_per_slot<M: Embedding>(
        spec: &SecuritySpec,
        plan: &ProtocolConfig<M>,
    ) {
        for r in plan.rounds() {
            let mask_info = r.mask_oracle_info();
            let cs = r.code_switch();
            assert_pow_closes_gap(
                spec,
                sumcheck_solver::analytic_error_bits(&cs.source, mask_info),
                &r.sumcheck().round_pow,
            );
            assert_pow_closes_gap(
                spec,
                code_switch_solver::analytic_error_bits(
                    &cs.source,
                    &cs.target,
                    cs.out_domain_samples,
                    mask_info,
                ),
                &cs.pow,
            );
            if let Some(mo) = r.mask_oracle() {
                let mp = mo.mask_proximity();
                assert_pow_closes_gap(
                    spec,
                    mask_proximity_solver::analytic_error_bits(&mp.c_zk_commit, mp.num_masks),
                    &mp.pow,
                );
            }
        }
        assert_pow_closes_gap(
            spec,
            sumcheck_solver::analytic_error_bits(&plan.basecase().commit, None),
            &plan.basecase().sumcheck.round_pow,
        );
        // γ-slot is ZK-only.
        if matches!(
            plan.basecase().mode,
            crate::protocols::basecase::BasecaseMode::ZeroKnowledge
        ) {
            assert_pow_closes_gap(
                spec,
                basecase_solver::analytic_error_bits(&plan.basecase().commit),
                &plan.basecase().pow,
            );
        }
    }

    proptest! {
        /// End-to-end soundness (Standard): every PoW slot in the derived plan
        /// closes the gap `analytic + pow ≥ target` against the spec target.
        #[test]
        fn derived_plan_meets_target_per_slot_standard(tuning in arb_tuning()) {
            let spec = test_spec(Mode::Standard);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec.clone(), tuning).unwrap();
            assert_plan_meets_target_per_slot(&spec, &plan);
        }

        /// End-to-end soundness (ZK): same as above, plus the per-round
        /// mask-proximity slot and the basecase γ-slot.
        #[test]
        fn derived_plan_meets_target_per_slot_zk(tuning in arb_tuning()) {
            let log_threshold =
                tuning.folding_factor.at_round(0) + tuning.folding_factor.at_round(1);
            prop_assume!(tuning.vector_size.trailing_zeros() as usize >= log_threshold);
            let spec = test_spec(Mode::ZeroKnowledge);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec.clone(), tuning).unwrap();
            assert_plan_meets_target_per_slot(&spec, &plan);
        }

        /// Standard mode: derive succeeds for any tuning shape, no per-round
        /// mask oracle, and basecase covers the post-fold tail.
        #[test]
        fn derive_standard_succeeds_over_tunings(tuning in arb_tuning()) {
            let spec = test_spec(Mode::Standard);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec, tuning).unwrap();
            for r in plan.rounds() {
                prop_assert!(matches!(r.mode(), RoundMode::Standard));
                prop_assert!(r.mask_oracle().is_none());
            }
            prop_assert!(matches!(
                plan.basecase().mode,
                crate::protocols::basecase::BasecaseMode::Standard
            ));
            prop_assert_eq!(plan.basecase().commit.interleaving_depth, 1);
        }

        /// ZK mode: each round has its own mask oracle sized for `k + 1`
        /// masks; basecase is ZK-flagged when shapes are non-empty.
        #[test]
        fn derive_zk_succeeds_over_tunings(tuning in arb_tuning()) {
            let log_threshold =
                tuning.folding_factor.at_round(0) + tuning.folding_factor.at_round(1);
            prop_assume!(tuning.vector_size.trailing_zeros() as usize >= log_threshold);

            let spec = test_spec(Mode::ZeroKnowledge);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec, tuning).unwrap();
            for r in plan.rounds() {
                let mask_oracle = r
                    .mask_oracle()
                    .expect("ZK round must have a mask oracle");
                let RoundMode::ZeroKnowledge { t_ood, .. } = r.mode() else {
                    panic!("expected ZK round");
                };
                let cs = r.code_switch();
                let k = cs.source.interleaving_depth.trailing_zeros() as usize;
                let num_masks = k + 1;
                prop_assert_eq!(mask_oracle.c_zk().num_vectors, 2 * num_masks);
                prop_assert_eq!(mask_oracle.mask_proximity().num_masks, num_masks);
                // Theorem 9.6 / Lemma 9.3: ℓ_zk ≥ r + t_ood for this round.
                let source_mask = cs.source.mask_length();
                prop_assert!(mask_oracle.l_zk().get() >= source_mask + t_ood.get());
            }
            prop_assert!(matches!(
                plan.basecase().mode,
                crate::protocols::basecase::BasecaseMode::ZeroKnowledge
            ));
        }

        /// `analytic_bits` is finite and non-negative for any tuning the
        /// planner accepts in Standard mode.
        #[test]
        fn analytic_bits_finite_and_non_negative_standard(tuning in arb_tuning()) {
            let spec = test_spec(Mode::Standard);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec, tuning).unwrap();
            let analytic = f64::from(plan.analytic_bits());
            prop_assert!(analytic.is_finite());
            prop_assert!(analytic >= 0.0);
        }
    }
}
