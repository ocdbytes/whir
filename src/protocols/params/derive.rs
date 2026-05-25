//! Derives a [`ProtocolConfig`] from a spec + tuning.

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        irs_commit::Config as IrsConfig,
        params::{
            basecase as basecase_params,
            bounds::usize_to_f64,
            code_switch as code_switch_params,
            error::{DeriveError, Pow},
            irs_commit as irs_params, mask_proximity as mask_proximity_params,
            protocol_config::{MaskOracleConfig, ProtocolConfig, RoundConfig, RoundMode},
            spec::{
                DecodingRegime, LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget,
                RoundContext, SecuritySpec, TuningSpec, ZkSpec,
            },
            sumcheck as sumcheck_params, SolveMode,
        },
    },
};

const T_OOD_MAX_ITER: usize = 32;

/// Mode flag for the OOD security bound in [`solve_t_ood`] /
/// [`ood_security_bits_at`].
#[derive(Clone, Copy)]
pub(super) enum OodMode {
    Standard,
    ZeroKnowledge { c_zk_log_inv_rate: f64 },
}

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Fails with [`DeriveError`] when the spec/tuning combination is
    /// infeasible.
    pub fn derive(spec: SecuritySpec, tuning: TuningSpec) -> Result<Self, DeriveError> {
        let RoundLayout {
            shapes,
            basecase_vector_size,
            basecase_log_inv_rate,
        } = round_layout(&tuning);

        let mode = match spec.mode {
            Mode::Standard => RoundBuildMode::Standard,
            Mode::ZeroKnowledge => RoundBuildMode::ZeroKnowledge {
                zk_spec: ZkSpec::try_new(&spec).expect("matched Mode::ZeroKnowledge above"),
                c_zk_log_inv_rate: LogInvRate::new(tuning.starting_log_inv_rate),
            },
        };

        let rounds: Vec<RoundConfig<M>> = shapes
            .iter()
            .map(|shape| build_round_config::<M>(&spec, shape, mode))
            .collect::<Result<_, _>>()?;

        let basecase = basecase_params::solve(&spec, basecase_vector_size, basecase_log_inv_rate)?;

        let plan = Self::new(spec, tuning, rounds, basecase);
        plan.validate()?;
        Ok(plan)
    }
}

/// Mode-dispatch input for [`build_round_config`].
#[derive(Clone, Copy)]
enum RoundBuildMode<'a> {
    Standard,
    ZeroKnowledge {
        zk_spec: ZkSpec<'a>,
        c_zk_log_inv_rate: LogInvRate,
    },
}

impl RoundBuildMode<'_> {
    fn to_ood_mode(self) -> OodMode {
        match self {
            Self::Standard => OodMode::Standard,
            Self::ZeroKnowledge {
                c_zk_log_inv_rate, ..
            } => OodMode::ZeroKnowledge {
                c_zk_log_inv_rate: f64::from(c_zk_log_inv_rate.get()),
            },
        }
    }
}

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

fn round_layout(tuning: &TuningSpec) -> RoundLayout {
    assert!(tuning.vector_size.is_power_of_two());
    assert!(tuning.folding_factor.min() >= 1);

    let mut num_vars = tuning.vector_size.trailing_zeros() as usize;
    let mut log_inv_rate = tuning.starting_log_inv_rate;
    let mut shapes = Vec::new();

    loop {
        let round = shapes.len();
        let source_folding = tuning.folding_factor.at_round(round);
        let target_folding = tuning.folding_factor.at_round(round.saturating_add(1));
        if num_vars < source_folding.saturating_add(target_folding) {
            break;
        }
        shapes.push(RoundShape {
            round_index: round,
            source_vector_size: 1usize << num_vars,
            source_log_inv_rate: log_inv_rate,
            source_folding_factor: source_folding as u32,
            target_folding_factor: target_folding as u32,
        });
        num_vars = num_vars.saturating_sub(source_folding);
        log_inv_rate = log_inv_rate.saturating_add((source_folding as u32).saturating_sub(1));
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
        log_inv_rate: shape
            .source_log_inv_rate
            .saturating_add(shape.source_folding_factor.saturating_sub(1)),
        folding_factor: shape.target_folding_factor,
    }
}

fn solve_round_source<M: Embedding + Default>(
    spec: &SecuritySpec,
    shape: &RoundShape,
    ood_mode: OodMode,
) -> Result<(IrsConfig<M>, usize), DeriveError> {
    let src_ctx = round_context(shape);
    let target_log_inv_rate = f64::from(
        shape
            .source_log_inv_rate
            .saturating_add(shape.source_folding_factor.saturating_sub(1)),
    );
    let target_log_degree = f64::from(
        shape
            .source_vector_size
            .trailing_zeros()
            .saturating_sub(shape.source_folding_factor),
    );
    let target_list_size = spec
        .decoding_regime
        .list_size_estimate(target_log_degree, target_log_inv_rate);
    solve_t_ood::<M>(
        spec,
        &src_ctx,
        target_list_size,
        ood_mode,
        shape.round_index,
    )
}

/// ZK-only: assemble the per-round mask oracle (C_zk codeword + mask-proximity
/// check).
fn build_mask_oracle<M: Embedding>(
    zk_spec: ZkSpec<'_>,
    source: &IrsConfig<M>,
    t_ood: usize,
    num_masks: usize,
    c_zk_log_inv_rate: LogInvRate,
    round_index: usize,
) -> Result<MaskOracleConfig<M::Target>, DeriveError> {
    let spec = zk_spec.as_inner();
    let l_zk = compute_l_zk(source, t_ood);
    let c_zk: IrsConfig<Identity<M::Target>> = irs_params::solve_mask_code(
        zk_spec,
        l_zk,
        source.mask_length(),
        c_zk_log_inv_rate,
        2 * num_masks,
    );
    let c_zk_list_size_estimate = spec.decoding_regime.list_size_estimate(
        (l_zk.get() as f64).log2(),
        f64::from(c_zk_log_inv_rate.get()),
    );
    debug_assert!(
        (c_zk.list_size() - c_zk_list_size_estimate).abs()
            < 1e-9 * c_zk_list_size_estimate.max(1.0),
        "c_zk.list_size() {} drifted from planner estimate {}",
        c_zk.list_size(),
        c_zk_list_size_estimate,
    );
    let mask_proximity = mask_proximity_params::solve(spec, c_zk.clone(), num_masks, round_index)?;
    Ok(MaskOracleConfig::new(c_zk, l_zk, mask_proximity))
}

fn build_round_config<M: Embedding + Default>(
    spec: &SecuritySpec,
    shape: &RoundShape,
    mode: RoundBuildMode<'_>,
) -> Result<RoundConfig<M>, DeriveError> {
    let ctx = round_context(shape);
    let (source, t_ood) = solve_round_source::<M>(spec, shape, mode.to_ood_mode())?;

    let (target_budget, solve_mode, round_mode) = match mode {
        RoundBuildMode::Standard => (
            OodSampleBudget::ZERO,
            SolveMode::Standard,
            RoundMode::Standard,
        ),
        RoundBuildMode::ZeroKnowledge {
            zk_spec,
            c_zk_log_inv_rate,
        } => {
            let num_masks =
                sumcheck_params::masks_required(&ctx) + code_switch_params::masks_required();
            let mask_oracle = build_mask_oracle::<M>(
                zk_spec,
                &source,
                t_ood,
                num_masks,
                c_zk_log_inv_rate,
                shape.round_index,
            )?;
            let solve_mode = SolveMode::ZeroKnowledge {
                mask_oracle: mask_oracle.info(),
            };
            let round_mode = RoundMode::ZeroKnowledge {
                t_ood: OodSampleBudget::new(t_ood),
                mask_oracle: Box::new(mask_oracle),
            };
            (OodSampleBudget::new(t_ood), solve_mode, round_mode)
        }
    };

    let target: IrsConfig<Identity<M::Target>> =
        irs_params::solve(spec, &target_context(shape, &source), target_budget);
    let sumcheck = sumcheck_params::solve(
        spec,
        &ctx,
        &source,
        solve_mode,
        Pow::RoundSumcheck {
            index: shape.round_index,
        },
    )?;
    let code_switch =
        code_switch_params::solve(spec, source, target, t_ood, solve_mode, shape.round_index)?;

    Ok(RoundConfig::new(
        shape.round_index,
        sumcheck,
        code_switch,
        round_mode,
    ))
}

/// `ℓ_zk = next_pow2(r + t_ood)` (Theorem 9.6 + Lemma 9.3).
pub(super) const fn compute_l_zk<M: Embedding>(
    source: &IrsConfig<M>,
    t_ood: usize,
) -> MaskCodeMessageLen {
    MaskCodeMessageLen::new(
        source
            .mask_length()
            .saturating_add(t_ood)
            .next_power_of_two(),
    )
}

/// Per-round `(source, t_ood)`.
///
/// Under `Unique`, `t_ood = 1` is pinned (the `log(L·(L−1)/2)` term degenerates
/// when `L = 1`, and Construction 9.7 requires `out_domain_samples ≥ 1`).
/// Otherwise linear search over `t_ood = 1..=T_OOD_MAX_ITER` for the smallest
/// value where [`ood_security_bits_at`] meets `protocol_security_target_bits`.
pub(super) fn solve_t_ood<M: Embedding + Default>(
    spec: &SecuritySpec,
    src_ctx: &RoundContext,
    target_list_size: f64,
    ood_mode: OodMode,
    round_index: usize,
) -> Result<(IrsConfig<M>, usize), DeriveError> {
    if matches!(spec.decoding_regime, DecodingRegime::Unique) {
        let source = irs_params::solve(spec, src_ctx, OodSampleBudget::new(1));
        return Ok((source, 1));
    }

    let security_target = f64::from(spec.protocol_security_target_bits());
    let field_bits = M::Target::field_size_bits();

    for t_ood in 1..=T_OOD_MAX_ITER {
        let source: IrsConfig<M> = irs_params::solve(spec, src_ctx, OodSampleBudget::new(t_ood));
        let bits =
            ood_security_bits_at(spec, &source, t_ood, target_list_size, ood_mode, field_bits);
        if bits >= security_target {
            return Ok((source, t_ood));
        }
    }
    Err(DeriveError::FixedPointDidNotConverge { round_index })
}

/// OOD security bits at candidate `t_ood`, per STIR Lemma 4.5:
/// `bits = t · (|F| − log d) − log(L · (L − 1) / 2) ≈ t·(|F| − log d) − 2·log L + 1`.
fn ood_security_bits_at<M: Embedding>(
    spec: &SecuritySpec,
    source: &IrsConfig<M>,
    t_ood: usize,
    target_list_size: f64,
    ood_mode: OodMode,
    field_bits: f64,
) -> f64 {
    let (log_degree, log_combined_list) = match ood_mode {
        OodMode::Standard => (
            usize_to_f64(source.message_length()).log2(),
            target_list_size.log2(),
        ),
        OodMode::ZeroKnowledge { c_zk_log_inv_rate } => {
            let l_zk = source
                .mask_length()
                .saturating_add(t_ood)
                .next_power_of_two();
            let c_zk_list = spec
                .decoding_regime
                .list_size_estimate(usize_to_f64(l_zk).log2(), c_zk_log_inv_rate);
            (
                usize_to_f64(source.message_length().saturating_add(l_zk)).log2(),
                (target_list_size * c_zk_list).log2(),
            )
        }
    };
    let ood = usize_to_f64(t_ood);
    ood * (field_bits - log_degree) - 2.0 * log_combined_list + 1.0
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

    const FIXTURE_FOLDING_FACTOR: usize = 2;
    const FIXTURE_LOG_INV_RATE: u32 = 1;

    const LOG_VECTOR_SIZE_NO_ROUNDS: u32 = 3;
    const LOG_VECTOR_SIZE_MULTI_ROUND: u32 = 8;

    const VARIED_INITIAL_FOLDING: usize = 3;
    const VARIED_STEADY_FOLDING: usize = 2;

    fn tuning_with(vector_size: usize) -> TuningSpec {
        TuningSpec {
            vector_size,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
        }
    }

    const PLAN_FIXTURE_TARGET_BITS: u32 = 40;

    fn test_spec(mode: Mode) -> SecuritySpec {
        SecuritySpec {
            mode,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: PLAN_FIXTURE_TARGET_BITS,
            pow_budget: PowBudget::per_slot(LOOSE_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        }
    }

    const RATE_STEPPING_STARTING_LOG_INV_RATE: u32 = 2;
    const MIN_ROUNDS_FOR_CHAINING_TEST: usize = 2;

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

    #[test]
    fn round_layout_basecase_size_consumes_remaining_num_vars() {
        let tuning = tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND);
        let layout = round_layout(&tuning);
        let consumed: u32 = layout.shapes.iter().map(|s| s.source_folding_factor).sum();
        let initial_num_vars = tuning.vector_size.trailing_zeros();
        let remaining = initial_num_vars - consumed;
        assert_eq!(layout.basecase_vector_size, 1usize << remaining);
    }

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

    #[test]
    fn t_ood_nonzero_in_johnson_zk() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Johnson,
            ..test_spec(Mode::ZeroKnowledge)
        };
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
    fn t_ood_pinned_to_one_in_unique_zk() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::ZeroKnowledge)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        for r in plan.rounds() {
            let RoundMode::ZeroKnowledge { t_ood, .. } = r.mode() else {
                panic!("expected ZK round")
            };
            assert_eq!(t_ood.get(), 1);
        }
    }

    #[test]
    fn c_zk_keeps_code_switch_mask_under_unique() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::ZeroKnowledge)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        for r in plan.rounds() {
            let mask_oracle = r.mask_oracle().expect("ZK round has a mask oracle");
            let k = r.code_switch().source.interleaving_depth.trailing_zeros() as usize;
            let expected_num_masks = k + 1;
            assert_eq!(mask_oracle.c_zk().num_vectors, 2 * expected_num_masks);
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
        assert_eq!(plan.basecase().sumcheck.final_size(), 1);
    }

    const LOOSE_POW_BUDGET_BITS: u32 = 60;
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

    #[test]
    fn check_pow_bits_passes_on_derived_plan() {
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            test_spec(Mode::ZeroKnowledge),
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(plan.check_pow_bits());
    }

    #[test]
    fn check_pow_bits_detects_over_budget_slot() {
        use crate::{bits::Bits, protocols::proof_of_work::Config as PowConfig};
        const MODERATE_POW_BUDGET_BITS: u32 = 30;
        let spec = SecuritySpec {
            pow_budget: PowBudget::per_slot(MODERATE_POW_BUDGET_BITS),
            ..test_spec(Mode::ZeroKnowledge)
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

    #[test]
    fn validate_round_chaining_detects_adjacent_round_mismatch() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let mut plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let n = plan.rounds().len();
        assert!(n >= 2, "need ≥ 2 rounds to break a mid-chain link");
        assert!(plan.check_all_invariants(), "fresh plan must validate");

        let bad_size = plan.rounds()[0].code_switch().target.vector_size + 1;
        plan.corrupt_round_target_vector_size_for_test(0, bad_size);

        let err = plan
            .validate_round_chaining()
            .expect_err("adjacent-round mismatch must trip the chain check");
        assert!(
            matches!(
                err,
                DeriveError::RoundChainBroken {
                    from: crate::protocols::params::error::ChainSource::Round(0),
                    to: crate::protocols::params::error::ChainTarget::NextRound(1),
                    ..
                }
            ),
            "got {err:?}",
        );
        assert!(!plan.check_all_invariants());
    }

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

    #[test]
    fn derive_reports_pow_ungrindable() {
        const UNREACHABLE_TARGET_BITS: u32 = 200;
        let spec = SecuritySpec {
            target_security_bits: UNREACHABLE_TARGET_BITS,
            ..test_spec(Mode::Standard)
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

    #[test]
    fn derive_reports_pow_budget_exceeded() {
        const TIGHT_MAX_POW: u32 = 5;
        let spec = SecuritySpec {
            pow_budget: PowBudget::per_slot(TIGHT_MAX_POW),
            ..test_spec(Mode::ZeroKnowledge)
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

    #[test]
    fn derive_threads_unique_decoding_standard() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::Standard)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_NO_ROUNDS),
        )
        .unwrap();
        assert!(plan.rounds().is_empty());
        assert!(plan.basecase().commit.unique_decoding());
    }

    #[test]
    fn derive_threads_unique_decoding_zk() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::ZeroKnowledge)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_NO_ROUNDS),
        )
        .unwrap();
        assert!(plan.rounds().is_empty());
        assert!(plan.basecase().commit.unique_decoding());
    }

    #[test]
    fn derive_multi_round_unique_decoding_succeeds() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::Standard)
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
            assert!(cs.out_domain_samples >= 1);
        }
        assert!(plan.basecase().commit.unique_decoding());
    }

    #[test]
    fn derive_multi_round_unique_decoding_zk_succeeds() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::ZeroKnowledge)
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

    #[test]
    fn derive_multi_round_capacity_decoding_succeeds() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Capacity,
            ..test_spec(Mode::Standard)
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

    #[test]
    fn derive_multi_round_capacity_decoding_zk_succeeds() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Capacity,
            ..test_spec(Mode::ZeroKnowledge)
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

    fn assert_plan_meets_target_per_slot<M: Embedding>(
        spec: &SecuritySpec,
        plan: &ProtocolConfig<M>,
    ) {
        for r in plan.rounds() {
            let mask_info = r.mask_oracle_info();
            let cs = r.code_switch();
            assert_pow_closes_gap(
                spec,
                sumcheck_params::analytic_error_bits(&cs.source, mask_info),
                &r.sumcheck().round_pow,
            );
            assert_pow_closes_gap(
                spec,
                code_switch_params::analytic_error_bits(
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
                    mask_proximity_params::analytic_error_bits(&mp.c_zk_commit, mp.num_masks),
                    &mp.pow,
                );
            }
        }
        assert_pow_closes_gap(
            spec,
            sumcheck_params::analytic_error_bits(&plan.basecase().commit, None),
            &plan.basecase().sumcheck.round_pow,
        );
        if matches!(
            plan.basecase().mode,
            crate::protocols::basecase::BasecaseMode::ZeroKnowledge
        ) {
            assert_pow_closes_gap(
                spec,
                basecase_params::analytic_error_bits(&plan.basecase().commit),
                &plan.basecase().pow,
            );
        }
    }

    proptest! {
        #[test]
        fn derived_plan_meets_target_per_slot_standard(tuning in arb_tuning()) {
            let spec = test_spec(Mode::Standard);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec.clone(), tuning).unwrap();
            assert_plan_meets_target_per_slot(&spec, &plan);
        }

        #[test]
        fn derived_plan_meets_target_per_slot_zk(tuning in arb_tuning()) {
            let log_threshold =
                tuning.folding_factor.at_round(0) + tuning.folding_factor.at_round(1);
            prop_assume!(tuning.vector_size.trailing_zeros() as usize >= log_threshold);
            let spec = test_spec(Mode::ZeroKnowledge);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec.clone(), tuning).unwrap();
            assert_plan_meets_target_per_slot(&spec, &plan);
        }

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
                let source_mask = cs.source.mask_length();
                prop_assert!(mask_oracle.l_zk().get() >= source_mask + t_ood.get());
            }
            prop_assert!(matches!(
                plan.basecase().mode,
                crate::protocols::basecase::BasecaseMode::ZeroKnowledge
            ));
        }

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
