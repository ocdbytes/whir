//! Derives a [`ParameterPlan`] from a spec + tuning. All cross-protocol
//! coordination — per-round loop, `t_ood ↔ r` and `ℓ_zk ↔ c_zk` fixed-points,
//! shared C_zk + mask-proximity — lives here.

use std::marker::PhantomData;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        irs_commit::{self, Config as IrsConfig},
        params::{
            basecase as bc_solver, code_switch as cs_solver, irs_commit as irs_solver,
            mask_proximity as mp_solver,
            plan::{
                MaskOracleInfo, MaskOraclePlan, ParameterPlan, RoundMode, RoundPlan, SharedPlan,
            },
            spec::{
                LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget, RoundContext, SecuritySpec,
                TuningSpec,
            },
            sumcheck as sc_solver,
        },
    },
};

const L_ZK_MAX_ITER: usize = 16;
/// Smallest pow2 ≥ 1 — satisfies `solve_mask_code`'s pow2 assertion.
const L_ZK_BOOTSTRAP: usize = 2;

impl<M: Embedding + Default> ParameterPlan<M> {
    /// ZK mode runs a global ℓ_zk fixed-point so one C_zk covers every round.
    pub fn derive(spec: SecuritySpec<M>, tuning: TuningSpec) -> Self {
        let RoundLayout {
            shapes,
            basecase_vector_size,
            basecase_log_inv_rate,
        } = round_layout(&tuning);
        match spec.mode {
            Mode::Standard { .. } => derive_standard(
                spec,
                tuning,
                &shapes,
                basecase_vector_size,
                basecase_log_inv_rate,
            ),
            Mode::ZeroKnowledge => derive_zk(
                spec,
                tuning,
                &shapes,
                basecase_vector_size,
                basecase_log_inv_rate,
            ),
        }
    }
}

// Round layout
// ---------------------------------------------------------------------------

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

/// Shapes plus the basecase tail (size + rate of the message after every fold).
struct RoundLayout {
    shapes: Vec<RoundShape>,
    basecase_vector_size: usize,
    basecase_log_inv_rate: u32,
}

struct RoundData<M: Embedding> {
    source: IrsConfig<M>,
    target: IrsConfig<Identity<M::Target>>,
    t_ood: usize,
}

/// Stops when there's no room for both a valid source and a valid target IRS.
fn round_layout(tuning: &TuningSpec) -> RoundLayout {
    assert!(tuning.vector_size.is_power_of_two());
    assert!(tuning.folding_factor >= 1);
    assert!(tuning.initial_folding_factor >= 1);

    let mut num_vars = tuning.vector_size.trailing_zeros() as usize;
    let mut log_inv_rate = tuning.starting_log_inv_rate;
    let mut source_folding = tuning.initial_folding_factor;
    let target_folding = tuning.folding_factor;
    let mut shapes = Vec::new();

    while num_vars >= source_folding + target_folding {
        #[allow(clippy::cast_possible_truncation)]
        shapes.push(RoundShape {
            round_index: shapes.len(),
            source_vector_size: 1usize << num_vars,
            source_log_inv_rate: log_inv_rate,
            source_folding_factor: source_folding as u32,
            target_folding_factor: target_folding as u32,
        });
        num_vars -= source_folding;
        #[allow(clippy::cast_possible_truncation)]
        {
            log_inv_rate += (source_folding as u32).saturating_sub(1);
        }
        source_folding = target_folding;
    }

    RoundLayout {
        shapes,
        basecase_vector_size: 1usize << num_vars,
        basecase_log_inv_rate: log_inv_rate,
    }
}

const fn round_context(shape: &RoundShape) -> RoundContext {
    RoundContext {
        round_index: shape.round_index,
        vector_size: shape.source_vector_size,
        log_inv_rate: shape.source_log_inv_rate,
        folding_factor: shape.source_folding_factor,
    }
}

fn target_context<M: Embedding>(shape: &RoundShape, source: &IrsConfig<M>) -> RoundContext {
    RoundContext {
        round_index: shape.round_index,
        vector_size: source.message_length(),
        log_inv_rate: shape.source_log_inv_rate + shape.source_folding_factor.saturating_sub(1),
        folding_factor: shape.target_folding_factor,
    }
}

// Standard mode
// ---------------------------------------------------------------------------

fn derive_standard<M: Embedding + Default>(
    spec: SecuritySpec<M>,
    tuning: TuningSpec,
    shapes: &[RoundShape],
    basecase_vector_size: usize,
    basecase_log_inv_rate: u32,
) -> ParameterPlan<M> {
    let target_spec = transfer_spec_to_target(&spec);
    let rounds = shapes
        .iter()
        .map(|shape| build_round(&spec, shape, None))
        .collect();
    let basecase = bc_solver::solve(&target_spec, basecase_vector_size, basecase_log_inv_rate);
    ParameterPlan {
        security: spec,
        tuning,
        shared: SharedPlan { mask_oracle: None },
        rounds,
        basecase,
    }
}

// Zero-knowledge mode — global ℓ_zk fixed-point + shared C_zk
// ---------------------------------------------------------------------------

fn derive_zk<M: Embedding + Default>(
    spec: SecuritySpec<M>,
    tuning: TuningSpec,
    shapes: &[RoundShape],
    basecase_vector_size: usize,
    basecase_log_inv_rate: u32,
) -> ParameterPlan<M> {
    let target_spec: SecuritySpec<Identity<M::Target>> = transfer_spec_to_target(&spec);
    let c_zk_log_inv_rate = LogInvRate::new(tuning.starting_log_inv_rate);

    // Lemma 6.4: one mask polynomial per sumcheck round. C_zk holds 2×.
    let total_masks: usize = shapes
        .iter()
        .map(|s| s.source_folding_factor as usize)
        .sum();
    assert!(total_masks > 0, "ZK requires ≥ 1 mask polynomial");
    let c_zk_num_vectors = 2 * total_masks;

    let mut l_zk = MaskCodeMessageLen::new(L_ZK_BOOTSTRAP);
    let mut c_zk =
        irs_solver::solve_mask_code(&target_spec, l_zk, 0, c_zk_log_inv_rate, c_zk_num_vectors);

    let mut last_round_data: Vec<RoundData<M>> = Vec::new();

    for _ in 0..L_ZK_MAX_ITER {
        let round_data: Vec<RoundData<M>> = shapes
            .iter()
            .map(|shape| build_zk_round_data(&spec, shape, c_zk.list_size()))
            .collect();

        let max_r_plus_t_ood = round_data
            .iter()
            .map(|r| r.source.mask_length() + r.t_ood)
            .max()
            .expect("non-empty rounds");
        let new_l_zk = MaskCodeMessageLen::new(max_r_plus_t_ood.next_power_of_two());

        if new_l_zk.get() == l_zk.get() {
            last_round_data = round_data;
            break;
        }

        l_zk = new_l_zk;
        // Solve_mask_code asserts `ℓ_zk ≥ r`; pass the max so it always holds.
        let max_source_mask = round_data
            .iter()
            .map(|r| r.source.mask_length())
            .max()
            .unwrap_or(0);
        c_zk = irs_solver::solve_mask_code(
            &target_spec,
            l_zk,
            max_source_mask,
            c_zk_log_inv_rate,
            c_zk_num_vectors,
        );
        last_round_data = round_data;
    }

    let mask_oracle_info = MaskOracleInfo {
        c_zk_list_size: c_zk.list_size(),
        l_zk,
    };

    let rounds = shapes
        .iter()
        .zip(last_round_data)
        .map(|(shape, data)| finalize_zk_round(&spec, shape, data, mask_oracle_info))
        .collect();

    let mask_proximity = mp_solver::solve(&target_spec, c_zk.clone(), total_masks);
    let mask_oracle = MaskOraclePlan {
        c_zk,
        l_zk,
        mask_proximity,
    };

    let basecase = bc_solver::solve(&target_spec, basecase_vector_size, basecase_log_inv_rate);

    ParameterPlan {
        security: spec,
        tuning,
        shared: SharedPlan {
            mask_oracle: Some(mask_oracle),
        },
        rounds,
        basecase,
    }
}

/// Local fixed point: `source.mask_length` covers `t_ood` queries; `t_ood` is
/// sized against `source.message + source.mask`.
fn build_zk_round_data<M: Embedding + Default>(
    spec: &SecuritySpec<M>,
    shape: &RoundShape,
    c_zk_list_size: f64,
) -> RoundData<M> {
    const LOCAL_MAX_ITER: usize = 16;

    let src_ctx = round_context(shape);
    let mut source = irs_solver::solve(spec, &src_ctx, OodSampleBudget::new(0));
    let mut t_ood = 0;
    let mut target = irs_solver::solve(
        &transfer_spec_to_target(spec),
        &target_context(shape, &source),
        OodSampleBudget::new(0),
    );

    for _ in 0..LOCAL_MAX_ITER {
        let new_t_ood = compute_t_ood(spec, &source, target.list_size(), Some(c_zk_list_size));
        let new_source = irs_solver::solve(spec, &src_ctx, OodSampleBudget::new(new_t_ood));
        let new_target = irs_solver::solve(
            &transfer_spec_to_target(spec),
            &target_context(shape, &new_source),
            OodSampleBudget::new(new_t_ood),
        );

        if new_t_ood == t_ood
            && new_source.codeword_length == source.codeword_length
            && new_target.codeword_length == target.codeword_length
        {
            return RoundData {
                source: new_source,
                target: new_target,
                t_ood: new_t_ood,
            };
        }

        source = new_source;
        target = new_target;
        t_ood = new_t_ood;
    }

    panic!("per-round ZK fixed-point did not converge");
}

fn finalize_zk_round<M: Embedding>(
    spec: &SecuritySpec<M>,
    shape: &RoundShape,
    data: RoundData<M>,
    mask_oracle: MaskOracleInfo,
) -> RoundPlan<M> {
    let RoundData {
        source,
        target,
        t_ood,
    } = data;
    let src_ctx = round_context(shape);
    let sumcheck = sc_solver::solve(spec, &src_ctx, &source, Some(mask_oracle));
    let code_switch = cs_solver::solve(spec, source, target, t_ood, Some(mask_oracle));
    RoundPlan {
        round_index: shape.round_index,
        sumcheck,
        code_switch,
        mode: RoundMode::ZeroKnowledge {
            t_ood: OodSampleBudget::new(t_ood),
            mask_oracle,
        },
    }
}

// Standard mode per-round builder
// ---------------------------------------------------------------------------

fn build_round<M: Embedding + Default>(
    spec: &SecuritySpec<M>,
    shape: &RoundShape,
    mask_oracle: Option<MaskOracleInfo>,
) -> RoundPlan<M> {
    debug_assert!(mask_oracle.is_none(), "ZK path uses finalize_zk_round");

    let target_spec = transfer_spec_to_target(spec);
    let src_ctx = round_context(shape);
    let source = irs_solver::solve(spec, &src_ctx, OodSampleBudget::new(0));

    let mut target = irs_solver::solve(
        &target_spec,
        &target_context(shape, &source),
        OodSampleBudget::new(0),
    );
    let mut t_ood = compute_t_ood(spec, &source, target.list_size(), None);
    for _ in 0..8 {
        let new_target = irs_solver::solve(
            &target_spec,
            &target_context(shape, &source),
            OodSampleBudget::new(t_ood),
        );
        let new_t_ood = compute_t_ood(spec, &source, new_target.list_size(), None);
        if new_target.codeword_length == target.codeword_length && new_t_ood == t_ood {
            target = new_target;
            t_ood = new_t_ood;
            break;
        }
        target = new_target;
        t_ood = new_t_ood;
    }

    let sumcheck = sc_solver::solve(spec, &src_ctx, &source, None);
    let code_switch = cs_solver::solve(spec, source, target, t_ood, None);
    RoundPlan {
        round_index: shape.round_index,
        sumcheck,
        code_switch,
        mode: RoundMode::Standard,
    }
}

// Cross-protocol bound helpers
// ---------------------------------------------------------------------------

/// Per-round `ℓ_zk = next_power_of_two(r + t_ood)` (Lemma 9.3). The global
/// ℓ_zk in [`derive_zk`] is the max-then-pad over all rounds, computed inline.
#[allow(dead_code)]
pub(super) const fn compute_l_zk<M: Embedding>(
    source: &IrsConfig<M>,
    t_ood: usize,
) -> MaskCodeMessageLen {
    MaskCodeMessageLen::new((source.mask_length() + t_ood).next_power_of_two())
}

/// Solves Lemma 9.9 term 1 for `t_ood`. In ZK, `degree = ℓ + r + t_ood`
/// couples back to `t_ood`, so iterate.
pub(super) fn compute_t_ood<M: Embedding>(
    spec: &SecuritySpec<M>,
    source: &IrsConfig<M>,
    target_list_size: f64,
    c_zk_list_size: Option<f64>,
) -> usize {
    const MAX_ITER: usize = 32;

    let security_target = spec.protocol_security_target_bits();
    let field_bits = M::Target::field_size_bits();
    let unique_decoding = spec.mode.unique_decoding();
    let combined_list_size = target_list_size * c_zk_list_size.unwrap_or(1.0);
    let message_length = source.message_length();
    let source_mask_length = source.mask_length();

    let solve_for_degree = |degree: usize| {
        irs_commit::num_ood_samples(
            unique_decoding,
            security_target,
            field_bits,
            combined_list_size,
            degree,
        )
    };

    if !matches!(spec.mode, Mode::ZeroKnowledge) {
        return solve_for_degree(message_length);
    }

    let mut t_ood = 0;
    for _ in 0..MAX_ITER {
        let new_t_ood = solve_for_degree(message_length + source_mask_length + t_ood);
        if new_t_ood == t_ood {
            return t_ood;
        }
        t_ood = new_t_ood;
    }
    panic!("compute_t_ood did not converge in {MAX_ITER} iterations");
}

// SecuritySpec helpers
// ---------------------------------------------------------------------------

/// C_zk lives in `Identity<M::Target>`; copy the rest of the spec across.
const fn transfer_spec_to_target<M: Embedding>(
    spec: &SecuritySpec<M>,
) -> SecuritySpec<Identity<M::Target>> {
    SecuritySpec {
        mode: spec.mode,
        target_security_bits: spec.target_security_bits,
        max_pow_bits: spec.max_pow_bits,
        hash_id: spec.hash_id,
        _embedding: PhantomData,
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::{
        hash,
        protocols::params::{bounds::SoundnessBounded, test_utils::TestEmbedding},
    };

    fn tuning_with(vector_size: usize) -> TuningSpec {
        TuningSpec {
            vector_size,
            starting_log_inv_rate: 1,
            initial_folding_factor: 2,
            folding_factor: 2,
        }
    }

    /// Keeps PoW below the 60-bit cap for small test tunings.
    fn test_spec<M: Embedding>(mode: Mode) -> SecuritySpec<M> {
        SecuritySpec {
            mode,
            target_security_bits: 40,
            max_pow_bits: None,
            hash_id: hash::BLAKE3,
            _embedding: PhantomData,
        }
    }

    #[test]
    fn round_shapes_match_old_whir_loop() {
        let tuning = tuning_with(1 << 10);
        let layout = round_layout(&tuning);
        assert!(!layout.shapes.is_empty());
        assert_eq!(layout.shapes[0].source_vector_size, 1 << 10);
        assert_eq!(layout.shapes[0].source_folding_factor, 2);
    }

    #[test]
    fn derive_standard_assembles() {
        let spec: SecuritySpec<TestEmbedding> = test_spec(Mode::Standard {
            unique_decoding: false,
        });
        let tuning = tuning_with(1 << 8);
        let plan = ParameterPlan::derive(spec, tuning);
        assert!(
            plan.shared.mask_oracle.is_none(),
            "Standard ⇒ no mask oracle"
        );
        assert!(!plan.rounds.is_empty());
        for r in &plan.rounds {
            assert!(matches!(r.mode, RoundMode::Standard));
        }
    }

    #[test]
    fn derive_zk_produces_shared_mask_oracle() {
        let spec: SecuritySpec<TestEmbedding> = test_spec(Mode::ZeroKnowledge);
        let tuning = tuning_with(1 << 8);
        let plan = ParameterPlan::derive(spec, tuning);
        let mask_oracle = plan
            .shared
            .mask_oracle
            .as_ref()
            .expect("ZK plan must produce a mask oracle");

        // Bound 3: ℓ_zk dominates every round's r + t_ood.
        for r in &plan.rounds {
            let RoundMode::ZeroKnowledge {
                t_ood,
                mask_oracle: round_oracle,
            } = r.mode
            else {
                panic!("expected ZK round");
            };
            let source_mask = r.code_switch.source.mask_length();
            assert!(mask_oracle.l_zk.get() >= source_mask + t_ood.get());
            assert_eq!(round_oracle.l_zk.get(), mask_oracle.l_zk.get());
            assert_eq!(round_oracle.c_zk_list_size, mask_oracle.c_zk.list_size());
        }

        let total_masks: usize = plan.rounds.iter().map(|r| r.sumcheck.num_rounds).sum();
        assert_eq!(mask_oracle.c_zk.num_vectors, 2 * total_masks);
        assert_eq!(mask_oracle.mask_proximity.num_masks, total_masks);
    }

    fn basecase_min_bits<M: Embedding>(plan: &ParameterPlan<M>) -> f64 {
        let sumcheck = f64::from(sc_solver::analytic_error_bits(&plan.basecase.commit, None));
        if matches!(
            plan.basecase.mode,
            crate::protocols::basecase::Mode::ZeroKnowledge
        ) {
            sumcheck.min(f64::from(bc_solver::analytic_error_bits(
                &plan.basecase.commit,
            )))
        } else {
            sumcheck
        }
    }

    #[test]
    fn analytic_bits_finite_and_positive_standard() {
        let spec: SecuritySpec<TestEmbedding> = test_spec(Mode::Standard {
            unique_decoding: false,
        });
        let plan = ParameterPlan::derive(spec, tuning_with(1 << 8));
        let bits: f64 = plan.analytic_bits().into();
        assert!(bits.is_finite() && bits > 0.0, "bits = {bits}");
        let min_round = plan
            .rounds
            .iter()
            .map(|r| f64::from(r.analytic_bits()))
            .fold(f64::INFINITY, f64::min);
        let expected = min_round.min(basecase_min_bits(&plan));
        assert!((bits - expected).abs() < 1e-9, "{bits} vs {expected}");
    }

    #[test]
    fn analytic_bits_includes_mask_oracle_in_zk() {
        let spec: SecuritySpec<TestEmbedding> = test_spec(Mode::ZeroKnowledge);
        let plan = ParameterPlan::derive(spec, tuning_with(1 << 8));
        let plan_bits: f64 = plan.analytic_bits().into();
        let mo_bits: f64 = plan
            .shared
            .mask_oracle
            .as_ref()
            .expect("ZK has mask oracle")
            .analytic_bits()
            .into();
        let min_round = plan
            .rounds
            .iter()
            .map(|r| f64::from(r.analytic_bits()))
            .fold(f64::INFINITY, f64::min);
        let expected = mo_bits.min(min_round).min(basecase_min_bits(&plan));
        assert!(
            (plan_bits - expected).abs() < 1e-9,
            "{plan_bits} vs {expected}"
        );
    }

    #[test]
    fn derive_plans_basecase() {
        let spec: SecuritySpec<TestEmbedding> = test_spec(Mode::ZeroKnowledge);
        let plan = ParameterPlan::derive(spec, tuning_with(1 << 8));
        assert!(matches!(
            plan.basecase.mode,
            crate::protocols::basecase::Mode::ZeroKnowledge
        ));
        assert_eq!(plan.basecase.commit.interleaving_depth, 1);
        // Sumcheck folds basecase to size 1.
        assert_eq!(plan.basecase.sumcheck.final_size(), 1);
    }
}
