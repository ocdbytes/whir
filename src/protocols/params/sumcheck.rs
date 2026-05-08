use crate::{
    algebra::embedding::Embedding,
    protocols::{
        irs_commit,
        params::{
            bounds::{self, CodeParams},
            spec::{Mode, RoundContext, SecuritySpec},
        },
        proof_of_work, sumcheck,
    },
};

pub fn solve<M: Embedding>(
    spec: &SecuritySpec<M>,
    ctx: &RoundContext,
    irs_source: &irs_commit::Config<M>,
) -> sumcheck::Config<M::Target> {
    let num_rounds = num_sumcheck_rounds(spec, ctx);
    let mode = match spec.mode {
        Mode::Standard { .. } => sumcheck::SumcheckMode::Standard,
        Mode::ZeroKnowledge => sumcheck::SumcheckMode::ZeroKnowledge {
            mask_length: mask_length(),
        },
    };
    let round_pow = solve_sumcheck_round_pow(spec, irs_source);
    sumcheck::Config::new(ctx.vector_size, round_pow, num_rounds, mode)
}

const fn num_sumcheck_rounds<M: Embedding>(spec: &SecuritySpec<M>, ctx: &RoundContext) -> usize {
    if ctx.round_index == 0 {
        spec.initial_folding_factor
    } else {
        spec.folding_factor
    }
}

pub const fn masks_required<M: Embedding>(spec: &SecuritySpec<M>, ctx: &RoundContext) -> usize {
    match spec.mode {
        Mode::Standard { .. } => 0,
        Mode::ZeroKnowledge => num_sumcheck_rounds(spec, ctx),
    }
}

/// 3 coefficients = constant + linear + quadratic, sufficient to mask each
/// degree-2 sumcheck round polynomial.
const fn mask_length() -> usize {
    3
}

/// Sumcheck-specific PoW sizing: closes the per-round Lemma 6.5 soundness gap.
fn solve_sumcheck_round_pow<M: Embedding>(
    spec: &SecuritySpec<M>,
    irs_source: &irs_commit::Config<M>,
) -> proof_of_work::Config {
    let code = CodeParams::from_irs(irs_source);

    // Lemma 6.5 per-round error has two terms; security in bits is the min.
    // TODO: extend with `ℓ_zk · |Λ_C_zk|` factors in ZK mode once mask-code
    // params are available (PR 2).
    let sec_mca = -bounds::eps_mca_log2(&code);
    let sec_combination =
        code.field_bits - bounds::list_size_log2(code.log_inv_rate, code.johnson_slack) - 1.0;
    let achieved = sec_mca.min(sec_combination);

    let pow_bits = bounds::pow_bits_to_close_gap(spec.target_security_bits, achieved);
    proof_of_work::Config::from_difficulty(pow_bits)
}
