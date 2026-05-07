//! Parameter selection for the IRS commit protocol.

use crate::{
    algebra::embedding::Embedding,
    protocols::{
        irs_commit::{self, num_in_domain_queries, IrsMode},
        params::spec::{Mode, RoundContext, SecuritySpec},
    },
};

/// Solve IRS-commit parameters for a single round.
///
/// `out_domain_samples` is the OOD-query budget owed by Construction 9.7 /
/// Bound 2; in ZK mode it is part of the per-row randomness budget (Lemma 9.5
/// requires `mask_length ≥ in_domain_samples + out_domain_samples`).
pub fn solve<M: Embedding + Default>(
    spec: &SecuritySpec<M>,
    ctx: &RoundContext,
    out_domain_samples: usize,
) -> irs_commit::Config<M> {
    assert!(
        !(matches!(spec.mode, Mode::ZeroKnowledge) && spec.unique_decoding),
        "ZK mode requires Johnson regime (code-switch needs OOD samples)"
    );

    let security_target = f64::from(
        spec.target_security_bits
            .saturating_sub(spec.max_pow_bits.unwrap_or(0)),
    );
    let rate = 2_f64.powf(-f64::from(ctx.log_inv_rate));
    let interleaving_depth = 1_usize << ctx.folding_factor;

    let mode = match spec.mode {
        Mode::Standard => IrsMode::Standard,
        Mode::ZeroKnowledge => IrsMode::ZeroKnowledge {
            mask_length: mask_length(
                spec.unique_decoding,
                security_target,
                rate,
                out_domain_samples,
            ),
        },
    };

    irs_commit::Config::new(
        security_target,
        spec.unique_decoding,
        spec.hash_id,
        1,
        ctx.vector_size,
        interleaving_depth,
        rate,
        mode,
    )
}

/// Solve the shared C_zk IRS-commit config used to commit mask polynomials.
///
/// C_zk itself carries no IRS randomness (`IrsMode::Standard`); the masks it
/// commits to already are the randomness.
///
/// - `l_zk` — C_zk message length (Theorem 9.6: ℓ_zk ≥ r).
/// - `log_inv_rate` — C_zk's rate, chosen by the orchestrator.
/// - `num_vectors` — total mask polynomials per commit (e.g. `2 * num_masks`
///   for mask-proximity's original/fresh pairs).
pub fn solve_mask_code<M: Embedding + Default>(
    spec: &SecuritySpec<M>,
    l_zk: usize,
    log_inv_rate: u32,
    num_vectors: usize,
) -> irs_commit::Config<M> {
    assert!(
        matches!(spec.mode, Mode::ZeroKnowledge),
        "C_zk only exists in ZK mode"
    );
    assert!(
        !spec.unique_decoding,
        "code-switch requires Johnson regime (OOD samples needed)"
    );

    let security_target = f64::from(
        spec.target_security_bits
            .saturating_sub(spec.max_pow_bits.unwrap_or(0)),
    );
    let rate = 2_f64.powf(-f64::from(log_inv_rate));

    irs_commit::Config::new(
        security_target,
        spec.unique_decoding,
        spec.hash_id,
        num_vectors,
        l_zk,
        1,
        rate,
        IrsMode::Standard,
    )
}

/// Lemma 9.5 ZK budget: cover every query that reveals a polynomial value.
fn mask_length(
    unique_decoding: bool,
    security_target: f64,
    rate: f64,
    out_domain_samples: usize,
) -> usize {
    let in_domain = num_in_domain_queries(unique_decoding, security_target, rate);
    in_domain + out_domain_samples
}
