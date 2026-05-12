//! Shared primitives for parameter selection: RS bounds + PoW sizing.

use std::{f64::consts::LOG2_10, ops::Neg};

use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    bits::Bits,
    protocols::irs_commit,
};

/// `johnson_slack == 0.0` selects the unique-decoding regime.
#[derive(Debug, Clone, Copy)]
pub struct CodeParams {
    pub log_inv_rate: f64,
    pub johnson_slack: f64,
    pub message_length: usize,
    pub field_bits: f64,
}

impl CodeParams {
    pub fn from_irs<M: Embedding>(irs: &irs_commit::Config<M>) -> Self {
        Self {
            log_inv_rate: irs.rate().log2().neg(),
            johnson_slack: irs.johnson_slack.into_inner(),
            message_length: irs.masked_message_length(),
            field_bits: M::Target::field_size_bits(),
        }
    }
}

fn rate(log_inv_rate: f64) -> f64 {
    2_f64.powf(-log_inv_rate)
}

fn unique_decoding(johnson_slack: f64) -> bool {
    johnson_slack == 0.0
}

/// log2 |Λ(C, δ)|.
pub fn list_size_log2(log_inv_rate: f64, johnson_slack: f64) -> f64 {
    if unique_decoding(johnson_slack) {
        0.0
    } else {
        // Johnson: |Λ| = 1 / (2 η √ρ).
        -1.0 - johnson_slack.log2() + 0.5 * log_inv_rate
    }
}

/// log2 ε_mca(C, δ).
pub fn eps_mca_log2(p: &CodeParams) -> f64 {
    let log_k = (p.message_length as f64).log2();

    let error = if unique_decoding(p.johnson_slack) {
        log_k + p.log_inv_rate
    } else {
        debug_assert!(p.johnson_slack.log2() >= -(0.5 * p.log_inv_rate + LOG2_10 + 1.0) - 1e-6);
        7.0 * LOG2_10 + 3.5 * p.log_inv_rate + 2.0 * log_k
    };

    error - p.field_bits
}

/// log2(1 - δ).
pub fn one_minus_distance_log2(log_inv_rate: f64, johnson_slack: f64) -> f64 {
    let one_minus_delta = if unique_decoding(johnson_slack) {
        f64::midpoint(1.0, rate(log_inv_rate))
    } else {
        rate(log_inv_rate).sqrt() + johnson_slack
    };
    one_minus_delta.log2()
}

/// log2 of the per-OOD-sample Schwartz-Zippel error: (k-1)/|F|.
pub fn ood_per_sample_log2(message_length: usize, field_bits: f64) -> f64 {
    ((message_length - 1) as f64).log2() - field_bits
}

/// PoW difficulty to close a soundness gap: max(0, target − achieved).
///
/// Currently unused — solvers emit `Config::none()` PoW. Will be re-wired by
/// the cross-protocol PoW pass.
#[allow(dead_code)]
pub fn pow_bits_to_close_gap(target_security_bits: f64, achieved_security_bits: f64) -> Bits {
    Bits::new((target_security_bits - achieved_security_bits).max(0.0))
}
