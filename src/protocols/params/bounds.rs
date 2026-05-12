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

#[cfg(test)]
mod tests {
    use super::*;

    /// Within 1e-9 of expected — formulas use `log2`, so floats are inexact.
    fn approx_eq(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-9,
            "expected ≈ {b}, got {a} (diff {})",
            (a - b).abs()
        );
    }

    #[test]
    fn list_size_unique_decoding_is_one() {
        // Unique decoding → |Λ| = 1 → log2 = 0.
        approx_eq(list_size_log2(1.0, 0.0), 0.0);
        approx_eq(list_size_log2(5.0, 0.0), 0.0);
    }

    #[test]
    fn list_size_johnson_grows_as_slack_shrinks() {
        // Same rate, smaller slack → larger list.
        let big_slack = list_size_log2(2.0, 0.5);
        let small_slack = list_size_log2(2.0, 0.05);
        assert!(small_slack > big_slack, "{small_slack} > {big_slack}");
    }

    #[test]
    fn list_size_johnson_grows_as_rate_drops() {
        // Lower rate (larger log_inv_rate) → larger list.
        let high_rate = list_size_log2(1.0, 0.1);
        let low_rate = list_size_log2(4.0, 0.1);
        assert!(low_rate > high_rate, "{low_rate} > {high_rate}");
    }

    fn code(log_inv_rate: f64, johnson_slack: f64, message_length: usize) -> CodeParams {
        CodeParams {
            log_inv_rate,
            johnson_slack,
            message_length,
            field_bits: 64.0,
        }
    }

    #[test]
    fn eps_mca_grows_with_message_length() {
        // Longer message → larger ε (less negative log) → less security.
        let short = eps_mca_log2(&code(2.0, 0.1, 16));
        let long = eps_mca_log2(&code(2.0, 0.1, 1024));
        assert!(long > short, "{long} > {short}");
    }

    #[test]
    fn eps_mca_grows_with_log_inv_rate() {
        // Lower rate (larger log_inv_rate) → larger ε.
        let high_rate = eps_mca_log2(&code(1.0, 0.1, 128));
        let low_rate = eps_mca_log2(&code(4.0, 0.1, 128));
        assert!(low_rate > high_rate, "{low_rate} > {high_rate}");
    }

    #[test]
    fn one_minus_distance_unique_is_midpoint() {
        // Unique decoding: 1 - δ = (1 + ρ) / 2.
        // log_inv_rate = 1 → ρ = 0.5 → (1 + 0.5)/2 = 0.75.
        approx_eq(one_minus_distance_log2(1.0, 0.0), 0.75_f64.log2());
    }

    #[test]
    fn one_minus_distance_johnson_more_negative_than_unique() {
        // Johnson allows larger δ than unique decoding → smaller (1-δ) → more
        // negative log.
        let unique = one_minus_distance_log2(2.0, 0.0);
        let johnson = one_minus_distance_log2(2.0, 0.1);
        assert!(johnson < unique, "{johnson} < {unique}");
    }

    #[test]
    fn ood_per_sample_exact() {
        // (k-1)/|F| with k=2, |F|=2^64 → log2 = -64.
        approx_eq(ood_per_sample_log2(2, 64.0), -64.0);
        // k=9, |F|=2^64 → (8)/2^64 → log2 = 3 - 64 = -61.
        approx_eq(ood_per_sample_log2(9, 64.0), -61.0);
    }

    #[test]
    fn pow_bits_zero_when_achieved_meets_target() {
        assert!(pow_bits_to_close_gap(80.0, 100.0).is_zero());
        assert!(pow_bits_to_close_gap(80.0, 80.0).is_zero());
    }

    #[test]
    fn pow_bits_fills_gap_to_target() {
        let bits = pow_bits_to_close_gap(80.0, 50.0);
        approx_eq(f64::from(bits), 30.0);
    }
}
