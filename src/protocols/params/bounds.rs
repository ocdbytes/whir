//! Shared RS-code primitives + the [`SoundnessBounded`] abstraction.

use std::{f64::consts::LOG2_10, ops::Neg};

use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    bits::Bits,
    protocols::irs_commit::Config as IrsConfig,
};

/// Analytic soundness bits (excluding PoW) delivered by a protocol-level unit.
/// Sub-protocol `Config` types lack the cross-protocol context to self-report.
#[allow(dead_code)]
pub trait SoundnessBounded {
    fn analytic_bits(&self) -> Bits;
}

/// `johnson_slack == 0.0` selects the unique-decoding regime.
#[derive(Debug, Clone, Copy)]
pub struct CodeParams {
    pub log_inv_rate: f64,
    pub johnson_slack: f64,
    pub message_length: usize,
    pub field_bits: f64,
}

impl CodeParams {
    pub fn from_irs<M: Embedding>(irs: &IrsConfig<M>) -> Self {
        Self {
            log_inv_rate: irs.rate().log2().neg(),
            johnson_slack: irs.johnson_slack.into_inner(),
            message_length: irs.masked_message_length(),
            field_bits: M::Target::field_size_bits(),
        }
    }
}

/// `ρ = 2^-log_inv_rate`. Centralized so the rate formula lives in one place.
pub(super) fn rate(log_inv_rate: f64) -> f64 {
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

/// `|Λ(C)|` for a Johnson-regime code derived purely from the rate, using the
/// canonical `η = √ρ / 20` slack.
pub fn johnson_list_size(log_inv_rate: f64) -> f64 {
    let rate = 2_f64.powf(-log_inv_rate);
    let johnson_slack = rate.sqrt() / 20.0;
    2_f64.powf(list_size_log2(log_inv_rate, johnson_slack))
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

/// PoW difficulty to close a soundness gap: `max(0, target − achieved)`.
// TODO(phase-6): re-wire from the cross-protocol PoW pass.
#[allow(dead_code)]
pub fn pow_bits_to_close_gap(target_security_bits: f64, achieved_security_bits: f64) -> Bits {
    Bits::new((target_security_bits - achieved_security_bits).max(0.0))
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    /// Johnson list size: `|Λ| = 1 / (2η√ρ)`, log₂ form. Hand-evaluated at
    /// `log_inv_rate = 2`, `η = 0.1`: `−1 − log₂(0.1) + 1 ≈ 3.3219`.
    #[test]
    fn list_size_log2_johnson_formula() {
        let got = list_size_log2(2.0, 0.1);
        let expected = -1.0 - 0.1_f64.log2() + 0.5 * 2.0;
        assert!((got - expected).abs() < EPS, "got {got} vs {expected}");
    }

    /// Unique-decoding regime (`η = 0`) gives `|Λ| = 1`, i.e. log = 0.
    #[test]
    fn list_size_log2_unique_decoding_is_zero() {
        assert_eq!(list_size_log2(2.0, 0.0), 0.0);
    }

    /// `η = √ρ / 20` substituted into `|Λ| = 1/(2η√ρ)` simplifies to `10/ρ`.
    /// So `johnson_list_size(b) = 10 · 2^b`.
    #[test]
    fn johnson_list_size_closed_form() {
        for b in [1.0, 2.0, 3.0, 5.0] {
            let got = johnson_list_size(b);
            let expected = 10.0 * 2_f64.powf(b);
            assert!(
                (got - expected).abs() / expected < 1e-12,
                "log_inv_rate={b}: got {got} vs {expected}",
            );
        }
    }

    /// `johnson_list_size(b) = 2^list_size_log2(b, √ρ/20)` must match `Config::list_size`
    /// once a config is built at the same rate. Keeps the bounds helper in sync with
    /// `irs_commit::Config::new`'s `johnson_slack = √ρ / 20` policy.
    #[test]
    fn johnson_list_size_matches_config_list_size() {
        use crate::{
            algebra::{embedding::Identity, fields::Field64},
            hash,
            protocols::irs_commit::{Config, IrsMode},
        };
        let log_inv_rate = 2;
        let config: Config<Identity<Field64>> = Config::new(
            80.0,
            false,
            hash::BLAKE3,
            2,
            8,
            1,
            2_f64.powf(-f64::from(log_inv_rate)),
            IrsMode::Standard,
        );
        let got = johnson_list_size(f64::from(log_inv_rate));
        let expected = config.list_size();
        assert!(
            (got - expected).abs() / expected < 1e-12,
            "bounds helper ({got}) vs Config::list_size ({expected})",
        );
    }

    /// OOD per-sample Schwartz–Zippel: `log₂((k−1) / |F|) = log₂(k−1) − field_bits`.
    #[test]
    fn ood_per_sample_log2_formula() {
        let got = ood_per_sample_log2(129, 64.0);
        let expected = 128_f64.log2() - 64.0;
        assert!((got - expected).abs() < EPS, "got {got} vs {expected}");
        // (k−1)/|F| < 1 for sane parameters ⇒ log is negative.
        assert!(got < 0.0);
    }

    /// `1 − δ` in unique-decoding mode: midpoint of 1 and ρ.
    #[test]
    fn one_minus_distance_log2_unique() {
        let log_inv_rate = 2.0;
        let got = one_minus_distance_log2(log_inv_rate, 0.0);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = f64::midpoint(1.0, rho).log2();
        assert!((got - expected).abs() < EPS, "got {got} vs {expected}");
    }

    /// `1 − δ` in Johnson regime: `√ρ + η`.
    #[test]
    fn one_minus_distance_log2_johnson() {
        let log_inv_rate = 2.0;
        let eta = 0.1;
        let got = one_minus_distance_log2(log_inv_rate, eta);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = (rho.sqrt() + eta).log2();
        assert!((got - expected).abs() < EPS, "got {got} vs {expected}");
    }

    /// MCA error, unique-decoding branch: `log k + log_inv_rate − field_bits`.
    #[test]
    fn eps_mca_log2_unique_decoding_formula() {
        let p = CodeParams {
            log_inv_rate: 2.0,
            johnson_slack: 0.0,
            message_length: 16,
            field_bits: 64.0,
        };
        let got = eps_mca_log2(&p);
        let expected = 16_f64.log2() + 2.0 - 64.0;
        assert!((got - expected).abs() < EPS, "got {got} vs {expected}");
    }

    /// MCA error, Johnson branch: `7·log₂10 + 3.5·log_inv_rate + 2·log k − field_bits`.
    #[test]
    fn eps_mca_log2_johnson_formula() {
        let p = CodeParams {
            log_inv_rate: 2.0,
            // Stay within the debug assertion's slack range: johnson_slack.log2() ≥
            // -(0.5·log_inv_rate + log₂10 + 1) ≈ -5.32.
            johnson_slack: 0.1,
            message_length: 16,
            field_bits: 64.0,
        };
        let got = eps_mca_log2(&p);
        let expected = 7.0 * LOG2_10 + 3.5 * 2.0 + 2.0 * 16_f64.log2() - 64.0;
        assert!((got - expected).abs() < EPS, "got {got} vs {expected}");
    }

    /// `pow_bits_to_close_gap` clamps negative gaps to zero (no anti-grind).
    #[test]
    fn pow_bits_to_close_gap_saturates_at_zero() {
        assert_eq!(f64::from(pow_bits_to_close_gap(100.0, 120.0)), 0.0);
        assert_eq!(f64::from(pow_bits_to_close_gap(100.0, 100.0)), 0.0);
        assert_eq!(f64::from(pow_bits_to_close_gap(100.0, 60.0)), 40.0);
    }
}
