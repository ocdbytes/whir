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
    use crate::protocols::params::test_utils::assert_close;

    /// Tighter tolerance for tests doing relative-error checks (`(got - exp).abs() / exp`)
    /// against an alternative-derived expected value with the same operations.
    const TIGHT_EPS: f64 = 1e-12;

    /// Johnson list size: `|Λ| = 1 / (2η√ρ)`, log₂ form. Hand-evaluated at
    /// `log_inv_rate = 2`, `η = 0.1`: `−1 − log₂(0.1) + 1 ≈ 3.3219`.
    #[test]
    fn list_size_log2_johnson_formula() {
        let got = list_size_log2(2.0, 0.1);
        let expected = -1.0 - 0.1_f64.log2() + 0.5 * 2.0;
        assert_close(got, expected);
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
                (got - expected).abs() / expected < TIGHT_EPS,
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
        // All shape values except rate are placeholders — `list_size()` depends
        // only on `johnson_slack`, which is itself a function of rate.
        const PLACEHOLDER_SECURITY_TARGET_BITS: f64 = 80.0;
        const PLACEHOLDER_NUM_VECTORS: usize = 2;
        const PLACEHOLDER_VECTOR_SIZE: usize = 8;
        const PLACEHOLDER_INTERLEAVING_DEPTH: usize = 1;
        const LOG_INV_RATE: u32 = 2;

        let config: Config<Identity<Field64>> = Config::new(
            PLACEHOLDER_SECURITY_TARGET_BITS,
            false, // unique_decoding
            hash::BLAKE3,
            PLACEHOLDER_NUM_VECTORS,
            PLACEHOLDER_VECTOR_SIZE,
            PLACEHOLDER_INTERLEAVING_DEPTH,
            2_f64.powf(-f64::from(LOG_INV_RATE)),
            IrsMode::Standard,
        );
        let got = johnson_list_size(f64::from(LOG_INV_RATE));
        let expected = config.list_size();
        assert!(
            (got - expected).abs() / expected < TIGHT_EPS,
            "bounds helper ({got}) vs Config::list_size ({expected})",
        );
    }

    /// OOD per-sample Schwartz–Zippel: `log₂((k−1) / |F|) = log₂(k−1) − field_bits`.
    #[test]
    fn ood_per_sample_log2_formula() {
        // `k = 129` so `k − 1 = 128 = 2^7` for exact `log2`.
        const K: usize = 129;
        const FIELD_BITS: f64 = 64.0;

        let got = ood_per_sample_log2(K, FIELD_BITS);
        let expected = ((K - 1) as f64).log2() - FIELD_BITS;
        assert_close(got, expected);
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
        assert_close(got, expected);
    }

    /// `1 − δ` in Johnson regime: `√ρ + η`.
    #[test]
    fn one_minus_distance_log2_johnson() {
        let log_inv_rate = 2.0;
        let eta = 0.1;
        let got = one_minus_distance_log2(log_inv_rate, eta);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = (rho.sqrt() + eta).log2();
        assert_close(got, expected);
    }

    /// MCA fixture — `message_length = 16 = 2^4` and `log_inv_rate = 2` give
    /// exact `log2(k) = 4`. `field_bits = 64.0` for Field64.
    const MCA_MESSAGE_LENGTH: usize = 16;
    const MCA_LOG_INV_RATE: f64 = 2.0;
    const MCA_FIELD_BITS: f64 = 64.0;

    /// MCA error, unique-decoding branch: `log k + log_inv_rate − field_bits`.
    #[test]
    fn eps_mca_log2_unique_decoding_formula() {
        let p = CodeParams {
            log_inv_rate: MCA_LOG_INV_RATE,
            johnson_slack: 0.0,
            message_length: MCA_MESSAGE_LENGTH,
            field_bits: MCA_FIELD_BITS,
        };
        let got = eps_mca_log2(&p);
        let expected = (MCA_MESSAGE_LENGTH as f64).log2() + MCA_LOG_INV_RATE - MCA_FIELD_BITS;
        assert_close(got, expected);
    }

    /// MCA error, Johnson branch: `7·log₂10 + 3.5·log_inv_rate + 2·log k − field_bits`.
    #[test]
    fn eps_mca_log2_johnson_formula() {
        // `η = 0.1` stays within the debug assertion's slack range:
        // `η.log2() ≥ −(0.5·log_inv_rate + log₂10 + 1) ≈ −5.32`.
        const JOHNSON_SLACK: f64 = 0.1;

        let p = CodeParams {
            log_inv_rate: MCA_LOG_INV_RATE,
            johnson_slack: JOHNSON_SLACK,
            message_length: MCA_MESSAGE_LENGTH,
            field_bits: MCA_FIELD_BITS,
        };
        let got = eps_mca_log2(&p);
        let expected =
            7.0 * LOG2_10 + 3.5 * MCA_LOG_INV_RATE + 2.0 * (MCA_MESSAGE_LENGTH as f64).log2()
                - MCA_FIELD_BITS;
        assert_close(got, expected);
    }

    /// `pow_bits_to_close_gap` clamps negative gaps to zero (no anti-grind).
    #[test]
    fn pow_bits_to_close_gap_saturates_at_zero() {
        assert_eq!(f64::from(pow_bits_to_close_gap(100.0, 120.0)), 0.0);
        assert_eq!(f64::from(pow_bits_to_close_gap(100.0, 100.0)), 0.0);
        assert_eq!(f64::from(pow_bits_to_close_gap(100.0, 60.0)), 40.0);
    }
}
