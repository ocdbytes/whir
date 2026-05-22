//! Reed–Solomon decoding regime — materialized per-round parameters and the
//! analytic helpers that depend on them.
//!
//! Spec-level policy lives in [`super::spec::DecodingRegime`] (rate-independent,
//! a user choice). The data-carrying [`DecodingRegimeParams`] is what gets
//! stored on per-round configs once a rate is known: [`Self::from_policy`]
//! is the single materialization point.

use std::f64::consts::LOG2_10;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::protocols::params::{
    bounds::{rate, usize_to_f64},
    spec::DecodingRegime,
};

/// Materialized decoding-regime parameters.
///
/// `Unique` carries no data; `Johnson { slack }` carries `η`. The two variants
/// are statically distinct — there is no "Johnson with η = 0" representation,
/// so callers can pattern-match without a sentinel-comparison branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecodingRegimeParams {
    Unique,
    Johnson { slack: OrderedFloat<f64> },
}

impl DecodingRegimeParams {
    /// Materialize spec policy at a known rate. The canonical Johnson slack
    /// (`η = √ρ / 20`) is centralized here — any tuning of `η` lives at this
    /// site and propagates to every per-round config.
    // TODO: Optimize picking η.
    pub fn from_policy(policy: DecodingRegime, rate: f64) -> Self {
        match policy {
            DecodingRegime::Unique => Self::Unique,
            DecodingRegime::Johnson => Self::johnson_canonical(rate),
        }
    }

    /// Johnson regime with the canonical `η = √ρ / 20` slack.
    pub fn johnson_canonical(rate: f64) -> Self {
        Self::Johnson {
            slack: OrderedFloat(rate.sqrt() / 20.0),
        }
    }

    pub const fn is_unique(self) -> bool {
        matches!(self, Self::Unique)
    }

    /// `log₂ |Λ(C, δ)|`.
    pub fn list_size_log2(self, log_inv_rate: f64) -> f64 {
        match self {
            Self::Unique => 0.0,
            // Johnson: |Λ| = 1 / (2 η √ρ).
            Self::Johnson { slack } => -1.0 - slack.into_inner().log2() + 0.5 * log_inv_rate,
        }
    }

    /// `|Λ(C, δ)|`.
    pub fn list_size(self, log_inv_rate: f64) -> f64 {
        2_f64.powf(self.list_size_log2(log_inv_rate))
    }

    /// `log₂(1 − δ)`.
    pub fn one_minus_distance_log2(self, log_inv_rate: f64) -> f64 {
        let one_minus_delta = match self {
            Self::Unique => f64::midpoint(1.0, rate(log_inv_rate)),
            Self::Johnson { slack } => rate(log_inv_rate).sqrt() + slack.into_inner(),
        };
        one_minus_delta.log2()
    }

    /// `log₂ ε_mca(C, δ)`.
    pub fn eps_mca_log2(self, log_inv_rate: f64, message_length: usize, field_bits: f64) -> f64 {
        let log_k = usize_to_f64(message_length).log2();
        let error = match self {
            Self::Unique => log_k + log_inv_rate,
            Self::Johnson { slack } => {
                debug_assert!(
                    slack.into_inner().log2() >= -(0.5 * log_inv_rate + LOG2_10 + 1.0) - 1e-6
                );
                7.0 * LOG2_10 + 3.5 * log_inv_rate + 2.0 * log_k
            }
        };
        error - field_bits
    }
}

/// Johnson list size at the canonical `η = √ρ / 20` slack, as a function of
/// `log_inv_rate` only. Used by planners that need a list-size estimate before
/// a target config exists.
pub fn johnson_list_size(log_inv_rate: f64) -> f64 {
    DecodingRegimeParams::johnson_canonical(rate(log_inv_rate)).list_size(log_inv_rate)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::protocols::params::test_utils::assert_close;

    /// Tighter tolerance for tests doing relative-error checks against an
    /// alternative-derived expected value with the same operations.
    const TIGHT_EPS: f64 = 1e-12;

    fn johnson(slack: f64) -> DecodingRegimeParams {
        DecodingRegimeParams::Johnson {
            slack: OrderedFloat(slack),
        }
    }

    /// Johnson list size: `|Λ| = 1 / (2η√ρ)`, log₂ form. Hand-evaluated at
    /// `log_inv_rate = 2`, `η = 0.1`: `−1 − log₂(0.1) + 1 ≈ 3.3219`.
    #[test]
    fn list_size_log2_johnson_formula() {
        let got = johnson(0.1).list_size_log2(2.0);
        let expected = -1.0 - 0.1_f64.log2() + 0.5 * 2.0;
        assert_close(got, expected);
    }

    /// Unique-decoding regime gives `|Λ| = 1`, i.e. log = 0.
    #[test]
    fn list_size_log2_unique_decoding_is_zero() {
        assert_eq!(DecodingRegimeParams::Unique.list_size_log2(2.0), 0.0);
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

    /// `johnson_list_size(b)` must match `Config::list_size` once a config is
    /// built at the same rate. Keeps the rate-only helper in sync with
    /// `irs_commit::Config::new`'s canonical-slack materialization.
    #[test]
    fn johnson_list_size_matches_config_list_size() {
        use crate::{
            algebra::{embedding::Identity, fields::Field64},
            hash,
            protocols::irs_commit::{Config, IrsMode},
        };
        const PLACEHOLDER_SECURITY_TARGET_BITS: f64 = 80.0;
        const PLACEHOLDER_NUM_VECTORS: usize = 2;
        const PLACEHOLDER_VECTOR_SIZE: usize = 8;
        const PLACEHOLDER_INTERLEAVING_DEPTH: usize = 1;
        const LOG_INV_RATE: u32 = 2;

        let config: Config<Identity<Field64>> = Config::new(
            PLACEHOLDER_SECURITY_TARGET_BITS,
            DecodingRegime::Johnson,
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
            "regime helper ({got}) vs Config::list_size ({expected})",
        );
    }

    /// `1 − δ` in unique-decoding mode: midpoint of 1 and ρ.
    #[test]
    fn one_minus_distance_log2_unique() {
        let log_inv_rate = 2.0;
        let got = DecodingRegimeParams::Unique.one_minus_distance_log2(log_inv_rate);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = f64::midpoint(1.0, rho).log2();
        assert_close(got, expected);
    }

    /// `1 − δ` in Johnson regime: `√ρ + η`.
    #[test]
    fn one_minus_distance_log2_johnson() {
        let log_inv_rate = 2.0;
        let eta = 0.1;
        let got = johnson(eta).one_minus_distance_log2(log_inv_rate);
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
        let got = DecodingRegimeParams::Unique.eps_mca_log2(
            MCA_LOG_INV_RATE,
            MCA_MESSAGE_LENGTH,
            MCA_FIELD_BITS,
        );
        let expected = (MCA_MESSAGE_LENGTH as f64).log2() + MCA_LOG_INV_RATE - MCA_FIELD_BITS;
        assert_close(got, expected);
    }

    /// MCA error, Johnson branch: `7·log₂10 + 3.5·log_inv_rate + 2·log k − field_bits`.
    #[test]
    fn eps_mca_log2_johnson_formula() {
        // `η = 0.1` stays within the debug assertion's slack range.
        const JOHNSON_SLACK: f64 = 0.1;

        let got = johnson(JOHNSON_SLACK).eps_mca_log2(
            MCA_LOG_INV_RATE,
            MCA_MESSAGE_LENGTH,
            MCA_FIELD_BITS,
        );
        let expected =
            7.0 * LOG2_10 + 3.5 * MCA_LOG_INV_RATE + 2.0 * (MCA_MESSAGE_LENGTH as f64).log2()
                - MCA_FIELD_BITS;
        assert_close(got, expected);
    }

    /// `from_policy(Unique, _)` ignores rate; `from_policy(Johnson, rate)`
    /// produces the same materialization as `johnson_canonical(rate)`.
    #[test]
    fn from_policy_matches_canonical() {
        assert_eq!(
            DecodingRegimeParams::from_policy(DecodingRegime::Unique, 0.25),
            DecodingRegimeParams::Unique,
        );
        assert_eq!(
            DecodingRegimeParams::from_policy(DecodingRegime::Johnson, 0.25),
            DecodingRegimeParams::johnson_canonical(0.25),
        );
    }
}
