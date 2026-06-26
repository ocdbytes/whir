//! Generic sumcheck protocol.
//!
//! The transcript / mask / challenge machinery is degree-agnostic. The
//! round polynomial computation is delegated to a [`RoundPolyOracle`]:
//!   - [`Config::prove`] is a thin wrapper that builds a dot-product oracle
//!     (degree 2) — the legacy `⟨a, b⟩ = sum` reduction.
//!   - [`Config::prove_with_oracle`] takes any oracle and is used by the
//!     selector sumcheck (degree 3) and any future higher-degree variants.
//!
//! The verifier ([`Config::verify`]) is fully degree-generic: it derives `c_1`
//! from the sumcheck invariant `p(0) + p(1) = sum` and accepts any
//! degree-`d` round polynomial whose `d` non-`c_1` coefficients the prover
//! sends.

use std::{fmt, num::NonZeroUsize};

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        dot,
        sumcheck::{compute_sumcheck_polynomial, fold, fold_and_compute_polynomial},
        univariate_evaluate,
    },
    protocols::proof_of_work,
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Type,
    utils::chunks_exact_or_empty,
};

/// Output from the sumcheck protocol (shared by prover and verifier).
#[must_use]
pub struct SumcheckOpening<F: Field> {
    pub round_challenges: Vec<F>,
    pub mask_rlc: F,
}

/// ZK sumcheck mask polynomial dimension.
///
/// Validated at construction to be at least `MIN = 3` — the round polynomial
/// has 3 coefficients (degree-2), so the mask must have at least as many to
/// hide it. Lemma 6.4 itself only requires `ℓ_zk ≥ 2`; the `3` floor is a
/// WHIR design choice tied to the degree-2 round polynomial (see
/// `params::sumcheck::zk_mask_length`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SumcheckMaskLen(usize);

impl SumcheckMaskLen {
    pub const MIN: usize = 3;

    pub const fn new(n: usize) -> Self {
        assert!(n >= Self::MIN);
        Self(n)
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SumcheckMode {
    Standard,
    ZeroKnowledge { mask_length: SumcheckMaskLen },
}

/// Per-round polynomial provider for [`Config::prove_with_oracle`].
///
/// The prover loop owns the transcript, masking, challenge sampling, and
/// per-round PoW; the oracle owns the round polynomial computation and the
/// internal-state fold. Implementations should pick the in-place fold
/// strategy that best fits their state representation — the dot-product
/// oracle fuses fold and round-poly into one pass via
/// [`crate::algebra::sumcheck::fold_and_compute_polynomial`], for example.
pub trait RoundPolyOracle<F: Field> {
    /// Degree `d` of every round polynomial.
    fn degree(&self) -> usize;

    /// Compute the round polynomial's `d` non-`c_1` coefficients
    /// `[c_0, c_2, c_3, …, c_d]` for the current round. If `prev_challenge`
    /// is `Some(r)`, the oracle must first fold its internal state by `r`
    /// (the challenge sampled in the previous round) and then compute the
    /// new round polynomial.
    ///
    /// `c_1` is the verifier-derivable coefficient
    /// `c_1 = sum − 2·c_0 − Σ_{i≥2} c_i` and is recovered by the prover
    /// loop, so the oracle never returns it.
    fn fold_and_compute(&mut self, prev_challenge: Option<F>) -> Vec<F>;

    /// Apply the final round's challenge so the oracle's internal state
    /// reflects the fully folded representation when the loop exits.
    fn finalize(&mut self, final_challenge: F);
}

#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F>
where
    F: Field,
{
    field: Type<F>,
    initial_size: usize,
    round_pow: proof_of_work::Config,
    num_rounds: usize,
    mode: SumcheckMode,
    /// Maximum degree of each round polynomial in the folded variable.
    /// `2` for the legacy dot-product sumcheck; `3` for the selector
    /// sumcheck (where `P_r(X) = eq(X, β) · ⟨M(X), V(X)⟩` is cubic).
    degree: NonZeroUsize,
}

impl<F: Field> Config<F> {
    pub fn new(
        initial_size: usize,
        round_pow: proof_of_work::Config,
        num_rounds: usize,
        mode: SumcheckMode,
        degree: NonZeroUsize,
    ) -> Self {
        assert!(num_rounds == 0 || initial_size.next_power_of_two() >= 1 << num_rounds);
        assert!(degree.get() >= 2, "sumcheck degree must be ≥ 2");
        // `SumcheckMaskLen::new` already enforces the ≥ 3 floor at construction;
        // here we only need the field-characteristic precondition from Lemma 6.4.
        if matches!(mode, SumcheckMode::ZeroKnowledge { .. }) {
            assert!(
                !F::ONE.double().is_zero(),
                "ZK sumcheck requires char(F) ≠ 2"
            );
        }
        Self {
            field: Type::new(),
            initial_size,
            round_pow,
            num_rounds,
            mode,
            degree,
        }
    }

    pub const fn degree(&self) -> NonZeroUsize {
        self.degree
    }

    pub const fn initial_size(&self) -> usize {
        self.initial_size
    }

    pub const fn round_pow(&self) -> proof_of_work::Config {
        self.round_pow
    }

    pub const fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    pub const fn mode(&self) -> &SumcheckMode {
        &self.mode
    }

    const fn mask_length(&self) -> usize {
        match &self.mode {
            SumcheckMode::Standard => 0,
            SumcheckMode::ZeroKnowledge { mask_length } => mask_length.get(),
        }
    }

    #[cfg(test)]
    pub(crate) const fn override_round_pow_for_test(&mut self, round_pow: proof_of_work::Config) {
        self.round_pow = round_pow;
    }

    pub fn final_size(&self) -> usize {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        if self.initial_size == 0 || self.num_rounds == 0 {
            self.initial_size
        } else {
            self.initial_size.next_power_of_two() >> self.num_rounds
        }
    }

    /// Reduce a claim `dot(a, b) == sum` via the degree-2 dot-product sumcheck.
    ///
    /// Thin wrapper around [`Self::prove_with_oracle`] that builds a
    /// dot-product oracle. The oracle folds `a` and `b` in place; on return
    /// they contain the post-loop folded values.
    ///
    /// # Panics
    ///
    /// Panics if `self.degree() != 2`. Use [`Self::prove_with_oracle`]
    /// directly for higher-degree sumchecks.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        a: &mut Vec<F>,
        b: &mut Vec<F>,
        sum: &mut F,
        masks: &[F],
    ) -> SumcheckOpening<F>
    where
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert_eq!(
            self.degree.get(),
            2,
            "Config::prove only supports degree-2 dot-product sumchecks; use prove_with_oracle"
        );
        assert_eq!(a.len(), self.initial_size);
        assert_eq!(b.len(), self.initial_size);
        debug_assert_eq!(dot(a, b), *sum);

        self.prove_with_oracle(prover_state, sum, masks, DotProductOracle { a, b })
    }

    /// Generic sumcheck prover. Drives the transcript / mask / challenge /
    /// PoW loop; delegates per-round polynomial computation to `oracle`.
    ///
    /// The `degree()` reported by `oracle` must match this config's degree.
    /// On return, `oracle`'s internal state reflects the fully folded
    /// representation (via [`RoundPolyOracle::finalize`]).
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove_with_oracle<O, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        sum: &mut F,
        masks: &[F],
        mut oracle: O,
    ) -> SumcheckOpening<F>
    where
        O: RoundPolyOracle<F>,
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        assert_eq!(
            oracle.degree(),
            self.degree.get(),
            "RoundPolyOracle::degree must match Config::degree"
        );
        assert_eq!(masks.len(), self.num_rounds * self.mask_length());

        let degree = self.degree.get();
        let half = F::from(2).inverse().unwrap();
        let polynomial_len = self.mask_length().max(degree + 1);

        let (mut mask_sum, mask_rlc) = self.maybe_send_initial_mask_sum(prover_state, masks);

        let mut univariate = Vec::with_capacity(polynomial_len);
        let mut round_challenges = Vec::with_capacity(self.num_rounds);
        let mut prev_round_challenge = None;
        for (round, mask) in
            chunks_exact_or_empty(masks, self.mask_length(), self.num_rounds).enumerate()
        {
            // Oracle computes the round polynomial's non-c1 coefficients
            // `[c_0, c_2, c_3, …, c_d]` (length d). The fold of the oracle's
            // internal state by `prev_round_challenge` is fused into this call
            // when the oracle supports it.
            let coeffs_no_c1 = oracle.fold_and_compute(prev_round_challenge);
            debug_assert_eq!(coeffs_no_c1.len(), degree);
            let c0 = coeffs_no_c1[0];
            let high = &coeffs_no_c1[1..]; // c_2, c_3, …, c_d
            let c1 = *sum - c0.double() - high.iter().copied().sum::<F>();

            // Build round polynomial. In Standard (`mask = []`, `mask_rlc = 1`,
            // `mask_sum = 0`) this collapses to `[c_0, c_1, c_2, …, c_d]`.
            univariate.clear();
            univariate.resize(polynomial_len, F::ZERO);
            let sum_multiple = F::from(1 << self.num_rounds.saturating_sub(round + 1));
            for (u, m) in univariate.iter_mut().zip(mask.iter()) {
                *u = sum_multiple * *m;
            }
            univariate[0] += (mask_sum - sum_multiple * eval_01(mask)) * half;
            univariate[0] += mask_rlc * c0;
            univariate[1] += mask_rlc * c1;
            for (slot, c) in univariate.iter_mut().skip(2).zip(high.iter()) {
                *slot += mask_rlc * *c;
            }

            prover_state.prover_message(&univariate[0]);
            prover_state.prover_messages(&univariate[2..]);

            // Receive the random evaluation point and update the sum.
            self.round_pow.prove(prover_state);
            let r = prover_state.verifier_message::<F>();
            round_challenges.push(r);
            // Update sum to p(r). Horner over [c_0, c_1, c_2, …, c_d].
            let mut s = *high.last().unwrap_or(&F::ZERO);
            for &c in high.iter().rev().skip(1) {
                s = s * r + c;
            }
            s = s * r + c1;
            s = s * r + c0;
            *sum = s;

            mask_sum = univariate_evaluate(&univariate, r) - mask_rlc * *sum;
            prev_round_challenge = Some(r);
        }
        if let Some(r) = prev_round_challenge {
            oracle.finalize(r);
        }

        *sum = mask_sum + mask_rlc * *sum;
        SumcheckOpening {
            round_challenges,
            mask_rlc,
        }
    }

    fn maybe_send_initial_mask_sum<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        masks: &[F],
    ) -> (F, F)
    where
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
    {
        match &self.mode {
            SumcheckMode::Standard => (F::ZERO, F::ONE),
            SumcheckMode::ZeroKnowledge { mask_length } => {
                if self.num_rounds == 0 {
                    return (F::ZERO, F::ONE);
                }
                let sum_multiple = F::from(1 << self.num_rounds.saturating_sub(1));
                let mask_sum = masks
                    .chunks_exact(mask_length.get())
                    .map(eval_01)
                    .sum::<F>()
                    * sum_multiple;
                prover_state.prover_message(&mask_sum);
                let mask_rlc = prover_state.verifier_message();
                (mask_sum, mask_rlc)
            }
        }
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut F,
    ) -> VerificationResult<SumcheckOpening<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );

        let mask_rlc = self.maybe_receive_initial_mask_sum(verifier_state, sum)?;

        let mut univariate = vec![F::ZERO; self.mask_length().max(self.degree.get() + 1)];
        let mut round_challenges = Vec::with_capacity(self.num_rounds);
        for _ in 0..self.num_rounds {
            // Receive all but linear coefficient.
            univariate[0] = verifier_state.prover_message()?;
            for c in &mut univariate[2..] {
                *c = verifier_state.prover_message()?;
            }

            // Derive linear coefficient from relation `univariate(0) + univariate(1) = sum`.
            univariate[1] = *sum - univariate[0].double() - univariate[2..].iter().sum::<F>();

            // Check proof of work (if any).
            self.round_pow.verify(verifier_state)?;

            // Receive the random evaluation point.
            let round_challenge = verifier_state.verifier_message::<F>();
            round_challenges.push(round_challenge);

            // Update the sum.
            *sum = univariate_evaluate(&univariate, round_challenge);
        }
        Ok(SumcheckOpening {
            round_challenges,
            mask_rlc,
        })
    }

    fn maybe_receive_initial_mask_sum<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut F,
    ) -> VerificationResult<F>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
    {
        match &self.mode {
            SumcheckMode::Standard => Ok(F::ONE),
            SumcheckMode::ZeroKnowledge { .. } => {
                if self.num_rounds == 0 {
                    return Ok(F::ONE);
                }
                let mask_sum: F = verifier_state.prover_message()?;
                let mask_rlc = verifier_state.verifier_message();
                *sum = mask_sum + mask_rlc * *sum;
                Ok(mask_rlc)
            }
        }
    }
}

impl<F: Field> fmt::Display for Config<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode_str = match &self.mode {
            SumcheckMode::Standard => "standard".to_string(),
            SumcheckMode::ZeroKnowledge { mask_length } => {
                format!("zk ℓ_zk={}", mask_length.get())
            }
        };
        write!(
            f,
            "size {} rounds {} pow {:.2} {}",
            self.initial_size,
            self.num_rounds,
            self.round_pow.difficulty(),
            mode_str,
        )
    }
}

// Evaluated a univariate as p(0) + p(1)
fn eval_01<F: Field>(coefficients: &[F]) -> F {
    if coefficients.is_empty() {
        return F::ZERO;
    }
    coefficients[0] + coefficients.iter().sum::<F>()
}

/// Degree-2 dot-product oracle for [`Config::prove_with_oracle`]: reduces
/// `⟨a, b⟩ = sum` via the legacy quadratic sumcheck. Folds `a` and `b` in
/// place using [`fold_and_compute_polynomial`] to fuse the previous-round
/// fold with the current-round polynomial computation in a single pass.
struct DotProductOracle<'a, F: Field> {
    a: &'a mut Vec<F>,
    b: &'a mut Vec<F>,
}

impl<F: Field> RoundPolyOracle<F> for DotProductOracle<'_, F> {
    fn degree(&self) -> usize {
        2
    }

    fn fold_and_compute(&mut self, prev_challenge: Option<F>) -> Vec<F> {
        let (c0, c2) = match prev_challenge {
            Some(w) => fold_and_compute_polynomial(self.a, self.b, w),
            None => compute_sumcheck_polynomial(self.a, self.b),
        };
        vec![c0, c2]
    }

    fn finalize(&mut self, final_challenge: F) {
        fold(self.a, final_challenge);
        fold(self.b, final_challenge);
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::{Distribution, Standard},
        rngs::StdRng,
        SeedableRng,
    };
    use proptest::{prelude::Just, prop_oneof, proptest, strategy::Strategy};
    #[cfg(feature = "tracing")]
    use tracing::instrument;

    use super::*;
    use crate::{
        algebra::{
            fields::{self, Field64},
            multilinear_extend, random_vector,
        },
        transcript::DomainSeparator,
    };

    impl<F: Field + 'static> Config<F>
    where
        Standard: Distribution<F>,
    {
        pub fn arbitrary() -> impl Strategy<Value = Self> {
            let mode_strategy = prop_oneof![
                3 => Just(SumcheckMode::Standard),
                7 => (3_usize..20).prop_map(|n| SumcheckMode::ZeroKnowledge {
                    mask_length: SumcheckMaskLen::new(n),
                }),
            ];
            (0_usize..(1 << 12), 0_usize..12, mode_strategy).prop_map(
                |(initial_size, num_rounds, mode)| {
                    let num_rounds =
                        num_rounds.min(initial_size.next_power_of_two().trailing_zeros() as usize);
                    Self::new(
                        initial_size,
                        proof_of_work::Config::none(),
                        num_rounds,
                        mode,
                        NonZeroUsize::new(2).expect("2 is non-zero"),
                    )
                },
            )
        }
    }

    #[cfg_attr(feature = "tracing", instrument)]
    fn test_config<F>(seed: u64, config: &Config<F>)
    where
        F: Field + Codec<[u8]> + 'static,
        Standard: Distribution<F>,
    {
        // Pseudo-random Instance
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let initial_vector = random_vector(&mut rng, config.initial_size);
        let initial_covector = random_vector(&mut rng, config.initial_size);
        let initial_sum = dot(&initial_vector, &initial_covector);
        let masks = random_vector(&mut rng, config.mask_length() * config.num_rounds);

        // Prover
        let mut vector = initial_vector.clone();
        let mut covector = initial_covector.clone();
        let mut sum = initial_sum;
        let mut prover_state = ProverState::new_std(&ds);
        let SumcheckOpening {
            round_challenges: point,
            mask_rlc,
        } = config.prove(
            &mut prover_state,
            &mut vector,
            &mut covector,
            &mut sum,
            &masks,
        );
        assert_eq!(vector.len(), config.final_size());
        assert_eq!(covector.len(), config.final_size());
        if config.final_size() == 1 {
            assert_eq!(multilinear_extend(&initial_vector, &point), vector[0]);
            assert_eq!(multilinear_extend(&initial_covector, &point), covector[0]);
        } else {
            // TODO: Check correct folding.
        }

        let expected_mask_sum: F =
            chunks_exact_or_empty(&masks, config.mask_length(), config.num_rounds)
                .zip(&point)
                .map(|(m, x)| univariate_evaluate(m, *x))
                .sum();
        assert_eq!(sum, expected_mask_sum + mask_rlc * dot(&vector, &covector));

        let proof = prover_state.proof();

        // Verifier
        let mut verifier_sum = initial_sum;
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let SumcheckOpening {
            round_challenges: verifier_point,
            mask_rlc: verifier_mask_rlc,
        } = config
            .verify(&mut verifier_state, &mut verifier_sum)
            .unwrap();
        assert_eq!(verifier_point, point);
        assert_eq!(verifier_mask_rlc, mask_rlc);
        assert_eq!(verifier_sum, sum);
        verifier_state.check_eof().unwrap();

        // Standard path: mask_rlc defaults to ONE (no combination randomness sampled).
        if matches!(config.mode, SumcheckMode::Standard) || config.num_rounds == 0 {
            assert_eq!(mask_rlc, F::ONE);
        }
    }

    fn test<F: Field + Codec<[u8]> + 'static>()
    where
        Standard: Distribution<F>,
    {
        crate::tests::init();
        proptest!(|(seed: u64, config in Config::arbitrary())| {
            test_config(seed, &config);
        });
    }

    #[test]
    fn test_single_round() {
        test_config(
            0,
            &Config::<Field64>::new(
                2,
                proof_of_work::Config::none(),
                1,
                SumcheckMode::ZeroKnowledge {
                    mask_length: SumcheckMaskLen::new(3),
                },
                NonZeroUsize::new(2).expect("2 is non-zero"),
            ),
        );
    }

    #[test]
    fn test_two_rounds() {
        test_config(
            0,
            &Config::<Field64>::new(
                3,
                proof_of_work::Config::none(),
                2,
                SumcheckMode::ZeroKnowledge {
                    mask_length: SumcheckMaskLen::new(3),
                },
                NonZeroUsize::new(2).expect("2 is non-zero"),
            ),
        );
    }

    #[test]
    fn test_three_rounds() {
        test_config(
            0,
            &Config::<Field64>::new(
                5,
                proof_of_work::Config::none(),
                3,
                SumcheckMode::ZeroKnowledge {
                    mask_length: SumcheckMaskLen::new(3),
                },
                NonZeroUsize::new(2).expect("2 is non-zero"),
            ),
        );
    }

    #[test]
    fn test_field64_1() {
        test::<fields::Field64>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_2() {
        test::<fields::Field64_2>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_3() {
        test::<fields::Field64_3>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        test::<fields::Field128>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field192() {
        test::<fields::Field192>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        test::<fields::Field256>();
    }
}
