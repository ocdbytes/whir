use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, Rng, RngCore};

/// A point `(x_1, ..., x_n)` in `F^n` for some field `F`.
///
/// Often, `x_i` are binary. If strictly binary, `BinaryHypercubePoint` is used.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct MultilinearPoint<F>(pub Vec<F>);

impl<F> MultilinearPoint<F>
where
    F: Field,
{
    /// Returns the number of variables (dimension `n`).
    #[inline]
    pub const fn num_variables(&self) -> usize {
        self.0.len()
    }
}

impl<F> MultilinearPoint<F>
where
    Standard: Distribution<F>,
{
    pub fn rand(rng: &mut impl RngCore, num_variables: usize) -> Self {
        Self((0..num_variables).map(|_| rng.gen()).collect())
    }
}

impl<F> From<F> for MultilinearPoint<F> {
    fn from(value: F) -> Self {
        Self(vec![value])
    }
}

#[cfg(test)]
#[allow(
    clippy::identity_op,
    clippy::cast_sign_loss,
    clippy::erasing_op,
    clippy::should_panic_without_expect
)]
mod tests {
    use ark_std::rand::thread_rng;

    use super::*;
    use crate::algebra::fields::Field64;

    #[test]
    fn test_n_variables() {
        let point =
            MultilinearPoint::<Field64>(vec![Field64::from(1), Field64::from(0), Field64::from(1)]);
        assert_eq!(point.num_variables(), 3);
    }

    #[test]
    fn test_multilinear_point_rand_not_all_same() {
        const K: usize = 20; // Number of trials
        const N: usize = 10; // Number of variables

        let mut rng = thread_rng();

        let mut all_same_count = 0;

        for _ in 0..K {
            let point = MultilinearPoint::<Field64>::rand(&mut rng, N);
            let first = point.0[0];

            // Check if all coordinates are the same as the first one
            if point.0.iter().all(|&x| x == first) {
                all_same_count += 1;
            }
        }

        // If all K trials are completely uniform, the RNG is suspicious
        assert!(
            all_same_count < K,
            "rand generated uniform points in all {K} trials"
        );
    }
}
