// Backward compatibility layer for ZK-WHIR
// This wraps the new LinearForm trait to provide the old Weights interface

use std::{fmt::Debug, ops::Index};

use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

use super::{
    linear_form::{Evaluate, LinearForm, UnivariateEvaluation},
    polynomials::{CoefficientList, EvaluationsList, MultilinearPoint},
};
use crate::algebra::embedding::Embedding;

// Helper to convert UnivariateEvaluation to Weights
impl<F: Field> From<UnivariateEvaluation<F>> for Weights<F> {
    fn from(eval: UnivariateEvaluation<F>) -> Self {
        Self::univariate(eval.point, eval.size)
    }
}

/// Represents a weight function used in polynomial evaluations.
///
/// A `Weights<F>` instance allows evaluating or accumulating weighted contributions
/// to a multilinear polynomial stored in evaluation form. It supports two modes:
///
/// - Evaluation mode: Represents an equality constraint at a specific `MultilinearPoint<F>`.
/// - Linear mode: Represents a set of per-corner weights stored as `EvaluationsList<F>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Field + CanonicalSerialize + CanonicalDeserialize")]
pub enum Weights<F> {
    /// Represents a weight function that enforces equality constraints at a specific point.
    Evaluation { point: MultilinearPoint<F> },
    /// Represents a weight function defined as a precomputed set of evaluations.
    Linear { weight: EvaluationsList<F> },
}

impl<F: Field> Weights<F> {
    /// Constructs a weight in evaluation mode, enforcing an equality constraint at `point`.
    pub const fn evaluation(point: MultilinearPoint<F>) -> Self {
        Self::Evaluation { point }
    }

    /// Construct weights for a univariate evaluation
    pub fn univariate(point: F, size: usize) -> Self {
        Self::Evaluation {
            point: MultilinearPoint::expand_from_univariate(point, size),
        }
    }

    /// Constructs a weight in linear mode, applying a set of precomputed weights.
    pub const fn linear(weight: EvaluationsList<F>) -> Self {
        Self::Linear { weight }
    }

    /// Returns the number of variables involved in the weight function.
    pub const fn num_variables(&self) -> usize {
        match self {
            Self::Evaluation { point } => point.num_variables(),
            Self::Linear { weight } => weight.num_variables(),
        }
    }

    pub const fn deferred(&self) -> bool {
        matches!(self, Self::Linear { .. })
    }

    pub fn mixed_evaluate<M>(&self, embedding: &M, poly: &CoefficientList<M::Source>) -> M::Target
    where
        M: Embedding<Target = F>,
    {
        assert_eq!(self.num_variables(), poly.num_variables());
        match self {
            Self::Evaluation { point } => poly.mixed_evaluate(embedding, point),
            Self::Linear { weight } => {
                let coeffs_eval = EvaluationsList::from(poly.lift(embedding));
                crate::algebra::dot(weight.evals(), coeffs_eval.evals())
            }
        }
    }

    pub fn evaluate(&self, poly: &CoefficientList<F>) -> F {
        match self {
            Self::Evaluation { point } => poly.evaluate(point),
            Self::Linear { weight } => {
                let evals = EvaluationsList::from(poly.clone());
                crate::algebra::dot(weight.evals(), evals.evals())
            }
        }
    }

    pub fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        match self {
            Self::Evaluation { point } => {
                // Compute eq polynomial evaluations and accumulate
                let eq_evals = crate::algebra::polynomials::compute_eq_evals(&point.0);
                for (acc, eq_val) in accumulator.iter_mut().zip(eq_evals.iter()) {
                    *acc += scalar * eq_val;
                }
            }
            Self::Linear { weight } => {
                for (acc, w) in accumulator.iter_mut().zip(weight.evals().iter()) {
                    *acc += scalar * w;
                }
            }
        }
    }

    pub fn compute(&self, point: &MultilinearPoint<F>) -> F {
        match self {
            Self::Evaluation { point: eval_point } => {
                let eq_evals = crate::algebra::polynomials::compute_eq_evals(&eval_point.0);
                let lagrange_evals = crate::algebra::polynomials::compute_eq_evals(&point.0);
                crate::algebra::dot(&eq_evals, &lagrange_evals)
            }
            Self::Linear { weight } => weight.eval_extension(point),
        }
    }
}

// Implement LinearForm for Weights<F>
impl<F: Field> LinearForm<F> for Weights<F> {
    fn size(&self) -> usize {
        1 << self.num_variables()
    }

    fn deferred(&self) -> bool {
        self.deferred()
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        self.compute(&MultilinearPoint(point.to_vec()))
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        self.accumulate(accumulator, scalar)
    }
}

// Implement Evaluate for Weights<F>
impl<F: Field, M: Embedding<Target = F>> Evaluate<M> for Weights<F> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        let poly = CoefficientList::new(vector.to_vec());
        self.mixed_evaluate(embedding, &poly)
    }
}
