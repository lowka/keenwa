//! Cost-model.

use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::{ExprRef, Properties};
use crate::statistics::Statistics;

pub mod simple;

//FIXME replace usize with f64
//FIXME Wrap into struct to prevent overflows.
pub type Cost = usize;

/// Estimates a cost of a physical expression.
pub trait CostEstimator {
    /// Estimates the cost of the given physical expression.
    fn estimate_cost(&self, expr: &PhysicalExpr, ctx: &CostEstimationContext, statistics: Option<&Statistics>) -> Cost;
}

/// Provides information that can be used to estimate a cost of an expression.
#[derive(Debug)]
pub struct CostEstimationContext {
    pub(crate) inputs: Vec<ExprRef>,
}

impl CostEstimationContext {
    /// Returns statistics of the i-th child expression.
    pub fn child_statistics(&self, i: usize) -> Option<&Statistics> {
        let best_expr = &self.inputs[i];
        match best_expr.props() {
            Properties::Relational(props) => props.logical.statistics(),
            Properties::Scalar(_) => None,
        }
    }
}
