//! Cost-model.

use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::{ExprRef, Properties};
use crate::statistics::Statistics;

pub mod simple;

/// The cost of a plan.
//FIXME Wrap into struct to prevent overflows?
pub type Cost = f64;

/// Additional cost per operator.
/// This extra cost is added to each an operator by the optimizer so a plan with more operator
/// has a higher cost than a plan with less operators.
pub const COST_PER_OPERATOR: f64 = 1f64;

/// Estimates a cost of a physical expression.
pub trait CostEstimator {
    /// Estimates the cost of the given physical expression excluding the cost of its inputs.
    fn estimate_cost(&self, expr: &PhysicalExpr, ctx: &CostEstimationContext, statistics: Option<&Statistics>) -> Cost;
}

/// Provides information that can be used to estimate a cost of an expression.
#[derive(Debug)]
pub struct CostEstimationContext {
    pub(crate) inputs: Vec<ExprRef>,
}

impl CostEstimationContext {
    /// Returns statistics of the i-th child expression.
    /// Because scalar expressions do not have independent statistics this method returns `None`
    /// if the `i`-th expression is a scalar expression. See [Statistics].
    pub fn child_statistics(&self, i: usize) -> Option<&Statistics> {
        let best_expr = &self.inputs[i];
        match best_expr.props() {
            Properties::Relational(props) => props.logical.statistics(),
            Properties::Scalar(_) => None,
        }
    }
}
