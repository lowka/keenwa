use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::{GroupRef, Properties};
use crate::statistics::Statistics;

pub mod simple;

//FIXME replace usize f64
//FIXME Wrap into struct to prevent from overflows.
pub type Cost = usize;

/// Estimates a cost of a physical expression.
pub trait CostEstimator {
    /// Estimates the cost of the given physical expression.
    fn estimate_cost(&self, expr: &PhysicalExpr, ctx: &CostEstimationContext, statistics: Option<&Statistics>) -> Cost;
}

/// Provides information that can be used to estimate a cost of an expression.
#[derive(Debug)]
pub struct CostEstimationContext {
    pub(crate) input_groups: Vec<GroupRef>,
}

impl CostEstimationContext {
    /// Returns statistics of the i-th child expression.
    pub fn child_statistics(&self, i: usize) -> Option<&Statistics> {
        let group = &self.input_groups[i];
        match group.props() {
            Properties::Relational(props) => props.logical.statistics(),
            Properties::Scalar(_) => None,
        }
    }
}
