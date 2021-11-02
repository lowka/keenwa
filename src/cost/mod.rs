use std::fmt::Debug;

use crate::operators::physical::PhysicalExpr;
use crate::operators::GroupRef;
use crate::properties::statistics::Statistics;

pub mod simple;

//FIXME replace usize f64
//FIXME Wrap into struct to prevent from overflows.
pub type Cost = usize;

/// Estimates a cost of a physical expression.
pub trait CostEstimator: Debug {
    /// Estimates the cost of the given physical expression.
    fn estimate_cost(&self, expr: &PhysicalExpr, ctx: &CostEstimationContext, statistics: Option<&Statistics>) -> Cost;
}

/// Provides information that can be used to estimate a cost of an expression.
#[derive(Debug)]
pub struct CostEstimationContext {
    pub(crate) input_costs: Vec<usize>,
    pub(crate) input_groups: Vec<GroupRef>,
}

impl CostEstimationContext {
    /// Returns the cost of the i-th input expression.
    pub fn input_cost(&self, i: usize) -> usize {
        self.input_costs[i]
    }

    /// Returns statistics of the i-th input expression.
    pub fn input_statistics(&self, i: usize) -> Option<&Statistics> {
        let group = &self.input_groups[i];
        group.attrs().logical().statistics()
    }
}
