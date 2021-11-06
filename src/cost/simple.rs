use std::fmt::{Debug, Formatter};

use crate::cost::{Cost, CostEstimationContext, CostEstimator};
use crate::operators::physical::PhysicalExpr;
use crate::properties::statistics::Statistics;

/// A very simple implementation of a [CostEstimator].
///
/// [CostEstimator]: crate::cost::CostEstimator
pub struct SimpleCostEstimator {
    _private: (),
}

impl SimpleCostEstimator {
    pub fn new() -> Self {
        SimpleCostEstimator { _private: () }
    }
}

impl CostEstimator for SimpleCostEstimator {
    fn estimate_cost(&self, expr: &PhysicalExpr, ctx: &CostEstimationContext, statistics: Option<&Statistics>) -> Cost {
        match expr {
            PhysicalExpr::Projection { .. } => ctx.input_cost(0),
            PhysicalExpr::Select { .. } => {
                let input_cost = ctx.input_cost(0);
                let input_stats = ctx.input_statistics(0).unwrap();
                let row_count = input_stats.row_count();
                let filter_cost = (row_count / 10.0) as usize;
                filter_cost + input_cost
            }
            PhysicalExpr::HashAggregate {
                aggr_exprs,
                group_exprs,
                ..
            } => {
                let input_cost = ctx.input_cost(0);
                let input_stats = ctx.input_statistics(0).unwrap();
                let row_count = input_stats.row_count() as usize;

                input_cost + aggr_exprs.len() * row_count + group_exprs.len() * row_count
            }
            PhysicalExpr::HashJoin { .. } => {
                let left_cost = ctx.input_cost(0);
                let left_stats = ctx.input_statistics(0).unwrap();
                let right_cost = ctx.input_cost(1);
                let right_stats = ctx.input_statistics(1).unwrap();

                let left_rows = left_stats.row_count();
                let right_rows = right_stats.row_count();
                let hashtable_access = (left_rows as f64 * 0.01) as usize;

                hashtable_access + (left_rows as usize) + (right_rows as usize) + (left_cost + right_cost)
            }
            PhysicalExpr::MergeSortJoin { .. } => {
                let left_cost = ctx.input_cost(0);
                let left_stats = ctx.input_statistics(0).unwrap();
                let right_cost = ctx.input_cost(1);
                let right_stats = ctx.input_statistics(1).unwrap();

                let left_rows = left_stats.row_count() as usize;
                let right_rows = right_stats.row_count() as usize;

                left_cost + right_cost + left_rows + right_rows
            }
            PhysicalExpr::Scan { .. } => {
                let row_count = statistics.unwrap().row_count();
                row_count as usize
            }
            PhysicalExpr::IndexScan { .. } => {
                let row_count = statistics.unwrap().row_count();
                (row_count / 2.0) as usize
            }
            PhysicalExpr::Sort { .. } => {
                let input_cost = ctx.input_cost(0);
                let input_stats = ctx.input_statistics(0).unwrap();
                let row_count = input_stats.row_count();

                input_cost + (row_count.ln() * row_count) as usize
            }
            PhysicalExpr::Expr { .. } => 1,
        }
    }
}

impl Debug for SimpleCostEstimator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SimpleCostEstimator")
    }
}
