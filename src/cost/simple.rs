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
            PhysicalExpr::Projection { .. } => 1,
            PhysicalExpr::Select { .. } => {
                let input_stats = ctx.input_statistics(0).unwrap();
                let row_count = input_stats.row_count();

                (row_count / 10.0) as usize
            }
            PhysicalExpr::HashAggregate {
                aggr_exprs,
                group_exprs,
                ..
            } => {
                let input_stats = ctx.input_statistics(0).unwrap();
                let row_count = input_stats.row_count() as usize;

                aggr_exprs.len() * row_count + group_exprs.len() * row_count
            }
            PhysicalExpr::HashJoin { .. } => {
                let left_stats = ctx.input_statistics(0).unwrap();
                let right_stats = ctx.input_statistics(1).unwrap();

                let left_rows = left_stats.row_count();
                let right_rows = right_stats.row_count();
                let hashtable_access = (left_rows as f64 * 0.01) as usize;

                hashtable_access + (left_rows as usize) + (right_rows as usize)
            }
            PhysicalExpr::MergeSortJoin { .. } => {
                let left_stats = ctx.input_statistics(0).unwrap();
                let right_stats = ctx.input_statistics(1).unwrap();

                let left_rows = left_stats.row_count() as usize;
                let right_rows = right_stats.row_count() as usize;

                left_rows + right_rows
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
                let input_stats = ctx.input_statistics(0).unwrap();
                let row_count = input_stats.row_count();

                (row_count.ln() * row_count) as usize
            }
        }
    }
}

impl Debug for SimpleCostEstimator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SimpleCostEstimator")
    }
}
