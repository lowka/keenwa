//! A basic implementation of the optimizer's cost-model.

use std::fmt::{Debug, Formatter};

use crate::cost::{Cost, CostEstimationContext, CostEstimator};
use crate::operators::relational::physical::exchanger::Exchanger;
use crate::operators::relational::physical::{
    HashAggregate, HashedSetOp, Limit, Offset, PhysicalExpr, StreamingAggregate, Unique, Values,
};
use crate::properties::partitioning::Partitioning;
use crate::statistics::Statistics;

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
            PhysicalExpr::Projection(_) => 1.0,
            PhysicalExpr::Select(_) => {
                let input_stats = ctx.child_statistics(0).unwrap();
                let row_count = input_stats.row_count();
                let selectivity = statistics.map(|s| s.selectivity()).unwrap_or(Statistics::DEFAULT_SELECTIVITY);

                row_count * selectivity
            }
            PhysicalExpr::HashAggregate(HashAggregate {
                aggr_exprs,
                group_exprs,
                ..
            }) => {
                let input_stats = ctx.child_statistics(0).unwrap();
                let row_count = input_stats.row_count();

                aggr_exprs.len() as f64 * row_count + group_exprs.len() as f64 * row_count
            }
            PhysicalExpr::HashJoin(_) => {
                let left_stats = ctx.child_statistics(0).unwrap();
                let right_stats = ctx.child_statistics(1).unwrap();

                let left_rows = left_stats.row_count();
                let right_rows = right_stats.row_count();
                let hashtable_access = left_rows * 0.01;

                hashtable_access + left_rows + right_rows
            }
            PhysicalExpr::MergeSortJoin(_) => {
                let left_stats = ctx.child_statistics(0).unwrap();
                let right_stats = ctx.child_statistics(1).unwrap();

                let left_rows = left_stats.row_count();
                let right_rows = right_stats.row_count();

                left_rows + right_rows
            }
            PhysicalExpr::NestedLoopJoin(_) => {
                let left_stats = ctx.child_statistics(0).unwrap();
                let right_stats = ctx.child_statistics(1).unwrap();

                let left_rows = left_stats.row_count();
                let right_rows = right_stats.row_count();

                left_rows * right_rows
            }
            PhysicalExpr::StreamingAggregate(StreamingAggregate { aggr_exprs, .. }) => {
                let input_stats = ctx.child_statistics(0).unwrap();
                let row_count = input_stats.row_count();

                aggr_exprs.len() as f64 * row_count
            }
            PhysicalExpr::Scan(_) => statistics.unwrap().row_count(),
            PhysicalExpr::IndexScan(_) => {
                let row_count = statistics.unwrap().row_count();
                row_count / 2.0
            }
            PhysicalExpr::Sort(_) => {
                let input_stats = ctx.child_statistics(0).unwrap();
                let row_count = input_stats.row_count();
                if row_count > 1.0 {
                    row_count.ln() * row_count
                } else {
                    1.0
                }
            }
            PhysicalExpr::Unique(Unique { inputs, .. }) => {
                // Inputs are sorted
                let row_count = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let stats = ctx.child_statistics(i).unwrap();
                        stats.row_count()
                    })
                    .sum();
                // TODO: Reduce the number of output rows by on_expr.

                row_count
            }
            PhysicalExpr::Append(_) => {
                let left_stats = ctx.child_statistics(0).unwrap();
                let right_stats = ctx.child_statistics(1).unwrap();

                left_stats.row_count() + right_stats.row_count()
            }
            PhysicalExpr::HashedSetOp(HashedSetOp { intersect, all, .. }) => {
                let left_stats = ctx.child_statistics(0).unwrap();
                let right_stats = ctx.child_statistics(1).unwrap();

                let (rows, hashtable_cost) = if *intersect {
                    // For INTERSECT a hash table is built from the left input
                    let hashtable_cost = left_stats.row_count();
                    let hashtable_access_cost = hashtable_cost * 0.01;
                    let right_rows = right_stats.row_count();

                    (right_rows, hashtable_access_cost)
                } else {
                    // For EXCEPT a hash table is built from the right input
                    let hashtable_cost = right_stats.row_count();
                    let hashtable_access_cost = hashtable_cost * 0.01;
                    let left_rows = left_stats.row_count();

                    (left_rows, hashtable_access_cost)
                };

                let deduplication_cost = if !*all {
                    // For both INTERSECT and EXCEPT operators an operator requires a hashtable
                    // with at most number_of_rows(left_side) records.
                    left_stats.row_count()
                } else {
                    0.0
                };

                rows + hashtable_cost + deduplication_cost
            }
            PhysicalExpr::Limit(Limit { rows, .. }) => {
                // Limit operator is executed at most `rows` times.
                *rows as f64
            }
            PhysicalExpr::Offset(Offset { rows, .. }) => {
                // The cost of an offset operator is equal to the number of rows it must skip
                // after that it is just a passthrough operator.
                *rows as f64
            }
            PhysicalExpr::Values(Values { values, .. }) => {
                // Values returns exactly len values rows.
                values.len() as f64
            }
            PhysicalExpr::Empty(_) => 0.0,
            PhysicalExpr::Exchanger(Exchanger { partitioning, .. }) => {
                let input_stats = ctx.child_statistics(0).unwrap();
                let input_row_count = input_stats.row_count();

                match partitioning {
                    Partitioning::Singleton => input_row_count,
                    Partitioning::Partitioned(cols)
                    | Partitioning::OrderedPartitioning(cols)
                    | Partitioning::HashPartitioning(cols) => {
                        // let's assume that the cost of partitioning
                        // is a linear function of the number of columns + some overhead.
                        let partitioning_cost = cols.len() as f64;
                        let overhead = 0.01f64;
                        input_row_count * (partitioning_cost + overhead)
                    }
                }
            }
        }
    }
}

impl Debug for SimpleCostEstimator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleCostEstimator").finish()
    }
}
