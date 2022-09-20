use crate::meta::ColumnId;
use crate::operators::relational::physical::exchanger::Exchanger;
use crate::operators::relational::physical::{
    IndexScan, MergeSortJoin, PhysicalExpr, Projection, Sort, StreamingAggregate, Unique,
};
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarExpr;
use crate::properties::partitioning::Partitioning;
use crate::properties::OrderingChoice;
use crate::rules::enforcers::PhysicalProperty;
use crate::rules::DerivePropertyMode;

impl PhysicalProperty for OrderingChoice {
    fn derive_from_expr(&self, expr: &PhysicalExpr) -> DerivePropertyMode {
        use DerivePropertyMode::*;
        match expr {
            PhysicalExpr::Projection(Projection { exprs, .. }) => {
                let simple_projection = exprs
                    .iter()
                    .all(|expr| matches!(expr.expr(), ScalarExpr::Scalar(_) | ScalarExpr::Column(_)));
                if simple_projection {
                    PropertyIsRetained
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::Select(_) => PropertyIsRetained,
            PhysicalExpr::HashAggregate(_) => ApplyEnforcer,
            PhysicalExpr::HashJoin(_) => ApplyEnforcer,
            PhysicalExpr::MergeSortJoin(MergeSortJoin {
                left_ordering,
                right_ordering,
                ..
            }) => {
                if join_provides_ordering(self, left_ordering, right_ordering) {
                    PropertyIsProvided
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::NestedLoopJoin(_) => ApplyEnforcer,
            PhysicalExpr::StreamingAggregate(StreamingAggregate {
                ordering: output_ordering,
                ..
            }) => {
                if self.prefix_of(output_ordering) {
                    PropertyIsProvided
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::Scan(_) => ApplyEnforcer,
            PhysicalExpr::IndexScan(IndexScan {
                ordering: index_ordering,
                ..
            }) => {
                let result = index_ordering.as_ref().map(|ord| self.prefix_of(ord));
                if result.unwrap_or_default() {
                    PropertyIsProvided
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::Sort(Sort {
                ordering: sort_ordering,
                ..
            }) => {
                if self.prefix_of(sort_ordering) {
                    PropertyIsProvided
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::Append(_) => ApplyEnforcer,
            PhysicalExpr::Unique(Unique {
                inputs,
                columns: output_columns,
                ordering: output_ordering,
                ..
            }) => {
                if unique_provides_ordering(self, inputs, output_ordering, output_columns) {
                    PropertyIsProvided
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::HashedSetOp(_) => ApplyEnforcer,
            PhysicalExpr::Limit(_) => PropertyIsRetained,
            PhysicalExpr::Offset(_) => PropertyIsRetained,
            PhysicalExpr::Values(_) => ApplyEnforcer,
            PhysicalExpr::Empty(_) => ApplyEnforcer,
            PhysicalExpr::Exchanger(Exchanger { partitioning, .. }) => match partitioning {
                Partitioning::Singleton => ApplyEnforcer,
                Partitioning::Partitioned(_) => ApplyEnforcer,
                // Partitioning::OrderedPartitioning(cols) if cols == &self.clone().into_columns() => PropertyIsProvided,
                Partitioning::OrderedPartitioning(_) => ApplyEnforcer,
                Partitioning::HashPartitioning(_) => ApplyEnforcer,
            },
        }
    }
}

fn join_provides_ordering(
    ordering: &OrderingChoice,
    left_ordering: &OrderingChoice,
    right_ordering: &OrderingChoice,
) -> bool {
    // Convert the required ordering to the ordering where all columns from the right side
    // of the join are replaced with columns from the left side of the join.
    let left_columns: Vec<_> = left_ordering.clone().into_columns();
    let right_columns: Vec<_> = right_ordering.clone().into_columns();

    let normalized = ordering.with_mapping(&right_columns, &left_columns);
    // the resulting ordering must be a prefix of the left ordering.
    normalized.prefix_of(left_ordering)
}

fn unique_provides_ordering(
    ordering: &OrderingChoice,
    inputs: &[RelNode],
    output_ordering: &[OrderingChoice],
    output_columns: &[ColumnId],
) -> bool {
    // All inputs of a unique operator are ordered in the same way
    // and the required ordering uses columns from its output. E.g.:
    //
    // Inputs:
    //  - a1 asc, a2 asc, a3 desc
    //  - b2 asc, b2 asc, b3 desc
    // Output:
    //    o1 asc, o2 asc, o3 desc
    //
    // Ordering:
    //    o1 asc, o2 asc
    //
    // Replace output columns from the required ordering with
    // columns of the first input and check whether the required
    // ordering if the prefix of the resulting one.
    //
    let input = &inputs[0];
    let input_columns = input.props().logical().output_columns();
    let input_ordering = &output_ordering[0];
    let ord = input_ordering.with_mapping(input_columns, output_columns);

    ordering.prefix_of(&ord)
}
