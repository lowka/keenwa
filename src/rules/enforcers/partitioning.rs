use crate::operators::relational::join::{get_non_empty_join_columns_pair, JoinCondition};
use crate::operators::relational::physical::exchanger::Exchanger;
use crate::operators::relational::physical::{HashJoin, MergeSortJoin, PhysicalExpr, StreamingAggregate};
use crate::operators::relational::RelNode;
use crate::properties::partitioning::Partitioning;
use crate::rules::enforcers::PhysicalProperty;
use crate::rules::DerivePropertyMode;

impl PhysicalProperty for Partitioning {
    fn create_enforcer_expr(&self, input: RelNode) -> PhysicalExpr {
        PhysicalExpr::Exchanger(Exchanger {
            input,
            partitioning: self.clone(),
        })
    }

    fn derive_from_expr(&self, expr: &PhysicalExpr) -> DerivePropertyMode {
        use DerivePropertyMode::*;
        match expr {
            PhysicalExpr::Projection(_) => PropertyIsRetained,
            PhysicalExpr::Select(_) => PropertyIsRetained,
            PhysicalExpr::HashAggregate(_) => ApplyEnforcer,
            PhysicalExpr::HashJoin(HashJoin {
                left, right, condition, ..
            }) => {
                if join_provides_partitioning(self, left, right, condition) {
                    PropertyIsProvided
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::MergeSortJoin(MergeSortJoin {
                left, right, condition, ..
            }) => {
                if join_provides_partitioning(self, left, right, condition) {
                    PropertyIsProvided
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::NestedLoopJoin(_) => ApplyEnforcer,
            PhysicalExpr::StreamingAggregate(StreamingAggregate { ordering, .. }) => {
                let partitioning = Partitioning::Partitioned(ordering.clone().into_columns());
                if self.subset_of(&partitioning) {
                    PropertyIsProvided
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::Scan(_) => ApplyEnforcer,
            PhysicalExpr::IndexScan(_) => ApplyEnforcer,
            PhysicalExpr::Sort(_) => ApplyEnforcer,
            PhysicalExpr::Append(_) => ApplyEnforcer,
            PhysicalExpr::Unique(_) => ApplyEnforcer,
            PhysicalExpr::HashedSetOp(_) => ApplyEnforcer,
            PhysicalExpr::Limit(_) => ApplyEnforcer,
            PhysicalExpr::Offset(_) => ApplyEnforcer,
            PhysicalExpr::Values(_) => ApplyEnforcer,
            PhysicalExpr::Empty(_) => ApplyEnforcer,
            PhysicalExpr::Exchanger(Exchanger { partitioning, .. }) => {
                if self.subset_of(partitioning) {
                    PropertyIsProvided
                } else {
                    ApplyEnforcer
                }
            }
        }
    }
}

fn join_provides_partitioning(
    partitioning: &Partitioning,
    left: &RelNode,
    right: &RelNode,
    condition: &JoinCondition,
) -> bool {
    get_non_empty_join_columns_pair(left, right, condition)
        .map(|(left, right)| {
            // Convert the given partitioning into a new partitioning where all columns from the
            // right side of the join are replaced with the columns from the left side of
            // the join. Also normalize the resulting partitioning.
            let normalized = partitioning.with_mapping(&right, &left).normalize();
            // This partitioning can be provided by the join operator iff
            // this partitioning is a subset of the partitioning by which data returned
            // the join is partitioned.
            partitioning.subset_of(&normalized)
        })
        .unwrap_or_default()
}
