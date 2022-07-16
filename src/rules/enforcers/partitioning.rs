use crate::operators::relational::join::{get_non_empty_join_columns_pair, JoinCondition};
use crate::operators::relational::physical::exchanger::Exchanger;
use crate::operators::relational::physical::{
    HashAggregate, HashJoin, MergeSortJoin, PhysicalExpr, StreamingAggregate,
};
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarExpr;
use crate::properties::partitioning::Partitioning;
use crate::rules::enforcers::PhysicalProperty;
use crate::rules::DerivePropertyMode;

impl PhysicalProperty for Partitioning {
    fn derive_from_expr(&self, expr: &PhysicalExpr) -> DerivePropertyMode {
        use DerivePropertyMode::*;
        match expr {
            PhysicalExpr::Projection(_) => PropertyIsRetained,
            PhysicalExpr::Select(_) => PropertyIsRetained,
            PhysicalExpr::HashAggregate(HashAggregate { group_exprs, .. }) => {
                let grouping_columns: Vec<_> = group_exprs
                    .iter()
                    .filter_map(|e| match e.expr() {
                        ScalarExpr::Column(id) => Some(*id),
                        _ => None,
                    })
                    .collect();
                let hash_partitioning = Partitioning::HashPartitioning(grouping_columns);
                if self.subset_of(&hash_partitioning) {
                    PropertyIsRetained
                } else {
                    // ApplyNewEnforcer(RequiredProperties::new_with_partitioning(hash_partitioning))
                    unimplemented!()
                }
            }
            PhysicalExpr::HashJoin(HashJoin {
                left, right, condition, ..
            }) => {
                // Hash(a1, a2)
                // left: a1, a2
                // right: b1, b2
                // condition: a1 = b1 and b1 = b2

                // if join_provides_partitioning(self, left, right, condition) {
                //     PropertyIsProvided
                // } else {
                //     ApplyEnforcer
                // }
                match get_non_empty_join_columns_pair(left, right, condition) {
                    Some((left, right)) => {
                        // Replace the right columns in the join condition with the left columns.

                        // Convert the given partitioning into a new partitioning where all columns from the
                        // right side of the join are replaced with the columns from the left side of
                        // the join. Also normalize the resulting partitioning.
                        let normalized = self.with_mapping(&right, &left).canonical_form();
                        // This partitioning can be provided by the join operator iff
                        // this partitioning is a subset of the partitioning by which data returned
                        // the join is partitioned.

                        match self {
                            Partitioning::Singleton => PropertyIsRetained,
                            Partitioning::Partitioned(_) => {
                                unreachable!("HashJoin: Any partitioning scheme should have been replaced")
                            }
                            Partitioning::OrderedPartitioning(_) => ApplyEnforcer,
                            Partitioning::HashPartitioning(_) => {
                                if self.subset_of(&normalized) {
                                    PropertyIsRetained
                                } else {
                                    ApplyEnforcer
                                }
                            }
                        }
                    }
                    None => {
                        unreachable!("HashJoin: Invalid join condition: {:?}", condition)
                    }
                }
            }
            PhysicalExpr::MergeSortJoin(MergeSortJoin {
                left, right, condition, ..
            }) => {
                if join_provides_partitioning(self, left, right, condition) {
                    PropertyIsRetained
                } else {
                    ApplyEnforcer
                }
            }
            PhysicalExpr::NestedLoopJoin(_) => ApplyEnforcer,
            PhysicalExpr::StreamingAggregate(StreamingAggregate { ordering, .. }) => {
                let partitioning = Partitioning::OrderedPartitioning(ordering.clone().into_columns());
                let p = self.clone().canonical_form();
                if p.subset_of(&partitioning) {
                    PropertyIsRetained
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
