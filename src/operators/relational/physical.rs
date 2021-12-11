use crate::memo::{ExprContext, MemoExprFormatter};
use crate::meta::ColumnId;
use crate::operators::relational::join::{get_join_columns_pair, JoinCondition, JoinOn};
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn, OperatorInputs};
use crate::properties::physical::PhysicalProperties;
use crate::properties::OrderingChoice;

// TODO: Docs
/// A physical expression represents an algorithm that can be used to implement a [logical expression].
///
/// [logical expression]: crate::operators::relational::logical::LogicalExpr
#[derive(Debug, Clone)]
pub enum PhysicalExpr {
    Projection {
        input: RelNode,
        columns: Vec<ColumnId>,
    },
    Select {
        input: RelNode,
        filter: Option<ScalarNode>,
    },
    HashAggregate {
        input: RelNode,
        aggr_exprs: Vec<ScalarNode>,
        group_exprs: Vec<ScalarNode>,
    },
    HashJoin {
        left: RelNode,
        right: RelNode,
        condition: JoinCondition,
    },
    MergeSortJoin {
        left: RelNode,
        right: RelNode,
        condition: JoinCondition,
    },
    NestedLoop {
        left: RelNode,
        right: RelNode,
        condition: Option<ScalarNode>,
    },
    Scan {
        source: String,
        columns: Vec<ColumnId>,
    },
    IndexScan {
        source: String,
        columns: Vec<ColumnId>,
    },
    Sort {
        input: RelNode,
        ordering: OrderingChoice,
    },
    Append {
        left: RelNode,
        right: RelNode,
    },
    Unique {
        left: RelNode,
        right: RelNode,
    },
    HashedSetOp {
        left: RelNode,
        right: RelNode,
        /// Whether this is an INTERSECT or EXCEPT operator.
        intersect: bool,
        /// If `true` this an INTERSECT ALL/EXCEPT ALL operator.
        all: bool,
    },
}

impl PhysicalExpr {
    pub(crate) fn traverse<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        match self {
            PhysicalExpr::Projection { input, .. } => {
                visitor.visit_rel(expr_ctx, input);
            }
            PhysicalExpr::Select { input, filter } => {
                visitor.visit_rel(expr_ctx, input);
                visitor.visit_opt_scalar(expr_ctx, filter.as_ref());
            }
            PhysicalExpr::HashAggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                visitor.visit_rel(expr_ctx, input);
                for aggr_expr in aggr_exprs {
                    visitor.visit_scalar(expr_ctx, aggr_expr);
                }
                for group_expr in group_exprs {
                    visitor.visit_scalar(expr_ctx, group_expr)
                }
            }
            PhysicalExpr::HashJoin { left, right, condition } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
                visitor.visit_join_condition(expr_ctx, condition);
            }
            PhysicalExpr::MergeSortJoin { left, right, condition } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
                visitor.visit_join_condition(expr_ctx, condition);
            }
            PhysicalExpr::NestedLoop { left, right, condition } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
                visitor.visit_opt_scalar(expr_ctx, condition.as_ref())
            }
            PhysicalExpr::Scan { .. } => {}
            PhysicalExpr::IndexScan { .. } => {}
            PhysicalExpr::Sort { input, .. } => {
                visitor.visit_rel(expr_ctx, input);
            }
            PhysicalExpr::Unique { left, right, .. } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
            }
            PhysicalExpr::Append { left, right } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
            }
            PhysicalExpr::HashedSetOp { left, right, .. } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
            }
        }
    }

    pub fn with_new_inputs(&self, inputs: &mut OperatorInputs) -> Self {
        match self {
            PhysicalExpr::Projection { columns, .. } => {
                inputs.expect_len(1, "Projection");
                PhysicalExpr::Projection {
                    input: inputs.rel_node(),
                    columns: columns.clone(),
                }
            }
            PhysicalExpr::Select { filter, .. } => {
                let num_opt = filter.as_ref().map(|_| 1).unwrap_or_default();
                inputs.expect_len(1 + num_opt, "Select");
                PhysicalExpr::Select {
                    input: inputs.rel_node(),
                    filter: filter.as_ref().map(|_| inputs.scalar_node()),
                }
            }
            PhysicalExpr::HashAggregate {
                aggr_exprs,
                group_exprs,
                ..
            } => {
                inputs.expect_len(1 + aggr_exprs.len() + group_exprs.len(), "HashAggregate");
                PhysicalExpr::HashAggregate {
                    input: inputs.rel_node(),
                    aggr_exprs: inputs.scalar_nodes(aggr_exprs.len()),
                    group_exprs: inputs.scalar_nodes(group_exprs.len()),
                }
            }
            PhysicalExpr::HashJoin { condition, .. } => {
                let num_opt = match condition {
                    JoinCondition::Using(_) => 0,
                    JoinCondition::On(_) => 1,
                };
                inputs.expect_len(2 + num_opt, "HashJoin");
                PhysicalExpr::HashJoin {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    condition: match condition {
                        JoinCondition::Using(_) => condition.clone(),
                        JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node())),
                    },
                }
            }
            PhysicalExpr::MergeSortJoin { condition, .. } => {
                let num_opt = match condition {
                    JoinCondition::Using(_) => 0,
                    JoinCondition::On(_) => 1,
                };
                inputs.expect_len(2 + num_opt, "MergeSortJoin");
                PhysicalExpr::MergeSortJoin {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    condition: match condition {
                        JoinCondition::Using(_) => condition.clone(),
                        JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node())),
                    },
                }
            }
            PhysicalExpr::NestedLoop { condition, .. } => {
                let num_opt = condition.as_ref().map(|_| 1).unwrap_or_default();
                inputs.expect_len(2 + num_opt, "NestedLoop");
                PhysicalExpr::NestedLoop {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    condition: condition.as_ref().map(|_| inputs.scalar_node()),
                }
            }
            PhysicalExpr::Scan { source, columns } => {
                inputs.expect_len(0, "Scan");
                PhysicalExpr::Scan {
                    source: source.clone(),
                    columns: columns.clone(),
                }
            }
            PhysicalExpr::IndexScan { source, columns } => {
                inputs.expect_len(0, "IndexScan");
                PhysicalExpr::IndexScan {
                    source: source.clone(),
                    columns: columns.clone(),
                }
            }
            PhysicalExpr::Sort { ordering, .. } => {
                inputs.expect_len(1, "Sort");
                PhysicalExpr::Sort {
                    input: inputs.rel_node(),
                    ordering: ordering.clone(),
                }
            }
            PhysicalExpr::Unique { .. } => {
                inputs.expect_len(2, "Unique");
                PhysicalExpr::Unique {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                }
            }
            PhysicalExpr::Append { .. } => {
                inputs.expect_len(2, "Append");
                PhysicalExpr::Append {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                }
            }
            PhysicalExpr::HashedSetOp { intersect, all, .. } => {
                inputs.expect_len(2, "SortedSetOp");
                PhysicalExpr::HashedSetOp {
                    left: inputs.rel_node(),
                    intersect: *intersect,
                    right: inputs.rel_node(),
                    all: *all,
                }
            }
        }
    }

    pub fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        match self {
            PhysicalExpr::Projection { .. } => None,
            PhysicalExpr::Select { .. } => None,
            PhysicalExpr::HashJoin { .. } => None,
            PhysicalExpr::MergeSortJoin {
                left, right, condition, ..
            } => match get_join_columns_pair(left, right, condition) {
                Some((left, right)) if !left.is_empty() && !right.is_empty() => {
                    let left_ordering = PhysicalProperties::new(OrderingChoice::new(left));
                    let right_ordering = PhysicalProperties::new(OrderingChoice::new(right));

                    let requirements = vec![left_ordering, right_ordering];
                    Some(requirements)
                }
                _ => None,
            },
            PhysicalExpr::NestedLoop { .. } => None,
            PhysicalExpr::Scan { .. } => None,
            PhysicalExpr::IndexScan { .. } => None,
            PhysicalExpr::Sort { .. } => None,
            PhysicalExpr::HashAggregate { .. } => None,
            PhysicalExpr::Unique { left, right, .. } | PhysicalExpr::HashedSetOp { left, right, .. } => {
                let left = left.props().logical().output_columns().iter().copied().collect();
                let right = right.props().logical().output_columns().iter().copied().collect();
                let left_ordering = PhysicalProperties::new(OrderingChoice::new(left));
                let right_ordering = PhysicalProperties::new(OrderingChoice::new(right));

                let requirements = vec![left_ordering, right_ordering];
                Some(requirements)
            }
            PhysicalExpr::Append { .. } => None,
        }
    }

    pub(crate) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            PhysicalExpr::Projection { input, columns } => {
                f.write_name("Projection");
                f.write_expr("input", input);
                f.write_values("cols", columns)
            }
            PhysicalExpr::Select { input, filter } => {
                f.write_name("Select");
                f.write_expr("input", input);
                f.write_expr_if_present("filter", filter.as_ref())
            }
            PhysicalExpr::HashJoin { left, right, condition } => {
                f.write_name("HashJoin");
                f.write_expr("left", left);
                f.write_expr("right", right);
                match condition {
                    JoinCondition::Using(using) => f.write_value("using", using),
                    JoinCondition::On(on) => f.write_value("on", on),
                };
            }
            PhysicalExpr::MergeSortJoin { left, right, condition } => {
                f.write_name("MergeSortJoin");
                f.write_expr("left", left);
                f.write_expr("right", right);
                match condition {
                    JoinCondition::Using(using) => f.write_value("using", using),
                    JoinCondition::On(on) => f.write_value("on", on),
                }
            }
            PhysicalExpr::NestedLoop { left, right, condition } => {
                f.write_name("NestedLoopJoin");
                f.write_expr("left", left);
                f.write_expr("right", right);
                if let Some(condition) = condition {
                    f.write_expr("condition", condition);
                }
            }
            PhysicalExpr::Scan { source, columns } => {
                f.write_name("Scan");
                f.write_source(source);
                f.write_values("cols", columns)
            }
            PhysicalExpr::IndexScan { source, columns } => {
                f.write_name("IndexScan");
                f.write_source(source);
                f.write_values("cols", columns)
            }
            PhysicalExpr::Sort { input, ordering } => {
                f.write_name("Sort");
                f.write_expr("input", input);
                f.write_value("ord", format!("{:?}", ordering.columns()).as_str())
            }
            PhysicalExpr::HashAggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                f.write_name("HashAggregate");
                f.write_expr("input", input);
                for aggr_expr in aggr_exprs {
                    f.write_expr("", aggr_expr);
                }
                for group_expr in group_exprs {
                    f.write_expr("", group_expr);
                }
            }
            PhysicalExpr::Unique { left, right } => {
                f.write_name("Unique");
                f.write_expr("left", left);
                f.write_expr("right", right);
            }
            PhysicalExpr::Append { left, right } => {
                f.write_name("Append");
                f.write_expr("left", left);
                f.write_expr("right", right);
            }
            PhysicalExpr::HashedSetOp {
                left,
                right,
                intersect,
                all,
            } => {
                f.write_name("HashedSetOp");
                f.write_expr("left", left);
                f.write_expr("right", right);
                f.write_value("intersect", intersect);
                f.write_value("all", all);
            }
        }
    }
}