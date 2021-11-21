use crate::memo::{ExprContext, MemoExprFormatter};
use crate::meta::ColumnId;
use crate::operators::join::JoinCondition;
use crate::operators::{Operator, OperatorCopyIn, OperatorInputs, RelNode, ScalarNode};
use crate::properties::physical::PhysicalProperties;
use crate::properties::OrderingChoice;

// TODO: Docs
/// A physical expression represents an algorithm that can be used to implement a [logical expression].
///
/// [logical expression]: crate::operators::logical::LogicalExpr
#[derive(Debug, Clone)]
pub enum PhysicalExpr {
    Projection {
        input: RelNode,
        columns: Vec<ColumnId>,
    },
    Select {
        input: RelNode,
        filter: ScalarNode,
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
}

impl PhysicalExpr {
    pub(crate) fn traverse(&self, visitor: &mut OperatorCopyIn, expr_ctx: &mut ExprContext<Operator>) {
        match self {
            PhysicalExpr::Projection { input, .. } => {
                visitor.visit_rel(expr_ctx, input);
            }
            PhysicalExpr::Select { input, filter } => {
                visitor.visit_rel(expr_ctx, input);
                visitor.visit_scalar(expr_ctx, filter)
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
            PhysicalExpr::HashJoin { left, right, .. } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
            }
            PhysicalExpr::MergeSortJoin { left, right, .. } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
            }
            PhysicalExpr::Scan { .. } => {}
            PhysicalExpr::IndexScan { .. } => {}
            PhysicalExpr::Sort { input, .. } => {
                visitor.visit_rel(expr_ctx, input);
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
            PhysicalExpr::Select { .. } => {
                inputs.expect_len(2, "Select");
                PhysicalExpr::Select {
                    input: inputs.rel_node(),
                    filter: inputs.scalar_node(),
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
                inputs.expect_len(2, "HashJoin");
                PhysicalExpr::HashJoin {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    condition: condition.clone(),
                }
            }
            PhysicalExpr::MergeSortJoin { condition, .. } => {
                inputs.expect_len(2, "MergeSortJoin");
                PhysicalExpr::MergeSortJoin {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    condition: condition.clone(),
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
        }
    }

    pub fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        match self {
            PhysicalExpr::Projection { .. } => None,
            PhysicalExpr::Select { .. } => None,
            PhysicalExpr::HashJoin { .. } => None,
            PhysicalExpr::MergeSortJoin { condition, .. } => {
                let (left, right) = match condition {
                    JoinCondition::Using(using) => using.as_columns_pair(),
                };
                let left_ordering = PhysicalProperties::new(OrderingChoice::new(left));
                let right_ordering = PhysicalProperties::new(OrderingChoice::new(right));

                let requirements = vec![left_ordering, right_ordering];
                Some(requirements)
            }
            PhysicalExpr::Scan { .. } => None,
            PhysicalExpr::IndexScan { .. } => None,
            PhysicalExpr::Sort { .. } => None,
            PhysicalExpr::HashAggregate { .. } => None,
        }
    }

    pub(crate) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            PhysicalExpr::Projection { input, columns } => {
                f.write_name("Projection");
                f.write_input("input", input);
                f.write_values("cols", columns)
            }
            PhysicalExpr::Select { input, filter } => {
                f.write_name("Select");
                f.write_input("input", input);
                f.write_input("filter", filter)
            }
            PhysicalExpr::HashJoin { left, right, condition } => {
                f.write_name("HashJoin");
                f.write_input("left", left);
                f.write_input("right", right);
                match condition {
                    JoinCondition::Using(using) => f.write_value("using", format!("{}", using).as_str()),
                };
            }
            PhysicalExpr::MergeSortJoin { left, right, condition } => {
                f.write_name("MergeSortJoin");
                f.write_input("left", left);
                f.write_input("right", right);
                match condition {
                    JoinCondition::Using(using) => f.write_value("using", format!("{}", using).as_str()),
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
                f.write_input("input", input);
                f.write_value("ord", format!("{:?}", ordering.columns()).as_str())
            }
            PhysicalExpr::HashAggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                f.write_name("HashAggregate");
                f.write_input("input", input);
                for aggr_expr in aggr_exprs {
                    f.write_input("", aggr_expr);
                }
                for group_expr in group_exprs {
                    f.write_input("", group_expr);
                }
            }
        }
    }
}
