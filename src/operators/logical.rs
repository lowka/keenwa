use crate::memo::{ExprContext, MemoExprFormatter};
use crate::meta::ColumnId;
use crate::operators::join::JoinCondition;
use crate::operators::{Operator, OperatorExpr};
use crate::operators::{OperatorCopyIn, OperatorInputs, RelExpr, RelNode, ScalarNode};

// TODO: Docs
/// A logical expression describes a high-level operator without specifying an implementation algorithm to be used.
#[derive(Debug, Clone)]
pub enum LogicalExpr {
    Projection {
        input: RelNode,
        columns: Vec<ColumnId>,
    },
    Select {
        input: RelNode,
        filter: ScalarNode,
    },
    Aggregate {
        input: RelNode,
        aggr_exprs: Vec<ScalarNode>,
        group_exprs: Vec<ScalarNode>,
    },
    Join {
        left: RelNode,
        right: RelNode,
        condition: JoinCondition,
    },
    Get {
        source: String,
        columns: Vec<ColumnId>,
    },
}

impl LogicalExpr {
    pub(crate) fn traverse(&self, visitor: &mut OperatorCopyIn, expr_ctx: &mut ExprContext<Operator>) {
        match self {
            LogicalExpr::Projection { input, .. } => {
                visitor.visit_rel(expr_ctx, input);
            }
            LogicalExpr::Select { input, filter } => {
                visitor.visit_rel(expr_ctx, input);
                visitor.visit_scalar(expr_ctx, filter);
            }
            LogicalExpr::Join { left, right, .. } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
            }
            LogicalExpr::Get { .. } => {}
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                visitor.visit_rel(expr_ctx, input);
                for expr in aggr_exprs {
                    visitor.visit_scalar(expr_ctx, expr);
                }
                for group_expr in group_exprs {
                    visitor.visit_scalar(expr_ctx, group_expr);
                }
            }
        }
    }

    pub fn with_new_inputs(&self, inputs: &mut OperatorInputs) -> Self {
        match self {
            LogicalExpr::Projection { columns, .. } => {
                inputs.expect_len(1, "LogicalProjection");
                LogicalExpr::Projection {
                    input: inputs.rel_node(),
                    columns: columns.clone(),
                }
            }
            LogicalExpr::Select { .. } => {
                inputs.expect_len(2, "LogicalSelect");
                LogicalExpr::Select {
                    input: inputs.rel_node(),
                    filter: inputs.scalar_node(),
                }
            }
            LogicalExpr::Join { condition, .. } => {
                inputs.expect_len(2, "LogicalJoin");
                LogicalExpr::Join {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    condition: condition.clone(),
                }
            }
            LogicalExpr::Get { source, columns } => {
                inputs.expect_len(0, "LogicalGet");
                LogicalExpr::Get {
                    source: source.clone(),
                    columns: columns.clone(),
                }
            }
            LogicalExpr::Aggregate {
                aggr_exprs,
                group_exprs,
                ..
            } => {
                inputs.expect_len(1 + aggr_exprs.len() + group_exprs.len(), "LogicalAggregate");
                LogicalExpr::Aggregate {
                    input: inputs.rel_node(),
                    aggr_exprs: inputs.scalar_nodes(aggr_exprs.len()),
                    group_exprs: inputs.scalar_nodes(group_exprs.len()),
                }
            }
        }
    }

    pub(crate) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            LogicalExpr::Projection { input, columns } => {
                f.write_name("LogicalProjection");
                f.write_input("input", input);
                f.write_values("cols", columns);
            }
            LogicalExpr::Select { input, filter } => {
                f.write_name("LogicalSelect");
                f.write_input("input", input);
                f.write_input("filter", filter);
            }
            LogicalExpr::Join { left, right, condition } => {
                f.write_name("LogicalJoin");
                f.write_input("left", left);
                f.write_input("right", right);
                match condition {
                    JoinCondition::Using(using) => f.write_value("using", using),
                }
            }
            LogicalExpr::Get { source, columns } => {
                f.write_name("LogicalGet");
                f.write_source(source);
                f.write_values("cols", columns);
            }
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                f.write_name("LogicalAggregate");
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

    // FIXME: ??? ToOperator trait
    pub fn to_operator(self) -> Operator {
        let expr = RelExpr::Logical(Box::new(self));
        let expr = OperatorExpr::Relational(expr);
        Operator::from(expr)
    }
}
