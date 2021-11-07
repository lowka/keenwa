use crate::memo::{ExprContext, MemoExprFormatter, TraversalContext};
use crate::meta::ColumnId;
use crate::operators::expressions::Expr;
use crate::operators::join::JoinCondition;
use crate::operators::InputExpr;
use crate::operators::{Operator, OperatorExpr};

// TODO: Docs
/// A logical expression describes a high-level operator without specifying an implementation algorithm to be used.
#[derive(Debug, Clone)]
pub enum LogicalExpr {
    Projection {
        input: InputExpr,
        columns: Vec<ColumnId>,
    },
    Select {
        input: InputExpr,
        filter: Expr,
    },
    Aggregate {
        input: InputExpr,
        aggr_exprs: Vec<Expr>,
        group_exprs: Vec<Expr>,
    },
    Join {
        left: InputExpr,
        right: InputExpr,
        condition: JoinCondition,
    },
    Get {
        source: String,
        columns: Vec<ColumnId>,
    },
    Expr {
        expr: Expr,
    },
}

impl LogicalExpr {
    pub(crate) fn traverse(&self, expr_ctx: &mut ExprContext<Operator>, ctx: &mut TraversalContext<Operator>) {
        match self {
            LogicalExpr::Projection { input, .. } => {
                ctx.visit_input(expr_ctx, input);
            }
            LogicalExpr::Select { input, .. } => {
                ctx.visit_input(expr_ctx, input);
            }
            LogicalExpr::Join { left, right, .. } => {
                ctx.visit_input(expr_ctx, left);
                ctx.visit_input(expr_ctx, right);
            }
            LogicalExpr::Get { .. } => {}
            LogicalExpr::Expr { .. } => {}
            LogicalExpr::Aggregate { input, .. } => {
                ctx.visit_input(expr_ctx, input);
            }
        }
    }

    pub fn with_new_inputs(&self, mut inputs: Vec<InputExpr>) -> Self {
        match self {
            LogicalExpr::Projection { columns, .. } => {
                assert_eq!(1, inputs.len(), "Projection operator expects 1 input but got {:?}", inputs);
                LogicalExpr::Projection {
                    input: inputs.swap_remove(0),
                    columns: columns.clone(),
                }
            }
            LogicalExpr::Select { filter, .. } => {
                assert_eq!(1, inputs.len(), "Select operator expects 1 input but got {:?}", inputs);
                LogicalExpr::Select {
                    input: inputs.swap_remove(0),
                    filter: filter.clone(),
                }
            }
            LogicalExpr::Join { condition, .. } => {
                assert_eq!(2, inputs.len(), "Join operator expects 2 inputs but got {:?}", inputs);
                LogicalExpr::Join {
                    left: inputs.swap_remove(0),
                    right: inputs.swap_remove(0),
                    condition: condition.clone(),
                }
            }
            LogicalExpr::Get { source, columns } => {
                assert_eq!(0, inputs.len(), "Get operator expects 0 inputs but got {:?}", inputs);
                LogicalExpr::Get {
                    source: source.clone(),
                    columns: columns.clone(),
                }
            }
            LogicalExpr::Expr { expr } => {
                assert_eq!(0, inputs.len(), "Expr operator expects 0 inputs but got {:?}", inputs);
                LogicalExpr::Expr { expr: expr.clone() }
            }
            LogicalExpr::Aggregate {
                aggr_exprs,
                group_exprs,
                ..
            } => {
                assert_eq!(1, inputs.len(), "Aggregate operator expects 1 input but got {:?}", inputs);
                LogicalExpr::Aggregate {
                    input: inputs.swap_remove(0),
                    aggr_exprs: aggr_exprs.clone(),
                    group_exprs: group_exprs.clone(),
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
                f.write_value("filter", filter);
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
            LogicalExpr::Expr { expr } => {
                f.write_name(format!("Logica lExpr {}", expr).as_str());
            }
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                f.write_name("LogicalAggregate");
                f.write_input("input", input);
                f.write_values("aggrs", &aggr_exprs);
                f.write_values("groups", &group_exprs);
            }
        }
    }

    // FIXME: ??? ToOperator trait
    pub fn to_operator(self) -> Operator {
        Operator::from(OperatorExpr::from(self))
    }
}
