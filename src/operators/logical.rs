use crate::memo::{ExprContext, MemoExprDigest, MemoExprFormatter, TraversalContext};
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

    pub(crate) fn make_digest<D>(&self, digest: &mut D)
    where
        D: MemoExprDigest,
    {
        match self {
            LogicalExpr::Projection { input, columns } => {
                digest.append_expr_type("l:Projection");
                digest.append_input("input", input);
                digest.append_property("cols", columns);
            }
            LogicalExpr::Select { input, filter } => {
                digest.append_expr_type("l:Select");
                digest.append_input("input", input);
                digest.append_value(format!("filter={}", filter).as_str());
            }
            LogicalExpr::Join { left, right, condition } => {
                digest.append_expr_type("l:Join");
                digest.append_input("left", left);
                digest.append_input("right", right);
                digest.append_value(format!("condition={:?}", condition).as_str());
            }
            LogicalExpr::Get { source, columns } => {
                digest.append_expr_type("l:Get");
                digest.append_value(source);
                digest.append_property("cols", columns);
            }
            LogicalExpr::Expr { expr } => {
                digest.append_expr_type("l:Expr");
                digest.append_value(format!("{}", expr).as_str());
            }
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                digest.append_expr_type("l:Aggregate");
                digest.append_input("input", input);
                digest.append_property("aggrs", aggr_exprs);
                digest.append_property("groups", group_exprs);
            }
        }
    }

    pub(crate) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            LogicalExpr::Projection { input, columns } => {
                f.write_name("Projection");
                f.write_input("input", input);
                f.write_value("cols", format!("{:?}", columns));
            }
            LogicalExpr::Select { input, filter } => {
                f.write_name("Select");
                f.write_input("input", input);
                f.write_value("filter", format!("{}", filter));
            }
            LogicalExpr::Join { left, right, condition } => {
                f.write_name("Join");
                f.write_input("left", left);
                f.write_input("right", right);
                match condition {
                    JoinCondition::Using(using) => f.write_value("using", format!("{}", using)),
                }
            }
            LogicalExpr::Get { source, columns } => {
                f.write_name("Get");
                f.write_source(source);
                f.write_value("cols", format!("{:?}", columns));
            }
            LogicalExpr::Expr { expr } => {
                f.write_name(format!("Expr {}", expr).as_str());
            }
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                f.write_name("Aggregate");
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
