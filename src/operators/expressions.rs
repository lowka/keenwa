use crate::memo::{CopyInNestedExprs, MemoExprFormatter};
use crate::meta::ColumnId;
use crate::operators::scalar::ScalarValue;
use crate::operators::{Operator, OperatorInputs, RelNode};
use itertools::Itertools;
use std::fmt::{Display, Formatter};

/// Expressions supported by the optimizer.
#[derive(Debug, Clone)]
pub enum Expr {
    Column(ColumnId),
    Scalar(ScalarValue),
    BinaryExpr {
        lhs: Box<Expr>,
        op: BinaryOp,
        rhs: Box<Expr>,
    },
    Not(Box<Expr>),
    Aggregate {
        func: AggregateFunction,
        args: Vec<Expr>,
        filter: Option<Box<Expr>>,
    },
    SubQuery(RelNode),
}

impl Expr {
    pub fn with_new_inputs(&self, inputs: &mut OperatorInputs) -> Self {
        match self {
            Expr::Column(_) => self.clone(),
            Expr::Scalar(_) => self.clone(),
            Expr::BinaryExpr { lhs, op, rhs } => {
                let lhs = lhs.with_new_inputs(inputs);
                let rhs = rhs.with_new_inputs(inputs);
                Expr::BinaryExpr {
                    lhs: Box::new(lhs),
                    op: op.clone(),
                    rhs: Box::new(rhs),
                }
            }
            Expr::Not(expr) => {
                let expr = expr.with_new_inputs(inputs);
                Expr::Not(Box::new(expr))
            }
            Expr::Aggregate { func, args, filter } => Expr::Aggregate {
                func: func.clone(),
                args: args.iter().map(|e| e.with_new_inputs(inputs)).collect(),
                filter: filter.as_ref().map(|f| Box::new(f.with_new_inputs(inputs))),
            },
            Expr::SubQuery(_) => Expr::SubQuery(inputs.rel_node()),
        }
    }

    pub fn copy_in_nested(&self, collector: &mut CopyInNestedExprs<Operator>) {
        match self {
            Expr::Column(_) => {}
            Expr::Scalar(_) => {}
            Expr::BinaryExpr { lhs, rhs, .. } => {
                lhs.copy_in_nested(collector);
                rhs.copy_in_nested(collector);
            }
            Expr::Not(expr) => expr.copy_in_nested(collector),
            Expr::Aggregate { args, filter, .. } => {
                for arg in args {
                    arg.copy_in_nested(collector);
                }
                filter.as_ref().map(|e| e.copy_in_nested(collector));
            }
            Expr::SubQuery(expr) => collector.visit_expr(expr.get_ref()),
        }
    }

    pub fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Expr");
        f.write_value("", self);
    }
}

/// Binary operators.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum BinaryOp {
    And,
    Or,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Column(column_id) => write!(f, "col:{:}", column_id),
            Expr::Scalar(value) => write!(f, "{}", value),
            Expr::BinaryExpr { lhs, op, rhs } => write!(f, "{} {} {}", lhs, op, rhs),
            Expr::Aggregate { func, args, filter } => {
                write!(f, "{}({})", func, DisplayArgs(args))?;
                if let Some(filter) = filter {
                    write!(f, " filter (where {})", filter)?;
                }
                Ok(())
            }
            Expr::Not(expr) => write!(f, "NOT {}", &*expr),
            Expr::SubQuery(expr) => match expr {
                RelNode::Expr(expr) => {
                    let ptr: *const Operator = &**expr;
                    write!(f, "SubQuery ptr {:?}", ptr)
                }
                RelNode::Group(group) => write!(f, "SubQuery {}", group.id()),
            },
        }
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::And => write!(f, "AND"),
            BinaryOp::Or => write!(f, "OR"),
            BinaryOp::Eq => write!(f, "="),
            BinaryOp::NotEq => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::LtEq => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::GtEq => write!(f, ">="),
        }
    }
}

/// Supported aggregate functions.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum AggregateFunction {
    Avg,
    Count,
    Max,
    Min,
    Sum,
}

impl Display for AggregateFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateFunction::Avg => write!(f, "avg"),
            AggregateFunction::Count => write!(f, "count"),
            AggregateFunction::Max => write!(f, "max"),
            AggregateFunction::Min => write!(f, "min"),
            AggregateFunction::Sum => write!(f, "sum"),
        }
    }
}

struct DisplayArgs<'b>(&'b Vec<Expr>);

impl Display for DisplayArgs<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.0.len() == 1 {
            write!(f, "{}", self.0[0])
        } else {
            write!(f, "[{}]", self.0.iter().join(", ").as_str())
        }
    }
}
