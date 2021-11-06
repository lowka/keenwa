use crate::meta::ColumnId;
use crate::operators::scalar::ScalarValue;
use std::fmt::{Display, Formatter};

/// Expressions supported by the optimizer.
#[derive(Debug, Clone, Eq, PartialEq)]
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
            write!(f, "[")?;
            for (i, expr) in self.0.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", expr)?;
            }
            write!(f, "]")
        }
    }
}
