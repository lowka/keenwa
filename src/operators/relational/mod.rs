use crate::memo::{MemoExpr, MemoGroupRef};
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::scalar::expr::NestedExpr;
use crate::operators::{Operator, Properties};
use std::fmt::Formatter;

pub mod join;
pub mod logical;
pub mod physical;

/// A relational expression. Relational expressions can be either [logical] or [physical].
#[derive(Debug, Clone)]
pub enum RelExpr {
    Logical(Box<LogicalExpr>),
    Physical(Box<PhysicalExpr>),
}

impl RelExpr {
    /// Returns a reference to the underlying logical expression.
    ///
    /// # Panics
    ///
    /// If this expression is not a logical expression this methods panics.
    pub fn as_logical(&self) -> &LogicalExpr {
        match self {
            RelExpr::Logical(expr) => expr.as_ref(),
            RelExpr::Physical(_) => {
                panic!("Expected a logical expression but got: {:?}", self)
            }
        }
    }

    /// Returns a reference to the underlying physical expression.
    ///
    /// # Panics
    ///
    /// If this expression is not a physical expression this methods panics.
    pub fn as_physical(&self) -> &PhysicalExpr {
        match self {
            RelExpr::Logical(_) => {
                panic!("Expected a physical expression but got: {:?}", self)
            }
            RelExpr::Physical(expr) => expr.as_ref(),
        }
    }
}

/// A relational node of an operator tree.
///
/// Should not be created directly and it is responsibility of the caller to provide a instance of `Operator`
/// which is a valid relational expression.
#[derive(Debug, Clone)]
pub enum RelNode {
    /// A node is an expression.
    Expr(Box<Operator>),
    /// A node is a memo-group.
    Group(MemoGroupRef<Operator>),
}

impl RelNode {
    /// Returns a reference to a relational expression stored inside this node:
    /// * if this node is an expression returns a reference to the underlying expression.
    /// * If this node is a memo group returns a reference to the first expression of this memo group.
    pub fn expr(&self) -> &RelExpr {
        match self {
            RelNode::Expr(expr) => expr.expr().as_relational(),
            RelNode::Group(group) => group.expr().as_relational(),
        }
    }

    /// Returns a reference to properties associated with this node:
    /// * if this node is an expression returns properties of the expression.
    /// * If this node is a memo group returns a reference to the properties of this memo group.
    pub fn props(&self) -> &Properties {
        match self {
            RelNode::Expr(expr) => expr.props(),
            RelNode::Group(group) => group.props(),
        }
    }
}

impl NestedExpr for RelNode {
    fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RelNode::Expr(expr) => {
                let ptr: *const Operator = &**expr;
                write!(f, "ptr {:?}", ptr)
            }
            RelNode::Group(group) => write!(f, "{}", group.id()),
        }
    }
}
