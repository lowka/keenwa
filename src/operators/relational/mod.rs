//! Relational expressions supported by the optimizer.

use std::fmt::{Display, Formatter};

use crate::memo::MemoExprState;
use crate::meta::ColumnId;
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::scalar::expr::NestedExpr;
use crate::operators::Operator;

pub mod join;
pub mod logical;
pub mod physical;

/// The type of the relational nodes used by [operators](crate::operators::Operator).
pub type RelNode = crate::memo::RelNode<Operator>;

/// A relational expression. Relational expressions can be either [logical] or [physical].
///
/// [logical]: crate::operators::relational::logical::LogicalExpr
/// [physical]: crate::operators::relational::physical::PhysicalExpr
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
    pub fn logical(&self) -> &LogicalExpr {
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
    pub fn physical(&self) -> &PhysicalExpr {
        match self {
            RelExpr::Logical(_) => {
                panic!("Expected a physical expression but got: {:?}", self)
            }
            RelExpr::Physical(expr) => expr.as_ref(),
        }
    }
}

impl NestedExpr for RelNode {
    fn output_columns(&self) -> &[ColumnId] {
        self.props().logical.output_columns()
    }

    fn outer_columns(&self) -> &[ColumnId] {
        &self.props().logical.outer_columns
    }

    fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.state() {
            MemoExprState::Owned(_) => {
                // let ptr: *const Operator = &**expr;
                // write!(f, "ptr {:?}", ptr)
                write!(f, "*ptr")
            }
            MemoExprState::Memo(state) => write!(f, "{}", state.group_id()),
        }
    }
}

/// The type of a set operator.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SetOperator {
    /// Union.
    Union,
    /// Intersect.
    Intersect,
    /// Except.
    Except,
}

impl Display for SetOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SetOperator::Union => write!(f, "Union"),
            SetOperator::Intersect => write!(f, "Intersect"),
            SetOperator::Except => write!(f, "Except"),
        }
    }
}
