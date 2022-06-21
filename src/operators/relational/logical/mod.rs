//! Logical expressions supported by the optimizer.

use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::relational::join::JoinCondition;
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::ExprVisitor;
use crate::operators::scalar::{get_subquery, ScalarExpr};
use crate::operators::{Operator, OperatorCopyIn};

pub use aggregates::{LogicalAggregate, LogicalWindowAggregate};
pub use distinct::LogicalDistinct;
pub use empty::LogicalEmpty;
pub use get::LogicalGet;
pub use join::LogicalJoin;
pub use limit::LogicalLimit;
pub use offset::LogicalOffset;
pub use projection::LogicalProjection;
pub use select::LogicalSelect;
pub use set_ops::{LogicalExcept, LogicalIntersect, LogicalUnion};
pub use values::LogicalValues;

mod aggregates;
mod distinct;
mod empty;
mod get;
mod join;
mod limit;
mod offset;
mod projection;
mod select;
mod set_ops;
mod values;

// TODO: Docs
/// A logical expression describes a high-level operator without specifying an implementation algorithm to be used.
#[derive(Debug, Clone)]
pub enum LogicalExpr {
    /// Logical projection operator.
    Projection(LogicalProjection),
    /// Logical select operator.
    Select(LogicalSelect),
    /// Logical aggregate operator.
    Aggregate(LogicalAggregate),
    /// Logical window aggregate operator.
    WindowAggregate(LogicalWindowAggregate),
    /// Logical join operator.
    Join(LogicalJoin),
    /// Logical get/scan operator.
    Get(LogicalGet),
    /// Logical union operator.
    Union(LogicalUnion),
    /// Logical intersect operator.
    Intersect(LogicalIntersect),
    /// Logical except operator.
    Except(LogicalExcept),
    /// Logical distinct operator.
    Distinct(LogicalDistinct),
    /// Logical limit operator.
    Limit(LogicalLimit),
    /// Logical offset operator.
    Offset(LogicalOffset),
    /// Logical values operator.
    Values(LogicalValues),
    /// Relation that produces no rows.
    Empty(LogicalEmpty),
}

impl LogicalExpr {
    pub(crate) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        match self {
            LogicalExpr::Projection(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Select(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Join(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Get(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Aggregate(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::WindowAggregate(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Union(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Intersect(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Except(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Distinct(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Limit(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Offset(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Values(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Empty(expr) => expr.copy_in(visitor, expr_ctx),
        }
    }

    pub fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        let expr = match self {
            LogicalExpr::Projection(expr) => LogicalExpr::Projection(expr.with_new_inputs(inputs)?),
            LogicalExpr::Select(expr) => LogicalExpr::Select(expr.with_new_inputs(inputs)?),
            LogicalExpr::Join(expr) => LogicalExpr::Join(expr.with_new_inputs(inputs)?),
            LogicalExpr::Get(expr) => LogicalExpr::Get(expr.with_new_inputs(inputs)?),
            LogicalExpr::Aggregate(expr) => LogicalExpr::Aggregate(expr.with_new_inputs(inputs)?),
            LogicalExpr::WindowAggregate(expr) => LogicalExpr::WindowAggregate(expr.with_new_inputs(inputs)?),
            LogicalExpr::Union(expr) => LogicalExpr::Union(expr.with_new_inputs(inputs)?),
            LogicalExpr::Intersect(expr) => LogicalExpr::Intersect(expr.with_new_inputs(inputs)?),
            LogicalExpr::Except(expr) => LogicalExpr::Except(expr.with_new_inputs(inputs)?),
            LogicalExpr::Distinct(expr) => LogicalExpr::Distinct(expr.with_new_inputs(inputs)?),
            LogicalExpr::Limit(expr) => LogicalExpr::Limit(expr.with_new_inputs(inputs)?),
            LogicalExpr::Offset(expr) => LogicalExpr::Offset(expr.with_new_inputs(inputs)?),
            LogicalExpr::Values(expr) => LogicalExpr::Values(expr.with_new_inputs(inputs)?),
            LogicalExpr::Empty(expr) => LogicalExpr::Empty(expr.with_new_inputs(inputs)?),
        };
        Ok(expr)
    }

    pub fn num_children(&self) -> usize {
        match self {
            LogicalExpr::Projection(expr) => expr.num_children(),
            LogicalExpr::Select(expr) => expr.num_children(),
            LogicalExpr::Join(expr) => expr.num_children(),
            LogicalExpr::Get(expr) => expr.num_children(),
            LogicalExpr::Aggregate(expr) => expr.num_children(),
            LogicalExpr::WindowAggregate(expr) => expr.num_children(),
            LogicalExpr::Union(expr) => expr.num_children(),
            LogicalExpr::Intersect(expr) => expr.num_children(),
            LogicalExpr::Except(expr) => expr.num_children(),
            LogicalExpr::Distinct(expr) => expr.num_children(),
            LogicalExpr::Limit(expr) => expr.num_children(),
            LogicalExpr::Offset(expr) => expr.num_children(),
            LogicalExpr::Values(expr) => expr.num_children(),
            LogicalExpr::Empty(expr) => expr.num_children(),
        }
    }

    pub fn get_child(&self, i: usize) -> Option<&Operator> {
        match self {
            LogicalExpr::Projection(expr) => expr.get_child(i),
            LogicalExpr::Select(expr) => expr.get_child(i),
            LogicalExpr::Aggregate(expr) => expr.get_child(i),
            LogicalExpr::WindowAggregate(expr) => expr.get_child(i),
            LogicalExpr::Join(expr) => expr.get_child(i),
            LogicalExpr::Get(expr) => expr.get_child(i),
            LogicalExpr::Union(expr) => expr.get_child(i),
            LogicalExpr::Intersect(expr) => expr.get_child(i),
            LogicalExpr::Except(expr) => expr.get_child(i),
            LogicalExpr::Distinct(expr) => expr.get_child(i),
            LogicalExpr::Limit(expr) => expr.get_child(i),
            LogicalExpr::Offset(expr) => expr.get_child(i),
            LogicalExpr::Values(expr) => expr.get_child(i),
            LogicalExpr::Empty(expr) => expr.get_child(i),
        }
    }

    pub fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            LogicalExpr::Projection(expr) => expr.format_expr(f),
            LogicalExpr::Select(expr) => expr.format_expr(f),
            LogicalExpr::Join(expr) => expr.format_expr(f),
            LogicalExpr::Get(expr) => expr.format_expr(f),
            LogicalExpr::Aggregate(expr) => expr.format_expr(f),
            LogicalExpr::WindowAggregate(expr) => expr.format_expr(f),
            LogicalExpr::Union(expr) => expr.format_expr(f),
            LogicalExpr::Intersect(expr) => expr.format_expr(f),
            LogicalExpr::Except(expr) => expr.format_expr(f),
            LogicalExpr::Distinct(expr) => expr.format_expr(f),
            LogicalExpr::Limit(expr) => expr.format_expr(f),
            LogicalExpr::Offset(expr) => expr.format_expr(f),
            LogicalExpr::Values(expr) => expr.format_expr(f),
            LogicalExpr::Empty(expr) => expr.format_expr(f),
        }
    }

    /// Performs a depth-first traversal of this expression tree calling methods of the given `visitor`.
    ///
    /// If [LogicalExprVisitor::pre_visit] returns `Ok(false)` then child expressions of the expression are not visited.
    ///
    /// If an error is returned then traversal terminates.
    pub fn accept<T>(&self, visitor: &mut T) -> Result<(), OptimizerError>
    where
        T: LogicalExprVisitor,
    {
        if !visitor.pre_visit(self)? {
            return Ok(());
        }

        struct VisitNestedLogicalExprs<'a, T> {
            parent_expr: &'a LogicalExpr,
            visitor: &'a mut T,
        }
        impl<T> ExprVisitor<RelNode> for VisitNestedLogicalExprs<'_, T>
        where
            T: LogicalExprVisitor,
        {
            type Error = OptimizerError;

            fn post_visit(&mut self, expr: &ScalarExpr) -> Result<(), Self::Error> {
                if let Some(query) = get_subquery(expr) {
                    if self.visitor.pre_visit_subquery(self.parent_expr, query)? {
                        query.expr().logical().accept(self.visitor)?;
                    }
                }
                Ok(())
            }
        }
        let mut expr_visitor = VisitNestedLogicalExprs {
            parent_expr: self,
            visitor,
        };

        match self {
            LogicalExpr::Projection(LogicalProjection { input, exprs, .. }) => {
                for expr in exprs {
                    expr.expr().accept(&mut expr_visitor)?;
                }
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Select(LogicalSelect { input, filter }) => {
                if let Some(expr) = filter.as_ref() {
                    expr.expr().accept(&mut expr_visitor)?;
                }
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Aggregate(LogicalAggregate {
                input,
                aggr_exprs,
                group_exprs,
                ..
            }) => {
                for aggr_expr in aggr_exprs {
                    aggr_expr.expr().accept(&mut expr_visitor)?;
                }
                for group_expr in group_exprs {
                    group_expr.expr().accept(&mut expr_visitor)?;
                }
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::WindowAggregate(LogicalWindowAggregate { input, window_expr, .. }) => {
                window_expr.expr().accept(&mut expr_visitor)?;
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Join(LogicalJoin {
                left, right, condition, ..
            }) => {
                match condition {
                    JoinCondition::Using(_) => (),
                    JoinCondition::On(on) => on.expr().expr().accept(&mut expr_visitor)?,
                };
                left.expr().logical().accept(visitor)?;
                right.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Get(_) => {}
            LogicalExpr::Union(LogicalUnion { left, right, .. })
            | LogicalExpr::Intersect(LogicalIntersect { left, right, .. })
            | LogicalExpr::Except(LogicalExcept { left, right, .. }) => {
                left.expr().logical().accept(visitor)?;
                right.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Distinct(LogicalDistinct { input, .. }) => {
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Limit(LogicalLimit { input, .. }) => {
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Offset(LogicalOffset { input, .. }) => {
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Values(LogicalValues { values, .. }) => {
                for values in values {
                    values.expr().accept(&mut expr_visitor)?;
                }
            }
            LogicalExpr::Empty(_) => {}
        }
        visitor.post_visit(self)
    }
}

/// Called by [LogicalExpr::accept] during a traversal of an expression tree.
pub trait LogicalExprVisitor {
    /// Called before all child expressions of `expr` are visited.
    /// Default implementation always returns`Ok(true)`.
    fn pre_visit(&mut self, _expr: &LogicalExpr) -> Result<bool, OptimizerError> {
        Ok(true)
    }

    /// Called before the given subquery is visited.
    /// Default implementation returns `Ok(true)`.
    fn pre_visit_subquery(&mut self, _expr: &LogicalExpr, _subquery: &RelNode) -> Result<bool, OptimizerError> {
        Ok(true)
    }

    /// Called after all child expressions of `expr` are visited.
    fn post_visit(&mut self, expr: &LogicalExpr) -> Result<(), OptimizerError>;
}
