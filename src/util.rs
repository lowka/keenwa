use crate::cost::Cost;
use crate::memo::GroupId;
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::scalar::ScalarExpr;
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use std::fmt::Debug;

/// A reference to the best expression in a memo-group.
#[derive(Debug, Clone)]
pub enum BestExprRef<'a> {
    /// A relational expression.
    Relational(&'a PhysicalExpr),
    /// A scalar expression.
    Scalar(&'a ScalarExpr),
}

/// A callback called by the optimizer when an optimized plan is being built.
pub trait ResultCallback {
    /// Called for each expression in the optimized plan.
    fn on_best_expr<C>(&self, expr: BestExprRef, ctx: &C)
    where
        C: BestExprContext;
}

/// Provides additional information about an expression chosen by the optimizer as the best expression.
pub trait BestExprContext {
    /// Returns the cost of the expression.
    fn cost(&self) -> Cost;

    /// Returns the logical properties of the expression.
    fn logical(&self) -> &LogicalProperties;

    /// Returns physical properties required by the expression.
    fn required(&self) -> &PhysicalProperties;

    /// Returns the identifier of a group the expression belongs to.
    fn group_id(&self) -> GroupId;

    /// Returns the number of child expressions.
    fn num_children(&self) -> usize;

    /// Returns the identifier of a group of the i-th child expression.
    fn child_group_id(&self, i: usize) -> GroupId;

    /// Returns physical properties required by the i-th child expression.
    fn child_required(&self, i: usize) -> &PhysicalProperties;
}

/// A [`ResultCallback`](crate::util::ResultCallback) that does nothing.
#[derive(Debug)]
pub struct NoOpResultCallback;

impl ResultCallback for NoOpResultCallback {
    fn on_best_expr<C>(&self, _expr: BestExprRef, _ctx: &C)
    where
        C: BestExprContext,
    {
        //no-op
    }
}
