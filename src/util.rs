use crate::cost::Cost;
use crate::memo::GroupId;
use crate::operators::expressions::Expr;
use crate::operators::physical::PhysicalExpr;
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use std::fmt::Debug;

/// A reference to the best expression in a memo-group.
#[derive(Debug, Clone)]
pub enum BestExprRef<'a> {
    /// A relational expression.
    Relational(&'a PhysicalExpr),
    /// A scalar expression.
    Scalar(&'a Expr),
}

/// A helper trait that is called by the optimizer when an optimized operator tree is being built.
pub trait ResultCallback: Debug {
    /// Called for each expression in the final operator tree.
    fn on_best_expr<C>(&self, expr: BestExprRef, ctx: &C)
    where
        C: BestExprContext;
}

/// Provides additional information about an expression selected by the optimizer.
pub trait BestExprContext: Debug {
    /// Returns the cost of the expression.
    fn cost(&self) -> Cost;

    /// Returns the logical properties of the expression.
    fn logical(&self) -> &LogicalProperties;

    /// Returns the physical properties required by the expression.
    fn required(&self) -> &PhysicalProperties;

    /// Returns the identifier of a group the expression belongs to.
    fn group_id(&self) -> GroupId;

    /// Returns the number of input expressions.
    fn inputs_num(&self) -> usize;

    /// Returns the identifier of a group of the i-th input expression.
    fn input_group_id(&self, i: usize) -> GroupId;

    /// Returns the physical properties required by the i-th input expression.
    fn input_required(&self, i: usize) -> &PhysicalProperties;
}

/// A [ResultCallback] that does nothing.
///
/// [ResultCallback]: crate::util::ResultCallback
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
