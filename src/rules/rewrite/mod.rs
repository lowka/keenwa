//! Experimental. Logical rewrite rules.

use crate::error::OptimizerError;
use std::collections::VecDeque;

use crate::memo::{MemoExpr, NewChildExprs};
use crate::operators::relational::RelNode;
use crate::operators::Operator;

mod filter_push_down;
mod remove_redundant_projections;
#[cfg(test)]
mod testing;

/// Rewrites relational child expressions of the relational expression `expr` using the given function.
/// See [with_new_rel_inputs] for error conditions.
pub fn rewrite_rel_inputs<T>(expr: &RelNode, mut rewrite: T) -> Result<RelNode, OptimizerError>
where
    T: FnMut(&RelNode) -> Result<RelNode, OptimizerError>,
{
    let children: Result<Vec<RelNode>, _> = expr
        .mexpr()
        .children()
        .filter(|expr| expr.expr().is_relational())
        .map(|expr| RelNode::try_from(expr.clone()).and_then(|expr| rewrite(&expr)))
        // .filter_map(|child_expr| {
        //     if child_expr.expr().is_relational() {
        //         let child_expr = RelNode::tr(child_expr.clone());
        //         let child_expr = (rewrite)(&child_expr);
        //         Some(child_expr)
        //     } else {
        //         None
        //     }
        // })
        .collect();

    with_new_rel_inputs(expr, children?)
}

/// Creates a new relational expression from the given expression by replacing its child relational expressions.
/// When `inputs` is are empty returns a clone of `expr`.
///
/// # Errors
///  
/// This method returns an error if inputs can not be used to replace relational expression of the given expression.
/// This usually happens when the number of child relational expressions of the given expressions
/// and the number of expressions in the `inputs` argument do not match.
pub fn with_new_rel_inputs(expr: &RelNode, mut inputs: Vec<RelNode>) -> Result<RelNode, OptimizerError> {
    if inputs.is_empty() {
        return Ok(expr.clone());
    }

    let mut new_children = VecDeque::new();
    for child_expr in expr.mexpr().children() {
        let child_expr = if child_expr.expr().is_relational() {
            inputs.swap_remove(0).into_inner()
        } else {
            child_expr.clone()
        };
        new_children.push_back(child_expr);
    }

    let expr = Operator::expr_with_new_children(expr.mexpr().expr(), NewChildExprs::new(new_children))?;
    RelNode::try_from(Operator::from(expr))
}
