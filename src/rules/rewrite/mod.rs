use std::collections::VecDeque;

use crate::memo::{MemoExpr, NewChildExprs};
use crate::operators::relational::RelNode;
use crate::operators::Operator;

mod filter_push_down;
mod remove_redundant_projections;
#[cfg(test)]
mod testing;

/// Rewrites relational child expressions of the relational expression `expr` using the given function.
pub fn rewrite_rel_inputs<T>(expr: &RelNode, mut rewrite: T) -> RelNode
where
    T: FnMut(&RelNode) -> RelNode,
{
    let children = expr
        .mexpr()
        .children()
        .filter_map(|child_expr| {
            if child_expr.expr().is_relational() {
                let child_expr = RelNode::from(child_expr.clone());
                let child_expr = (rewrite)(&child_expr);
                Some(child_expr)
            } else {
                None
            }
        })
        .collect();

    with_new_rel_inputs(expr, children)
}

/// Creates a new relational expression from the given expression by replacing its child relational expressions.
pub fn with_new_rel_inputs(expr: &RelNode, mut inputs: Vec<RelNode>) -> RelNode {
    if inputs.is_empty() {
        return expr.clone();
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

    let expr = Operator::expr_with_new_children(expr.mexpr().expr(), NewChildExprs::new(new_children));
    RelNode::from(Operator::from(expr))
}
