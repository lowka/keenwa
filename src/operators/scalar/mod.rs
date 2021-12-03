use crate::memo::{MemoExpr, MemoGroupRef};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::ExprRewriter;
use crate::operators::{Operator, OperatorInputs, Properties};

pub mod expr;
pub mod value;

/// The type of scalar expressions supported by the optimizer.
pub type ScalarExpr = self::expr::Expr<RelNode>;

/// A scalar node of an operator tree.
///
/// Should not be created directly and it is responsibility of the caller to provide a instance of `Operator`
/// which is a valid scalar expression.
#[derive(Debug, Clone)]
pub enum ScalarNode {
    /// A node is an expression.
    Expr(Box<Operator>),
    /// A node is a memo-group.
    Group(MemoGroupRef<Operator>),
}

impl ScalarNode {
    /// Returns a reference to a scalar expression stored inside this node:
    /// * if this node is an expression returns a reference to the underlying expression.
    /// * If this node is a memo group returns a reference to the first expression of this memo group.
    pub fn expr(&self) -> &ScalarExpr {
        match self {
            ScalarNode::Expr(expr) => expr.expr().as_scalar(),
            ScalarNode::Group(group) => group.expr().as_scalar(),
        }
    }

    /// Returns a reference to properties of the underlying expression.
    /// * If this is an expression returns a references to the properties of that expression.
    /// * If this is a memo-group returns a reference to the properties of the first expression in that group.
    pub fn props(&self) -> &Properties {
        match self {
            ScalarNode::Expr(expr) => expr.props(),
            ScalarNode::Group(group) => group.props(),
        }
    }
}

/// Replaces all relational expressions in of this expression tree with
/// relational expressions provided by `inputs` object. if this expression tree does not
/// contain nested sub-queries this method returns a copy of this `Expr`.
pub fn expr_with_new_inputs(expr: &ScalarExpr, inputs: &mut OperatorInputs) -> ScalarExpr {
    struct RelInputsRewriter<'a> {
        inputs: &'a mut OperatorInputs,
    }
    impl ExprRewriter<RelNode> for RelInputsRewriter<'_> {
        fn rewrite(&mut self, expr: ScalarExpr) -> ScalarExpr {
            if let ScalarExpr::SubQuery(_) = expr {
                let rel_node = self.inputs.rel_node();
                ScalarExpr::SubQuery(rel_node)
            } else {
                expr
            }
        }
    }
    let expr = expr.clone();
    let mut rewriter = RelInputsRewriter { inputs };
    expr.rewrite(&mut rewriter)
}
