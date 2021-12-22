use crate::memo::NewChildExprs;
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::ExprRewriter;
use crate::operators::Operator;

pub mod expr;
pub mod value;

/// The type of scalar expressions supported by the optimizer.
pub type ScalarExpr = self::expr::Expr<RelNode>;
/// The type of the scalar nodes used by [operators](crate::operators::Operator).
pub type ScalarNode = crate::memo::ScalarNode<Operator>;

/// Replaces all relational expressions in of this expression tree with
/// relational expressions provided by `inputs` object. if this expression tree does not
/// contain nested sub-queries this method returns a copy of this `Expr`.
pub fn expr_with_new_inputs(expr: &ScalarExpr, inputs: &mut NewChildExprs<Operator>) -> ScalarExpr {
    struct RelInputsRewriter<'a> {
        inputs: &'a mut NewChildExprs<Operator>,
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
