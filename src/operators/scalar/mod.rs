use crate::memo::NewChildExprs;
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::{ExprRewriter, Scalar};
use crate::operators::scalar::value::ScalarValue;
use crate::operators::Operator;
use std::convert::Infallible;

pub mod expr;
pub mod exprs;
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
        type Error = Infallible;
        fn rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
            if let ScalarExpr::SubQuery(_) = expr {
                let rel_node = self.inputs.rel_node();
                Ok(ScalarExpr::SubQuery(rel_node))
            } else {
                Ok(expr)
            }
        }
    }
    let expr = expr.clone();
    let mut rewriter = RelInputsRewriter { inputs };
    // Never returns an error
    expr.rewrite(&mut rewriter).unwrap()
}

/// Creates a column reference expression.
pub fn col(name: &str) -> ScalarExpr {
    ScalarExpr::ColumnName(name.to_owned())
}

/// Creates a `vec` of column reference expressions.
pub fn cols(cols: Vec<impl Into<String>>) -> Vec<ScalarExpr> {
    cols.into_iter().map(|name| ScalarExpr::ColumnName(name.into())).collect()
}

/// Creates a scalar value expression.
pub fn scalar<T>(val: T) -> ScalarExpr
where
    T: Scalar,
{
    ScalarExpr::Scalar(val.get_value())
}
