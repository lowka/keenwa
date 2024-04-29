//! Scalar expressions supported by the optimizer.

use crate::datatypes::DataType;
use crate::error::OptimizerError;
use std::ops::Deref;

use crate::memo::NewChildExprs;
use crate::meta::{ColumnId, MutableMetadata};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::ExprRewriter;
use crate::operators::scalar::types::ColumnTypeRegistry;
use crate::operators::scalar::value::Scalar;
use crate::operators::Operator;

pub mod aggregates;
pub mod expr;
pub mod exprs;
pub mod func;
pub mod funcs;
pub mod types;
pub mod value;

/// The type of scalar expressions supported by the optimizer.
pub type ScalarExpr = self::expr::Expr<RelNode>;
/// The type of the scalar nodes used by [operators](crate::operators::Operator).
pub type ScalarNode = crate::memo::ScalarNode<Operator>;

/// Replaces all relational expressions of this expression tree with
/// relational expressions provided by `inputs` object. if this expression tree does not
/// contain nested sub-queries this method returns a copy of this `Expr`.
pub fn expr_with_new_inputs(
    expr: &ScalarExpr,
    inputs: &mut NewChildExprs<Operator>,
) -> Result<ScalarExpr, OptimizerError> {
    struct RelInputsRewriter<'a> {
        inputs: &'a mut NewChildExprs<Operator>,
    }
    impl ExprRewriter<RelNode> for RelInputsRewriter<'_> {
        type Error = OptimizerError;
        fn rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
            if let ScalarExpr::SubQuery(_) = expr {
                let rel_node = self.inputs.rel_node()?;
                Ok(ScalarExpr::SubQuery(rel_node))
            } else {
                Ok(expr)
            }
        }
    }
    let expr = expr.clone();
    let mut rewriter = RelInputsRewriter { inputs };
    expr.rewrite(&mut rewriter)
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

/// Creates a wildcard expression `*`.
pub fn wildcard() -> ScalarExpr {
    ScalarExpr::Wildcard(None)
}

/// Creates a qualified wildcard expression (eg. `alias.*` ).
// TODO: Move to projection builder.
pub fn qualified_wildcard(qualifier: &str) -> ScalarExpr {
    ScalarExpr::Wildcard(Some(qualifier.into()))
}

/// Returns a subquery if the given expression contains one.
pub fn get_subquery(expr: &ScalarExpr) -> Option<&RelNode> {
    match expr {
        ScalarExpr::Column(_) => None,
        ScalarExpr::ColumnName(_) => None,
        ScalarExpr::Scalar(_) => None,
        ScalarExpr::BinaryExpr { .. } => None,
        ScalarExpr::Cast { .. } => None,
        ScalarExpr::Not(_) => None,
        ScalarExpr::Negation(_) => None,
        ScalarExpr::InList { .. } => None,
        ScalarExpr::ArrayIndex { .. } => None,
        ScalarExpr::IsNull { .. } => None,
        ScalarExpr::IsFalse { .. } => None,
        ScalarExpr::IsTrue { .. } => None,
        ScalarExpr::IsUnknown { .. } => None,
        ScalarExpr::Between { .. } => None,
        ScalarExpr::Alias(_, _) => None,
        ScalarExpr::Case { .. } => None,
        ScalarExpr::Tuple(_) => None,
        ScalarExpr::Array(_) => None,
        ScalarExpr::ScalarFunction { .. } => None,
        ScalarExpr::Like { .. } => None,
        ScalarExpr::Aggregate { .. } => None,
        ScalarExpr::WindowAggregate { .. } => None,
        ScalarExpr::Ordering { .. } => None,
        ScalarExpr::SubQuery(query) => Some(query),
        ScalarExpr::Exists { query, .. } => Some(query),
        ScalarExpr::InSubQuery { query, .. } => Some(query),
        ScalarExpr::Wildcard(_) => None,
    }
}

// TODO: Implement ColumnRegistry for all metadata types.
impl<T> ColumnTypeRegistry for T
where
    T: Deref<Target = MutableMetadata>,
{
    fn get_column_type(&self, col_id: &ColumnId) -> DataType {
        let target = self.deref();
        target.get_column(col_id).data_type().clone()
    }
}
