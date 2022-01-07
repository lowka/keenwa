use crate::meta::ColumnId;

use crate::operators::scalar::expr::{Expr, ExprVisitor, NestedExpr};
use std::convert::Infallible;

/// Collects column identifiers from the given expression.
pub fn collect_columns<T>(expr: &Expr<T>) -> Vec<ColumnId>
where
    T: NestedExpr,
{
    let mut columns = Vec::new();
    let mut visitor = CollectColumns { columns: &mut columns };
    // Never returns an error
    expr.accept(&mut visitor).unwrap();
    columns
}

struct CollectColumns<'a> {
    columns: &'a mut Vec<ColumnId>,
}
impl<T> ExprVisitor<T> for CollectColumns<'_>
where
    T: NestedExpr,
{
    type Error = Infallible;

    fn post_visit(&mut self, expr: &Expr<T>) -> Result<(), Self::Error> {
        if let Expr::Column(id) = expr {
            self.columns.push(*id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::operators::scalar::expr::{BinaryOp, NestedExpr};
    use crate::operators::scalar::exprs::collect_columns;
    use crate::operators::scalar::value::ScalarValue;
    use std::fmt::{Debug, Formatter};
    use std::hash::Hash;

    #[derive(Debug, Eq, PartialEq, Clone, Hash)]
    struct DummyExpr;

    impl NestedExpr for DummyExpr {
        fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    type Expr = crate::operators::scalar::expr::Expr<DummyExpr>;

    #[test]
    fn test_collect_columns() {
        let expr = Expr::Column(1);
        let columns = collect_columns(&expr);
        assert_eq!(columns, vec![1], "one column");

        let expr = Expr::BinaryExpr {
            lhs: Box::new(Expr::Column(1)),
            op: BinaryOp::And,
            rhs: Box::new(Expr::Not(Box::new(Expr::Column(2)))),
        };
        let columns = collect_columns(&expr);
        assert_eq!(columns, vec![1, 2], "nested columns");

        let expr = Expr::Scalar(ScalarValue::Int32(1));
        let columns = collect_columns(&expr);
        assert_eq!(columns, vec![], "no columns");
    }
}
