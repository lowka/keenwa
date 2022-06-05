use std::convert::Infallible;

use crate::meta::ColumnId;
use crate::operators::scalar::expr::{Expr, ExprVisitor, NestedExpr};

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
        match expr {
            Expr::SubQuery(query) | Expr::InSubQuery { query, .. } | Expr::Exists { query, .. } => {
                self.columns.extend_from_slice(query.outer_columns());
            }
            Expr::Column(id) => {
                self.columns.push(*id);
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::fmt::{Debug, Formatter};
    use std::hash::Hash;

    use crate::meta::testing::TestMetadata;
    use crate::meta::ColumnId;
    use crate::operators::scalar::expr::NestedExpr;
    use crate::operators::scalar::exprs::collect_columns;
    use crate::operators::scalar::value::{Scalar, ScalarValue};

    #[derive(Debug, Eq, PartialEq, Clone, Hash, Default)]
    struct DummyExpr {
        output_columns: Vec<ColumnId>,
        outer_columns: Vec<ColumnId>,
    }

    impl NestedExpr for DummyExpr {
        fn output_columns(&self) -> &[ColumnId] {
            &self.output_columns
        }

        fn outer_columns(&self) -> &[ColumnId] {
            &self.outer_columns
        }

        fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    type Expr = crate::operators::scalar::expr::Expr<DummyExpr>;

    #[test]
    fn test_collect_columns() {
        let mut metadata = TestMetadata::new();
        let col1 = metadata.synthetic_column().build();
        let col2 = metadata.synthetic_column().build();

        let expr = Expr::Column(col1);
        let columns = collect_columns(&expr);
        assert_eq!(columns, vec![col1], "one column");

        let expr = !Expr::Column(col1).and(Expr::Column(col2));
        let columns = collect_columns(&expr);
        assert_eq!(columns, vec![col1, col2], "nested columns");

        let expr = Expr::Scalar(1.get_value());
        let columns = collect_columns(&expr);
        let empty = Vec::<ColumnId>::new();
        assert_eq!(columns, empty, "no columns");
    }

    #[test]
    fn test_collect_columns_from_nested_query() {
        let mut metadata = TestMetadata::new();
        let col1 = metadata.synthetic_column().build();

        test_subqueries(
            &DummyExpr {
                output_columns: vec![col1],
                outer_columns: vec![],
            },
            vec![],
            "output columns must not be returned",
        );

        test_subqueries(
            &DummyExpr {
                output_columns: vec![],
                outer_columns: vec![col1],
            },
            vec![col1],
            "outer columns must be included",
        );
    }

    fn expect_columns(expr: &Expr, expected: Vec<ColumnId>, message: &str) {
        let columns = collect_columns(&expr);
        assert_eq!(expected, columns, "{} ", message);
    }

    fn test_subqueries(expr: &DummyExpr, expected: Vec<ColumnId>, message: &str) {
        let subquery = Expr::SubQuery(expr.clone());
        expect_columns(&subquery, expected.clone(), message);

        let subquery = Expr::Exists {
            not: false,
            query: expr.clone(),
        };
        expect_columns(&subquery, expected.clone(), message);

        let subquery = Expr::InSubQuery {
            not: false,
            expr: Box::new(Expr::Scalar(1.get_value())),
            query: expr.clone(),
        };
        expect_columns(&subquery, expected, message);
    }
}
