use crate::datatypes::DataType;
use crate::memo::MemoExprFormatter;
use crate::meta::ColumnId;
use crate::operators::scalar::value::ScalarValue;
use itertools::Itertools;
use std::fmt::{Debug, Display, Formatter};

/// Expressions supported by the optimizer.
#[derive(Debug, Clone)]
pub enum Expr<T>
where
    T: NestedExpr,
{
    Column(ColumnId),
    // TestOperatorTreeBuilder: AliasColumn(table, column_name)
    // AliasColumn should be used instead of Column(column_id) to simplify testing.
    // TestOperatorTreeBuilder replaces all instances of AliasColumn expressions to corresponding Column(column_id) expressions.
    Scalar(ScalarValue),
    BinaryExpr {
        lhs: Box<Expr<T>>,
        op: BinaryOp,
        rhs: Box<Expr<T>>,
    },
    Not(Box<Expr<T>>),
    Aggregate {
        func: AggregateFunction,
        args: Vec<Expr<T>>,
        filter: Option<Box<Expr<T>>>,
    },
    SubQuery(T),
}

/// Trait that must be implemented by other expressions that can be nested inside [Expr](self::Expr).  
pub trait NestedExpr: Debug + Clone {
    /// Writes this nested expression to the given formatter.
    fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
}

impl<T> Expr<T>
where
    T: NestedExpr,
{
    /// Performs a depth-first traversal of this expression tree calling methods of the given `visitor`.
    pub fn accept<V>(&self, visitor: &mut V)
    where
        V: ExprVisitor<T>,
    {
        visitor.pre_visit(self);
        match self {
            Expr::Column(_) => {}
            Expr::Scalar(_) => {}
            Expr::BinaryExpr { lhs, rhs, .. } => {
                lhs.accept(visitor);
                rhs.accept(visitor);
            }
            Expr::Not(expr) => {
                expr.accept(visitor);
            }
            Expr::Aggregate { args, filter, .. } => {
                for arg in args {
                    arg.accept(visitor);
                }
                if let Some(f) = filter.as_ref() {
                    f.accept(visitor)
                }
            }
            Expr::SubQuery(_) => {}
        }
        visitor.post_visit(self);
    }

    /// Recursively rewrites this expression using the given `rewriter`.
    /// It first calls [`Expr::rewrite`](Self::rewrite) for every child expressions and
    /// then calls [`ExprRewriter::rewrite`] on this expression.
    pub fn rewrite<V>(self, rewriter: &mut V) -> Self
    where
        V: ExprRewriter<T>,
    {
        match self {
            Expr::Column(_) => rewriter.rewrite(self),
            Expr::Scalar(_) => rewriter.rewrite(self),
            Expr::BinaryExpr { lhs, rhs, op } => {
                let lhs = rewrite_boxed(lhs, rewriter);
                let rhs = rewrite_boxed(rhs, rewriter);
                Expr::BinaryExpr { lhs, op, rhs }
            }
            Expr::Not(expr) => expr.rewrite(rewriter),
            Expr::Aggregate { func, args, filter } => Expr::Aggregate {
                func,
                args: rewrite_vec(args, rewriter),
                filter: rewrite_boxed_option(filter, rewriter),
            },
            Expr::SubQuery(_) => rewriter.rewrite(self),
        }
    }

    pub fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Expr");
        f.write_value("", self);
    }
}

/// Called by [`Expr::accept`](self::Expr::accept) during a traversal of an expression tree.
pub trait ExprVisitor<T>
where
    T: NestedExpr,
{
    /// Called before all child expressions of `expr` are visited.  
    fn pre_visit(&mut self, _expr: &Expr<T>) {}

    /// Called after all child expressions of `expr` are visited.
    fn post_visit(&mut self, expr: &Expr<T>);
}

/// Provides methods to rewrite an expression tree.
pub trait ExprRewriter<T>
where
    T: NestedExpr,
{
    /// Rewrites the given expression. Called after all children of the given expression are visited.
    fn rewrite(&mut self, expr: Expr<T>) -> Expr<T>;
}

fn rewrite_boxed<T, V>(expr: Box<Expr<T>>, rewriter: &mut V) -> Box<Expr<T>>
where
    V: ExprRewriter<T>,
    T: NestedExpr,
{
    let new_expr = (*expr).rewrite(rewriter);
    Box::new(new_expr)
}

fn rewrite_vec<T, V>(exprs: Vec<Expr<T>>, rewriter: &mut V) -> Vec<Expr<T>>
where
    V: ExprRewriter<T>,
    T: NestedExpr,
{
    exprs.into_iter().map(|e| e.rewrite(rewriter)).collect()
}

fn rewrite_boxed_option<T, V>(expr: Option<Box<Expr<T>>>, rewriter: &mut V) -> Option<Box<Expr<T>>>
where
    V: ExprRewriter<T>,
    T: NestedExpr,
{
    expr.map(|expr| rewrite_boxed(expr, rewriter))
}

/// Binary operators.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum BinaryOp {
    And,
    Or,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl BinaryOp {
    pub fn return_type(&self) -> DataType {
        DataType::Bool
    }
}

impl<T> Display for Expr<T>
where
    T: NestedExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Column(column_id) => write!(f, "col:{:}", column_id),
            Expr::Scalar(value) => write!(f, "{}", value),
            Expr::BinaryExpr { lhs, op, rhs } => write!(f, "{} {} {}", lhs, op, rhs),
            Expr::Aggregate { func, args, filter } => {
                write!(f, "{}({})", func, DisplayArgs(args))?;
                if let Some(filter) = filter {
                    write!(f, " filter (where {})", filter)?;
                }
                Ok(())
            }
            Expr::Not(expr) => write!(f, "NOT {}", &*expr),
            Expr::SubQuery(expr) => {
                write!(f, "SubQuery ")?;
                expr.write_to_fmt(f)
            }
        }
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::And => write!(f, "AND"),
            BinaryOp::Or => write!(f, "OR"),
            BinaryOp::Eq => write!(f, "="),
            BinaryOp::NotEq => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::LtEq => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::GtEq => write!(f, ">="),
        }
    }
}

/// Supported aggregate functions.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum AggregateFunction {
    Avg,
    Count,
    Max,
    Min,
    Sum,
}

impl Display for AggregateFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateFunction::Avg => write!(f, "avg"),
            AggregateFunction::Count => write!(f, "count"),
            AggregateFunction::Max => write!(f, "max"),
            AggregateFunction::Min => write!(f, "min"),
            AggregateFunction::Sum => write!(f, "sum"),
        }
    }
}

struct DisplayArgs<'b, T>(&'b Vec<Expr<T>>)
where
    T: NestedExpr;

impl<T> Display for DisplayArgs<'_, T>
where
    T: NestedExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.0.len() == 1 {
            write!(f, "{}", self.0[0])
        } else {
            write!(f, "[{}]", self.0.iter().join(", ").as_str())
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Clone)]
    struct DummyRelExpr;

    impl NestedExpr for DummyRelExpr {
        fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    type Expr = super::Expr<DummyRelExpr>;

    #[test]
    fn column_traversal() {
        let expr = Expr::Column(1);
        expect_traversal_order(&expr, vec!["pre:col:1", "post:col:1"]);
    }

    #[test]
    fn scalar_traversal() {
        let expr = Expr::Scalar(ScalarValue::Int32(1));
        expect_traversal_order(&expr, vec!["pre:1", "post:1"]);
    }

    #[test]
    fn binary_expr_traversal() {
        let expr = Expr::BinaryExpr {
            lhs: Box::new(Expr::Scalar(ScalarValue::Int32(1))),
            op: BinaryOp::NotEq,
            rhs: Box::new(Expr::Scalar(ScalarValue::Int32(2))),
        };
        expect_traversal_order(&expr, vec!["pre:1 != 2", "pre:1", "post:1", "pre:2", "post:2", "post:1 != 2"]);
    }

    #[test]
    fn not_expr_traversal() {
        let expr = Expr::Not(Box::new(Expr::Scalar(ScalarValue::Bool(true))));
        expect_traversal_order(&expr, vec!["pre:NOT true", "pre:true", "post:true", "post:NOT true"])
    }

    #[test]
    fn aggr_expr_traversal_no_filter() {
        let expr = Expr::Aggregate {
            func: AggregateFunction::Avg,
            args: vec![Expr::Column(1)],
            filter: None,
        };
        expect_traversal_order(&expr, vec!["pre:avg(col:1)", "pre:col:1", "post:col:1", "post:avg(col:1)"])
    }

    #[test]
    fn aggr_expr_traversal_with_filter() {
        let expr = Expr::Aggregate {
            func: AggregateFunction::Avg,
            args: vec![Expr::Column(1)],
            filter: Some(Box::new(Expr::Scalar(ScalarValue::Bool(true)))),
        };
        expect_traversal_order(
            &expr,
            vec![
                "pre:avg(col:1) filter (where true)",
                "pre:col:1",
                "post:col:1",
                "pre:true",
                "post:true",
                "post:avg(col:1) filter (where true)",
            ],
        )
    }

    #[test]
    fn sub_query_traversal() {
        let expr = Expr::SubQuery(DummyRelExpr);
        expect_traversal_order(&expr, vec!["pre:SubQuery DummyRelExpr", "post:SubQuery DummyRelExpr"]);
    }

    #[test]
    fn rewriter() {
        struct ColumnRewriter;
        impl ExprRewriter<DummyRelExpr> for ColumnRewriter {
            fn rewrite(&mut self, expr: Expr) -> Expr {
                if let Expr::Column(1) = expr {
                    Expr::Column(2)
                } else {
                    expr
                }
            }
        }

        let expr = Expr::BinaryExpr {
            lhs: Box::new(Expr::Column(1)),
            op: BinaryOp::Or,
            rhs: Box::new(Expr::BinaryExpr {
                lhs: Box::new(Expr::Column(3)),
                op: BinaryOp::And,
                rhs: Box::new(Expr::Column(1)),
            }),
        };
        expect_rewritten(expr, ColumnRewriter, "col:2 OR col:3 AND col:2");
    }

    fn expect_traversal_order(expr: &Expr, expected: Vec<&str>) {
        struct TraversalTester {
            exprs: Vec<String>,
        }
        impl ExprVisitor<DummyRelExpr> for TraversalTester {
            fn pre_visit(&mut self, expr: &Expr) {
                self.exprs.push(format!("pre:{}", expr));
            }

            fn post_visit(&mut self, expr: &Expr) {
                self.exprs.push(format!("post:{}", expr));
            }
        }

        let mut visitor = TraversalTester { exprs: Vec::new() };
        expr.accept(&mut visitor);

        // let actual: Vec<String> = visitor
        //     .exprs
        //     .into_iter()
        //     .map(|s| s.split(" ptr 0x").take(1).collect::<String>())
        //     .collect();

        let expected: Vec<String> = expected.into_iter().map(|s| s.to_string()).collect();
        assert_eq!(visitor.exprs, expected, "traversal order does not match");
    }

    fn expect_rewritten<V>(expr: Expr, mut rewriter: V, result: &str)
    where
        V: ExprRewriter<DummyRelExpr>,
    {
        let rewritten_expr = expr.rewrite(&mut rewriter);
        assert_eq!(format!("{}", rewritten_expr), result.to_string(), "rewritten expression does not match");
    }
}
