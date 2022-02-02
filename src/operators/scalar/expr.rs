use crate::datatypes::DataType;
use crate::memo::MemoExprFormatter;
use crate::meta::ColumnId;
use crate::operators::scalar::expr::Expr::BinaryExpr;
use crate::operators::scalar::value::ScalarValue;
use itertools::Itertools;
use std::convert::TryFrom;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

/// Expressions supported by the optimizer.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Expr<T>
where
    T: NestedExpr,
{
    /// A column reference expression. In contrast to [column name expression](Self::ColumnName) it stores an internal identifier.
    // ???: Make Expr generic over column id.
    Column(ColumnId),
    /// A column reference expression. The optimizer replaces such expressions with [column(id)](Self::Column) expressions.
    ColumnName(String),
    /// A scalar value.
    Scalar(ScalarValue),
    /// A binary expression (eg. 5 + 1).
    BinaryExpr {
        /// The left operand.
        lhs: Box<Expr<T>>,
        /// The operator.
        op: BinaryOp,
        /// The right operand.
        rhs: Box<Expr<T>>,
    },
    /// Negation of an expression.
    Not(Box<Expr<T>>),
    /// An expression with the given name (eg. 1 + 1 as two).
    Alias(Box<Expr<T>>, String),
    /// An aggregate expression.
    Aggregate {
        /// The aggregate function.
        func: AggregateFunction,
        /// A list of arguments.
        args: Vec<Expr<T>>,
        /// A filter expression. Specifies a condition which selects input rows used by this aggregate expression.
        /// (eg. `sum(a) FILTER (WHERE a > 10)`).
        filter: Option<Box<Expr<T>>>,
    },
    /// A subquery expression.
    //TODO: Implement table subquery operators such as ALL, ANY, SOME, EXISTS, UNIQUE.
    SubQuery(T),
}

/// Trait that must be implemented by other expressions that can be nested inside [Expr](self::Expr).
pub trait NestedExpr: Debug + Clone + Eq + Hash {
    /// Writes this nested expression to the given formatter.
    fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
}

impl<T> Expr<T>
where
    T: NestedExpr,
{
    /// Performs a depth-first traversal of this expression tree calling methods of the given `visitor`.
    ///
    /// If [ExprVisitor::pre_visit] returns `Ok(false)` then child expressions of the expression are not visited.
    ///
    /// If an error is returned then traversal terminates.
    pub fn accept<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: ExprVisitor<T>,
    {
        if !visitor.pre_visit(self)? {
            return Ok(());
        }
        match self {
            Expr::Column(_) => {}
            Expr::ColumnName(_) => {}
            Expr::Scalar(_) => {}
            Expr::BinaryExpr { lhs, rhs, .. } => {
                lhs.accept(visitor)?;
                rhs.accept(visitor)?;
            }
            Expr::Not(expr) => {
                expr.accept(visitor)?;
            }
            Expr::Alias(expr, _) => {
                expr.accept(visitor)?;
            }
            Expr::Aggregate { args, filter, .. } => {
                for arg in args {
                    arg.accept(visitor)?;
                }
                if let Some(f) = filter.as_ref() {
                    f.accept(visitor)?;
                }
            }
            Expr::SubQuery(_) => {
                // Should be handled by visitor::pre_visit because Expr is generic over
                // the type of a nested sub query.
            }
        }
        visitor.post_visit(self)
    }

    /// Performs a depth-first traversal of this expression and recursively rewrites it using the given `rewriter`.
    ///
    /// If [ExprRewriter::pre_rewrite] returns `Ok(false)` then child expressions of the expression are not visited.
    ///
    /// If an error is returned then traversal terminates.
    pub fn rewrite<V>(self, rewriter: &mut V) -> Result<Self, V::Error>
    where
        V: ExprRewriter<T>,
    {
        if !rewriter.pre_rewrite(&self)? {
            return Ok(self);
        }
        let expr = match self {
            Expr::Column(_) => rewriter.rewrite(self)?,
            Expr::ColumnName(_) => rewriter.rewrite(self)?,
            Expr::Scalar(_) => rewriter.rewrite(self)?,
            Expr::BinaryExpr { lhs, rhs, op } => {
                let lhs = rewrite_boxed(*lhs, rewriter)?;
                let rhs = rewrite_boxed(*rhs, rewriter)?;
                Expr::BinaryExpr { lhs, op, rhs }
            }
            Expr::Not(expr) => expr.rewrite(rewriter)?,
            Expr::Alias(expr, name) => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                Expr::Alias(expr, name)
            }
            Expr::Aggregate { func, args, filter } => Expr::Aggregate {
                func,
                args: rewrite_vec(args, rewriter)?,
                filter: rewrite_boxed_option(filter, rewriter)?,
            },
            Expr::SubQuery(_) => rewriter.rewrite(self)?,
        };
        rewriter.post_rewrite(expr)
    }

    pub fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Expr");
        f.write_value("", self);
    }

    /// Returns child expressions of this expression.
    pub fn get_children(&self) -> Vec<Expr<T>> {
        match self {
            Expr::Column(_) => vec![],
            Expr::ColumnName(_) => vec![],
            Expr::Scalar(_) => vec![],
            Expr::BinaryExpr { lhs, rhs, .. } => vec![lhs.as_ref().clone(), rhs.as_ref().clone()],
            Expr::Not(expr) => vec![expr.as_ref().clone()],
            Expr::Alias(expr, _) => vec![expr.as_ref().clone()],
            Expr::Aggregate { args, filter, .. } => {
                let mut children: Vec<_> = args.to_vec();
                if let Some(filter) = filter.clone() {
                    children.push(filter.as_ref().clone())
                }
                children
            }
            Expr::SubQuery(_) => vec![],
        }
    }

    /// Creates a new expressions that retains all properties of this expression but contains the given child expressions.
    ///
    /// # Panics
    ///
    /// This method panics if the number of elements in `children` is not equal to the
    /// number of child expressions of this expression.
    pub fn with_children(&self, mut children: Vec<Expr<T>>) -> Self {
        fn expect_children(expr: &str, actual: usize, expected: usize) {
            assert_eq!(actual, expected, "{}: Unexpected number of child expressions", expr);
        }

        match self {
            Expr::Column(id) => {
                expect_children("Column", children.len(), 0);
                Expr::Column(*id)
            }
            Expr::ColumnName(name) => {
                expect_children("ColumnName", children.len(), 0);
                Expr::ColumnName(name.clone())
            }
            Expr::Scalar(value) => {
                expect_children("Scalar", children.len(), 0);
                Expr::Scalar(value.clone())
            }
            Expr::BinaryExpr { op, .. } => {
                expect_children("BinaryExpr", children.len(), 2);
                Expr::BinaryExpr {
                    lhs: Box::new(children.swap_remove(0)),
                    op: op.clone(),
                    rhs: Box::new(children.swap_remove(0)),
                }
            }
            Expr::Not(_) => {
                expect_children("Not", children.len(), 1);
                Expr::Not(Box::new(children.swap_remove(0)))
            }
            Expr::Alias(_, name) => {
                expect_children("Alias", children.len(), 1);
                Expr::Alias(Box::new(children.swap_remove(0)), name.clone())
            }
            Expr::Aggregate { func, args, filter } => {
                expect_children("Aggregate", children.len(), args.len() + filter.as_ref().map(|_| 1).unwrap_or(0));
                let (args, filter) = if filter.is_some() {
                    // filter is the last expression (see Expr::get_children)
                    let filter = children.remove(args.len() - 1);
                    // use [0..len-1] expressions as arguments.
                    let args = children;
                    (args, Some(Box::new(filter)))
                } else {
                    (children, None)
                };
                Expr::Aggregate {
                    func: func.clone(),
                    args,
                    filter,
                }
            }
            Expr::SubQuery(node) => {
                expect_children("SubQuery", children.len(), 0);
                Expr::SubQuery(node.clone())
            }
        }
    }

    /// Returns `this_expr as name` expression.
    pub fn alias(self, name: &str) -> Self {
        Expr::Alias(Box::new(self), name.to_owned())
    }

    /// Returns `this_expr AND rhs` expression.
    pub fn and(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::And, rhs)
    }

    /// Returns `this_expr OR rhs` expression.
    pub fn or(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Or, rhs)
    }

    /// Returns `!this_expr` expression.
    pub fn not(self) -> Self {
        Expr::Not(Box::new(self))
    }

    /// Returns `this_expr = rhs` expression.
    pub fn eq(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Eq, rhs)
    }

    /// Returns `this_expr != rhs` expression.
    pub fn ne(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::NotEq, rhs)
    }

    /// Returns `this_expr < rhs` expression.
    pub fn lt(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Lt, rhs)
    }

    /// Returns `this_expr <= rhs` expression.
    pub fn lte(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::LtEq, rhs)
    }

    /// Returns `this_expr > rhs` expression.
    pub fn gt(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Gt, rhs)
    }

    /// Returns `this_expr >= rhs` expression.
    pub fn gte(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::GtEq, rhs)
    }

    /// Returns `this_expr + rhs` expression.
    pub fn add(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Plus, rhs)
    }

    /// Returns `this_expr - rhs` expression.
    pub fn sub(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Minus, rhs)
    }

    /// Returns `this_expr * rhs` expression.
    pub fn mul(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Multiply, rhs)
    }

    /// Returns `this_expr / rhs` expression.
    pub fn div(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Divide, rhs)
    }

    /// Returns `this_expr % rhs` expression.
    pub fn modulo(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Modulo, rhs)
    }

    fn binary_expr(&self, op: BinaryOp, rhs: Expr<T>) -> Self {
        Expr::BinaryExpr {
            lhs: Box::new(self.clone()),
            op,
            rhs: Box::new(rhs),
        }
    }
}

/// Called by [Expr::accept] during a traversal of an expression tree.
pub trait ExprVisitor<T>
where
    T: NestedExpr,
{
    /// The error type returned when operation fails.
    type Error;

    /// Called before all child expressions of `expr` are visited.
    ///
    /// Default implementation always returns `Ok(true)`.  
    fn pre_visit(&mut self, _expr: &Expr<T>) -> Result<bool, Self::Error> {
        Ok(true)
    }

    /// Called after all child expressions of `expr` are visited.
    fn post_visit(&mut self, expr: &Expr<T>) -> Result<(), Self::Error>;
}

/// Called by [Expr::rewrite] during a traversal of an expression tree.
pub trait ExprRewriter<T>
where
    T: NestedExpr,
{
    /// The error type returned when operation fails.
    type Error;

    /// Called before all child expressions of `expr` are rewritten.
    ///
    /// Default implementation always returns `Ok(true)`.
    fn pre_rewrite(&mut self, _expr: &Expr<T>) -> Result<bool, Self::Error> {
        Ok(true)
    }

    /// Rewrites the given expression. Called after all children of the given expression are rewritten.
    fn rewrite(&mut self, expr: Expr<T>) -> Result<Expr<T>, Self::Error>;

    /// Called after all child expressions are rewritten. `expr` contains the expression produced by [ExprRewriter::rewrite].
    ///
    /// Default implementation always returns the provided expression.
    fn post_rewrite(&mut self, expr: Expr<T>) -> Result<Expr<T>, Self::Error> {
        Ok(expr)
    }
}

fn rewrite_boxed<T, V>(expr: Expr<T>, rewriter: &mut V) -> Result<Box<Expr<T>>, V::Error>
where
    V: ExprRewriter<T>,
    T: NestedExpr,
{
    let new_expr = expr.rewrite(rewriter)?;
    Ok(Box::new(new_expr))
}

fn rewrite_vec<T, V>(exprs: Vec<Expr<T>>, rewriter: &mut V) -> Result<Vec<Expr<T>>, V::Error>
where
    V: ExprRewriter<T>,
    T: NestedExpr,
{
    exprs.into_iter().map(|e| e.rewrite(rewriter)).collect()
}

fn rewrite_boxed_option<T, V>(expr: Option<Box<Expr<T>>>, rewriter: &mut V) -> Result<Option<Box<Expr<T>>>, V::Error>
where
    V: ExprRewriter<T>,
    T: NestedExpr,
{
    match expr {
        None => Ok(None),
        Some(expr) => Ok(Some(rewrite_boxed(*expr, rewriter)?)),
    }
}

/// Binary operators.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum BinaryOp {
    /// Logical AND.
    And,
    /// Logical OR.
    Or,
    /// Equals.
    Eq,
    /// Not equals.
    NotEq,
    /// Less than.
    Lt,
    /// Less than or equals.
    LtEq,
    /// Greater than.
    Gt,
    /// Greater than or equals.
    GtEq,
    /// Addition.
    Plus,
    /// Subtraction.
    Minus,
    /// Multiplication.
    Multiply,
    /// Division.
    Divide,
    /// Modulo/Remainder.
    Modulo,
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
            Expr::Column(column_id) => write!(f, "col:{}", column_id),
            Expr::ColumnName(name) => write!(f, "col:{}", name),
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
            Expr::Alias(expr, name) => write!(f, "{} AS {}", &*expr, name),
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
            BinaryOp::Plus => write!(f, "+"),
            BinaryOp::Minus => write!(f, "-"),
            BinaryOp::Multiply => write!(f, "*"),
            BinaryOp::Divide => write!(f, "/"),
            BinaryOp::Modulo => write!(f, "%"),
        }
    }
}

/// Supported aggregate functions.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum AggregateFunction {
    Avg,
    Count,
    Max,
    Min,
    Sum,
}

impl TryFrom<&str> for AggregateFunction {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            _ if value.eq_ignore_ascii_case("avg") => Ok(AggregateFunction::Avg),
            _ if value.eq_ignore_ascii_case("count") => Ok(AggregateFunction::Count),
            _ if value.eq_ignore_ascii_case("max") => Ok(AggregateFunction::Max),
            _ if value.eq_ignore_ascii_case("min") => Ok(AggregateFunction::Min),
            _ if value.eq_ignore_ascii_case("sum") => Ok(AggregateFunction::Sum),
            _ => Err(()),
        }
    }
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

/// Converts a type to a [scalar value](super::value::ScalarValue).
pub trait Scalar {
    /// Convert this type to a scalar value.
    fn get_value(&self) -> ScalarValue;
}

impl Scalar for bool {
    fn get_value(&self) -> ScalarValue {
        ScalarValue::Bool(*self)
    }
}

impl Scalar for i32 {
    fn get_value(&self) -> ScalarValue {
        ScalarValue::Int32(*self)
    }
}

impl Scalar for &str {
    fn get_value(&self) -> ScalarValue {
        ScalarValue::String((*self).to_owned())
    }
}

impl Scalar for String {
    fn get_value(&self) -> ScalarValue {
        ScalarValue::String(self.to_owned())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::cell::Cell;
    use std::convert::Infallible;
    use std::hash::Hasher;
    use std::rc::Rc;

    #[derive(Debug, Clone)]
    struct DummyRelExpr;

    impl Eq for DummyRelExpr {}

    impl PartialEq<Self> for DummyRelExpr {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }

    impl Hash for DummyRelExpr {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_usize(0)
        }
    }

    impl NestedExpr for DummyRelExpr {
        fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    type Expr = super::Expr<DummyRelExpr>;

    #[test]
    fn expr_methods() {
        let expr = Expr::Scalar(ScalarValue::Int32(1));
        let rhs = Expr::Scalar(ScalarValue::Int32(10));

        expect_expr(expr.clone().and(rhs.clone()), "1 AND 10");
        expect_expr(expr.clone().or(rhs.clone()), "1 OR 10");
        expect_expr(expr.clone().not(), "NOT 1");

        expect_expr(expr.clone().eq(rhs.clone()), "1 = 10");
        expect_expr(expr.clone().ne(rhs.clone()), "1 != 10");

        expect_expr(expr.clone().lt(rhs.clone()), "1 < 10");
        expect_expr(expr.clone().lte(rhs.clone()), "1 <= 10");

        expect_expr(expr.clone().gt(rhs.clone()), "1 > 10");
        expect_expr(expr.clone().gte(rhs.clone()), "1 >= 10");

        expect_expr(expr.clone().add(rhs.clone()), "1 + 10");
        expect_expr(expr.clone().sub(rhs.clone()), "1 - 10");
        expect_expr(expr.clone().mul(rhs.clone()), "1 * 10");
        expect_expr(expr.clone().div(rhs.clone()), "1 / 10");
        expect_expr(expr.clone().modulo(rhs.clone()), "1 % 10");
    }

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
        let expr = Expr::Scalar(ScalarValue::Int32(1)).ne(Expr::Scalar(ScalarValue::Int32(2)));
        expect_traversal_order(&expr, vec!["pre:1 != 2", "pre:1", "post:1", "pre:2", "post:2", "post:1 != 2"]);
    }

    #[test]
    fn not_expr_traversal() {
        let expr = Expr::Scalar(ScalarValue::Bool(true)).not();
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
        struct ColumnRewriter {
            skip_column: ColumnId,
            post_rewrites: Rc<Cell<usize>>,
        }
        impl ExprRewriter<DummyRelExpr> for ColumnRewriter {
            type Error = Infallible;

            fn pre_rewrite(&mut self, expr: &Expr) -> Result<bool, Self::Error> {
                if let Expr::Column(column) = expr {
                    Ok(*column != self.skip_column)
                } else {
                    Ok(true)
                }
            }

            fn rewrite(&mut self, expr: Expr) -> Result<Expr, Self::Error> {
                if let Expr::Column(1) = expr {
                    Ok(Expr::Column(2))
                } else {
                    Ok(expr)
                }
            }

            fn post_rewrite(&mut self, expr: Expr) -> Result<Expr, Self::Error> {
                let cell = self.post_rewrites.as_ref();
                cell.set(cell.get() + 1);
                Ok(expr)
            }
        }

        let expr = Expr::Column(1).or(Expr::Column(3).and(Expr::Column(1)));
        let post_rewrites = Rc::new(Cell::new(0));
        let rewriter = ColumnRewriter {
            skip_column: 0,
            post_rewrites: post_rewrites.clone(),
        };
        expect_rewritten(expr.clone(), rewriter, "col:2 OR col:3 AND col:2");
        assert_eq!(post_rewrites.get(), 5, "post rewrite calls");

        let post_rewrites = Rc::new(Cell::new(0));
        let rewriter = ColumnRewriter {
            skip_column: 1,
            post_rewrites: post_rewrites.clone(),
        };
        expect_rewritten(expr, rewriter, "col:1 OR col:3 AND col:1");
        assert_eq!(post_rewrites.get(), 3, "post rewrite calls");
    }

    #[test]
    fn rewriter_fails() {
        #[derive(Default)]
        struct FailingRewriter {
            visited: usize,
            rewritten: usize,
        }

        impl ExprRewriter<DummyRelExpr> for FailingRewriter {
            type Error = ();

            fn pre_rewrite(&mut self, _expr: &Expr) -> Result<bool, Self::Error> {
                self.visited += 1;
                Ok(true)
            }

            fn rewrite(&mut self, expr: Expr) -> Result<Expr, Self::Error> {
                self.rewritten += 1;
                if let Expr::Column(1) = expr {
                    Err(())
                } else {
                    Ok(expr)
                }
            }
        }

        let expr = Expr::Column(1).or(Expr::Column(3).and(Expr::Column(1)));

        let mut rewriter = FailingRewriter::default();
        let result = expr.rewrite(&mut rewriter);
        let _ = result.expect_err("Expected an error");

        assert_eq!(rewriter.visited, 2);
        assert_eq!(rewriter.rewritten, 1);
    }

    #[test]
    fn aggr_func_try_from_is_case_insensitive() {
        fn expect_parsed(s: &str, expected: AggregateFunction) {
            let f = AggregateFunction::try_from(s)
                .ok()
                .unwrap_or_else(|| panic!("Failed to parse string: {}", s));
            assert_eq!(f, expected);
        }

        expect_parsed("Avg", AggregateFunction::Avg);
        expect_parsed("CoUNT", AggregateFunction::Count);
        expect_parsed("MAX", AggregateFunction::Max);
        expect_parsed("min", AggregateFunction::Min);
        expect_parsed("Sum", AggregateFunction::Sum);
    }

    fn expect_traversal_order(expr: &Expr, expected: Vec<&str>) {
        struct TraversalTester {
            exprs: Vec<String>,
        }
        impl ExprVisitor<DummyRelExpr> for TraversalTester {
            type Error = Infallible;

            fn pre_visit(&mut self, expr: &Expr) -> Result<bool, Self::Error> {
                self.exprs.push(format!("pre:{}", expr));
                Ok(true)
            }

            fn post_visit(&mut self, expr: &Expr) -> Result<(), Self::Error> {
                self.exprs.push(format!("post:{}", expr));
                Ok(())
            }
        }

        let mut visitor = TraversalTester { exprs: Vec::new() };
        expr.accept(&mut visitor).unwrap();

        let expected: Vec<String> = expected.into_iter().map(|s| s.to_string()).collect();
        assert_eq!(visitor.exprs, expected, "traversal order does not match");
    }

    fn expect_rewritten<V>(expr: Expr, mut rewriter: V, result: &str)
    where
        V: ExprRewriter<DummyRelExpr, Error = Infallible>,
    {
        let rewritten_expr = expr.rewrite(&mut rewriter).unwrap();
        assert_eq!(format!("{}", rewritten_expr), result.to_string(), "rewritten expression does not match");
    }

    fn expect_expr(expr: Expr, expected: &str) {
        assert_eq!(expected, format!("{}", expr))
    }
}
