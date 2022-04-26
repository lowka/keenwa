use std::convert::TryFrom;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Not, Rem, Sub};

use itertools::Itertools;

use crate::datatypes::DataType;
use crate::memo::MemoExprFormatter;
use crate::meta::ColumnId;
use crate::operators::scalar::value::ScalarValue;

/// Expressions supported by the optimizer.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Expr<T>
where
    T: NestedExpr,
{
    /// A column identifier expression. In contrast to [column name expression](Self::ColumnName) it stores an internal identifier.
    // ???: Make Expr generic over column id.
    // TODO: Rename to this expr to ColumnId, rename ColumnName to Column.
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
    /// Cast operator. Coverts an expression to another type.
    Cast {
        /// The expression.
        expr: Box<Expr<T>>,
        /// The type the expression should be converted into.
        data_type: DataType,
    },
    /// Negation of an expression.
    Not(Box<Expr<T>>),
    /// Negation of an arithmetic expression (eg. - 1).
    Negation(Box<Expr<T>>),
    /// IN/ NOT IN (item1, .., itemN) expression.
    InList {
        not: bool,
        expr: Box<Expr<T>>,
        exprs: Vec<Expr<T>>,
    },
    /// IS NULL/ IS NOT NULL expression.
    IsNull { not: bool, expr: Box<Expr<T>> },
    /// An expression with the given name (eg. 1 + 1 as two).
    // TODO: Move to projection builder.
    Alias(Box<Expr<T>>, String),
    /// Case expression.
    Case {
        /// Base case expression.
        expr: Option<Box<Expr<T>>>,
        /// A collection of WHEN <expr> THEN <expr> clauses.
        when_then_exprs: Vec<(Expr<T>, Expr<T>)>,
        /// ELSE <expr>.
        else_expr: Option<Box<Expr<T>>>,
    },
    /// An aggregate expression.
    Aggregate {
        /// The aggregate function.
        func: AggregateFunction,
        /// If set the aggregate function will accept only distinct values from its input.
        distinct: bool,
        /// A list of arguments.
        args: Vec<Expr<T>>,
        /// A filter expression. Specifies a condition which selects input rows used by this aggregate expression.
        /// (eg. `sum(a) FILTER (WHERE a > 10)`).
        filter: Option<Box<Expr<T>>>,
    },
    /// A subquery expression.
    //TODO: Implement table subquery operators such as ALL, ANY, SOME, UNIQUE.
    SubQuery(T),
    /// An EXIST / NOT EXISTS (<subquery>) expression.
    Exists { not: bool, query: T },
    /// IN / NOT IN (<subquery>) expression.
    InSubQuery { not: bool, expr: Box<Expr<T>>, query: T },
    /// Wildcard expression (eg. `*`, `alias.*` etc).
    Wildcard(Option<String>),
}

/// Trait that must be implemented by other expressions that can be nested inside [Expr](self::Expr).
pub trait NestedExpr: Debug + Clone + Eq + Hash {
    /// Returns the output columns returned by this nested expression.
    fn output_columns(&self) -> &[ColumnId];

    /// Outer columns used by this expression.
    fn outer_columns(&self) -> &[ColumnId];

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
            Expr::Cast { expr, .. } => {
                expr.accept(visitor)?;
            }
            Expr::Not(expr) => {
                expr.accept(visitor)?;
            }
            Expr::Negation(expr) => {
                expr.accept(visitor)?;
            }
            Expr::Alias(expr, _) => {
                expr.accept(visitor)?;
            }
            Expr::InList { expr, exprs, .. } => {
                expr.accept(visitor)?;
                for expr in exprs {
                    expr.accept(visitor)?;
                }
            }
            Expr::IsNull { expr, .. } => {
                expr.accept(visitor)?;
            }
            Expr::Case {
                expr,
                when_then_exprs,
                else_expr,
            } => {
                if let Some(expr) = expr {
                    expr.accept(visitor)?;
                }
                for (when, then) in when_then_exprs {
                    when.accept(visitor)?;
                    then.accept(visitor)?;
                }
                if let Some(expr) = else_expr {
                    expr.accept(visitor)?;
                }
            }
            Expr::Aggregate { args, filter, .. } => {
                for arg in args {
                    arg.accept(visitor)?;
                }
                if let Some(f) = filter.as_ref() {
                    f.accept(visitor)?;
                }
            }
            Expr::SubQuery(_) | Expr::Exists { .. } => {
                // Should be handled by visitor::pre_visit because Expr is generic over
                // the type of a nested sub query.
            }
            Expr::InSubQuery { expr, .. } => {
                // See the comment for Expr::SubQuery above.
                expr.accept(visitor)?;
            }
            Expr::Wildcard(_) => {}
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
            Expr::Cast { expr, data_type } => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                Expr::Cast { expr, data_type }
            }
            Expr::Not(expr) => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                Expr::Not(expr)
            }
            Expr::Negation(expr) => {
                let expr = expr.rewrite(rewriter)?;
                Expr::Negation(Box::new(expr))
            }
            Expr::Alias(expr, name) => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                Expr::Alias(expr, name)
            }
            Expr::InList { not, expr, exprs } => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                let exprs = rewrite_vec(exprs, rewriter)?;
                Expr::InList { not, expr, exprs }
            }
            Expr::IsNull { not, expr } => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                Expr::IsNull { not, expr }
            }
            Expr::Case {
                expr,
                when_then_exprs,
                else_expr,
            } => Expr::Case {
                expr: rewrite_boxed_option(expr, rewriter)?,
                when_then_exprs: rewrite_pairs_vec(when_then_exprs, rewriter)?,
                else_expr: rewrite_boxed_option(else_expr, rewriter)?,
            },
            Expr::Aggregate {
                func,
                distinct,
                args,
                filter,
            } => Expr::Aggregate {
                func,
                distinct,
                args: rewrite_vec(args, rewriter)?,
                filter: rewrite_boxed_option(filter, rewriter)?,
            },
            Expr::SubQuery(_) => rewriter.rewrite(self)?,
            Expr::Exists { .. } => rewriter.rewrite(self)?,
            Expr::InSubQuery { not, expr, query } => Expr::InSubQuery {
                not,
                expr: rewrite_boxed(*expr, rewriter)?,
                query,
            },
            Expr::Wildcard(qualifier) => Expr::Wildcard(qualifier),
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
            Expr::Cast { expr, .. } => vec![expr.as_ref().clone()],
            Expr::Not(expr) => vec![expr.as_ref().clone()],
            Expr::Negation(expr) => vec![expr.as_ref().clone()],
            Expr::Alias(expr, _) => vec![expr.as_ref().clone()],
            Expr::Case {
                expr,
                when_then_exprs,
                else_expr,
            } => {
                let mut children = vec![];
                if let Some(expr) = expr {
                    children.push(expr.as_ref().clone());
                }
                for (when, then) in when_then_exprs.iter() {
                    children.push(when.clone());
                    children.push(then.clone());
                }
                if let Some(expr) = else_expr {
                    children.push(expr.as_ref().clone());
                }
                children
            }
            Expr::InList { expr, exprs, .. } => {
                let mut result = Vec::with_capacity(1 + exprs.len());
                result.push(*expr.clone());
                result.extend(exprs.clone().into_iter());
                result
            }
            Expr::IsNull { expr, .. } => vec![expr.as_ref().clone()],
            Expr::Aggregate { args, filter, .. } => {
                let mut children: Vec<_> = args.to_vec();
                if let Some(filter) = filter.clone() {
                    children.push(filter.as_ref().clone())
                }
                children
            }
            Expr::SubQuery(_) => vec![],
            Expr::Exists { .. } => vec![],
            Expr::InSubQuery { .. } => vec![],
            Expr::Wildcard(_) => vec![],
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
            Expr::Cast { data_type, .. } => {
                expect_children("Cast", children.len(), 1);
                Expr::Cast {
                    expr: Box::new(children.swap_remove(0)),
                    data_type: data_type.clone(),
                }
            }
            Expr::Not(_) => {
                expect_children("Not", children.len(), 1);
                Expr::Not(Box::new(children.swap_remove(0)))
            }
            Expr::Negation(_) => {
                expect_children("Negation", children.len(), 1);
                Expr::Not(Box::new(children.swap_remove(0)))
            }
            Expr::Alias(_, name) => {
                expect_children("Alias", children.len(), 1);
                Expr::Alias(Box::new(children.swap_remove(0)), name.clone())
            }
            Expr::InList { not, exprs, .. } => {
                expect_children("InList", children.len(), 1 + exprs.len());
                Expr::InList {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                    exprs: children,
                }
            }
            Expr::IsNull { not, .. } => {
                expect_children("IsNull", children.len(), 1);
                Expr::IsNull {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                }
            }
            Expr::Case {
                expr,
                when_then_exprs,
                else_expr,
            } => {
                let opt_num =
                    expr.as_ref().map(|_| 1).unwrap_or_default() + else_expr.as_ref().map(|_| 1).unwrap_or_default();
                expect_children("Case", children.len(), opt_num + when_then_exprs.len() * 2);

                let expr = if expr.is_some() {
                    let expr = Box::new(children.remove(0));
                    Some(expr)
                } else {
                    None
                };

                let else_expr = if else_expr.is_some() {
                    Some(Box::new(children.swap_remove(children.len() - 1)))
                } else {
                    None
                };

                let mut idx = 0;
                let (when, then): (Vec<_>, Vec<_>) = children.into_iter().partition(|_| {
                    let odd = idx % 2 == 0;
                    idx += 1;
                    odd
                });

                let when_then_exprs: Vec<_> = when.into_iter().zip(then).collect();

                Expr::Case {
                    expr,
                    when_then_exprs,
                    else_expr,
                }
            }
            Expr::Aggregate {
                func,
                distinct,
                args,
                filter,
            } => {
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
                    distinct: *distinct,
                    args,
                    filter,
                }
            }
            Expr::SubQuery(node) => {
                expect_children("SubQuery", children.len(), 0);
                Expr::SubQuery(node.clone())
            }
            Expr::Exists { not, query } => {
                expect_children("Exists", children.len(), 0);
                Expr::Exists {
                    not: *not,
                    query: query.clone(),
                }
            }
            Expr::InSubQuery { not, query, .. } => {
                expect_children("InSubQuery", children.len(), 1);
                Expr::InSubQuery {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                    query: query.clone(),
                }
            }
            Expr::Wildcard(qualifier) => {
                expect_children("Wildcard", children.len(), 0);
                Expr::Wildcard(qualifier.clone())
            }
        }
    }

    /// Returns `this_expr as name` expression.
    pub fn alias(self, name: &str) -> Self {
        Expr::Alias(Box::new(self), name.to_owned())
    }

    /// Returns `this_expr <op> rhs` expression.
    pub fn binary_expr(self, op: BinaryOp, rhs: Expr<T>) -> Self {
        Expr::BinaryExpr {
            lhs: Box::new(self),
            op,
            rhs: Box::new(rhs),
        }
    }

    /// Returns `this_expr AND rhs` expression.
    pub fn and(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::And, rhs)
    }

    /// Returns `this_expr OR rhs` expression.
    pub fn or(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Or, rhs)
    }

    /// Returns `this_expr = rhs` expression.
    pub fn eq(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::Eq, rhs)
    }

    /// Returns `this_expr != rhs` expression.
    pub fn ne(self, rhs: Expr<T>) -> Self {
        self.binary_expr(BinaryOp::NotEq, rhs)
    }

    /// Returns `CAST(this_expr as data_type)` expression.
    pub fn cast(self, data_type: DataType) -> Self {
        Expr::Cast {
            expr: Box::new(self),
            data_type,
        }
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

    /// Returns `-this_expr` expression.
    pub fn negate(self) -> Self {
        Expr::Negation(Box::new(self))
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

// Result<Vec<(expr, expr)>, V:Error> is used only once.
#[allow(clippy::type_complexity)]
fn rewrite_pairs_vec<T, V>(
    exprs: Vec<(Expr<T>, Expr<T>)>,
    rewriter: &mut V,
) -> Result<Vec<(Expr<T>, Expr<T>)>, V::Error>
where
    V: ExprRewriter<T>,
    T: NestedExpr,
{
    let mut result = vec![];
    for (left, right) in exprs {
        let left = left.rewrite(rewriter)?;
        let right = right.rewrite(rewriter)?;
        result.push((left, right));
    }
    Ok(result)
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
            Expr::Cast { expr, data_type } => write!(f, "CAST({} as {})", expr, data_type),
            Expr::Aggregate {
                func,
                distinct,
                args,
                filter,
            } => {
                write!(f, "{}(", func)?;
                if *distinct {
                    write!(f, "distinct ")?;
                }
                write!(f, "{})", DisplayArgs(args))?;
                if let Some(filter) = filter {
                    write!(f, " filter (where {})", filter)?;
                }
                Ok(())
            }
            Expr::Not(expr) => write!(f, "NOT {}", &*expr),
            Expr::Negation(expr) => write!(f, "-{}", &*expr),
            Expr::Alias(expr, name) => write!(f, "{} AS {}", &*expr, name),
            Expr::Case {
                expr,
                when_then_exprs,
                else_expr,
            } => {
                let newline = match expr {
                    None => {
                        write!(f, "CASE ")?;
                        false
                    }
                    Some(expr) => {
                        write!(f, "CASE {}", expr)?;
                        true
                    }
                };
                for (i, (when, then)) in when_then_exprs.iter().enumerate() {
                    if i == 0 && newline || i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "WHEN {} THEN {}", when, then)?;
                }
                if let Some(expr) = else_expr {
                    write!(f, " ELSE {}", expr)?;
                }
                Ok(())
            }
            Expr::InList { not, expr, exprs } => {
                if *not {
                    write!(f, "{} NOT IN (", &*expr)?;
                } else {
                    write!(f, "{} IN (", &*expr)?;
                }
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, ")")
            }
            Expr::IsNull { not, expr } => {
                if *not {
                    write!(f, "IS NOT NULL {}", &*expr)
                } else {
                    write!(f, "IS NULL {}", &*expr)
                }
            }
            Expr::SubQuery(query) => {
                write!(f, "SubQuery ")?;
                query.write_to_fmt(f)
            }
            Expr::Exists { not, query } => {
                if *not {
                    write!(f, "NOT EXISTS ")?;
                } else {
                    write!(f, "EXISTS ")?;
                }
                query.write_to_fmt(f)
            }
            Expr::InSubQuery { not, expr, query } => {
                write!(f, "{}", expr)?;
                if *not {
                    write!(f, " NOT IN ")?;
                } else {
                    write!(f, " IN ")?;
                }
                query.write_to_fmt(f)
            }
            Expr::Wildcard(qualifier) => match qualifier {
                None => write!(f, "*"),
                Some(qualifier) => write!(f, "{}.*", qualifier),
            },
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

impl<T> Add for Expr<T>
where
    T: NestedExpr,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary_expr(BinaryOp::Plus, rhs)
    }
}

impl<T> Sub for Expr<T>
where
    T: NestedExpr,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binary_expr(BinaryOp::Minus, rhs)
    }
}

impl<T> Mul for Expr<T>
where
    T: NestedExpr,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary_expr(BinaryOp::Multiply, rhs)
    }
}

impl<T> Div for Expr<T>
where
    T: NestedExpr,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.binary_expr(BinaryOp::Divide, rhs)
    }
}

impl<T> Rem for Expr<T>
where
    T: NestedExpr,
{
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        self.binary_expr(BinaryOp::Modulo, rhs)
    }
}

impl<T> Not for Expr<T>
where
    T: NestedExpr,
{
    type Output = Self;

    fn not(self) -> Self::Output {
        Expr::Not(Box::new(self))
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
    use std::cell::Cell;
    use std::convert::Infallible;
    use std::hash::Hasher;
    use std::rc::Rc;

    use super::*;

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
        fn output_columns(&self) -> &[ColumnId] {
            &[]
        }

        fn outer_columns(&self) -> &[ColumnId] {
            &[]
        }

        fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    type Expr = super::Expr<DummyRelExpr>;

    fn col(name: &str) -> Expr {
        Expr::ColumnName(String::from(name))
    }

    #[test]
    fn expr_methods() {
        let expr = Expr::Scalar(ScalarValue::Int32(1));
        let rhs = Expr::Scalar(ScalarValue::Int32(10));

        expect_expr(expr.clone().and(rhs.clone()), "1 AND 10");
        expect_expr(expr.clone().or(rhs.clone()), "1 OR 10");
        expect_expr(expr.clone().not(), "NOT 1");
        expect_expr(expr.clone().cast(DataType::Int32), "CAST(1 as Int32)");

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
        expect_expr(expr.clone().rem(rhs), "1 % 10");

        expect_expr(expr.negate(), "-1")
    }

    #[test]
    fn column_traversal() {
        let expr = col("a1");
        expect_traversal_order(&expr, vec!["pre:col:a1", "post:col:a1"]);
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
    fn is_null_expr_traversal() {
        let expr = Expr::IsNull {
            not: false,
            expr: Box::new(Expr::Scalar(ScalarValue::Bool(true))),
        };
        expect_traversal_order(&expr, vec!["pre:IS NULL true", "pre:true", "post:true", "post:IS NULL true"])
    }

    #[test]
    fn in_list_expr_traversal() {
        let expr = Expr::InList {
            not: false,
            expr: Box::new(Expr::Scalar(ScalarValue::Bool(false))),
            exprs: vec![Expr::Scalar(ScalarValue::Bool(true)), Expr::Scalar(ScalarValue::Int32(1))],
        };
        expect_traversal_order_with_depth(
            &expr,
            1,
            vec!["pre:false IN (true, 1)", "false", "true", "1", "post:false IN (true, 1)"],
        )
    }

    #[test]
    fn is_not_null_expr_traversal() {
        let expr = Expr::IsNull {
            not: true,
            expr: Box::new(Expr::Scalar(ScalarValue::Bool(true))),
        };
        expect_traversal_order(&expr, vec!["pre:IS NOT NULL true", "pre:true", "post:true", "post:IS NOT NULL true"])
    }

    #[test]
    fn aggr_expr_traversal_no_filter() {
        let expr = Expr::Aggregate {
            func: AggregateFunction::Avg,
            distinct: false,
            args: vec![col("a1")],
            filter: None,
        };
        expect_traversal_order(&expr, vec!["pre:avg(col:a1)", "pre:col:a1", "post:col:a1", "post:avg(col:a1)"])
    }

    #[test]
    fn aggr_expr_traversal_with_filter() {
        let expr = Expr::Aggregate {
            func: AggregateFunction::Avg,
            distinct: false,
            args: vec![col("a1")],
            filter: Some(Box::new(Expr::Scalar(ScalarValue::Bool(true)))),
        };
        expect_traversal_order(
            &expr,
            vec![
                "pre:avg(col:a1) filter (where true)",
                "pre:col:a1",
                "post:col:a1",
                "pre:true",
                "post:true",
                "post:avg(col:a1) filter (where true)",
            ],
        )
    }

    #[test]
    fn case_traversal() {
        fn val(i: i32) -> Expr {
            Expr::Scalar(ScalarValue::Int32(i))
        }

        let col1 = col("a1");

        let expr = Expr::Case {
            expr: Some(Box::new(col1.clone().gt(val(10)))),
            when_then_exprs: vec![
                (col1.clone().eq(val(11)), val(10) + val(1)),
                (col1.clone().eq(val(12)), val(10) + val(2)),
            ],
            else_expr: Some(Box::new(col1.clone())),
        };
        expect_traversal_order_with_depth(
            &expr,
            1,
            vec![
                "pre:CASE col:a1 > 10 WHEN col:a1 = 11 THEN 10 + 1 WHEN col:a1 = 12 THEN 10 + 2 ELSE col:a1",
                // EXPR
                "col:a1 > 10",
                // WHEN
                "col:a1 = 11",
                // THEN
                "10 + 1",
                // WHEN
                "col:a1 = 12",
                // THEN
                "10 + 2",
                // ELSE
                "col:a1",
                "post:CASE col:a1 > 10 WHEN col:a1 = 11 THEN 10 + 1 WHEN col:a1 = 12 THEN 10 + 2 ELSE col:a1",
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
            skip_column: &'static str,
            post_rewrites: Rc<Cell<usize>>,
        }
        impl ExprRewriter<DummyRelExpr> for ColumnRewriter {
            type Error = Infallible;

            fn pre_rewrite(&mut self, expr: &Expr) -> Result<bool, Self::Error> {
                match expr {
                    Expr::ColumnName(name) => Ok(name != self.skip_column),
                    _ => Ok(true),
                }
            }

            fn rewrite(&mut self, expr: Expr) -> Result<Expr, Self::Error> {
                match expr {
                    Expr::ColumnName(name) if name == "a1" => Ok(col("a2")),
                    _ => Ok(expr),
                }
            }

            fn post_rewrite(&mut self, expr: Expr) -> Result<Expr, Self::Error> {
                let cell = self.post_rewrites.as_ref();
                cell.set(cell.get() + 1);
                Ok(expr)
            }
        }

        let expr = col("a1").or(col("a3")).and(col("a1"));

        let post_rewrites = Rc::new(Cell::new(0));
        let rewriter = ColumnRewriter {
            skip_column: "-",
            post_rewrites: post_rewrites.clone(),
        };
        expect_rewritten(expr.clone(), rewriter, "col:a2 OR col:a3 AND col:a2");
        assert_eq!(post_rewrites.get(), 5, "post rewrite calls");

        let post_rewrites = Rc::new(Cell::new(0));
        let rewriter = ColumnRewriter {
            skip_column: "a1",
            post_rewrites: post_rewrites.clone(),
        };
        expect_rewritten(expr, rewriter, "col:a1 OR col:a3 AND col:a1");
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
                match expr {
                    Expr::ColumnName(name) if name == "a1" => Err(()),
                    _ => Ok(expr),
                }
            }
        }

        let expr = col("a1").or(col("a2").and(col("1a")));

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
        expect_traversal_order_with_depth(expr, usize::MAX, expected)
    }

    fn expect_traversal_order_with_depth(expr: &Expr, max_depth: usize, expected: Vec<&str>) {
        struct TraversalTester {
            exprs: Vec<String>,
            depth: usize,
            max_depth: usize,
        }
        impl ExprVisitor<DummyRelExpr> for TraversalTester {
            type Error = Infallible;

            fn pre_visit(&mut self, expr: &Expr) -> Result<bool, Self::Error> {
                if self.depth >= self.max_depth {
                    self.exprs.push(format!("{}", expr));
                    Ok(false)
                } else {
                    self.exprs.push(format!("pre:{}", expr));
                    self.depth += 1;
                    Ok(true)
                }
            }

            fn post_visit(&mut self, expr: &Expr) -> Result<(), Self::Error> {
                self.exprs.push(format!("post:{}", expr));
                self.depth -= 1;
                Ok(())
            }
        }

        let mut visitor = TraversalTester {
            exprs: Vec::new(),
            depth: 0,
            max_depth,
        };
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
