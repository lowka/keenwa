use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Not, Rem, Sub};

use itertools::Itertools;

use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::memo::MemoExprFormatter;
use crate::meta::ColumnId;
use crate::operators::scalar::aggregates::{AggregateFunction, WindowOrAggregateFunction};
use crate::operators::scalar::funcs::ScalarFunction;
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
    /// `<expr> IN [NOT] (item1, .., itemN)` expression.
    InList {
        not: bool,
        expr: Box<Expr<T>>,
        exprs: Vec<Expr<T>>,
    },
    /// `IS [NOT] NULL` expression.
    IsNull { not: bool, expr: Box<Expr<T>> },
    /// `IS [NOT] FALSE` expression.
    IsFalse { not: bool, expr: Box<Expr<T>> },
    /// `IS [NOT] TRUE` expression.
    IsTrue { not: bool, expr: Box<Expr<T>> },
    /// `IS [NOT] UNKNOWN` expression.
    IsUnknown { not: bool, expr: Box<Expr<T>> },
    /// An expression that checks whether the given value is within the specified range.
    Between {
        /// Negated or not.
        not: bool,
        /// The expression.
        expr: Box<Expr<T>>,
        /// The lower bound.
        low: Box<Expr<T>>,
        /// The upper bound.
        high: Box<Expr<T>>,
    },
    /// An expression with the given name (eg. 1 + 1 as two).
    // TODO: Move to projection builder.
    Alias(Box<Expr<T>>, String),
    /// Case expression.
    Case {
        /// Base case expression.
        expr: Option<Box<Expr<T>>>,
        /// A collection of `WHEN <expr> THEN <expr>` clauses.
        when_then_exprs: Vec<(Expr<T>, Expr<T>)>,
        /// `ELSE <expr>`.
        else_expr: Option<Box<Expr<T>>>,
    },
    /// An ordered list of expressions.
    Tuple(Vec<Expr<T>>),
    /// An array expression.
    Array(Vec<Expr<T>>),
    /// Array access expression. (eg. `arr[0]` ).
    ArrayIndex { expr: Box<Expr<T>>, index: Box<Expr<T>> },
    /// Array slice expression `arr[0:1]` with an optional stride `[0:10:2]`.
    ArraySlice {
        expr: Box<Expr<T>>,
        lower_bound: Option<Box<Expr<T>>>,
        upper_bound: Option<Box<Expr<T>>>,
        stride: Option<Box<Expr<T>>>,
    },
    /// A scalar function expression.
    ScalarFunction { func: ScalarFunction, args: Vec<Expr<T>> },
    /// A `[NOT] LIKE <pattern> [ESCAPE <escape_character>]` expression.
    Like {
        /// Negated or not.
        not: bool,
        /// The Expression.
        expr: Box<Expr<T>>,
        /// A pattern.
        pattern: Box<Expr<T>>,
        /// An escape character.
        escape_char: Option<String>,
        /// Case-insensitive or not.
        case_insensitive: bool,
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
    /// A window aggregate expression.
    WindowAggregate {
        /// The window function.
        func: WindowOrAggregateFunction,
        /// A list of arguments.
        args: Vec<Expr<T>>,
        /// A partitioning.
        partition_by: Vec<Expr<T>>,
        /// An ordering within the partition.
        order_by: Vec<Expr<T>>,
    },
    /// The expression in ORDER BY clause with its options.
    Ordering {
        /// The expression.
        expr: Box<Expr<T>>,
        /// Whether ordering is descending or not.
        descending: bool,
        /// NULLS first/last.
        nulls_first: Option<bool>,
    },
    /// A subquery expression.
    //TODO: Implement table subquery operators such as ALL, ANY, SOME, UNIQUE.
    SubQuery(T),
    /// An `EXIST / NOT EXISTS (<subquery>)` expression.
    Exists { not: bool, query: T },
    /// `IN / NOT IN (<subquery>)` expression.
    InSubQuery { not: bool, expr: Box<Expr<T>>, query: T },
    /// Wildcard expression (eg. `*`, `alias.*` etc).
    Wildcard(Option<String>),
}

/// Trait that must be implemented by other expressions that can be nested inside [Expr].
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
            Expr::IsTrue { expr, .. } => {
                expr.accept(visitor)?;
            }
            Expr::IsFalse { expr, .. } => {
                expr.accept(visitor)?;
            }
            Expr::IsUnknown { expr, .. } => {
                expr.accept(visitor)?;
            }
            Expr::Between { expr, low, high, .. } => {
                expr.accept(visitor)?;
                low.accept(visitor)?;
                high.accept(visitor)?;
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
            Expr::Tuple(exprs) => {
                for expr in exprs {
                    expr.accept(visitor)?;
                }
            }
            Expr::Array(exprs) => {
                for expr in exprs {
                    expr.accept(visitor)?
                }
            }
            Expr::ArrayIndex { expr, index } => {
                expr.accept(visitor)?;
                index.accept(visitor)?;
            }
            Expr::ArraySlice {
                expr,
                lower_bound,
                upper_bound,
                stride,
            } => {
                expr.accept(visitor)?;
                if let Some(expr) = lower_bound {
                    expr.accept(visitor)?;
                }
                if let Some(expr) = upper_bound {
                    expr.accept(visitor)?;
                }
                if let Some(expr) = stride {
                    expr.accept(visitor)?;
                }
            }
            Expr::ScalarFunction { args, .. } => {
                for arg in args {
                    arg.accept(visitor)?;
                }
            }
            Expr::Like { expr, pattern, .. } => {
                expr.accept(visitor)?;
                pattern.accept(visitor)?;
            }
            Expr::Aggregate { args, filter, .. } => {
                for arg in args {
                    arg.accept(visitor)?;
                }
                if let Some(f) = filter.as_ref() {
                    f.accept(visitor)?;
                }
            }
            Expr::WindowAggregate {
                args,
                partition_by,
                order_by,
                ..
            } => {
                for arg in args {
                    arg.accept(visitor)?;
                }
                for partition in partition_by {
                    partition.accept(visitor)?;
                }
                for order in order_by {
                    order.accept(visitor)?;
                }
            }
            Expr::Ordering { expr, .. } => expr.accept(visitor)?,
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
            Expr::ArrayIndex { expr, index } => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                let index = rewrite_boxed(*index, rewriter)?;
                Expr::ArrayIndex { expr, index }
            }
            Expr::ArraySlice {
                expr,
                lower_bound,
                upper_bound,
                stride,
            } => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                let lower_bound = rewrite_boxed_option(lower_bound, rewriter)?;
                let upper_bound = rewrite_boxed_option(upper_bound, rewriter)?;
                let stride = rewrite_boxed_option(stride, rewriter)?;
                Expr::ArraySlice {
                    expr,
                    lower_bound,
                    upper_bound,
                    stride,
                }
            }
            Expr::IsNull { not, expr } => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                Expr::IsNull { not, expr }
            }
            Expr::IsFalse { not, expr } => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                Expr::IsFalse { not, expr }
            }
            Expr::IsTrue { not, expr } => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                Expr::IsTrue { not, expr }
            }
            Expr::IsUnknown { not, expr } => {
                let expr = rewrite_boxed(*expr, rewriter)?;
                Expr::IsUnknown { not, expr }
            }
            Expr::Between { not, expr, low, high } => Expr::Between {
                not,
                expr: rewrite_boxed(*expr, rewriter)?,
                low: rewrite_boxed(*low, rewriter)?,
                high: rewrite_boxed(*high, rewriter)?,
            },
            Expr::Case {
                expr,
                when_then_exprs,
                else_expr,
            } => Expr::Case {
                expr: rewrite_boxed_option(expr, rewriter)?,
                when_then_exprs: rewrite_pairs_vec(when_then_exprs, rewriter)?,
                else_expr: rewrite_boxed_option(else_expr, rewriter)?,
            },
            Expr::Tuple(exprs) => Expr::Tuple(rewrite_vec(exprs, rewriter)?),
            Expr::Array(exprs) => Expr::Array(rewrite_vec(exprs, rewriter)?),
            Expr::ScalarFunction { func, args } => Expr::ScalarFunction {
                func,
                args: rewrite_vec(args, rewriter)?,
            },
            Expr::Like {
                not,
                expr,
                pattern,
                escape_char,
                case_insensitive,
            } => Expr::Like {
                not,
                expr: rewrite_boxed(*expr, rewriter)?,
                pattern: rewrite_boxed(*pattern, rewriter)?,
                escape_char,
                case_insensitive,
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
            Expr::WindowAggregate {
                func,
                args,
                partition_by,
                order_by,
            } => Expr::WindowAggregate {
                func,
                args: rewrite_vec(args, rewriter)?,
                partition_by: rewrite_vec(partition_by, rewriter)?,
                order_by: rewrite_vec(order_by, rewriter)?,
            },
            Expr::Ordering {
                expr,
                descending,
                nulls_first: nulls,
            } => Expr::Ordering {
                expr: rewrite_boxed(*expr, rewriter)?,
                descending,
                nulls_first: nulls,
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
                result.extend(exprs.clone());
                result
            }
            Expr::IsNull { expr, .. } => vec![expr.as_ref().clone()],
            Expr::IsFalse { expr, .. } => vec![expr.as_ref().clone()],
            Expr::IsTrue { expr, .. } => vec![expr.as_ref().clone()],
            Expr::IsUnknown { expr, .. } => vec![expr.as_ref().clone()],
            Expr::Between { expr, low, high, .. } => {
                vec![expr.as_ref().clone(), low.as_ref().clone(), high.as_ref().clone()]
            }
            Expr::Tuple(exprs) => exprs.clone(),
            Expr::Array(exprs) => exprs.clone(),
            Expr::ArrayIndex { expr, index } => {
                vec![*expr.clone(), *index.clone()]
            }
            Expr::ArraySlice {
                expr,
                lower_bound,
                upper_bound,
                stride,
            } => {
                let mut result = vec![];
                result.push(*expr.clone());
                if let Some(expr) = &lower_bound {
                    result.push(*expr.clone());
                }
                if let Some(expr) = &upper_bound {
                    result.push(*expr.clone());
                }
                if let Some(expr) = &stride {
                    result.push(*expr.clone());
                }
                result
            }
            Expr::ScalarFunction { args, .. } => args.clone(),
            Expr::Like { expr, pattern, .. } => {
                vec![expr.as_ref().clone(), pattern.as_ref().clone()]
            }
            Expr::Aggregate { args, filter, .. } => {
                let mut children: Vec<_> = args.to_vec();
                if let Some(filter) = filter.clone() {
                    children.push(filter.as_ref().clone())
                }
                children
            }
            Expr::WindowAggregate {
                args,
                partition_by,
                order_by,
                ..
            } => {
                let mut children: Vec<_> = args.to_vec();
                children.extend_from_slice(partition_by);
                children.extend_from_slice(order_by);
                children
            }
            Expr::Ordering { expr, .. } => vec![expr.as_ref().clone()],
            Expr::SubQuery(_) => vec![],
            Expr::Exists { .. } => vec![],
            Expr::InSubQuery { .. } => vec![],
            Expr::Wildcard(_) => vec![],
        }
    }

    /// Creates a new expressions that retains all properties of this expression but contains the given child expressions.
    /// This method returns an error if the number of elements in `children` is not equal to the
    /// number of child expressions of this expression.
    pub fn with_children(&self, mut children: Vec<Expr<T>>) -> Result<Self, OptimizerError> {
        fn expect_children(expr: &str, actual: usize, expected: usize) -> Result<(), OptimizerError> {
            if actual != expected {
                let message = format!(
                    "{}: Unexpected number of child expressions. Expected {} but got {}",
                    expr, expected, actual
                );
                Err(OptimizerError::argument(message))
            } else {
                Ok(())
            }
        }

        let expr = match self {
            Expr::Column(id) => {
                expect_children("Column", children.len(), 0)?;
                Expr::Column(*id)
            }
            Expr::ColumnName(name) => {
                expect_children("ColumnName", children.len(), 0)?;
                Expr::ColumnName(name.clone())
            }
            Expr::Scalar(value) => {
                expect_children("Scalar", children.len(), 0)?;
                Expr::Scalar(value.clone())
            }
            Expr::BinaryExpr { op, .. } => {
                expect_children("BinaryExpr", children.len(), 2)?;
                Expr::BinaryExpr {
                    lhs: Box::new(children.swap_remove(0)),
                    op: op.clone(),
                    rhs: Box::new(children.swap_remove(0)),
                }
            }
            Expr::Cast { data_type, .. } => {
                expect_children("Cast", children.len(), 1)?;
                Expr::Cast {
                    expr: Box::new(children.swap_remove(0)),
                    data_type: data_type.clone(),
                }
            }
            Expr::Not(_) => {
                expect_children("Not", children.len(), 1)?;
                Expr::Not(Box::new(children.swap_remove(0)))
            }
            Expr::Negation(_) => {
                expect_children("Negation", children.len(), 1)?;
                Expr::Not(Box::new(children.swap_remove(0)))
            }
            Expr::Alias(_, name) => {
                expect_children("Alias", children.len(), 1)?;
                Expr::Alias(Box::new(children.swap_remove(0)), name.clone())
            }
            Expr::InList { not, exprs, .. } => {
                expect_children("InList", children.len(), 1 + exprs.len())?;
                Expr::InList {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                    exprs: children,
                }
            }
            Expr::IsNull { not, .. } => {
                expect_children("IsNull", children.len(), 1)?;
                Expr::IsNull {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                }
            }
            Expr::IsFalse { not, .. } => {
                expect_children("IsFalse", children.len(), 1)?;
                Expr::IsFalse {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                }
            }
            Expr::IsTrue { not, .. } => {
                expect_children("IsTrue", children.len(), 1)?;
                Expr::IsFalse {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                }
            }
            Expr::IsUnknown { not, .. } => {
                expect_children("IsUnknown", children.len(), 1)?;
                Expr::IsFalse {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                }
            }
            Expr::Between { not, .. } => {
                expect_children("Between", children.len(), 3)?;
                Expr::Between {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                    low: Box::new(children.swap_remove(0)),
                    high: Box::new(children.swap_remove(0)),
                }
            }
            Expr::Case {
                expr,
                when_then_exprs,
                else_expr,
            } => {
                let opt_num =
                    expr.as_ref().map(|_| 1).unwrap_or_default() + else_expr.as_ref().map(|_| 1).unwrap_or_default();
                expect_children("Case", children.len(), opt_num + when_then_exprs.len() * 2)?;

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
            Expr::Tuple(exprs) => {
                expect_children("Tuple", children.len(), exprs.len())?;
                Expr::Tuple(children)
            }
            Expr::Array(exprs) => {
                expect_children("Array", children.len(), exprs.len())?;
                Expr::Array(children)
            }
            Expr::ArrayIndex { .. } => {
                expect_children("ArrayIndex", children.len(), 2)?;
                Expr::ArrayIndex {
                    expr: Box::new(children.swap_remove(0)),
                    index: Box::new(children.swap_remove(0)),
                }
            }
            Expr::ArraySlice {
                lower_bound,
                upper_bound,
                stride,
                ..
            } => {
                let opt_num = lower_bound.as_ref().map(|_| 1).unwrap_or_default()
                    + upper_bound.as_ref().map(|_| 1).unwrap_or_default()
                    + stride.as_ref().map(|_| 1).unwrap_or_default();

                expect_children("ArraySlice", children.len(), 1 + opt_num)?;

                let expr = children.swap_remove(0);
                let lower_bound = if let Some(_) = lower_bound {
                    Some(children.swap_remove(0))
                } else {
                    None
                };
                let upper_bound = if let Some(_) = upper_bound {
                    Some(children.swap_remove(0))
                } else {
                    None
                };
                let stride = if let Some(_) = stride {
                    Some(children.swap_remove(0))
                } else {
                    None
                };

                Expr::ArraySlice {
                    expr: Box::new(expr),
                    lower_bound: lower_bound.map(Box::new),
                    upper_bound: upper_bound.map(Box::new),
                    stride: stride.map(Box::new),
                }
            }
            Expr::ScalarFunction { func, args } => {
                expect_children("ScalarFunction", children.len(), args.len())?;
                Expr::ScalarFunction {
                    func: func.clone(),
                    args: children,
                }
            }
            Expr::Like {
                not,
                escape_char,
                case_insensitive,
                ..
            } => {
                expect_children("Like", children.len(), 2)?;
                Expr::Like {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                    pattern: Box::new(children.swap_remove(0)),
                    escape_char: escape_char.clone(),
                    case_insensitive: *case_insensitive,
                }
            }
            Expr::Aggregate {
                func,
                distinct,
                args,
                filter,
            } => {
                expect_children("Aggregate", children.len(), args.len() + filter.as_ref().map(|_| 1).unwrap_or(0))?;
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
            Expr::WindowAggregate {
                func,
                args,
                partition_by,
                order_by,
            } => {
                expect_children("WindowFunction", children.len(), args.len() + partition_by.len() + order_by.len())?;
                let args: Vec<Expr<T>> = children.drain(0..args.len()).collect();
                let partition_by: Vec<Expr<T>> = children.drain(0..partition_by.len()).collect();
                let order_by: Vec<Expr<T>> = children.drain(0..order_by.len()).collect();

                Expr::WindowAggregate {
                    func: func.clone(),
                    args,
                    partition_by,
                    order_by,
                }
            }
            Expr::Ordering {
                descending,
                nulls_first,
                ..
            } => {
                expect_children("Ordering", children.len(), 1)?;
                Expr::Ordering {
                    expr: Box::new(children.swap_remove(0)),
                    descending: *descending,
                    nulls_first: *nulls_first,
                }
            }
            Expr::SubQuery(node) => {
                expect_children("SubQuery", children.len(), 0)?;
                Expr::SubQuery(node.clone())
            }
            Expr::Exists { not, query } => {
                expect_children("Exists", children.len(), 0)?;
                Expr::Exists {
                    not: *not,
                    query: query.clone(),
                }
            }
            Expr::InSubQuery { not, query, .. } => {
                expect_children("InSubQuery", children.len(), 1)?;
                Expr::InSubQuery {
                    not: *not,
                    expr: Box::new(children.swap_remove(0)),
                    query: query.clone(),
                }
            }
            Expr::Wildcard(qualifier) => {
                expect_children("Wildcard", children.len(), 0)?;
                Expr::Wildcard(qualifier.clone())
            }
        };
        Ok(expr)
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

    /// Returns `this_expr LIKE pattern [ESCAPE <escape_char>]`
    pub fn like(self, pattern: Expr<T>, escape_char: Option<char>) -> Self {
        Expr::Like {
            not: false,
            expr: Box::new(self),
            pattern: Box::new(pattern),
            escape_char: escape_char.map(|c| c.to_string()),
            case_insensitive: false,
        }
    }

    /// Returns `this_expr ILIKE pattern [ESCAPE <escape_char>]`
    pub fn ilike(self, pattern: Expr<T>, escape_char: Option<char>) -> Self {
        Expr::Like {
            not: false,
            expr: Box::new(self),
            pattern: Box::new(pattern),
            escape_char: escape_char.map(|c| c.to_string()),
            case_insensitive: true,
        }
    }

    /// Returns `this_expr IS UNKNOWN` expression.
    pub fn is_null(self) -> Self {
        Expr::IsNull {
            not: false,
            expr: Box::new(self),
        }
    }

    /// Returns `this_expr IS FALSE` expression.
    pub fn is_false(self) -> Self {
        Expr::IsFalse {
            not: false,
            expr: Box::new(self),
        }
    }

    /// Returns `this_expr IS TRUE` expression.
    pub fn is_true(self) -> Self {
        Expr::IsTrue {
            not: false,
            expr: Box::new(self),
        }
    }

    /// Returns `this_expr IS UNKNOWN` expression.
    pub fn is_unknown(self) -> Self {
        Expr::IsUnknown {
            not: false,
            expr: Box::new(self),
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
    /// String concatenation.
    Concat,
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
            Expr::ScalarFunction { func, args } => {
                write!(f, "{}(", func)?;
                write!(f, "{})", DisplayArgs(args))
            }
            Expr::Like {
                not,
                expr,
                pattern,
                escape_char,
                case_insensitive,
            } => {
                write!(f, "{} ", expr)?;

                if *not {
                    write!(f, "NOT ")?;
                }

                if *case_insensitive {
                    write!(f, "ILIKE {}", pattern)?;
                } else {
                    write!(f, "LIKE {}", pattern)?;
                }

                if let Some(escape_char) = escape_char {
                    write!(f, " ESCAPE {}", escape_char)?;
                }

                Ok(())
            }
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
            Expr::WindowAggregate {
                func,
                args,
                partition_by,
                order_by,
            } => {
                write!(f, "{}(", func)?;
                write!(f, "{})", DisplayArgs(args))?;
                write!(f, " OVER(")?;
                if !partition_by.is_empty() {
                    write!(f, "PARTITION BY {}", partition_by.iter().join(", "))?;
                }
                if !order_by.is_empty() {
                    write!(
                        f,
                        "{pad}ORDER BY {}",
                        order_by.iter().join(", "),
                        pad = if partition_by.is_empty() { "" } else { " " }
                    )?;
                }
                write!(f, ")")
            }
            Expr::Not(expr) => write!(f, "NOT {}", expr),
            Expr::Negation(expr) => write!(f, "-{}", expr),
            Expr::Alias(expr, name) => write!(f, "{} AS {}", expr, name),
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
                    write!(f, "{} NOT IN (", expr)?;
                } else {
                    write!(f, "{} IN (", expr)?;
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
                    write!(f, "{} IS NOT NULL", expr)
                } else {
                    write!(f, "{} IS NULL", expr)
                }
            }
            Expr::IsFalse { not, expr } => {
                if *not {
                    write!(f, "{} IS NOT FALSE", expr)
                } else {
                    write!(f, "{} IS FALSE", expr)
                }
            }
            Expr::IsTrue { not, expr } => {
                if *not {
                    write!(f, "{} IS NOT TRUE", expr)
                } else {
                    write!(f, "{} IS TRUE", expr)
                }
            }
            Expr::IsUnknown { not, expr } => {
                if *not {
                    write!(f, "{} IS NOT UNKNOWN", expr)
                } else {
                    write!(f, "{} IS UNKNOWN", expr)
                }
            }
            Expr::Between { not, expr, low, high } => {
                if *not {
                    write!(f, "{} NOT BETWEEN {} AND {}", expr, low, high)
                } else {
                    write!(f, "{} BETWEEN {} AND {}", expr, low, high)
                }
            }
            Expr::Tuple(exprs) => {
                write!(f, "({})", exprs.iter().join(", "))
            }
            Expr::Array(exprs) => {
                write!(f, "[{}]", exprs.iter().join(", "))
            }
            Expr::ArrayIndex { expr, index } => {
                write!(f, "{}[{}]", expr, index)
            }
            Expr::ArraySlice {
                expr,
                lower_bound,
                upper_bound,
                stride,
            } => {
                write!(f, "{}[", expr)?;
                for (i, expr) in vec![lower_bound, upper_bound, stride]
                    .into_iter()
                    .filter(|e| e.is_some())
                    .map(|e| e.as_ref().unwrap())
                    .enumerate()
                {
                    if i > 0 {
                        write!(f, ":")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, "]")
            }
            Expr::Ordering {
                expr,
                descending,
                nulls_first: nulls,
            } => {
                write!(f, "{}", expr)?;
                if *descending {
                    write!(f, " DESC")?;
                }
                match nulls {
                    Some(true) => write!(f, " NULLS FIRST"),
                    Some(false) => write!(f, " NULLS LAST"),
                    _ => Ok(()),
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
            BinaryOp::Concat => write!(f, "||"),
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
        match self {
            Expr::Column(_) => Expr::Not(Box::new(self)),
            Expr::ColumnName(_) => Expr::Not(Box::new(self)),
            Expr::Scalar(_) => Expr::Not(Box::new(self)),
            Expr::BinaryExpr { .. } => Expr::Not(Box::new(self)),
            Expr::Cast { .. } => Expr::Not(Box::new(self)),
            Expr::Not(_) => Expr::Not(Box::new(self)),
            Expr::Negation(_) => Expr::Not(Box::new(self)),
            Expr::InList { not, expr, exprs } => Expr::InList { not: !not, expr, exprs },
            Expr::IsNull { not, expr } => Expr::IsNull { not: !not, expr },
            Expr::IsFalse { not, expr } => Expr::IsFalse { not: !not, expr },
            Expr::IsTrue { not, expr } => Expr::IsTrue { not: !not, expr },
            Expr::IsUnknown { not, expr } => Expr::IsUnknown { not: !not, expr },
            Expr::Between { not, expr, low, high } => Expr::Between {
                not: !not,
                expr,
                low,
                high,
            },
            Expr::Alias(_, _) => Expr::Not(Box::new(self)),
            Expr::Case { .. } => Expr::Not(Box::new(self)),
            Expr::Tuple(_) => Expr::Not(Box::new(self)),
            Expr::Array(_) => Expr::Not(Box::new(self)),
            Expr::ArrayIndex { .. } => Expr::Not(Box::new(self)),
            Expr::ArraySlice { .. } => Expr::Not(Box::new(self)),
            Expr::ScalarFunction { .. } => Expr::Not(Box::new(self)),
            Expr::Like {
                not,
                expr,
                pattern,
                escape_char,
                case_insensitive,
            } => Expr::Like {
                not: !not,
                expr,
                pattern,
                escape_char,
                case_insensitive,
            },
            Expr::Aggregate { .. } => Expr::Not(Box::new(self)),
            Expr::WindowAggregate { .. } => Expr::Not(Box::new(self)),
            Expr::Ordering { .. } => Expr::Not(Box::new(self)),
            Expr::SubQuery(_) => Expr::Not(Box::new(self)),
            Expr::Exists { not, query } => Expr::Exists { not: !not, query },
            Expr::InSubQuery { not, expr, query } => Expr::InSubQuery { not: !not, expr, query },
            Expr::Wildcard(_) => Expr::Not(Box::new(self)),
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
        match self.0.len() {
            0 => Ok(()),
            1 => write!(f, "{}", self.0[0]),
            _ => write!(f, "[{}]", self.0.iter().join(", ").as_str()),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::operators::scalar::aggregates::WindowFunction;
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

    fn str_val(val: &str) -> Expr {
        Expr::Scalar(ScalarValue::String(Some(String::from(val))))
    }

    fn bool_val(val: bool) -> Expr {
        Expr::Scalar(ScalarValue::Bool(Some(val)))
    }

    fn int_val(val: i32) -> Expr {
        Expr::Scalar(ScalarValue::Int32(Some(val)))
    }

    #[test]
    fn expr_methods() {
        let expr = int_val(1);
        let rhs = int_val(10);

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
    fn string_concat() {
        let expr = str_val("hello");
        let rhs = str_val("world");
        expect_expr(expr.binary_expr(BinaryOp::Concat, rhs), "hello || world");
    }

    #[test]
    fn column_traversal() {
        let expr = col("a1");
        expect_traversal_order(&expr, vec!["pre:col:a1", "post:col:a1"]);
    }

    #[test]
    fn scalar_traversal() {
        let expr = int_val(1);
        expect_traversal_order(&expr, vec!["pre:1", "post:1"]);
    }

    #[test]
    fn binary_expr_traversal() {
        let expr = int_val(1).ne(int_val(2));
        expect_traversal_order(&expr, vec!["pre:1 != 2", "pre:1", "post:1", "pre:2", "post:2", "post:1 != 2"]);
    }

    #[test]
    fn not_expr_traversal() {
        let expr = bool_val(true).not();
        expect_traversal_order(&expr, vec!["pre:NOT true", "pre:true", "post:true", "post:NOT true"])
    }

    #[test]
    fn is_null_expr_traversal() {
        let expr = bool_val(true).is_null();
        expect_traversal_order(&expr, vec!["pre:true IS NULL", "pre:true", "post:true", "post:true IS NULL"])
    }

    #[test]
    fn is_not_null_expr_traversal() {
        let expr = bool_val(true).is_null().not();
        expect_traversal_order(&expr, vec!["pre:true IS NOT NULL", "pre:true", "post:true", "post:true IS NOT NULL"])
    }

    #[test]
    fn is_false_expr_traversal() {
        let expr = bool_val(true).is_false();
        expect_traversal_order(&expr, vec!["pre:true IS FALSE", "pre:true", "post:true", "post:true IS FALSE"])
    }

    #[test]
    fn is_not_false_expr_traversal() {
        let expr = bool_val(true).is_false().not();
        expect_traversal_order(&expr, vec!["pre:true IS NOT FALSE", "pre:true", "post:true", "post:true IS NOT FALSE"])
    }

    #[test]
    fn is_true_expr_traversal() {
        let expr = bool_val(true).is_true();
        expect_traversal_order(&expr, vec!["pre:true IS TRUE", "pre:true", "post:true", "post:true IS TRUE"])
    }

    #[test]
    fn is_not_true_expr_traversal() {
        let expr = bool_val(true).is_true().not();
        expect_traversal_order(&expr, vec!["pre:true IS NOT TRUE", "pre:true", "post:true", "post:true IS NOT TRUE"])
    }

    #[test]
    fn is_unknown_expr_traversal() {
        let expr = bool_val(true).is_unknown();
        expect_traversal_order(&expr, vec!["pre:true IS UNKNOWN", "pre:true", "post:true", "post:true IS UNKNOWN"])
    }

    #[test]
    fn is_not_unknown_expr_traversal() {
        let expr = bool_val(true).is_unknown().not();
        expect_traversal_order(
            &expr,
            vec!["pre:true IS NOT UNKNOWN", "pre:true", "post:true", "post:true IS NOT UNKNOWN"],
        )
    }

    #[test]
    fn in_list_expr_traversal() {
        let expr = Expr::InList {
            not: false,
            expr: Box::new(bool_val(false)),
            exprs: vec![bool_val(true), int_val(1)],
        };
        expect_traversal_order_with_depth(
            &expr,
            1,
            vec!["pre:false IN (true, 1)", "false", "true", "1", "post:false IN (true, 1)"],
        )
    }

    #[test]
    fn not_in_list_expr_traversal() {
        let expr = Expr::InList {
            not: false,
            expr: Box::new(bool_val(false)),
            exprs: vec![bool_val(true), int_val(1)],
        }
        .not();
        expect_traversal_order_with_depth(
            &expr,
            1,
            vec!["pre:false NOT IN (true, 1)", "false", "true", "1", "post:false NOT IN (true, 1)"],
        )
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
    fn function_expr_traversal() {
        let expr = Expr::ScalarFunction {
            func: ScalarFunction::BitLength,
            args: vec![col("a1")],
        };
        expect_traversal_order(
            &expr,
            vec!["pre:bit_length(col:a1)", "pre:col:a1", "post:col:a1", "post:bit_length(col:a1)"],
        )
    }

    #[test]
    fn aggr_expr_traversal_with_filter() {
        let expr = Expr::Aggregate {
            func: AggregateFunction::Avg,
            distinct: false,
            args: vec![col("a1")],
            filter: Some(Box::new(bool_val(true))),
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
    fn like_expr_traversal() {
        let expr = str_val("str").like(str_val("p"), None);
        expect_traversal_order(
            &expr,
            vec!["pre:str LIKE p", "pre:str", "post:str", "pre:p", "post:p", "post:str LIKE p"],
        );

        let expr = str_val("str").like(str_val("p"), Some('c'));
        expect_traversal_order(
            &expr,
            vec![
                "pre:str LIKE p ESCAPE c",
                "pre:str",
                "post:str",
                "pre:p",
                "post:p",
                "post:str LIKE p ESCAPE c",
            ],
        )
    }

    #[test]
    fn not_like_expr_traversal() {
        let expr = str_val("str").like(str_val("p"), None).not();
        expect_traversal_order(
            &expr,
            vec!["pre:str NOT LIKE p", "pre:str", "post:str", "pre:p", "post:p", "post:str NOT LIKE p"],
        );

        let expr = str_val("str").like(str_val("p"), Some('c')).not();
        expect_traversal_order(
            &expr,
            vec![
                "pre:str NOT LIKE p ESCAPE c",
                "pre:str",
                "post:str",
                "pre:p",
                "post:p",
                "post:str NOT LIKE p ESCAPE c",
            ],
        )
    }

    #[test]
    fn ilike_expr_traversal() {
        let expr = str_val("str").ilike(str_val("p"), None);
        expect_traversal_order(
            &expr,
            vec!["pre:str ILIKE p", "pre:str", "post:str", "pre:p", "post:p", "post:str ILIKE p"],
        );

        let expr = str_val("str").ilike(str_val("p"), Some('c'));
        expect_traversal_order(
            &expr,
            vec![
                "pre:str ILIKE p ESCAPE c",
                "pre:str",
                "post:str",
                "pre:p",
                "post:p",
                "post:str ILIKE p ESCAPE c",
            ],
        )
    }

    #[test]
    fn not_ilike_expr_traversal() {
        let expr = str_val("str").ilike(str_val("p"), None).not();
        expect_traversal_order(
            &expr,
            vec!["pre:str NOT ILIKE p", "pre:str", "post:str", "pre:p", "post:p", "post:str NOT ILIKE p"],
        );

        let expr = str_val("str").ilike(str_val("p"), Some('c')).not();
        expect_traversal_order(
            &expr,
            vec![
                "pre:str NOT ILIKE p ESCAPE c",
                "pre:str",
                "post:str",
                "pre:p",
                "post:p",
                "post:str NOT ILIKE p ESCAPE c",
            ],
        )
    }

    #[test]
    fn case_traversal() {
        let col1 = col("a1");

        let expr = Expr::Case {
            expr: Some(Box::new(col1.clone().gt(int_val(10)))),
            when_then_exprs: vec![
                (col1.clone().eq(int_val(11)), int_val(10) + int_val(1)),
                (col1.clone().eq(int_val(12)), int_val(10) + int_val(2)),
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
    fn between_expr_traversal() {
        let expr = Expr::Between {
            not: false,
            expr: Box::new(col("a1")),
            low: Box::new(int_val(2)),
            high: Box::new(int_val(4)),
        };
        expect_traversal_order(
            &expr,
            vec![
                "pre:col:a1 BETWEEN 2 AND 4",
                "pre:col:a1",
                "post:col:a1",
                "pre:2",
                "post:2",
                "pre:4",
                "post:4",
                "post:col:a1 BETWEEN 2 AND 4",
            ],
        )
    }

    #[test]
    fn not_between_expr_traversal() {
        let expr = Expr::Between {
            not: true,
            expr: Box::new(col("a1")),
            low: Box::new(int_val(2)),
            high: Box::new(int_val(4)),
        };

        let expected = vec![
            "pre:col:a1 NOT BETWEEN 2 AND 4",
            "pre:col:a1",
            "post:col:a1",
            "pre:2",
            "post:2",
            "pre:4",
            "post:4",
            "post:col:a1 NOT BETWEEN 2 AND 4",
        ];
        expect_traversal_order(&expr, expected.clone());

        let expr = Expr::Between {
            not: false,
            expr: Box::new(col("a1")),
            low: Box::new(int_val(2)),
            high: Box::new(int_val(4)),
        }
        .not();
        expect_traversal_order(&expr, expected)
    }

    #[test]
    fn tuple_traversal() {
        let expr = Expr::Tuple(vec![bool_val(true), int_val(1), col("a1")]);
        expect_traversal_order(
            &expr,
            vec![
                "pre:(true, 1, col:a1)",
                "pre:true",
                "post:true",
                "pre:1",
                "post:1",
                "pre:col:a1",
                "post:col:a1",
                "post:(true, 1, col:a1)",
            ],
        )
    }

    #[test]
    fn array_traversal() {
        let expr = Expr::Array(vec![bool_val(true), bool_val(false), col("a1")]);
        expect_traversal_order(
            &expr,
            vec![
                "pre:[true, false, col:a1]",
                "pre:true",
                "post:true",
                "pre:false",
                "post:false",
                "pre:col:a1",
                "post:col:a1",
                "post:[true, false, col:a1]",
            ],
        )
    }

    #[test]
    fn array_slice() {
        let expr = Expr::ArraySlice {
            expr: Box::new(col("a")),
            lower_bound: None,
            upper_bound: None,
            stride: None,
        };
        expect_traversal_order(&expr, vec!["pre:col:a[]", "pre:col:a", "post:col:a", "post:col:a[]"]);

        let expr = Expr::ArraySlice {
            expr: Box::new(col("a")),
            lower_bound: Some(Box::new(int_val(0))),
            upper_bound: None,
            stride: None,
        };
        expect_traversal_order(
            &expr,
            vec!["pre:col:a[0]", "pre:col:a", "post:col:a", "pre:0", "post:0", "post:col:a[0]"],
        );

        let expr = Expr::ArraySlice {
            expr: Box::new(col("a")),
            lower_bound: Some(Box::new(int_val(0))),
            upper_bound: Some(Box::new(int_val(5))),
            stride: None,
        };
        expect_traversal_order(
            &expr,
            vec![
                "pre:col:a[0:5]",
                "pre:col:a",
                "post:col:a",
                "pre:0",
                "post:0",
                "pre:5",
                "post:5",
                "post:col:a[0:5]",
            ],
        );

        let expr = Expr::ArraySlice {
            expr: Box::new(col("a")),
            lower_bound: Some(Box::new(int_val(0))),
            upper_bound: Some(Box::new(int_val(5))),
            stride: Some(Box::new(int_val(2))),
        };
        expect_traversal_order(
            &expr,
            vec![
                "pre:col:a[0:5:2]",
                "pre:col:a",
                "post:col:a",
                "pre:0",
                "post:0",
                "pre:5",
                "post:5",
                "pre:2",
                "post:2",
                "post:col:a[0:5:2]",
            ],
        )
    }

    #[test]
    fn window_aggregate_traversal() {
        fn ordering(col_name: &str) -> Vec<Expr> {
            let expr = Expr::Ordering {
                expr: Box::new(col(col_name)),
                descending: true,
                nulls_first: None,
            };
            vec![expr]
        }

        let expr = Expr::WindowAggregate {
            func: WindowFunction::FirstValue.into(),
            args: vec![col("a1")],
            partition_by: vec![col("a2")],
            order_by: ordering("a3"),
        };
        expect_traversal_order(
            &expr,
            vec![
                "pre:first_value(col:a1) OVER(PARTITION BY col:a2 ORDER BY col:a3 DESC)",
                "pre:col:a1",
                "post:col:a1",
                "pre:col:a2",
                "post:col:a2",
                "pre:col:a3 DESC",
                "pre:col:a3",
                "post:col:a3",
                "post:col:a3 DESC",
                "post:first_value(col:a1) OVER(PARTITION BY col:a2 ORDER BY col:a3 DESC)",
            ],
        );

        let expr = Expr::WindowAggregate {
            func: WindowFunction::FirstValue.into(),
            args: vec![col("a1")],
            partition_by: vec![],
            order_by: ordering("a3"),
        };
        expect_traversal_order(
            &expr,
            vec![
                "pre:first_value(col:a1) OVER(ORDER BY col:a3 DESC)",
                "pre:col:a1",
                "post:col:a1",
                "pre:col:a3 DESC",
                "pre:col:a3",
                "post:col:a3",
                "post:col:a3 DESC",
                "post:first_value(col:a1) OVER(ORDER BY col:a3 DESC)",
            ],
        );

        let expr = Expr::WindowAggregate {
            func: WindowFunction::FirstValue.into(),
            args: vec![col("a1")],
            partition_by: vec![col("a2")],
            order_by: vec![],
        };
        expect_traversal_order(
            &expr,
            vec![
                "pre:first_value(col:a1) OVER(PARTITION BY col:a2)",
                "pre:col:a1",
                "post:col:a1",
                "pre:col:a2",
                "post:col:a2",
                "post:first_value(col:a1) OVER(PARTITION BY col:a2)",
            ],
        );

        let expr = Expr::WindowAggregate {
            func: WindowFunction::FirstValue.into(),
            args: vec![col("a1")],
            partition_by: vec![],
            order_by: vec![],
        };
        expect_traversal_order(
            &expr,
            vec![
                "pre:first_value(col:a1) OVER()",
                "pre:col:a1",
                "post:col:a1",
                "post:first_value(col:a1) OVER()",
            ],
        );
    }

    #[test]
    fn ordering_traversal() {
        let expr = Expr::Ordering {
            expr: Box::new(col("a1")),
            descending: true,
            nulls_first: Some(true),
        };
        expect_traversal_order(
            &expr,
            vec!["pre:col:a1 DESC NULLS FIRST", "pre:col:a1", "post:col:a1", "post:col:a1 DESC NULLS FIRST"],
        );
    }

    #[test]
    fn in_subquery_traversal() {
        let expr = Expr::InSubQuery {
            not: false,
            expr: Box::new(col("a1")),
            query: DummyRelExpr,
        };
        expect_traversal_order(
            &expr,
            vec!["pre:col:a1 IN DummyRelExpr", "pre:col:a1", "post:col:a1", "post:col:a1 IN DummyRelExpr"],
        );
    }

    #[test]
    fn not_in_subquery_traversal() {
        let expr = Expr::InSubQuery {
            not: false,
            expr: Box::new(col("a1")),
            query: DummyRelExpr,
        }
        .not();
        expect_traversal_order(
            &expr,
            vec![
                "pre:col:a1 NOT IN DummyRelExpr",
                "pre:col:a1",
                "post:col:a1",
                "post:col:a1 NOT IN DummyRelExpr",
            ],
        );
    }

    #[test]
    fn exists_traversal() {
        let expr = Expr::Exists {
            not: false,
            query: DummyRelExpr,
        };
        expect_traversal_order(&expr, vec!["pre:EXISTS DummyRelExpr", "post:EXISTS DummyRelExpr"]);
    }

    #[test]
    fn not_exists_traversal() {
        let expr = Expr::Exists {
            not: false,
            query: DummyRelExpr,
        }
        .not();
        expect_traversal_order(&expr, vec!["pre:NOT EXISTS DummyRelExpr", "post:NOT EXISTS DummyRelExpr"]);
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
        result.expect_err("Expected an error");

        assert_eq!(rewriter.visited, 2);
        assert_eq!(rewriter.rewritten, 1);
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
