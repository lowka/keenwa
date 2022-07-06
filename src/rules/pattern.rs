//! Patterns to match operator trees.

use crate::memo::MemoExpr;
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::relational::RelExpr;
use crate::operators::scalar::ScalarExpr;
use crate::operators::{Operator, OperatorExpr};

/// A type alias for the function type that matches [operators](Operator).
type Matcher = Box<dyn Fn(&Operator) -> bool>;

/// A pattern for matching relational and scalar expressions.
/// Use [PatternBuilder] to create an instance of a pattern.
pub struct Pattern {
    matcher: Matcher,
    inputs: Vec<(usize, Pattern)>,
}

impl Pattern {
    fn new(f: Matcher) -> Self {
        Pattern {
            matcher: f,
            inputs: Vec::new(),
        }
    }

    /// Tests whether the given expression matches this pattern.
    pub fn matches(&self, expr: &Operator) -> bool {
        if !(self.matcher)(expr) {
            false
        } else {
            for (i, input_matcher) in self.inputs.iter() {
                match expr.get_child(*i) {
                    Some(expr) => {
                        if !input_matcher.matches(expr) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
            true
        }
    }
}

/// Builds instances of [Pattern]. A pattern builder can be created using one of the following functions:
/// - [logical_expr()](logical_expr) creates a builder for a pattern that expects a root
/// of an expression tree to be a logical expression.
/// - [physical_expr()](physical_expr) creates a builder for a pattern that expects a root
/// of an expression tree to be a physical expression.
/// - [scalar_expr()](scalar_expr) creates a builder for a pattern that expects a root
/// of an expression tree to be a scalar expression.
///
/// The above functions can be used in combination with the methods of [PatternBuilder] to match
/// complex expression trees. E.g:
///
/// ```
///  # use keenwa::rules::pattern::{logical_expr, physical_expr};
///  # use keenwa::operators::relational::physical::{PhysicalExpr, Projection, Scan};
///  # use keenwa::operators::relational::logical::{LogicalExpr};
///  # use keenwa::operators::scalar::{scalar, ScalarNode, ScalarExpr};
///  # use keenwa::operators::scalar::value::ScalarValue;
///
///  let pattern = logical_expr(|expr| matches!(expr, LogicalExpr::Select(_)))
///   .child(
///       0,
///       physical_expr(|expr| matches!(expr, PhysicalExpr::Projection(_)))
///           .physical(0, |expr| matches!(expr, PhysicalExpr::Scan(_)))
///           .scalar(1, |expr| matches!(expr, ScalarExpr::Scalar(ScalarValue::Bool(Some(true)))))
///           .build(),
///   )
///   .build();
/// ```
/// Matches the following expression tree:
/// ```text
///   LogicalExpr Select
///     input PhysicalExpr Projection
///       input PhysicalExpr Scan
///     filter: ScalarExpr true
/// ```
///
pub struct PatternBuilder {
    matcher: Matcher,
    inputs: Vec<(usize, Pattern)>,
}

impl PatternBuilder {
    /// Expects the `i`-th child expression to match the given pattern.
    pub fn child(mut self, i: usize, pattern: Pattern) -> Self {
        self.inputs.push((i, Pattern::new(Box::new(move |expr| pattern.matches(expr)))));
        self
    }

    /// Expects the `i`-th child expression to match a [logical expression](LogicalExpr).
    pub fn logical<F>(mut self, i: usize, f: F) -> Self
    where
        F: Fn(&LogicalExpr) -> bool + 'static,
    {
        self.inputs.push((i, Pattern::new(logical_expr_matcher(f))));
        self
    }

    /// Expects the `i`-th child expression to match a [physical expression](PhysicalExpr).
    pub fn physical<F>(mut self, i: usize, f: F) -> Self
    where
        F: Fn(&PhysicalExpr) -> bool + 'static,
    {
        self.inputs.push((i, Pattern::new(physical_expr_matcher(f))));
        self
    }

    /// Expects the `i`-th child expression to match a [scalar expression](ScalarExpr).
    pub fn scalar<F>(mut self, i: usize, f: F) -> Self
    where
        F: Fn(&ScalarExpr) -> bool + 'static,
    {
        self.inputs.push((i, Pattern::new(scalar_expr_matcher(f))));
        self
    }

    /// Build an [expression pattern](Pattern).
    pub fn build(self) -> Pattern {
        Pattern {
            matcher: self.matcher,
            inputs: self.inputs,
        }
    }
}

/// Creates a pattern that matches any expression.
/// This method is a shorthand for `any_expr().build()`.
pub fn any() -> Pattern {
    any_expr().build()
}

/// Creates a [pattern builder](PatternBuilder) for a pattern that matches any expression.
pub fn any_expr() -> PatternBuilder {
    PatternBuilder {
        matcher: Box::new(|_| true),
        inputs: Vec::new(),
    }
}

/// Creates a [pattern builder](PatternBuilder) that builds a pattern that matches an expression
/// if a root expression of that expression tree is a [logical expressions](LogicalExpr).
pub fn logical_expr<F>(f: F) -> PatternBuilder
where
    F: Fn(&LogicalExpr) -> bool + 'static,
{
    PatternBuilder {
        matcher: logical_expr_matcher(f),
        inputs: Vec::new(),
    }
}

/// Creates a [pattern builder](PatternBuilder) that builds a pattern that matches an expression
/// if a root expression of that expression tree is a [physical expressions](PhysicalExpr).
pub fn physical_expr<F>(f: F) -> PatternBuilder
where
    F: Fn(&PhysicalExpr) -> bool + 'static,
{
    PatternBuilder {
        matcher: physical_expr_matcher(f),
        inputs: Vec::new(),
    }
}

/// Creates a [pattern builder](PatternBuilder) that builds a pattern that matches an expression
/// if a root expression of that expression tree is a [scalar expressions](ScalarExpr).
pub fn scalar_expr<F>(f: F) -> PatternBuilder
where
    F: Fn(&ScalarExpr) -> bool + 'static,
{
    PatternBuilder {
        matcher: scalar_expr_matcher(f),
        inputs: Vec::new(),
    }
}

fn logical_expr_matcher<F>(f: F) -> Matcher
where
    F: Fn(&LogicalExpr) -> bool + 'static,
{
    Box::new(move |expr| match expr.expr() {
        OperatorExpr::Relational(RelExpr::Logical(expr)) => {
            let expr = &**expr;
            (f)(expr)
        }
        _ => false,
    })
}

fn physical_expr_matcher<F>(f: F) -> Matcher
where
    F: Fn(&PhysicalExpr) -> bool + 'static,
{
    Box::new(move |expr| match expr.expr() {
        OperatorExpr::Relational(RelExpr::Physical(expr)) => {
            let expr = &**expr;
            (f)(expr)
        }
        _ => false,
    })
}

fn scalar_expr_matcher<F>(f: F) -> Matcher
where
    F: Fn(&ScalarExpr) -> bool + 'static,
{
    Box::new(move |expr| match expr.expr() {
        OperatorExpr::Scalar(expr) => (f)(expr),
        _ => false,
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::operators::relational::logical::{LogicalEmpty, LogicalGet, LogicalSelect};
    use crate::operators::relational::physical::{Projection, Scan, Select};
    use crate::operators::relational::RelNode;
    use crate::operators::scalar::value::ScalarValue;
    use crate::operators::scalar::{scalar, ScalarNode};

    #[test]
    fn test_any_expr() {
        let expr = Operator::from(PhysicalExpr::Scan(Scan {
            source: "A".to_string(),
            columns: vec![],
        }));
        assert!(any_expr().build().matches(&expr));
        assert!(any().matches(&expr));

        let select_get = Operator::from(PhysicalExpr::Select(Select {
            input: RelNode::from(LogicalExpr::Get(LogicalGet {
                source: "A".to_string(),
                columns: vec![],
            })),
            filter: None,
        }));

        let any_with_scan = any_expr()
            .child(0, logical_expr(|expr| matches!(expr, LogicalExpr::Get(_))).build())
            .build();

        let select_empty = Operator::from(PhysicalExpr::Select(Select {
            input: RelNode::from(LogicalExpr::Empty(LogicalEmpty { return_one_row: false })),
            filter: None,
        }));

        assert!(any_with_scan.matches(&select_get));
        assert!(!any_with_scan.matches(&select_empty));
    }

    #[test]
    fn test_logical_expr() {
        let select_get = logical_expr(|expr| matches!(expr, LogicalExpr::Select(_)))
            .logical(0, |expr| matches!(expr, LogicalExpr::Get(_)))
            .build();

        let expr = Operator::from(LogicalExpr::Select(LogicalSelect {
            input: RelNode::from(LogicalExpr::Get(LogicalGet {
                source: "A".to_string(),
                columns: vec![],
            })),
            filter: None,
        }));

        assert!(select_get.matches(&expr));
    }

    #[test]
    fn test_physical_expr() {
        let select_get = physical_expr(|expr| matches!(expr, PhysicalExpr::Select(_)))
            .physical(0, |expr| matches!(expr, PhysicalExpr::Scan(_)))
            .build();

        let expr = Operator::from(PhysicalExpr::Select(Select {
            input: RelNode::from(PhysicalExpr::Scan(Scan {
                source: "A".to_string(),
                columns: vec![],
            })),
            filter: None,
        }));

        assert!(select_get.matches(&expr));
    }

    #[test]
    fn test_complex_expr() {
        let projection = physical_expr(|expr| matches!(expr, PhysicalExpr::Projection(_)))
            .child(
                0,
                logical_expr(|expr| matches!(expr, LogicalExpr::Select(_)))
                    .logical(0, |expr| matches!(expr, LogicalExpr::Get(_)))
                    .scalar(1, |expr| matches!(expr, ScalarExpr::Scalar(ScalarValue::Bool(Some(true)))))
                    .build(),
            )
            .build();

        let expr = Operator::from(PhysicalExpr::Projection(Projection {
            input: RelNode::from(LogicalExpr::Select(LogicalSelect {
                input: RelNode::from(LogicalExpr::Get(LogicalGet {
                    source: "A".to_string(),
                    columns: vec![],
                })),
                filter: Some(ScalarNode::from(scalar(true))),
            })),
            exprs: vec![],
            columns: vec![],
        }));

        assert!(projection.matches(&expr));
    }
}
