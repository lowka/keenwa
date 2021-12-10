use crate::memo::{
    CopyInExprs, CopyInNestedExprs, ExprContext, ExprNode, ExprNodeRef, MemoExpr, MemoExprCallback, MemoExprFormatter,
    NewChildExprs,
};
use crate::operators::scalar::expr_with_new_inputs;
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use crate::properties::statistics::Statistics;
use relational::join::JoinCondition;
use relational::logical::LogicalExpr;
use relational::physical::PhysicalExpr;
use relational::{RelExpr, RelNode};
use scalar::expr::ExprVisitor;
use scalar::{ScalarExpr, ScalarNode};
use std::fmt::Debug;

pub mod builder;
pub mod relational;
pub mod scalar;

pub type ExprMemo = crate::memo::Memo<Operator>;
pub type GroupRef = crate::memo::MemoGroupRef<Operator>;
pub type ExprRef = crate::memo::MemoExprRef<Operator>;
pub type ExprCallback = dyn MemoExprCallback<Expr = Operator, Props = Properties>;

/// An operator is an expression (which can be either logical or physical) with a set of properties.
/// A tree of operators can represent both initial (unoptimized) and optimized query plans.
// TODO: Docs
#[derive(Debug, Clone)]
pub struct Operator {
    pub(crate) expr: OperatorExpr,
    pub(crate) properties: Properties,
}

impl Operator {
    /// Creates a new operator from the given expression and properties.
    pub fn new(expr: OperatorExpr, properties: Properties) -> Self {
        Operator { expr, properties }
    }

    /// Returns an expression associated with this operator.
    pub fn expr(&self) -> &OperatorExpr {
        &self.expr
    }

    /// Physical properties required by this expression.
    pub fn required(&self) -> &PhysicalProperties {
        self.properties.required()
    }

    /// Creates a new operator from this one but with new required properties.
    pub fn with_required(self, required: PhysicalProperties) -> Self {
        let Operator {
            expr,
            properties: old_properties,
            ..
        } = self;
        assert!(expr.is_relational(), "Scalar expressions do not support physical properties: {:?}", expr);
        Operator {
            expr,
            properties: Properties {
                logical: old_properties.logical,
                required,
            },
        }
    }

    /// Creates a new operator from this one but with new statistics.
    pub fn with_statistics(self, statistics: Statistics) -> Self {
        let Operator {
            expr,
            properties: old_properties,
            ..
        } = self;
        let Properties { logical, required } = old_properties;
        let columns = logical.output_columns().to_vec();
        let logical = LogicalProperties::new(columns, Some(statistics));

        Operator {
            expr,
            properties: Properties { logical, required },
        }
    }
}

/// Expression can be either a [relational] or [scalar] expression.
///
/// [relational]: crate::operators::relational::RelExpr
/// [scalar]: crate::operators::scalar::expr::Expr
// TODO: Docs
#[derive(Debug, Clone)]
pub enum OperatorExpr {
    /// A relational expression.
    Relational(RelExpr),
    /// A scalar expression.
    Scalar(ScalarExpr),
}

impl OperatorExpr {
    /// Returns the underlying relational expression.
    ///
    /// # Panics
    ///
    /// This method panics if this is not a relational expression.
    pub fn as_relational(&self) -> &RelExpr {
        match self {
            OperatorExpr::Relational(expr) => expr,
            OperatorExpr::Scalar(expr) => panic!("Expected a relational expression but got {:?}", expr),
        }
    }

    /// Returns the underlying scalar expression.
    ///
    /// # Panics
    ///
    /// This method panics if this is not a scalar expression.
    pub fn as_scalar(&self) -> &ScalarExpr {
        match self {
            OperatorExpr::Relational(expr) => panic!("Expected a scalar expression but got {:?}", expr),
            OperatorExpr::Scalar(expr) => expr,
        }
    }

    /// Returns `true` if this is a relational expression.
    pub fn is_relational(&self) -> bool {
        match self {
            OperatorExpr::Relational(_) => true,
            OperatorExpr::Scalar(_) => false,
        }
    }

    /// Returns `true` if this is a scalar expression.
    pub fn is_scalar(&self) -> bool {
        !self.is_relational()
    }
}

/// Properties of an operator.
#[derive(Debug, Clone, Default)]
pub struct Properties {
    pub(crate) logical: LogicalProperties,
    pub(crate) required: PhysicalProperties,
}

impl Properties {
    pub fn new(logical: LogicalProperties, required: PhysicalProperties) -> Self {
        Properties { logical, required }
    }

    /// Logical properties shared by an expression and equivalent expressions inside a group the expression belongs to.
    pub fn logical(&self) -> &LogicalProperties {
        &self.logical
    }

    /// Physical properties required by an expression.
    pub fn required(&self) -> &PhysicalProperties {
        &self.required
    }
}

impl crate::memo::Properties for Properties {}

impl crate::memo::Expr for OperatorExpr {}

impl MemoExpr for Operator {
    type Expr = OperatorExpr;
    type Props = Properties;

    fn expr(&self) -> &Self::Expr {
        &self.expr
    }

    fn props(&self) -> &Self::Props {
        &self.properties
    }

    fn copy_in(&self, visitor: &mut CopyInExprs<Self>) {
        let mut visitor = OperatorCopyIn { visitor };
        let mut expr_ctx = visitor.enter_expr(self);
        match self.expr() {
            OperatorExpr::Relational(expr) => match expr {
                RelExpr::Logical(expr) => expr.traverse(&mut visitor, &mut expr_ctx),
                RelExpr::Physical(expr) => expr.traverse(&mut visitor, &mut expr_ctx),
            },
            OperatorExpr::Scalar(expr) => {
                visitor.copy_in_nested(&mut expr_ctx, expr);
            }
        }
        visitor.copy_in(self, expr_ctx);
    }

    fn with_new_children(&self, children: NewChildExprs<Self>) -> Self {
        let mut inputs = OperatorInputs::from(children);
        let expr = match self.expr() {
            OperatorExpr::Relational(expr) => match expr {
                RelExpr::Logical(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)),
                RelExpr::Physical(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)),
            },
            OperatorExpr::Scalar(expr) => OperatorExpr::from(expr_with_new_inputs(expr, &mut inputs)),
        };
        let properties = self.properties.clone();
        Operator::new(expr, properties)
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self.expr() {
            OperatorExpr::Relational(expr) => match expr {
                RelExpr::Logical(expr) => expr.format_expr(f),
                RelExpr::Physical(expr) => expr.format_expr(f),
            },
            OperatorExpr::Scalar(expr) => expr.format_expr(f),
        }
    }
}

/// Provides methods to copy an operator tree into a [memo].
///
///[memo]: crate::memo::Memo
pub struct OperatorCopyIn<'c, 'm> {
    visitor: &'c mut CopyInExprs<'m, Operator>,
}

impl OperatorCopyIn<'_, '_> {
    /// Starts traversal of the given expression `expr`.
    pub fn enter_expr(&mut self, expr: &Operator) -> ExprContext<Operator> {
        self.visitor.enter_expr(expr)
    }

    /// Visits the given relational expression and copies it into a memo.
    /// See [`memo`][crate::memo::CopyInExprs::visit_expr_node] for details.
    pub fn visit_rel(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: &RelNode) {
        self.visitor.visit_expr_node(expr_ctx, expr);
    }

    /// Visits the given optional relational expression if it is present and copies it into a memo. This method is equivalent to:
    /// ```text
    ///   if let Some(expr) = expr {
    ///     visitor.visit_rel(expr_ctx, expr);
    ///   }
    /// ```
    /// See [`memo::CopyInExprs`][crate::memo::CopyInExprs::visit_opt_expr_node] for details.
    pub fn visit_opt_rel(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: Option<&RelNode>) {
        self.visitor.visit_opt_expr_node(expr_ctx, expr);
    }

    /// Visits the given scalar expression and copies it into a memo.
    /// See [`memo`][crate::memo::CopyInExprs::visit_expr_node] for details.
    pub fn visit_scalar(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: &ScalarNode) {
        self.visitor.visit_expr_node(expr_ctx, expr);
    }

    /// Visits the given optional scalar expression if it is present and copies it into a memo.
    /// See [`memo::CopyInExprs`][crate::memo::CopyInExprs::visit_opt_expr_node] for details.
    pub fn visit_opt_scalar(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: Option<&ScalarNode>) {
        self.visitor.visit_opt_expr_node(expr_ctx, expr);
    }

    /// Visits expressions in the given join condition and copies them into a memo.
    pub fn visit_join_condition(&mut self, expr_ctx: &mut ExprContext<Operator>, condition: &JoinCondition) {
        if let JoinCondition::On(on) = condition {
            self.visitor.visit_expr_node(expr_ctx, on.expr())
        };
    }

    /// Traverses the given scalar expression and all of its nested relational expressions into a memo.
    /// See [`memo`][crate::memo::CopyInExprs::visit_expr_node] for details.
    pub fn copy_in_nested(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: &ScalarExpr) {
        struct CopyNestedRelExprs<'b, 'a, 'c> {
            collector: &'b mut CopyInNestedExprs<'a, 'c, Operator>,
        }
        impl ExprVisitor<RelNode> for CopyNestedRelExprs<'_, '_, '_> {
            fn post_visit(&mut self, expr: &ScalarExpr) {
                if let ScalarExpr::SubQuery(rel_node) = expr {
                    self.collector.visit_expr(rel_node.into())
                }
            }
        }

        let nested_ctx = CopyInNestedExprs::new(self.visitor, expr_ctx);
        nested_ctx.execute(expr, |expr, collector: &mut CopyInNestedExprs<Operator>| {
            let mut visitor = CopyNestedRelExprs { collector };
            expr.accept(&mut visitor)
        });
    }

    /// Copies the given expression `expr` into a memo.
    pub fn copy_in(self, expr: &Operator, expr_ctx: ExprContext<Operator>) {
        self.visitor.copy_in(expr, expr_ctx)
    }
}

/// A wrapper round [`NewChildExprs`](crate::memo::NewChildExprs) that provides convenient methods
/// to retrieve both relational and scalar expressions.
//???: Improve error reporting
#[derive(Debug)]
pub struct OperatorInputs {
    inputs: NewChildExprs<Operator>,
}

impl OperatorInputs {
    /// Ensures that this container holds exactly `n` expressions and panics if this condition is not met.
    pub fn expect_len(&self, n: usize, operator: &str) {
        assert_eq!(self.inputs.len(), n, "{}: Unexpected number of child expressions", operator);
    }

    /// Retrieves the next relational expression.
    ///
    /// # Panics
    ///
    /// This method panics if there is no relational expressions left or the retrieved expression is
    /// not a relational expression.
    pub fn rel_node(&mut self) -> RelNode {
        let expr = self.inputs.expr();
        Self::expect_relational(&expr);
        RelNode::from(expr)
    }

    /// Retrieves the next `n` relational expressions.
    ///
    /// # Panics
    ///
    /// This method panics if there is not enough expressions left or some of the retrieved expressions are
    /// not relational expressions.
    pub fn rel_nodes(&mut self, n: usize) -> Vec<RelNode> {
        self.inputs
            .exprs(n)
            .into_iter()
            .map(|i| {
                Self::expect_relational(&i);
                RelNode::from(i)
            })
            .collect()
    }

    /// Retrieves the next scalar expression.
    ///
    /// # Panics
    ///
    /// This method panics if there is no expressions left or the retrieved expression is
    /// not a scalar expression.
    pub fn scalar_node(&mut self) -> ScalarNode {
        let expr = self.inputs.expr();
        Self::expect_scalar(&expr);
        ScalarNode::from(expr)
    }

    /// Retrieves the next `n` scalar expressions.
    ///
    /// # Panics
    ///
    /// This method panics if there is not enough expressions left or some of the retrieved expressions are
    /// not scalar expressions.
    pub fn scalar_nodes(&mut self, n: usize) -> Vec<ScalarNode> {
        self.inputs
            .exprs(n)
            .into_iter()
            .map(|i| {
                Self::expect_scalar(&i);
                ScalarNode::from(i)
            })
            .collect()
    }

    fn expect_relational(expr: &ExprNode<Operator>) {
        assert!(expr.expr().is_relational(), "expected a relational expression");
    }

    fn expect_scalar(expr: &ExprNode<Operator>) {
        assert!(expr.expr().is_scalar(), "expected a scalar expression");
    }
}

impl From<NewChildExprs<Operator>> for OperatorInputs {
    fn from(inputs: NewChildExprs<Operator>) -> Self {
        OperatorInputs { inputs }
    }
}

impl From<OperatorExpr> for Operator {
    fn from(expr: OperatorExpr) -> Self {
        Operator {
            expr,
            properties: Default::default(),
        }
    }
}

impl From<LogicalExpr> for OperatorExpr {
    fn from(expr: LogicalExpr) -> Self {
        let expr = RelExpr::Logical(Box::new(expr));
        OperatorExpr::Relational(expr)
    }
}

impl From<PhysicalExpr> for OperatorExpr {
    fn from(expr: PhysicalExpr) -> Self {
        let expr = RelExpr::Physical(Box::new(expr));
        OperatorExpr::Relational(expr)
    }
}

// For testing
impl From<LogicalExpr> for RelNode {
    fn from(expr: LogicalExpr) -> Self {
        let expr = RelExpr::Logical(Box::new(expr));
        let expr = OperatorExpr::Relational(expr);
        RelNode::Expr(Box::new(Operator::from(expr)))
    }
}

impl<'a> From<&'a RelNode> for ExprNodeRef<'a, Operator> {
    fn from(expr: &'a RelNode) -> Self {
        match expr {
            RelNode::Expr(expr) => ExprNodeRef::Expr(expr),
            RelNode::Group(group) => ExprNodeRef::Group(group),
        }
    }
}

impl From<ExprNode<Operator>> for RelNode {
    fn from(expr: ExprNode<Operator>) -> Self {
        match expr {
            ExprNode::Expr(expr) => RelNode::Expr(expr),
            ExprNode::Group(group) => RelNode::Group(group),
        }
    }
}

// For testing
impl From<Operator> for RelNode {
    fn from(expr: Operator) -> Self {
        RelNode::Expr(Box::new(expr))
    }
}

impl<'a> From<&'a ScalarNode> for ExprNodeRef<'a, Operator> {
    fn from(expr: &'a ScalarNode) -> Self {
        match expr {
            ScalarNode::Expr(expr) => ExprNodeRef::Expr(expr),
            ScalarNode::Group(group) => ExprNodeRef::Group(group),
        }
    }
}

impl From<ExprNode<Operator>> for ScalarNode {
    fn from(expr: ExprNode<Operator>) -> Self {
        match expr {
            ExprNode::Expr(expr) => ScalarNode::Expr(expr),
            ExprNode::Group(group) => ScalarNode::Group(group),
        }
    }
}

// For testing
impl From<ScalarExpr> for ScalarNode {
    fn from(expr: ScalarExpr) -> Self {
        let expr = OperatorExpr::Scalar(expr);
        ScalarNode::Expr(Box::new(Operator::from(expr)))
    }
}

impl From<ScalarExpr> for OperatorExpr {
    fn from(expr: ScalarExpr) -> Self {
        OperatorExpr::Scalar(expr)
    }
}
