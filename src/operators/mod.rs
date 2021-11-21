use crate::memo::{
    CopyInExprs, CopyInNestedExprs, ExprContext, ExprNode, ExprNodeRef, MemoExpr, MemoExprCallback, MemoExprFormatter,
    MemoGroupRef, NewChildExprs,
};
use crate::operators::expr::Expr;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use crate::properties::statistics::Statistics;
use std::fmt::Debug;

pub mod expr;
pub mod join;
pub mod logical;
pub mod physical;
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
    expr: OperatorExpr,
    properties: Properties,
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

    /// Logical properties shared by this expression and equivalent expressions inside the group this expression belongs to.
    pub fn logical(&self) -> &LogicalProperties {
        self.properties.logical()
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
/// [relational]: crate::operators::RelExpr
/// [scalar]: crate::operators::expressions::Expr
// TODO: Docs
#[derive(Debug, Clone)]
pub enum OperatorExpr {
    /// A relational expression.
    Relational(RelExpr),
    /// A scalar expression.
    Scalar(Expr),
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
    pub fn as_scalar(&self) -> &Expr {
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

/// A relational expression. Relational expressions can be either [logical] or [physical].
#[derive(Debug, Clone)]
pub enum RelExpr {
    Logical(Box<LogicalExpr>),
    Physical(Box<PhysicalExpr>),
}

impl RelExpr {
    /// Returns a reference to the underlying logical expression.
    ///
    /// # Panics
    ///
    /// If this expression is not a logical expression this methods panics.
    pub fn as_logical(&self) -> &LogicalExpr {
        match self {
            RelExpr::Logical(expr) => expr.as_ref(),
            RelExpr::Physical(_) => {
                panic!("Expected a logical expression but got: {:?}", self)
            }
        }
    }

    /// Returns a reference to the underlying physical expression.
    ///
    /// # Panics
    ///
    /// If this expression is not a physical expression this methods panics.
    pub fn as_physical(&self) -> &PhysicalExpr {
        match self {
            RelExpr::Logical(_) => {
                panic!("Expected a physical expression but got: {:?}", self)
            }
            RelExpr::Physical(expr) => expr.as_ref(),
        }
    }
}

/// A relational node of an operator tree.
///
/// Should not be created directly and it is responsibility of the caller to provide a instance of `Operator`
/// which is a valid relational expression.
#[derive(Debug, Clone)]
pub enum RelNode {
    /// A node is an expression.
    Expr(Box<Operator>),
    /// A node is a memo-group.
    Group(MemoGroupRef<Operator>),
}

impl RelNode {
    /// Returns a reference to a relational expression stored inside this node:
    /// * if this node is an expression returns a reference to the underlying expression.
    /// * If this node is a memo group returns a reference to the first expression of this memo group.
    pub fn expr(&self) -> &RelExpr {
        match self {
            RelNode::Expr(expr) => expr.expr().as_relational(),
            RelNode::Group(group) => group.expr().as_relational(),
        }
    }

    /// Returns a reference to properties associated with this node:
    /// * if this node is an expression returns properties of the expression.
    /// * If this node is a memo group returns a reference to the properties of this memo group.
    pub fn props(&self) -> &Properties {
        match self {
            RelNode::Expr(expr) => expr.props(),
            RelNode::Group(group) => group.props(),
        }
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

/// A scalar node of an operator tree.
///
/// Should not be created directly and it is responsibility of the caller to provide a instance of `Operator`
/// which is a valid scalar expression.
#[derive(Debug, Clone)]
pub enum ScalarNode {
    /// A node is an expression.
    Expr(Box<Operator>),
    /// A node is a memo-group.
    Group(MemoGroupRef<Operator>),
}

impl ScalarNode {
    /// Returns a reference to a scalar expression stored inside this node:
    /// * if this node is an expression returns a reference to the underlying expression.
    /// * If this node is a memo group returns a reference to the first expression of this memo group.
    pub fn expr(&self) -> &Expr {
        match self {
            ScalarNode::Expr(expr) => expr.expr().as_scalar(),
            ScalarNode::Group(group) => group.expr().as_scalar(),
        }
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
            OperatorExpr::Scalar(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)),
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
    /// See [`memo`][crate::memo::CopyInExprs::visit_input] for details.
    pub fn visit_rel(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: &RelNode) {
        self.visitor.visit_expr_node(expr_ctx, expr);
    }

    /// Visits the given scalar expression and copies it into a memo.
    /// See [`memo`][crate::memo::CopyInExprs::visit_input] for details.
    pub fn visit_scalar(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: &ScalarNode) {
        self.visitor.visit_expr_node(expr_ctx, expr);
    }

    /// Traverses the given scalar expression and all of its nested relational expressions into a memo.
    /// See [`memo`][crate::memo::CopyInExprs::visit_input] for details.
    pub fn copy_in_nested(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: &Expr) {
        let nested_ctx = CopyInNestedExprs::new(self.visitor, expr_ctx);
        nested_ctx.execute(expr, |expr, collector: &mut CopyInNestedExprs<Operator>| {
            expr.copy_in_nested(collector);
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

impl From<Expr> for OperatorExpr {
    fn from(expr: Expr) -> Self {
        OperatorExpr::Scalar(expr)
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

// For testing
impl From<Operator> for RelNode {
    fn from(expr: Operator) -> Self {
        RelNode::Expr(Box::new(expr))
    }
}

// For testing
impl From<Expr> for ScalarNode {
    fn from(expr: Expr) -> Self {
        let expr = OperatorExpr::Scalar(expr);
        ScalarNode::Expr(Box::new(Operator::from(expr)))
    }
}
