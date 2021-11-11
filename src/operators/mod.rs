use crate::memo::{
    CopyInExprs, CopyInNestedExprs, ExprContext, InputNode, InputNodeRef, MemoExpr, MemoExprCallback,
    MemoExprFormatter, MemoGroupRef, NewInputs,
};
use crate::operators::expressions::Expr;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use crate::properties::statistics::Statistics;
use std::fmt::Debug;

pub mod expressions;
pub mod join;
pub mod logical;
pub mod physical;
pub mod scalar;

pub type ExprMemo = crate::memo::Memo<Operator>;
pub type GroupRef = crate::memo::MemoGroupRef<Operator>;
pub type ExprRef = crate::memo::MemoExprRef<Operator>;
pub type ExprCallback = dyn MemoExprCallback<Expr = Operator, Attrs = Properties>;

/// A node of an operator tree that represent both initial and optimized query plan.
/// An operator is an expression (which can be either logical or physical) with a set of attributes.
// TODO: Docs
#[derive(Debug, Clone)]
pub struct Operator {
    expr: OperatorExpr,
    attributes: Properties,
}

impl Operator {
    /// Creates a new operator from the given expression and attributes.
    pub fn new(expr: OperatorExpr, attributes: Properties) -> Self {
        Operator { expr, attributes }
    }

    /// Returns an expression associated with this operator.
    pub fn expr(&self) -> &OperatorExpr {
        &self.expr
    }

    /// Logical properties shared by this expression and equivalent expressions inside the group this expression belongs to.
    pub fn logical(&self) -> &LogicalProperties {
        self.attributes.logical()
    }

    /// Physical properties required by this expression.
    pub fn required(&self) -> &PhysicalProperties {
        self.attributes.required()
    }

    /// Creates a new operator from this one but with new required properties.
    pub fn with_required(self, required: PhysicalProperties) -> Self {
        let Operator { expr, attributes, .. } = self;
        Operator {
            expr,
            attributes: Properties {
                logical: attributes.logical,
                required,
            },
        }
    }

    /// Creates a new operator from this one but with new statistics.
    pub fn with_statistics(self, statistics: Statistics) -> Self {
        let Operator { expr, attributes, .. } = self;
        let Properties { logical, required } = attributes;
        let columns = logical.output_columns().to_vec();
        let logical = LogicalProperties::new(columns, Some(statistics));

        Operator {
            expr,
            attributes: Properties { logical, required },
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

    /// Returns `true` if this is a relational expression.
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
#[derive(Debug, Clone)]
pub enum RelNode {
    Expr(Box<Operator>),
    Group(MemoGroupRef<Operator>),
}

impl RelNode {
    pub fn expr(&self) -> &RelExpr {
        match self {
            RelNode::Expr(expr) => expr.expr().as_relational(),
            RelNode::Group(group) => group.expr().as_relational(),
        }
    }

    pub fn attrs(&self) -> &Properties {
        match self {
            RelNode::Expr(expr) => expr.attrs(),
            RelNode::Group(group) => group.attrs(),
        }
    }

    pub(crate) fn get_ref(&self) -> InputNodeRef<Operator> {
        match self {
            RelNode::Expr(expr) => InputNodeRef::Expr(&**expr),
            RelNode::Group(group) => InputNodeRef::Group(group),
        }
    }
}

impl<'a> From<&'a RelNode> for InputNodeRef<'a, Operator> {
    fn from(expr: &'a RelNode) -> Self {
        match expr {
            RelNode::Expr(expr) => InputNodeRef::Expr(expr),
            RelNode::Group(group) => InputNodeRef::Group(group),
        }
    }
}

impl From<InputNode<Operator>> for RelNode {
    fn from(expr: InputNode<Operator>) -> Self {
        match expr {
            InputNode::Expr(expr) => RelNode::Expr(expr),
            InputNode::Group(group) => RelNode::Group(group),
        }
    }
}

/// A scalar node of an operator tree.
#[derive(Debug, Clone)]
pub enum ScalarNode {
    Expr(Box<Operator>),
    Group(MemoGroupRef<Operator>),
}

impl ScalarNode {
    pub fn expr(&self) -> &Expr {
        match self {
            ScalarNode::Expr(expr) => expr.expr().as_scalar(),
            ScalarNode::Group(group) => group.expr().as_scalar(),
        }
    }
}

impl<'a> From<&'a ScalarNode> for InputNodeRef<'a, Operator> {
    fn from(expr: &'a ScalarNode) -> Self {
        match expr {
            ScalarNode::Expr(expr) => InputNodeRef::Expr(expr),
            ScalarNode::Group(group) => InputNodeRef::Group(group),
        }
    }
}

impl From<InputNode<Operator>> for ScalarNode {
    fn from(expr: InputNode<Operator>) -> Self {
        match expr {
            InputNode::Expr(expr) => ScalarNode::Expr(expr),
            InputNode::Group(group) => ScalarNode::Group(group),
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

impl crate::memo::Attributes for Properties {}

impl crate::memo::Expr for OperatorExpr {}

impl MemoExpr for Operator {
    type Expr = OperatorExpr;
    type Attrs = Properties;

    fn expr(&self) -> &Self::Expr {
        &self.expr
    }

    fn attrs(&self) -> &Self::Attrs {
        &self.attributes
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

    fn with_new_inputs(&self, inputs: NewInputs<Self>) -> Self {
        let mut inputs = OperatorInputs::from(inputs);
        let expr = match self.expr() {
            OperatorExpr::Relational(expr) => match expr {
                RelExpr::Logical(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)),
                RelExpr::Physical(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)),
            },
            OperatorExpr::Scalar(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)),
        };
        let attrs = self.attributes.clone();
        Operator::new(expr, attrs)
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
        self.visitor.visit_input(expr_ctx, expr);
    }

    /// Visits the given scalar expression and copies it into a memo.
    /// See [`memo`][crate::memo::CopyInExprs::visit_input] for details.
    pub fn visit_scalar(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: &ScalarNode) {
        self.visitor.visit_input(expr_ctx, expr);
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

/// A wrapper round [`NewInputs`](crate::memo::NewInputs) that provides convenient methods to retrieve both relational and scalar expressions.
//???: Improve error reporting
#[derive(Debug)]
pub struct OperatorInputs {
    inputs: NewInputs<Operator>,
}

impl OperatorInputs {
    /// Ensures that this container holds exactly `n` inputs expressions and panics if this condition is not met.
    pub fn expect_len(&self, n: usize, operator: &str) {
        assert_eq!(self.inputs.len(), n, "{}: Unexpected number of sub-expressions", operator);
    }

    /// Retrieves the next relational expression.
    ///
    /// # Panics
    ///
    /// This method panics if there is no relational expressions left.
    pub fn rel_node(&mut self) -> RelNode {
        RelNode::from(self.inputs.input())
    }

    /// Retrieves the next `n` relational expressions.
    ///
    /// # Panics
    ///
    /// This method panics if there is not enough expressions left.
    pub fn rel_nodes(&mut self, n: usize) -> Vec<RelNode> {
        self.inputs.inputs(n).into_iter().map(|i| RelNode::from(i)).collect()
    }

    /// Retrieves the next scalar expression.
    ///
    /// # Panics
    ///
    /// This method panics if there is no expressions left.
    pub fn scalar_node(&mut self) -> ScalarNode {
        ScalarNode::from(self.inputs.input())
    }

    /// Retrieves the next `n` scalar expressions.
    ///
    /// # Panics
    ///
    /// This method panics if there is not enough expressions left.
    pub fn scalar_nodes(&mut self, n: usize) -> Vec<ScalarNode> {
        self.inputs.inputs(n).into_iter().map(|i| ScalarNode::from(i)).collect()
    }
}

impl From<NewInputs<Operator>> for OperatorInputs {
    fn from(inputs: NewInputs<Operator>) -> Self {
        OperatorInputs { inputs }
    }
}

impl From<OperatorExpr> for Operator {
    fn from(expr: OperatorExpr) -> Self {
        Operator {
            expr,
            attributes: Default::default(),
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
