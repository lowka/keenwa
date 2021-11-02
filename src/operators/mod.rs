use crate::memo::{
    Attributes, InputNode, MemoExpr, MemoExprCallback, MemoExprDigest, MemoExprFormatter, TraversalContext,
};
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
pub type InputExpr = crate::memo::InputNode<Operator>;
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

/// A [logical] or [physical] expression.
///
/// [logical]: crate::operators::logical::LogicalExpr
/// [physical]: crate::operators::physical::PhysicalExpr
// TODO: Docs
#[derive(Debug, Clone)]
pub enum OperatorExpr {
    Logical(Box<LogicalExpr>),
    Physical(Box<PhysicalExpr>),
}

impl OperatorExpr {
    /// Returns a reference to the underlying logical expression.
    /// If this expression is not a logical expression this methods panics.
    pub fn as_logical(&self) -> &LogicalExpr {
        match self {
            OperatorExpr::Logical(expr) => expr.as_ref(),
            OperatorExpr::Physical(_) => {
                panic!("Expected a logical expression but got: {:?}", self)
            }
        }
    }

    /// Returns a reference to the underlying physical expression.
    /// If this expression is not a physical expression this methods panics.
    pub fn as_physical(&self) -> &PhysicalExpr {
        match self {
            OperatorExpr::Logical(_) => {
                panic!("Expected a physical expression but got: {:?}", self)
            }
            OperatorExpr::Physical(expr) => expr.as_ref(),
        }
    }

    /// Creates a new expression from this expression by replacing its input expressions with the new ones.
    pub fn with_new_inputs(&self, inputs: Vec<InputExpr>) -> Self {
        match self {
            OperatorExpr::Logical(expr) => OperatorExpr::from(expr.with_new_inputs(inputs)),
            OperatorExpr::Physical(expr) => OperatorExpr::from(expr.with_new_inputs(inputs)),
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

impl Attributes for Properties {}

impl MemoExpr for Operator {
    type Expr = OperatorExpr;
    type Attrs = Properties;

    fn expr(&self) -> &Self::Expr {
        &self.expr
    }

    fn attrs(&self) -> &Self::Attrs {
        &self.attributes
    }

    fn copy_in(&self, ctx: &mut TraversalContext<Self>) {
        let mut expr_ctx = ctx.enter_expr(self);
        match self.expr() {
            OperatorExpr::Logical(expr) => expr.traverse(&mut expr_ctx, ctx),
            OperatorExpr::Physical(expr) => expr.traverse(&mut expr_ctx, ctx),
        }
        ctx.copy_in(self, expr_ctx);
    }

    fn with_new_inputs(&self, inputs: Vec<InputNode<Self>>) -> Self {
        let expr = match self.expr() {
            OperatorExpr::Logical(expr) => OperatorExpr::from(expr.with_new_inputs(inputs)),
            OperatorExpr::Physical(expr) => OperatorExpr::from(expr.with_new_inputs(inputs)),
        };
        let attrs = self.attributes.clone();
        Operator::new(expr, attrs)
    }

    fn make_digest<D>(&self, digest: &mut D)
    where
        D: MemoExprDigest,
    {
        match self.expr() {
            OperatorExpr::Logical(expr) => expr.make_digest(digest),
            OperatorExpr::Physical(expr) => expr.make_digest(digest),
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self.expr() {
            OperatorExpr::Logical(expr) => expr.format_expr(f),
            OperatorExpr::Physical(expr) => expr.format_expr(f),
        }
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
        OperatorExpr::Logical(Box::new(expr))
    }
}

impl From<PhysicalExpr> for OperatorExpr {
    fn from(expr: PhysicalExpr) -> Self {
        OperatorExpr::Physical(Box::new(expr))
    }
}

// For testing
impl From<LogicalExpr> for InputExpr {
    fn from(expr: LogicalExpr) -> Self {
        InputExpr::from(Operator::from(OperatorExpr::from(expr)))
    }
}
