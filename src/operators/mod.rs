use crate::memo::{
    CopyInExprs, CopyInNestedExprs, ExprContext, ExprGroupRef, MemoExpr, MemoExprFormatter, MemoGroupCallback,
    NewChildExprs, Props,
};
use crate::meta::MutableMetadata;
use crate::operators::scalar::expr_with_new_inputs;
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use relational::join::JoinCondition;
use relational::logical::LogicalExpr;
use relational::physical::PhysicalExpr;
use relational::{RelExpr, RelNode};
use scalar::expr::ExprVisitor;
use scalar::{ScalarExpr, ScalarNode};
use std::convert::Infallible;
use std::fmt::Debug;
use std::rc::Rc;

pub mod builder;
pub mod properties;
pub mod relational;
pub mod scalar;

pub type OperatorMetadata = Rc<MutableMetadata>;
pub type ExprMemo = crate::memo::Memo<Operator, OperatorMetadata>;
pub type GroupRef = crate::memo::MemoGroupRef<Operator>;
pub type ExprRef = crate::memo::MemoExprRef<Operator>;
pub type ExprCallback = dyn MemoGroupCallback<Expr = Operator, Props = Properties, Metadata = OperatorMetadata>;

/// An operator is an expression (which can be either logical or physical) with a set of properties.
/// A tree of operators can represent both initial (unoptimized) and optimized query plans.
// TODO: Docs
#[derive(Debug, Clone)]
pub struct Operator {
    pub(crate) expr: crate::memo::ExprRef<Operator>,
    pub(crate) group: crate::memo::ExprGroupRef<Operator>,
}

impl Operator {
    /// Creates a new operator from the given expression and properties.
    pub fn new(expr: OperatorExpr, properties: Properties) -> Self {
        Operator {
            expr: crate::memo::ExprRef::Detached(Box::new(expr)),
            group: crate::memo::ExprGroupRef::Detached(Box::new(properties)),
        }
    }

    /// Returns an expression associated with this operator.
    pub fn expr(&self) -> &OperatorExpr {
        self.expr.expr()
    }

    /// Creates a new operator from this one but with new required properties.
    pub fn with_required(self, required: PhysicalProperties) -> Self {
        let Operator { expr, group, .. } = self;
        assert!(expr.expr().is_relational(), "Scalar expressions do not support physical properties: {:?}", expr);
        let old_properties = match group.props().clone() {
            Properties::Relational(props) => props,
            Properties::Scalar(_) => panic!("Relational operator with scalar properties"),
        };
        let props = RelationalProperties {
            logical: old_properties.logical,
            required,
        };
        Operator {
            expr,
            group: crate::memo::ExprGroupRef::Detached(Box::new(Properties::Relational(props))),
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

/// Properties of an operator. Properties are shared between all operators (expressions) in a [memo group](crate::memo::MemoGroupRef).
#[derive(Debug, Clone)]
pub enum Properties {
    Relational(RelationalProperties),
    Scalar(ScalarProperties),
}

impl Properties {
    pub fn new_relational_properties(logical: LogicalProperties, required: PhysicalProperties) -> Self {
        Properties::Relational(RelationalProperties { logical, required })
    }

    pub fn new_scalar_properties(nested_sub_queries: Vec<Operator>) -> Self {
        Properties::Scalar(ScalarProperties { nested_sub_queries })
    }
}

impl Props for Properties {
    type RelProps = RelationalProperties;
    type ScalarProps = ScalarProperties;

    fn new_rel(props: Self::RelProps) -> Self {
        Properties::Relational(props)
    }

    fn new_scalar(props: Self::ScalarProps) -> Self {
        Properties::Scalar(props)
    }

    fn as_relational(&self) -> &Self::RelProps {
        match self {
            Properties::Relational(props) => props,
            Properties::Scalar(_) => panic!("Expected relational properties"),
        }
    }

    fn as_scalar(&self) -> &Self::ScalarProps {
        match self {
            Properties::Relational(_) => panic!("Expected scalar properties"),
            Properties::Scalar(props) => props,
        }
    }
}

/// Properties of a relational operator.
#[derive(Debug, Clone, Default)]
pub struct RelationalProperties {
    pub(crate) logical: LogicalProperties,
    pub(crate) required: PhysicalProperties,
}

impl RelationalProperties {
    /// Logical properties shared by an expression and equivalent expressions inside a group the expression belongs to.
    pub fn logical(&self) -> &LogicalProperties {
        &self.logical
    }

    /// Physical properties required by an expression.
    pub fn required(&self) -> &PhysicalProperties {
        &self.required
    }
}

/// Properties of a scalar operator.
#[derive(Debug, Clone, Default)]
pub struct ScalarProperties {
    pub(crate) nested_sub_queries: Vec<Operator>,
}

impl ScalarProperties {
    /// Returns nested sub-queries of scalar expression tree these properties are associated with.
    pub fn nested_sub_queries(&self) -> &[Operator] {
        &self.nested_sub_queries
    }
}

impl crate::memo::Expr for OperatorExpr {
    type RelExpr = crate::operators::relational::RelExpr;
    type ScalarExpr = crate::operators::scalar::ScalarExpr;

    fn new_rel(expr: Self::RelExpr) -> Self {
        OperatorExpr::Relational(expr)
    }

    fn new_scalar(expr: Self::ScalarExpr) -> Self {
        OperatorExpr::Scalar(expr)
    }

    fn as_relational(&self) -> &Self::RelExpr {
        match self {
            OperatorExpr::Relational(expr) => expr,
            OperatorExpr::Scalar(_) => panic!("Expected a relational expression"),
        }
    }

    fn as_scalar(&self) -> &Self::ScalarExpr {
        match self {
            OperatorExpr::Relational(_) => panic!("Expected a scalar expression"),
            OperatorExpr::Scalar(expr) => expr,
        }
    }

    fn is_scalar(&self) -> bool {
        match self {
            OperatorExpr::Relational(_) => false,
            OperatorExpr::Scalar(_) => true,
        }
    }
}

impl MemoExpr for Operator {
    type Expr = OperatorExpr;
    type Props = Properties;

    fn create(expr: crate::memo::ExprRef<Self>, group: crate::memo::ExprGroupRef<Self>) -> Self {
        Operator { expr, group }
    }

    fn expr_ref(&self) -> &crate::memo::ExprRef<Self> {
        &self.expr
    }

    fn group_ref(&self) -> &ExprGroupRef<Self> {
        &self.group
    }

    fn copy_in<T>(&self, visitor: &mut CopyInExprs<Self, T>) {
        let mut visitor = OperatorCopyIn { visitor };
        let mut expr_ctx = visitor.enter_expr(self);
        match self.expr() {
            OperatorExpr::Relational(expr) => match expr {
                RelExpr::Logical(expr) => expr.copy_in(&mut visitor, &mut expr_ctx),
                RelExpr::Physical(expr) => expr.copy_in(&mut visitor, &mut expr_ctx),
            },
            OperatorExpr::Scalar(expr) => {
                visitor.copy_in_nested(&mut expr_ctx, expr);
            }
        }
        visitor.copy_in(self, expr_ctx);
    }

    fn expr_with_new_children(expr: &Self::Expr, mut inputs: NewChildExprs<Self>) -> Self::Expr {
        match expr {
            OperatorExpr::Relational(expr) => match expr {
                RelExpr::Logical(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)),
                RelExpr::Physical(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)),
            },
            OperatorExpr::Scalar(expr) => OperatorExpr::from(expr_with_new_inputs(expr, &mut inputs)),
        }
    }

    fn new_properties_with_nested_sub_queries(
        _props: Self::Props,
        sub_queries: impl Iterator<Item = Self>,
    ) -> Self::Props {
        Properties::new_scalar_properties(sub_queries.collect())
    }

    fn num_children(&self) -> usize {
        match self.expr() {
            OperatorExpr::Relational(RelExpr::Logical(e)) => e.num_children(),
            OperatorExpr::Relational(RelExpr::Physical(e)) => e.num_children(),
            OperatorExpr::Scalar(_) => self.props().as_scalar().nested_sub_queries.len(),
        }
    }

    fn get_child(&self, i: usize) -> Option<&Self> {
        match self.expr() {
            OperatorExpr::Relational(RelExpr::Logical(e)) => e.get_child(i),
            OperatorExpr::Relational(RelExpr::Physical(e)) => e.get_child(i),
            OperatorExpr::Scalar(_) if i < self.props().as_scalar().nested_sub_queries().len() => {
                self.props().as_scalar().nested_sub_queries.get(i)
            }
            OperatorExpr::Scalar(_) => None,
        }
    }

    fn format_expr<F>(expr: &Self::Expr, _props: &Self::Props, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match expr {
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
pub struct OperatorCopyIn<'c, 'm, T> {
    visitor: &'c mut CopyInExprs<'m, Operator, T>,
}

impl<T> OperatorCopyIn<'_, '_, T> {
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
        struct CopyNestedRelExprs<'b, 'a, 'c, T> {
            collector: &'b mut CopyInNestedExprs<'a, 'c, Operator, T>,
        }
        impl<T> ExprVisitor<RelNode> for CopyNestedRelExprs<'_, '_, '_, T> {
            type Error = Infallible;

            fn post_visit(&mut self, expr: &ScalarExpr) -> Result<(), Self::Error> {
                if let ScalarExpr::SubQuery(rel_node) = expr {
                    self.collector.visit_expr(rel_node)
                }
                Ok(())
            }
        }

        let nested_ctx = CopyInNestedExprs::new(self.visitor, expr_ctx);
        nested_ctx.execute(expr, |expr, collector: &mut CopyInNestedExprs<Operator, T>| {
            let mut visitor = CopyNestedRelExprs { collector };
            // Never returns an error
            expr.accept(&mut visitor).unwrap()
        });
    }

    /// Copies the given expression `expr` into a memo.
    pub fn copy_in(self, expr: &Operator, expr_ctx: ExprContext<Operator>) {
        self.visitor.copy_in(expr, expr_ctx)
    }
}

impl From<OperatorExpr> for Operator {
    fn from(expr: OperatorExpr) -> Self {
        let props = match expr {
            OperatorExpr::Relational(_) => Properties::Relational(RelationalProperties::default()),
            OperatorExpr::Scalar(_) => Properties::Scalar(ScalarProperties::default()),
        };
        Operator {
            expr: crate::memo::ExprRef::Detached(Box::new(expr)),
            group: crate::memo::ExprGroupRef::Detached(Box::new(props)),
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
        RelNode::from_mexpr(Operator::from(expr))
    }
}

// For testing
impl From<Operator> for RelNode {
    fn from(expr: Operator) -> Self {
        RelNode::from_mexpr(expr)
    }
}

// For testing
impl From<ScalarExpr> for ScalarNode {
    fn from(expr: ScalarExpr) -> Self {
        let expr = OperatorExpr::Scalar(expr);
        ScalarNode::from_mexpr(Operator::from(expr))
    }
}

impl From<ScalarExpr> for OperatorExpr {
    fn from(expr: ScalarExpr) -> Self {
        OperatorExpr::Scalar(expr)
    }
}

#[cfg(test)]
mod test {
    use crate::operators::scalar::ScalarExpr;
    use crate::operators::{Operator, OperatorExpr, Properties, ScalarProperties};

    #[test]
    fn test_operator_must_be_sync_and_send() {
        let expr = OperatorExpr::Scalar(ScalarExpr::Column(0));
        let props = Properties::Scalar(ScalarProperties::default());
        let expr = Operator::new(expr, props);

        ensure_sync_and_send(expr);
    }

    fn ensure_sync_and_send<T>(_o: T)
    where
        T: Sync + Send,
    {
        //
    }
}
