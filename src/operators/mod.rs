//! Operators.

use std::fmt::Debug;
use std::rc::Rc;

use crate::error::OptimizerError;
use relational::join::JoinCondition;
use relational::logical::LogicalExpr;
use relational::physical::PhysicalExpr;
use relational::{RelExpr, RelNode};
use scalar::expr::ExprVisitor;
use scalar::{ScalarExpr, ScalarNode};

use crate::memo::{
    CopyInExprs, CopyInNestedExprs, ExprContext, Memo, MemoBuilder, MemoExpr, MemoExprFormatter, MemoExprRef,
    MemoExprState, MemoFormatterFlags, MemoGroupCallback, MemoGroupCallbackRef, MemoGroupRef, NewChildExprs, Props,
};
use crate::meta::{ColumnId, MutableMetadata};
use crate::operators::format::PropertiesFormatter;
use crate::operators::properties::{LogicalPropertiesBuilder, PropertiesProvider};
use crate::operators::scalar::{expr_with_new_inputs, get_subquery};
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use crate::statistics::StatisticsBuilder;

pub mod builder;
pub mod format;
// mod partitioning;
pub mod properties;
pub mod relational;
pub mod scalar;

pub type OperatorMetadata = Rc<MutableMetadata>;
pub type ExprMemo = Memo<Operator, OperatorMetadata>;
pub type ExprGroupRef<'a> = MemoGroupRef<'a, Operator, OperatorMetadata>;
pub type ExprRef = MemoExprRef<Operator>;
pub type ExprCallback = MemoGroupCallbackRef<Operator, OperatorMetadata>;

/// An operator is an expression (which can be either logical or physical) with a set of properties.
/// A tree of operators can represent both initial (unoptimized) and optimized query plans.
// TODO: Docs
#[derive(Debug, Clone)]
pub struct Operator {
    pub state: MemoExprState<Operator>,
}

impl Operator {
    /// Creates a new operator from the given expression and properties.
    pub fn new(expr: OperatorExpr, properties: Properties) -> Self {
        Operator {
            state: MemoExprState::new(expr, properties),
        }
    }

    /// Returns an expression associated with this operator.
    pub fn expr(&self) -> &OperatorExpr {
        self.state.expr()
    }

    /// Returns a reference properties associated with this operator (see [MemoExpr#props](crate::memo::MemoExpr::props)).
    pub fn props(&self) -> &Properties {
        self.state.props()
    }

    /// Creates a new operator from this one but with new [physical properties](PhysicalProperties).
    pub fn with_physical(self, physical: PhysicalProperties) -> Self {
        self.with_rel_properties(|p| RelationalProperties {
            logical: p.logical,
            physical,
        })
    }

    fn with_rel_properties<F>(self, modify_properties: F) -> Self
    where
        F: FnOnce(RelationalProperties) -> RelationalProperties,
    {
        let Operator { state } = self;
        assert!(
            state.expr().is_relational(),
            "Scalar expressions do not support relational properties: {:?}",
            state.expr()
        );
        let old_properties = match state.props().clone() {
            Properties::Relational(props) => props,
            Properties::Scalar(_) => panic!("Relational operator with scalar properties"),
        };
        let properties = (modify_properties)(old_properties);
        let expr = state.expr().clone();
        Operator {
            state: MemoExprState::new(expr, Properties::Relational(properties)),
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
    pub fn relational(&self) -> &RelExpr {
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
    pub fn scalar(&self) -> &ScalarExpr {
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
    pub fn new_scalar_properties(nested_sub_queries: Vec<Operator>) -> Self {
        Properties::Scalar(ScalarProperties {
            nested_sub_queries,
            has_correlated_sub_queries: false,
        })
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

    fn relational(&self) -> &Self::RelProps {
        match self {
            Properties::Relational(props) => props,
            Properties::Scalar(_) => panic!("Expected relational properties"),
        }
    }

    fn scalar(&self) -> &Self::ScalarProps {
        match self {
            Properties::Relational(_) => panic!("Expected scalar properties"),
            Properties::Scalar(props) => props,
        }
    }

    fn to_relational(self) -> Self::RelProps {
        match self {
            Properties::Relational(props) => props,
            Properties::Scalar(_) => panic!("Expected relational properties"),
        }
    }

    fn to_scalar(self) -> Self::ScalarProps {
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
    pub(crate) physical: PhysicalProperties,
}

impl RelationalProperties {
    /// Logical properties shared by an expression and equivalent expressions inside a group the expression belongs to.
    pub fn logical(&self) -> &LogicalProperties {
        &self.logical
    }

    /// Physical properties of an expression.
    pub fn physical(&self) -> &PhysicalProperties {
        &self.physical
    }
}

/// Properties of a scalar operator.
#[derive(Debug, Clone, Default)]
pub struct ScalarProperties {
    pub nested_sub_queries: Vec<Operator>,
    pub has_correlated_sub_queries: bool,
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

    fn relational(&self) -> &Self::RelExpr {
        OperatorExpr::relational(self)
    }

    fn scalar(&self) -> &Self::ScalarExpr {
        OperatorExpr::scalar(self)
    }

    fn is_relational(&self) -> bool {
        match self {
            OperatorExpr::Relational(_) => true,
            OperatorExpr::Scalar(_) => false,
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
    type Scope = OuterScope;

    fn from_state(state: MemoExprState<Self>) -> Self {
        Operator { state }
    }

    fn state(&self) -> &MemoExprState<Self> {
        &self.state
    }

    fn copy_in<T>(&self, visitor: &mut CopyInExprs<Self, T>) -> Result<(), OptimizerError> {
        let mut visitor = OperatorCopyIn { visitor };
        let mut expr_ctx = visitor.enter_expr(self);
        match self.expr() {
            OperatorExpr::Relational(expr) => match expr {
                RelExpr::Logical(expr) => expr.copy_in(&mut visitor, &mut expr_ctx)?,
                RelExpr::Physical(expr) => expr.copy_in(&mut visitor, &mut expr_ctx)?,
            },
            OperatorExpr::Scalar(expr) => visitor.copy_in_nested(&mut expr_ctx, expr)?,
        }
        visitor.copy_in(self, expr_ctx)
    }

    fn expr_with_new_children(
        expr: &Self::Expr,
        mut inputs: NewChildExprs<Self>,
    ) -> Result<Self::Expr, OptimizerError> {
        let expr = match expr {
            OperatorExpr::Relational(expr) => match expr {
                RelExpr::Logical(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)?),
                RelExpr::Physical(expr) => OperatorExpr::from(expr.with_new_inputs(&mut inputs)?),
            },
            OperatorExpr::Scalar(expr) => OperatorExpr::from(expr_with_new_inputs(expr, &mut inputs)?),
        };
        Ok(expr)
    }

    fn new_properties_with_nested_sub_queries(
        props: Self::Props,
        sub_queries: impl Iterator<Item = Self>,
    ) -> Self::Props {
        if let Properties::Scalar(scalar) = props {
            Properties::Scalar(ScalarProperties {
                nested_sub_queries: sub_queries.collect(),
                has_correlated_sub_queries: scalar.has_correlated_sub_queries,
            })
        } else {
            unreachable!(
                "new_properties_with_nested_sub_queries has been called with relational properties: {:?}",
                props
            )
        }
    }

    fn num_children(&self) -> usize {
        match self.expr() {
            OperatorExpr::Relational(RelExpr::Logical(e)) => e.num_children(),
            OperatorExpr::Relational(RelExpr::Physical(e)) => e.num_children(),
            OperatorExpr::Scalar(_) => self.props().scalar().nested_sub_queries.len(),
        }
    }

    fn get_child(&self, i: usize) -> Option<&Self> {
        match self.expr() {
            OperatorExpr::Relational(RelExpr::Logical(e)) => e.get_child(i),
            OperatorExpr::Relational(RelExpr::Physical(e)) => e.get_child(i),
            OperatorExpr::Scalar(_) if i < self.props().scalar().nested_sub_queries().len() => {
                self.props().scalar().nested_sub_queries.get(i)
            }
            OperatorExpr::Scalar(_) => None,
        }
    }

    fn format_expr<F>(expr: &Self::Expr, props: &Self::Props, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match expr {
            OperatorExpr::Relational(expr) => {
                match expr {
                    RelExpr::Logical(expr) => {
                        expr.format_expr(f);
                    }
                    RelExpr::Physical(expr) => expr.format_expr(f),
                };
            }
            OperatorExpr::Scalar(expr) => expr.format_expr(f),
        }
        if f.flags().has_flag(&MemoFormatterFlags::IncludeProps) {
            let fmt = PropertiesFormatter::new(f);
            fmt.format(props)
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
    pub fn visit_rel(&mut self, expr_ctx: &mut ExprContext<Operator>, expr: &RelNode) -> Result<(), OptimizerError> {
        self.visitor.visit_expr_node(expr_ctx, expr)
    }

    /// Visits the given optional relational expression if it is present and copies it into a memo. This method is equivalent to:
    /// ```text
    ///   if let Some(expr) = expr {
    ///     visitor.visit_rel(expr_ctx, expr);
    ///   }
    /// ```
    /// See [`memo::CopyInExprs`][crate::memo::CopyInExprs::visit_opt_expr_node] for details.
    pub fn visit_opt_rel(
        &mut self,
        expr_ctx: &mut ExprContext<Operator>,
        expr: Option<&RelNode>,
    ) -> Result<(), OptimizerError> {
        self.visitor.visit_opt_expr_node(expr_ctx, expr)
    }

    /// Visits the given scalar expression and copies it into a memo.
    /// See [`memo`][crate::memo::CopyInExprs::visit_expr_node] for details.
    pub fn visit_scalar(
        &mut self,
        expr_ctx: &mut ExprContext<Operator>,
        expr: &ScalarNode,
    ) -> Result<(), OptimizerError> {
        self.visitor.visit_expr_node(expr_ctx, expr)
    }

    /// Visits the given optional scalar expression if it is present and copies it into a memo.
    /// See [`memo::CopyInExprs`][crate::memo::CopyInExprs::visit_opt_expr_node] for details.
    pub fn visit_opt_scalar(
        &mut self,
        expr_ctx: &mut ExprContext<Operator>,
        expr: Option<&ScalarNode>,
    ) -> Result<(), OptimizerError> {
        self.visitor.visit_opt_expr_node(expr_ctx, expr)
    }

    /// Visits expressions in the given join condition and copies them into a memo.
    pub fn visit_join_condition(
        &mut self,
        expr_ctx: &mut ExprContext<Operator>,
        condition: &JoinCondition,
    ) -> Result<(), OptimizerError> {
        if let JoinCondition::On(on) = condition {
            self.visitor.visit_expr_node(expr_ctx, on.expr())
        } else {
            Ok(())
        }
    }

    /// Traverses the given scalar expression and all of its nested relational expressions into a memo.
    /// See [`memo`][crate::memo::CopyInExprs::visit_expr_node] for details.
    pub fn copy_in_nested(
        &mut self,
        expr_ctx: &mut ExprContext<Operator>,
        expr: &ScalarExpr,
    ) -> Result<(), OptimizerError> {
        struct CopyNestedRelExprs<'b, 'a, 'c, T> {
            collector: &'b mut CopyInNestedExprs<'a, 'c, Operator, T>,
        }
        impl<T> ExprVisitor<RelNode> for CopyNestedRelExprs<'_, '_, '_, T> {
            type Error = OptimizerError;

            fn post_visit(&mut self, expr: &ScalarExpr) -> Result<(), Self::Error> {
                if let Some(subquery) = get_subquery(expr) {
                    self.collector.visit_expr(subquery)
                } else {
                    Ok(())
                }
            }
        }

        let nested_ctx = CopyInNestedExprs::new(self.visitor, expr_ctx);
        nested_ctx.execute(expr, |expr, collector: &mut CopyInNestedExprs<Operator, T>| {
            let mut visitor = CopyNestedRelExprs { collector };
            expr.accept(&mut visitor)
        })
    }

    /// Copies the given expression `expr` into a memo.
    pub fn copy_in(self, expr: &Operator, expr_ctx: ExprContext<Operator>) -> Result<(), OptimizerError> {
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
            state: MemoExprState::new(expr, props),
        }
    }
}

impl From<LogicalExpr> for Operator {
    fn from(expr: LogicalExpr) -> Self {
        let expr = OperatorExpr::from(expr);
        Operator::from(expr)
    }
}

impl From<LogicalExpr> for OperatorExpr {
    fn from(expr: LogicalExpr) -> Self {
        let expr = RelExpr::Logical(Box::new(expr));
        OperatorExpr::Relational(expr)
    }
}

impl From<LogicalExpr> for RelNode {
    fn from(expr: LogicalExpr) -> Self {
        let expr = RelExpr::Logical(Box::new(expr));
        RelNode::new_expr(expr)
    }
}

impl From<PhysicalExpr> for Operator {
    fn from(expr: PhysicalExpr) -> Self {
        let expr = OperatorExpr::from(expr);
        Operator::from(expr)
    }
}

impl From<PhysicalExpr> for OperatorExpr {
    fn from(expr: PhysicalExpr) -> Self {
        let expr = RelExpr::Physical(Box::new(expr));
        OperatorExpr::Relational(expr)
    }
}

impl From<PhysicalExpr> for RelNode {
    fn from(expr: PhysicalExpr) -> Self {
        let expr = RelExpr::Physical(Box::new(expr));
        RelNode::new_expr(expr)
    }
}

impl From<ScalarExpr> for OperatorExpr {
    fn from(expr: ScalarExpr) -> Self {
        OperatorExpr::Scalar(expr)
    }
}

impl From<ScalarExpr> for ScalarNode {
    fn from(expr: ScalarExpr) -> Self {
        ScalarNode::new_expr(expr)
    }
}

/// ExprScope stores information that is used to build logical properties of an operator.
/// See [MemoExpr::Scope].
//???: Rename OperatorScope to BuildScope, Rename this struct to OperatorScope.
pub struct OuterScope {
    pub outer_columns: Vec<ColumnId>,
}

impl OuterScope {
    /// Creates a scope for the root of an operator tree.
    pub fn root() -> Self {
        OuterScope {
            outer_columns: Vec::new(),
        }
    }

    /// Creates a scope from the given properties.
    pub fn from_properties(props: &Properties) -> Self {
        match props {
            Properties::Relational(props) => OuterScope {
                outer_columns: props.logical.outer_columns.to_vec(),
            },
            Properties::Scalar(_) => OuterScope {
                // Currently properties of a scalar expression do not include the outer columns
                // referenced in by it because the outer columns are only used by relational operators.
                outer_columns: Vec::new(),
            },
        }
    }
}

impl From<RelationalProperties> for OuterScope {
    fn from(props: RelationalProperties) -> Self {
        OuterScope {
            outer_columns: props.logical.output_columns,
        }
    }
}

/// Callback that sets logical properties when expression is added into a [memo](crate::memo::Memo).
#[derive(Debug)]
pub struct SetPropertiesCallback<P> {
    properties_provider: Rc<P>,
}

impl<P> SetPropertiesCallback<P>
where
    P: PropertiesProvider,
{
    /// Creates a new callback with the given `properties_provider`.
    pub fn new(properties_provider: Rc<P>) -> Self {
        SetPropertiesCallback { properties_provider }
    }
}

impl<P> MemoGroupCallback for SetPropertiesCallback<P>
where
    P: PropertiesProvider,
{
    type Expr = OperatorExpr;
    type Props = Properties;
    type Scope = OuterScope;
    type Metadata = OperatorMetadata;

    fn new_group(
        &self,
        expr: &Self::Expr,
        scope: &Self::Scope,
        provided_props: Self::Props,
        metadata: &Self::Metadata,
    ) -> Result<Self::Props, OptimizerError> {
        // Every time a new expression is added into a memo we need to compute logical properties of that expression.
        self.properties_provider.build_properties(expr, scope, provided_props, metadata.get_ref())
    }
}

/// Builder to create a [memo](Memo).
pub struct OperatorMemoBuilder {
    metadata: OperatorMetadata,
}

impl OperatorMemoBuilder {
    pub fn new(metadata: OperatorMetadata) -> Self {
        OperatorMemoBuilder { metadata }
    }

    /// Creates a memo with the given [PropertiesProvider].
    pub fn build_with_properties<T>(self, properties: T) -> ExprMemo
    where
        T: PropertiesProvider + 'static,
    {
        let propagate_properties = SetPropertiesCallback::new(Rc::new(properties));
        let memo_callback = Rc::new(propagate_properties);

        MemoBuilder::new(self.metadata).set_callback(memo_callback).build()
    }

    /// Creates a memo with the given [StatisticsBuilder].
    pub fn build_with_statistics<T>(self, statistics_builder: T) -> ExprMemo
    where
        T: StatisticsBuilder + 'static,
    {
        let properties_builder = LogicalPropertiesBuilder::new(statistics_builder);
        let propagate_properties = SetPropertiesCallback::new(Rc::new(properties_builder));
        let memo_callback = Rc::new(propagate_properties);

        MemoBuilder::new(self.metadata).set_callback(memo_callback).build()
    }
}

#[cfg(test)]
mod test {
    use crate::operators::scalar::col;
    use crate::operators::{Operator, OperatorExpr, Properties, ScalarProperties};

    #[test]
    fn test_operator_must_be_sync_and_send() {
        let expr = OperatorExpr::Scalar(col("a1"));
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
