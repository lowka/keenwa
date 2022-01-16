use crate::catalog::CatalogRef;
use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::memo::MemoExprState;
use crate::meta::{ColumnId, ColumnMetadata, MutableMetadata};
use crate::operators::relational::join::{JoinCondition, JoinOn, JoinType};
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalEmpty, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin,
    LogicalProjection, LogicalSelect, LogicalUnion, SetOperator,
};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::{AggregateFunction, BinaryOp, Expr, ExprRewriter};
use crate::operators::scalar::{ScalarExpr, ScalarNode};
use crate::operators::{ExprMemo, Operator, OperatorExpr};
use crate::properties::physical::PhysicalProperties;
use crate::properties::OrderingChoice;
use itertools::Itertools;
use std::cell::RefCell;
use std::convert::TryFrom;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

/// Ordering options.
#[derive(Debug, Clone)]
pub struct OrderingOptions {
    options: Vec<OrderingOption>,
}

impl OrderingOptions {
    pub fn new(options: Vec<OrderingOption>) -> Self {
        OrderingOptions { options }
    }

    /// Returns ordering options.
    fn options(&self) -> &[OrderingOption] {
        &self.options
    }
}

#[derive(Debug, Clone)]
pub struct OrderingOption {
    expr: ScalarExpr,
    descending: bool,
}

impl OrderingOption {
    /// Creates a new ordering option.
    pub fn new(expr: ScalarExpr, descending: bool) -> Self {
        OrderingOption { expr, descending }
    }

    /// Create new ordering option. A shorthand for `OrderingOption::new(ScalarExpr::ColumnName(<column>), <desc/asc>)`.
    pub fn by<T>(column_desc: (T, bool)) -> Self
    where
        T: Into<String>,
    {
        let column_name: String = column_desc.0.into();
        let descending = column_desc.1;
        Self::new(ScalarExpr::ColumnName(column_name), descending)
    }

    /// Returns the expression.
    pub fn expr(&self) -> &ScalarExpr {
        &self.expr
    }

    /// Returns `true` if ordering is descending and `false` otherwise.
    pub fn descending(&self) -> bool {
        self.descending
    }
}

impl From<OrderingOption> for OrderingOptions {
    fn from(o: OrderingOption) -> Self {
        OrderingOptions::new(vec![o])
    }
}

/// Callback invoked when a new operator is added to an operator tree by a [operator builder](self::OperatorBuilder).
pub trait OperatorCallback {
    /// Called when a new relational expression is added to an operator tree.
    fn new_rel_expr(&self, expr: Operator) -> RelNode;

    /// Called when a new scalar expression is added to an operator tree.
    fn new_scalar_expr(&self, expr: Operator) -> ScalarNode;
}

/// Provides API to build a tree of [operator tree](super::Operator).
/// When an operator is added to the tree this builder calls an appropriate method of an [operator callback](self::OperatorCallback).
#[derive(Clone)]
pub struct OperatorBuilder {
    callback: Rc<dyn OperatorCallback>,
    catalog: CatalogRef,
    metadata: Rc<MutableMetadata>,
    operator: Option<(Operator, OperatorScope)>,
    sub_query_builder: bool,
}

impl OperatorBuilder {
    /// Creates a new instance of OperatorBuilder.
    pub fn new(callback: Rc<dyn OperatorCallback>, catalog: CatalogRef, metadata: Rc<MutableMetadata>) -> Self {
        OperatorBuilder {
            callback,
            catalog,
            metadata,
            operator: None,
            sub_query_builder: false,
        }
    }

    /// Creates a new builder that shares all properties of the given builder
    /// and uses the the given operator as its current on operator tree.
    fn from_builder(builder: &OperatorBuilder, operator: Operator, scope: OperatorScope) -> Self {
        OperatorBuilder {
            callback: builder.callback.clone(),
            catalog: builder.catalog.clone(),
            metadata: builder.metadata.clone(),
            operator: Some((operator, scope)),
            sub_query_builder: builder.sub_query_builder,
        }
    }

    /// Adds scan operator to an operator tree. A scan operator can only be added as leaf node of a tree.
    pub fn get(mut self, source: &str, columns: Vec<impl Into<String>>) -> Result<Self, OptimizerError> {
        if self.operator.is_some() {
            return Result::Err(OptimizerError::Internal(
                "Adding a scan operator on top of another operator is not allowed".to_string(),
            ));
        }
        let columns: Vec<String> = columns.into_iter().map(|c| c.into()).collect();
        let table = self
            .catalog
            .get_table(source)
            .ok_or_else(|| OptimizerError::Argument(format!("Table does not exist. Table: {}", source)))?;

        let columns: Result<Vec<(String, ColumnId)>, OptimizerError> = columns
            .iter()
            .map(|name| {
                table.get_column(name).ok_or_else(|| {
                    OptimizerError::Argument(format!("Column does not exist. Column: {}. Table: {}", name, source))
                })
            })
            .map_ok(|column| {
                let column_name = column.name().clone();
                let metadata =
                    ColumnMetadata::new_table_column(column_name.clone(), column.data_type().clone(), source.into());
                let column_id = self.metadata.add_column(metadata);
                (column_name, column_id)
            })
            .collect();
        let columns = columns?;
        let column_ids: Vec<ColumnId> = columns.clone().into_iter().map(|(_, id)| id).collect();

        let expr = LogicalExpr::Get(LogicalGet {
            source: source.into(),
            columns: column_ids,
        });

        self.add_operator(expr, OperatorScope { columns });
        Ok(self)
    }

    /// Adds a select operator to an operator tree.
    pub fn select(mut self, filter: Option<ScalarExpr>) -> Result<Self, OptimizerError> {
        let (input, scope) = self.rel_node()?;
        let filter = if let Some(filter) = filter {
            let mut rewriter = RewriteExprs::new(&scope);
            let filter = filter.rewrite(&mut rewriter)?;
            let expr = self.add_scalar_node(filter);
            Some(expr)
        } else {
            None
        };

        let expr = LogicalExpr::Select(LogicalSelect { input, filter });

        self.add_operator(expr, scope);
        Ok(self)
    }

    /// Adds a projection operator to an operator tree.
    pub fn project(mut self, exprs: Vec<ScalarExpr>) -> Result<Self, OptimizerError> {
        let (input, scope) = self.rel_node()?;

        let mut column_ids = Vec::new();
        let mut output_columns = Vec::new();
        let mut new_exprs = vec![];

        for expr in exprs {
            let mut rewriter = RewriteExprs::for_projection(&scope, &output_columns);
            let expr = expr.clone().rewrite(&mut rewriter)?;
            let (id, name) = match expr.clone() {
                ScalarExpr::Column(id) => {
                    // Should never panic because RewriteExprs set an error when encounters an unknown column id.
                    let meta = self.metadata.get_column(&id);
                    (id, meta.name().clone())
                }
                _ => {
                    let (expr, name) = if let ScalarExpr::Alias(expr, name) = expr.clone() {
                        (*expr, name)
                    } else {
                        (expr.clone(), "?column?".to_string())
                    };
                    let data_type = resolve_expr_type(&expr, &self.metadata);
                    let column_meta = ColumnMetadata::new_synthetic_column(name.clone(), data_type, Some(expr));
                    let id = self.metadata.add_column(column_meta);
                    (id, name)
                }
            };
            column_ids.push(id);
            output_columns.push((name.clone(), id));
            new_exprs.push(expr.clone());
        }

        let expr = LogicalExpr::Projection(LogicalProjection {
            input,
            exprs: new_exprs,
            columns: column_ids,
        });

        self.add_operator(
            expr,
            OperatorScope {
                columns: output_columns,
            },
        );
        Ok(self)
    }

    /// Adds a join operator to an operator tree.
    pub fn join_using(
        mut self,
        mut right: OperatorBuilder,
        columns: Vec<(impl Into<String>, impl Into<String>)>,
    ) -> Result<Self, OptimizerError> {
        let (left, left_scope) = self.rel_node()?;
        let (right, right_scope) = right.rel_node()?;
        let mut columns_ids = Vec::new();

        for (l, r) in columns {
            let l: String = l.into();
            let r: String = r.into();

            let left_id = left_scope
                .columns
                .iter()
                .find(|(name, _)| name == &l)
                .map(|(_, id)| *id)
                .ok_or_else(|| OptimizerError::Argument("Join left side".into()))?;
            let right_id = right_scope
                .columns
                .iter()
                .find(|(name, _)| name == &r)
                .map(|(_, id)| *id)
                .ok_or_else(|| OptimizerError::Argument("Join right side".into()))?;
            columns_ids.push((left_id, right_id));
        }

        let mut output_columns = left_scope.columns;
        output_columns.extend_from_slice(&right_scope.columns);

        let condition = JoinCondition::using(columns_ids);
        let expr = LogicalExpr::Join(LogicalJoin {
            join_type: JoinType::Inner,
            left,
            right,
            condition,
        });

        self.add_operator(
            expr,
            OperatorScope {
                columns: output_columns,
            },
        );
        Ok(self)
    }

    /// Adds a join operator with ON <expr> condition to an operator tree.
    pub fn join_on(mut self, mut right: OperatorBuilder, expr: ScalarExpr) -> Result<Self, OptimizerError> {
        let (left, left_scope) = self.rel_node()?;
        let (right, right_scope) = right.rel_node()?;

        let mut output_columns = left_scope.columns;
        output_columns.extend_from_slice(&right_scope.columns);

        let scope = OperatorScope {
            columns: output_columns,
        };

        let mut rewriter = RewriteExprs::new(&scope);
        let expr = expr.rewrite(&mut rewriter)?;
        let expr = self.add_scalar_node(expr);
        let condition = JoinCondition::On(JoinOn::new(expr));
        let expr = LogicalExpr::Join(LogicalJoin {
            join_type: JoinType::Inner,
            left,
            right,
            condition,
        });

        self.add_operator(expr, scope);
        Ok(self)
    }

    /// Set ordering requirements for the current node of an operator tree.
    pub fn order_by(mut self, ordering: impl Into<OrderingOptions>) -> Result<Self, OptimizerError> {
        if let Some((operator, scope)) = self.operator.take() {
            let OrderingOptions { options } = ordering.into();
            let mut ordering_columns = Vec::with_capacity(options.len());

            for option in options {
                let OrderingOption {
                    expr,
                    descending: _descending,
                } = option;
                let mut rewriter = RewriteExprs::new(&scope);
                let expr = expr.rewrite(&mut rewriter)?;
                let column_id = if let ScalarExpr::Column(id) = expr {
                    id
                } else {
                    let expr_type = resolve_expr_type(&expr, &self.metadata);
                    let column_meta = ColumnMetadata::new_synthetic_column("".into(), expr_type, Some(expr));
                    self.metadata.add_column(column_meta)
                };
                ordering_columns.push(column_id);
            }

            let ordering = OrderingChoice::new(ordering_columns);
            let required = PhysicalProperties::new(ordering);
            self.operator = Some((operator.with_required(required), scope));
            Ok(self)
        } else {
            Err(OptimizerError::Internal("No input operator".to_string()))
        }
    }

    /// Adds a union operator.
    pub fn union(mut self, right: OperatorBuilder) -> Result<Self, OptimizerError> {
        self.build_set_operator(SetOperator::Union, false, right)?;
        Ok(self)
    }

    /// Adds a union all operator.
    pub fn union_all(mut self, right: OperatorBuilder) -> Result<Self, OptimizerError> {
        self.build_set_operator(SetOperator::Union, true, right)?;
        Ok(self)
    }

    /// Adds a relation that produces no rows.
    /// The `return_one_row` flag with value `true` can be used to support queries like `SELECT 1`
    /// where projection has no input operator.
    ///
    /// This operator can only be added as a leaf node of an operator tree
    /// (the first operator build created by an operator builder).
    pub fn empty(mut self, return_one_row: bool) -> Result<Self, OptimizerError> {
        if self.operator.is_some() {
            return Result::Err(OptimizerError::Internal(
                "Empty relation can not be added on top of another operator".to_string(),
            ));
        };
        let expr = LogicalExpr::Empty(LogicalEmpty { return_one_row });
        self.add_operator(expr, OperatorScope { columns: vec![] });
        Ok(self)
    }

    /// Returns a builder to construct an aggregate operator.
    pub fn aggregate_builder(&mut self) -> AggregateBuilder {
        AggregateBuilder {
            builder: self,
            aggr_exprs: vec![],
            group_exprs: vec![],
        }
    }

    /// Creates a builder that can be used to construct a nested sub-query.
    pub fn sub_query_builder(&self) -> Self {
        OperatorBuilder {
            callback: self.callback.clone(),
            catalog: self.catalog.clone(),
            metadata: self.metadata.clone(),
            operator: None,
            sub_query_builder: true,
        }
    }

    /// Creates an operator tree and returns its metadata.
    /// If this builder has been created via call to [Self::sub_query_builder()] this method returns an error.
    pub fn build(self) -> Result<Operator, OptimizerError> {
        if self.sub_query_builder {
            Err(OptimizerError::Internal("Use to_sub_query() to create sub queries".to_string()))
        } else {
            let (operator, _) = self.operator.expect("No operator");
            Ok(operator)
        }
    }

    /// Creates a nested sub query expression from this nested operator builder.
    /// If this builder has not been created via call to [Self::sub_query_builder()] this method returns an error.
    pub fn to_sub_query(mut self) -> Result<ScalarExpr, OptimizerError> {
        if self.sub_query_builder {
            let (operator, _) = self.rel_node()?;
            Ok(ScalarExpr::SubQuery(operator))
        } else {
            let message = "Sub queries can only be created using builders instantiated via call to OperatorBuilder::sub_query_builder()";
            Err(OptimizerError::Internal(message.into()))
        }
    }

    fn build_set_operator(
        &mut self,
        set_op: SetOperator,
        all: bool,
        mut right: OperatorBuilder,
    ) -> Result<(), OptimizerError> {
        let (left, left_scope) = self.rel_node()?;
        let (right, right_scope) = right.rel_node()?;

        if left_scope.columns.len() != right_scope.columns.len() {
            return Err(OptimizerError::Argument("Union: Number of columns does not match".to_string()));
        }

        let mut columns = Vec::new();
        let mut column_ids = Vec::new();
        for (i, (l, r)) in left_scope.columns.iter().zip(right_scope.columns.iter()).enumerate() {
            let data_type = {
                let l = self.metadata.get_column(&l.1);
                let r = self.metadata.get_column(&r.1);

                if l.data_type() != r.data_type() {
                    let message =
                        format!("Union: Data type of {}-th column does not match. Type casting is not implemented", i);
                    return Err(OptimizerError::NotImplemented(message));
                }
                l.data_type().clone()
            };

            let column_name = "".to_string();
            let column_meta = ColumnMetadata::new_synthetic_column(column_name.clone(), data_type, None);
            let column_id = self.metadata.add_column(column_meta);
            columns.push((column_name, column_id));
            column_ids.push(column_id);
        }

        let expr = match set_op {
            SetOperator::Union => LogicalExpr::Union(LogicalUnion {
                left,
                right,
                all,
                columns: column_ids,
            }),
            SetOperator::Intersect => LogicalExpr::Intersect(LogicalIntersect {
                left,
                right,
                all,
                columns: column_ids,
            }),
            SetOperator::Except => LogicalExpr::Except(LogicalExcept {
                left,
                right,
                all,
                columns: column_ids,
            }),
        };

        self.add_operator(expr, OperatorScope { columns });
        Ok(())
    }

    fn add_operator(&mut self, expr: LogicalExpr, output_columns: OperatorScope) {
        let operator = Operator::from(OperatorExpr::from(expr));
        self.operator = Some((operator, output_columns));
    }

    fn rel_node(&mut self) -> Result<(RelNode, OperatorScope), OptimizerError> {
        let (operator, scope) = self
            .operator
            .take()
            .ok_or_else(|| OptimizerError::Internal("No input operator".to_string()))?;

        let rel_node = self.callback.new_rel_expr(operator);
        Ok((rel_node, scope))
    }

    fn add_scalar_node(&self, expr: ScalarExpr) -> ScalarNode {
        let operator = Operator::from(OperatorExpr::from(expr));
        self.callback.new_scalar_expr(operator)
    }
}

impl Debug for OperatorBuilder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        //[E0658]: trait upcasting coercion is experimental
        let catalog = format!("{:?}", self.catalog.as_ref());
        if let Some((operator, scope)) = self.operator.as_ref() {
            f.debug_struct("OperatorBuilder")
                .field("catalog", &catalog)
                .field("metadata", self.metadata.as_ref())
                .field("operator", operator)
                .field("scope", scope)
                .finish()
        } else {
            f.debug_struct("OperatorBuilder")
                .field("catalog", &catalog)
                .field("metadata", self.metadata.as_ref())
                .finish()
        }
    }
}

/// Stores columns available to the current node of an operator tree.
#[derive(Debug, Clone)]
struct OperatorScope {
    // (alias, column_id)
    columns: Vec<(String, ColumnId)>,
}

struct RewriteExprs<'a> {
    scope: &'a OperatorScope,
    // Aliased expressions must be visible inside a scope of a projection.
    projection: &'a [(String, ColumnId)],
}

impl<'a> RewriteExprs<'a> {
    fn new(scope: &'a OperatorScope) -> Self {
        RewriteExprs { scope, projection: &[] }
    }

    fn for_projection(scope: &'a OperatorScope, projection: &'a [(String, ColumnId)]) -> Self {
        RewriteExprs { scope, projection }
    }
}

impl ExprRewriter<RelNode> for RewriteExprs<'_> {
    type Error = OptimizerError;

    fn pre_rewrite(&mut self, _expr: &Expr<RelNode>) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
        match expr {
            ScalarExpr::Column(column_id) => {
                let exists = self.scope.columns.iter().any(|(_, id)| column_id == *id);
                if exists {
                    return Err(OptimizerError::Internal(format!(
                        "Projection: Unexpected column : {}. Input columns: {:?}",
                        column_id, self.scope.columns
                    )));
                }
                Ok(expr)
            }
            ScalarExpr::ColumnName(ref column_name) => {
                let column_id = self.scope.columns.iter().find(|(name, _)| column_name == name).map(|(_, id)| *id);
                let column_id = column_id
                    .or_else(|| self.projection.iter().find(|(name, _)| column_name == name).map(|(_, id)| *id));

                match column_id {
                    Some(column_id) => Ok(ScalarExpr::Column(column_id)),
                    None => {
                        return Err(OptimizerError::Internal(format!(
                            "Projection: Unexpected column : {}. Input columns: {:?}",
                            column_name, self.scope.columns
                        )));
                    }
                }
            }
            ScalarExpr::SubQuery(ref rel_node) => {
                match rel_node.state() {
                    MemoExprState::Owned(_) => {
                        //FIXME: Add method to handle nested relational expressions to OperatorCallback?
                        Err(OptimizerError::Internal(
                            "Use OperatorBuilder::sub_query_builder to build a nested sub query".to_string(),
                        ))
                    }
                    MemoExprState::Memo(_) => Ok(expr),
                }
            }
            _ => Ok(expr),
        }
    }
}

/// A builder for aggregate operators.  
pub struct AggregateBuilder<'a> {
    builder: &'a mut OperatorBuilder,
    aggr_exprs: Vec<AggrExpr>,
    group_exprs: Vec<String>,
}

enum AggrExpr {
    Function { func: AggregateFunction, column: String },
    Column(String),
}

impl AggregateBuilder<'_> {
    /// Adds aggregate function `func` with the given column as an argument.
    pub fn add_func(mut self, func: &str, column_name: &str) -> Result<Self, OptimizerError> {
        let func = AggregateFunction::try_from(func)
            .map_err(|_| OptimizerError::Argument(format!("Unknown aggregate function {}", func)))?;
        let aggr_expr = AggrExpr::Function {
            func,
            column: column_name.into(),
        };
        self.aggr_exprs.push(aggr_expr);
        Ok(self)
    }

    /// Adds column to the aggregate operator. The given column must appear in group_by clause.
    pub fn add_column(mut self, column_name: &str) -> Result<Self, OptimizerError> {
        self.aggr_exprs.push(AggrExpr::Column(column_name.into()));
        Ok(self)
    }

    /// Adds grouping by the given column.
    pub fn group_by(mut self, column_name: &str) -> Result<Self, OptimizerError> {
        self.group_exprs.push(column_name.into());
        Ok(self)
    }

    /// Builds an aggregate operator and adds in to an operator tree.
    pub fn build(mut self) -> Result<OperatorBuilder, OptimizerError> {
        let (input, scope) = self.builder.rel_node()?;

        let aggr_exprs = std::mem::take(&mut self.aggr_exprs);
        let aggr_exprs: Result<Vec<(ScalarNode, (String, ColumnId))>, OptimizerError> = aggr_exprs
            .into_iter()
            .map(|expr| match expr {
                AggrExpr::Function { func, column } => {
                    let mut rewriter = RewriteExprs::new(&scope);
                    let expr = ScalarExpr::ColumnName(column);
                    let expr = expr.rewrite(&mut rewriter)?;

                    let name = format!("{}", func);
                    let expr = ScalarExpr::Aggregate {
                        func,
                        args: vec![expr],
                        filter: None,
                    };
                    let metadata = &self.builder.metadata;
                    let data_type = resolve_expr_type(&expr, metadata);

                    let column_meta = ColumnMetadata::new_synthetic_column(name.clone(), data_type, Some(expr.clone()));
                    let id = metadata.add_column(column_meta);
                    let expr = self.builder.add_scalar_node(expr);

                    Ok((expr, (name, id)))
                }
                AggrExpr::Column(name) => {
                    let mut rewriter = RewriteExprs::new(&scope);
                    let expr = ScalarExpr::ColumnName(name.clone());
                    let expr = expr.rewrite(&mut rewriter)?;
                    let expr = self.builder.add_scalar_node(expr);
                    let id = if let ScalarExpr::Column(id) = expr.expr() {
                        *id
                    } else {
                        unreachable!("Column name has not been replaced with metadata id")
                    };
                    if self.group_exprs.iter().any(|group_column| group_column == &name) {
                        Ok((expr, (name, id)))
                    } else {
                        Err(OptimizerError::Internal(format!("Column {} must appear in group by clause", name)))
                    }
                }
            })
            .collect();
        let aggr_exprs = aggr_exprs?;
        let (aggr_exprs, output_columns): (Vec<ScalarNode>, Vec<(String, ColumnId)>) = aggr_exprs.into_iter().unzip();
        let column_ids: Vec<ColumnId> = output_columns.iter().map(|(_, id)| *id).collect();

        let group_exprs = std::mem::take(&mut self.group_exprs);
        let group_exprs: Result<Vec<ScalarNode>, OptimizerError> = group_exprs
            .into_iter()
            .map(|name| {
                let mut rewriter = RewriteExprs::new(&scope);
                let expr = ScalarExpr::ColumnName(name);
                let expr = expr.rewrite(&mut rewriter)?;
                let expr = self.builder.add_scalar_node(expr);

                Ok(expr)
            })
            .collect();
        let group_exprs = group_exprs?;

        let aggregate = LogicalExpr::Aggregate(LogicalAggregate {
            input,
            aggr_exprs,
            group_exprs,
            columns: column_ids,
        });
        let operator = Operator::from(OperatorExpr::from(aggregate));

        Ok(OperatorBuilder::from_builder(
            self.builder,
            operator,
            OperatorScope {
                columns: output_columns,
            },
        ))
    }
}

/// [OperatorCallback] that does nothing.
#[derive(Debug)]
pub struct NoOpOperatorCallback;

impl OperatorCallback for NoOpOperatorCallback {
    fn new_rel_expr(&self, expr: Operator) -> RelNode {
        assert!(expr.expr().is_relational(), "Not a relational expression");
        RelNode::from(expr)
    }

    fn new_scalar_expr(&self, expr: Operator) -> ScalarNode {
        assert!(expr.expr().is_scalar(), "Not a scalar expression");
        ScalarNode::from_mexpr(expr)
    }
}

/// A callback that copies operators into a memo.
#[derive(Debug)]
pub struct MemoizeOperatorCallback {
    // RefCell is necessary because the memo is shared by multiple operator builders.
    memo: RefCell<ExprMemo>,
}

impl MemoizeOperatorCallback {
    /// Creates a callback that uses the given memo.
    pub fn new(memo: ExprMemo) -> Self {
        MemoizeOperatorCallback {
            memo: RefCell::new(memo),
        }
    }

    /// Consumes this callback and returns the underlying memo.
    pub fn into_inner(self) -> ExprMemo {
        self.memo.into_inner()
    }
}

impl OperatorCallback for MemoizeOperatorCallback {
    fn new_rel_expr(&self, expr: Operator) -> RelNode {
        assert!(expr.expr().is_relational(), "Expected a relational expression but got {:?}", expr);
        let mut memo = self.memo.borrow_mut();
        let expr = memo.insert_group(expr);
        RelNode::from_mexpr(expr)
    }

    fn new_scalar_expr(&self, expr: Operator) -> ScalarNode {
        assert!(expr.expr().is_scalar(), "Expected a scalar expression but got {:?}", expr);
        let mut memo = self.memo.borrow_mut();
        let expr = memo.insert_group(expr);
        ScalarNode::from_mexpr(expr)
    }
}

fn resolve_expr_type(expr: &ScalarExpr, metadata: &MutableMetadata) -> DataType {
    match expr {
        ScalarExpr::Column(column_id) => metadata.get_column(column_id).data_type().clone(),
        ScalarExpr::ColumnName(_) => panic!("Expr Column(name) should have been replaced with Column(id)"),
        ScalarExpr::Scalar(value) => value.data_type(),
        ScalarExpr::BinaryExpr { lhs, op, rhs } => {
            let left_tpe = resolve_expr_type(lhs, metadata);
            let right_tpe = resolve_expr_type(rhs, metadata);
            resolve_binary_expr_type(left_tpe, op, right_tpe)
        }
        ScalarExpr::Not(expr) => {
            let tpe = resolve_expr_type(expr, metadata);
            assert_eq!(tpe, DataType::Bool, "Invalid argument type for NOT operator");
            tpe
        }
        ScalarExpr::Alias(expr, _) => resolve_expr_type(expr, metadata),
        ScalarExpr::Aggregate { func, args, .. } => {
            for (i, arg) in args.iter().enumerate() {
                let arg_tpe = resolve_expr_type(arg, metadata);
                let expected_tpe = match func {
                    AggregateFunction::Avg
                    | AggregateFunction::Max
                    | AggregateFunction::Min
                    | AggregateFunction::Sum => DataType::Int32,
                    AggregateFunction::Count => arg_tpe.clone(),
                };
                assert_eq!(
                    &arg_tpe, &expected_tpe,
                    "Invalid argument type for aggregate function {}. Argument#{} {}",
                    func, i, arg
                );
            }
            DataType::Int32
        }
        ScalarExpr::SubQuery(_) => DataType::Int32,
    }
}

fn resolve_binary_expr_type(lhs: DataType, _op: &BinaryOp, rhs: DataType) -> DataType {
    assert_eq!(lhs, rhs, "Types does not match");
    // TODO: Correctly resolve data types.
    DataType::Bool
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::catalog::mutable::MutableCatalog;
    use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
    use crate::memo::format_memo;
    use crate::operators::properties::LogicalPropertiesBuilder;
    use crate::operators::scalar::value::ScalarValue;
    use crate::operators::Properties;
    use crate::optimizer::SetPropertiesCallback;
    use crate::rules::testing::format_operator_tree;
    use crate::statistics::NoStatisticsBuilder;
    use std::sync::Arc;

    #[test]
    fn test_get() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| builder.get("A", vec!["a1"])?.build());

        tester.expect_expr(
            r#"
LogicalGet A cols=[1]
  output cols: [1]
Metadata:
  col:1 A.a1 Int32
Memo:
  00 LogicalGet A cols=[1]
"#,
        );
    }

    #[test]
    fn test_get_order_by() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let ord = OrderingOption::by(("a1", false));
            let from_a = builder.get("A", vec!["a1"])?.order_by(ord)?;

            from_a.build()
        });

        //TODO: include properties
        tester.expect_expr(
            r#"
LogicalGet A cols=[1]
  output cols: [1]
Metadata:
  col:1 A.a1 Int32
Memo:
  00 LogicalGet A cols=[1]
"#,
        );
    }

    #[test]
    fn test_select() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let filter = ScalarExpr::BinaryExpr {
                lhs: Box::new(ScalarExpr::ColumnName("a1".into())),
                op: BinaryOp::Gt,
                rhs: Box::new(ScalarExpr::Scalar(ScalarValue::Int32(100))),
            };

            builder.get("A", vec!["a1"])?.select(Some(filter))?.build()
        });

        tester.expect_expr(
            r#"
LogicalSelect
  input: LogicalGet A cols=[1]
  filter: Expr col:1 > 100
  output cols: [1]
Metadata:
  col:1 A.a1 Int32
Memo:
  02 LogicalSelect input=00 filter=01
  01 Expr col:1 > 100
  00 LogicalGet A cols=[1]
"#,
        );
    }

    #[test]
    fn test_projection() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a1", "a2"])?;

            let col_a2 = ScalarExpr::ColumnName("a2".into());
            let val_i100 = ScalarExpr::Scalar(ScalarValue::Int32(100));
            let alias = ScalarExpr::Alias(Box::new(col_a2.clone()), "a2_alias".into());
            let use_alias = ScalarExpr::Alias(Box::new(alias.clone()), "a2_alias_alias".into());
            let projection_list = vec![col_a2, val_i100, alias, use_alias];

            from_a.project(projection_list)?.build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[2, 3, 4, 5] exprs=[col:2, 100, col:2 AS a2_alias, col:2 AS a2_alias AS a2_alias_alias]
  input: LogicalGet A cols=[1, 2]
  output cols: [2, 3, 4, 5]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 ?column? Int32, expr: 100
  col:4 a2_alias Int32, expr: col:2
  col:5 a2_alias_alias Int32, expr: col:2 AS a2_alias
Memo:
  01 LogicalProjection input=00 cols=[2, 3, 4, 5] exprs=[col:2, 100, col:2 AS a2_alias, col:2 AS a2_alias AS a2_alias_alias]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_projection_of_projection() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let projection_list1 = vec![
                ScalarExpr::ColumnName("a2".into()),
                ScalarExpr::Scalar(ScalarValue::Int32(100)),
                ScalarExpr::ColumnName("a1".into()),
            ];

            let projection_list2 = vec![ScalarExpr::ColumnName("a2".into())];

            builder
                .get("A", vec!["a1", "a2"])?
                .project(projection_list1)?
                .project(projection_list2)?
                .build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[2] exprs=[col:2]
  input: LogicalProjection cols=[2, 3, 1] exprs=[col:2, 100, col:1]
      input: LogicalGet A cols=[1, 2]
  output cols: [2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 ?column? Int32, expr: 100
Memo:
  02 LogicalProjection input=01 cols=[2] exprs=[col:2]
  01 LogicalProjection input=00 cols=[2, 3, 1] exprs=[col:2, 100, col:1]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_join_using() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.clone().get("A", vec!["a1", "a2"])?;
            let right = builder.get("B", vec!["b1", "b2"])?;
            let join = left.join_using(right, vec![("a1", "b2")])?;

            join.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin using=[(1, 4)]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  02 LogicalJoin left=00 right=01 using=[(1, 4)]
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_join_on() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.clone().get("A", vec!["a1", "a2"])?;
            let right = builder.get("B", vec!["b1", "b2"])?;
            let expr = ScalarExpr::BinaryExpr {
                lhs: Box::new(ScalarExpr::ColumnName("a1".into())),
                op: BinaryOp::Eq,
                rhs: Box::new(ScalarExpr::ColumnName("b1".into())),
            };
            let join = left.join_on(right, expr)?;

            join.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin on=col:1 = col:3
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  03 LogicalJoin left=00 right=01 on=col:1 = col:3
  02 Expr col:1 = col:3
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_empty() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| builder.empty(true)?.build());

        tester.expect_expr(
            r#"
LogicalEmpty return_one_row=true
  output cols: []
Metadata:
Memo:
  00 LogicalEmpty return_one_row=true
"#,
        );
    }

    #[test]
    fn test_prohibit_empty_as_an_intermediate_operator() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            builder.get("A", vec!["a1"])?.empty(false)?;

            unreachable!()
        });

        tester.expect_error("Empty relation can not be added on top of another operator");
    }

    #[test]
    fn test_nested() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.clone().get("A", vec!["a1", "a2"])?;

            let sub_query = builder
                .sub_query_builder()
                .empty(true)?
                .project(vec![ScalarExpr::Scalar(ScalarValue::Bool(true))])?
                .to_sub_query()?;

            let filter = ScalarExpr::BinaryExpr {
                lhs: Box::new(sub_query),
                op: BinaryOp::Eq,
                rhs: Box::new(ScalarExpr::Scalar(ScalarValue::Bool(true))),
            };
            let select = from_a.select(Some(filter))?;

            select.build()
        });

        tester.expect_expr(
            r#"
LogicalSelect
  input: LogicalGet A cols=[1, 2]
  filter: Expr SubQuery 01 = true
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 ?column? Bool, expr: true
Memo:
  04 LogicalSelect input=02 filter=03
  03 Expr SubQuery 01 = true
  02 LogicalGet A cols=[1, 2]
  01 LogicalProjection input=00 cols=[3] exprs=[true]
  00 LogicalEmpty return_one_row=true
"#,
        );
    }

    #[test]
    fn test_prohibit_nested_sub_queries_not_created_via_sub_query_builder() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_b = builder.get("B", vec!["b1"])?;
            let _sub_query = from_b.to_sub_query()?;

            unreachable!()
        });

        tester.expect_error("call to OperatorBuilder::sub_query_builder()")
    }

    #[test]
    fn test_prohibit_non_memoized_expressions_in_nested_sub_queries() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.clone().get("A", vec!["a1", "a2"])?;

            let sub_query = builder
                .empty(false)?
                .project(vec![ScalarExpr::Scalar(ScalarValue::Bool(true))])?
                .build()?;
            let expr = ScalarExpr::SubQuery(RelNode::from(sub_query));

            let filter = ScalarExpr::BinaryExpr {
                lhs: Box::new(expr),
                op: BinaryOp::Eq,
                rhs: Box::new(ScalarExpr::Scalar(ScalarValue::Bool(true))),
            };

            let _select = from_a.select(Some(filter))?;

            unreachable!()
        });

        tester.expect_error("Use OperatorBuilder::sub_query_builder to build a nested sub query")
    }

    #[test]
    fn test_aggregate() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let mut from_a = builder.get("A", vec!["a1", "a2", "a3"])?;
            let sum = from_a
                .aggregate_builder()
                .add_column("a1")?
                .add_func("sum", "a1")?
                .group_by("a1")?
                .build()?;

            sum.build()
        });
        tester.expect_expr(
            r#"
LogicalAggregate
  input: LogicalGet A cols=[1, 2, 3]
  : Expr col:1
  : Expr sum(col:1)
  : Expr col:1
  output cols: [1, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 sum Int32, expr: sum(col:1)
Memo:
  03 LogicalAggregate input=00 01 02 01
  02 Expr sum(col:1)
  01 Expr col:1
  00 LogicalGet A cols=[1, 2, 3]
"#,
        );
    }

    struct OperatorBuilderTester {
        operator: Box<dyn Fn(OperatorBuilder) -> Result<Operator, OptimizerError>>,
        metadata: Rc<MutableMetadata>,
        memoization: Rc<MemoizeOperatorCallback>,
    }

    impl OperatorBuilderTester {
        fn new() -> Self {
            let properties_builder = Rc::new(LogicalPropertiesBuilder::new(NoStatisticsBuilder));
            let metadata = Rc::new(MutableMetadata::new());
            let memoization = Rc::new(MemoizeOperatorCallback::new(ExprMemo::with_callback(
                metadata.clone(),
                Rc::new(SetPropertiesCallback::new(properties_builder)),
            )));

            OperatorBuilderTester {
                operator: Box::new(|_| panic!("Operator has not been specified")),
                metadata,
                memoization,
            }
        }

        fn build_operator<F>(&mut self, f: F)
        where
            F: Fn(OperatorBuilder) -> Result<Operator, OptimizerError> + 'static,
        {
            self.operator = Box::new(f)
        }

        fn expect_error(mut self, msg: &str) {
            let result = self.do_build_operator();
            let err = result.expect_err("Expected an error");
            let actual_err_str = format!("{}", err);

            assert!(actual_err_str.contains(msg), "Unexpected error message. Expected {}. Error: {}", msg, err);
        }

        fn expect_expr(mut self, expected: &str) {
            let result = self.do_build_operator();
            let expr = result.expect("Failed to build an operator");

            let mut buf = String::new();
            buf.push('\n');

            let memoization = Rc::try_unwrap(self.memoization).unwrap();
            let mut memo = memoization.into_inner();
            let expr = memo.insert_group(expr);

            buf.push_str(format_operator_tree(&expr).as_str());
            buf.push('\n');

            let props = expr.props();
            match props {
                Properties::Relational(props) => {
                    buf.push_str(format!("  output cols: {:?}\n", props.logical().output_columns()).as_str());
                }
                Properties::Scalar(_) => {}
            }
            buf.push_str("Metadata:\n");

            let columns: Vec<ColumnMetadata> =
                self.metadata.get_columns().into_iter().sorted_by(|a, b| a.id().cmp(b.id())).collect();

            for column in columns {
                let id = column.id();
                let expr = column.expr();
                let table = column.table();
                let column_name = if !column.name().is_empty() {
                    column.name().as_str()
                } else {
                    "?column?"
                };
                let column_info = match (expr, table) {
                    (None, None) => format!("  col:{} {} {:?}", id, column_name, column.data_type()),
                    (None, Some(table)) => format!("  col:{} {}.{} {:?}", id, table, column_name, column.data_type()),
                    (Some(expr), _) => format!("  col:{} {} {:?}, expr: {}", id, column_name, column.data_type(), expr),
                };
                buf.push_str(column_info.as_str());
                buf.push('\n');
            }

            buf.push_str("Memo:\n");
            let memo_as_string = format_memo(&memo);
            let lines = memo_as_string
                .split('\n')
                .map(|l| {
                    if !l.is_empty() {
                        let mut s = String::new();
                        s.push_str("  ");
                        s.push_str(l);
                        s
                    } else {
                        l.to_string()
                    }
                })
                .join("\n");

            buf.push_str(lines.as_str());

            assert_eq!(buf.as_str(), expected);
        }

        fn do_build_operator(&mut self) -> Result<Operator, OptimizerError> {
            let catalog = Arc::new(MutableCatalog::new());
            catalog.add_table(
                DEFAULT_SCHEMA,
                TableBuilder::new("A")
                    .add_column("a1", DataType::Int32)
                    .add_column("a2", DataType::Int32)
                    .add_column("a3", DataType::Int32)
                    .add_column("a4", DataType::Int32)
                    .build(),
            );

            catalog.add_table(
                DEFAULT_SCHEMA,
                TableBuilder::new("B")
                    .add_column("b1", DataType::Int32)
                    .add_column("b2", DataType::Int32)
                    .add_column("b3", DataType::Int32)
                    .build(),
            );

            let builder = OperatorBuilder::new(self.memoization.clone(), catalog, self.metadata.clone());
            (self.operator)(builder)
        }
    }
}
