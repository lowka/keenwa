use std::cell::RefCell;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{Debug, Display, Formatter};
use std::iter::FromIterator;
use std::rc::Rc;

use aggregate::AggregateBuilder;
use itertools::Itertools;
use projection::{ProjectionList, ProjectionListBuilder};
use scope::{OperatorScope, RelationInScope};

use crate::catalog::CatalogRef;
use crate::error::OptimizerError;
use crate::memo::{MemoExprState, Props};
use crate::meta::{ColumnId, ColumnMetadata, MetadataRef, MutableMetadata};
use crate::operators::builder::aggregate::{add_window_aggregates, has_window_aggregates};
use crate::operators::builder::scope::CteInScope;
use crate::operators::builder::subqueries::{
    decorrelate_subqueries, possibly_has_correlated_subqueries, BuildFilter, BuildOperators,
};
use crate::operators::relational::join::{JoinCondition, JoinOn, JoinType, JoinUsing};
use crate::operators::relational::logical::{
    LogicalDistinct, LogicalEmpty, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin, LogicalLimit,
    LogicalOffset, LogicalProjection, LogicalSelect, LogicalUnion, LogicalValues,
};
use crate::operators::relational::{RelNode, SetOperator};
use crate::operators::scalar::expr::ExprRewriter;
use crate::operators::scalar::types::resolve_expr_type;
use crate::operators::scalar::value::ScalarValue;
use crate::operators::scalar::{get_subquery, scalar, ScalarExpr, ScalarNode};
use crate::operators::{ExprMemo, Operator, OperatorExpr, OperatorMetadata, OuterScope, Properties, ScalarProperties};
use crate::properties::partitioning::Partitioning;
use crate::properties::physical::{PhysicalProperties, Presentation, RequiredProperties};
use crate::properties::{OrderingChoice, OrderingColumn};
use crate::{not_implemented, not_supported};

mod aggregate;
mod projection;
mod scope;
mod subqueries;

/// Table alias.
#[derive(Debug, Clone)]
pub struct TableAlias {
    /// New name.
    pub name: String,
    /// columns to rename.
    pub columns: Vec<String>,
}

/// Ordering options.
#[derive(Debug, Clone)]
pub struct OrderingOptions {
    pub options: Vec<OrderingOption>,
}

impl OrderingOptions {
    pub fn new(options: Vec<OrderingOption>) -> Self {
        OrderingOptions { options }
    }

    /// Returns ordering options.
    pub fn options(&self) -> &[OrderingOption] {
        &self.options
    }
}

/// Specifies how the data is sorted.
#[derive(Debug, Clone)]
pub struct OrderingOption {
    /// An instance of an [ScalarExpr::Ordering].
    ///
    /// [ScalarExpr::Ordering]: crate::operators::scalar::expr::Expr::Ordering
    pub expr: ScalarExpr,
    /// Whether this ordering is descending or not.
    pub descending: bool,
    //TODO: Add null_first
}

impl OrderingOption {
    /// Creates an instance of [OrderingOption].
    ///
    /// # Panics
    ///
    /// This method panics if the given expression is an instance of [ScalarExpr::Ordering].
    ///
    /// [ScalarExpr::Ordering]: crate::operators::scalar::expr::Expr::Ordering
    pub fn new(expr: ScalarExpr, descending: bool) -> Self {
        assert!(
            !matches!(expr, ScalarExpr::Ordering { .. }),
            "Can not to create OrderingOption from Ordering expression"
        );
        OrderingOption { expr, descending }
    }

    /// Creates an option to order data by `i`-th column (1-based)
    pub fn by_position(i: usize, descending: bool) -> Self {
        OrderingOption::new(ScalarExpr::Scalar(ScalarValue::Int32(Some(i as i32))), descending)
    }

    /// Creates an option to order data by the given column.
    pub fn by<T>(column: T, descending: bool) -> Self
    where
        T: Into<String>,
    {
        let expr = ScalarExpr::ColumnName(column.into());
        OrderingOption::new(expr, descending)
    }

    /// Returns a reference to the expression.
    pub fn expr(&self) -> &ScalarExpr {
        &self.expr
    }
}

impl From<OrderingOption> for OrderingOptions {
    fn from(o: OrderingOption) -> Self {
        OrderingOptions::new(vec![o])
    }
}

/// Partitioning options.
pub enum PartitioningOptions {
    /// Singleton partitioned.
    Singleton,
    /// Tuples partitioned by the given columns.
    Partitioned(Vec<String>),
}

impl<T, S> From<T> for PartitioningOptions
where
    T: IntoIterator<Item = S>,
    S: Into<String>,
{
    fn from(opts: T) -> Self {
        let mut cols = Vec::new();
        for v in opts.into_iter() {
            let name = v.into();
            cols.push(name);
        }
        PartitioningOptions::Partitioned(cols)
    }
}

/// Callback invoked when a new operator is added to an operator tree by a [operator builder](self::OperatorBuilder).
pub trait OperatorCallback {
    /// Called when a new relational expression is added to an operator tree.
    fn new_rel_expr(&self, expr: Operator, scope: &OperatorScope) -> Result<RelNode, OptimizerError>;

    /// Called when a new scalar expression is added to an operator tree.
    fn new_scalar_expr(&self, expr: Operator, scope: &OperatorScope) -> Result<ScalarNode, OptimizerError>;
}

/// Provides API to build a tree of [operator tree](super::Operator).
/// When an operator is added to the tree this builder calls an appropriate method of an [operator callback](self::OperatorCallback).
#[derive(Clone)]
pub struct OperatorBuilder {
    callback: Rc<dyn OperatorCallback>,
    catalog: CatalogRef,
    metadata: Rc<MutableMetadata>,
    scope: Option<OperatorScope>,
    operator: Option<Operator>,
    decorrelate_sub_queries: bool,
}

/// Additional configuration options for a [OperatorBuilder].
#[derive(Debug, Clone, Default)]
pub struct OperatorBuilderConfig {
    // Make not public.
    // Use builder-lite pattern.
    pub decorrelate_subqueries: bool,
}

impl OperatorBuilder {
    /// Creates a new instance of OperatorBuilder.
    pub fn new(callback: Rc<dyn OperatorCallback>, catalog: CatalogRef, metadata: Rc<MutableMetadata>) -> Self {
        OperatorBuilder::with_config(OperatorBuilderConfig::default(), callback, catalog, metadata)
    }

    /// Creates a new instance of OperatorBuilder with the given configuration options.
    pub fn with_config(
        config: OperatorBuilderConfig,
        callback: Rc<dyn OperatorCallback>,
        catalog: CatalogRef,
        metadata: Rc<MutableMetadata>,
    ) -> Self {
        OperatorBuilder {
            callback,
            catalog,
            metadata,
            scope: None,
            operator: None,
            decorrelate_sub_queries: config.decorrelate_subqueries,
        }
    }

    /// Creates a new builder that shares all properties of the given builder
    /// and uses the given operator as its parent node in an operator tree.
    fn from_builder(
        builder: &OperatorBuilder,
        operator: Operator,
        scope: &OperatorScope,
        columns: Vec<(String, ColumnId)>,
    ) -> Self {
        let scope = if let Some(scope) = &builder.scope {
            // Set columns if scope exists.
            scope.with_new_columns(columns)
        } else {
            // If the given builder does not have the scope yet
            // create a new scope from the given columns.
            OperatorScope::from_columns(columns, scope.outer_columns().to_vec())
        };
        OperatorBuilder {
            callback: builder.callback.clone(),
            catalog: builder.catalog.clone(),
            metadata: builder.metadata.clone(),
            scope: Some(scope),
            operator: Some(operator),
            decorrelate_sub_queries: builder.decorrelate_sub_queries,
        }
    }

    /// Adds a common table expression with the given table alias.
    pub fn with_table_expr(mut self, alias: TableAlias, expr: OperatorBuilder) -> Result<Self, OptimizerError> {
        // Add a projection when the alias contains column names.
        // Otherwise the names of the output columns are replaced by RewriteExprs
        // and we get the original names instead of aliases.
        let mut expr = if !alias.columns.is_empty() {
            let scope = expr
                .scope
                .as_ref()
                .map(Ok)
                .unwrap_or_else(|| Err(OptimizerError::internal("CTE: No scope")))?;

            let num_columns = alias.columns.len();

            let columns: Vec<_> = scope
                .columns()
                .iter()
                .take(num_columns)
                .enumerate()
                .map(|(i, (_, col_id))| {
                    let col_name = alias.columns[i].clone();
                    // if a column metadata has no expression then it is a reference to some column and
                    // we should and an alias in order to preserve its name.
                    let col_metadata = self.metadata.get_column(col_id);
                    match col_metadata.expr() {
                        Some(expr) => expr.clone(),
                        None => ScalarExpr::Column(*col_id).alias(col_name.as_str()),
                    }
                })
                .collect();

            expr.project(columns)?
        } else {
            expr
        };

        let (rel_node, scope) = expr.rel_node()?;
        let name = alias.name;

        let cte = CteInScope {
            expr: rel_node,
            relation: scope.output_relation().columns().to_vec(),
        };

        let mut scope = if let Some(scope) = self.scope.take() {
            scope
        } else {
            scope
        };

        if !scope.add_cte(name.clone(), cte) {
            return Err(OptimizerError::argument(format!("WITH query name '{}' specified more than once", name)));
        }
        self.scope = Some(scope);

        Ok(self)
    }

    /// Add a scan operator to an operator tree. A scan operator can only be added as a leaf node of a tree.
    ///
    /// Unlike [OperatorBuilder::get] this method adds an operator
    /// that returns all the columns from the given `source`.
    pub fn from(self, source: &str) -> Result<Self, OptimizerError> {
        self.add_scan_operator::<String>(source, None)
    }

    /// Adds a scan operator to an operator tree. A scan operator can only be added as leaf node of a tree.
    pub fn get(self, source: &str, columns: Vec<impl Into<String>>) -> Result<Self, OptimizerError> {
        self.add_scan_operator(source, Some(columns))
    }

    /// Adds an operator that returns the given list of rows.
    pub fn values(mut self, values: Vec<Vec<ScalarExpr>>) -> Result<Self, OptimizerError> {
        if values.is_empty() {
            return Err(OptimizerError::argument("VALUES list is empty"));
        }

        let value_list_length = values[0].len();
        let mut result_list = Vec::with_capacity(values.len());
        let mut columns = vec![];
        let relation_id = self.metadata.add_table("");

        for (i, value_list) in values.into_iter().enumerate() {
            let list_length = value_list.len();
            let tuple = ScalarExpr::Tuple(value_list);
            let metadata = self.metadata.clone();
            let validator = ValidateFilterExpr::where_clause();

            let value_list_expr = if let Some(scope) = self.scope.as_ref() {
                let mut rewriter = RewriteExprs::new(scope, metadata, validator);
                let tuple = tuple.rewrite(&mut rewriter)?;
                self.add_scalar_node(tuple, scope)?
            } else {
                let scope = OperatorScope::from_columns(vec![], vec![]);
                let mut rewriter = RewriteExprs::new(&scope, metadata, validator);
                let tuple = tuple.rewrite(&mut rewriter)?;
                self.add_scalar_node(tuple, &scope)?
            };

            if i == 0 {
                if let ScalarExpr::Tuple(values) = value_list_expr.expr() {
                    let columns_from_values: Result<Vec<(String, ColumnId)>, OptimizerError> = values
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            let col_type = resolve_expr_type(value, &self.metadata)?;
                            let col_name = format!("column{}", (i + 1));
                            let column =
                                ColumnMetadata::new_table_column(col_name.clone(), col_type, relation_id, "".into());
                            let col_id = self.metadata.add_column(column);

                            Ok((col_name, col_id))
                        })
                        .collect();
                    let columns_from_values = columns_from_values?;
                    columns.extend(columns_from_values.into_iter());
                }
            }

            result_list.push(value_list_expr);

            if list_length != value_list_length {
                return Err(OptimizerError::argument("VALUE lists must all have the same length"));
            }
        }

        let expr = LogicalExpr::Values(LogicalValues {
            values: result_list,
            columns: columns.iter().map(|(_, id)| *id).collect(),
        });
        let relation = RelationInScope::new(relation_id, "".into(), columns);

        self.add_input_operator(expr, relation);

        Ok(self)
    }

    /// Adds a select operator to an operator tree.
    pub fn select(mut self, filter: Option<ScalarExpr>) -> Result<Self, OptimizerError> {
        let (input, scope) = self.rel_node()?;
        let filter = if let Some(filter) = filter {
            let metadata = self.metadata.clone();
            let mut rewriter = RewriteExprs::new(&scope, metadata, ValidateFilterExpr::where_clause());
            let filter = filter.rewrite(&mut rewriter)?;
            let correlated = possibly_has_correlated_subqueries(&filter);

            let expr = self.add_scalar_node(filter, &scope)?;
            Some((expr, correlated))
        } else {
            None
        };

        let decorrelate_sub_queries = self.decorrelate_sub_queries;
        let mut builder = BuildLogicalExprs {
            scope: Some(scope),
            builder: &mut self,
        };

        let expr = match &filter {
            Some((filter, true)) if decorrelate_sub_queries => {
                decorrelate_subqueries(input.clone(), filter.clone(), &mut builder)?
            }
            _ => None,
        };
        let expr = match expr {
            Some(expr) => expr,
            None => builder.select(input, filter.map(|(f, _)| BuildFilter::Constructed(f)))?,
        };
        let scope = builder.take_scope();

        self.add_operator_and_scope(expr, scope);
        Ok(self)
    }

    /// Adds a projection operator to an operator tree.
    /// Use [aggregate_builder](Self::aggregate_builder) to add aggregate operator.
    ///
    /// # NOTE
    ///
    /// This method do not accept aggregate expressions but accepts window aggregate expressions
    /// with or without aggregate functions in their arguments.
    pub fn project(self, exprs: Vec<ScalarExpr>) -> Result<Self, OptimizerError> {
        self.projection_with_window_functions(exprs, false, true, false)
    }

    pub(crate) fn projection_with_window_functions(
        mut self,
        exprs: Vec<ScalarExpr>,
        allow_aggregates: bool,
        check_window_functions: bool,
        has_window_functions: bool,
    ) -> Result<Self, OptimizerError> {
        let (input, scope) = self.rel_node()?;
        let input_is_projection = matches!(input.expr().logical(), &LogicalExpr::Projection(_));
        let mut projection_builder = ProjectionListBuilder::new(&mut self, &scope, allow_aggregates);

        for expr in exprs {
            projection_builder.add_expr(expr)?;
        }

        let ProjectionList {
            column_ids,
            output_columns,
            projection: new_exprs,
        } = projection_builder.build();

        let add_window_functions = if check_window_functions {
            has_window_aggregates(new_exprs.iter().map(|expr| expr.expr()))
        } else {
            has_window_functions
        };

        let (input, scope, new_exprs) = if add_window_functions {
            add_window_aggregates(&mut self, input, scope, &column_ids, new_exprs)?
        } else {
            (input, scope, new_exprs)
        };

        let expr = LogicalExpr::Projection(LogicalProjection {
            input,
            exprs: new_exprs,
            columns: column_ids,
        });

        self.add_projection(expr, scope, output_columns, input_is_projection);
        Ok(self)
    }

    /// Adds a join operator to an operator tree.
    pub fn join_using(
        mut self,
        mut right: OperatorBuilder,
        join_type: JoinType,
        columns: Vec<(impl Into<String>, impl Into<String>)>,
    ) -> Result<Self, OptimizerError> {
        let (left, left_scope) = self.rel_node()?;
        let (right, right_scope) = right.rel_node()?;

        if join_type == JoinType::Cross {
            return Err(OptimizerError::argument("CROSS JOIN: USING (columns) condition is not supported"));
        }

        let mut columns_ids = Vec::new();

        for (l, r) in columns {
            let l: String = l.into();
            let r: String = r.into();

            let left_id = left_scope
                .columns()
                .iter()
                .find(|(name, _)| name == &l)
                .map(|(_, id)| *id)
                .ok_or_else(|| OptimizerError::argument("Join left side"))?;
            let right_id = right_scope
                .columns()
                .iter()
                .find(|(name, _)| name == &r)
                .map(|(_, id)| *id)
                .ok_or_else(|| OptimizerError::argument("Join right side"))?;
            columns_ids.push((left_id, right_id));
        }

        let condition = BuildJoinCondition::Using(JoinUsing::new(columns_ids));
        self.add_join(left, right, left_scope, right_scope, join_type, condition)?;

        Ok(self)
    }

    /// Adds a join operator with the given join type and ON <expr> condition to an operator tree.
    pub fn join_on(
        mut self,
        mut right: OperatorBuilder,
        join_type: JoinType,
        expr: ScalarExpr,
    ) -> Result<Self, OptimizerError> {
        let (left, left_scope) = self.rel_node()?;
        let (right, right_scope) = right.rel_node()?;

        if join_type == JoinType::Cross {
            match expr {
                ScalarExpr::Scalar(ScalarValue::Bool(Some(true))) => {}
                _ => {
                    return Err(OptimizerError::argument(format!(
                        "CROSS JOIN: Invalid expression in ON <expr> condition: {}",
                        expr
                    )))
                }
            }
        }

        let condition = BuildJoinCondition::On(expr);
        self.add_join(left, right, left_scope, right_scope, join_type, condition)?;

        Ok(self)
    }

    /// Adds a natural join operator to an operator tree.
    pub fn natural_join(mut self, mut right: OperatorBuilder, join_type: JoinType) -> Result<Self, OptimizerError> {
        let (left, left_scope) = self.rel_node()?;
        let (right, right_scope) = right.rel_node()?;

        let error = match join_type {
            JoinType::Cross => Some("CROSS JOIN"),
            JoinType::LeftSemi => Some("LEFT SEMI JOIN"),
            JoinType::RightSemi => Some("RIGHT SEMI JOIN"),
            JoinType::Anti => Some("ANTI JOIN"),
            _ => None,
        };
        if let Some(error) = error {
            return Err(OptimizerError::argument(format!("{}: Natural join condition is not allowed", error)));
        }

        let left_name_id: HashMap<String, ColumnId, RandomState> = HashMap::from_iter(left_scope.columns().to_vec());
        let mut column_ids = vec![];
        for (right_name, right_id) in right_scope.columns().iter() {
            if let Some(left_id) = left_name_id.get(right_name) {
                column_ids.push((*left_id, *right_id));
            }
        }

        let scope = left_scope.join(right_scope);
        let condition = if !column_ids.is_empty() {
            JoinCondition::Using(JoinUsing::new(column_ids))
        } else {
            let expr = self.add_scalar_node(scalar(true), &scope)?;
            JoinCondition::On(JoinOn::new(expr))
        };

        let expr = LogicalExpr::Join(LogicalJoin {
            join_type,
            left,
            right,
            condition,
        });

        self.add_operator_and_scope(expr, scope);
        Ok(self)
    }

    /// Set ordering requirements for the current node of an operator tree.
    pub fn order_by(mut self, ordering: impl Into<OrderingOptions>) -> Result<Self, OptimizerError> {
        match (self.operator.take(), self.scope.take()) {
            (Some(operator), Some(scope)) => {
                let OrderingOptions { options } = ordering.into();
                let mut ordering_columns = Vec::with_capacity(options.len());

                for option in options {
                    let OrderingOption { expr, descending } = option;

                    let metadata = self.metadata.clone();
                    let mut rewriter = RewriteExprs::new(&scope, metadata, ValidateProjectionExpr::projection_expr());
                    let expr = expr.rewrite(&mut rewriter)?;
                    let columns = scope.columns();

                    let column_id = match expr {
                        ScalarExpr::Scalar(ScalarValue::Int32(Some(pos))) => match usize::try_from(pos - 1).ok() {
                            Some(pos) if pos < columns.len() => {
                                let (_, id) = columns[pos];
                                id
                            }
                            Some(_) | None => {
                                return Err(OptimizerError::argument(format!(
                                    "ORDER BY position {} is not in select list",
                                    pos
                                )))
                            }
                        },
                        ScalarExpr::Column(id) => id,
                        _ => {
                            let expr_type = resolve_expr_type(&expr, &self.metadata)?;
                            let column_meta = ColumnMetadata::new_synthetic_column("".into(), expr_type, Some(expr));
                            self.metadata.add_column(column_meta)
                        }
                    };
                    ordering_columns.push(OrderingColumn::ord(column_id, descending));
                }
                // FIXME: Add null_last_{col_name}/null_first_{column} to support NULLS FIRST/LAST option
                let ordering = OrderingChoice::new(ordering_columns);
                let required = match operator.props().relational().physical().required.clone() {
                    Some(required) => required.with_ordering(ordering),
                    None => RequiredProperties::new_with_ordering(ordering),
                };
                let physical = PhysicalProperties::new_with_required(required);
                self.operator = Some(operator.with_physical(physical));
                self.scope = Some(scope);
                Ok(self)
            }
            _ => Err(OptimizerError::internal("No input operator")),
        }
    }

    /// Sets partitioning scheme required by the current node of an operator tree.
    pub fn partition_by(mut self, partitioning: impl Into<PartitioningOptions>) -> Result<Self, OptimizerError> {
        match (self.operator.take(), self.scope.take()) {
            (Some(operator), Some(scope)) => {
                let partitioning = partitioning.into();
                let partitioning = match partitioning {
                    PartitioningOptions::Singleton => Partitioning::Singleton,
                    PartitioningOptions::Partitioned(columns) => {
                        let columns: Result<Vec<_>, _> = columns
                            .into_iter()
                            .map(|col_name| {
                                let metadata = self.metadata.clone();
                                let mut rewriter =
                                    RewriteExprs::new(&scope, metadata, ValidateProjectionExpr::projection_expr());
                                let expr = ScalarExpr::ColumnName(col_name.clone());
                                let expr = expr.rewrite(&mut rewriter)?;
                                match expr {
                                    ScalarExpr::Column(id) => Ok(id),
                                    _ => Err(OptimizerError::argument(format!(
                                        "PARTITION BY unexpected column {}",
                                        col_name
                                    ))),
                                }
                            })
                            .collect();
                        Partitioning::Partitioned(columns?)
                    }
                };
                let partitioning = partitioning.normalize();
                let required = match operator.props().relational().physical().required.clone() {
                    Some(required) => required.with_partitioning(partitioning),
                    None => RequiredProperties::new_with_partitioning(partitioning),
                };
                let physical = PhysicalProperties::new_with_required(required);
                self.operator = Some(operator.with_physical(physical));
                self.scope = Some(scope);
                Ok(self)
            }
            _ => Err(OptimizerError::internal("No input operator")),
        }
    }

    /// Adds a union operator.
    pub fn union(self, right: OperatorBuilder) -> Result<Self, OptimizerError> {
        self.add_set_operator(SetOperator::Union, false, right)
    }

    /// Adds a union all operator.
    pub fn union_all(self, right: OperatorBuilder) -> Result<Self, OptimizerError> {
        self.add_set_operator(SetOperator::Union, true, right)
    }

    /// Adds an except operator.
    pub fn except(self, right: OperatorBuilder) -> Result<Self, OptimizerError> {
        self.add_set_operator(SetOperator::Except, false, right)
    }

    /// Adds an except all operator.
    pub fn except_all(self, right: OperatorBuilder) -> Result<Self, OptimizerError> {
        self.add_set_operator(SetOperator::Except, true, right)
    }

    /// Adds an intersect operator.
    pub fn intersect(self, right: OperatorBuilder) -> Result<Self, OptimizerError> {
        self.add_set_operator(SetOperator::Intersect, false, right)
    }

    /// Adds an intersect all operator.
    pub fn intersect_all(self, right: OperatorBuilder) -> Result<Self, OptimizerError> {
        self.add_set_operator(SetOperator::Intersect, true, right)
    }

    /// Adds a relation that produces no rows.
    /// The `return_one_row` flag with value `true` can be used to support queries like `SELECT 1`
    /// where projection has no input operator.
    ///
    /// This operator can only be added as a leaf node of an operator tree
    /// (the first operator build created by an operator builder).
    pub fn empty(mut self, return_one_row: bool) -> Result<Self, OptimizerError> {
        if self.operator.is_some() {
            return Err(OptimizerError::argument(
                "Empty relation can not be added on top of another operator".to_string(),
            ));
        };
        let expr = LogicalExpr::Empty(LogicalEmpty { return_one_row });
        self.add_input_operator(expr, RelationInScope::from_columns(vec![]));
        Ok(self)
    }

    /// Adds a distinct operator.
    pub fn distinct(mut self, on_expr: Option<ScalarExpr>) -> Result<Self, OptimizerError> {
        let (input, input_scope) = self.rel_node()?;
        let on_expr = if let Some(on_expr) = on_expr {
            let metadata = self.metadata.clone();
            let mut rewriter = RewriteExprs::new(
                &input_scope,
                metadata,
                ValidateFilterExpr {
                    allow_aggregates: false,
                },
            );
            let on_expr = on_expr.rewrite(&mut rewriter)?;
            let expr = self.add_scalar_node(on_expr, &input_scope)?;
            Some(expr)
        } else {
            None
        };

        let columns = input_scope.columns().iter().map(|(_, id)| *id).collect();
        let expr = LogicalExpr::Distinct(LogicalDistinct {
            input,
            on_expr,
            columns,
        });
        self.add_operator_and_scope(expr, input_scope);
        Ok(self)
    }

    /// Adds a limit operator.
    pub fn limit(mut self, rows: usize) -> Result<Self, OptimizerError> {
        let (input, input_scope) = self.rel_node()?;
        let expr = LogicalExpr::Limit(LogicalLimit { input, rows });
        self.add_operator_and_scope(expr, input_scope);
        Ok(self)
    }

    /// Adds an offset operator.
    pub fn offset(mut self, rows: usize) -> Result<Self, OptimizerError> {
        let (input, input_scope) = self.rel_node()?;
        let expr = LogicalExpr::Offset(LogicalOffset { input, rows });
        self.add_operator_and_scope(expr, input_scope);
        Ok(self)
    }

    /// Returns a builder to construct an aggregate operator that consists of a list of
    /// non-window aggregate functions, group by expressions, and a filter (HAVING clause).
    pub fn aggregate_builder(&mut self) -> AggregateBuilder {
        AggregateBuilder {
            builder: self,
            aggr_exprs: vec![],
            group_exprs: vec![],
            having: None,
        }
    }

    /// Creates a builder that can be used to construct subqueries that
    /// can reference tables, columns and other objects  
    /// accessible to the current topmost operator in the operator tree.
    pub fn sub_query_builder(&self) -> Self {
        OperatorBuilder {
            callback: self.callback.clone(),
            catalog: self.catalog.clone(),
            metadata: self.metadata.clone(),
            scope: self.scope.as_ref().map(|scope| scope.new_child_scope(self.metadata.get_ref())),
            operator: None,
            decorrelate_sub_queries: self.decorrelate_sub_queries,
        }
    }

    /// Creates a builder that shares the metadata with this builder and
    /// reuses outer columns from the current scope.
    /// Unlike [sub_query_builder](Self::sub_query_builder) this method should be used
    /// to operator trees that will be joined with this operator.
    pub fn new_relation_builder(&self) -> Self {
        OperatorBuilder {
            callback: self.callback.clone(),
            catalog: self.catalog.clone(),
            metadata: self.metadata.clone(),
            scope: self.scope.as_ref().map(|scope| scope.new_scope(self.metadata.get_ref())),
            operator: None,
            decorrelate_sub_queries: self.decorrelate_sub_queries,
        }
    }

    /// Creates a builder that shares the metadata with this one.
    /// Unlike [sub_query_builder](Self::sub_query_builder) this method should be used
    /// to build independent operator trees.
    pub fn new_query_builder(&self) -> Self {
        OperatorBuilder {
            callback: self.callback.clone(),
            catalog: self.catalog.clone(),
            metadata: self.metadata.clone(),
            scope: None,
            operator: None,
            decorrelate_sub_queries: self.decorrelate_sub_queries,
        }
    }

    /// Creates an operator tree and returns its metadata.
    /// If this builder has been created via call to [Self::sub_query_builder()] this method returns an error.
    pub fn build(self) -> Result<Operator, OptimizerError> {
        match (self.operator, self.scope) {
            (Some(operator), Some(scope)) => {
                validate_rel_node(&operator, &self.metadata)?;
                let rel_node = self.callback.new_rel_expr(operator, &scope)?;
                Ok(rel_node.into_inner())
            }
            (None, _) => Err(OptimizerError::internal("Build: No operator")),
            (_, _) => Err(OptimizerError::internal("Build: No scope")),
        }
    }

    /// Sets an alias to the current operator. If there is no operator returns an error.
    pub fn with_alias(mut self, alias: TableAlias) -> Result<Self, OptimizerError> {
        if let Some(scope) = self.scope.as_mut() {
            let output_relation = scope.output_relation();
            let output_columns = output_relation.columns();

            for (i, column) in alias.columns.iter().enumerate() {
                let id = output_columns[i].1;
                self.metadata.rename_column(id, column.clone());
            }

            scope.set_alias(alias, &self.metadata)?;
            Ok(self)
        } else {
            Err(OptimizerError::argument("ALIAS: no operator"))
        }
    }

    /// Set the [presentation](Presentation) physical property of the the current node of an operator tree.
    pub fn with_presentation(mut self) -> Result<Self, OptimizerError> {
        match (self.operator.take(), self.scope.as_ref()) {
            (Some(operator), Some(scope)) => {
                let presentation = Presentation {
                    columns: scope.columns().to_vec(),
                };
                let mut physical = operator.props().relational().physical().clone();
                physical.presentation = Some(presentation);

                self.operator = Some(operator.with_physical(physical));
                Ok(self)
            }
            (None, _) => Err(OptimizerError::internal("Presentation: No operator")),
            (_, _) => Err(OptimizerError::internal("Presentation: No scope")),
        }
    }

    fn add_scan_operator<T>(mut self, source: &str, columns: Option<Vec<T>>) -> Result<Self, OptimizerError>
    where
        T: Into<String>,
    {
        if self.operator.is_some() {
            return Err(OptimizerError::internal(
                "Adding a scan operator on top of another operator is not allowed".to_string(),
            ));
        }

        if let Some((input_scope, cte)) =
            self.scope.as_ref().and_then(|scope| scope.find_cte(source).map(|cte| (scope, cte)))
        {
            let column_ids = cte.expr.props().logical().output_columns();
            let columns: Result<Vec<_>, OptimizerError> = match columns {
                Some(columns) => columns
                    .into_iter()
                    .map(|name| name.into())
                    .zip(column_ids.iter())
                    .map(|(name, _)| {
                        let mut rewriter = RewriteExprs::new(
                            input_scope,
                            self.metadata.clone(),
                            ValidateProjectionExpr::projection_expr(),
                        );
                        let expr = ScalarExpr::ColumnName(name.clone());
                        let expr = expr.rewrite(&mut rewriter)?;
                        if let ScalarExpr::Column(id) = expr {
                            Ok((name, id))
                        } else {
                            unreachable!()
                        }
                    })
                    .collect(),
                None => Ok(cte.relation.clone()),
            };
            let columns = columns?;
            let operator = Operator::from(OperatorExpr::from(cte.expr.expr().logical().clone()));

            // Set new scope which includes:
            //  + outer columns from the input scope
            //  + output columns returned by CTE.
            //  + relations from the input scope (technically, the relations from the outer scope)
            let outer_columns = input_scope.outer_columns().to_vec();
            let mut scope = OperatorScope::from_columns(columns, outer_columns);
            scope.add_relations(input_scope.clone());

            self.operator = Some(operator);
            self.scope = Some(scope);

            return Ok(self);
        }

        let table = self
            .catalog
            .get_table(source)
            .ok_or_else(|| OptimizerError::argument(format!("Table does not exist. Table: {}", source)))?;

        let columns: Vec<_> = match columns {
            None => table.columns().iter().map(|c| c.name().to_string()).collect(),
            Some(columns) => columns.into_iter().map(|s| s.into()).collect(),
        };

        let relation_name = String::from(source);
        let relation_id = self.metadata.add_table(relation_name.as_str());
        //TODO: use a single method to add relation and its columns.
        let columns: Result<Vec<(String, ColumnId)>, OptimizerError> = columns
            .iter()
            .map(|name| {
                table.get_column(name).ok_or_else(|| {
                    OptimizerError::argument(format!("Column does not exist. Column: {}. Table: {}", name, source))
                })
            })
            .map_ok(|column| {
                let column_name = column.name().to_string();
                let metadata = ColumnMetadata::new_table_column(
                    column_name.clone(),
                    column.data_type().clone(),
                    relation_id,
                    source.into(),
                );
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

        let relation = RelationInScope::new(relation_id, relation_name, columns);
        self.add_input_operator(expr, relation);
        Ok(self)
    }

    fn add_set_operator(
        mut self,
        set_op: SetOperator,
        all: bool,
        mut right: OperatorBuilder,
    ) -> Result<Self, OptimizerError> {
        let (left, left_scope) = self.rel_node()?;
        let (right, right_scope) = right.rel_node()?;

        if left_scope.columns().len() != right_scope.columns().len() {
            return Err(OptimizerError::argument(format!("{}: Number of columns does not match", set_op)));
        }

        let outer_columns = left_scope.outer_columns().to_vec();

        let mut columns = Vec::new();
        let mut column_ids = Vec::new();
        let left_columns = left_scope.into_columns();
        let right_columns = right_scope.into_columns();
        for (i, (l, r)) in left_columns.iter().zip(right_columns.iter()).enumerate() {
            let data_type = {
                let l = self.metadata.get_column(&l.1);
                let r = self.metadata.get_column(&r.1);
                let left_col_type = l.data_type();
                let right_col_type = r.data_type();

                if left_col_type != right_col_type {
                    let message = format!(
                        "{}: Data type of {}-th column does not match. Left: {}, right: {}",
                        set_op, i, left_col_type, right_col_type
                    );
                    return Err(OptimizerError::argument(message));
                }
                left_col_type.clone()
            };

            // use column names of the first operand.
            let column_name = l.0.clone();
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

        let operator = Operator::from(OperatorExpr::from(expr));
        self.operator = Some(operator);

        let relation = RelationInScope::from_columns(columns);
        let mut scope = if let Some(mut scope) = self.scope.take() {
            scope.set_relation(relation);
            scope
        } else {
            OperatorScope::new_source(relation)
        };
        scope.set_outer_columns(outer_columns);
        self.scope = Some(scope);

        Ok(self)
    }

    fn add_projection(
        &mut self,
        expr: LogicalExpr,
        input: OperatorScope,
        columns: Vec<(String, ColumnId)>,
        input_is_projection: bool,
    ) {
        let operator = Operator::from(OperatorExpr::from(expr));
        self.operator = Some(operator);

        let outer_columns = input.outer_columns().to_vec();
        let mut scope = OperatorScope::from_columns(columns, outer_columns);
        if !input_is_projection {
            scope.add_relations(input);
        }
        self.scope = Some(scope);
    }

    fn add_join(
        &mut self,
        left: RelNode,
        right: RelNode,
        left_scope: OperatorScope,
        right_scope: OperatorScope,
        join_type: JoinType,
        condition: BuildJoinCondition,
    ) -> Result<(), OptimizerError> {
        fn build_join_expr_and_scope(
            builder: &mut OperatorBuilder,
            expr_scope: &OperatorScope,
            left: RelNode,
            right: RelNode,
            join_type: JoinType,
            condition: BuildJoinCondition,
        ) -> Result<LogicalExpr, OptimizerError> {
            let metadata = builder.metadata.clone();
            let condition = match condition {
                BuildJoinCondition::Using(using) => JoinCondition::Using(using),
                BuildJoinCondition::On(expr) => {
                    let mut rewriter = RewriteExprs::new(expr_scope, metadata, ValidateFilterExpr::join_clause());
                    let expr = expr.rewrite(&mut rewriter)?;
                    let expr = builder.add_scalar_node(expr, expr_scope)?;
                    JoinCondition::On(JoinOn::new(expr))
                }
            };
            let expr = LogicalExpr::Join(LogicalJoin {
                join_type,
                left,
                right,
                condition,
            });

            Ok(expr)
        }

        let (expr, scope) = match join_type {
            JoinType::LeftSemi | JoinType::Anti => {
                let scope = left_scope.clone().join(right_scope);
                let expr = build_join_expr_and_scope(self, &scope, left, right, join_type, condition)?;
                // LeftSemi and anti join return only left columns
                (expr, left_scope)
            }
            JoinType::RightSemi => {
                let scope = right_scope.clone().join(left_scope);
                let expr = build_join_expr_and_scope(self, &scope, left, right, join_type, condition)?;
                // RightSemi join return only right columns
                (expr, right_scope)
            }
            _ => {
                let scope = left_scope.join(right_scope);
                let expr = build_join_expr_and_scope(self, &scope, left, right, join_type, condition)?;
                (expr, scope)
            }
        };

        self.add_operator_and_scope(expr, scope);
        Ok(())
    }

    fn add_input_operator(&mut self, expr: LogicalExpr, relation: RelationInScope) {
        let operator = Operator::from(OperatorExpr::from(expr));
        self.operator = Some(operator);

        if let Some(mut scope) = self.scope.take() {
            scope.set_relation(relation);
            self.scope = Some(scope);
        } else {
            self.scope = Some(OperatorScope::new_source(relation));
        }
    }

    fn add_operator_and_scope<T>(&mut self, expr: T, scope: OperatorScope)
    where
        T: Into<Operator>,
    {
        let operator = expr.into();
        self.operator = Some(operator);
        self.scope = Some(scope);
    }

    fn rel_node(&mut self) -> Result<(RelNode, OperatorScope), OptimizerError> {
        let operator = self.operator.take().ok_or_else(|| OptimizerError::internal("No input operator"))?;

        validate_rel_node(&operator, &self.metadata)?;

        let scope = self.scope.take().ok_or_else(|| OptimizerError::internal("No scope"))?;
        let rel_node = self.callback.new_rel_expr(operator, &scope)?;
        Ok((rel_node, scope))
    }

    fn add_scalar_node(&self, expr: ScalarExpr, scope: &OperatorScope) -> Result<ScalarNode, OptimizerError> {
        let operator = Operator::new(OperatorExpr::from(expr), Properties::Scalar(ScalarProperties::default()));
        self.callback.new_scalar_expr(operator, scope)
    }
}

fn validate_rel_node(operator: &Operator, metadata: &OperatorMetadata) -> Result<(), OptimizerError> {
    // Move validation logic somewhere else.
    if let OperatorExpr::Relational(rel_expr) = operator.expr() {
        if let LogicalExpr::Values(values) = rel_expr.logical() {
            let col_id = values.columns[0];
            let column = metadata.get_column(&col_id);
            let relation_id = column.relation_id().expect("VALUES column without relation_id");
            let relation = metadata.get_relation(&relation_id);
            if relation.name().is_empty() {
                return Err(OptimizerError::argument("VALUES in FROM must have an alias".to_string()));
            }
        }
    }
    Ok(())
}

impl Debug for OperatorBuilder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorBuilder")
            .field("catalog", &self.catalog)
            .field("metadata", self.metadata.as_ref())
            .field("operator", &self.operator)
            .field("scope", &self.scope)
            .finish()
    }
}

// TODO: Remove alias is post_rewrite
struct RewriteExprs<'a, T> {
    scope: &'a OperatorScope,
    validator: ExprValidator<T>,
    metadata: Rc<MutableMetadata>,
    referenced_outer_columns: Vec<ColumnId>,
}

impl<'a, T> RewriteExprs<'a, T>
where
    T: ValidateExpr,
{
    fn new(scope: &'a OperatorScope, metadata: Rc<MutableMetadata>, validator: T) -> Self {
        RewriteExprs {
            scope,
            metadata,
            validator: ExprValidator {
                aggregate_depth: 0,
                window_func_depth: 0,
                alias_depth: 0,
                expr_index: 0,
                validator,
            },
            referenced_outer_columns: Vec::new(),
        }
    }
}

impl<T> ExprRewriter<RelNode> for RewriteExprs<'_, T>
where
    T: ValidateExpr,
{
    type Error = OptimizerError;

    fn pre_rewrite(&mut self, expr: &ScalarExpr) -> Result<bool, Self::Error> {
        self.validator.before_expr(expr)?;
        Ok(true)
    }

    fn rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
        fn unexpected_column(col_name: &str, scope: &OperatorScope, metadata: MetadataRef) -> OptimizerError {
            let outer_columns: Vec<_> = scope
                .outer_columns()
                .iter()
                .map(|id| {
                    let col_name = metadata.get_column(id);
                    (col_name.name().to_string(), *id)
                })
                .collect();

            OptimizerError::argument(format!(
                "Unexpected column: {}. Input columns: {}, Outer columns: {}",
                col_name,
                DisplayColumns(scope.columns()),
                DisplayColumns(&outer_columns),
            ))
        }

        let expr = match expr {
            ScalarExpr::Column(column_id) => {
                let exists = self.scope.find_column_by_id(column_id).is_some();
                if !exists {
                    let col_name = format!("<{}>", column_id);
                    Err(unexpected_column(col_name.as_str(), self.scope, self.metadata.get_ref()))
                } else {
                    Ok(expr)
                }
            }
            ScalarExpr::ColumnName(ref column_name) => {
                let column_id = self.scope.find_column_by_name(column_name);

                match column_id {
                    Some(column_id) => Ok(ScalarExpr::Column(column_id)),
                    None => Err(unexpected_column(column_name, self.scope, self.metadata.get_ref())),
                }
            }
            _ => {
                if let Some(query) = get_subquery(&expr) {
                    let output_columns = query.props().logical().output_columns();
                    if output_columns.len() != 1 {
                        return Err(OptimizerError::argument("Subquery must return exactly one column"));
                    }

                    match query.state() {
                        MemoExprState::Owned(_) => {
                            //FIXME: Add method to handle nested relational expressions to OperatorCallback?
                            Err(OptimizerError::internal(
                                "Use OperatorBuilder::sub_query_builder to build a nested sub query",
                            ))
                        }
                        MemoExprState::Memo(_) => Ok(expr),
                    }
                } else {
                    Ok(expr)
                }
            }
        }?;
        self.validator.validate(&expr)?;
        Ok(expr)
    }

    fn post_rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
        if let ScalarExpr::Column(id) = &expr {
            if self.scope.outer_columns().contains(id) {
                self.referenced_outer_columns.push(*id);
            }
        }
        self.validator.after_expr(&expr)?;
        Ok(expr)
    }
}

enum BuildJoinCondition {
    Using(JoinUsing),
    On(ScalarExpr),
}

struct DisplayColumns<'a>(&'a [(String, ColumnId)]);

impl Display for DisplayColumns<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.0.iter().map(|(name, id)| format!("({}, {})", name, id)).join(", "))
    }
}

struct ExprValidator<T> {
    aggregate_depth: usize,
    window_func_depth: usize,
    alias_depth: usize,
    expr_index: usize,
    validator: T,
}

impl<T> ExprValidator<T>
where
    T: ValidateExpr,
{
    fn before_expr(&mut self, expr: &ScalarExpr) -> Result<(), OptimizerError> {
        self.expr_index += 1;
        match expr {
            ScalarExpr::Alias(_, _) => self.alias_depth += 1,
            ScalarExpr::Aggregate { .. } => self.aggregate_depth += 1,
            ScalarExpr::WindowAggregate { .. } => {
                let is_root = self.expr_index - self.alias_depth == 1;
                not_implemented!(!is_root, "WINDOW FUNCTIONS: must be a root of an expression tree: {}", expr);
                self.window_func_depth += 1
            }
            // Do not allow ScalarValues of type Array and Tuple both of those are created
            // via ScalarExprs of the same type.
            ScalarExpr::Scalar(ScalarValue::Array(_, _)) => {
                not_implemented!("ARRAYS: scalar value of type array")
            }
            ScalarExpr::Scalar(ScalarValue::Tuple(_, _)) => {
                not_implemented!("TUPLES: scalar value of type tuple")
            }
            _ => {}
        }
        if self.aggregate_depth > 1 {
            // query error
            return Err(OptimizerError::argument("Nested aggregate functions are not allowed".to_string()));
        }
        if self.alias_depth > 1 {
            // query error
            return Err(OptimizerError::argument("Nested alias expressions are not allowed".to_string()));
        }
        if self.window_func_depth > 1 {
            // query error
            return Err(OptimizerError::argument("Nested window functions are not allowed".to_string()));
        }
        self.validator.pre_validate(expr)?;
        Ok(())
    }

    fn validate(&mut self, expr: &ScalarExpr) -> Result<(), OptimizerError> {
        self.validator.validate(expr)?;
        Ok(())
    }

    fn after_expr(&mut self, expr: &ScalarExpr) -> Result<(), OptimizerError> {
        match expr {
            ScalarExpr::Alias(_, _) => self.alias_depth -= 1,
            ScalarExpr::Aggregate { .. } => self.aggregate_depth -= 1,
            ScalarExpr::WindowAggregate {
                func,
                args,
                partition_by,
                ..
            } => {
                let non_column_args = args.iter().any(|expr| {
                    !matches!(expr, ScalarExpr::Column(_) | ScalarExpr::Scalar(_) | ScalarExpr::Aggregate { .. })
                });
                not_implemented!(non_column_args, "WINDOW FUNCTION: non column arguments in function: {}", func);

                let non_column_exprs = partition_by.iter().any(|expr| !matches!(expr, ScalarExpr::Column(_)));
                not_implemented!(
                    non_column_exprs,
                    "WINDOW FUNCTION: non column expressions are allowed in PARTITION BY clause: {}",
                    expr
                );

                self.window_func_depth -= 1;
            }
            _ => {
                self.validate_array_expr(expr)?;
            }
        }
        Ok(())
    }

    fn validate_array_expr(&self, expr: &ScalarExpr) -> Result<(), OptimizerError> {
        #[derive(Debug, Eq, PartialEq, Clone)]
        enum ArrayDimensions {
            Value,
            Array {
                size: usize,
                element_type: Box<ArrayDimensions>,
            },
        }

        fn check_type(expr: &ScalarExpr) -> Result<ArrayDimensions, OptimizerError> {
            if let ScalarExpr::Array(elements) = expr {
                let mut first_element_type = None;

                for element in elements.iter() {
                    if let ScalarExpr::Scalar(ScalarValue::Null) = element {
                        // Ignore NULLs because NULL can be both an array and a value.
                    } else {
                        match &first_element_type {
                            None => {
                                first_element_type = Some(check_type(element)?);
                            }
                            Some(element_type) => {
                                let current_type = check_type(element)?;
                                if element_type != &current_type {
                                    return Err(OptimizerError::argument(
                                        "Multidimensional arrays must have array expressions with matching dimensions",
                                    ));
                                }
                            }
                        }
                    }
                }

                match first_element_type {
                    None => {
                        not_supported!("Empty array expression");
                    }
                    Some(element_type) => Ok(ArrayDimensions::Array {
                        size: elements.len(),
                        element_type: Box::new(element_type),
                    }),
                }
            } else {
                Ok(ArrayDimensions::Value)
            }
        }

        check_type(expr)?;
        Ok(())
    }
}

trait ValidateExpr {
    fn pre_validate(&mut self, _expr: &ScalarExpr) -> Result<(), OptimizerError> {
        Ok(())
    }

    fn validate(&mut self, expr: &ScalarExpr) -> Result<(), OptimizerError>;
}

struct ValidateFilterExpr {
    allow_aggregates: bool,
}

impl ValidateFilterExpr {
    fn where_clause() -> Self {
        ValidateFilterExpr {
            allow_aggregates: false,
        }
    }

    fn join_clause() -> Self {
        ValidateFilterExpr {
            allow_aggregates: false,
        }
    }
}

impl ValidateExpr for ValidateFilterExpr {
    fn validate(&mut self, expr: &ScalarExpr) -> Result<(), OptimizerError> {
        match expr {
            ScalarExpr::Alias(_, _) => {
                // query error
                Err(OptimizerError::argument("aliases are not allowed in filter expressions".to_string()))
            }
            ScalarExpr::Aggregate { .. } if !self.allow_aggregates => {
                // query error
                //TODO: Include clause (WHERE, JOIN etc)
                Err(OptimizerError::argument("aggregates are not allowed".to_string()))
            }
            _ => Ok(()),
        }
    }
}

struct ValidateProjectionExpr {}

impl ValidateProjectionExpr {
    fn projection_expr() -> Self {
        ValidateProjectionExpr {}
    }
}

impl ValidateExpr for ValidateProjectionExpr {
    fn validate(&mut self, _expr: &ScalarExpr) -> Result<(), OptimizerError> {
        Ok(())
    }
}

/// [OperatorCallback] that does nothing.
#[derive(Debug)]
pub struct NoOpOperatorCallback;

impl OperatorCallback for NoOpOperatorCallback {
    fn new_rel_expr(&self, expr: Operator, _scope: &OperatorScope) -> Result<RelNode, OptimizerError> {
        RelNode::try_from(expr)
    }

    fn new_scalar_expr(&self, expr: Operator, _scope: &OperatorScope) -> Result<ScalarNode, OptimizerError> {
        ScalarNode::try_from(expr)
    }
}

/// A helper that to populate a memo data structure.
///
/// Usage:
/// ```text
///   let memo = ...;
///   let memoization = MemoizeOperators::new(memo);
///   let operator_callback = memoization.take_callback();
///
///   let operator_builder = OperatorBuilder::new(operator_callback,
///     // provide other dependencies
///     ...
///   );
///
///   // .. add some nodes to the operator tree.
///   
///   // operator_builder consumes its reference the `operator_callback`.
///   let operator = operator_builder.build()?;
///
///   // consume the helper and return the underlying memo.
///   let memo = memoization.into_memo();
///
///   // We can now use both the operator and the memo.
/// ```
#[derive(Debug)]
pub struct MemoizeOperators {
    inner: Rc<OperatorCallbackImpl>,
    returned_callback: bool,
}

#[derive(Debug)]
struct OperatorCallbackImpl {
    // RefCell is necessary because the memo is shared by multiple operator builders.
    memo: RefCell<ExprMemo>,
}

impl MemoizeOperators {
    /// Creates a helper that uses the given memo.
    pub fn new(memo: ExprMemo) -> Self {
        let callback = OperatorCallbackImpl {
            memo: RefCell::new(memo),
        };
        MemoizeOperators {
            inner: Rc::new(callback),
            returned_callback: false,
        }
    }

    /// Returns a callback that will be called when a new operator is added to an operator tree.
    /// All operators are copied to a [memo](crate::memo::Memo).
    ///
    /// # Panics
    ///
    /// Only a single callback can exist at a time. This method panics if it is called more than once.
    pub fn take_callback(&mut self) -> Rc<dyn OperatorCallback> {
        assert!(!self.returned_callback, "callback has already been taken");

        self.returned_callback = true;
        self.inner.clone()
    }

    /// Consumes this helper and returns the underlying memo.
    ///
    /// # Panics
    ///
    /// This method panics if other references to a callback returned by [take_callback](Self::take_callback) still exists.
    pub fn into_memo(self) -> ExprMemo {
        let inner =
            Rc::try_unwrap(self.inner).expect("References to the callback still exist outside of MemoizeOperators");
        inner.memo.into_inner()
    }
}

impl OperatorCallback for OperatorCallbackImpl {
    fn new_rel_expr(&self, expr: Operator, scope: &OperatorScope) -> Result<RelNode, OptimizerError> {
        let mut memo = self.memo.borrow_mut();
        let scope = OuterScope {
            outer_columns: scope.outer_columns().to_vec(),
        };
        let expr = memo.insert_group(expr, &scope)?;
        RelNode::try_from(expr)
    }

    fn new_scalar_expr(&self, expr: Operator, scope: &OperatorScope) -> Result<ScalarNode, OptimizerError> {
        let mut memo = self.memo.borrow_mut();
        let scope = OuterScope {
            outer_columns: scope.outer_columns().to_vec(),
        };
        let expr = memo.insert_group(expr, &scope)?;
        ScalarNode::try_from(expr)
    }
}

struct BuildLogicalExprs<'a> {
    pub scope: Option<OperatorScope>,
    pub builder: &'a mut OperatorBuilder,
}

impl BuildLogicalExprs<'_> {
    fn add_scalar_expr(&self, expr: BuildFilter, scope: Option<&OperatorScope>) -> Result<ScalarNode, OptimizerError> {
        match expr {
            BuildFilter::New(expr) => {
                if let Some(scope) = scope {
                    self.builder.add_scalar_node(expr, scope)
                } else {
                    Err(OptimizerError::internal("No scope"))
                }
            }
            BuildFilter::Constructed(expr) => Ok(expr),
        }
    }

    fn take_scope(self) -> OperatorScope {
        self.scope.expect("No scope")
    }
}

impl<'a> BuildOperators for BuildLogicalExprs<'a> {
    fn metadata(&self) -> MetadataRef {
        self.builder.metadata.get_ref()
    }

    fn add_operator(&mut self, expr: Operator) -> Result<RelNode, OptimizerError> {
        let scope = self.scope.take().expect("No scope");
        self.builder.add_operator_and_scope(expr, scope);
        let (node, scope) = self.builder.rel_node()?;
        self.scope = Some(scope);
        Ok(node)
    }

    fn select(&mut self, input: RelNode, filter: Option<BuildFilter>) -> Result<Operator, OptimizerError> {
        let filter = match filter {
            Some(filter) => Some(self.add_scalar_expr(filter, self.scope.as_ref())?),
            None => None,
        };
        let select = LogicalExpr::Select(LogicalSelect { input, filter });
        Ok(select.into())
    }

    fn left_semi_join(
        &mut self,
        left: RelNode,
        right: RelNode,
        condition: BuildFilter,
    ) -> Result<Operator, OptimizerError> {
        let condition = self.add_scalar_expr(condition, self.scope.as_ref())?;
        let join = LogicalExpr::Join(LogicalJoin {
            join_type: JoinType::LeftSemi,
            left,
            right,
            condition: JoinCondition::On(JoinOn::new(condition)),
        });

        Ok(join.into())
    }

    fn left_join(&mut self, left: RelNode, right: RelNode, condition: BuildFilter) -> Result<Operator, OptimizerError> {
        let condition = self.add_scalar_expr(condition, self.scope.as_ref())?;

        let join = LogicalExpr::Join(LogicalJoin {
            join_type: JoinType::Left,
            left,
            right,
            condition: JoinCondition::On(JoinOn::new(condition)),
        });

        Ok(join.into())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::catalog::mutable::MutableCatalog;
    use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
    use crate::datatypes::DataType;
    use crate::operators::format::{AppendMemo, AppendMetadata, OperatorTreeFormatter, SubQueriesFormatter};
    use crate::operators::properties::LogicalPropertiesBuilder;
    use crate::operators::scalar::aggregates::{AggregateFunction, WindowFunction};
    use crate::operators::scalar::value::{Array, Scalar};
    use crate::operators::scalar::{col, qualified_wildcard, scalar, wildcard};
    use crate::operators::{OperatorMemoBuilder, Properties, RelationalProperties};
    use crate::properties::logical::LogicalProperties;
    use crate::statistics::NoStatisticsBuilder;

    use super::*;

    #[test]
    fn test_from() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| builder.from("A")?.build());

        tester.expect_expr(
            r#"
LogicalGet A cols=[1, 2, 3, 4]
  output cols: [1, 2, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 A.a4 Int32
Memo:
  00 LogicalGet A cols=[1, 2, 3, 4]
"#,
        );
    }

    #[test]
    fn test_from_alias_rename_columns() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            builder
                .from("A")?
                .with_alias(TableAlias {
                    name: "X".to_owned(),
                    columns: vec!["x".to_owned(), "y".to_owned()],
                })?
                .project(vec![col("y"), col("x")])?
                .build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[2, 1] exprs: [col:2, col:1]
  input: LogicalGet A cols=[1, 2, 3, 4]
  output cols: [2, 1]
Metadata:
  col:1 A.x Int32
  col:2 A.y Int32
  col:3 A.a3 Int32
  col:4 A.a4 Int32
Memo:
  03 LogicalProjection input=00 exprs=[01, 02] cols=[2, 1]
  02 Expr col:1
  01 Expr col:2
  00 LogicalGet A cols=[1, 2, 3, 4]
"#,
        );
    }

    #[test]
    fn test_from_alias_rename_columns_does_not_introduce_new_columns() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            builder
                .from("A")?
                .with_alias(TableAlias {
                    name: "X".to_owned(),
                    columns: vec!["x".to_owned(), "y".to_owned()],
                })?
                .project(vec![wildcard()])?
                .build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[1, 2, 3, 4] exprs: [col:1, col:2, col:3, col:4]
  input: LogicalGet A cols=[1, 2, 3, 4]
  output cols: [1, 2, 3, 4]
Metadata:
  col:1 A.x Int32
  col:2 A.y Int32
  col:3 A.a3 Int32
  col:4 A.a4 Int32
Memo:
  05 LogicalProjection input=00 exprs=[01, 02, 03, 04] cols=[1, 2, 3, 4]
  04 Expr col:4
  03 Expr col:3
  02 Expr col:2
  01 Expr col:1
  00 LogicalGet A cols=[1, 2, 3, 4]
"#,
        );
    }

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
    fn test_get_order_by_column_name() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a1"])?;
            let ord = OrderingOption::by("a1", false);

            from_a.order_by(ord)?.build()
        });

        tester.expect_expr(
            r#"
LogicalGet A cols=[1] ordering=[+1]
  output cols: [1]
Metadata:
  col:1 A.a1 Int32
Memo:
  00 LogicalGet A cols=[1]
"#,
        );
    }

    #[test]
    fn test_get_order_by_alias() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a1", "a2"])?;
            let projection = from_a.project(vec![col("a1").alias("x"), col("a2")])?;

            let ord = OrderingOption::by("x", false);
            projection.order_by(ord)?.build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[3, 2] ordering=[+3] exprs: [col:1 AS x, col:2]
  input: LogicalGet A cols=[1, 2]
  output cols: [3, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 x Int32, expr: col:1
Memo:
  03 LogicalProjection input=00 exprs=[01, 02] cols=[3, 2]
  02 Expr col:2
  01 Expr col:1 AS x
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_get_order_by_position() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a2", "a1", "a3"])?;
            let ord = OrderingOption::by_position(2, false);

            from_a.order_by(ord)?.build()
        });

        tester.expect_expr(
            r#"
LogicalGet A cols=[1, 2, 3] ordering=[+2]
  output cols: [1, 2, 3]
Metadata:
  col:1 A.a2 Int32
  col:2 A.a1 Int32
  col:3 A.a3 Int32
Memo:
  00 LogicalGet A cols=[1, 2, 3]
"#,
        );
    }

    #[test]
    fn test_partition_by_singleton() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a2", "a1", "a3"])?;
            let partitioning = PartitioningOptions::Singleton;

            from_a.partition_by(partitioning)?.build()
        });

        tester.expect_expr(
            r#"
LogicalGet A cols=[1, 2, 3] partitioning=()
  output cols: [1, 2, 3]
Metadata:
  col:1 A.a2 Int32
  col:2 A.a1 Int32
  col:3 A.a3 Int32
Memo:
  00 LogicalGet A cols=[1, 2, 3]
"#,
        );
    }

    #[test]
    fn test_partition_by_columns() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a2", "a1", "a3"])?;

            from_a.partition_by(vec!["a1", "a2"])?.build()
        });

        tester.expect_expr(
            r#"
LogicalGet A cols=[1, 2, 3] partitioning=[1, 2]
  output cols: [1, 2, 3]
Metadata:
  col:1 A.a2 Int32
  col:2 A.a1 Int32
  col:3 A.a3 Int32
Memo:
  00 LogicalGet A cols=[1, 2, 3]
"#,
        );
    }

    #[test]
    fn test_partitioning_and_ordering() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a2", "a1", "a3"])?;
            let partitioned = from_a.partition_by(vec!["a1", "a2"])?;
            let ordered = partitioned.order_by(OrderingOption::by("a1", false))?;

            ordered.build()
        });

        tester.expect_expr(
            r#"
LogicalGet A cols=[1, 2, 3] ordering=[+2] partitioning=[1, 2]
  output cols: [1, 2, 3]
Metadata:
  col:1 A.a2 Int32
  col:2 A.a1 Int32
  col:3 A.a3 Int32
Memo:
  00 LogicalGet A cols=[1, 2, 3]
"#,
        );
    }

    #[test]
    fn test_select() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let filter = col("a1").gt(scalar(100));

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

            let col_a2 = col("A2");
            let val_i100 = scalar(100);
            let alias = col_a2.clone().alias("a2_alias");
            let projection_list = vec![col_a2, val_i100, alias];

            from_a.project(projection_list)?.build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[2, 3, 4] exprs: [col:2, 100, col:2 AS a2_alias]
  input: LogicalGet A cols=[1, 2]
  output cols: [2, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 ?column? Int32, expr: 100
  col:4 a2_alias Int32, expr: col:2
Memo:
  04 LogicalProjection input=00 exprs=[01, 02, 03] cols=[2, 3, 4]
  03 Expr col:2 AS a2_alias
  02 Expr 100
  01 Expr col:2
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_projection_with_wildcard() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.from("A")?;
            let projection_list = vec![col("a1"), wildcard(), col("a1")];

            from_a.project(projection_list)?.build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[1, 1, 2, 3, 4, 1] exprs: [col:1, col:1, col:2, col:3, col:4, col:1]
  input: LogicalGet A cols=[1, 2, 3, 4]
  output cols: [1, 1, 2, 3, 4, 1]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 A.a4 Int32
Memo:
  05 LogicalProjection input=00 exprs=[01, 01, 02, 03, 04, 01] cols=[1, 1, 2, 3, 4, 1]
  04 Expr col:4
  03 Expr col:3
  02 Expr col:2
  01 Expr col:1
  00 LogicalGet A cols=[1, 2, 3, 4]
"#,
        );
    }

    #[test]
    fn test_projection_with_qualified_wildcard() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.from("A")?;
            let projection_list = vec![col("a1"), qualified_wildcard("a"), col("a.a2")];

            from_a.project(projection_list)?.build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[1, 1, 2, 3, 4, 2] exprs: [col:1, col:1, col:2, col:3, col:4, col:2]
  input: LogicalGet A cols=[1, 2, 3, 4]
  output cols: [1, 1, 2, 3, 4, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 A.a4 Int32
Memo:
  05 LogicalProjection input=00 exprs=[01, 01, 02, 03, 04, 02] cols=[1, 1, 2, 3, 4, 2]
  04 Expr col:4
  03 Expr col:3
  02 Expr col:2
  01 Expr col:1
  00 LogicalGet A cols=[1, 2, 3, 4]
"#,
        );
    }

    #[test]
    fn test_projection_of_projection() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let projection_list1 = vec![col("a2"), scalar(100), col("a1")];

            let projection_list2 = vec![col("a2")];

            builder
                .get("A", vec!["a1", "a2"])?
                .project(projection_list1)?
                .project(projection_list2)?
                .build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[2] exprs: [col:2]
  input: LogicalProjection cols=[2, 3, 1] exprs: [col:2, 100, col:1]
    input: LogicalGet A cols=[1, 2]
  output cols: [2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 ?column? Int32, expr: 100
Memo:
  05 LogicalProjection input=04 exprs=[01] cols=[2]
  04 LogicalProjection input=00 exprs=[01, 02, 03] cols=[2, 3, 1]
  03 Expr col:1
  02 Expr 100
  01 Expr col:2
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_projection_with_aggr_expressions_are_not_allowed() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.from("A")?;
            let count_a2 = ScalarExpr::Aggregate {
                func: AggregateFunction::Avg,
                distinct: false,
                args: vec![col("a2")],
                filter: None,
            };
            let projection_list = vec![col("a1"), count_a2];

            from_a.project(projection_list)?.build()
        });

        tester.expect_error("Aggregate expressions are not allowed in projection operator");
    }

    #[test]
    fn test_join_using() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let join = left.join_using(right, JoinType::Inner, vec![("a1", "b2")])?;

            join.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=Inner using=[(1, 4)]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  02 LogicalJoin left=00 right=01 type=Inner using=[(1, 4)]
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_join_using_reject_cross_join() {
        fn expect_cross_join_is_rejected(columns: Vec<(String, String)>) {
            let mut tester = OperatorBuilderTester::new();

            tester.build_operator(move |builder| {
                let left = builder.get("A", vec!["a1", "a2"])?;
                let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
                let join = left.join_using(right, JoinType::Cross, columns.clone())?;

                join.build()
            });

            tester.expect_error("CROSS JOIN: USING (columns) condition is not supported");
        }

        expect_cross_join_is_rejected(vec![]);
        expect_cross_join_is_rejected(vec![("a1".into(), "b2".into())]);
    }

    #[test]
    fn test_join_on() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let expr = col("a1").eq(col("b1"));
            let join = left.join_on(right, JoinType::Inner, expr)?;

            join.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=Inner on=col:1 = col:3
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  03 LogicalJoin left=00 right=01 type=Inner on=col:1 = col:3
  02 Expr col:1 = col:3
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_join_on_reject_cross_join_for_non_true_expr() {
        fn expect_cross_join_is_rejected(expr: ScalarExpr) {
            let mut tester = OperatorBuilderTester::new();

            tester.build_operator(move |builder| {
                let left = builder.get("A", vec!["a1", "a2"])?;
                let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
                let join = left.join_on(right, JoinType::Cross, expr.clone())?;

                join.build()
            });

            tester.expect_error("CROSS JOIN: Invalid expression in ON <expr> condition");
        }

        expect_cross_join_is_rejected(ScalarExpr::Scalar(false.get_value()));
        expect_cross_join_is_rejected(ScalarExpr::Scalar(1.get_value()));
    }

    #[test]
    fn test_natural_join() {
        let mut tester = OperatorBuilderTester::new();

        tester.update_catalog(|catalog| {
            catalog
                .add_table(
                    DEFAULT_SCHEMA,
                    TableBuilder::new("A2")
                        .add_column("a2", DataType::Int32)
                        .add_column("a1", DataType::Int32)
                        .add_column("a22", DataType::Int32)
                        .build()
                        .expect("Table A2"),
                )
                .unwrap();
        });

        tester.build_operator(|builder| {
            let left = builder.from("A")?;
            let right = left.new_query_builder().from("A2")?;
            let join = left.natural_join(right, JoinType::Inner)?;

            join.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=Inner using=[(2, 5), (1, 6)]
  left: LogicalGet A cols=[1, 2, 3, 4]
  right: LogicalGet A2 cols=[5, 6, 7]
  output cols: [1, 2, 3, 4, 5, 6, 7]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 A.a4 Int32
  col:5 A2.a2 Int32
  col:6 A2.a1 Int32
  col:7 A2.a22 Int32
Memo:
  02 LogicalJoin left=00 right=01 type=Inner using=[(2, 5), (1, 6)]
  01 LogicalGet A2 cols=[5, 6, 7]
  00 LogicalGet A cols=[1, 2, 3, 4]
"#,
        );
    }

    #[test]
    fn test_natural_join_to_join_on_true() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let join = left.natural_join(right, JoinType::Inner)?;

            join.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=Inner on=true
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  03 LogicalJoin left=00 right=01 type=Inner on=true
  02 Expr true
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_natural_join_condition_is_not_allowed_for_cross_join() {
        do_not_allow_natural_join(JoinType::Cross, "CROSS JOIN");
    }

    #[test]
    fn test_natural_join_condition_is_not_allowed_for_left_semi_join() {
        do_not_allow_natural_join(JoinType::LeftSemi, "LEFT SEMI JOIN");
    }

    #[test]
    fn test_natural_join_condition_is_not_allowed_for_right_semi_join() {
        do_not_allow_natural_join(JoinType::RightSemi, "RIGHT SEMI JOIN");
    }

    #[test]
    fn test_natural_join_condition_is_not_allowed_for_anti_join() {
        do_not_allow_natural_join(JoinType::Anti, "ANTI JOIN");
    }

    fn do_not_allow_natural_join(join_type: JoinType, join_type_str: &str) {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(move |builder| {
            let join_type = join_type.clone();
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let join = left.natural_join(right, join_type)?;

            join.build()
        });

        tester.expect_error(format!("{}: Natural join condition is not allowed", join_type_str).as_str());
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
            let from_a = builder.get("A", vec!["a1", "a2"])?;

            let sub_query = from_a.new_query_builder().empty(true)?.project(vec![scalar(true)])?.build()?;
            let sub_query = ScalarExpr::SubQuery(RelNode::try_from(sub_query)?);

            let filter = sub_query.eq(scalar(true));
            let select = from_a.select(Some(filter))?;

            select.build()
        });

        tester.expect_expr(
            r#"
LogicalSelect
  input: LogicalGet A cols=[1, 2]
  filter: Expr SubQuery 02 = true
  output cols: [1, 2]

Sub query from filter SubQuery 02 = true:
LogicalProjection cols=[3] exprs: [true]
  input: LogicalEmpty return_one_row=true
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 ?column? Bool, expr: true
Memo:
  05 LogicalSelect input=03 filter=04
  04 Expr SubQuery 02 = true
  03 LogicalGet A cols=[1, 2]
  02 LogicalProjection input=00 exprs=[01] cols=[3]
  01 Expr true
  00 LogicalEmpty return_one_row=true
"#,
        );
    }

    #[test]
    //FIXME: This restriction is unnecessary.
    fn test_prohibit_non_memoized_expressions_in_nested_sub_queries() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a1", "a2"])?;

            let col_a1 = &from_a.metadata.get_columns()[0];
            assert_eq!("a1", col_a1.name(), "a1 must be the fist column");

            let Operator { state } =
                from_a.new_query_builder().empty(false)?.project(vec![scalar(true)])?.operator.unwrap();

            let subquery_props = Properties::Relational(RelationalProperties {
                logical: LogicalProperties::new(vec![*col_a1.id()], None),
                physical: PhysicalProperties::none(),
            });
            let subquery = Operator::new(state.expr().clone(), subquery_props);
            let expr = ScalarExpr::SubQuery(RelNode::try_from(subquery)?);

            let filter = expr.eq(scalar(true));

            let _select = from_a.select(Some(filter))?;

            unreachable!()
        });

        tester.expect_error("Use OperatorBuilder::sub_query_builder to build a nested sub query")
    }

    #[test]
    fn test_prohibit_sub_queries_with_more_than_one_output_column() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let from_b = builder.new_query_builder().get("B", vec!["b1", "b2"])?.build()?;
            let subquery = ScalarExpr::SubQuery(RelNode::try_from(from_b)?);
            let _projection = builder.empty(true)?.project(vec![subquery])?;

            unreachable!()
        });

        tester.expect_error("Subquery must return exactly one column")
    }

    #[test]
    fn test_aggregate() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let mut from_a = builder.get("A", vec!["a1", "a2"])?;
            let sum = from_a
                .aggregate_builder()
                .add_column("a1")?
                .add_func("sum", "a1")?
                .group_by("a1")?
                .having(col("a1").gt(scalar(100)))?
                .build()?;

            sum.with_presentation()?.build()
        });
        tester.expect_expr(
            r#"
LogicalAggregate cols=[1, 3] presentation=[a1:1, sum:3]
  aggr_exprs: [col:1, sum(col:1)]
  group_exprs: [col:1]
  input: LogicalGet A cols=[1, 2]
  having: Expr col:1 > 100
  output cols: [1, 3]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 sum Int32, expr: sum(col:1)
Memo:
  04 LogicalAggregate input=00 aggr_exprs=[01, 02] group_exprs=[01] having=03 cols=[1, 3]
  03 Expr col:1 > 100
  02 Expr sum(col:1)
  01 Expr col:1
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_window_aggregate_function() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a1", "a2"])?;
            let row_number = ScalarExpr::WindowAggregate {
                func: WindowFunction::RowNumber.into(),
                args: vec![],
                partition_by: vec![col("a1")],
                order_by: vec![],
            };
            let window = from_a.project(vec![col("a1"), row_number])?;

            window.with_presentation()?.build()
        });
        tester.expect_expr(
            r#"
LogicalProjection cols=[1, 3] presentation=[a1:1, row_number:3] exprs: [col:1, col:3]
  input: LogicalWindowAggregate cols=[1, 3]
    input: LogicalGet A cols=[1, 2]
    window_expr: Expr row_number() OVER(PARTITION BY col:1)
  output cols: [1, 3]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 row_number Int32, expr: row_number() OVER(PARTITION BY col:1)
Memo:
  05 LogicalProjection input=03 exprs=[01, 04] cols=[1, 3]
  04 Expr col:3
  03 LogicalWindowAggregate input=00 window_expr=02 cols=[1, 3]
  02 Expr row_number() OVER(PARTITION BY col:1)
  01 Expr col:1
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_window_aggregate_reuse_operator() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a1", "a2"])?;
            let first_value = ScalarExpr::WindowAggregate {
                func: WindowFunction::FirstValue.into(),
                args: vec![col("a2")],
                partition_by: vec![col("a1")],
                order_by: vec![],
            };
            let window = from_a.project(vec![col("a1"), first_value.clone(), first_value])?;

            window.with_presentation()?.build()
        });
        tester.expect_expr(
            r#"
LogicalProjection cols=[1, 3, 4] presentation=[a1:1, first_value:3, first_value:4] exprs: [col:1, col:3, col:4]
  input: LogicalWindowAggregate cols=[1, 3]
    input: LogicalGet A cols=[1, 2]
    window_expr: Expr first_value(col:2) OVER(PARTITION BY col:1)
  output cols: [1, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 first_value Int32, expr: first_value(col:2) OVER(PARTITION BY col:1)
  col:4 first_value Int32, expr: first_value(col:2) OVER(PARTITION BY col:1)
Memo:
  06 LogicalProjection input=03 exprs=[01, 04, 05] cols=[1, 3, 4]
  05 Expr col:4
  04 Expr col:3
  03 LogicalWindowAggregate input=00 window_expr=02 cols=[1, 3]
  02 Expr first_value(col:2) OVER(PARTITION BY col:1)
  01 Expr col:1
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_multiple_window_aggregate_functions() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let from_a = builder.get("A", vec!["a1", "a2"])?;

            let rank = ScalarExpr::WindowAggregate {
                func: WindowFunction::Rank.into(),
                args: vec![],
                partition_by: vec![col("a1")],
                order_by: vec![],
            };
            let row_number = ScalarExpr::WindowAggregate {
                func: WindowFunction::RowNumber.into(),
                args: vec![],
                partition_by: vec![],
                order_by: vec![],
            };
            let window = from_a.project(vec![rank, col("a2"), row_number])?;

            window.with_presentation()?.build()
        });
        tester.expect_expr(
            r#"
LogicalProjection cols=[3, 2, 4] presentation=[rank:3, a2:2, row_number:4] exprs: [col:3, col:2, col:4]
  input: LogicalWindowAggregate cols=[3, 2, 4]
    input: LogicalWindowAggregate cols=[3, 2]
      input: LogicalGet A cols=[1, 2]
      window_expr: Expr rank() OVER(PARTITION BY col:1)
    window_expr: Expr row_number() OVER()
  output cols: [3, 2, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 rank Int32, expr: rank() OVER(PARTITION BY col:1)
  col:4 row_number Int32, expr: row_number() OVER()
Memo:
  08 LogicalProjection input=05 exprs=[06, 02, 07] cols=[3, 2, 4]
  07 Expr col:4
  06 Expr col:3
  05 LogicalWindowAggregate input=04 window_expr=03 cols=[3, 2, 4]
  04 LogicalWindowAggregate input=00 window_expr=01 cols=[3, 2]
  03 Expr row_number() OVER()
  02 Expr col:2
  01 Expr rank() OVER(PARTITION BY col:1)
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_outer_columns_in_sub_query_filter_expr() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let filter = col("b1").eq(col("a3"));

            // SELECT a1, (SELECT b1 FROM B WHERE b1=a3) FROM A
            let from_a = builder.get("A", vec!["a1", "a2", "a3"])?;
            let b1 = from_a
                .sub_query_builder()
                .get("B", vec!["b1"])?
                .select(Some(filter))?
                .with_presentation()?
                .build()?;

            let b1 = ScalarExpr::SubQuery(RelNode::try_from(b1)?);
            let project = from_a.project(vec![col("a1"), b1])?;

            project.build()
        });
        tester.expect_expr(
            r#"
LogicalProjection cols=[1, 5] exprs: [col:1, SubQuery 02]
  input: LogicalGet A cols=[1, 2, 3]
  output cols: [1, 5]

Sub query from column 5:
LogicalSelect outer_cols=[3] presentation=[b1:4]
  input: LogicalGet B cols=[4]
  filter: Expr col:4 = col:3
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 B.b1 Int32
  col:5 b1 Int32, expr: SubQuery 02
Memo:
  06 LogicalProjection input=03 exprs=[04, 05] cols=[1, 5]
  05 Expr SubQuery 02
  04 Expr col:1
  03 LogicalGet A cols=[1, 2, 3]
  02 LogicalSelect input=00 filter=01
  01 Expr col:4 = col:3
  00 LogicalGet B cols=[4]
"#,
        )
    }

    #[test]
    fn test_outer_columns_in_sub_query_join_condition() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            // SELECT a1, (SELECT b1 FROM B JOIN C ON b1=c1 AND c1=a3) FROM A
            let from_a = builder.clone().get("A", vec!["a1", "a2", "a3"])?;
            let from_c = builder.new_query_builder().get("C", vec!["c1"])?;
            let join_condition = (col("b1").eq(col("c1"))).and(col("c1").eq(col("a3")));

            let b1 = from_a
                .sub_query_builder()
                .get("B", vec!["b1"])?
                .join_on(from_c, JoinType::Inner, join_condition)?
                .project(vec![col("b1")])?
                .with_presentation()?
                .build()?;

            let b1 = ScalarExpr::SubQuery(RelNode::try_from(b1)?);
            let project = from_a.project(vec![col("a1"), b1])?;

            project.with_presentation()?.build()
        });
        tester.expect_expr(
            r#"
LogicalProjection cols=[1, 6] presentation=[a1:1, b1:6] exprs: [col:1, SubQuery 05]
  input: LogicalGet A cols=[1, 2, 3]
  output cols: [1, 6]

Sub query from column 6:
LogicalProjection cols=[5] outer_cols=[3] presentation=[b1:5] exprs: [col:5]
  input: LogicalJoin type=Inner on=col:5 = col:4 AND col:4 = col:3 outer_cols=[3]
    left: LogicalGet B cols=[5]
    right: LogicalGet C cols=[4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 C.c1 Int32
  col:5 B.b1 Int32
  col:6 b1 Int32, expr: SubQuery 05
Memo:
  09 LogicalProjection input=06 exprs=[07, 08] cols=[1, 6]
  08 Expr SubQuery 05
  07 Expr col:1
  06 LogicalGet A cols=[1, 2, 3]
  05 LogicalProjection input=03 exprs=[04] cols=[5]
  04 Expr col:5
  03 LogicalJoin left=00 right=01 type=Inner on=col:5 = col:4 AND col:4 = col:3
  02 Expr col:5 = col:4 AND col:4 = col:3
  01 LogicalGet C cols=[4]
  00 LogicalGet B cols=[5]
"#,
        )
    }

    #[test]
    fn test_union() {
        expect_set_op(SetOperator::Union, false, "LogicalUnion");
        expect_set_op(SetOperator::Union, true, "LogicalUnion");
    }

    #[test]
    fn test_intersect() {
        expect_set_op(SetOperator::Intersect, false, "LogicalIntersect");
        expect_set_op(SetOperator::Intersect, true, "LogicalIntersect");
    }

    #[test]
    fn test_except() {
        expect_set_op(SetOperator::Except, false, "LogicalExcept");
        expect_set_op(SetOperator::Except, true, "LogicalExcept");
    }

    #[test]
    fn test_distinct() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| builder.from("A")?.distinct(None)?.with_presentation()?.build());

        tester.expect_expr(
            r#"
LogicalDistinct cols=[1, 2, 3, 4] presentation=[a1:1, a2:2, a3:3, a4:4]
  input: LogicalGet A cols=[1, 2, 3, 4]
  output cols: [1, 2, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 A.a4 Int32
Memo:
  01 LogicalDistinct input=00 cols=[1, 2, 3, 4]
  00 LogicalGet A cols=[1, 2, 3, 4]
"#,
        );
    }

    #[test]
    fn test_left_semi_join() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let expr = col("a1").eq(col("b1"));
            let join = left.join_on(right, JoinType::LeftSemi, expr)?;

            join.with_presentation()?.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=LeftSemi on=col:1 = col:3 presentation=[a1:1, a2:2]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  03 LogicalJoin left=00 right=01 type=LeftSemi on=col:1 = col:3
  02 Expr col:1 = col:3
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_left_semi_join_using() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let join = left.join_using(right, JoinType::LeftSemi, vec![("a1", "b1")])?;

            join.with_presentation()?.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=LeftSemi using=[(1, 3)] presentation=[a1:1, a2:2]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  02 LogicalJoin left=00 right=01 type=LeftSemi using=[(1, 3)]
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_right_semi_join() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let expr = col("a1").eq(col("b1"));
            let join = left.join_on(right, JoinType::RightSemi, expr)?;

            join.with_presentation()?.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=RightSemi on=col:1 = col:3 presentation=[b1:3, b2:4]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  03 LogicalJoin left=00 right=01 type=RightSemi on=col:1 = col:3
  02 Expr col:1 = col:3
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_right_semi_join_using() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let join = left.join_using(right, JoinType::RightSemi, vec![("a1", "b1")])?;

            join.with_presentation()?.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=RightSemi using=[(1, 3)] presentation=[b1:3, b2:4]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  02 LogicalJoin left=00 right=01 type=RightSemi using=[(1, 3)]
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_anti_join() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let expr = col("a1").eq(col("b1"));
            let join = left.join_on(right, JoinType::Anti, expr)?;

            join.with_presentation()?.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=Anti on=col:1 = col:3 presentation=[a1:1, a2:2]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  03 LogicalJoin left=00 right=01 type=Anti on=col:1 = col:3
  02 Expr col:1 = col:3
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_anti_join_using() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let join = left.join_using(right, JoinType::Anti, vec![("a1", "b1")])?;

            join.with_presentation()?.build()
        });

        tester.expect_expr(
            r#"
LogicalJoin type=Anti using=[(1, 3)] presentation=[a1:1, a2:2]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  02 LogicalJoin left=00 right=01 type=Anti using=[(1, 3)]
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    pub fn test_values() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let values = builder.values(vec![vec![scalar(1), scalar(true)]])?;
            let values = values.with_alias(TableAlias {
                name: "v".to_string(),
                columns: vec![],
            })?;

            values.with_presentation()?.build()
        });

        tester.expect_expr(
            r#"
LogicalValues cols=[1, 2] presentation=[column1:1, column2:2] values: [(1, true)]
  output cols: [1, 2]
Metadata:
  col:1 v.column1 Int32
  col:2 v.column2 Bool
Memo:
  01 LogicalValues values=[00] cols=[1, 2]
  00 Expr (1, true)
"#,
        );
    }

    #[test]
    pub fn test_values_empty_list_are_not_allowed() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let values = builder.values(vec![vec![scalar(1), scalar(true)]])?;
            let projection = values.project(vec![col("column1")])?;
            projection.build()
        });

        tester.expect_error("VALUES in FROM must have an alias");

        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let values = builder.values(vec![vec![scalar(1), scalar(true)]])?;
            values.build()
        });

        tester.expect_error("VALUES in FROM must have an alias");
    }

    fn expect_set_op(set_op: SetOperator, all: bool, logical_op: &str) {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(move |builder| {
            let left = builder.clone().get("A", vec!["a1", "a2"])?;
            let right = builder.get("B", vec!["b1", "b2"])?;

            let op = match set_op {
                SetOperator::Union => {
                    if all {
                        left.union_all(right)?
                    } else {
                        left.union(right)?
                    }
                }
                SetOperator::Intersect => {
                    if all {
                        left.intersect_all(right)?
                    } else {
                        left.intersect(right)?
                    }
                }
                SetOperator::Except => {
                    if all {
                        left.except_all(right)?
                    } else {
                        left.except(right)?
                    }
                }
            };

            op.with_presentation()?.build()
        });

        tester.expect_expr(
            r#"
:set_op all=:all cols=[5, 6] presentation=[a1:5, a2:6]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [5, 6]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 a1 Int32
  col:6 a2 Int32
Memo:
  02 :set_op left=00 right=01 all=:all cols=[5, 6]
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#
            .replace(":set_op", logical_op)
            .replace(":all", format!("{}", all).as_str())
            .as_str(),
        );
    }

    #[test]
    fn test_cte() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let a1_vals = builder.new_relation_builder().from("a")?.project(vec![col("a1")])?;
            let builder = builder.with_table_expr(
                TableAlias {
                    name: "a1_vals".to_string(),
                    columns: vec![],
                },
                a1_vals,
            )?;
            let project = builder.from("a1_vals")?.project(vec![col("a1")])?;
            project.with_presentation()?.build()
        });
        tester.expect_expr(
            r#"
LogicalProjection cols=[1] presentation=[a1:1] exprs: [col:1]
  input: LogicalProjection cols=[1] exprs: [col:1]
    input: LogicalGet a cols=[1, 2, 3, 4]
  output cols: [1]
Metadata:
  col:1 a.a1 Int32
  col:2 a.a2 Int32
  col:3 a.a3 Int32
  col:4 a.a4 Int32
Memo:
  03 LogicalProjection input=02 exprs=[01] cols=[1]
  02 LogicalProjection input=00 exprs=[01] cols=[1]
  01 Expr col:1
  00 LogicalGet a cols=[1, 2, 3, 4]
"#,
        );
    }

    #[test]
    fn test_reject_scalar_array_values() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let empty = builder.empty(true)?;
            let element = 1.get_value();
            let element_type = element.data_type();

            let array = ScalarValue::Array(
                Some(Array {
                    elements: vec![element],
                }),
                Box::new(element_type),
            );
            let array = empty.project(vec![ScalarExpr::Scalar(array)])?;

            array.with_presentation()?.build()
        });
        tester.expect_error("ARRAYS: scalar value of type array");
    }

    #[test]
    fn test_reject_scalar_tuple_values() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let empty = builder.empty(true)?;
            let element = 1.get_value();
            let element_type = element.data_type();

            let tuple = ScalarValue::Tuple(Some(vec![element]), Box::new(DataType::Tuple(vec![element_type])));
            let tuple = empty.project(vec![ScalarExpr::Scalar(tuple)])?;

            tuple.with_presentation()?.build()
        });
        tester.expect_error("TUPLES: scalar value of type tuple");
    }

    struct OperatorBuilderTester {
        operator: Box<dyn Fn(OperatorBuilder) -> Result<Operator, OptimizerError>>,
        update_catalog: Box<dyn Fn(&MutableCatalog)>,
        metadata: Rc<MutableMetadata>,
        memoization: MemoizeOperators,
    }

    impl OperatorBuilderTester {
        fn new() -> Self {
            let properties_builder = LogicalPropertiesBuilder::new(NoStatisticsBuilder);
            let metadata = Rc::new(MutableMetadata::new());
            let memo = OperatorMemoBuilder::new(metadata.clone()).build_with_properties(properties_builder);
            let memoization = MemoizeOperators::new(memo);

            OperatorBuilderTester {
                operator: Box::new(|_| panic!("Operator has not been specified")),
                update_catalog: Box::new(|_| {}),
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

        fn update_catalog<F>(&mut self, f: F)
        where
            F: Fn(&MutableCatalog) + 'static,
        {
            self.update_catalog = Box::new(f)
        }

        fn expect_error(mut self, msg: &str) {
            let result = self.do_build_operator();
            let err = result.expect_err("Expected an error");
            let actual_err_str = format!("{}", err);

            assert!(actual_err_str.contains(msg), "Unexpected error message. Expected: {}.\nActual: {}", msg, err);
        }

        fn expect_expr(mut self, expected: &str) {
            let result = self.do_build_operator();
            let expr = match result {
                Ok(expr) => expr,
                Err(err) => panic!("Unexpected error: {}", err),
            };

            let mut buf = String::new();
            buf.push('\n');

            let memo = self.memoization.into_memo();
            let metadata_formatter = AppendMetadata::new(self.metadata.clone());
            let memo_formatter = AppendMemo::new(memo);
            let subquery_formatter = SubQueriesFormatter::new(self.metadata.clone());

            let formatter = OperatorTreeFormatter::new()
                .properties_last()
                .add_formatter(Box::new(subquery_formatter))
                .add_formatter(Box::new(metadata_formatter))
                .add_formatter(Box::new(memo_formatter));

            buf.push_str(formatter.format(&expr).as_str());

            assert_eq!(buf.as_str(), expected);
        }

        fn do_build_operator(&mut self) -> Result<Operator, OptimizerError> {
            let catalog = Arc::new(MutableCatalog::new());
            catalog
                .add_table(
                    DEFAULT_SCHEMA,
                    TableBuilder::new("A")
                        .add_column("a1", DataType::Int32)
                        .add_column("a2", DataType::Int32)
                        .add_column("a3", DataType::Int32)
                        .add_column("a4", DataType::Int32)
                        .build()?,
                )
                .unwrap();

            catalog
                .add_table(
                    DEFAULT_SCHEMA,
                    TableBuilder::new("B")
                        .add_column("b1", DataType::Int32)
                        .add_column("b2", DataType::Int32)
                        .add_column("b3", DataType::Int32)
                        .build()?,
                )
                .unwrap();

            catalog
                .add_table(
                    DEFAULT_SCHEMA,
                    TableBuilder::new("C")
                        .add_column("c1", DataType::Int32)
                        .add_column("c2", DataType::Int32)
                        .add_column("c3", DataType::Int32)
                        .build()?,
                )
                .unwrap();

            (self.update_catalog)(catalog.as_ref());

            let builder = OperatorBuilder::new(self.memoization.take_callback(), catalog, self.metadata.clone());
            (self.operator)(builder)
        }
    }
}
