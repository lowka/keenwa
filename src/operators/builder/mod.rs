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
use crate::memo::MemoExprState;
use crate::meta::{ColumnId, ColumnMetadata, MetadataRef, MutableMetadata};
use crate::operators::relational::join::{JoinCondition, JoinOn, JoinType, JoinUsing};
use crate::operators::relational::logical::{
    LogicalDistinct, LogicalEmpty, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin, LogicalLimit,
    LogicalOffset, LogicalProjection, LogicalSelect, LogicalUnion, SetOperator,
};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::ExprRewriter;
use crate::operators::scalar::types::resolve_expr_type;
use crate::operators::scalar::value::ScalarValue;
use crate::operators::scalar::{get_subquery, ScalarExpr, ScalarNode};
use crate::operators::{ExprMemo, Operator, OperatorExpr, OuterScope};
use crate::properties::physical::{PhysicalProperties, RequiredProperties};
use crate::properties::OrderingChoice;

mod aggregate;
mod projection;
mod scope;

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
    options: Vec<OrderingOption>,
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
    fn new_rel_expr(&self, expr: Operator, scope: &OperatorScope) -> RelNode;

    /// Called when a new scalar expression is added to an operator tree.
    fn new_scalar_expr(&self, expr: Operator, scope: &OperatorScope) -> ScalarNode;
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
    sub_query_builder: bool,
}

impl OperatorBuilder {
    /// Creates a new instance of OperatorBuilder.
    pub fn new(callback: Rc<dyn OperatorCallback>, catalog: CatalogRef, metadata: Rc<MutableMetadata>) -> Self {
        OperatorBuilder {
            callback,
            catalog,
            metadata,
            scope: None,
            operator: None,
            sub_query_builder: false,
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
            sub_query_builder: builder.sub_query_builder,
        }
    }

    /// Add a scan operator to an operator tree. A scan operator can only be added as a leaf node of a tree.
    ///
    /// Unlike [OperatorBuilder::get] this method adds an operator
    /// that returns all the columns from the given `source`.
    pub fn from(self, source: &str) -> Result<Self, OptimizerError> {
        let table = self
            .catalog
            .get_table(source)
            .ok_or_else(|| OptimizerError::Argument(format!("Table does not exist. Table: {}", source)))?;
        let columns = table.columns().iter().map(|c| c.name()).cloned().collect();
        self.add_scan_operator(source, columns)
    }

    /// Adds a scan operator to an operator tree. A scan operator can only be added as leaf node of a tree.
    pub fn get(self, source: &str, columns: Vec<impl Into<String>>) -> Result<Self, OptimizerError> {
        self.add_scan_operator(source, columns)
    }

    /// Adds a select operator to an operator tree.
    pub fn select(mut self, filter: Option<ScalarExpr>) -> Result<Self, OptimizerError> {
        let (input, scope) = self.rel_node()?;
        let filter = if let Some(filter) = filter {
            let metadata = self.metadata.clone();
            let mut rewriter = RewriteExprs::new(&scope, metadata, ValidateFilterExpr::where_clause());
            let filter = filter.rewrite(&mut rewriter)?;
            let expr = self.add_scalar_node(filter, &scope);
            Some(expr)
        } else {
            None
        };

        let expr = LogicalExpr::Select(LogicalSelect { input, filter });

        self.add_operator_and_scope(expr, scope);
        Ok(self)
    }

    /// Adds a projection operator to an operator tree.
    pub fn project(mut self, exprs: Vec<ScalarExpr>) -> Result<Self, OptimizerError> {
        let (input, scope) = self.rel_node()?;
        let input_is_projection = matches!(input.expr().logical(), &LogicalExpr::Projection(_));

        let mut projection_builder = ProjectionListBuilder::new(&mut self, &scope);

        for expr in exprs {
            projection_builder.add_expr(expr)?;
        }

        let ProjectionList {
            column_ids,
            output_columns,
            projection: new_exprs,
        } = projection_builder.build();

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
            return Err(OptimizerError::Argument("CROSS JOIN: USING (columns) condition is not supported".to_string()));
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
                .ok_or_else(|| OptimizerError::Argument("Join left side".into()))?;
            let right_id = right_scope
                .columns()
                .iter()
                .find(|(name, _)| name == &r)
                .map(|(_, id)| *id)
                .ok_or_else(|| OptimizerError::Argument("Join right side".into()))?;
            columns_ids.push((left_id, right_id));
        }

        let scope = left_scope.join(right_scope);
        let condition = JoinCondition::using(columns_ids);
        let expr = LogicalExpr::Join(LogicalJoin {
            join_type,
            left,
            right,
            condition,
        });

        self.add_operator_and_scope(expr, scope);
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
                ScalarExpr::Scalar(ScalarValue::Bool(true)) => {}
                _ => {
                    return Err(OptimizerError::Argument(format!(
                        "CROSS JOIN: Invalid expression in ON <expr> condition: {}",
                        expr
                    )))
                }
            }
        }

        let scope = left_scope.join(right_scope);

        let metadata = self.metadata.clone();
        let mut rewriter = RewriteExprs::new(&scope, metadata, ValidateFilterExpr::join_clause());
        let expr = expr.rewrite(&mut rewriter)?;
        let expr = self.add_scalar_node(expr, &scope);
        let condition = JoinCondition::On(JoinOn::new(expr));
        let expr = LogicalExpr::Join(LogicalJoin {
            join_type,
            left,
            right,
            condition,
        });

        self.add_operator_and_scope(expr, scope);
        Ok(self)
    }

    /// Adds a natural join operator to an operator tree.
    pub fn natural_join(mut self, mut right: OperatorBuilder, join_type: JoinType) -> Result<Self, OptimizerError> {
        let (left, left_scope) = self.rel_node()?;
        let (right, right_scope) = right.rel_node()?;

        if join_type == JoinType::Cross {
            return Err(OptimizerError::Argument("CROSS JOIN: Natural join condition is not allowed".to_string()));
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
            let expr = self.add_scalar_node(ScalarExpr::Scalar(ScalarValue::Bool(true)), &scope);
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
                    let OrderingOption {
                        expr,
                        descending: _descending,
                    } = option;

                    let metadata = self.metadata.clone();
                    let mut rewriter = RewriteExprs::new(&scope, metadata, ValidateProjectionExpr::projection_expr());
                    let expr = expr.rewrite(&mut rewriter)?;
                    let columns = scope.columns();
                    let column_id = match expr {
                        ScalarExpr::Scalar(ScalarValue::Int32(pos)) => match usize::try_from(pos - 1).ok() {
                            Some(pos) if pos < columns.len() => {
                                let (_, id) = columns[pos];
                                id
                            }
                            Some(_) | None => {
                                return Err(OptimizerError::Argument(format!(
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
                    ordering_columns.push(column_id);
                }

                let ordering = OrderingChoice::new(ordering_columns);
                let required = RequiredProperties::new_with_ordering(ordering);
                let physical = PhysicalProperties::with_required(required);
                self.operator = Some(operator.with_physical(physical));
                self.scope = Some(scope);
                Ok(self)
            }
            _ => Err(OptimizerError::Internal("No input operator".to_string())),
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
            return Result::Err(OptimizerError::Internal(
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
            let expr = self.add_scalar_node(on_expr, &input_scope);
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

    /// Returns a builder to construct an aggregate operator.
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
            sub_query_builder: true,
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
            sub_query_builder: true,
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
            sub_query_builder: false,
        }
    }

    /// Creates an operator tree and returns its metadata.
    /// If this builder has been created via call to [Self::sub_query_builder()] this method returns an error.
    pub fn build(self) -> Result<Operator, OptimizerError> {
        match (self.operator, self.scope) {
            (Some(operator), Some(scope)) => {
                let rel_node = self.callback.new_rel_expr(operator, &scope);
                Ok(rel_node.into_inner())
            }
            (None, _) => Err(OptimizerError::Internal("Build: No operator".to_string())),
            (_, _) => Err(OptimizerError::Internal("Build: No scope".to_string())),
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

            scope.set_alias(alias)?;
            Ok(self)
        } else {
            Err(OptimizerError::Argument("ALIAS: no operator".to_string()))
        }
    }

    fn add_scan_operator(mut self, source: &str, columns: Vec<impl Into<String>>) -> Result<Self, OptimizerError> {
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

        let relation_name = String::from(source);
        let relation_id = self.metadata.add_table(relation_name.as_str());
        //TODO: use a single method to add relation and its columns.
        let columns: Result<Vec<(String, ColumnId)>, OptimizerError> = columns
            .iter()
            .map(|name| {
                table.get_column(name).ok_or_else(|| {
                    OptimizerError::Argument(format!("Column does not exist. Column: {}. Table: {}", name, source))
                })
            })
            .map_ok(|column| {
                let column_name = column.name().clone();
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
            return Err(OptimizerError::Argument("Union: Number of columns does not match".to_string()));
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

                if l.data_type() != r.data_type() {
                    let message =
                        format!("Union: Data type of {}-th column does not match. Type casting is not implemented", i);
                    return Err(OptimizerError::NotImplemented(message));
                }
                l.data_type().clone()
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

    fn add_operator_and_scope(&mut self, expr: LogicalExpr, scope: OperatorScope) {
        let operator = Operator::from(OperatorExpr::from(expr));
        self.operator = Some(operator);
        self.scope = Some(scope);
    }

    fn rel_node(&mut self) -> Result<(RelNode, OperatorScope), OptimizerError> {
        let operator = self
            .operator
            .take()
            .ok_or_else(|| OptimizerError::Internal("No input operator".to_string()))?;

        let scope = self.scope.take().ok_or_else(|| OptimizerError::Internal("No scope".to_string()))?;

        let rel_node = self.callback.new_rel_expr(operator, &scope);
        Ok((rel_node, scope))
    }

    fn add_scalar_node(&self, expr: ScalarExpr, scope: &OperatorScope) -> ScalarNode {
        let operator = Operator::from(OperatorExpr::from(expr));
        self.callback.new_scalar_expr(operator, scope)
    }
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
                alias_depth: 0,
                validator,
            },
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
            // Store Rc<metadata> instead of MetadataRef<'_> in
            //TODO: Include names of the outer columns.
            let outer_columns: Vec<_> = scope
                .outer_columns()
                .iter()
                .map(|id| {
                    let col_name = metadata.get_column(id);
                    (col_name.name().clone(), *id)
                })
                .collect();

            OptimizerError::Internal(format!(
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
                    let col_name = format!("!{}", column_id);
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
                        let message = "Subquery must return exactly one column";
                        return Err(OptimizerError::Internal(message.into()));
                    }

                    match query.state() {
                        MemoExprState::Owned(_) => {
                            //FIXME: Add method to handle nested relational expressions to OperatorCallback?
                            Err(OptimizerError::Internal(
                                "Use OperatorBuilder::sub_query_builder to build a nested sub query".to_string(),
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
        self.validator.after_expr(&expr);
        Ok(expr)
    }
}

struct DisplayColumns<'a>(&'a [(String, ColumnId)]);

impl Display for DisplayColumns<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.0.iter().map(|(name, id)| format!("({}, {})", name, id)).join(", "))
    }
}

struct ExprValidator<T> {
    aggregate_depth: usize,
    alias_depth: usize,
    validator: T,
}

impl<T> ExprValidator<T>
where
    T: ValidateExpr,
{
    fn before_expr(&mut self, expr: &ScalarExpr) -> Result<(), OptimizerError> {
        match expr {
            ScalarExpr::Alias(_, _) => self.alias_depth += 1,
            ScalarExpr::Aggregate { .. } => self.aggregate_depth += 1,
            _ => {}
        }
        if self.aggregate_depth > 1 {
            // query error
            return Err(OptimizerError::Internal("Nested aggregate functions are not allowed".to_string()));
        }
        if self.alias_depth > 1 {
            // query error
            return Err(OptimizerError::Internal("Nested alias expressions are not allowed".to_string()));
        }
        self.validator.pre_validate(expr)?;
        Ok(())
    }

    fn validate(&mut self, expr: &ScalarExpr) -> Result<(), OptimizerError> {
        self.validator.validate(expr)?;
        Ok(())
    }

    fn after_expr(&mut self, expr: &ScalarExpr) {
        match expr {
            ScalarExpr::Alias(_, _) => self.alias_depth -= 1,
            ScalarExpr::Aggregate { .. } => self.aggregate_depth -= 1,
            _ => {}
        }
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
                Err(OptimizerError::Internal("aliases are not allowed in filter expressions".to_string()))
            }
            ScalarExpr::Aggregate { .. } if !self.allow_aggregates => {
                // query error
                //TODO: Include clause (WHERE, JOIN etc)
                Err(OptimizerError::Internal("aggregates are not allowed".to_string()))
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
    fn new_rel_expr(&self, expr: Operator, _scope: &OperatorScope) -> RelNode {
        assert!(expr.expr().is_relational(), "Not a relational expression");
        RelNode::from(expr)
    }

    fn new_scalar_expr(&self, expr: Operator, _scope: &OperatorScope) -> ScalarNode {
        assert!(expr.expr().is_scalar(), "Not a scalar expression");
        ScalarNode::from_mexpr(expr)
    }
}

/// A helper that to populate a memo data structure.
///
/// Usage:
/// ```text
///   let memo = ...;
///   let memoization = MemoizeOperators::new(memo);
///   let operator_callback = memoization.get_callback();
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
    fn new_rel_expr(&self, expr: Operator, scope: &OperatorScope) -> RelNode {
        assert!(expr.expr().is_relational(), "Expected a relational expression but got {:?}", expr);
        let mut memo = self.memo.borrow_mut();
        let scope = OuterScope {
            outer_columns: scope.outer_columns().to_vec(),
        };
        let expr = memo.insert_group(expr, &scope);
        RelNode::from_mexpr(expr)
    }

    fn new_scalar_expr(&self, expr: Operator, scope: &OperatorScope) -> ScalarNode {
        assert!(expr.expr().is_scalar(), "Expected a scalar expression but got {:?}", expr);
        let mut memo = self.memo.borrow_mut();
        let scope = OuterScope {
            outer_columns: scope.outer_columns().to_vec(),
        };
        let expr = memo.insert_group(expr, &scope);
        ScalarNode::from_mexpr(expr)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::catalog::mutable::MutableCatalog;
    use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
    use crate::datatypes::DataType;
    use crate::memo::format_memo;
    use crate::operators::format::{OperatorFormatter, OperatorTreeFormatter, SubQueriesFormatter};
    use crate::operators::properties::LogicalPropertiesBuilder;
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
            let ord = OrderingOption::by(("a1", false));

            from_a.order_by(ord)?.build()
        });

        tester.expect_expr(
            r#"
LogicalGet A cols=[1] ordering=[1]
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

            let ord = OrderingOption::by(("x", false));
            projection.order_by(ord)?.build()
        });

        tester.expect_expr(
            r#"
LogicalProjection cols=[3, 2] ordering=[3] exprs: [col:1 AS x, col:2]
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
            let ord = OrderingOption::new(scalar(2), false);

            from_a.order_by(ord)?.build()
        });

        tester.expect_expr(
            r#"
LogicalGet A cols=[1, 2, 3] ordering=[2]
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

        expect_cross_join_is_rejected(ScalarExpr::Scalar(ScalarValue::Bool(false)));
        expect_cross_join_is_rejected(ScalarExpr::Scalar(ScalarValue::Int32(1)));
    }

    #[test]
    fn test_natural_join() {
        let mut tester = OperatorBuilderTester::new();

        tester.update_catalog(|catalog| {
            catalog.add_table(
                DEFAULT_SCHEMA,
                TableBuilder::new("A2")
                    .add_column("a2", DataType::Int32)
                    .add_column("a1", DataType::Int32)
                    .add_column("a22", DataType::Int32)
                    .build(),
            );
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
    fn test_natural_join_do_not_allow_cross_join_type() {
        let mut tester = OperatorBuilderTester::new();

        tester.build_operator(|builder| {
            let left = builder.get("A", vec!["a1", "a2"])?;
            let right = left.new_query_builder().get("B", vec!["b1", "b2"])?;
            let join = left.natural_join(right, JoinType::Cross)?;

            join.build()
        });

        tester.expect_error("CROSS JOIN: Natural join condition is not allowed");
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
            let sub_query = ScalarExpr::SubQuery(RelNode::from(sub_query));

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
            let expr = ScalarExpr::SubQuery(RelNode::from(subquery));

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
            let subquery = ScalarExpr::SubQuery(RelNode::from(from_b));
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

            sum.build()
        });
        tester.expect_expr(
            r#"
LogicalAggregate cols=[1, 3]
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
    fn test_outer_columns_in_sub_query_filter_expr() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let filter = col("b1").eq(col("a3"));

            // SELECT a1, (SELECT b1 FROM B WHERE b1=a3) FROM A
            let from_a = builder.get("A", vec!["a1", "a2", "a3"])?;
            let b1 = from_a.sub_query_builder().get("B", vec!["b1"])?.select(Some(filter))?.build()?;
            let b1 = ScalarExpr::SubQuery(RelNode::from(b1));
            let project = from_a.project(vec![col("a1"), b1])?;

            project.build()
        });
        tester.expect_expr(
            r#"
LogicalProjection cols=[1, 5] exprs: [col:1, SubQuery 02]
  input: LogicalGet A cols=[1, 2, 3]
  output cols: [1, 5]

Sub query from column 5:
LogicalSelect outer_cols=[3]
  input: LogicalGet B cols=[4]
  filter: Expr col:4 = col:3
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 B.b1 Int32
  col:5 ?column? Int32, expr: SubQuery 02
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
                .build()?;
            let b1 = ScalarExpr::SubQuery(RelNode::from(b1));
            let project = from_a.project(vec![col("a1"), b1])?;

            project.build()
        });
        tester.expect_expr(
            r#"
LogicalProjection cols=[1, 6] exprs: [col:1, SubQuery 05]
  input: LogicalGet A cols=[1, 2, 3]
  output cols: [1, 6]

Sub query from column 6:
LogicalProjection cols=[5] outer_cols=[3] exprs: [col:5]
  input: LogicalJoin type=Inner on=col:5 = col:4 AND col:4 = col:3 outer_cols=[3]
    left: LogicalGet B cols=[5]
    right: LogicalGet C cols=[4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 C.c1 Int32
  col:5 B.b1 Int32
  col:6 ?column? Int32, expr: SubQuery 05
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

        tester.build_operator(|builder| builder.from("A")?.distinct(None)?.build());

        tester.expect_expr(
            r#"
LogicalDistinct cols=[1, 2, 3, 4]
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

            op.build()
        });

        tester.expect_expr(
            r#"
:set_op all=:all cols=[5, 6]
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
            let metadata_formatter = AppendMetadata {
                metadata: self.metadata.clone(),
            };
            let memo_formatter = AppendMemo { memo };
            let subquery_formatter = SubQueriesFormatter::new(self.metadata.clone());

            let formatter = OperatorTreeFormatter::new()
                .properties_last()
                .add_formatter(Box::new(subquery_formatter))
                .add_formatter(Box::new(metadata_formatter))
                .add_formatter(Box::new(memo_formatter));

            buf.push_str(formatter.format(&expr).as_str());

            assert_eq!(buf.as_str(), expected);

            struct AppendMetadata {
                metadata: Rc<MutableMetadata>,
            }

            impl OperatorFormatter for AppendMetadata {
                fn write_operator(&self, _operator: &Operator, buf: &mut String) {
                    buf.push_str("Metadata:\n");

                    for column in self.metadata.get_columns() {
                        let id = column.id();
                        let expr = column.expr();
                        let table = column.table();
                        let column_name = column.name();
                        let column_info = match (expr, table) {
                            (None, None) => format!("  col:{} {} {:?}", id, column_name, column.data_type()),
                            (None, Some(table)) => {
                                format!("  col:{} {}.{} {:?}", id, table, column_name, column.data_type())
                            }
                            (Some(expr), _) => {
                                format!("  col:{} {} {:?}, expr: {}", id, column_name, column.data_type(), expr)
                            }
                        };
                        buf.push_str(column_info.as_str());
                        buf.push('\n');
                    }
                }
            }

            struct AppendMemo {
                memo: ExprMemo,
            }

            impl OperatorFormatter for AppendMemo {
                fn write_operator(&self, _operator: &Operator, buf: &mut String) {
                    buf.push_str("Memo:\n");
                    // TODO: Do not include properties.
                    let memo_as_string = format_memo(&self.memo);
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
                }
            }
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

            catalog.add_table(
                DEFAULT_SCHEMA,
                TableBuilder::new("C")
                    .add_column("c1", DataType::Int32)
                    .add_column("c2", DataType::Int32)
                    .add_column("c3", DataType::Int32)
                    .build(),
            );

            (self.update_catalog)(catalog.as_ref());

            let builder = OperatorBuilder::new(self.memoization.take_callback(), catalog, self.metadata.clone());
            (self.operator)(builder)
        }
    }
}
