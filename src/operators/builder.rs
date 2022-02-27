use std::cell::RefCell;
use std::collections::hash_map::RandomState;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt::{Debug, Formatter};
use std::iter::FromIterator;
use std::rc::Rc;

use itertools::Itertools;

use crate::catalog::CatalogRef;
use crate::error::OptimizerError;
use crate::memo::MemoExprState;
use crate::meta::{ColumnId, ColumnMetadata, MutableMetadata, RelationId};
use crate::operators::relational::join::{JoinCondition, JoinOn, JoinType, JoinUsing};
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalEmpty, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin,
    LogicalProjection, LogicalSelect, LogicalUnion, SetOperator,
};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::{AggregateFunction, ExprRewriter, ExprVisitor};
use crate::operators::scalar::exprs::collect_columns;
use crate::operators::scalar::types::resolve_expr_type;
use crate::operators::scalar::value::ScalarValue;
use crate::operators::scalar::{ScalarExpr, ScalarNode};
use crate::operators::{ExprMemo, Operator, OperatorExpr};
use crate::properties::physical::PhysicalProperties;
use crate::properties::OrderingChoice;

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
    fn from_builder(builder: &OperatorBuilder, operator: Operator, columns: Vec<(String, ColumnId)>) -> Self {
        let scope = if let Some(scope) = &builder.scope {
            // Set columns if scope exists.
            scope.with_new_columns(columns)
        } else {
            // If the given builder does not have the scope yet
            // create a new scope from the given columns.
            OperatorScope::from_columns(columns)
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
            let mut rewriter = RewriteExprs::new(&scope, ValidateFilterExpr::where_clause());
            let filter = filter.rewrite(&mut rewriter)?;
            let expr = self.add_scalar_node(filter);
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
            return Err(OptimizerError::Argument(format!("CROSS JOIN: USING (columns) condition is not supported")));
        }

        let mut columns_ids = Vec::new();

        for (l, r) in columns {
            let l: String = l.into();
            let r: String = r.into();

            let left_id = left_scope
                .relation
                .columns
                .iter()
                .find(|(name, _)| name == &l)
                .map(|(_, id)| *id)
                .ok_or_else(|| OptimizerError::Argument("Join left side".into()))?;
            let right_id = right_scope
                .relation
                .columns
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

        let mut rewriter = RewriteExprs::new(&scope, ValidateFilterExpr::join_clause());
        let expr = expr.rewrite(&mut rewriter)?;
        let expr = self.add_scalar_node(expr);
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
            return Err(OptimizerError::Argument(format!("CROSS JOIN: Natural join condition is not allowed")));
        }

        let left_name_id: HashMap<String, ColumnId, RandomState> =
            HashMap::from_iter(left_scope.relation.columns.clone().into_iter());
        let mut column_ids = vec![];
        for (right_name, right_id) in right_scope.relation.columns.iter() {
            if let Some(left_id) = left_name_id.get(right_name) {
                column_ids.push((*left_id, *right_id));
            }
        }

        let condition = if !column_ids.is_empty() {
            JoinCondition::Using(JoinUsing::new(column_ids))
        } else {
            let expr = self.add_scalar_node(ScalarExpr::Scalar(ScalarValue::Bool(true)));
            JoinCondition::On(JoinOn::new(expr))
        };

        let scope = left_scope.join(right_scope);
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
                    let mut rewriter = RewriteExprs::new(&scope, ValidateProjectionExpr::projection_expr());
                    let expr = expr.rewrite(&mut rewriter)?;
                    let columns = &scope.relation.columns;
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
                let required = PhysicalProperties::new(ordering);
                self.operator = Some(operator.with_required(required));
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
            scope: self.scope.as_ref().map(|scope| scope.new_child_scope()),
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
        match self.operator {
            Some(operator) => {
                let rel_node = self.callback.new_rel_expr(operator);
                Ok(rel_node.into_inner())
            }
            None => Err(OptimizerError::Internal("Build: No operator".to_string())),
        }
    }

    /// Creates a nested sub query expression from this nested operator builder.
    /// If this builder has not been created via call to [Self::sub_query_builder()] this method returns an error.
    #[deprecated]
    pub fn to_sub_query(mut self) -> Result<ScalarExpr, OptimizerError> {
        if self.sub_query_builder {
            let (operator, _) = self.rel_node()?;
            if operator.props().logical.output_columns().len() != 1 {
                let message = "Sub query must return only one column";
                Err(OptimizerError::Internal(message.into()))
            } else {
                Ok(ScalarExpr::SubQuery(operator))
            }
        } else {
            let message = "Sub queries can only be created using builders instantiated via call to OperatorBuilder::sub_query_builder()";
            Err(OptimizerError::Internal(message.into()))
        }
    }

    /// Sets an alias to the current operator. If there is no operator returns an error.
    pub fn with_alias(mut self, alias: &str) -> Result<Self, OptimizerError> {
        if let Some(scope) = self.scope.as_mut() {
            scope.set_alias(alias.to_owned())?;
            Ok(self)
        } else {
            Err(OptimizerError::Argument(format!("ALIAS: no operator")))
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

        if left_scope.relation.columns.len() != right_scope.relation.columns.len() {
            return Err(OptimizerError::Argument("Union: Number of columns does not match".to_string()));
        }

        let mut columns = Vec::new();
        let mut column_ids = Vec::new();
        for (i, (l, r)) in left_scope.relation.columns.iter().zip(right_scope.relation.columns.iter()).enumerate() {
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

        self.add_input_operator(expr, RelationInScope::from_columns(columns));
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

        let mut scope = OperatorScope::from_columns(columns);
        if !input_is_projection {
            scope.relations.add_all(input.relations);
        }
        self.scope = Some(scope);
    }

    fn add_input_operator(&mut self, expr: LogicalExpr, relation: RelationInScope) {
        let operator = Operator::from(OperatorExpr::from(expr));
        self.operator = Some(operator);

        if let Some(mut scope) = self.scope.take() {
            scope.relation = relation;
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
        f.debug_struct("OperatorBuilder")
            .field("catalog", &self.catalog)
            .field("metadata", self.metadata.as_ref())
            .field("operator", &self.operator)
            .field("scope", &self.scope)
            .finish()
    }
}

/// Stores columns and relations available to the current node of an operator tree.
#[derive(Debug, Clone)]
pub struct OperatorScope {
    //FIXME: It is not necessary to store output relation in both `relation` and `relations` fields.
    relation: RelationInScope,
    relations: RelationsInScope,
    parent: Option<Rc<OperatorScope>>,
}

impl OperatorScope {
    fn new_source(relation: RelationInScope) -> Self {
        let relations = RelationsInScope::from_relation(relation.clone());

        OperatorScope {
            relation,
            parent: None,
            relations,
        }
    }

    fn from_columns(columns: Vec<(String, ColumnId)>) -> Self {
        let relation = RelationInScope::from_columns(columns);
        let relations = RelationsInScope::from_relation(relation.clone());

        OperatorScope {
            relation,
            parent: None,
            relations,
        }
    }

    fn new_child_scope(&self) -> Self {
        let mut relations = RelationsInScope::from_relation(self.relation.clone());
        relations.add_all(self.relations.clone());

        OperatorScope {
            relation: RelationInScope::from_columns(vec![]),
            parent: Some(Rc::new(self.clone())),
            relations,
        }
    }

    fn with_new_columns(&self, columns: Vec<(String, ColumnId)>) -> Self {
        let relation = RelationInScope::from_columns(columns);
        let relations = RelationsInScope::from_relation(relation.clone());

        OperatorScope {
            relation,
            parent: self.parent.clone(),
            relations,
        }
    }

    fn join(self, right: OperatorScope) -> Self {
        let mut columns = self.relation.columns;
        columns.extend_from_slice(&right.relation.columns);

        let mut relations = self.relations;
        relations.add(right.relation);
        relations.add_all(right.relations);

        OperatorScope {
            relation: RelationInScope::from_columns(columns),
            relations,
            parent: self.parent,
        }
    }

    fn set_alias(&mut self, alias: String) -> Result<(), OptimizerError> {
        fn set_alias(relation: &mut RelationInScope, alias: String) -> Result<(), OptimizerError> {
            if relation.relation_id.is_some() {
                relation.alias = Some(alias.clone());
                Ok(())
            } else if relation.name.is_empty() {
                relation.name = alias.clone();
                relation.alias = Some(alias.clone());
                Ok(())
            } else {
                let message = format!("BUG: a relation has no relation_id but has a name. Relation: {:?}", relation);
                Err(OptimizerError::Internal(message))
            }
        }

        if let Some(relation) = self.relations.relations.first_mut() {
            let output_relation = &mut self.relation;
            set_alias(relation, alias.clone())?;
            set_alias(output_relation, alias)
        } else {
            let message = format!("OperatorScope is empty!");
            return Err(OptimizerError::Internal(message));
        }
    }

    fn resolve_columns(
        &self,
        qualifier: Option<&str>,
        metadata: &MutableMetadata,
    ) -> Result<Vec<(String, ColumnId)>, OptimizerError> {
        if let Some(qualifier) = qualifier {
            let relation = self.relations.find_relation(qualifier);

            match relation {
                Some(relation) => {
                    let columns = relation
                        .columns
                        .iter()
                        .map(|(_, id)| {
                            // Should never panic because relation contains only valid ids.
                            let meta = metadata.get_column(id);
                            let column_name = meta.name().clone();
                            (column_name, *id)
                        })
                        .collect();
                    Ok(columns)
                }
                None if self.relations.find_relation_by_name(qualifier).is_some() => {
                    Err(OptimizerError::Internal(format!("Invalid reference to relation {}", qualifier)))
                }
                None => Err(OptimizerError::Internal(format!("Unknown relation {}", qualifier))),
            }
        } else {
            Ok(self.relation.columns.iter().cloned().collect())
        }
    }

    fn find_column_by_name(&self, name: &str) -> Option<ColumnId> {
        let f = |r: &RelationInScope, rs: &RelationsInScope| {
            r.find_column_by_name(name).or_else(|| rs.find_column_by_name(name))
        };
        self.find_in_scope(&f)
    }

    fn find_column_by_id(&self, id: ColumnId) -> Option<ColumnId> {
        let f =
            |r: &RelationInScope, rs: &RelationsInScope| r.find_column_by_id(&id).or_else(|| rs.find_column_by_id(&id));
        self.find_in_scope(&f)
    }

    fn find_in_scope<F>(&self, f: &F) -> Option<ColumnId>
    where
        F: Fn(&RelationInScope, &RelationsInScope) -> Option<ColumnId>,
    {
        let mut scope = self;
        loop {
            let relations = &scope.relations;
            if let Some(id) = (f)(&self.relation, relations) {
                return Some(id);
            } else if let Some(parent) = &scope.parent {
                scope = parent.as_ref();
            } else {
                return None;
            }
        }
    }
}

#[derive(Debug, Clone)]
struct RelationsInScope {
    relations: Vec<RelationInScope>,
}

impl RelationsInScope {
    fn new() -> Self {
        RelationsInScope { relations: Vec::new() }
    }

    fn from_relation(relation: RelationInScope) -> Self {
        if !relation.columns.is_empty() {
            RelationsInScope {
                relations: vec![relation],
            }
        } else {
            RelationsInScope::new()
        }
    }

    fn add(&mut self, relation: RelationInScope) {
        for r in self.relations.iter() {
            if r.relation_id == relation.relation_id {
                return;
            }
        }
        self.relations.push(relation);
    }

    fn add_all(&mut self, relations: RelationsInScope) {
        let existing: Vec<RelationId> = self.relations.iter().filter_map(|r| r.relation_id).collect();
        self.relations.extend(relations.relations.into_iter().filter(|r| match &r.relation_id {
            Some(id) if existing.contains(id) => false,
            _ => true,
        }))
    }

    fn find_relation(&self, name_or_alias: &str) -> Option<&RelationInScope> {
        self.relations.iter().find(|r| r.has_name_or_alias(name_or_alias))
    }

    fn find_relation_by_name(&self, name: &str) -> Option<&RelationInScope> {
        self.relations.iter().find(|r| r.has_name(name))
    }

    fn find_column_by_id(&self, id: &ColumnId) -> Option<ColumnId> {
        self.relations
            .iter()
            .find_map(|r| r.columns.iter().find(|(_, col_id)| col_id == id).map(|(_, id)| *id))
    }

    fn find_column_by_name(&self, name: &str) -> Option<ColumnId> {
        // FIXME: use object names instead of str.
        if let Some(pos) = name.find('.') {
            let (relation_name, col_name) = name.split_at(pos);
            let relation = self.relations.iter().find(|r| r.has_name_or_alias(relation_name));
            if let Some(relation) = relation {
                relation
                    .columns
                    .iter()
                    .find(|(name, _id)| name.eq_ignore_ascii_case(&col_name[1..]))
                    .map(|(_, id)| *id)
            } else {
                None
            }
        } else {
            self.relations.iter().find_map(|r| {
                r.columns
                    .iter()
                    .find(|(col_name, id)| col_name.eq_ignore_ascii_case(name))
                    .map(|(_, id)| *id)
            })
        }
    }
}

#[derive(Debug, Clone)]
struct RelationInScope {
    name: String,
    alias: Option<String>,
    relation_id: Option<RelationId>,
    // (name/alias, column_id)
    columns: Vec<(String, ColumnId)>,
}

impl RelationInScope {
    fn new(relation_id: RelationId, name: String, columns: Vec<(String, ColumnId)>) -> Self {
        RelationInScope {
            name,
            alias: None,
            relation_id: Some(relation_id),
            columns,
        }
    }

    fn from_columns(columns: Vec<(String, ColumnId)>) -> Self {
        RelationInScope {
            name: String::from(""),
            alias: None,
            relation_id: None,
            columns,
        }
    }

    fn has_name_or_alias(&self, name_or_alias: &str) -> bool {
        if let Some(alias) = &self.alias {
            alias.eq_ignore_ascii_case(name_or_alias)
        } else {
            self.has_name(name_or_alias)
        }
    }

    fn has_name(&self, name: &str) -> bool {
        self.name.eq_ignore_ascii_case(name)
    }

    fn find_column_by_name(&self, name: &str) -> Option<ColumnId> {
        let f = |(col_name, _): &(String, ColumnId)| col_name == name;
        self.find_column(&f)
    }

    fn find_column_by_id(&self, id: &ColumnId) -> Option<ColumnId> {
        let f = |(_, col_id): &(String, ColumnId)| col_id == id;
        self.find_column(&f)
    }

    fn find_column<F>(&self, predicate: &F) -> Option<ColumnId>
    where
        F: Fn(&(String, ColumnId)) -> bool,
    {
        self.columns
            .iter()
            .find_map(|column| if (predicate)(column) { Some(column.1) } else { None })
    }
}

// TODO: Remove alias is post_rewrite
struct RewriteExprs<'a, T> {
    scope: &'a OperatorScope,
    validator: ExprValidator<T>,
}

impl<'a, T> RewriteExprs<'a, T>
where
    T: ValidateExpr,
{
    fn new(scope: &'a OperatorScope, validator: T) -> Self {
        RewriteExprs {
            scope,
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
        let expr = match expr {
            ScalarExpr::Column(column_id) => {
                let exists = self.scope.find_column_by_id(column_id).is_some();
                if !exists {
                    return Err(OptimizerError::Internal(format!(
                        "Unexpected column id: {}. Input columns: {:?}",
                        column_id, self.scope.relation.columns
                    )));
                }
                Ok(expr)
            }
            ScalarExpr::ColumnName(ref column_name) => {
                let column_id = self.scope.find_column_by_name(column_name);

                match column_id {
                    Some(column_id) => Ok(ScalarExpr::Column(column_id)),
                    None => {
                        return Err(OptimizerError::Internal(format!(
                            "Unexpected column: {}. Input columns: {:?}, Outer columns: {:?}",
                            column_name,
                            self.scope.relation.columns,
                            self.scope.parent.as_ref().map(|s| &s.relation.columns).unwrap_or(&vec![]),
                        )));
                    }
                }
            }
            ScalarExpr::SubQuery(ref rel_node) => {
                let output_columns = rel_node.props().logical().output_columns();
                if output_columns.len() != 1 {
                    let message = "Subquery must return exactly one column";
                    return Err(OptimizerError::Internal(message.into()));
                }

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
        }?;
        self.validator.validate(&expr)?;
        Ok(expr)
    }

    fn post_rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
        self.validator.after_expr(&expr);
        Ok(expr)
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
            return Err(OptimizerError::Internal(format!("Nested aggregate functions are not allowed")));
        }
        if self.alias_depth > 1 {
            // query error
            return Err(OptimizerError::Internal(format!("Nested alias expressions are not allowed")));
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
                Err(OptimizerError::Internal(format!("aliases are not allowed in filter expressions")))
            }
            ScalarExpr::Aggregate { .. } if !self.allow_aggregates => {
                // query error
                //TODO: Include clause (WHERE, JOIN etc)
                Err(OptimizerError::Internal(format!("aggregates are not allowed")))
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

/// A builder for aggregate operators.  
pub struct AggregateBuilder<'a> {
    builder: &'a mut OperatorBuilder,
    aggr_exprs: Vec<AggrExpr>,
    group_exprs: Vec<ScalarExpr>,
    having: Option<ScalarExpr>,
}

enum AggrExpr {
    Function { func: AggregateFunction, column: String },
    Column(String),
    Expr(ScalarExpr),
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

    /// Adds an expression to the aggregate operator.
    pub fn add_expr(mut self, expr: ScalarExpr) -> Result<Self, OptimizerError> {
        self.aggr_exprs.push(AggrExpr::Expr(expr));
        Ok(self)
    }

    /// Adds grouping by the given column.
    pub fn group_by(mut self, column_name: &str) -> Result<Self, OptimizerError> {
        self.group_exprs.push(ScalarExpr::ColumnName(column_name.into()));
        Ok(self)
    }

    /// Adds grouping by the given expression.
    pub fn group_by_expr(mut self, expr: ScalarExpr) -> Result<Self, OptimizerError> {
        self.group_exprs.push(expr);
        Ok(self)
    }

    /// Adds HAVING clause.
    pub fn having(mut self, expr: ScalarExpr) -> Result<Self, OptimizerError> {
        self.having = Some(expr);
        Ok(self)
    }

    /// Builds an aggregate operator and adds in to an operator tree.
    pub fn build(mut self) -> Result<OperatorBuilder, OptimizerError> {
        let (input, scope) = self.builder.rel_node()?;

        let aggr_exprs = std::mem::take(&mut self.aggr_exprs);
        let mut projection_builder = ProjectionListBuilder::new(self.builder, &scope);
        let mut non_aggregate_columns = HashSet::new();

        for expr in aggr_exprs {
            match expr {
                AggrExpr::Function { func, column } => {
                    let mut rewriter = RewriteExprs::new(&scope, ValidateProjectionExpr::projection_expr());
                    let expr = ScalarExpr::ColumnName(column);
                    let expr = expr.rewrite(&mut rewriter)?;
                    let aggr_expr = ScalarExpr::Aggregate {
                        func,
                        args: vec![expr],
                        filter: None,
                    };
                    projection_builder.add_expr(aggr_expr)?;
                }
                AggrExpr::Column(name) => {
                    let i = projection_builder.projection.projection.len();
                    let expr = ScalarExpr::ColumnName(name);
                    projection_builder.add_expr(expr)?;
                    let column_id = projection_builder.projection.column_ids[i];

                    non_aggregate_columns.insert(column_id);
                }
                AggrExpr::Expr(expr) => {
                    let i = projection_builder.projection.projection.len();
                    projection_builder.add_expr(expr)?;
                    let expr = projection_builder.projection.projection[i].expr();

                    // AggregateBuilder::disallow_nested_subqueries(expr, "AGGREGATE", "aggregate function/expression")?;

                    let used_columns = match &expr {
                        ScalarExpr::Aggregate { .. } => vec![],
                        ScalarExpr::Alias(expr, _) if matches!(expr.as_ref(), ScalarExpr::Aggregate { .. }) => {
                            vec![]
                        }
                        _ => collect_columns(&expr),
                    };
                    non_aggregate_columns.extend(used_columns);
                }
            }
        }

        let ProjectionList {
            column_ids,
            output_columns,
            projection: aggr_exprs,
        } = projection_builder.build();

        let mut group_exprs = vec![];
        let mut group_by_columns = HashSet::new();

        for expr in std::mem::take(&mut self.group_exprs) {
            let expr = if let ScalarExpr::Scalar(ScalarValue::Int32(pos)) = expr {
                let pos = pos - 1;
                if pos >= 0 && (pos as usize) < aggr_exprs.len() {
                    aggr_exprs[pos as usize].expr().clone()
                } else {
                    // query error
                    return Err(OptimizerError::Internal(format!("GROUP BY: position {} is not in select list", pos)));
                }
            } else {
                expr
            };

            // AggregateBuilder::disallow_nested_subqueries(&expr, "GROUP BY", "expressions")?;

            let expr = match expr {
                ScalarExpr::Column(_) | ScalarExpr::ColumnName(_) => {
                    let mut rewriter = RewriteExprs::new(&scope, ValidateProjectionExpr::projection_expr());
                    let expr = expr.rewrite(&mut rewriter)?;
                    let expr = self.builder.add_scalar_node(expr);
                    if let ScalarExpr::Column(id) = expr.expr() {
                        group_by_columns.insert(*id);
                    } else {
                        unreachable!("GROUP BY: positional argument")
                    }
                    expr
                }
                ScalarExpr::SubQuery(_) => ScalarNode::from(expr),
                ScalarExpr::Scalar(ScalarValue::Int32(_)) => unreachable!("GROUP BY: positional argument"),
                ScalarExpr::Scalar(_) => {
                    // query error
                    return Err(OptimizerError::Internal(format!("GROUP BY: not integer constant argument")));
                }
                ScalarExpr::Aggregate { .. } => {
                    // query error
                    return Err(OptimizerError::Internal(format!("GROUP BY: Aggregate functions are not allowed")));
                }
                _ => {
                    // query error
                    return Err(OptimizerError::Internal(format!("GROUP BY: unsupported expression: {}", expr)));
                }
            };
            group_exprs.push(expr);
        }

        if !group_by_columns.is_empty() {
            for col_id in non_aggregate_columns.iter() {
                if !group_by_columns.contains(col_id) {
                    // query error. Add table/table alias
                    let column = self.builder.metadata.get_column(col_id);
                    return Err(OptimizerError::Internal(format!(
                        "AGGREGATE column {} must appear in GROUP BY clause",
                        column.name()
                    )));
                }
            }
        }

        let having_expr = if let Some(expr) = self.having.take() {
            AggregateBuilder::disallow_nested_subqueries(&expr, "AGGREGATE", "HAVING clause")?;

            let mut rewriter = RewriteExprs::new(&scope, ValidateFilterExpr::where_clause());
            let expr = expr.rewrite(&mut rewriter)?;
            let mut validate = ValidateHavingClause {
                projection_columns: &non_aggregate_columns,
                group_by_columns: &group_by_columns,
            };

            expr.accept(&mut validate)?;

            struct ValidateHavingClause<'a> {
                projection_columns: &'a HashSet<ColumnId>,
                group_by_columns: &'a HashSet<ColumnId>,
            }

            impl<'a> ExprVisitor<RelNode> for ValidateHavingClause<'a> {
                type Error = OptimizerError;

                fn pre_visit(&mut self, expr: &ScalarExpr) -> Result<bool, Self::Error> {
                    match expr {
                        ScalarExpr::Aggregate { .. } => Ok(false),
                        ScalarExpr::Column(id) => {
                            if self.projection_columns.contains(id) || self.group_by_columns.contains(id) {
                                Ok(false)
                            } else {
                                Err(OptimizerError::Internal(format!(
                                    "AGGREGATE: Column {} must appear in GROUP BY clause or an aggregate function",
                                    id
                                )))
                            }
                        }
                        _ => Ok(true),
                    }
                }

                fn post_visit(&mut self, _expr: &ScalarExpr) -> Result<(), Self::Error> {
                    Ok(())
                }
            }

            Some(self.builder.add_scalar_node(expr))
        } else {
            None
        };

        let aggregate = LogicalExpr::Aggregate(LogicalAggregate {
            input,
            aggr_exprs,
            group_exprs,
            columns: column_ids,
            having: having_expr,
        });
        let operator = Operator::from(OperatorExpr::from(aggregate));

        Ok(OperatorBuilder::from_builder(self.builder, operator, output_columns))
    }

    fn disallow_nested_subqueries(expr: &ScalarExpr, clause: &str, location: &str) -> Result<(), OptimizerError> {
        struct DisallowNestedSubqueries<'a> {
            clause: &'a str,
            location: &'a str,
        }
        impl<'a> ExprVisitor<RelNode> for DisallowNestedSubqueries<'a> {
            type Error = OptimizerError;

            fn post_visit(&mut self, expr: &ScalarExpr) -> Result<(), Self::Error> {
                match expr {
                    ScalarExpr::Column(_) => {}
                    ScalarExpr::ColumnName(_) => {}
                    ScalarExpr::Scalar(_) => {}
                    ScalarExpr::BinaryExpr { .. } => {}
                    ScalarExpr::Cast { .. } => {}
                    ScalarExpr::Not(_) => {}
                    ScalarExpr::Negation(_) => {}
                    ScalarExpr::Alias(_, _) => {}
                    ScalarExpr::Aggregate { .. } => {}
                    ScalarExpr::SubQuery(_) => {
                        return Err(OptimizerError::NotImplemented(format!(
                            "{}: Subqueries in {} are not implemented",
                            self.clause, self.location
                        )))
                    }
                    ScalarExpr::Wildcard(_) => {
                        return Err(OptimizerError::Internal(format!(
                            "{}: Wildcard expressions are not allowed",
                            self.clause
                        )))
                    }
                }
                Ok(())
            }
        }
        let mut visitor = DisallowNestedSubqueries { clause, location };
        expr.accept(&mut visitor)
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

struct ProjectionListBuilder<'a> {
    builder: &'a mut OperatorBuilder,
    scope: &'a OperatorScope,
    projection: ProjectionList,
}

#[derive(Default)]
struct ProjectionList {
    column_ids: Vec<ColumnId>,
    output_columns: Vec<(String, ColumnId)>,
    projection: Vec<ScalarNode>,
}

impl ProjectionList {
    fn add_expr(&mut self, id: ColumnId, name: String, expr: ScalarNode) {
        self.projection.push(expr);
        self.column_ids.push(id);
        self.output_columns.push((name, id));
    }

    fn get_expr(&self, i: usize) -> Option<&ScalarExpr> {
        self.projection.get(i).map(|s| s.expr())
    }
}

impl<'a> ProjectionListBuilder<'a> {
    fn new(builder: &'a mut OperatorBuilder, scope: &'a OperatorScope) -> Self {
        ProjectionListBuilder {
            builder,
            scope,
            projection: ProjectionList::default(),
        }
    }

    fn add_expr(&mut self, expr: ScalarExpr) -> Result<(), OptimizerError> {
        match expr {
            ScalarExpr::Wildcard(qualifier) => {
                let columns = self.scope.resolve_columns(qualifier.as_deref(), &self.builder.metadata)?;

                for (name, id) in columns {
                    self.add_column(id, name)?;
                }
            }
            _ => {
                let mut rewriter = RewriteExprs::new(self.scope, ValidateProjectionExpr::projection_expr());
                let expr = expr.rewrite(&mut rewriter)?;
                match expr {
                    ScalarExpr::Column(id) => {
                        let name = {
                            // Should never panic because RewriteExprs set an error when encounters an unknown column id.
                            let meta = self.builder.metadata.get_column(&id);
                            meta.name().clone()
                        };
                        self.add_column(id, name)?
                    }
                    ScalarExpr::Alias(ref inner_expr, ref name) => {
                        //Stores inner expression in meta column metadata.
                        self.add_synthetic_column(expr.clone(), name.clone(), *inner_expr.clone())?
                    }
                    ScalarExpr::Aggregate { ref func, .. } => {
                        self.add_synthetic_column(expr.clone(), format!("{}", func), expr)?
                    }
                    _ => self.add_synthetic_column(expr.clone(), "?column?".into(), expr)?,
                };
            }
        };

        Ok(())
    }

    fn add_column(&mut self, id: ColumnId, name: String) -> Result<(), OptimizerError> {
        let expr = self.builder.add_scalar_node(ScalarExpr::Column(id));

        self.projection.add_expr(id, name, expr);

        Ok(())
    }

    fn add_synthetic_column(
        &mut self,
        expr: ScalarExpr,
        name: String,
        column_expr: ScalarExpr,
    ) -> Result<(), OptimizerError> {
        let data_type = resolve_expr_type(&expr, &self.builder.metadata)?;
        let column_meta = ColumnMetadata::new_synthetic_column(name.clone(), data_type, Some(column_expr));
        let id = self.builder.metadata.add_column(column_meta);
        let expr = self.builder.add_scalar_node(expr);

        self.projection.add_expr(id, name.clone(), expr);

        Ok(())
    }

    fn build(self) -> ProjectionList {
        self.projection
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::catalog::mutable::MutableCatalog;
    use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
    use crate::datatypes::DataType;
    use crate::memo::{format_memo, MemoBuilder};
    use crate::operators::properties::LogicalPropertiesBuilder;
    use crate::operators::scalar::{col, qualified_wildcard, scalar, wildcard};
    use crate::operators::{Properties, RelationalProperties};
    use crate::optimizer::SetPropertiesCallback;
    use crate::properties::logical::LogicalProperties;
    use crate::rules::testing::{OperatorFormatter, OperatorTreeFormatter, SubQueriesFormatter};
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
LogicalGet A cols=[1]
  output cols: [1]
  ordering: [1]
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
LogicalProjection cols=[3, 2] exprs: [col:1 AS x, col:2]
  input: LogicalGet A cols=[1, 2]
  output cols: [3, 2]
  ordering: [3]
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
LogicalGet A cols=[1, 2, 3]
  output cols: [1, 2, 3]
  ordering: [2]
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

            let Operator { state } =
                from_a.new_query_builder().empty(false)?.project(vec![scalar(true)])?.operator.unwrap();

            let subquery_props = Properties::Relational(RelationalProperties {
                logical: LogicalProperties::new(vec![1], None),
                required: PhysicalProperties::none(),
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
            let projection = builder.empty(true)?.project(vec![subquery])?;

            unreachable!()
        });

        tester.expect_error("Subquery must return exactly one column")
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
                .having(col("a1").gt(scalar(100)))?
                .build()?;

            sum.build()
        });
        tester.expect_expr(
            r#"
LogicalAggregate
  aggr_exprs: [col:1, sum(col:1)]
  group_exprs: [col:1]
  input: LogicalGet A cols=[1, 2, 3]
  having: Expr col:1 > 100
  output cols: [1, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 A.a3 Int32
  col:4 sum Int32, expr: sum(col:1)
Memo:
  04 LogicalAggregate input=00 aggr_exprs=[01, 02] group_exprs=[01] having=03
  03 Expr col:1 > 100
  02 Expr sum(col:1)
  01 Expr col:1
  00 LogicalGet A cols=[1, 2, 3]
"#,
        );
    }

    #[test]
    fn test_outer_columns_in_sub_query() {
        let mut tester = OperatorBuilderTester::new();
        tester.build_operator(|builder| {
            let filter = col("b1").eq(col("a1"));

            // SELECT a1, (SELECT b1 WHERE B b1=a1) FROM A
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
LogicalSelect
  input: LogicalGet B cols=[4]
  filter: Expr col:4 = col:1
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
  01 Expr col:4 = col:1
  00 LogicalGet B cols=[4]
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
        update_catalog: Box<dyn Fn(&MutableCatalog) -> ()>,
        metadata: Rc<MutableMetadata>,
        memoization: Rc<MemoizeOperatorCallback>,
    }

    impl OperatorBuilderTester {
        fn new() -> Self {
            let properties_builder = Rc::new(LogicalPropertiesBuilder::new(NoStatisticsBuilder));
            let metadata = Rc::new(MutableMetadata::new());
            let memo = MemoBuilder::new(metadata.clone())
                .set_callback(Rc::new(SetPropertiesCallback::new(properties_builder)))
                .build();
            let memoization = Rc::new(MemoizeOperatorCallback::new(memo));

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
            F: Fn(&MutableCatalog) -> () + 'static,
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

            let memoization = Rc::try_unwrap(self.memoization).unwrap();
            let memo = memoization.into_inner();

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

            (self.update_catalog)(catalog.as_ref());

            let builder = OperatorBuilder::new(self.memoization.clone(), catalog, self.metadata.clone());
            (self.operator)(builder)
        }
    }
}
