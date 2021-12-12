use crate::catalog::CatalogRef;
use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::meta::{ColumnId, ColumnMetadata, MutableMetadata};
use crate::operators::relational::join::{JoinCondition, JoinOn};
use crate::operators::relational::logical::{LogicalExpr, SetOperator};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::{AggregateFunction, BinaryOp, Expr, ExprRewriter};
use crate::operators::scalar::{ScalarExpr, ScalarNode};
use crate::operators::{ExprMemo, Operator, OperatorExpr, Properties};
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

/// Callback invoked when a new operator is created by an [operator builder](self::OperatorBuilder).
//FIXME: Rename to operator callback.
pub trait MemoizationHandler {
    fn memoize_rel(&self, expr: Operator) -> RelNode;

    fn memoize_scalar(&self, expr: Operator) -> ScalarNode;
}

/// Provides API to build an [operator tree](super::Operator).
//FIXME: Method that add operators must accept &mut (allows storing in SQL syntax tree parser as a field etc)
//FIXME: use column aliases instead of column ids in public API
#[derive(Debug, Clone)]
pub struct OperatorBuilder {
    memoization: Rc<dyn MemoizationHandler>,
    catalog: CatalogRef,
    metadata: Rc<MutableMetadata>,
    operator: Option<(Operator, OperatorScope)>,
    sub_query_builder: bool,
}

impl OperatorBuilder {
    /// Creates a new instance of operator builder that uses the given [catalog](crate::catalog::Catalog).
    pub fn new(memoization: Rc<dyn MemoizationHandler>, catalog: CatalogRef, metadata: Rc<MutableMetadata>) -> Self {
        OperatorBuilder {
            memoization,
            catalog,
            metadata,
            operator: None,
            sub_query_builder: false,
        }
    }

    fn with_parent(parent: &OperatorBuilder, operator: Operator, scope: OperatorScope) -> Self {
        OperatorBuilder {
            memoization: parent.memoization.clone(),
            catalog: parent.catalog.clone(),
            metadata: parent.metadata.clone(),
            operator: Some((operator, scope)),
            sub_query_builder: parent.sub_query_builder,
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

        let expr = LogicalExpr::Get {
            source: source.into(),
            columns: column_ids,
        };

        self.add_operator(expr, OperatorScope { columns });
        Ok(self)
    }

    /// Adds a select operator to an operator tree.
    pub fn select(mut self, filter: Option<ScalarExpr>) -> Result<Self, OptimizerError> {
        let (input, scope) = self.rel_node()?;
        let filter = if let Some(filter) = filter {
            let mut rewriter = RewriteExprs {
                scope: &scope,
                result: Ok(()),
            };
            let filter = filter.rewrite(&mut rewriter);
            rewriter.result?;

            let expr = OperatorExpr::Scalar(filter);
            let expr = self.memoization.memoize_scalar(Operator::from(expr));
            Some(expr)
        } else {
            None
        };

        let expr = LogicalExpr::Select { input, filter };

        self.add_operator(expr, scope);
        Ok(self)
    }

    /// Adds a projection operator to an operator tree.
    pub fn project_cols(self, columns: Vec<impl Into<String>>) -> Result<Self, OptimizerError> {
        let exprs = columns.into_iter().map(|name| ScalarExpr::ColumnName(name.into())).collect();
        self.project(exprs)
    }

    /// Adds a projection operator to an operator tree.
    pub fn project(mut self, exprs: Vec<ScalarExpr>) -> Result<Self, OptimizerError> {
        let (input, scope) = self.rel_node()?;

        let mut column_ids = Vec::new();
        let mut output_columns = Vec::new();
        for expr in exprs {
            let mut rewriter = RewriteExprs {
                scope: &scope,
                result: Ok(()),
            };
            let expr = expr.rewrite(&mut rewriter);
            rewriter.result?;
            let (id, name) = match expr {
                ScalarExpr::Column(id) => {
                    // Should never panic because RewriteExprs set an error when encounters an unknown column id.
                    let meta = self.metadata.get_column(&id);
                    (id, meta.name().clone())
                }
                _ => {
                    let name = "?column?".to_string();
                    let data_type = resolve_expr_type(&expr, &self.metadata);
                    let column_meta = ColumnMetadata::new_synthetic_column(name.clone(), data_type, Some(expr));
                    let id = self.metadata.add_column(column_meta);
                    (id, name)
                }
            };
            column_ids.push(id);
            output_columns.push((name, id))
        }

        let expr = LogicalExpr::Projection {
            input,
            exprs: vec![],
            columns: column_ids,
        };

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
        let expr = LogicalExpr::Join { left, right, condition };

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

        let mut rewriter = RewriteExprs {
            scope: &scope,
            result: Ok(()),
        };
        let expr = expr.rewrite(&mut rewriter);
        rewriter.result?;

        let condition = JoinCondition::On(JoinOn::new(ScalarNode::from(expr)));
        let expr = LogicalExpr::Join { left, right, condition };

        self.add_operator(expr, scope);
        Ok(self)
    }

    /// Set ordering requirements for the root of an operator tree.
    pub fn order_by(mut self, ordering: impl Into<OrderingOptions>) -> Result<Self, OptimizerError> {
        if let Some((operator, scope)) = self.operator.take() {
            let OrderingOptions { options } = ordering.into();
            let mut ordering_columns = Vec::with_capacity(options.len());

            for option in options {
                let OrderingOption {
                    expr,
                    descending: _descending,
                } = option;
                let mut rewriter = RewriteExprs {
                    scope: &scope,
                    result: Ok(()),
                };
                let expr = expr.rewrite(&mut rewriter);
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
            memoization: self.memoization.clone(),
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
            SetOperator::Union => LogicalExpr::Union {
                left,
                right,
                all,
                columns: column_ids,
            },
            SetOperator::Intersect => LogicalExpr::Intersect {
                left,
                right,
                all,
                columns: column_ids,
            },
            SetOperator::Except => LogicalExpr::Except {
                left,
                right,
                all,
                columns: column_ids,
            },
        };

        self.add_operator(expr, OperatorScope { columns });
        Ok(())
    }

    fn add_operator(&mut self, expr: LogicalExpr, output_columns: OperatorScope) {
        let operator = Operator::from(OperatorExpr::from(expr));
        self.operator = Some((operator, output_columns));
    }

    fn expect_column(
        &self,
        description: &str,
        column_id: &ColumnId,
        input_properties: &Properties,
    ) -> Result<(), OptimizerError> {
        let input_columns = input_properties.logical().output_columns();
        if !input_columns.contains(column_id) {
            Err(OptimizerError::Argument(format!(
                "{}: Unexpected column {}. input columns: {:?}",
                description, column_id, input_columns
            )))
        } else {
            Ok(())
        }
    }

    fn rel_node(&mut self) -> Result<(RelNode, OperatorScope), OptimizerError> {
        let (operator, scope) = self
            .operator
            .take()
            .ok_or_else(|| OptimizerError::Internal("No input operator".to_string()))?;

        let rel_node = self.memoization.memoize_rel(operator);
        Ok((rel_node, scope))
    }
}

#[derive(Debug, Clone)]
struct OperatorScope {
    // (alias, column_id)
    columns: Vec<(String, ColumnId)>,
}

struct RewriteExprs<'a> {
    scope: &'a OperatorScope,
    result: Result<(), OptimizerError>,
}

impl ExprRewriter<RelNode> for RewriteExprs<'_> {
    fn pre_rewrite(&mut self, _expr: &Expr<RelNode>) -> bool {
        self.result.is_ok()
    }

    fn rewrite(&mut self, expr: ScalarExpr) -> ScalarExpr {
        match expr {
            ScalarExpr::Column(column_id) => {
                let exists = self.scope.columns.iter().any(|(_, id)| column_id == *id);
                if exists {
                    self.result = Err(OptimizerError::Internal(format!(
                        "Projection: Unexpected column : {}. Input columns: {:?}",
                        column_id, self.scope.columns
                    )));
                }
                expr
            }
            ScalarExpr::ColumnName(ref column_name) => {
                let column_id = self.scope.columns.iter().find(|(name, _id)| column_name == name).map(|(_, id)| *id);
                match column_id {
                    Some(column_id) => ScalarExpr::Column(column_id),
                    None => {
                        self.result = Err(OptimizerError::Internal(format!(
                            "Projection: Unexpected column : {}. Input columns: {:?}",
                            column_name, self.scope.columns
                        )));
                        expr
                    }
                }
            }
            ScalarExpr::SubQuery(RelNode::Expr(_)) => {
                // FIXME: Return the following error when Memoization will be enabled in OptimizerTester::builder()
                // self.result = Err(OptimizerError::Internal(format!(
                //     "Use sub_query_builder/to_sub_query to construct nested sub queries"
                // )));
                expr
            }
            _ => expr,
        }
    }
}

/// A builder for aggregate operators.  
pub struct AggregateBuilder<'a> {
    builder: &'a mut OperatorBuilder,
    aggr_exprs: Vec<(AggregateFunction, String)>,
    group_exprs: Vec<String>,
}

impl AggregateBuilder<'_> {
    /// Adds aggregate function `func` with the given column as an argument.
    pub fn add_func(mut self, func: &str, column_name: &str) -> Result<Self, OptimizerError> {
        let func = AggregateFunction::try_from(func)
            .map_err(|_| OptimizerError::Argument(format!("Unknown aggregate function {}", func)))?;
        self.aggr_exprs.push((func, column_name.into()));
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
            .map(|(f, a)| {
                let mut rewriter = RewriteExprs {
                    scope: &scope,
                    result: Ok(()),
                };
                let expr = ScalarExpr::ColumnName(a);
                let expr = expr.rewrite(&mut rewriter);
                rewriter.result?;

                let name = format!("{}", f);
                let expr = ScalarExpr::Aggregate {
                    func: f,
                    args: vec![expr],
                    filter: None,
                };
                let metadata = &self.builder.metadata;
                let data_type = resolve_expr_type(&expr, metadata);

                let column_meta = ColumnMetadata::new_synthetic_column(name.clone(), data_type, Some(expr.clone()));
                let id = metadata.add_column(column_meta);

                Ok((ScalarNode::from(expr), (name, id)))
            })
            .collect();
        let aggr_exprs = aggr_exprs?;
        let (aggr_exprs, output_columns): (Vec<ScalarNode>, Vec<(String, ColumnId)>) = aggr_exprs.into_iter().unzip();
        let column_ids: Vec<ColumnId> = output_columns.iter().map(|(_, id)| *id).collect();

        let group_exprs = std::mem::take(&mut self.group_exprs);
        let group_exprs: Result<Vec<ScalarNode>, OptimizerError> = group_exprs
            .into_iter()
            .map(|name| {
                let mut rewriter = RewriteExprs {
                    scope: &scope,
                    result: Ok(()),
                };
                let expr = ScalarExpr::ColumnName(name);
                let expr = expr.rewrite(&mut rewriter);
                rewriter.result?;

                Ok(ScalarNode::from(expr))
            })
            .collect();
        let group_exprs = group_exprs?;

        let aggregate = LogicalExpr::Aggregate {
            input,
            aggr_exprs,
            group_exprs,
            columns: column_ids,
        };
        let operator = Operator::from(OperatorExpr::from(aggregate));

        Ok(OperatorBuilder::with_parent(
            self.builder,
            operator,
            OperatorScope {
                columns: output_columns,
            },
        ))
    }
}

/// No memoization.
pub struct NoMemoization;

impl MemoizationHandler for NoMemoization {
    fn memoize_rel(&self, expr: Operator) -> RelNode {
        RelNode::from(expr)
    }

    fn memoize_scalar(&self, expr: Operator) -> ScalarNode {
        ScalarNode::Expr(Box::new(expr))
    }
}

impl Debug for dyn MemoizationHandler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoizationHandler")
    }
}

/// A callback that copies operators into a memo.
#[derive(Debug)]
pub struct MemoizeWithMemo {
    // RefCell is necessary because the memo is shared by multiple operator builders.
    memo: RefCell<ExprMemo>,
}

impl MemoizeWithMemo {
    /// Creates a callback that uses the given memo.
    pub fn new(memo: ExprMemo) -> Self {
        MemoizeWithMemo {
            memo: RefCell::new(memo),
        }
    }

    /// Consumes this callback and returns the underlying memo.
    pub fn into_inner(self) -> ExprMemo {
        self.memo.into_inner()
    }
}

impl MemoizationHandler for MemoizeWithMemo {
    fn memoize_rel(&self, expr: Operator) -> RelNode {
        let mut memo = self.memo.borrow_mut();
        let (group, _) = memo.insert_group(expr);
        RelNode::Group(group)
    }

    fn memoize_scalar(&self, expr: Operator) -> ScalarNode {
        let mut memo = self.memo.borrow_mut();
        let (group, _) = memo.insert_group(expr);
        ScalarNode::Group(group)
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
    DataType::Bool
}

#[cfg(test)]
mod test {
    use crate::catalog::mutable::MutableCatalog;
    use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
    use crate::datatypes::DataType;
    use crate::error::OptimizerError;
    use crate::memo::format_memo;
    use crate::meta::{ColumnId, ColumnMetadata, MutableMetadata};
    use crate::operators::builder::{MemoizationHandler, MemoizeWithMemo, OperatorBuilder, OrderingOption};

    use crate::operators::scalar::expr::{AggregateFunction, BinaryOp};
    use crate::operators::scalar::value::ScalarValue;
    use crate::operators::scalar::ScalarExpr;
    use crate::operators::{ExprMemo, Operator, OperatorMetadata};

    use crate::operators::properties::LogicalPropertiesBuilder;
    use crate::operators::statistics::NoStatisticsBuilder;
    use crate::optimizer::SetPropertiesCallback;
    use crate::rules::testing::format_operator_tree;
    use itertools::Itertools;
    use std::rc::Rc;
    use std::sync::Arc;

    #[test]
    fn test_get() {
        let metadata = Rc::new(MutableMetadata::new());
        let memoization = memoization(metadata.clone());

        let expr = build_tree(metadata.clone(), memoization.clone(), |operator_builder| {
            operator_builder.get("A", vec!["a1"])?.build()
        })
        .unwrap();

        expect_expr(
            memoization,
            expr,
            metadata,
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
        let metadata = Rc::new(MutableMetadata::new());
        let memoization = memoization(metadata.clone());

        let expr = build_tree(metadata.clone(), memoization.clone(), |operator_builder| {
            let ord = OrderingOption::by(("a1", false));
            let tree = operator_builder.get("A", vec!["a1"])?.order_by(ord)?;

            tree.build()
        })
        .unwrap();

        //TODO: include properties
        expect_expr(
            memoization,
            expr,
            metadata,
            r#"
LogicalGet A cols=[1]
  output cols: [1]
Metadata:
  col:1 A.a1 Int32
Memo:
  00 LogicalGet A cols=[1]
"#,
        )
    }

    #[test]
    fn test_select() {
        let metadata = Rc::new(MutableMetadata::new());
        let memoization = memoization(metadata.clone());

        let expr = build_tree(metadata.clone(), memoization.clone(), |operator_builder| {
            let filter = ScalarExpr::BinaryExpr {
                lhs: Box::new(ScalarExpr::ColumnName("a1".into())),
                op: BinaryOp::Gt,
                rhs: Box::new(ScalarExpr::Scalar(ScalarValue::Int32(100))),
            };

            operator_builder.get("A", vec!["a1"])?.select(Some(filter))?.build()
        })
        .unwrap();

        expect_expr(
            memoization,
            expr,
            metadata,
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
        )
    }

    #[test]
    fn test_projection() {
        let metadata = Rc::new(MutableMetadata::new());
        let memoization = memoization(metadata.clone());

        let expr = build_tree(metadata.clone(), memoization.clone(), |operator_builder| {
            let projection_list =
                vec![ScalarExpr::ColumnName("a2".into()), ScalarExpr::Scalar(ScalarValue::Int32(100))];

            operator_builder.get("A", vec!["a1", "a2"])?.project(projection_list)?.build()
        })
        .unwrap();

        expect_expr(
            memoization,
            expr,
            metadata,
            r#"
LogicalProjection cols=[2, 3]
  input: LogicalGet A cols=[1, 2]
  output cols: [2, 3]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 ?column? Int32, expr: 100
Memo:
  01 LogicalProjection input=00 cols=[2, 3]
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_projection_of_projection() {
        let metadata = Rc::new(MutableMetadata::new());
        let memoization = memoization(metadata.clone());

        let expr = build_tree(metadata.clone(), memoization.clone(), |operator_builder| {
            let projection_list1 = vec![
                ScalarExpr::ColumnName("a2".into()),
                ScalarExpr::Scalar(ScalarValue::Int32(100)),
                ScalarExpr::ColumnName("a1".into()),
            ];

            let projection_list2 = vec![ScalarExpr::ColumnName("a2".into())];

            operator_builder
                .get("A", vec!["a1", "a2"])?
                .project(projection_list1)?
                .project(projection_list2)?
                .build()
        })
        .unwrap();

        expect_expr(
            memoization,
            expr,
            metadata,
            r#"
LogicalProjection cols=[2]
  input: LogicalProjection cols=[2, 3, 1]
      input: LogicalGet A cols=[1, 2]
  output cols: [2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 ?column? Int32, expr: 100
Memo:
  02 LogicalProjection input=01 cols=[2]
  01 LogicalProjection input=00 cols=[2, 3, 1]
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_join() {
        let metadata = Rc::new(MutableMetadata::new());
        let memoization = memoization(metadata.clone());

        let expr = build_tree(metadata.clone(), memoization.clone(), |operator_builder| {
            let left = operator_builder.clone().get("A", vec!["a1", "a2"])?;
            let right = operator_builder.get("B", vec!["b1", "b2"])?;
            let join = left.join_using(right, vec![("a1", "b2")])?;

            join.build()
        })
        .unwrap();

        expect_expr(
            memoization,
            expr,
            metadata,
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
        )
    }

    #[test]
    fn test_nested() {
        let metadata = Rc::new(MutableMetadata::new());
        let memoization = memoization(metadata.clone());

        let expr = build_tree(metadata.clone(), memoization.clone(), |operator_builder| {
            let from_a = operator_builder.get("A", vec!["a1", "a2"])?;

            let expr = ScalarExpr::Aggregate {
                func: AggregateFunction::Count,
                args: vec![ScalarExpr::ColumnName("b1".into())],
                filter: None,
            };

            let sub_query = from_a
                .sub_query_builder()
                .get("B", vec!["b1", "b2"])?
                .project(vec![expr])?
                .to_sub_query()?;

            let filter = ScalarExpr::BinaryExpr {
                lhs: Box::new(sub_query),
                op: BinaryOp::Gt,
                rhs: Box::new(ScalarExpr::Scalar(ScalarValue::Int32(37))),
            };
            let select = from_a.select(Some(filter))?;

            select.build()
        })
        .unwrap();

        expect_expr(
            memoization,
            expr,
            metadata,
            r#"
LogicalSelect
  input: LogicalGet A cols=[1, 2]
  filter: Expr SubQuery 01 > 37
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32, expr: count(col:3)
Memo:
  04 LogicalSelect input=02 filter=03
  03 Expr SubQuery 01 > 37
  02 LogicalGet A cols=[1, 2]
  01 LogicalProjection input=00 cols=[5]
  00 LogicalGet B cols=[3, 4]
"#,
        )
    }

    #[test]
    fn test_prohibit_nested_sub_queries_not_created_via_sub_query_builder() {
        let metadata = Rc::new(MutableMetadata::new());
        let memoization = memoization(metadata.clone());

        let _err = build_tree(metadata, memoization, |operator_builder| {
            let _from_a = operator_builder.clone().get("A", vec!["a1", "a2"])?;

            let _expr = ScalarExpr::Aggregate {
                func: AggregateFunction::Count,
                args: vec![ScalarExpr::ColumnName("b1".into())],
                filter: None,
            };

            let from_b = operator_builder.get("B", vec!["b1"])?;
            let _sub_query = from_b.to_sub_query()?;

            panic!("Created sub query from a non sub query build")
        });
    }

    fn build_tree<F>(
        metadata: OperatorMetadata,
        memoization: Rc<dyn MemoizationHandler>,
        mut f: F,
    ) -> Result<Operator, OptimizerError>
    where
        F: FnMut(OperatorBuilder) -> Result<Operator, OptimizerError>,
    {
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
        let operator_builder = OperatorBuilder::new(memoization, catalog, metadata);
        (f)(operator_builder)
    }

    fn memoization(metadata: OperatorMetadata) -> Rc<MemoizeWithMemo> {
        let properties_builder = Rc::new(LogicalPropertiesBuilder::new(NoStatisticsBuilder));
        Rc::new(MemoizeWithMemo::new(ExprMemo::with_callback(
            metadata,
            Rc::new(SetPropertiesCallback::new(properties_builder)),
        )))
    }

    fn expect_expr(memoization: Rc<MemoizeWithMemo>, expr: Operator, metadata: Rc<MutableMetadata>, expected: &str) {
        let mut buf = String::new();
        buf.push('\n');

        let memoization = Rc::try_unwrap(memoization).unwrap();
        let mut memo = memoization.into_inner();
        let (_, expr) = memo.insert_group(expr);

        buf.push_str(format_operator_tree(expr.mexpr()).as_str());
        buf.push('\n');

        buf.push_str(format!("  output cols: {:?}\n", expr.props().logical().output_columns()).as_str());
        buf.push_str("Metadata:\n");

        let columns: Vec<(ColumnId, ColumnMetadata)> =
            metadata.get_columns().into_iter().sorted_by(|a, b| a.0.cmp(&b.0)).collect();

        for (id, column) in columns {
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
}
