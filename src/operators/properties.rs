use std::fmt::{Debug, Formatter};

use crate::error::OptimizerError;
use crate::memo::Props;
use crate::meta::{ColumnId, MetadataRef};
use crate::operators::relational::join::{JoinCondition, JoinType};
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalDistinct, LogicalEmpty, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect,
    LogicalJoin, LogicalLimit, LogicalOffset, LogicalProjection, LogicalSelect, LogicalUnion, LogicalValues,
    LogicalWindowAggregate, SetOperator,
};
use crate::operators::relational::physical::{PhysicalExpr, Sort};
use crate::operators::relational::{RelExpr, RelNode};
use crate::operators::scalar::{exprs, ScalarNode};
use crate::operators::{OperatorExpr, OuterScope, Properties, RelationalProperties};
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use crate::statistics::StatisticsBuilder;

pub struct PropertiesContext<'a> {
    pub outer_columns: &'a [ColumnId],
    pub properties: Properties,
}

/// Provides logical properties for memo expressions.
pub trait PropertiesProvider {
    /// Builds properties for the given expression `expr`. `provided_props` contains properties that
    /// has been specified for the given expression in an operator tree.   
    fn build_properties(
        &self,
        expr: &OperatorExpr,
        scope: &OuterScope,
        provided_props: Properties,
        metadata: MetadataRef,
    ) -> Result<Properties, OptimizerError>;
}

/// Builds logical properties shared by expressions in a memo group.
pub struct LogicalPropertiesBuilder<T> {
    statistics: T,
}

impl<T> LogicalPropertiesBuilder<T>
where
    T: StatisticsBuilder,
{
    /// Creates a new logical properties builder.
    pub fn new(statistics: T) -> Self {
        LogicalPropertiesBuilder { statistics }
    }

    /// Builds logical properties for a get operator.
    pub fn build_get(&self, _source: &str, columns: &[ColumnId]) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();
        Ok(LogicalProperties::new(output_columns, None))
    }

    /// Builds logical properties for the given select operator.
    pub fn build_select(
        &self,
        input: &RelNode,
        filter: Option<&ScalarNode>,
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = input.props().logical().output_columns.to_vec();

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        outer_columns.add_input(input);
        if let Some(expr) = filter {
            outer_columns.add_expr(expr)
        }
        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    /// Builds logical properties for a projection operator.
    pub fn build_projection(
        &self,
        input: &RelNode,
        columns: &[ColumnId],
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        outer_columns.add_input(input);
        outer_columns.add_projection(columns);
        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    /// Builds logical properties for a join operator.
    pub fn build_join(
        &self,
        join_type: &JoinType,
        left: &RelNode,
        right: &RelNode,
        condition: &JoinCondition,
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = match join_type {
            JoinType::LeftSemi | JoinType::Anti => left.props().logical().output_columns().to_vec(),
            JoinType::RightSemi => right.props().logical().output_columns().to_vec(),
            _ => {
                let mut output_columns = left.props().logical().output_columns().to_vec();
                output_columns.extend_from_slice(right.props().logical().output_columns());
                output_columns
            }
        };

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        outer_columns.add_input(left);
        outer_columns.add_input(right);
        match condition {
            JoinCondition::Using(using) => outer_columns.add_expr(&using.get_expr()),
            JoinCondition::On(on_expr) => outer_columns.add_expr(on_expr.expr()),
        };
        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    /// Builds logical properties for the given aggregate operator.
    pub fn build_aggregate(
        &self,
        input: &RelNode,
        columns: &[ColumnId],
        group_exprs: &[ScalarNode],
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        outer_columns.add_input(input);
        outer_columns.add_projection(columns);
        for group_by in group_exprs {
            outer_columns.add_expr(group_by);
        }

        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    /// Builds logical properties for the given window aggregate operator.
    pub fn build_window_aggregate(
        &self,
        input: &RelNode,
        window_expr: &ScalarNode,
        columns: &[ColumnId],
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        outer_columns.add_input(input);
        outer_columns.add_expr(window_expr);
        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    /// Builds logical properties for the given set operator (Union, Intersect or Except).
    #[allow(clippy::too_many_arguments)]
    pub fn build_set_op(
        &self,
        set_op: SetOperator,
        left: &RelNode,
        right: &RelNode,
        all: bool,
        columns: &[ColumnId],
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        match set_op {
            SetOperator::Union => self.build_union(left, right, all, columns, metadata, outer_scope),
            SetOperator::Intersect => self.build_intersect(left, right, all, columns, metadata, outer_scope),
            SetOperator::Except => self.build_except(left, right, all, columns, metadata, outer_scope),
        }
    }

    /// Builds logical properties for a values operator.
    pub fn build_values(
        &self,
        values: &[ScalarNode],
        columns: &[ColumnId],
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        for value_list in values {
            outer_columns.add_expr(value_list);
        }
        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    /// Builds logical properties for an empty operator.
    pub fn build_empty(&self) -> Result<LogicalProperties, OptimizerError> {
        Ok(LogicalProperties::empty())
    }

    fn build_union(
        &self,
        left: &RelNode,
        right: &RelNode,
        _all: bool,
        columns: &[ColumnId],
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        outer_columns.add_input(left);
        outer_columns.add_input(right);
        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    fn build_intersect(
        &self,
        left: &RelNode,
        right: &RelNode,
        _all: bool,
        columns: &[ColumnId],
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        outer_columns.add_input(left);
        outer_columns.add_input(right);
        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    fn build_except(
        &self,
        left: &RelNode,
        right: &RelNode,
        _all: bool,
        columns: &[ColumnId],
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        outer_columns.add_input(left);
        outer_columns.add_input(right);
        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    fn build_distinct(
        &self,
        input: &RelNode,
        on_expr: Option<&ScalarNode>,
        metadata: MetadataRef,
        outer_scope: &OuterScope,
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = input.props().logical().output_columns.to_vec();

        let mut outer_columns = OuterColumnsBuilder::new(outer_scope, metadata);
        outer_columns.add_input(input);
        if let Some(expr) = on_expr {
            outer_columns.add_expr(expr);
        }
        let outer_columns = outer_columns.build();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }

    fn build_limit_offset(&self, input: &RelNode) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = input.props().logical().output_columns.to_vec();
        let outer_columns = input.props().logical.outer_columns.to_vec();

        Ok(LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries: false,
            statistics: None,
        })
    }
}

impl<T> Debug for LogicalPropertiesBuilder<T>
where
    T: StatisticsBuilder + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogicalPropertiesBuilder").field("statistics", &self.statistics).finish()
    }
}

impl<T> PropertiesProvider for LogicalPropertiesBuilder<T>
where
    T: StatisticsBuilder,
{
    fn build_properties(
        &self,
        expr: &OperatorExpr,
        scope: &OuterScope,
        props: Properties,
        metadata: MetadataRef,
    ) -> Result<Properties, OptimizerError> {
        match expr {
            OperatorExpr::Relational(RelExpr::Logical(expr)) => {
                let RelationalProperties { physical, .. } = props.to_relational();

                let LogicalProperties {
                    output_columns,
                    outer_columns,
                    has_correlated_subqueries,
                    // build_xxx methods return logical properties without statistics.
                    statistics: _,
                } = match &**expr {
                    LogicalExpr::Projection(LogicalProjection { input, columns, .. }) => {
                        self.build_projection(input, columns, metadata.clone(), scope)
                    }
                    LogicalExpr::Select(LogicalSelect { input, filter }) => {
                        self.build_select(input, filter.as_ref(), metadata.clone(), scope)
                    }
                    LogicalExpr::Aggregate(LogicalAggregate {
                        input,
                        columns,
                        group_exprs,
                        ..
                    }) => self.build_aggregate(input, columns, group_exprs, metadata.clone(), scope),
                    LogicalExpr::WindowAggregate(LogicalWindowAggregate {
                        input,
                        window_expr,
                        columns,
                    }) => self.build_window_aggregate(input, window_expr, columns, metadata.clone(), scope),
                    LogicalExpr::Join(LogicalJoin {
                        join_type,
                        left,
                        right,
                        condition,
                    }) => self.build_join(join_type, left, right, condition, metadata.clone(), scope),
                    LogicalExpr::Get(LogicalGet { source, columns }) => self.build_get(source, columns),
                    LogicalExpr::Union(LogicalUnion {
                        left,
                        right,
                        all,
                        columns,
                    }) => self.build_union(left, right, *all, columns, metadata.clone(), scope),
                    LogicalExpr::Intersect(LogicalIntersect {
                        left,
                        right,
                        all,
                        columns,
                    }) => self.build_intersect(left, right, *all, columns, metadata.clone(), scope),
                    LogicalExpr::Except(LogicalExcept {
                        left,
                        right,
                        all,
                        columns,
                    }) => self.build_except(left, right, *all, columns, metadata.clone(), scope),
                    LogicalExpr::Distinct(LogicalDistinct { input, on_expr, .. }) => {
                        self.build_distinct(input, on_expr.as_ref(), metadata.clone(), scope)
                    }
                    LogicalExpr::Limit(LogicalLimit { input, .. })
                    | LogicalExpr::Offset(LogicalOffset { input, .. }) => self.build_limit_offset(input),
                    LogicalExpr::Values(LogicalValues { values, columns }) => {
                        self.build_values(values, columns, metadata.clone(), scope)
                    }
                    LogicalExpr::Empty(LogicalEmpty { .. }) => self.build_empty(),
                }?;
                let logical = LogicalProperties {
                    output_columns,
                    outer_columns,
                    has_correlated_subqueries,
                    statistics: None,
                };
                let statistics = self.statistics.build_statistics(expr, &logical, metadata)?;
                let logical = if let Some(statistics) = statistics {
                    logical.with_statistics(statistics)
                } else {
                    logical
                };
                Ok(Properties::Relational(RelationalProperties { logical, physical }))
            }
            OperatorExpr::Relational(RelExpr::Physical(expr)) => {
                // Only enforcer operators are copied into a memo as groups.
                // Other physical expressions are copied as members of existing groups.
                if let PhysicalExpr::Sort(Sort { input, .. }) = &**expr {
                    // Enforcer returns the same logical properties as its input
                    let logical = input.props().logical().clone();
                    // Sort operator does not have any ordering requirements.
                    let PhysicalProperties { required, presentation } = input.props().physical().clone();
                    let physical = PhysicalProperties {
                        required: required.map(|r| r.without_ordering()),
                        // Use the same presentation as the input operator.
                        presentation,
                    };

                    Ok(Properties::Relational(RelationalProperties { logical, physical }))
                } else {
                    Err(OptimizerError::Internal(
                        "Physical expressions are not allowed in an operator tree".to_string(),
                    ))
                }
            }
            OperatorExpr::Scalar(_) => {
                let scalar = props.to_scalar();
                Ok(Properties::Scalar(scalar))
            }
        }
    }
}

struct OuterColumnsBuilder<'a> {
    scope: &'a OuterScope,
    metadata: MetadataRef<'a>,
    input_columns: Vec<ColumnId>,
    result: Vec<ColumnId>,
}

impl<'a> OuterColumnsBuilder<'a> {
    fn new(scope: &'a OuterScope, metadata: MetadataRef<'a>) -> Self {
        OuterColumnsBuilder {
            scope,
            metadata,
            input_columns: Vec::new(),
            result: Vec::new(),
        }
    }

    /// Adds outer columns used by the input operator.
    fn add_input(&mut self, input: &RelNode) {
        let output_columns = &input.props().logical.output_columns;
        self.input_columns.extend_from_slice(output_columns);

        let input_outer_columns = &input.props().logical.outer_columns;
        for col_id in input_outer_columns {
            assert!(
                self.scope.outer_columns.contains(col_id),
                "Input operator contains unexpected column in outer columns. Outer scope: {:?}. Input: {:?}",
                self.scope.outer_columns,
                input_outer_columns
            );
            self.add_outer_column(*col_id);
        }
    }

    /// Adds outer columns referenced by projection expressions.
    fn add_projection(&mut self, columns: &[ColumnId]) {
        // We can not borrow the metadata and modify the outer columns at the same time.
        let metadata = self.metadata.clone();
        columns
            .iter()
            .flat_map(|col_id| {
                let meta = metadata.get_column(col_id);
                if let Some(expr) = meta.expr() {
                    exprs::collect_columns(expr).into_iter()
                } else {
                    vec![*col_id].into_iter()
                }
            })
            .for_each(|col_id| {
                self.add_if_outer(col_id);
            });
    }

    /// Adds outer columns referenced by the given expressions.
    fn add_expr(&mut self, expr: &ScalarNode) {
        let columns = exprs::collect_columns(expr.expr());
        for col_id in columns {
            self.add_if_outer(col_id);
        }
    }

    fn build(self) -> Vec<ColumnId> {
        self.result
    }

    fn add_if_outer(&mut self, id: ColumnId) {
        if self.scope.outer_columns.contains(&id) {
            // Do not add an outer column if the same column is provided by the input operator.
            if !self.input_columns.contains(&id) {
                self.add_outer_column(id);
            }
        }
    }

    fn add_outer_column(&mut self, id: ColumnId) {
        if !self.result.contains(&id) {
            self.result.push(id);
        }
    }
}
