use crate::error::OptimizerError;
use crate::meta::{ColumnId, MetadataRef};
use crate::operators::relational::join::JoinCondition;
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalEmpty, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin,
    LogicalProjection, LogicalSelect, LogicalUnion, SetOperator,
};
use crate::operators::relational::physical::{PhysicalExpr, Sort};
use crate::operators::relational::{RelExpr, RelNode};
use crate::operators::scalar::ScalarNode;
use crate::operators::{OperatorExpr, Properties, RelationalProperties};
use crate::properties::logical::LogicalProperties;
use crate::statistics::StatisticsBuilder;
use std::fmt::{Debug, Formatter};

/// Provides logical properties for memo expressions.
pub trait PropertiesProvider {
    /// Builds properties for the given expression `expr`. `manual_props` contains properties that
    /// has been added by hand to modify statistics or other properties of the given expression to simplify testing.  
    fn build_properties(
        &self,
        expr: &OperatorExpr,
        manual_props: Properties,
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
        let output_columns = columns.iter().copied().collect();
        Ok(LogicalProperties::new(output_columns, None))
    }

    /// Builds logical properties for the given select operator.
    pub fn build_select(
        &self,
        input: &RelNode,
        _filter: Option<&ScalarNode>,
    ) -> Result<LogicalProperties, OptimizerError> {
        Ok(LogicalProperties::new(input.props().logical().output_columns.to_vec(), None))
    }

    /// Builds logical properties for a projection operator.
    pub fn build_projection(
        &self,
        _input: &RelNode,
        columns: &[ColumnId],
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();
        Ok(LogicalProperties::new(output_columns, None))
    }

    /// Builds logical properties for a join operator.
    pub fn build_join(
        &self,
        left: &RelNode,
        right: &RelNode,
        _condition: &JoinCondition,
    ) -> Result<LogicalProperties, OptimizerError> {
        let mut output_columns = left.props().logical().output_columns().to_vec();
        output_columns.extend_from_slice(right.props().logical().output_columns());

        Ok(LogicalProperties::new(output_columns, None))
    }

    /// Builds logical properties for the given aggregate operator.
    pub fn build_aggregate(&self, _input: &RelNode, columns: &[ColumnId]) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        Ok(LogicalProperties::new(output_columns, None))
    }

    /// Builds logical properties for the given set operator (Union, Intersect or Except).
    pub fn build_set_op(
        &self,
        set_op: SetOperator,
        left: &RelNode,
        right: &RelNode,
        all: bool,
        columns: &[ColumnId],
    ) -> Result<LogicalProperties, OptimizerError> {
        match set_op {
            SetOperator::Union => self.build_union(left, right, all, columns),
            SetOperator::Intersect => self.build_intersect(left, right, all, columns),
            SetOperator::Except => self.build_except(left, right, all, columns),
        }
    }

    /// Builds logical properties for an empty operator.
    pub fn build_empty(&self) -> Result<LogicalProperties, OptimizerError> {
        Ok(LogicalProperties::empty())
    }

    fn build_union(
        &self,
        _left: &RelNode,
        _right: &RelNode,
        _all: bool,
        columns: &[ColumnId],
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        Ok(LogicalProperties::new(output_columns, None))
    }

    fn build_intersect(
        &self,
        _left: &RelNode,
        _right: &RelNode,
        _all: bool,
        columns: &[ColumnId],
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        Ok(LogicalProperties::new(output_columns, None))
    }

    fn build_except(
        &self,
        _left: &RelNode,
        _right: &RelNode,
        _all: bool,
        columns: &[ColumnId],
    ) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.to_vec();

        Ok(LogicalProperties::new(output_columns, None))
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
        manual_props: Properties,
        metadata: MetadataRef,
    ) -> Result<Properties, OptimizerError> {
        match expr {
            OperatorExpr::Relational(RelExpr::Logical(expr)) => {
                let RelationalProperties { required, .. } = match manual_props {
                    Properties::Relational(props) => props,
                    Properties::Scalar(_) => return Err(OptimizerError::Internal("a".into())),
                };
                let LogicalProperties {
                    output_columns,
                    // build_xxx methods return logical properties without statistics.
                    statistics: _statistics,
                } = match &**expr {
                    LogicalExpr::Projection(LogicalProjection { input, columns, .. }) => {
                        self.build_projection(input, columns)
                    }
                    LogicalExpr::Select(LogicalSelect { input, filter }) => self.build_select(input, filter.as_ref()),
                    LogicalExpr::Aggregate(LogicalAggregate { input, columns, .. }) => {
                        self.build_aggregate(input, columns)
                    }
                    LogicalExpr::Join(LogicalJoin { left, right, condition }) => {
                        self.build_join(left, right, condition)
                    }
                    LogicalExpr::Get(LogicalGet { source, columns }) => self.build_get(source, columns),
                    LogicalExpr::Union(LogicalUnion {
                        left,
                        right,
                        all,
                        columns,
                    }) => self.build_union(left, right, *all, columns),
                    LogicalExpr::Intersect(LogicalIntersect {
                        left,
                        right,
                        all,
                        columns,
                    }) => self.build_intersect(left, right, *all, columns),
                    LogicalExpr::Except(LogicalExcept {
                        left,
                        right,
                        all,
                        columns,
                    }) => self.build_except(left, right, *all, columns),
                    LogicalExpr::Empty(LogicalEmpty { .. }) => self.build_empty(),
                }?;
                let logical = LogicalProperties::new(output_columns, None);
                let statistics = self.statistics.build_statistics(expr, &logical, metadata)?;
                let logical = if let Some(statistics) = statistics {
                    logical.with_statistics(statistics)
                } else {
                    logical
                };
                Ok(Properties::new_relational_properties(logical, required))
            }
            OperatorExpr::Relational(RelExpr::Physical(expr)) => {
                // Only enforcer operators are copied into a memo as groups.
                // Other physical expressions are copied as members of existing groups.
                if let PhysicalExpr::Sort(Sort { input, .. }) = &**expr {
                    // Enforcer returns the same logical properties as its input
                    let logical = input.props().logical().clone();
                    // Sort operator does not have any ordering requirements
                    let required = input.props().required().clone().without_ordering();

                    Ok(Properties::new_relational_properties(logical, required))
                } else {
                    Err(OptimizerError::Internal(
                        "Physical expressions are not allowed in an operator tree".to_string(),
                    ))
                }
            }
            OperatorExpr::Scalar(_) => match manual_props {
                Properties::Relational(_) => {
                    Err(OptimizerError::Internal("Scalar expression with relational properties".into()))
                }
                Properties::Scalar(scalar) => Ok(Properties::new_scalar_properties(scalar.nested_sub_queries)),
            },
        }
    }
}
