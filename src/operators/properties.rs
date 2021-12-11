use crate::error::OptimizerError;
use crate::meta::{ColumnId, MetadataRef};
use crate::operators::relational::join::JoinCondition;
use crate::operators::relational::logical::{LogicalExpr, SetOperator};
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::relational::{RelExpr, RelNode};
use crate::operators::scalar::expr::ExprVisitor;
use crate::operators::scalar::{ScalarExpr, ScalarNode};
use crate::operators::statistics::StatisticsBuilder;
use crate::operators::{Operator, OperatorExpr, Properties};
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use std::fmt::Debug;

/// Provides logical properties for memo expressions.
pub trait PropertiesProvider: Debug {
    /// Builds properties for the given expression `expr`. `manual_props` contains properties that
    /// has been added by hand to modify statistics or other properties of the given expression to simplify testing.  
    fn build_properties(
        &self,
        expr: &Operator,
        manual_props: Properties,
        metadata: MetadataRef,
    ) -> Result<Properties, OptimizerError>;
}

#[derive(Debug)]
pub struct LogicalPropertiesBuilder {
    statistics: Box<dyn StatisticsBuilder>,
}

impl LogicalPropertiesBuilder {
    pub fn new(statistics: Box<dyn StatisticsBuilder>) -> Self {
        LogicalPropertiesBuilder { statistics }
    }
}

impl LogicalPropertiesBuilder {
    /// Builds logical properties for a get operator.
    pub fn build_get(&self, _source: &str, columns: &[ColumnId]) -> Result<LogicalProperties, OptimizerError> {
        let output_columns = columns.iter().copied().collect();
        Ok(LogicalProperties::new(output_columns, None))
    }

    /// Builds logical properties for the given select operator.
    pub fn build_select(
        &self,
        input: &RelNode,
        filter: Option<&ScalarNode>,
    ) -> Result<LogicalProperties, OptimizerError> {
        // FIXME: Is this validation really necessary?
        if let Some(filter) = filter {
            expect_columns("Select", filter, input)?;
        };
        Ok(LogicalProperties::new(input.props().logical().output_columns.to_vec(), None))
    }

    /// Builds logical properties for a projection operator.
    pub fn build_projection(
        &self,
        _input: &RelNode,
        columns: &[ColumnId],
    ) -> Result<LogicalProperties, OptimizerError> {
        // FIXME: Is this validation really necessary?
        //let input_columns = input.props().logical().output_columns();
        // for column_id in columns {
        //     if !input_columns.contains(column_id) {
        //         return Err(OptimizerError::Argument(format!(
        //             "Output column {} does not exist. Input: {:?}",
        //             column_id, input_columns
        //         )));
        //     }
        // }
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

// Move to OperatorBuilder
fn expect_columns(operator: &str, expr: &ScalarNode, input: &RelNode) -> Result<(), OptimizerError> {
    struct ExpectColumns<'a> {
        operator: &'a str,
        input_columns: &'a [ColumnId],
        result: Result<(), OptimizerError>,
    }

    impl ExprVisitor<RelNode> for ExpectColumns<'_> {
        fn post_visit(&mut self, expr: &ScalarExpr) {
            if self.result.is_err() {
                return;
            }
            if let ScalarExpr::Column(id) = expr {
                if !self.input_columns.contains(id) {
                    self.result = Err(OptimizerError::Internal(format!(
                        "{}: Unexpected column {}. Input columns: {:?}",
                        self.operator, id, self.input_columns
                    )))
                }
            }
        }
    }
    let mut visitor = ExpectColumns {
        operator,
        input_columns: input.props().logical().output_columns(),
        result: Ok(()),
    };
    expr.expr().accept(&mut visitor);
    visitor.result
}

impl PropertiesProvider for LogicalPropertiesBuilder {
    fn build_properties(
        &self,
        expr: &Operator,
        manual_props: Properties,
        metadata: MetadataRef,
    ) -> Result<Properties, OptimizerError> {
        let Properties { logical, required } = manual_props;
        let statistics = logical.statistics;
        match expr.expr() {
            OperatorExpr::Relational(RelExpr::Logical(expr)) => {
                let LogicalProperties {
                    output_columns,
                    statistics: _statistics,
                } = match &**expr {
                    LogicalExpr::Projection { input, columns, .. } => self.build_projection(input, columns),
                    LogicalExpr::Select { input, filter } => self.build_select(input, filter.as_ref()),
                    LogicalExpr::Aggregate { input, columns, .. } => self.build_aggregate(input, columns),
                    LogicalExpr::Join { left, right, condition } => self.build_join(left, right, condition),
                    LogicalExpr::Get { source, columns } => self.build_get(source, columns),
                    LogicalExpr::Union {
                        left,
                        right,
                        all,
                        columns,
                    } => self.build_union(left, right, *all, columns),
                    LogicalExpr::Intersect {
                        left,
                        right,
                        all,
                        columns,
                    } => self.build_intersect(left, right, *all, columns),
                    LogicalExpr::Except {
                        left,
                        right,
                        all,
                        columns,
                    } => self.build_except(left, right, *all, columns),
                }?;
                let logical = LogicalProperties::new(output_columns, None);
                let statistics = self.statistics.build_statistics(expr, &logical, metadata)?;
                let logical = if let Some(statistics) = statistics {
                    logical.with_statistics(statistics)
                } else {
                    logical
                };
                Ok(Properties::new(logical, required))
            }
            OperatorExpr::Relational(RelExpr::Physical(expr)) => {
                // Only enforcer operators are copied into a memo as groups.
                // Other physical expressions are copied as members of existing groups.
                if let PhysicalExpr::Sort { input, .. } = &**expr {
                    // Enforcer returns the same logical properties as its input
                    let logical = input.props().logical().clone();
                    // Sort operator does not have any ordering requirements
                    let required = input.props().required().clone().without_ordering();

                    Ok(Properties::new(logical, required))
                } else {
                    Err(OptimizerError::Internal(format!("Physical expressions are not allowed in an operator tree")))
                }
            }
            OperatorExpr::Scalar(_) => {
                assert!(required.is_empty(), "Physical properties can not be set for scalar expressions");
                let logical = LogicalProperties::new(Vec::new(), statistics);
                Ok(Properties::new(logical, PhysicalProperties::none()))
            }
        }
    }
}
