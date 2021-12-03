use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::{OperatorExpr, Properties, RelExpr};
use crate::properties::physical::PhysicalProperties;
use crate::properties::statistics::{Statistics, StatisticsBuilder};
use std::fmt::Debug;

/// Properties that are identical across all expressions within a memo group.
#[derive(Debug, Clone)]
pub struct LogicalProperties {
    // FIXME: use a bit set instead of a vec.
    pub(crate) output_columns: Vec<ColumnId>,
    //FIXME: Make this non-optional when logical properties builder API becomes stable.
    pub(crate) statistics: Option<Statistics>,
}

impl LogicalProperties {
    /// Creates a new instance of `LogicalProperties` with the specified properties.
    pub fn new(output_columns: Vec<ColumnId>, statistics: Option<Statistics>) -> Self {
        LogicalProperties {
            output_columns,
            statistics,
        }
    }

    /// Creates an empty `LogicalProperties` object.
    ///
    /// Right now this method is used for physical expressions because physical expressions do not use logical properties
    /// of other expressions directly. They instead use logical properties of the memo groups of their child expressions.
    pub fn empty() -> Self {
        LogicalProperties {
            output_columns: Vec::with_capacity(0),
            statistics: None,
        }
    }

    /// Returns the columns produced by the expression.
    pub fn output_columns(&self) -> &[ColumnId] {
        &self.output_columns
    }

    /// Returns statistics for the expression.
    pub fn statistics(&self) -> Option<&Statistics> {
        self.statistics.as_ref()
    }
}

impl Default for LogicalProperties {
    fn default() -> Self {
        LogicalProperties::empty()
    }
}

/// Provides logical properties for memo expressions.
pub trait PropertiesProvider: Debug {
    /// Builds properties for the given expression `expr`. `manual_props` contains properties that
    /// has been added by hand to modify statistics or other properties of the given expression to simplify testing.  
    fn build_properties(&self, expr: &OperatorExpr, manual_props: Properties) -> Result<Properties, OptimizerError>;
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

impl PropertiesProvider for LogicalPropertiesBuilder {
    fn build_properties(&self, expr: &OperatorExpr, manual_props: Properties) -> Result<Properties, OptimizerError> {
        let Properties { logical, required } = manual_props;
        let output_columns = logical.output_columns;
        let statistics = logical.statistics;
        match expr {
            OperatorExpr::Relational(RelExpr::Logical(expr)) => {
                let statistics = self.statistics.build_statistics(expr, statistics.as_ref())?;
                let logical = LogicalProperties::new(output_columns, statistics);
                Ok(Properties::new(logical, required))
            }
            OperatorExpr::Relational(RelExpr::Physical(_)) => {
                //TODO: compute statistics for physical expressions
                let logical = LogicalProperties::new(output_columns, statistics);
                Ok(Properties::new(logical, required))
            }
            OperatorExpr::Scalar(_) => {
                assert!(required.is_empty(), "Physical properties can not be set for scalar expressions");
                let logical = LogicalProperties::new(output_columns, statistics);
                Ok(Properties::new(logical, PhysicalProperties::none()))
            }
        }
    }
}
