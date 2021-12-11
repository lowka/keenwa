use crate::catalog::{Catalog, CatalogRef};
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::relational::join::JoinCondition;
use crate::operators::relational::logical::{LogicalExpr, SetOperator};
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::properties::logical::LogicalProperties;
use crate::properties::statistics::Statistics;
use std::fmt::Debug;

pub mod simple;

/// Provide statistics
pub trait StatisticsBuilder: Debug {
    /// Builds statistics for the given expression.
    // Pass reference to a metadata.
    // MemoExprCallback should accept context
    fn build_statistics(
        &self,
        expr: &LogicalExpr,
        _logical_properties: &LogicalProperties,
    ) -> Result<Option<Statistics>, OptimizerError>;
}

/// Statistics builder that provides no statistics.
#[derive(Debug)]
pub struct NoStatisticsBuilder;

impl StatisticsBuilder for NoStatisticsBuilder {
    fn build_statistics(
        &self,
        _expr: &LogicalExpr,
        _logical_properties: &LogicalProperties,
    ) -> Result<Option<Statistics>, OptimizerError> {
        Ok(None)
    }
}
