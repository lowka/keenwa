//FIXME: Statistics must be a top level module.

use crate::error::OptimizerError;
use crate::meta::MetadataRef;
use crate::operators::relational::logical::LogicalExpr;
use crate::properties::logical::LogicalProperties;
use crate::properties::statistics::Statistics;
use std::fmt::Debug;

pub mod simple;

/// Provide statistics
pub trait StatisticsBuilder {
    /// Builds statistics for the given expression.
    fn build_statistics(
        &self,
        expr: &LogicalExpr,
        logical_properties: &LogicalProperties,
        metadata: MetadataRef,
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
        _metadata: MetadataRef,
    ) -> Result<Option<Statistics>, OptimizerError> {
        Ok(None)
    }
}
