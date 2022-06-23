//! Statistics.

use crate::error::OptimizerError;
use crate::meta::MetadataRef;
use crate::operators::relational::logical::LogicalExpr;
use crate::properties::logical::LogicalProperties;

pub mod simple;

/// The estimated number of rows returned by an operator.
pub type RowCount = f64;

/// The portion of rows that match a predicate. The valid range of selectivity is `[0.0; 1.0]`.
pub type Selectivity = f64;

/// Statistics associated with an operator.
///
/// * Statistics for most relational operators should be created from [RowCount] via `From` conversion.
///
/// * Statistics for `Restriction`/`Select`/`Filter` operators must be created by
/// providing both [RowCount] and [Selectivity] and the former must already include the latter.
/// For example: If `SELECT FROM a WHERE a1 > 10` produces 100 rows and
/// selectivity of `a1 > 10` is `0.1` then `Statistics` object is going have
/// `row_count` of `10` (`100*0.1`) and `selectivity` of `0.1`.
///
#[derive(Debug, Clone)]
pub struct Statistics {
    row_count: RowCount,
    selectivity: Selectivity,
}

impl Statistics {
    /// The default value of selectivity statistics.
    pub const DEFAULT_SELECTIVITY: Selectivity = 1.0;

    /// Creates new statistics with the given row count and selectivity.
    ///
    /// # Panics
    ///
    /// This method panics if the row_count is negative or the selectivity lies outside of `[0.0, 1.0]` bounds.
    pub fn new(row_count: f64, selectivity: f64) -> Self {
        assert!(row_count >= 0f64, "row_count must be non negative");
        assert!(
            (0f64..=Self::DEFAULT_SELECTIVITY).contains(&selectivity),
            "selectivity must be within [0.0, 1.0] range but got: {}",
            selectivity
        );
        Statistics { row_count, selectivity }
    }

    /// Creates a new statistics with selectivity set to the given value.
    ///
    /// # Panics
    ///
    /// This method panics if the given selectivity lies outside of `[0.0, 1.0]` bounds.
    pub fn from_selectivity(selectivity: f64) -> Self {
        Statistics::new(0.0, selectivity)
    }

    /// Creates a new statistics with row_count set to the given value.
    ///
    /// # Panics
    ///
    /// This method panics if row_count is negative.
    pub fn from_row_count(row_count: f64) -> Self {
        Statistics::new(row_count, Self::DEFAULT_SELECTIVITY)
    }

    /// The estimated number of rows returned by an operator.
    pub fn row_count(&self) -> RowCount {
        self.row_count
    }

    /// The selectivity of a predicate.
    pub fn selectivity(&self) -> Selectivity {
        self.selectivity
    }
}

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
