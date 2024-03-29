//! Logical properties. See [LogicalProperties].

use crate::meta::ColumnId;
use crate::statistics::Statistics;

/// Properties that are identical across all expressions within a memo group.
#[derive(Debug, Clone)]
pub struct LogicalProperties {
    // FIXME: use a bit set instead of a vec.
    pub output_columns: Vec<ColumnId>,
    /// Columns from the outer scope used by an operator.
    pub outer_columns: Vec<ColumnId>,
    pub has_correlated_subqueries: bool,
    //FIXME: Make this non-optional when logical properties builder API becomes stable.
    pub statistics: Option<Statistics>,
}

impl LogicalProperties {
    /// Creates a new instance of `LogicalProperties` with the specified properties.
    pub fn new(output_columns: Vec<ColumnId>, statistics: Option<Statistics>) -> Self {
        LogicalProperties {
            output_columns,
            outer_columns: Vec::with_capacity(0),
            has_correlated_subqueries: false,
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
            outer_columns: Vec::with_capacity(0),
            has_correlated_subqueries: false,
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

    /// Returns new logical properties with the given statistics.
    pub fn with_statistics(self, statistics: Statistics) -> Self {
        let LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries,
            statistics: _statistics,
        } = self;
        LogicalProperties {
            output_columns,
            outer_columns,
            has_correlated_subqueries,
            statistics: Some(statistics),
        }
    }
}

impl Default for LogicalProperties {
    fn default() -> Self {
        LogicalProperties::empty()
    }
}
