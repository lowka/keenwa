use crate::catalog::CatalogRef;
use std::fmt::Debug;

use crate::operators::logical::LogicalExpr;

/// The number of rows returned by an operator in case when no statistics is available.
pub const UNKNOWN_ROW_COUNT: f64 = 1000f64;

/// Statistics associated with an operator.
#[derive(Debug, Clone)]
pub struct Statistics {
    row_count: f64,
    selectivity: f64,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            row_count: UNKNOWN_ROW_COUNT,
            selectivity: 1.0,
        }
    }
}

impl Statistics {
    pub fn new(row_count: f64, selectivity: f64) -> Self {
        assert!(row_count >= 0f64, "row_count must be non negative");
        assert!(
            (0f64..=1.0f64).contains(&selectivity),
            "selectivity must be within [0.0, 1.0] range but got: {}",
            selectivity
        );
        Statistics { row_count, selectivity }
    }

    /// Creates a new statistics with selectivity set to the given value.
    pub fn from_selectivity(selectivity: f64) -> Self {
        Statistics::new(0.0, selectivity)
    }

    /// The estimated number of rows returned by an operator.
    pub fn row_count(&self) -> f64 {
        self.row_count
    }

    /// The selectivity of a predicate.
    pub fn selectivity(&self) -> f64 {
        self.selectivity
    }
}

pub trait StatisticsBuilder: Debug {
    /// Builds statistics for the given expression.
    fn build_statistics(&self, expr: &LogicalExpr, statistics: Option<&Statistics>) -> Option<Statistics>;
}

/// Builds statistics using information available in a [database catalog].
///
/// [database catalog]: crate::catalog::Catalog
#[derive(Debug)]
// TODO: Find a better name.
pub struct CatalogStatisticsBuilder {
    catalog: CatalogRef,
}

impl CatalogStatisticsBuilder {
    pub fn new(catalog: CatalogRef) -> Self {
        CatalogStatisticsBuilder { catalog }
    }
}

impl StatisticsBuilder for CatalogStatisticsBuilder {
    fn build_statistics(&self, expr: &LogicalExpr, statistics: Option<&Statistics>) -> Option<Statistics> {
        match expr {
            LogicalExpr::Projection { input, .. } => {
                let logical = input.attrs().logical();
                let statistics = logical.statistics().unwrap();
                Some(statistics.clone())
            }
            LogicalExpr::Select { input, .. } => {
                let logical = input.attrs().logical();
                let input_statistics = logical.statistics().unwrap();
                let selectivity = statistics.map_or(1.0, |s| s.selectivity());
                let row_count = selectivity * input_statistics.row_count();
                Some(Statistics::new(row_count, selectivity))
            }
            LogicalExpr::Join { left, .. } => {
                let logical = left.attrs().logical();
                let statistics = logical.statistics().unwrap();
                let row_count = statistics.row_count();
                // take selectivity of the join condition into account
                Some(Statistics::new(row_count, 1.0))
            }
            LogicalExpr::Get { source, .. } => {
                let table_ref = self
                    .catalog
                    .get_table(source)
                    .unwrap_or_else(|| panic!("Table '{}' does not exists", source));
                let row_count = table_ref
                    .statistics()
                    .map(|s| s.row_count())
                    .flatten()
                    .unwrap_or_else(|| panic!("No row count for table '{}'", source));
                Some(Statistics::new(row_count as f64, 1.0))
            }
            LogicalExpr::Expr { .. } => Some(Statistics::new(1.0, 1.0)),
        }
    }
}
