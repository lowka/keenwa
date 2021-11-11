use crate::catalog::CatalogRef;
use crate::error::OptimizerError;
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

    /// Creates a new statisticcs with row_count set to the given value.
    pub fn from_row_count(row_count: f64) -> Self {
        Statistics::new(row_count, 1.0)
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
    fn build_statistics(
        &self,
        expr: &LogicalExpr,
        statistics: Option<&Statistics>,
    ) -> Result<Option<Statistics>, OptimizerError>;
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
    fn build_statistics(
        &self,
        expr: &LogicalExpr,
        statistics: Option<&Statistics>,
    ) -> Result<Option<Statistics>, OptimizerError> {
        let statistics = match expr {
            LogicalExpr::Projection { input, .. } => {
                let logical = input.attrs().logical();
                let statistics = logical.statistics().unwrap();
                statistics.clone()
            }
            LogicalExpr::Select { input, .. } => {
                let logical = input.attrs().logical();
                let input_statistics = logical.statistics().unwrap();
                let selectivity = statistics.map_or(1.0, |s| s.selectivity());
                let row_count = selectivity * input_statistics.row_count();
                Statistics::new(row_count, selectivity)
            }
            LogicalExpr::Aggregate { group_exprs, .. } => {
                let max_groups = if group_exprs.is_empty() {
                    1.0
                } else {
                    group_exprs.len() as f64
                };
                Statistics::from_row_count(max_groups)
            }
            LogicalExpr::Join { left, .. } => {
                let logical = left.attrs().logical();
                let statistics = logical.statistics().unwrap();
                let row_count = statistics.row_count();
                // take selectivity of the join condition into account
                Statistics::new(row_count, 1.0)
            }
            LogicalExpr::Get { source, .. } => {
                let table_ref = match self.catalog.get_table(source) {
                    Some(table_ref) => table_ref,
                    None => return Err(OptimizerError::Internal(format!("Table '{}' does not exists", source))),
                };
                let row_count = match table_ref.statistics().map(|s| s.row_count()).flatten() {
                    Some(row_count) => row_count,
                    None => return Err(OptimizerError::Internal(format!("No row count for table '{}'", source))),
                };
                Statistics::new(row_count as f64, 1.0)
            }
            LogicalExpr::Expr { .. } => Statistics::new(1.0, 1.0),
        };
        Ok(Some(statistics))
    }
}

#[cfg(test)]
mod test {
    use crate::catalog::mutable::MutableCatalog;
    use crate::catalog::TableBuilder;
    use crate::datatypes::DataType;
    use crate::operators::expressions::{AggregateFunction, Expr};
    use crate::operators::logical::LogicalExpr;
    use crate::properties::statistics::{CatalogStatisticsBuilder, Statistics, StatisticsBuilder};
    use std::sync::Arc;

    fn new_statistics_builder(tables: Vec<(&str, usize)>) -> impl StatisticsBuilder {
        let catalog = MutableCatalog::new();

        for (name, _row_count) in tables {
            let table = TableBuilder::new(name)
                .add_column(format!("{}1", name).as_str(), DataType::Int32)
                .add_column(format!("{}2", name).as_str(), DataType::Int32)
                .build();

            catalog.add_table("s", table);
        }

        CatalogStatisticsBuilder::new(Arc::new(catalog))
    }

    fn new_aggregate(groups: Vec<Expr>) -> LogicalExpr {
        LogicalExpr::Aggregate {
            input: LogicalExpr::Get {
                source: "A".to_string(),
                columns: vec![1],
            }
            .into(),
            aggr_exprs: vec![Expr::Aggregate {
                func: AggregateFunction::Avg,
                args: vec![Expr::Column(1)],
                filter: None,
            }],
            group_exprs: groups,
        }
    }

    #[test]
    fn test_aggregate_statistics_no_groups() {
        let statistics_builder = new_statistics_builder(vec![("A", 0)]);

        let aggr = new_aggregate(vec![]);
        let stats = compute_statistics(&statistics_builder, &aggr, None);
        expect_statistics(&stats, 1.0, 1.0);
    }

    #[test]
    fn test_aggregate_statistics_multiple_groups() {
        let statistics_builder = new_statistics_builder(vec![("A", 0)]);

        let aggr = new_aggregate(vec![Expr::Column(1), Expr::Column(2)]);
        let stats = compute_statistics(&statistics_builder, &aggr, None);
        expect_statistics(&stats, 2.0, 1.0);
    }

    fn compute_statistics(
        statistics_builder: &impl StatisticsBuilder,
        expr: &LogicalExpr,
        statistics: Option<Statistics>,
    ) -> Statistics {
        let statistics = statistics_builder
            .build_statistics(expr, statistics.as_ref())
            .expect("Failed to build logical properties");
        statistics.expect("No statistics")
    }

    fn expect_statistics(statistics: &Statistics, row_count: f64, selectivity: f64) {
        assert_eq!(statistics.row_count(), row_count, "row_count");
        assert_eq!(statistics.selectivity(), selectivity, "selectivity");
    }
}
