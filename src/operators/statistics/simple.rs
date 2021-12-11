use crate::catalog::{Catalog, CatalogRef};
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::relational::join::JoinCondition;
use crate::operators::relational::logical::{LogicalExpr, SetOperator};
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::statistics::StatisticsBuilder;
use crate::properties::logical::LogicalProperties;
use crate::properties::statistics::Statistics;

/// A simple implementation of [StatisticsBuilder](super::StatisticsBuilder) that information available in a [database catalog].
///
/// [database catalog]: crate::catalog::Catalog
// FIXME: (MemoMetadata) Rename to mutable SimpleStatisticsBuilder (remove dependency on Catalog and instead add method to set statistics manually.
//  Add method that sets the number of rows per table.
//  Add method that specifies predicate selectivity (directly or indirectly ag. select filter selectivity = select (row count)/ input (row count) )
#[derive(Debug)]
pub struct SimpleCatalogStatisticsBuilder {
    catalog: CatalogRef,
}

impl SimpleCatalogStatisticsBuilder {
    pub fn new(catalog: CatalogRef) -> Self {
        SimpleCatalogStatisticsBuilder { catalog }
    }
}

impl SimpleCatalogStatisticsBuilder {
    fn build_get(&self, source: &str) -> Result<Option<Statistics>, OptimizerError> {
        let table_ref = match self.catalog.get_table(source) {
            Some(table_ref) => table_ref,
            None => return Err(OptimizerError::Internal(format!("Table '{}' does not exists", source))),
        };
        let row_count = match table_ref.statistics().map(|s| s.row_count()).flatten() {
            Some(row_count) => row_count,
            None => return Err(OptimizerError::Internal(format!("No row count for table '{}'", source))),
        };
        Ok(Some(Statistics::new(row_count as f64, 1.0)))
    }

    fn build_projection(&self, input: &RelNode, _columns: &[ColumnId]) -> Result<Option<Statistics>, OptimizerError> {
        let logical = input.props().logical();
        let statistics = logical.statistics().cloned();

        Ok(statistics)
    }

    fn build_select(&self, input: &RelNode, filter: Option<&ScalarNode>) -> Result<Option<Statistics>, OptimizerError> {
        let logical = input.props().logical();
        let input_statistics = logical.statistics().unwrap();
        let selectivity = if let Some(filter) = filter {
            let filter_statistics = filter.props().logical().statistics().unwrap();
            filter_statistics.selectivity()
        } else {
            1.0
        };
        let row_count = selectivity * input_statistics.row_count();
        Ok(Some(Statistics::new(row_count, selectivity)))
    }

    fn build_join(
        &self,
        left: &RelNode,
        _right: &RelNode,
        _condition: &JoinCondition,
    ) -> Result<Option<Statistics>, OptimizerError> {
        let logical = left.props().logical();
        let statistics = logical.statistics().unwrap();
        let row_count = statistics.row_count();
        // take selectivity of the join condition into account
        Ok(Some(Statistics::new(row_count, 1.0)))
    }

    fn build_aggregate(
        &self,
        _input: &RelNode,
        group_exprs: &[ScalarNode],
        _columns: &[ColumnId],
    ) -> Result<Option<Statistics>, OptimizerError> {
        let max_groups = if group_exprs.is_empty() {
            1.0
        } else {
            group_exprs.len() as f64
        };
        Ok(Some(Statistics::from_row_count(max_groups)))
    }

    fn build_set_operator(
        &self,
        _set_op: SetOperator,
        _all: bool,
        left: &RelNode,
        right: &RelNode,
        _columns: &[ColumnId],
    ) -> Result<Option<Statistics>, OptimizerError> {
        let left_statistics = left.props().logical().statistics().unwrap();
        let right_statistics = right.props().logical().statistics().unwrap();
        let row_count = left_statistics.row_count() + right_statistics.row_count();
        //FIXME: The result of a set operation with all=false should include only non-duplicate rows.
        Ok(Some(Statistics::from_row_count(row_count)))
    }
}

impl StatisticsBuilder for SimpleCatalogStatisticsBuilder {
    fn build_statistics(
        &self,
        expr: &LogicalExpr,
        _logical: &LogicalProperties,
    ) -> Result<Option<Statistics>, OptimizerError> {
        match expr {
            LogicalExpr::Projection { input, columns, .. } => self.build_projection(input, columns),
            LogicalExpr::Select { input, filter, .. } => self.build_select(input, filter.as_ref()),
            LogicalExpr::Aggregate { input, group_exprs, .. } => self.build_aggregate(input, group_exprs, &[]),
            LogicalExpr::Join { left, right, condition } => self.build_join(left, right, condition),
            LogicalExpr::Get { source, .. } => self.build_get(source),
            LogicalExpr::Union {
                left,
                right,
                all,
                columns,
            } => self.build_set_operator(SetOperator::Union, *all, left, right, columns),
            LogicalExpr::Intersect {
                left,
                right,
                all,
                columns,
            } => self.build_set_operator(SetOperator::Intersect, *all, left, right, columns),
            LogicalExpr::Except {
                left,
                right,
                all,
                columns,
            } => self.build_set_operator(SetOperator::Except, *all, left, right, columns),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::catalog::mutable::MutableCatalog;
    use crate::catalog::TableBuilder;
    use crate::datatypes::DataType;
    use crate::operators::relational::logical::LogicalExpr;
    use crate::operators::scalar::expr::AggregateFunction;
    use crate::operators::scalar::{ScalarExpr, ScalarNode};
    use crate::operators::statistics::simple::SimpleCatalogStatisticsBuilder;
    use crate::operators::statistics::StatisticsBuilder;
    use crate::operators::{Operator, OperatorExpr, Properties};
    use crate::properties::logical::LogicalProperties;
    use crate::properties::physical::PhysicalProperties;
    use crate::properties::statistics::Statistics;
    use std::sync::Arc;

    fn new_aggregate(groups: Vec<ScalarExpr>) -> LogicalExpr {
        LogicalExpr::Aggregate {
            input: LogicalExpr::Get {
                source: "A".to_string(),
                columns: vec![1],
            }
            .into(),
            aggr_exprs: vec![ScalarNode::from(ScalarExpr::Aggregate {
                func: AggregateFunction::Avg,
                args: vec![ScalarExpr::Column(1)],
                filter: None,
            })],
            group_exprs: groups.into_iter().map(ScalarNode::from).collect(),
            columns: vec![],
        }
    }

    #[test]
    fn test_aggregate_statistics_no_groups() {
        let tester = StatisticsTester::new(Vec::<(String, usize)>::new());

        let aggr = new_aggregate(vec![]);

        tester.expect_statistics(&aggr, Some(Statistics::new(1.0, 1.0)));
    }

    #[test]
    fn test_aggregate_statistics_multiple_groups() {
        let tester = StatisticsTester::new(Vec::<(String, usize)>::new());
        let aggr = new_aggregate(vec![ScalarExpr::Column(1), ScalarExpr::Column(2)]);

        tester.expect_statistics(&aggr, Some(Statistics::new(2.0, 1.0)));
    }

    #[test]
    fn test_union_statistics() {
        let tester = StatisticsTester::new(vec![("A", 10), ("B", 5)]);
        let left = tester.new_operator("A", OperatorStatistics::FromTable("A"));
        let right = tester.new_operator("A", OperatorStatistics::FromTable("B"));

        let union = LogicalExpr::Union {
            left: left.into(),
            right: right.into(),
            all: false,
            columns: vec![],
        };

        tester.expect_statistics(&union, Some(Statistics::from_row_count(15.0)))
    }

    enum OperatorStatistics {
        Provided(Statistics),
        FromTable(&'static str),
    }

    struct StatisticsTester {
        statistics_builder: SimpleCatalogStatisticsBuilder,
    }

    impl StatisticsTester {
        fn new(table_statistics: Vec<(impl Into<String>, usize)>) -> Self {
            let catalog = MutableCatalog::new();

            for (name, row_count) in table_statistics {
                let name = name.into();
                let table = TableBuilder::new(name.as_str())
                    .add_column(format!("{}1", name).as_str(), DataType::Int32)
                    .add_column(format!("{}2", name).as_str(), DataType::Int32)
                    .add_row_count(row_count)
                    .build();

                catalog.add_table(crate::catalog::DEFAULT_SCHEMA, table);
            }

            StatisticsTester {
                statistics_builder: SimpleCatalogStatisticsBuilder::new(Arc::new(catalog)),
            }
        }

        fn new_operator(&self, table: &str, statistics: OperatorStatistics) -> Operator {
            let statistics = match statistics {
                OperatorStatistics::Provided(statistics) => statistics,
                OperatorStatistics::FromTable(source) => {
                    let table = self
                        .statistics_builder
                        .catalog
                        .get_table(source)
                        .unwrap_or_else(|| panic!("No table {}", source));
                    let stats = table.statistics().unwrap_or_else(|| panic!("Table {} has no statistics", source));
                    Statistics::from_row_count(stats.row_count().expect("No row count") as f64)
                }
            };

            let expr = LogicalExpr::Get {
                source: table.into(),
                // Currently is not used when statistics are computed.
                columns: vec![],
            };

            let logical = LogicalProperties::new(vec![], Some(statistics));
            let properties = Properties::new(logical, PhysicalProperties::none());

            Operator::new(OperatorExpr::from(expr), properties)
        }

        fn expect_statistics(&self, expr: &LogicalExpr, expected: Option<Statistics>) {
            // At the moment logical properties are not used when statistics are computed.
            let logical_properties = LogicalProperties::new(vec![], None);
            let actual = self
                .statistics_builder
                .build_statistics(expr, &logical_properties)
                .expect("Failed to compute statistics");
            match expected {
                None => assert!(actual.is_none(), "Expected no statistics but got: {:?}. Operator: {:?}", actual, expr),
                Some(expected) => {
                    assert!(
                        actual.is_some(),
                        "Expected statistics no has been computed. Expected: {:?}, Operator: {:?}",
                        expected,
                        expr
                    );
                    let actual = actual.unwrap();
                    assert_eq!(actual.row_count(), expected.row_count(), "row count");
                    assert_eq!(actual.selectivity(), expected.selectivity(), "selectivity");
                }
            }
        }
    }
}
