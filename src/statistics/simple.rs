//! Statistics for available operators.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::Infallible;
use std::rc::Rc;

use crate::catalog::CatalogRef;
use crate::error::OptimizerError;
use crate::meta::{ColumnId, MetadataRef};
use crate::operators::relational::join::{JoinCondition, JoinType};
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalDistinct, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin,
    LogicalLimit, LogicalOffset, LogicalProjection, LogicalSelect, LogicalUnion, SetOperator,
};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::ExprRewriter;
use crate::operators::scalar::{ScalarExpr, ScalarNode};
use crate::properties::logical::LogicalProperties;
use crate::statistics::Statistics;
use crate::statistics::StatisticsBuilder;

/// A simple implementation of [StatisticsBuilder](super::StatisticsBuilder) that information available in a [database catalog].
///
/// [database catalog]: crate::catalog::Catalog
#[derive(Debug)]
pub struct SimpleCatalogStatisticsBuilder<T> {
    catalog: CatalogRef,
    selectivity_provider: T,
}

impl<T> SimpleCatalogStatisticsBuilder<T>
where
    T: SelectivityProvider,
{
    pub fn new(catalog: CatalogRef, selectivity_provider: T) -> Self {
        SimpleCatalogStatisticsBuilder {
            catalog,
            selectivity_provider,
        }
    }

    fn build_get(&self, source: &str) -> Result<Option<Statistics>, OptimizerError> {
        let table_ref = match self.catalog.get_table(source) {
            Some(table_ref) => table_ref,
            None => return Err(OptimizerError::Internal(format!("Table '{}' does not exists", source))),
        };
        let row_count = match table_ref.statistics().and_then(|s| s.row_count()) {
            Some(row_count) => row_count,
            None => return Err(OptimizerError::Internal(format!("No row count for table '{}'", source))),
        };
        Ok(Some(Statistics::from_row_count(row_count as f64)))
    }

    fn build_projection(&self, input: &RelNode, _columns: &[ColumnId]) -> Result<Option<Statistics>, OptimizerError> {
        let logical = input.props().logical();
        let statistics = logical.statistics().cloned();

        Ok(statistics)
    }

    fn build_select(
        &self,
        input: &RelNode,
        filter: Option<&ScalarNode>,
        metadata: MetadataRef,
    ) -> Result<Option<Statistics>, OptimizerError> {
        let selectivity = match filter {
            Some(filter) => self.build_filter(filter, metadata)?,
            None => None,
        }
        .unwrap_or(Statistics::DEFAULT_SELECTIVITY);

        let logical = input.props().logical();
        let input_statistics = logical.statistics().unwrap();
        let row_count = selectivity * input_statistics.row_count();
        Ok(Some(Statistics::new(row_count, selectivity)))
    }

    fn build_join(
        &self,
        _join_type: &JoinType,
        left: &RelNode,
        _right: &RelNode,
        _condition: &JoinCondition,
    ) -> Result<Option<Statistics>, OptimizerError> {
        let logical = left.props().logical();
        let statistics = logical.statistics().unwrap();
        let row_count = statistics.row_count();
        // take selectivity of the join condition into account
        Ok(Some(Statistics::from_row_count(row_count)))
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

    fn build_empty(&self) -> Result<Option<Statistics>, OptimizerError> {
        Ok(Some(Statistics::new(0f64, Statistics::DEFAULT_SELECTIVITY)))
    }

    fn build_filter(&self, expr: &ScalarNode, metadata: MetadataRef) -> Result<Option<f64>, OptimizerError> {
        let selectivity = self.selectivity_provider.get_selectivity(expr.expr(), metadata);

        Ok(selectivity)
    }

    fn build_distinct(
        &self,
        input: &RelNode,
        _on_expr: Option<&ScalarNode>,
    ) -> Result<Option<Statistics>, OptimizerError> {
        let input_statistics = input.props().logical().statistics().unwrap();
        let row_count = input_statistics.row_count();
        //FIXME: The result of a distinct operator should include only non-duplicate rows.
        Ok(Some(Statistics::from_row_count(row_count)))
    }

    fn build_limit(&self, input: &RelNode, rows: usize) -> Result<Option<Statistics>, OptimizerError> {
        let input_statistics = input.props().logical().statistics().unwrap();
        let input_rows = input_statistics.row_count();
        let rows = rows as f64;

        let rows = if rows > input_rows { input_rows } else { rows };

        Ok(Some(Statistics::from_row_count(rows as f64)))
    }

    fn build_offset(&self, input: &RelNode, rows: usize) -> Result<Option<Statistics>, OptimizerError> {
        let input_statistics = input.props().logical().statistics().unwrap();
        let input_rows = input_statistics.row_count();
        let rows = rows as f64;

        let rows = if input_rows > rows { rows } else { rows - input_rows };

        Ok(Some(Statistics::from_row_count(rows)))
    }
}

impl<T> StatisticsBuilder for SimpleCatalogStatisticsBuilder<T>
where
    T: SelectivityProvider,
{
    fn build_statistics(
        &self,
        expr: &LogicalExpr,
        _logical: &LogicalProperties,
        metadata: MetadataRef,
    ) -> Result<Option<Statistics>, OptimizerError> {
        match expr {
            LogicalExpr::Projection(LogicalProjection { input, columns, .. }) => self.build_projection(input, columns),
            LogicalExpr::Select(LogicalSelect { input, filter, .. }) => {
                self.build_select(input, filter.as_ref(), metadata)
            }
            LogicalExpr::Aggregate(LogicalAggregate { input, group_exprs, .. }) => {
                self.build_aggregate(input, group_exprs, &[])
            }
            LogicalExpr::Join(LogicalJoin {
                join_type,
                left,
                right,
                condition,
            }) => self.build_join(join_type, left, right, condition),
            LogicalExpr::Get(LogicalGet { source, .. }) => self.build_get(source),
            LogicalExpr::Union(LogicalUnion {
                left,
                right,
                all,
                columns,
            }) => self.build_set_operator(SetOperator::Union, *all, left, right, columns),
            LogicalExpr::Intersect(LogicalIntersect {
                left,
                right,
                all,
                columns,
            }) => self.build_set_operator(SetOperator::Intersect, *all, left, right, columns),
            LogicalExpr::Except(LogicalExcept {
                left,
                right,
                all,
                columns,
            }) => self.build_set_operator(SetOperator::Except, *all, left, right, columns),
            LogicalExpr::Distinct(LogicalDistinct { input, on_expr, .. }) => {
                self.build_distinct(input, on_expr.as_ref())
            }
            LogicalExpr::Limit(LogicalLimit { input, rows }) => self.build_limit(input, *rows),
            LogicalExpr::Offset(LogicalOffset { input, rows }) => self.build_offset(input, *rows),
            LogicalExpr::Empty(_) => self.build_empty(),
        }
    }
}

/// An extra trait that computes selectivity of predicate expressions.  
pub trait SelectivityProvider {
    /// Returns this selectivity statistics as [`Any`](std::any::Any) in order it can be downcast to its implementation.
    fn as_any(&self) -> &dyn Any;

    /// Returns selectivity of the given predicate expression `filter` in a context of the relational expression `expr`.
    fn get_selectivity(&self, filter: &ScalarExpr, metadata: MetadataRef) -> Option<f64>;
}

/// [SelectivityProvider](self::SelectivityProvider) that always returns selectivity of 1.0.
#[derive(Debug)]
pub struct DefaultSelectivityStatistics;

impl SelectivityProvider for DefaultSelectivityStatistics {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_selectivity(&self, _filter: &ScalarExpr, _metadata: MetadataRef) -> Option<f64> {
        Some(Statistics::DEFAULT_SELECTIVITY)
    }
}

/// [SelectivityProvider](self::SelectivityProvider) that allows to set selectivity on per predicate basis.
/// When selectivity for a predicate is not specified returns selectivity of 1.0.
#[derive(Debug)]
pub struct PrecomputedSelectivityStatistics {
    inner: RefCell<HashMap<String, f64>>,
}

impl PrecomputedSelectivityStatistics {
    /// Creates a new instance of `PrecomputedSelectivityStatistics`.
    pub fn new() -> Self {
        PrecomputedSelectivityStatistics {
            inner: RefCell::new(HashMap::new()),
        }
    }

    /// Sets selectivity for the given predicate `expr`.
    pub fn set_selectivity(&self, expr: &str, value: f64) {
        let mut inner = self.inner.borrow_mut();
        inner.insert(expr.into(), value);
    }
}

impl SelectivityProvider for PrecomputedSelectivityStatistics {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_selectivity(&self, filter: &ScalarExpr, metadata: MetadataRef) -> Option<f64> {
        struct ReplaceColumnIdsWithColumnNames<'a> {
            metadata: MetadataRef<'a>,
        }

        impl ExprRewriter<RelNode> for ReplaceColumnIdsWithColumnNames<'_> {
            type Error = Infallible;
            fn rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
                if let ScalarExpr::Column(id) = expr {
                    // Should not panic because columns has been assigned by the metadata itself.
                    let column = self.metadata.get_column(&id);
                    Ok(ScalarExpr::ColumnName(column.name().clone()))
                } else {
                    Ok(expr)
                }
            }
        }

        let filter = filter.clone();
        let mut rewriter = ReplaceColumnIdsWithColumnNames { metadata };
        // Never returns an error
        let filter = filter.rewrite(&mut rewriter).unwrap();
        let filter_str = format!("{}", filter);

        let inner = self.inner.borrow();
        inner.get(&filter_str).copied().or(Some(Statistics::DEFAULT_SELECTIVITY))
    }
}

impl<T> SelectivityProvider for Rc<T>
where
    T: SelectivityProvider + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self.as_ref()
    }

    fn get_selectivity(&self, filter: &ScalarExpr, metadata: MetadataRef) -> Option<f64> {
        self.as_ref().get_selectivity(filter, metadata)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::catalog::mutable::MutableCatalog;
    use crate::catalog::TableBuilder;
    use crate::datatypes::DataType;
    use crate::meta::MutableMetadata;
    use crate::operators::relational::logical::{
        LogicalAggregate, LogicalEmpty, LogicalExpr, LogicalGet, LogicalUnion,
    };
    use crate::operators::scalar::expr::AggregateFunction;
    use crate::operators::scalar::{ScalarExpr, ScalarNode};
    use crate::operators::{Operator, OperatorExpr, Properties, RelationalProperties};
    use crate::properties::logical::LogicalProperties;
    use crate::properties::physical::PhysicalProperties;
    use crate::statistics::simple::{DefaultSelectivityStatistics, SimpleCatalogStatisticsBuilder};
    use crate::statistics::Statistics;
    use crate::statistics::StatisticsBuilder;

    fn new_aggregate(groups: Vec<ScalarExpr>) -> LogicalExpr {
        LogicalExpr::Aggregate(LogicalAggregate {
            input: LogicalExpr::Get(LogicalGet {
                source: "A".to_string(),
                columns: vec![1],
            })
            .into(),
            aggr_exprs: vec![ScalarNode::from(ScalarExpr::Aggregate {
                func: AggregateFunction::Avg,
                args: vec![ScalarExpr::Column(1)],
                filter: None,
            })],
            group_exprs: groups.into_iter().map(ScalarNode::from).collect(),
            having: None,
            columns: vec![],
        })
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

        let union = LogicalExpr::Union(LogicalUnion {
            left: left.into(),
            right: right.into(),
            all: false,
            columns: vec![],
        });

        tester.expect_statistics(&union, Some(Statistics::from_row_count(15.0)))
    }

    #[test]
    fn test_empty_statistics() {
        let tester = StatisticsTester::new(vec![("A", 10)]);

        tester.expect_statistics(
            &LogicalExpr::Empty(LogicalEmpty { return_one_row: true }),
            Some(Statistics::new(0f64, Statistics::DEFAULT_SELECTIVITY)),
        )
    }

    enum OperatorStatistics {
        Provided(Statistics),
        FromTable(&'static str),
    }

    struct StatisticsTester {
        statistics_builder: SimpleCatalogStatisticsBuilder<DefaultSelectivityStatistics>,
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
            let selectivity_provider = DefaultSelectivityStatistics;

            StatisticsTester {
                statistics_builder: SimpleCatalogStatisticsBuilder::new(Arc::new(catalog), selectivity_provider),
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

            let expr = LogicalExpr::Get(LogicalGet {
                source: table.into(),
                // Currently is not used when statistics are computed.
                columns: vec![],
            });

            let logical = LogicalProperties::new(vec![], Some(statistics));
            let properties = Properties::Relational(RelationalProperties {
                logical,
                physical: PhysicalProperties::none(),
            });

            Operator::new(OperatorExpr::from(expr), properties)
        }

        fn expect_statistics(&self, expr: &LogicalExpr, expected: Option<Statistics>) {
            // At the moment logical properties are not used when statistics are computed.
            let logical_properties = LogicalProperties::new(vec![], None);
            let metadata = MutableMetadata::new();
            let actual = self
                .statistics_builder
                .build_statistics(expr, &logical_properties, metadata.get_ref())
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
