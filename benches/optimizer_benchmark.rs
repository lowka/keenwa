use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use keenwa::catalog::mutable::MutableCatalog;
use keenwa::catalog::{CatalogRef, TableBuilder, DEFAULT_SCHEMA};
use keenwa::cost::simple::SimpleCostEstimator;
use keenwa::datatypes::DataType;
use keenwa::error::OptimizerError;
use keenwa::meta::MutableMetadata;
use keenwa::operators::builder::{MemoizeOperators, OperatorBuilder, OrderingOption};
use keenwa::operators::relational::join::JoinType;
use keenwa::operators::scalar::{col, scalar};
use keenwa::operators::{Operator, OperatorMemoBuilder, OuterScope};
use keenwa::optimizer::Optimizer;
use keenwa::rules::implementation::{GetToScanRule, HashJoinRule, SelectRule};
use keenwa::rules::transformation::JoinCommutativityRule;
use keenwa::rules::{Rule, StaticRuleSet, StaticRuleSetBuilder};
use keenwa::statistics::simple::{PrecomputedSelectivityStatistics, SimpleCatalogStatisticsBuilder};

fn memo_bench(c: &mut Criterion) {
    fn add_benchmark(
        c: &mut Criterion,
        name: &str,
        f: fn(OperatorBuilder, &PrecomputedSelectivityStatistics) -> Result<Operator, OptimizerError>,
    ) {
        c.bench_function(format!("optimize_query_{}", name).as_str(), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::default();

                for _i in 0..iters {
                    let catalog = Arc::new(MutableCatalog::new());

                    catalog
                        .add_table(
                            DEFAULT_SCHEMA,
                            TableBuilder::new("A")
                                .add_column("a1", DataType::Int32)
                                .add_column("a2", DataType::Int32)
                                .add_row_count(100)
                                .build()
                                .expect("Table A"),
                        )
                        .expect("Cannot add table A");

                    catalog
                        .add_table(
                            DEFAULT_SCHEMA,
                            TableBuilder::new("B")
                                .add_column("b1", DataType::Int32)
                                .add_column("b2", DataType::Int32)
                                .add_row_count(100)
                                .build()
                                .expect("Table B"),
                        )
                        .expect("Cannot add table B");

                    catalog
                        .add_table(
                            DEFAULT_SCHEMA,
                            TableBuilder::new("C")
                                .add_column("c1", DataType::Int32)
                                .add_column("c2", DataType::Int32)
                                .add_row_count(100)
                                .build()
                                .expect("Table C"),
                        )
                        .expect("Cannot add table C");

                    let metadata = Rc::new(MutableMetadata::new());
                    let selectivity_provider = Rc::new(PrecomputedSelectivityStatistics::new());

                    let statistics_builder =
                        SimpleCatalogStatisticsBuilder::new(catalog.clone(), selectivity_provider.clone());

                    let memo = OperatorMemoBuilder::new(metadata.clone()).build_with_statistics(statistics_builder);

                    let mut memoization = MemoizeOperators::new(memo);
                    let operator_builder = OperatorBuilder::new(memoization.take_callback(), catalog.clone(), metadata);

                    let query = f(operator_builder, selectivity_provider.as_ref()).expect("Failed to build a query");
                    let scope = OuterScope::root();

                    let mut memo = memoization.into_memo();
                    // benchmark should not include the time spend in memoization.
                    let query = memo.insert_group(query, &scope).expect("Failed to insert an operator");

                    let optimizer = create_optimizer(catalog.clone());
                    let start = Instant::now();
                    let optimized_expr = optimizer.optimize(query, &mut memo).expect("Failed to optimize a query");

                    black_box(optimized_expr);

                    total += start.elapsed();
                }

                total
            })
        });
    }

    fn build_query(
        builder: OperatorBuilder,
        stats: &PrecomputedSelectivityStatistics,
    ) -> Result<Operator, OptimizerError> {
        let from_a = builder.clone().get("A", vec!["a1"])?;
        let from_b = builder.get("B", vec!["b1"])?;
        let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;

        let filter = col("a1").gt(scalar(100));

        stats.set_selectivity("col:a1 > 100", 0.1);

        let select = join.select(Some(filter))?;
        let order_by = select.order_by(OrderingOption::by("a1", false))?;

        order_by.build()
    }

    add_benchmark(c, "Join AxB ordered", build_query);
}

fn create_optimizer(catalog: CatalogRef) -> Optimizer<StaticRuleSet, SimpleCostEstimator> {
    let rules: Vec<Box<dyn Rule>> = vec![
        Box::new(GetToScanRule::new(catalog)),
        Box::new(SelectRule),
        Box::new(HashJoinRule),
        Box::new(JoinCommutativityRule),
    ];

    let rules = StaticRuleSetBuilder::new().add_rules(rules).build();
    let cost_estimator = SimpleCostEstimator::new();
    Optimizer::new(Arc::new(rules), Arc::new(cost_estimator))
}

criterion_group!(benches, memo_bench,);

criterion_main!(benches);
