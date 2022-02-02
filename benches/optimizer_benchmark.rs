use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use keenwa::catalog::mutable::MutableCatalog;
use keenwa::catalog::{CatalogRef, TableBuilder, DEFAULT_SCHEMA};
use keenwa::cost::simple::SimpleCostEstimator;
use keenwa::datatypes::DataType;
use keenwa::error::OptimizerError;
use keenwa::memo::MemoBuilder;
use keenwa::meta::MutableMetadata;
use keenwa::operators::builder::{MemoizeOperatorCallback, OperatorBuilder, OrderingOption};

use keenwa::operators::properties::LogicalPropertiesBuilder;
use keenwa::operators::scalar::{col, scalar};
use keenwa::operators::*;
use keenwa::optimizer::{Optimizer, SetPropertiesCallback};
use keenwa::rules::implementation::*;
use keenwa::rules::transformation::*;
use keenwa::rules::StaticRuleSet;
use keenwa::rules::*;
use keenwa::statistics::simple::{PrecomputedSelectivityStatistics, SimpleCatalogStatisticsBuilder};
use keenwa::util::NoOpResultCallback;

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

                    catalog.add_table(
                        DEFAULT_SCHEMA,
                        TableBuilder::new("A")
                            .add_column("a1", DataType::Int32)
                            .add_column("a2", DataType::Int32)
                            .add_row_count(100)
                            .build(),
                    );

                    catalog.add_table(
                        DEFAULT_SCHEMA,
                        TableBuilder::new("B")
                            .add_column("b1", DataType::Int32)
                            .add_column("b2", DataType::Int32)
                            .add_row_count(100)
                            .build(),
                    );

                    catalog.add_table(
                        DEFAULT_SCHEMA,
                        TableBuilder::new("C")
                            .add_column("c1", DataType::Int32)
                            .add_column("c2", DataType::Int32)
                            .add_row_count(100)
                            .build(),
                    );

                    let metadata = Rc::new(MutableMetadata::new());
                    let selectivity_provider = Rc::new(PrecomputedSelectivityStatistics::new());

                    let statistics_builder =
                        SimpleCatalogStatisticsBuilder::new(catalog.clone(), selectivity_provider.clone());
                    let properties_builder = Rc::new(LogicalPropertiesBuilder::new(statistics_builder));

                    let memo = MemoBuilder::new(metadata.clone())
                        .set_callback(Rc::new(SetPropertiesCallback::new(properties_builder.clone())))
                        .build();
                    let memoization = Rc::new(MemoizeOperatorCallback::new(memo));
                    let operator_builder = OperatorBuilder::new(memoization.clone(), catalog.clone(), metadata);

                    let query = f(operator_builder, selectivity_provider.as_ref()).expect("Failed to build a query");

                    // We can retrieve the underlying memoization handler because
                    // the only user of Rc (an operator_builder) has been consumed.
                    let memoization = Rc::try_unwrap(memoization).unwrap();
                    let mut memo = memoization.into_inner();
                    // benchmark should not include time spend in memoization.
                    let query = memo.insert_group(query);

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
        let join = from_a.join_using(from_b, vec![("a1", "b1")])?;

        let filter = col("a1").gt(scalar(100));

        stats.set_selectivity("col:a1 > 100", 0.1);

        let select = join.select(Some(filter))?;
        let order_by = select.order_by(OrderingOption::by(("a1", false)))?;

        order_by.build()
    }

    add_benchmark(c, "Join AxB ordered", build_query);
}

fn create_optimizer(catalog: CatalogRef) -> Optimizer<StaticRuleSet, SimpleCostEstimator, NoOpResultCallback> {
    let rules: Vec<Box<dyn Rule>> = vec![
        Box::new(GetToScanRule::new(catalog)),
        Box::new(SelectRule),
        Box::new(HashJoinRule),
        Box::new(JoinCommutativityRule),
    ];

    let rules = StaticRuleSet::new(rules);
    let cost_estimator = SimpleCostEstimator::new();
    Optimizer::new(Rc::new(rules), Rc::new(cost_estimator), Rc::new(NoOpResultCallback))
}

criterion_group!(benches, memo_bench,);

criterion_main!(benches);
