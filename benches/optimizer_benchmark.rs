use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use keenwa::catalog::mutable::MutableCatalog;
use keenwa::catalog::{Catalog, CatalogRef, TableBuilder, DEFAULT_SCHEMA};
use keenwa::cost::simple::SimpleCostEstimator;
use keenwa::datatypes::DataType;
use keenwa::meta::{ColumnId, ColumnMetadata, Metadata};
use keenwa::operators::expr::{BinaryOp, Expr};
use keenwa::operators::join::JoinCondition;
use keenwa::operators::logical::*;
use keenwa::operators::operator_tree::TestOperatorTreeBuilder;
use keenwa::operators::scalar::ScalarValue;
use keenwa::operators::*;
use keenwa::optimizer::{Optimizer, SetPropertiesCallback};
use keenwa::properties::logical::LogicalPropertiesBuilder;
use keenwa::properties::physical::PhysicalProperties;
use keenwa::properties::statistics::{CatalogStatisticsBuilder, Statistics};
use keenwa::properties::OrderingChoice;
use keenwa::rules::implementation::*;
use keenwa::rules::transformation::*;
use keenwa::rules::StaticRuleSet;
use keenwa::rules::*;
use keenwa::util::NoOpResultCallback;

fn memo_bench(c: &mut Criterion) {
    fn ordering(cols: Vec<ColumnId>) -> PhysicalProperties {
        PhysicalProperties::new(OrderingChoice::new(cols))
    }

    fn add_benchmark(c: &mut Criterion, name: &str, query: Operator) {
        c.bench_function(format!("optimize_query_{}", name).as_str(), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::default();

                for _i in 0..iters {
                    let tables = vec![
                        ("A".into(), vec![("a1".into(), DataType::Int32), ("a2".into(), DataType::Int32)]),
                        ("B".into(), vec![("b1".into(), DataType::Int32), ("b2".into(), DataType::Int32)]),
                        ("C".into(), vec![("c1".into(), DataType::Int32), ("c2".into(), DataType::Int32)]),
                    ];
                    let catalog = Arc::new(MutableCatalog::new());
                    let mut memo = create_memo(catalog.clone());

                    let table_access_costs = HashMap::from([("A".into(), 100), ("B".into(), 100), ("C".into(), 100)]);
                    let (query, metadata) =
                        prepare_query(catalog.clone(), &mut memo, query.clone(), tables, table_access_costs);

                    let optimizer = create_optimizer(catalog.clone());

                    let start = Instant::now();
                    let optimized_expr = optimizer
                        .optimize(query.clone(), metadata.clone(), &mut memo)
                        .expect("Failed to optimize a query");

                    black_box(optimized_expr);

                    total += start.elapsed();
                }

                total
            })
        });
    }

    let filter = Expr::BinaryExpr {
        lhs: Box::new(Expr::Column(1)),
        op: BinaryOp::Gt,
        rhs: Box::new(Expr::Scalar(ScalarValue::Int32(100))),
    };
    let query = LogicalExpr::Select {
        input: LogicalExpr::Join {
            left: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1],
            }
            .into(),
            right: LogicalExpr::Get {
                source: "B".into(),
                columns: vec![3],
            }
            .into(),
            condition: JoinCondition::using(vec![(1, 3)]),
        }
        .into(),
        filter: Some(ScalarNode::from(filter)),
    }
    .to_operator()
    .with_required(ordering(vec![1]))
    .with_statistics(Statistics::from_selectivity(0.1));

    add_benchmark(c, "Join AxB ordered", query);
}

fn prepare_query(
    catalog: CatalogRef,
    memo: &mut ExprMemo,
    query: Operator,
    tables: Vec<(String, Vec<(String, DataType)>)>,
    table_access_costs: HashMap<String, usize>,
) -> (Operator, Metadata) {
    let builder = TestOperatorTreeBuilder::new(memo, catalog, tables, table_access_costs);
    builder.build_initialized(query)
}

fn create_memo(catalog: CatalogRef) -> ExprMemo {
    let statistics_builder = CatalogStatisticsBuilder::new(catalog.clone());
    let properties_builder = LogicalPropertiesBuilder::new(Box::new(statistics_builder));
    let propagate_properties = SetPropertiesCallback::new(Rc::new(properties_builder));
    let memo_callback = Rc::new(propagate_properties);

    ExprMemo::with_callback(memo_callback.clone())
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
