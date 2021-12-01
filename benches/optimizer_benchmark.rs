use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use keenwa::catalog::mutable::MutableCatalog;
use keenwa::catalog::{Catalog, TableBuilder, DEFAULT_SCHEMA};
use keenwa::cost::simple::SimpleCostEstimator;
use keenwa::datatypes::DataType;
use keenwa::meta::{ColumnId, Metadata};
use keenwa::operators::expr::{BinaryOp, Expr};
use keenwa::operators::join::JoinCondition;
use keenwa::operators::logical::*;
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

fn metadata_from_catalog(catalog: &dyn Catalog) -> Metadata {
    let mut metadata = HashMap::new();
    let mut counter = 1;

    for schema in catalog.get_schemas() {
        for table in schema.get_tables() {
            for c in table.columns() {
                metadata.insert(counter, c.clone());
                counter += 1;
            }
        }
    }

    Metadata::new(metadata)
}

fn memo_bench(c: &mut Criterion) {
    fn ordering(cols: Vec<ColumnId>) -> PhysicalProperties {
        PhysicalProperties::new(OrderingChoice::new(cols))
    }

    fn add_benchmark(c: &mut Criterion, name: &str, query: Operator) {
        let catalog = MutableCatalog::new();
        let table1 = TableBuilder::new("A").add_column("a", DataType::Int32).add_row_count(100).build();
        let table2 = TableBuilder::new("B").add_column("b", DataType::Int32).add_row_count(110).build();
        catalog.add_table(DEFAULT_SCHEMA, table1);
        catalog.add_table(DEFAULT_SCHEMA, table2);

        let catalog = Arc::new(catalog);
        let metadata = metadata_from_catalog(catalog.as_ref());

        let rules: Vec<Box<dyn Rule>> = vec![
            Box::new(GetToScanRule::new(catalog.clone())),
            Box::new(SelectRule),
            Box::new(HashJoinRule),
            Box::new(JoinCommutativityRule),
        ];

        let rules = StaticRuleSet::new(rules);
        let cost_estimator = SimpleCostEstimator::new();
        let statistics_builder = CatalogStatisticsBuilder::new(catalog);
        let properties_builder = LogicalPropertiesBuilder::new(Box::new(statistics_builder));
        let propagate_properties = SetPropertiesCallback::new(Rc::new(properties_builder));
        let memo_callback = Rc::new(propagate_properties);

        let optimizer = Optimizer::new(Rc::new(rules), Rc::new(cost_estimator), Rc::new(NoOpResultCallback));
        let optimizer = Rc::new(optimizer);

        c.bench_function(format!("optimize_query_{}", name).as_str(), |b| {
            b.iter(|| {
                let mut memo = ExprMemo::with_callback(memo_callback.clone());
                let optimized_expr = optimizer
                    .optimize(query.clone(), metadata.clone(), &mut memo)
                    .expect("Failed to optimize a query");

                black_box(optimized_expr);
            });
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
                columns: vec![2],
            }
            .into(),
            condition: JoinCondition::using(vec![(1, 2)]),
        }
        .into(),
        filter: Some(ScalarNode::from(filter)),
    }
    .to_operator()
    .with_required(ordering(vec![1]))
    .with_statistics(Statistics::from_selectivity(0.1));

    add_benchmark(c, "Join AxB ordered", query);
}

criterion_group!(benches, memo_bench,);

criterion_main!(benches);
