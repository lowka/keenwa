use keenwa::catalog::mutable::MutableCatalog;
use keenwa::catalog::{CatalogRef, TableBuilder, DEFAULT_SCHEMA};
use keenwa::cost::simple::SimpleCostEstimator;
use keenwa::datatypes::DataType;
use keenwa::error::OptimizerError;
use keenwa::meta::MutableMetadata;
use keenwa::operators::builder::{MemoizeOperators, OperatorBuilder};
use keenwa::operators::format::format_operator_tree;
use keenwa::operators::scalar::col;
use keenwa::operators::OperatorMemoBuilder;
use keenwa::optimizer::{NoOpResultCallback, Optimizer};
use keenwa::rules::implementation::{EmptyRule, GetToScanRule, ProjectionRule, SelectRule};
use keenwa::rules::{Rule, StaticRuleSet};
use keenwa::statistics::simple::{DefaultSelectivityStatistics, SimpleCatalogStatisticsBuilder};
use std::rc::Rc;
use std::sync::Arc;

fn main() -> Result<(), OptimizerError> {
    let metadata = Rc::new(MutableMetadata::new());

    // 1. Create a catalog
    let catalog = create_catalog();

    // 2.  Create the optimizer.
    let optimizer = create_optimizer(catalog.clone());

    //3. Setup memo and logical properties builders.
    let mut memoization = create_memo(catalog.clone(), metadata.clone());

    //4. Create an operator with the memo callback.
    let operator_builder = OperatorBuilder::new(memoization.take_callback(), catalog, metadata);

    //5. Build a query
    let query = operator_builder.from("a")?.project(vec![col("a1")])?.build()?;

    //6. Retrieve the memo.
    let mut memo = memoization.into_memo();

    //7. Call optimizer to convert logical plan to physical.
    let result = optimizer.optimize(query, &mut memo)?;

    let actual_plan = format_operator_tree(&result);

    println!("-----");
    println!("{}", actual_plan);
    println!("-----");

    let expected = r#"Projection cols=[1] exprs: [col:1]
  input: Scan a cols=[1, 2]"#;

    assert_eq!(expected, actual_plan);

    Ok(())
}

fn create_memo(catalog: CatalogRef, metadata: Rc<MutableMetadata>) -> MemoizeOperators {
    let selectivity = DefaultSelectivityStatistics;
    let statistics_builder = SimpleCatalogStatisticsBuilder::new(catalog.clone(), selectivity);
    let memo = OperatorMemoBuilder::new(metadata).build_with_statistics(statistics_builder);

    MemoizeOperators::new(memo)
}

fn create_catalog() -> CatalogRef {
    let catalog = Arc::new(MutableCatalog::new());

    catalog.add_table(
        DEFAULT_SCHEMA,
        TableBuilder::new("a")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::String)
            .add_row_count(100)
            .build(),
    );

    catalog
}

fn create_optimizer(catalog: CatalogRef) -> Optimizer<StaticRuleSet, SimpleCostEstimator, NoOpResultCallback> {
    // Setup transformation/implementation rules.
    let rules: Vec<Box<dyn Rule>> = vec![
        Box::new(GetToScanRule::new(catalog)),
        Box::new(SelectRule),
        Box::new(ProjectionRule),
        Box::new(EmptyRule),
    ];

    let rule_set = StaticRuleSet::new(rules);
    let cost_estimator = SimpleCostEstimator::new();

    Optimizer::new(Rc::new(rule_set), Rc::new(cost_estimator), Rc::new(NoOpResultCallback))
}
