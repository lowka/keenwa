use keenwa::catalog::mutable::MutableCatalog;
use keenwa::catalog::{CatalogRef, TableBuilder, DEFAULT_SCHEMA};
use keenwa::cost::simple::SimpleCostEstimator;
use keenwa::datatypes::DataType;
use keenwa::error::OptimizerError;
use keenwa::meta::MutableMetadata;
use keenwa::operators::builder::{MemoizeOperators, OperatorBuilder};
use keenwa::operators::format::format_operator_tree;
use keenwa::operators::relational::join::JoinType;
use keenwa::operators::scalar::{col, scalar};
use keenwa::operators::{Operator, OperatorMemoBuilder};
use keenwa::optimizer::Optimizer;
use keenwa::rules::implementation::{EmptyRule, GetToScanRule, HashJoinRule, ProjectionRule, SelectRule};
use keenwa::rules::{Rule, StaticRuleSet, StaticRuleSetBuilder};
use keenwa::statistics::simple::{DefaultSelectivityStatistics, SimpleCatalogStatisticsBuilder};
use std::ops::Not;
use std::rc::Rc;
use std::sync::Arc;

fn main() -> Result<(), OptimizerError> {
    let metadata = Rc::new(MutableMetadata::new());

    // 1. Create a catalog
    let catalog = create_catalog()?;

    // 2.  Create the optimizer.
    let optimizer = create_optimizer(catalog.clone());

    //3. Setup memo and logical properties builders.
    let mut memoization = create_memo(catalog.clone(), metadata.clone());

    //4. Create an operator with the memo callback.
    let operator_builder = OperatorBuilder::new(memoization.take_callback(), catalog, metadata);

    //5. Build a query
    let from_a = operator_builder.clone().from("a")?;
    let from_b = operator_builder.from("b")?;
    let expr = col("a1").eq(col("b1"));
    let join = from_a.join_on(from_b, JoinType::Inner, expr)?;
    let query = join.project(vec![col("a1") + scalar(100), col("a2").not()])?.build()?;

    //6. Retrieve the memo.
    let mut memo = memoization.into_memo();

    //7. Call optimizer to convert logical plan to physical.
    let result = optimizer.optimize(query, &mut memo)?;

    let expected_plan = r#"
Projection cols=[5, 6] exprs: [col:1 + 100, NOT col:2]
  input: HashJoin type=Inner on=col:1 = col:4
    left: Scan a cols=[1, 2, 3]
    right: Scan b cols=[4]
"#;

    expect_plan(&result, "Optimized plan", expected_plan);

    Ok(())
}

fn create_memo(catalog: CatalogRef, metadata: Rc<MutableMetadata>) -> MemoizeOperators {
    let selectivity = DefaultSelectivityStatistics;
    let statistics_builder = SimpleCatalogStatisticsBuilder::new(catalog.clone(), selectivity);
    let memo = OperatorMemoBuilder::new(metadata).build_with_statistics(statistics_builder);

    MemoizeOperators::new(memo)
}

fn create_catalog() -> Result<CatalogRef, OptimizerError> {
    let catalog = Arc::new(MutableCatalog::new());

    let table_a = TableBuilder::new("a")
        .add_column("a1", DataType::Int32)
        .add_column("a2", DataType::Bool)
        .add_column("a3", DataType::String)
        .add_row_count(100)
        .build()?;

    let table_b = TableBuilder::new("b").add_column("b1", DataType::Int32).add_row_count(200).build()?;

    catalog.add_table(DEFAULT_SCHEMA, table_a)?;
    catalog.add_table(DEFAULT_SCHEMA, table_b)?;
    Ok(catalog)
}

fn create_optimizer(catalog: CatalogRef) -> Optimizer<StaticRuleSet, SimpleCostEstimator> {
    // Setup transformation/implementation rules.
    let rules: Vec<Box<dyn Rule>> = vec![
        Box::new(GetToScanRule::new(catalog)),
        Box::new(SelectRule),
        Box::new(ProjectionRule),
        Box::new(EmptyRule),
        Box::new(HashJoinRule),
    ];

    let rule_set = StaticRuleSetBuilder::new().add_rules(rules).build();
    let cost_estimator = SimpleCostEstimator::new();

    Optimizer::new(Arc::new(rule_set), Arc::new(cost_estimator))
}

fn expect_plan(operator: &Operator, plan_type: &str, expected_plan: &str) {
    let actual_plan = format_operator_tree(operator);
    println!("{}:\n{}", plan_type, actual_plan);
    println!("-----");

    assert_eq!(expected_plan, format!("\n{}\n", actual_plan), "{}", plan_type)
}
