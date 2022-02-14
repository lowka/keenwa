use keenwa::catalog::mutable::MutableCatalog;
use keenwa::catalog::{CatalogRef, TableBuilder, DEFAULT_SCHEMA};
use keenwa::cost::simple::SimpleCostEstimator;
use keenwa::datatypes::DataType;
use keenwa::error::OptimizerError;
use keenwa::operators::format::format_operator_tree;
use keenwa::operators::Operator;
use keenwa::optimizer::{NoOpResultCallback, Optimizer};
use keenwa::rules::implementation::{EmptyRule, GetToScanRule, HashJoinRule, ProjectionRule, SelectRule};
use keenwa::rules::{Rule, StaticRuleSet};
use keenwa::sql::OperatorFromSqlBuilder;
use keenwa::statistics::simple::{DefaultSelectivityStatistics, SimpleCatalogStatisticsBuilder};
use std::rc::Rc;
use std::sync::Arc;

fn main() -> Result<(), OptimizerError> {
    let query = r#"
SELECT a1 + 100, NOT a2 FROM a 
JOIN b ON a1 = b1 
"#;

    // 1. create a catalog.
    let catalog = create_catalog();

    // 2. setup statistics builder.
    let statistics = SimpleCatalogStatisticsBuilder::new(catalog.clone(), DefaultSelectivityStatistics);

    // 3. create operator from sql builder.
    let from_sql_builder = OperatorFromSqlBuilder::new(catalog.clone(), statistics);

    // 4. build an operator tree from the given sql query.
    let (operator, mut memo, _metadata) = from_sql_builder.build(query)?;

    // 5. create an instance of the optimizer.
    let optimizer = create_optimizer(catalog.clone());

    // 6. Optimize the logical plan.
    let result = optimizer.optimize(operator, &mut memo)?;

    let expected_plan = r#"
Projection cols=[5, 6] exprs: [col:1 + 100, NOT col:2]
  input: HashJoin type=Inner on=col:1 = col:4
    left: Scan a cols=[1, 2, 3]
    right: Scan b cols=[4]
"#;

    expect_plan(&result, "Optimized plan", expected_plan);

    Ok(())
}

fn create_catalog() -> CatalogRef {
    let catalog = Arc::new(MutableCatalog::new());

    let table_a = TableBuilder::new("a")
        .add_column("a1", DataType::Int32)
        .add_column("a2", DataType::Bool)
        .add_column("a3", DataType::String)
        .add_row_count(100)
        .build();

    let table_b = TableBuilder::new("b").add_column("b1", DataType::Int32).add_row_count(200).build();

    catalog.add_table(DEFAULT_SCHEMA, table_a);
    catalog.add_table(DEFAULT_SCHEMA, table_b);
    catalog
}

fn create_optimizer(catalog: CatalogRef) -> Optimizer<StaticRuleSet, SimpleCostEstimator, NoOpResultCallback> {
    // Setup transformation/implementation rules.
    let rules: Vec<Box<dyn Rule>> = vec![
        Box::new(GetToScanRule::new(catalog)),
        Box::new(SelectRule),
        Box::new(ProjectionRule),
        Box::new(EmptyRule),
        Box::new(HashJoinRule),
    ];

    let rule_set = StaticRuleSet::new(rules);
    let cost_estimator = SimpleCostEstimator::new();

    Optimizer::new(Rc::new(rule_set), Rc::new(cost_estimator), Rc::new(NoOpResultCallback))
}

fn expect_plan(operator: &Operator, plan_type: &str, expected_plan: &str) {
    let actual_plan = format_operator_tree(&operator);
    println!("{}:\n{}", plan_type, actual_plan);
    println!("-----");

    assert_eq!(expected_plan, format!("\n{}\n", actual_plan), "{}", plan_type)
}
