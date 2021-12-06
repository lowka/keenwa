use crate::catalog::{Catalog, IndexBuilder, DEFAULT_SCHEMA};
use crate::error::OptimizerError;
use crate::operators::builder::{OrderingOption, OrderingOptions};
use crate::operators::scalar::expr::*;
use crate::operators::scalar::value::ScalarValue;
use crate::operators::scalar::{ScalarExpr};

use crate::rules::implementation::*;
use crate::rules::transformation::*;
use crate::testing::OptimizerTester;

fn filter_expr(left: &str, value: ScalarValue) -> Option<ScalarExpr> {
    let expr = ScalarExpr::BinaryExpr {
        lhs: Box::new(ScalarExpr::ColumnName(left.into())),
        op: BinaryOp::Gt,
        rhs: Box::new(ScalarExpr::Scalar(value)),
    };
    Some(expr)
}

#[test]
fn test_get() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let from_a = tester.builder().get("A", vec!["a1", "a2"])?;
    let projection = from_a.project_cols(vec!["a1", "a2"])?;

    let query = tester.build_query(projection);
    tester.set_table_access_cost("A", 100);

    tester.optimize(
        query,
        r#"
01 Projection [00] cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_join() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder();
    let right = tester.builder();

    let join = left.get("A", vec!["a1", "a2"])?;
    let right = right.get("B", vec!["b1", "b2"])?;

    let join = join.join_using(right, vec![("a1", "b1")])?;
    let project = join.project_cols(vec!["a1", "a2", "b1"])?;
    let query = tester.build_query(project);

    tester.add_rules(|_| vec![Box::new(HashJoinRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 120);

    tester.optimize(
        query,
        r#"
03 Projection [02] cols=[1, 2, 3]
02 HashJoin [00 01] using=[(1, 3)]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_select() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let builder = tester.builder();

    let filter = filter_expr("a1", ScalarValue::Int32(10));
    let select = builder.get("A", vec!["a1", "a2"])?.select(filter)?;
    let query = tester.build_query(select);

    tester.set_table_access_cost("A", 100);

    tester.optimize(
        query,
        r#"
02 Select [00 01]
01 Expr col:1 > 10
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_select_with_a_nested_query() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let builder = tester.builder();

    let from_a = builder.get("A", vec!["a1", "a2"])?;
    let sub_query = tester.builder().sub_query_builder().get("B", vec!["b2"])?.to_sub_query()?;

    let filter = ScalarExpr::BinaryExpr {
        lhs: Box::new(sub_query),
        op: BinaryOp::Gt,
        rhs: Box::new(ScalarExpr::Scalar(ScalarValue::Int32(1))),
    };
    let select = from_a.select(Some(filter))?;
    let query = tester.build_query(select);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 100);

    tester.optimize(
        query,
        r#"
03 Select [00 02]
02 Expr SubQuery 01 > 1
01 Scan B cols=[3]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_get_ordered_top_level_enforcer() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let builder = tester.builder();
    builder.set_selectivity("col:a1 > 10", 0.1);

    let from_a = builder.get("A", vec!["a1", "a2"])?;
    let select = from_a.select(filter_expr("a1", ScalarValue::Int32(10)))?;
    let select = select.order_by(OrderingOption::by(("a2", false)))?;

    let query = tester.build_query(select);

    tester.set_table_access_cost("A", 100);

    tester.optimize(
        query,
        r#"
02 Sort [02] ord=[2]
02 Select [00 01]
01 Expr col:1 > 10
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_get_ordered_no_top_level_enforcer() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let builder = tester.builder();
    builder.set_selectivity("col:a1 > 10", 0.1);

    let from_a = builder.get("A", vec!["a1", "a2"])?;
    let select = from_a.select(filter_expr("a1", ScalarValue::Int32(10)))?;
    let select = select.order_by(OrderingOption::by(("a2", false)))?;

    let query = tester.build_query(select);

    tester.set_table_access_cost("A", 100);

    tester.explore_with_enforcer(false);
    tester.optimize(
        query,
        r#"
02 Select [ord:[2]=00 ord:[2]=01]
01 Expr col:1 > 10
00 Sort [00] ord=[2]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_get_ordered() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let from_a = tester.builder().get("A", vec!["a1", "a2"])?;
    let ordered = from_a.order_by(OrderingOption::by(("a1", false)))?;

    let query = tester.build_query(ordered);

    tester.add_rules(|_| vec![Box::new(SelectRule)]);

    tester.set_table_access_cost("A", 100);
    tester.optimize(
        query,
        r#"
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_join_commutativity() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1"])?;
    let right = tester.builder().get("B", vec!["b1"])?;

    let join = left.join_using(right, vec![("a1", "b1")])?;

    let select = join.select(filter_expr("a1", ScalarValue::Int32(10)))?;
    let query = tester.build_query(select);

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 10);

    tester.optimize(
        query,
        r#"
04 Select [02 03]
03 Expr col:1 > 10
02 HashJoin [01 00] using=[(2, 1)]
00 Scan A cols=[1]
01 Scan B cols=[2]
"#,
    )
}

#[test]
fn test_join_commutativity_ordered() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1"])?;
    let right = tester.builder().get("B", vec!["b1"])?;

    let join = left.join_using(right, vec![("a1", "b1")])?;
    join.set_selectivity("col:a1 > 10", 0.1);

    let select = join.select(filter_expr("a1", ScalarValue::Int32(10)))?;
    let ordered = select.order_by(OrderingOption::by(("a1", false)))?;

    let query = tester.build_query(ordered);

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 110);

    tester.optimize(
        query,
        r#"
04 Sort [04] ord=[1]
04 Select [02 03]
03 Expr col:1 > 10
02 HashJoin [00 01] using=[(1, 2)]
01 Scan B cols=[2]
00 Scan A cols=[1]
"#,
    )
}

#[test]
fn test_prefer_already_sorted_data() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let from_a = tester.builder().get("A", vec!["a1", "a2"])?;
    let ordering = vec![OrderingOption::by(("a1", false)), OrderingOption::by(("a2", false))];
    let ordered = from_a.order_by(OrderingOptions::new(ordering))?;

    tester.add_rules(|catalog| vec![Box::new(IndexOnlyScanRule::new(catalog))]);
    tester.update_catalog(|catalog| {
        let table = catalog.get_table("A").expect("Table does not exist");

        let index = IndexBuilder::new("my_index")
            .table("A")
            .add_column(table.get_column("a1").unwrap())
            .add_column(table.get_column("a2").unwrap())
            .build();

        catalog.add_index(DEFAULT_SCHEMA, index);
    });

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("Index:A", 20);

    let query = tester.build_query(ordered);

    tester.optimize(
        query,
        r#"
00 IndexScan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_merge_join_requires_sorted_inputs() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;
    let right = tester.builder().get("B", vec!["b1", "b2"])?;

    let join = left.join_using(right, vec![("a1", "b2")])?;
    let query = tester.build_query(join);

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 100);

    tester.optimize(
        query,
        r#"
02 MergeSortJoin [ord:[1]=00 ord:[4]=01] using=[(1, 4)]
01 Sort [01] ord=[4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_merge_join_satisfies_ordering_requirements() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;
    let right = tester.builder().get("B", vec!["b1", "b2"])?;

    let join = left.join_using(right, vec![("a1", "b2")])?;
    let ordered = join.order_by(OrderingOption::by(("a1", false)))?;

    let query = tester.build_query(ordered);

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 100);

    tester.optimize(
        query,
        r#"
02 MergeSortJoin [ord:[1]=00 ord:[4]=01] using=[(1, 4)]
01 Sort [01] ord=[4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_merge_join_does_no_satisfy_ordering_requirements() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;
    let right = tester.builder().get("B", vec!["b1", "b2"])?;

    let join = left.join_using(right, vec![("a1", "b2")])?;
    let ordered = join.order_by(OrderingOption::by(("b1", false)))?;

    let query = tester.build_query(ordered);

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 100);

    tester.optimize(
        query,
        r#"
02 Sort [02] ord=[3]
02 MergeSortJoin [ord:[1]=00 ord:[4]=01] using=[(1, 4)]
01 Sort [01] ord=[4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_self_joins() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;
    let right = tester.builder().get("A", vec!["a1", "a2"])?;

    let join = left.join_using(right, vec![("a1", "a1")])?;
    let ordered = join.order_by(OrderingOption::by(("a1", false)))?;
    let query = tester.build_query(ordered.clone());

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 50);

    tester.optimize(
        query,
        r#"
01 Sort [01] ord=[1]
01 HashJoin [00 00] using=[(1, 1)]
00 Scan A cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    )?;

    tester.disable_rules(|r| r.name() == "HashJoinRule");
    let query = tester.build_query(ordered);

    tester.optimize(
        query,
        r#"
01 MergeSortJoin [ord:[1]=00 ord:[1]=00] using=[(1, 1)]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_self_joins_inner_sort_should_be_ignored() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;
    let right = tester.builder().get("A", vec!["a1", "a2"])?;

    let join = left.join_using(right, vec![("a1", "a1")])?;
    let inner = join.order_by(OrderingOption::by(("a1", false)))?;

    let right = tester.builder().get("A", vec!["a1", "a2"])?;
    let join = inner.join_using(right, vec![("a1", "a1")])?;
    let ordered = join.order_by(OrderingOption::by(("a1", false)))?;

    let query = tester.build_query(ordered.clone());

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 50);

    tester.disable_rules(|r| r.name() == "HashJoinRule");

    tester.optimize(
        query,
        r#"
02 MergeSortJoin [ord:[1]=01 ord:[1]=00] using=[(1, 1)]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
01 MergeSortJoin [ord:[1]=00 ord:[1]=00] using=[(1, 1)]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    )?;

    tester.reset_rule_filters();

    let query = tester.build_query(ordered);
    tester.disable_rules(|r| r.name() == "MergeSortJoinRule");

    // Ignore ordering requirement from inner node because parent node also requires ordering
    tester.optimize(
        query,
        r#"
02 Sort [02] ord=[1]
02 HashJoin [01 00] using=[(1, 1)]
00 Scan A cols=[1, 2]
01 HashJoin [00 00] using=[(1, 1)]
00 Scan A cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_inner_sort_with_enforcer() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;
    let right = tester.builder().get("A", vec!["a1", "a2"])?;
    let join = left.join_using(right, vec![("a1", "a1")])?;
    let ordered = join.order_by(OrderingOption::by(("a1", false)))?;
    let select = ordered.select(filter_expr("a1", ScalarValue::Int32(10)))?;

    let query = tester.build_query(select);

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(SelectRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 50);

    tester.optimize(
        query,
        r#"
03 Select [ord:[1]=01 02]
02 Expr col:1 > 10
01 Sort [01] ord=[1]
01 HashJoin [00 00] using=[(1, 1)]
00 Scan A cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_inner_sort_satisfied_by_ordering_providing_operator() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;
    let right = tester.builder().get("A", vec!["a1", "a2"])?;
    let join = left.join_using(right, vec![("a1", "a1")])?;
    let ordered = join.order_by(OrderingOption::by(("a1", false)))?;

    ordered.set_selectivity("col:a1 > 10", 0.1);

    let select = ordered.select(filter_expr("a1", ScalarValue::Int32(10)))?;

    let query = tester.build_query(select);

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule), Box::new(SelectRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 50);

    tester.optimize(
        query,
        r#"
03 Select [ord:[1]=01 02]
02 Expr col:1 > 10
01 MergeSortJoin [ord:[1]=00 ord:[1]=00] using=[(1, 1)]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_join_associativity_ax_bxc() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;

    let right = tester.builder().get("B", vec!["b1", "b2"])?;
    let from_c = tester.builder().get("C", vec!["c1", "c2"])?;
    let right = right.join_using(from_c, vec![("b1", "c2")])?;

    let join = left.join_using(right, vec![("a1", "b1")])?;
    let query = tester.build_query(join.clone());

    tester
        .add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinAssociativityRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 200);
    tester.set_table_access_cost("C", 500);

    tester.disable_rules(|r| r.name() == "JoinCommutativityRule");

    tester.optimize(
        query,
        r#"query: Ax[BxC] => [AxB]xC
04 HashJoin [05 02] using=[(1, 6)]
02 Scan C cols=[5, 6]
05 HashJoin [00 01] using=[(1, 3)]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    )?;
    tester.reset_rule_filters();
    let query = tester.build_query(join.clone());

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 150);
    tester.set_table_access_cost("C", 250);

    tester.optimize(
        query,
        r#"query: Ax[BxC] => Ax[BxC]
04 HashJoin [00 03] using=[(1, 3)]
03 HashJoin [01 02] using=[(3, 6)]
02 Scan C cols=[5, 6]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    )?;

    tester.set_table_access_cost("A", 250);
    tester.set_table_access_cost("B", 100);
    tester.set_table_access_cost("C", 150);

    let query = tester.build_query(join);

    tester.optimize(
        query,
        r#"query: Ax[BxC] => [BxC]xA
04 HashJoin [03 00] using=[(3, 1)]
00 Scan A cols=[1, 2]
03 HashJoin [01 02] using=[(3, 6)]
02 Scan C cols=[5, 6]
01 Scan B cols=[3, 4]
"#,
    )
}

#[test]
fn test_join_associativity_axb_xc() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;
    let from_b = tester.builder().get("B", vec!["b1", "b2"])?;
    let left = left.join_using(from_b, vec![("a1", "b2")])?;

    let from_c = tester.builder().get("C", vec!["c1", "c2"])?;
    let join = left.join_using(from_c, vec![("a1", "c2")])?;

    let query = tester.build_query(join);

    tester
        .add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinAssociativityRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_access_cost("A", 500);
    tester.set_table_access_cost("B", 200);
    tester.set_table_access_cost("C", 300);

    tester.disable_rules(|r| r.name() == "JoinCommutativityRule");

    tester.optimize(
        query,
        r#"query: [AxB]xC => Ax[BxC]
04 HashJoin [00 05] using=[(1, 4)]
05 HashJoin [01 03] using=[(4, 6)]
03 Scan C cols=[5, 6]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_enforce_grouping() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let builder = tester.builder();
    let mut from_a = builder.get("A", vec!["a1", "a2"])?;
    let aggr = from_a.aggregate_builder().add_func("count", "a1")?.group_by("a2")?.build()?;

    let query = tester.build_query(aggr);

    tester.add_rules(|_| vec![Box::new(HashAggregateRule)]);

    tester.set_table_access_cost("A", 100);

    tester.optimize(
        query,
        r#"
03 HashAggregate [00 01 02]
02 Expr col:2
01 Expr count(col:1)
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_union() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let from_a = tester.builder().get("A", vec!["a1", "a2"])?;
    let from_b = tester.builder().get("B", vec!["b1", "b2"])?;

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 100);

    tester.add_rules(|_| vec![Box::new(UnionRule)]);

    let union = from_a.clone().union(from_b.clone())?;
    let union = tester.build_query(union);

    tester.optimize(
        union,
        r#"query: union all=false -> unique
02 Unique [ord:[1, 2]=00 ord:[3, 4]=01]
01 Sort [01] ord=[3, 4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[1, 2]
00 Scan A cols=[1, 2]
"#,
    )?;

    let union = from_a.union_all(from_b)?;
    let union = tester.build_query(union);

    tester.optimize(
        union,
        r#"query: union all=true -> append
02 Append [00 01]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_nested_loop_join() -> Result<(), OptimizerError> {
    let mut tester = OptimizerTester::new();

    let left = tester.builder().get("A", vec!["a1", "a2"])?;
    let right = tester.builder().get("B", vec!["b1", "b2"])?;

    let expr = ScalarExpr::BinaryExpr {
        lhs: Box::new(ScalarExpr::ColumnName("a1".into())),
        op: BinaryOp::Gt,
        rhs: Box::new(ScalarExpr::Scalar(ScalarValue::Int32(100))),
    };
    let join = left.join_on(right, expr)?;
    let projection = join.project_cols(vec!["a1", "a2", "b1"])?;
    let query = tester.build_query(projection);

    tester.add_rules(|_| vec![Box::new(NestedLoopJoin)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 120);

    tester.optimize(
        query,
        r#"
04 Projection [03] cols=[1, 2, 3]
03 NestedLoopJoin [00 01 02]
02 Expr col:1 > 100
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    )
}
