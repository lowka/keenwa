use crate::catalog::{Catalog, IndexBuilder, DEFAULT_SCHEMA};
use crate::operators::builder::{OperatorBuilder, OrderingOption, OrderingOptions};
use crate::operators::scalar::expr::*;
use crate::operators::scalar::value::ScalarValue;
use crate::operators::scalar::ScalarExpr;

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

fn columns_expr(cols: Vec<impl Into<String>>) -> Vec<ScalarExpr> {
    cols.into_iter().map(|c| ScalarExpr::ColumnName(c.into())).collect()
}

#[test]
fn test_get() {
    let mut tester = OptimizerTester::new();

    tester.set_table_row_count("A", 100);

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let columns = columns_expr(vec!["a1", "a2"]);
        let projection = from_a.project(columns)?;

        projection.build()
    });

    tester.optimize(
        r#"
01 Projection [00] cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_join() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let join = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("B", vec!["b1", "b2"])?;

        let join = join.join_using(right, vec![("a1", "b1")])?;
        let columns = columns_expr(vec!["a1", "a2", "b1"]);
        let project = join.project(columns)?;

        project.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 120);

    tester.optimize(
        r#"
03 Projection [02] cols=[1, 2, 3]
02 HashJoin [00 01] using=[(1, 3)]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_select() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let filter = filter_expr("a1", ScalarValue::Int32(10));
        let select = builder.get("A", vec!["a1", "a2"])?.select(filter)?;

        select.build()
    });

    tester.set_table_row_count("A", 100);

    tester.optimize(
        r#"
02 Select [00 01]
01 Expr col:1 > 10
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_select_with_a_nested_query() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
        let sub_query = builder.sub_query_builder().get("B", vec!["b2"])?.to_sub_query()?;

        let filter = ScalarExpr::BinaryExpr {
            lhs: Box::new(sub_query),
            op: BinaryOp::Gt,
            rhs: Box::new(ScalarExpr::Scalar(ScalarValue::Int32(1))),
        };
        let select = from_a.select(Some(filter))?;

        select.build()
    });

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
03 Select [01 02]
02 Expr SubQuery 00 > 1
00 Scan B cols=[3]
01 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_get_ordered_top_level_enforcer() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let select = from_a.select(filter_expr("a1", ScalarValue::Int32(10)))?;
        let select = select.order_by(OrderingOption::by(("a2", false)))?;

        select.build()
    });

    tester.set_table_row_count("A", 100);
    tester.update_statistics(|p| p.set_selectivity("col:a1 > 10", 0.1));

    tester.optimize(
        r#"
02 Sort [02] ord=[2]
02 Select [00 01]
01 Expr col:1 > 10
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_get_ordered_no_top_level_enforcer() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let select = from_a.select(filter_expr("a1", ScalarValue::Int32(10)))?;
        let select = select.order_by(OrderingOption::by(("a2", false)))?;

        select.build()
    });

    tester.set_table_row_count("A", 100);
    tester.update_statistics(|p| p.set_selectivity("col:a1 > 10", 0.1));

    tester.explore_with_enforcer(false);
    tester.optimize(
        r#"
02 Select [ord:[2]=00 ord:[2]=01]
01 Expr col:1 > 10
00 Sort [00] ord=[2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_get_ordered() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let ordered = from_a.order_by(OrderingOption::by(("a1", false)))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(SelectRule)]);

    tester.set_table_row_count("A", 100);
    tester.optimize(
        r#"
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_join_commutativity() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1"])?;
        let right = builder.get("B", vec!["b1"])?;

        let join = left.join_using(right, vec![("a1", "b1")])?;
        let select = join.select(filter_expr("a1", ScalarValue::Int32(10)))?;

        select.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 10);

    tester.optimize(
        r#"
04 Select [02 03]
03 Expr col:1 > 10
02 HashJoin [01 00] using=[(2, 1)]
00 Scan A cols=[1]
01 Scan B cols=[2]
"#,
    );
}

#[test]
fn test_join_commutativity_ordered() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1"])?;
        let right = builder.get("B", vec!["b1"])?;

        let join = left.join_using(right, vec![("a1", "b1")])?;

        let select = join.select(filter_expr("a1", ScalarValue::Int32(10)))?;
        let ordered = select.order_by(OrderingOption::by(("a1", false)))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 110);
    tester.update_statistics(|p| p.set_selectivity("col:a1 > 10", 0.1));

    tester.optimize(
        r#"
04 Sort [04] ord=[1]
04 Select [02 03]
03 Expr col:1 > 10
02 HashJoin [00 01] using=[(1, 2)]
01 Scan B cols=[2]
00 Scan A cols=[1]
"#,
    );
}

#[test]
fn test_prefer_already_sorted_data() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let ordering = vec![OrderingOption::by(("a1", false)), OrderingOption::by(("a2", false))];
        let ordered = from_a.order_by(OrderingOptions::new(ordering))?;

        ordered.build()
    });

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

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("Index:A", 20);

    tester.optimize(
        r#"
00 IndexScan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_merge_join_requires_sorted_inputs() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("B", vec!["b1", "b2"])?;
        let join = left.join_using(right, vec![("a1", "b2")])?;

        join.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
02 MergeSortJoin [ord:[1]=00 ord:[4]=01] using=[(1, 4)]
01 Sort [01] ord=[4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_merge_join_satisfies_ordering_requirements() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("B", vec!["b1", "b2"])?;

        let join = left.join_using(right, vec![("a1", "b2")])?;
        let ordered = join.order_by(OrderingOption::by(("a1", false)))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
02 MergeSortJoin [ord:[1]=00 ord:[4]=01] using=[(1, 4)]
01 Sort [01] ord=[4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_merge_join_does_no_satisfy_ordering_requirements() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("B", vec!["b1", "b2"])?;

        let join = left.join_using(right, vec![("a1", "b2")])?;
        let ordered = join.order_by(OrderingOption::by(("b1", false)))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
02 Sort [02] ord=[3]
02 MergeSortJoin [ord:[1]=00 ord:[4]=01] using=[(1, 4)]
01 Sort [01] ord=[4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_self_joins() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("A", vec!["a1", "a2"])?;

        let join = left.join_using(right, vec![("a1", "a1")])?;
        let ordered = join.order_by(OrderingOption::by(("a1", false)))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 50);

    tester.optimize(
        r#"
01 Sort [01] ord=[1]
01 HashJoin [00 00] using=[(1, 1)]
00 Scan A cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );

    tester.disable_rules(|r| r.name() == "HashJoinRule");

    tester.optimize(
        r#"
01 MergeSortJoin [ord:[1]=00 ord:[1]=00] using=[(1, 1)]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_self_joins_inner_sort_should_be_ignored() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.clone().get("A", vec!["a1", "a2"])?;

        let join = left.join_using(right, vec![("a1", "a1")])?;
        let inner = join.order_by(OrderingOption::by(("a1", false)))?;

        let right = builder.get("A", vec!["a1", "a2"])?;
        let join = inner.join_using(right, vec![("a1", "a1")])?;
        let ordered = join.order_by(OrderingOption::by(("a1", false)))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 50);

    tester.disable_rules(|r| r.name() == "HashJoinRule");

    tester.optimize(
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
    );

    tester.reset_rule_filters();

    tester.disable_rules(|r| r.name() == "MergeSortJoinRule");

    // Ignore ordering requirement from inner node because parent node also requires ordering
    tester.optimize(
        r#"
02 Sort [02] ord=[1]
02 HashJoin [01 00] using=[(1, 1)]
00 Scan A cols=[1, 2]
01 HashJoin [00 00] using=[(1, 1)]
00 Scan A cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_inner_sort_with_enforcer() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("A", vec!["a1", "a2"])?;
        let join = left.join_using(right, vec![("a1", "a1")])?;
        let ordered = join.order_by(OrderingOption::by(("a1", false)))?;
        let select = ordered.select(filter_expr("a1", ScalarValue::Int32(10)))?;

        select.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(SelectRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 50);

    tester.optimize(
        r#"
03 Select [ord:[1]=01 02]
02 Expr col:1 > 10
01 Sort [01] ord=[1]
01 HashJoin [00 00] using=[(1, 1)]
00 Scan A cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_inner_sort_satisfied_by_ordering_providing_operator() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("A", vec!["a1", "a2"])?;
        let join = left.join_using(right, vec![("a1", "a1")])?;
        let ordered = join.order_by(OrderingOption::by(("a1", false)))?;
        let select = ordered.select(filter_expr("a1", ScalarValue::Int32(10)))?;

        select.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule), Box::new(SelectRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 50);
    tester.update_statistics(|p| p.set_selectivity("col:a1 > 10", 0.1));

    tester.optimize(
        r#"
03 Select [ord:[1]=01 02]
02 Expr col:1 > 10
01 MergeSortJoin [ord:[1]=00 ord:[1]=00] using=[(1, 1)]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_join_associativity_ax_bxc() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;

        let right = builder.clone().get("B", vec!["b1", "b2"])?;
        let from_c = builder.get("C", vec!["c1", "c2"])?;
        let right = right.join_using(from_c, vec![("b1", "c2")])?;
        let join = left.join_using(right, vec![("a1", "b1")])?;

        join.build()
    });

    tester
        .add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinAssociativityRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 200);
    tester.set_table_row_count("C", 500);

    tester.disable_rules(|r| r.name() == "JoinCommutativityRule");

    tester.optimize(
        r#"query: Ax[BxC] => [AxB]xC
04 HashJoin [05 01] using=[(1, 6)]
01 Scan C cols=[5, 6]
05 HashJoin [02 00] using=[(1, 3)]
00 Scan B cols=[3, 4]
02 Scan A cols=[1, 2]
"#,
    );

    tester.reset_rule_filters();

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 150);
    tester.set_table_row_count("C", 250);

    tester.optimize(
        r#"query: Ax[BxC] => Ax[BxC]
04 HashJoin [02 03] using=[(1, 3)]
03 HashJoin [00 01] using=[(3, 6)]
01 Scan C cols=[5, 6]
00 Scan B cols=[3, 4]
02 Scan A cols=[1, 2]
"#,
    );

    tester.set_table_row_count("A", 250);
    tester.set_table_row_count("B", 100);
    tester.set_table_row_count("C", 150);

    tester.optimize(
        r#"query: Ax[BxC] => [BxC]xA
04 HashJoin [03 02] using=[(3, 1)]
02 Scan A cols=[1, 2]
03 HashJoin [00 01] using=[(3, 6)]
01 Scan C cols=[5, 6]
00 Scan B cols=[3, 4]
"#,
    );
}

#[test]
fn test_join_associativity_axb_xc() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let from_b = builder.clone().get("B", vec!["b1", "b2"])?;
        let left = left.join_using(from_b, vec![("a1", "b2")])?;
        let from_c = builder.get("C", vec!["c1", "c2"])?;
        let join = left.join_using(from_c, vec![("a1", "c2")])?;

        join.build()
    });

    tester
        .add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinAssociativityRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_row_count("A", 500);
    tester.set_table_row_count("B", 200);
    tester.set_table_row_count("C", 300);

    tester.disable_rules(|r| r.name() == "JoinCommutativityRule");

    tester.optimize(
        r#"query: [AxB]xC => Ax[BxC]
04 HashJoin [00 05] using=[(1, 4)]
05 HashJoin [01 03] using=[(4, 6)]
03 Scan C cols=[5, 6]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_enforce_grouping() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let mut from_a = builder.get("A", vec!["a1", "a2"])?;
        let aggr = from_a.aggregate_builder().add_func("count", "a1")?.group_by("a2")?.build()?;

        aggr.build()
    });

    tester.add_rules(|_| vec![Box::new(HashAggregateRule)]);

    tester.set_table_row_count("A", 100);

    tester.optimize(
        r#"
03 HashAggregate [00 01 02]
02 Expr col:2
01 Expr count(col:1)
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_union() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
        let from_b = builder.get("B", vec!["b1", "b2"])?;
        let union = from_a.union(from_b)?;

        union.build()
    });

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.add_rules(|_| vec![Box::new(UnionRule)]);

    tester.optimize(
        r#"query: union all=false -> unique
02 Unique [ord:[1, 2]=00 ord:[3, 4]=01]
01 Sort [01] ord=[3, 4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );

    tester.set_operator(|builder| {
        let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
        let from_b = builder.get("B", vec!["b1", "b2"])?;
        let union = from_a.union_all(from_b)?;
        union.build()
    });

    tester.optimize(
        r#"query: union all=true -> append
02 Append [00 01]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_nested_loop_join() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("B", vec!["b1", "b2"])?;

        let expr = ScalarExpr::BinaryExpr {
            lhs: Box::new(ScalarExpr::ColumnName("a1".into())),
            op: BinaryOp::Gt,
            rhs: Box::new(ScalarExpr::Scalar(ScalarValue::Int32(100))),
        };
        let join = left.join_on(right, expr)?;
        let columns = columns_expr(vec!["a1", "a2", "b1"]);
        let projection = join.project(columns)?;

        projection.build()
    });

    tester.add_rules(|_| vec![Box::new(NestedLoopJoin)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 120);

    tester.optimize(
        r#"
04 Projection [03] cols=[1, 2, 3]
03 NestedLoopJoin [00 01 02]
02 Expr col:1 > 100
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    );
}
