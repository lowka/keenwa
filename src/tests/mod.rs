use crate::catalog::{Catalog, IndexBuilder, OrderingBuilder, DEFAULT_SCHEMA};
use crate::operators::builder::{OrderingOption, OrderingOptions, PartitioningOptions};
use crate::operators::relational::join::JoinType;
use crate::operators::relational::RelNode;
use crate::operators::scalar::{col, cols, scalar, ScalarExpr};
use crate::rules::implementation::aggregate::{HashAggregateRule, StreamingAggregateRule};
use crate::rules::implementation::*;
use crate::rules::transformation::*;
use crate::testing::OptimizerTester;

#[test]
fn test_get() {
    let mut tester = OptimizerTester::new();

    tester.set_table_row_count("A", 100);

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let columns = cols(vec!["a1", "a2"]);
        let projection = from_a.project(columns)?;

        projection.build()
    });

    tester.optimize(
        r#"
03 Projection [00 01 02] cols=[1, 2]
02 Expr col:2
01 Expr col:1
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

        let join = join.join_using(right, JoinType::Inner, vec![("a1", "b1")])?;
        let columns = cols(vec!["a1", "a2", "b1"]);
        let project = join.project(columns)?;

        project.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 120);

    tester.optimize(
        r#"
06 Projection [02 03 04 05] cols=[1, 2, 3]
05 Expr col:3
04 Expr col:2
03 Expr col:1
02 HashJoin [00 01] type=Inner using=[(1, 3)]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_select() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let filter = col("a1").gt(scalar(10));
        let select = builder.get("A", vec!["a1", "a2"])?.select(Some(filter))?;

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
        let sub_query = builder.new_query_builder().get("B", vec!["b2"])?.build()?;
        let sub_query = ScalarExpr::SubQuery(RelNode::try_from(sub_query)?);
        let filter = sub_query.gt(scalar(1));
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
fn test_select_with_ordering_and_nested_query() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
        let sub_query = builder.new_query_builder().get("B", vec!["b2"])?.build()?;
        let sub_query = ScalarExpr::SubQuery(RelNode::try_from(sub_query)?);
        let filter = sub_query.gt(scalar(1));
        let select = from_a.select(Some(filter))?;
        let ordered = select.order_by(OrderingOption::by("a1", false))?;

        ordered.build()
    });

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
03 Select [ord:[+1]=01 02]
02 Expr SubQuery 00 > 1
00 Scan B cols=[3]
01 Sort [01] ord=[+1]
01 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_select_with_ordering_in_nested_query() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.clone().get("A", vec!["a1", "a2"])?;

        let ordered_sub_query = builder
            .new_query_builder()
            .get("B", vec!["b2"])?
            .order_by(OrderingOption::by("b2", true))?
            .build()?;

        let sub_query = ScalarExpr::SubQuery(RelNode::try_from(ordered_sub_query)?);

        let filter = sub_query.gt(scalar(1));
        let select = from_a.select(Some(filter))?;

        select.build()
    });

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
03 Select [01 02]
02 Expr SubQuery 00 > 1
00 Sort [00] ord=[-3]
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
        let filter = col("a1").gt(scalar(10));
        let select = from_a.select(Some(filter))?;
        let select = select.order_by(OrderingOption::by("a2", false))?;

        select.build()
    });

    tester.set_table_row_count("A", 100);
    tester.update_statistics(|p| p.set_selectivity("col:a1 > 10", 0.1));

    tester.optimize(
        r#"
02 Sort [02] ord=[+2]
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
        let filter = col("a1").gt(scalar(10));
        let select = from_a.select(Some(filter))?;
        let select = select.order_by(OrderingOption::by("a2", false))?;

        select.build()
    });

    tester.set_table_row_count("A", 100);
    tester.update_statistics(|p| p.set_selectivity("col:a1 > 10", 0.1));

    tester.explore_with_enforcer(false);
    tester.optimize(
        r#"
02 Select [ord:[+2]=00 01]
01 Expr col:1 > 10
00 Sort [00] ord=[+2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_pass_ordering_through_ordering_preserving_operators() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let projection = from_a.project(vec![col("a1"), col("a2")])?;
        let filter = col("a1").gt(scalar(10));
        let select = projection.select(Some(filter))?;
        let select = select.order_by(OrderingOption::by("a1", false))?;

        select.build()
    });

    tester.add_rules(|catalog| vec![Box::new(IndexOnlyScanRule::new(catalog))]);
    tester.update_catalog(|catalog| {
        let table = catalog.get_table("A").expect("Table does not exist");
        let index_ordering = OrderingBuilder::new(table.clone()).add_asc("a1").add_asc("a2").build()?;

        let index = IndexBuilder::new(table, "my_index")
            .add_column("a1")
            .add_column("a2")
            .ordering(index_ordering)
            .build()?;

        catalog.add_index(DEFAULT_SCHEMA, index)?;
        Ok(())
    });

    tester.update_statistics(|p| p.set_selectivity("col:a1 > 10", 0.1));
    tester.set_table_row_count("A", 100);

    tester.explore_with_enforcer(false);
    tester.optimize(
        r#"
05 Select [ord:[+1]=03 04]
04 Expr col:1 > 10
03 Projection [ord:[+1]=00 01 02] cols=[1, 2]
02 Expr col:2
01 Expr col:1
00 IndexScan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_get_ordered() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let ordered = from_a.order_by(OrderingOption::by("a1", false))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(SelectRule)]);

    tester.set_table_row_count("A", 100);
    tester.optimize(
        r#"
00 Sort [00] ord=[+1]
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

        let join = left.join_using(right, JoinType::Inner, vec![("a1", "b1")])?;
        let filter = col("a1").gt(scalar(10));
        let select = join.select(Some(filter))?;

        select.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 10);

    tester.optimize(
        r#"
04 Select [02 03]
03 Expr col:1 > 10
02 HashJoin [01 00] type=Inner using=[(2, 1)]
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

        let join = left.join_using(right, JoinType::Inner, vec![("a1", "b1")])?;
        let filter = col("a1").gt(scalar(10));
        let select = join.select(Some(filter))?;
        let ordered = select.order_by(OrderingOption::by("a1", false))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 110);
    tester.update_statistics(|p| p.set_selectivity("col:a1 > 10", 0.1));

    tester.optimize(
        r#"
04 Sort [04] ord=[+1]
04 Select [02 03]
03 Expr col:1 > 10
02 HashJoin [00 01] type=Inner using=[(1, 2)]
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
        let ordering = vec![OrderingOption::by("a1", false), OrderingOption::by("a2", false)];
        let ordered = from_a.order_by(OrderingOptions::new(ordering))?;

        ordered.build()
    });

    tester.add_rules(|catalog| vec![Box::new(IndexOnlyScanRule::new(catalog))]);
    tester.update_catalog(|catalog| {
        let table = catalog.get_table("A").expect("Table does not exist");
        let index_ordering = OrderingBuilder::new(table.clone()).add_asc("a1").add_asc("a2").build()?;

        let index = IndexBuilder::new(table, "my_index")
            .add_column("a1")
            .add_column("a2")
            .ordering(index_ordering)
            .build()?;

        catalog.add_index(DEFAULT_SCHEMA, index).unwrap();
        Ok(())
    });

    tester.set_table_row_count("A", 100);
    // tester.set_table_row_count("Index:A", 20);

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
        let join = left.join_using(right, JoinType::Inner, vec![("a1", "b2")])?;

        join.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
02 MergeSortJoin [ord:[+1]=00 ord:[+4]=01] type=Inner using=[(1, 4)] left_ord=[+1] right_ord=[+4]
01 Sort [01] ord=[+4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[+1]
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

        let join = left.join_using(right, JoinType::Inner, vec![("a1", "b2")])?;
        let ordered = join.order_by(OrderingOption::by("a1", false))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
02 MergeSortJoin [ord:[+1]=00 ord:[+4]=01] type=Inner using=[(1, 4)] left_ord=[+1] right_ord=[+4]
01 Sort [01] ord=[+4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[+1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_merge_join_satisfies_multi_column_ordering_requirements() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("B", vec!["b1", "b2"])?;

        let join = left.join_using(right, JoinType::Inner, vec![("a1", "b2"), ("a2", "b1")])?;
        let a1_asc = OrderingOption::by("a1", false);
        let b2_desc = OrderingOption::by("b1", true);
        let ordered = join.order_by(OrderingOptions::new(vec![a1_asc, b2_desc]))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
02 MergeSortJoin [ord:[+1, -2]=00 ord:[+4, -3]=01] type=Inner using=[(1, 4), (2, 3)] left_ord=[+1, -2] right_ord=[+4, -3]
01 Sort [01] ord=[+4, -3]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[+1, -2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_merge_join_does_not_satisfy_ordering_requirements() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("B", vec!["b1", "b2"])?;

        let join = left.join_using(right, JoinType::Inner, vec![("a1", "b2")])?;
        let ordered = join.order_by(OrderingOption::by("b1", false))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.optimize(
        r#"
02 Sort [02] ord=[+3]
02 MergeSortJoin [ord:[+1]=00 ord:[+4]=01] type=Inner using=[(1, 4)] left_ord=[+1] right_ord=[+4]
01 Sort [01] ord=[+4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[+1]
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

        let join = left.join_using(right, JoinType::Inner, vec![("a1", "a1")])?;
        let ordered = join.order_by(OrderingOption::by("a1", false))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 50);

    tester.optimize(
        r#"
01 Sort [01] ord=[+1]
01 HashJoin [00 00] type=Inner using=[(1, 1)]
00 Scan A cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );

    tester.disable_rules(|r| r.name() == "HashJoinRule");

    tester.optimize(
        r#"
01 MergeSortJoin [ord:[+1]=00 ord:[+1]=00] type=Inner using=[(1, 1)] left_ord=[+1] right_ord=[+1]
00 Sort [00] ord=[+1]
00 Scan A cols=[1, 2]
00 Sort [00] ord=[+1]
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

        let join = left.join_using(right, JoinType::Inner, vec![("a1", "a1")])?;
        let inner = join.order_by(OrderingOption::by("a1", false))?;

        let right = builder.get("A", vec!["a1", "a2"])?;
        let join = inner.join_using(right, JoinType::Inner, vec![("a1", "a1")])?;
        let ordered = join.order_by(OrderingOption::by("a1", false))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 50);

    tester.disable_rules(|r| r.name() == "HashJoinRule");

    tester.optimize(
        r#"
02 MergeSortJoin [ord:[+1]=01 ord:[+1]=00] type=Inner using=[(1, 1)] left_ord=[+1] right_ord=[+1]
00 Sort [00] ord=[+1]
00 Scan A cols=[1, 2]
01 MergeSortJoin [ord:[+1]=00 ord:[+1]=00] type=Inner using=[(1, 1)] left_ord=[+1] right_ord=[+1]
00 Sort [00] ord=[+1]
00 Scan A cols=[1, 2]
00 Sort [00] ord=[+1]
00 Scan A cols=[1, 2]
"#,
    );

    tester.reset_rule_filters();

    tester.disable_rules(|r| r.name() == "MergeSortJoinRule");

    // Ignore ordering requirement from inner node because parent node also requires ordering
    tester.optimize(
        r#"
02 Sort [02] ord=[+1]
02 HashJoin [01 00] type=Inner using=[(1, 1)]
00 Scan A cols=[1, 2]
01 HashJoin [00 00] type=Inner using=[(1, 1)]
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
        let join = left.join_using(right, JoinType::Inner, vec![("a1", "a1")])?;
        let ordered = join.order_by(OrderingOption::by("a1", false))?;
        let filter = col("a1").gt(scalar(10));
        let select = ordered.select(Some(filter))?;

        select.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(SelectRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 50);

    tester.optimize(
        r#"
03 Select [ord:[+1]=01 02]
02 Expr col:1 > 10
01 Sort [01] ord=[+1]
01 HashJoin [00 00] type=Inner using=[(1, 1)]
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
        let join = left.join_using(right, JoinType::Inner, vec![("a1", "a1")])?;
        let ordered = join.order_by(OrderingOption::by("a1", false))?;
        let filter = col("a1").gt(scalar(10));
        let select = ordered.select(Some(filter))?;

        select.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule), Box::new(SelectRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 50);
    tester.update_statistics(|p| p.set_selectivity("col:a1 > 10", 0.1));

    tester.optimize(
        r#"
03 Select [ord:[+1]=01 02]
02 Expr col:1 > 10
01 MergeSortJoin [ord:[+1]=00 ord:[+1]=00] type=Inner using=[(1, 1)] left_ord=[+1] right_ord=[+1]
00 Sort [00] ord=[+1]
00 Scan A cols=[1, 2]
00 Sort [00] ord=[+1]
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
        let right = right.join_using(from_c, JoinType::Inner, vec![("b1", "c2")])?;
        let join = left.join_using(right, JoinType::Inner, vec![("a1", "b1")])?;

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
04 HashJoin [05 01] type=Inner using=[(1, 6)]
01 Scan C cols=[5, 6]
05 HashJoin [02 00] type=Inner using=[(1, 3)]
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
04 HashJoin [02 03] type=Inner using=[(1, 3)]
03 HashJoin [00 01] type=Inner using=[(3, 6)]
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
04 HashJoin [03 02] type=Inner using=[(3, 1)]
02 Scan A cols=[1, 2]
03 HashJoin [00 01] type=Inner using=[(3, 6)]
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
        let left = left.join_using(from_b, JoinType::Inner, vec![("a1", "b2")])?;
        let from_c = builder.get("C", vec!["c1", "c2"])?;
        let join = left.join_using(from_c, JoinType::Inner, vec![("a1", "c2")])?;

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
04 HashJoin [00 05] type=Inner using=[(1, 4)]
05 HashJoin [01 03] type=Inner using=[(4, 6)]
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
03 HashAggregate [00 01 02] cols=[3]
02 Expr col:2
01 Expr count(col:1)
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_prefer_streaming_aggregate_when_data_is_sorted() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let sorted_a = from_a.order_by(OrderingOption::by("a2", false))?;
        let mut projection = sorted_a.project(vec![col("a1"), col("a2")])?;
        let aggr = projection.aggregate_builder().add_func("count", "a1")?.group_by("a2")?.build()?;

        aggr.build()
    });

    tester.add_rules(|_| vec![Box::new(HashAggregateRule), Box::new(StreamingAggregateRule)]);

    tester.set_table_row_count("A", 100);

    tester.optimize(
        r#"
05 StreamingAggregate [ord:[+2]=03 04 02] cols=[3] ordering=[+2]
02 Expr col:2
04 Expr count(col:1)
03 Projection [ord:[+2]=00 01 02] cols=[1, 2]
02 Expr col:2
01 Expr col:1
00 Sort [00] ord=[+2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_streaming_aggregate_provides_ordering() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let sorted_a = from_a.order_by(OrderingOption::by("a2", false))?;
        let mut projection = sorted_a.project(vec![col("a1"), col("a2")])?;
        let aggr = projection
            .aggregate_builder()
            .add_column("a2")?
            .add_func("count", "a1")?
            .group_by("a2")?
            .build()?;
        let ordered = aggr.order_by(OrderingOption::by("a2", false))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(HashAggregateRule), Box::new(StreamingAggregateRule)]);

    tester.set_table_row_count("A", 100);

    tester.optimize(
        r#"
05 StreamingAggregate [ord:[+2]=03 02 04 02] cols=[2, 3] ordering=[+2]
02 Expr col:2
04 Expr count(col:1)
02 Expr col:2
03 Projection [ord:[+2]=00 01 02] cols=[1, 2]
02 Expr col:2
01 Expr col:1
00 Sort [00] ord=[+2]
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
02 Unique [ord:[+1, +2]=00 ord:[+3, +4]=01] cols=[5, 6] left_ord=[+1, +2] right_ord=[+3, +4]
01 Sort [01] ord=[+3, +4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[+1, +2]
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
02 Append [00 01] cols=[5, 6]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    )
}

#[test]
fn test_unique_satisfies_ordering_requirements() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
        let from_b = builder.get("B", vec!["b1", "b2"])?;
        let union = from_a.union(from_b)?;
        let a1_asc = OrderingOption::by("a1", false);
        let a2_desc = OrderingOption::by("a2", true);
        let ordering = OrderingOptions::new(vec![a1_asc, a2_desc]);
        let ordered = union.order_by(ordering)?;

        ordered.build()
    });

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 100);

    tester.add_rules(|_| vec![Box::new(UnionRule)]);

    tester.optimize(
        r#"
02 Unique [ord:[+1, -2]=00 ord:[+3, -4]=01] cols=[5, 6] left_ord=[+1, -2] right_ord=[+3, -4]
01 Sort [01] ord=[+3, -4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[+1, -2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_nested_loop_join() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1", "a2"])?;
        let right = builder.get("B", vec!["b1", "b2"])?;

        let filter = col("a1").gt(scalar(100));
        let join = left.join_on(right, JoinType::Inner, filter)?;
        let columns = cols(vec!["a1", "a2", "b1"]);
        let projection = join.project(columns)?;

        projection.build()
    });

    tester.add_rules(|_| vec![Box::new(NestedLoopJoinRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 120);

    tester.optimize(
        r#"
07 Projection [03 04 05 06] cols=[1, 2, 3]
06 Expr col:3
05 Expr col:2
04 Expr col:1
03 NestedLoopJoin type=Inner [00 01 02]
02 Expr col:1 > 100
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_select_1() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from = builder.empty(true)?;
        let project = from.project(vec![scalar(1)])?;
        let select = project.select(None)?;

        select.build()
    });

    tester.optimize(
        r#"
03 Select [02]
02 Projection [00 01] cols=[1]
01 Expr 1
00 Empty return_one_row=true
"#,
    );
}

#[test]
fn test_distinct_on_expr_as_unique() {
    let mut tester = OptimizerTester::new();
    tester.set_operator(|builder| {
        let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
        let distinct = from_a.distinct(Some(col("a1")))?;

        distinct.build()
    });

    tester.add_rules(|_| vec![Box::new(DistinctRule)]);
    tester.set_table_row_count("A", 100);

    tester.optimize(
        r#"
02 Unique [ord:[+1, +2]=00 01] cols=[1, 2] ord=[+1, +2]
01 Expr col:1
00 Sort [00] ord=[+1, +2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_distinct_hash_aggregate() {
    let mut tester = OptimizerTester::new();
    tester.set_operator(|builder| {
        let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
        let distinct = from_a.distinct(None)?;

        distinct.build()
    });

    tester.add_rules(|_| vec![Box::new(DistinctRule)]);
    tester.set_table_row_count("A", 100);

    tester.optimize(
        r#"
01 HashAggregate [00 02 03 02 03] cols=[1, 2]
03 Expr col:2
02 Expr col:1
03 Expr col:2
02 Expr col:1
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_limit_offset() {
    let mut tester = OptimizerTester::new();
    tester.set_operator(|builder| {
        let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
        let result = from_a.offset(10)?.limit(4)?;

        result.build()
    });

    tester.add_rules(|_| vec![Box::new(DistinctRule)]);
    tester.set_table_row_count("A", 100);

    tester.optimize(
        r#"
02 Limit [01] rows=4
01 Offset [00] rows=10
00 Scan A cols=[1, 2]
"#,
    );
}

// PARTITIONING

#[test]
fn test_add_initial_partitioning_after_not_partitioned_input() {
    let mut tester = OptimizerTester::new();
    tester.enable_partitioning(vec!["hash"]);

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let partitioned = from_a.partition_by(vec!["a1"])?;

        partitioned.build()
    });

    tester.add_rules(|_| vec![Box::new(SelectRule)]);

    tester.set_table_row_count("A", 100);
    tester.optimize(
        r#"
00 InitialPartitioning [partitioning:()=00] hash-partitioning=[1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_add_full_merge_when_serial_input_is_requested() {
    let mut tester = OptimizerTester::new();
    tester.enable_partitioning(vec!["hash"]);

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let partitioned = from_a.partition_by(PartitioningOptions::Singleton)?;

        partitioned.build()
    });

    tester.add_rules(|_| vec![Box::new(SelectRule)]);

    tester.set_table_row_count("A", 100);
    tester.optimize(
        r#"
00 FullMerge [partitioning:hash()=00] partitioning=()
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_full_merge_prefer() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let partitioned = from_a.partition_by(vec!["a1", "a2"])?;
        let ordering = vec![OrderingOption::by("a1", false), OrderingOption::by("a2", false)];
        // let ordered = partitioned.order_by(OrderingOptions::new(ordering))?;

        partitioned.build()
    });

    tester.add_rules(|catalog| vec![Box::new(IndexOnlyScanRule::new(catalog))]);
    tester.update_catalog(|catalog| {
        let table = catalog.get_table("A").expect("Table does not exist");
        let index_ordering = OrderingBuilder::new(table.clone()).add_asc("a1").add_asc("a2").build()?;

        let index = IndexBuilder::new(table, "my_index")
            .add_column("a1")
            .add_column("a2")
            .ordering(index_ordering)
            .build()?;

        catalog.add_index(DEFAULT_SCHEMA, index).unwrap();
        Ok(())
    });

    tester.set_table_row_count("A", 100);
    // tester.set_table_row_count("Index:A", 20);

    tester.optimize(
        r#"
00 InitialPartitioning [partitioning:()=00] hash-partitioning=[1, 2]
00 IndexScan A cols=[1, 2]
"#,
    );

    tester.disable_rules(|r| r.name() == "IndexOnlyScanRule");

    tester.optimize(
        r#"
00 InitialPartitioning [partitioning:()=00] hash-partitioning=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_full_merge_prefer_with_ordering() {
    let mut tester = OptimizerTester::new();
    tester.enable_partitioning(vec!["ord"]);

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let partitioned = from_a.partition_by(vec!["a1", "a2"])?;
        let ordering = vec![OrderingOption::by("a1", false), OrderingOption::by("a2", false)];
        let ordered = partitioned.order_by(OrderingOptions::new(ordering))?;

        ordered.build()
    });

    tester.add_rules(|catalog| vec![Box::new(IndexOnlyScanRule::new(catalog))]);
    tester.update_catalog(|catalog| {
        let table = catalog.get_table("A").expect("Table does not exist");
        let index_ordering = OrderingBuilder::new(table.clone()).add_asc("a1").add_asc("a2").build()?;

        let index = IndexBuilder::new(table, "my_index")
            .add_column("a1")
            .add_column("a2")
            .ordering(index_ordering)
            .build()?;

        catalog.add_index(DEFAULT_SCHEMA, index).unwrap();
        Ok(())
    });

    tester.set_table_row_count("A", 100);
    // tester.set_table_row_count("Index:A", 20);

    //     tester.optimize(
    //         r#"
    // 00 InitialPartitioning [ord:[+1, +2]=00] ord-partitioning=[1, 2]
    // 00 IndexScan A cols=[1, 2]
    // "#,
    //     );

    tester.disable_rules(|r| r.name() == "IndexOnlyScanRule");

    tester.optimize(
        r#"
00 InitialPartitioning [ord:[+1, +2]=00] ord-partitioning=[1, 2]
00 Sort [ord=00] ord=[+1, +2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_add_initial_partitioning_when_partitioning_is_requested() {
    let mut tester = OptimizerTester::new();
    tester.enable_partitioning(vec!["hash"]);

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let projection = from_a.project(vec![col("a1"), col("a2")])?;
        let projection = projection.project(vec![col("a1"), col("a2")])?;
        let partitioned = projection.partition_by(vec!["a1"])?;

        partitioned.build()
    });

    tester.set_table_row_count("A", 100);
    tester.optimize(
        r#"
04 InitialPartitioning [partitioning:()=04] hash-partitioning=[1]
04 Projection [03 01 02] cols=[1, 2]
02 Expr col:2
01 Expr col:1
03 Projection [00 01 02] cols=[1, 2]
02 Expr col:2
01 Expr col:1
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_get_partitioned_and_ordered() {
    let mut tester = OptimizerTester::new();
    tester.enable_partitioning(vec!["hash"]);

    tester.set_operator(|builder| {
        let from_a = builder.get("A", vec!["a1", "a2"])?;
        let partitioned = from_a.partition_by(vec!["a1"])?;
        let ordered = partitioned.order_by(OrderingOption::by("a2", false))?;

        ordered.build()
    });

    tester.add_rules(|_| vec![Box::new(SelectRule)]);
    tester.explore_with_enforcer(false);

    tester.set_table_row_count("A", 100);
    tester.optimize(
        r#"
00 Sort [partitioning:hash(1)=01] ord=[+2]
01 InitialPartitioning [00] hash-partitioning=[1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_add_parallel_requirements_after_join() {
    let mut tester = OptimizerTester::new();
    tester.enable_partitioning(vec!["hash"]);

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1"])?;
        let right = builder.get("B", vec!["b1"])?;

        let join = left.join_using(right, JoinType::Inner, vec![("a1", "b1")])?;
        let project = join.project(vec![col("a1")])?;
        let project = project.partition_by(vec!["a1"])?;

        project.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);
    tester.disable_rules(|r| r.name() == "MergeSortJoinRule");

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 10);

    tester.optimize(
        r#"
04 InitialPartitioning [partitioning:()=04] hash-partitioning=[1]
04 Projection [02 03] cols=[1]
03 Expr col:1
02 HashJoin [00 01] type=Inner using=[(1, 2)]
01 Scan B cols=[2]
00 Scan A cols=[1]
"#,
    );

    tester.reset_rule_filters();
    tester.disable_rules(|r| r.name() == "HashJoinRule");

    tester.optimize(
        r#"
04 InitialPartitioning [partitioning:()=04] hash-partitioning=[1]
04 Projection [02 03] cols=[1]
03 Expr col:1
02 MergeSortJoin [ord:[+1]=00 ord:[+2]=01] type=Inner using=[(1, 2)] left_ord=[+1] right_ord=[+2]
01 Sort [01] ord=[+2]
01 Scan B cols=[2]
00 Sort [00] ord=[+1]
00 Scan A cols=[1]
"#,
    );
}

#[test]
fn test_add_serial_requirements_after_join() {
    let mut tester = OptimizerTester::new();
    tester.enable_partitioning(vec!["hash"]);

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1"])?;
        let right = builder.get("B", vec!["b1"])?;

        let left = left.partition_by(vec!["a1"])?;
        // retain partitioning by adding a projection
        let left = left.project(vec![col("a1")])?;

        let right = right.partition_by(vec!["b1"])?;
        // retain partitioning by adding a projection
        let right = right.project(vec![col("b1")])?;

        let join = left.join_using(right, JoinType::Inner, vec![("a1", "b1")])?;
        let project = join.project(vec![col("a1")])?;
        let project = project.partition_by(PartitioningOptions::Singleton)?;

        project.build()
    });

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);
    tester.disable_rules(|r| r.name() == "MergeSortJoinRule");

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 10);

    tester.optimize(
        r#"
07 FullMerge [partitioning:hash()=07] partitioning=()
07 Projection [06 01] cols=[1]
01 Expr col:1
06 HashJoin [04 05] type=Inner using=[(1, 2)]
05 Projection [partitioning:hash(2)=02 03] cols=[2]
03 Expr col:2
02 InitialPartitioning [partitioning:()=02] hash-partitioning=[2]
02 Scan B cols=[2]
04 Projection [partitioning:hash(1)=00 01] cols=[1]
01 Expr col:1
00 InitialPartitioning [partitioning:()=00] hash-partitioning=[1]
00 Scan A cols=[1]
"#,
    );

    tester.reset_rule_filters();
    tester.disable_rules(|r| r.name() == "HashJoinRule");

    tester.optimize(
        r#"
07 FullMerge [partitioning:hash()=07] partitioning=()
07 Projection [06 01] cols=[1]
01 Expr col:1
06 MergeSortJoin [ord:[+1]=04 ord:[+2]=05] type=Inner using=[(1, 2)] left_ord=[+1] right_ord=[+2]
05 Projection [ord:[+2]=02 03] cols=[2]
03 Expr col:2
02 Sort [02] ord=[+2]
02 InitialPartitioning [partitioning:()=02] ord-partitioning=[2]
02 Scan B cols=[2]
04 Projection [ord:[+1]=00 01] cols=[1]
01 Expr col:1
00 Sort [00] ord=[+1]
00 InitialPartitioning [partitioning:()=00] ord-partitioning=[1]
00 Scan A cols=[1]
"#,
    );
}

#[test]
fn test_merge_join_after_streaming_aggregate() {
    let mut tester = OptimizerTester::new();

    tester.set_operator(|builder| {
        let left = builder.clone().get("A", vec!["a1"])?;
        let right = builder.get("B", vec!["b1"])?;

        let mut join = left.join_using(right, JoinType::Inner, vec![("a1", "b1")])?;
        let aggr = join.aggregate_builder().add_column("a1")?.group_by("a1")?.build()?;
        let aggr = aggr.order_by(OrderingOption::by("a1", false))?;

        let filter = col("a1").gt(scalar(10));
        let select = aggr.select(Some(filter))?;
        let project = select.project(vec![col("a1")])?;

        project.build()
    });

    tester.add_rules(|_| vec![Box::new(MergeSortJoinRule), Box::new(StreamingAggregateRule)]);

    tester.set_table_row_count("A", 100);
    tester.set_table_row_count("B", 10);

    tester.optimize(
        r#"
07 Projection [06 03] cols=[1]
03 Expr col:1
06 Select [ord:[+1]=04 05]
05 Expr col:1 > 10
04 StreamingAggregate [ord:[+1]=02 03 03] cols=[1] ordering=[+1]
03 Expr col:1
03 Expr col:1
02 MergeSortJoin [ord:[+1]=00 ord:[+2]=01] type=Inner using=[(1, 2)] left_ord=[+1] right_ord=[+2]
01 Sort [01] ord=[+2]
01 Scan B cols=[2]
00 Sort [00] ord=[+1]
00 Scan A cols=[1]
"#,
    );
}
