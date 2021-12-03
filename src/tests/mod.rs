use crate::catalog::*;
use crate::meta::*;
use crate::operators::expr::*;
use crate::operators::join::*;
use crate::operators::logical::*;
use crate::operators::scalar::ScalarValue;
use crate::operators::{Operator, OperatorExpr, ScalarNode};
use crate::properties::physical::PhysicalProperties;
use crate::properties::statistics::Statistics;
use crate::properties::OrderingChoice;
use crate::rules::implementation::*;
use crate::rules::transformation::*;
use crate::testing::OptimizerTester;

fn ordering(cols: Vec<ColumnId>) -> PhysicalProperties {
    PhysicalProperties::new(OrderingChoice::new(cols))
}

fn filter(left: ColumnId, value: ScalarValue) -> Option<ScalarNode> {
    filter_with_selectivity(left, value, 1.0)
}

fn filter_with_selectivity(left: ColumnId, value: ScalarValue, selectivity: f64) -> Option<ScalarNode> {
    let expr = Expr::BinaryExpr {
        lhs: Box::new(Expr::Column(left)),
        op: BinaryOp::Gt,
        rhs: Box::new(Expr::Scalar(value)),
    };
    let operator = Operator::from(OperatorExpr::Scalar(expr));
    let statistics = Statistics::from_selectivity(selectivity);
    Some(ScalarNode::Expr(Box::new(operator.with_statistics(statistics))))
}

#[test]
fn test_get() {
    let query = LogicalExpr::Projection {
        input: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        columns: vec![1, 2],
        exprs: vec![],
    }
    .to_operator();

    let mut tester = OptimizerTester::new();

    tester.set_table_access_cost("A", 100);

    tester.optimize(
        query,
        r#"
01 Projection [00] cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_join() {
    let query = LogicalExpr::Projection {
        input: LogicalExpr::Join {
            left: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            right: LogicalExpr::Get {
                source: "B".into(),
                columns: vec![3, 4],
            }
            .into(),
            condition: JoinCondition::using(vec![(1, 3)]),
        }
        .into(),
        columns: vec![1, 2, 3],
        exprs: vec![],
    }
    .to_operator();

    let mut tester = OptimizerTester::new();

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
    );
}

#[test]
fn test_select() {
    let query = LogicalExpr::Select {
        input: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        filter: filter(1, ScalarValue::Int32(10)),
    }
    .to_operator();

    let mut tester = OptimizerTester::new();

    tester.set_table_access_cost("A", 100);

    tester.optimize(
        query,
        r#"
02 Select [00 01]
01 Expr col:1 > 10
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_select_with_a_nested_query() {
    let sub_query = LogicalExpr::Get {
        source: "B".into(),
        columns: vec![3],
    }
    .to_operator();
    let filter = Expr::BinaryExpr {
        lhs: Box::new(Expr::SubQuery(sub_query.into())),
        op: BinaryOp::Gt,
        rhs: Box::new(Expr::Scalar(ScalarValue::Int32(1))),
    };

    let query = LogicalExpr::Select {
        input: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        filter: Some(filter.into()),
    }
    .to_operator();

    let mut tester = OptimizerTester::new();

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
    );
}

#[test]
fn test_get_ordered_top_level_enforcer() {
    let query = LogicalExpr::Select {
        input: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        filter: filter_with_selectivity(1, ScalarValue::Int32(10), 0.1),
    }
    .to_operator()
    .with_required(ordering(vec![2]));
    // .with_statistics(Statistics::from_selectivity(0.1));

    let mut tester = OptimizerTester::new();

    tester.set_table_access_cost("A", 100);

    tester.optimize(
        query,
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
    let query = LogicalExpr::Select {
        input: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        filter: filter(1, ScalarValue::Int32(10)),
    }
    .to_operator()
    .with_required(ordering(vec![2]));

    let mut tester = OptimizerTester::new();

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
    );
}

#[test]
fn test_get_ordered() {
    let query = LogicalExpr::Get {
        source: "A".into(),
        columns: vec![1, 2],
    }
    .to_operator()
    .with_required(ordering(vec![1]));

    let mut tester = OptimizerTester::new();

    tester.add_rules(|_| vec![Box::new(SelectRule)]);

    tester.set_table_access_cost("A", 100);
    tester.optimize(
        query,
        r#"
00 Sort [00] ord=[1]
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_join_commutativity() {
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
        filter: filter(1, ScalarValue::Int32(10)),
    }
    .to_operator();

    let mut tester = OptimizerTester::new();

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 10);

    tester.optimize(
        query,
        r#"
04 Select [02 03]
03 Expr col:1 > 10
02 HashJoin [01 00] using=[(3, 1)]
00 Scan A cols=[1]
01 Scan B cols=[3]
"#,
    );
}

#[test]
fn test_join_commutativity_ordered() {
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
        filter: filter_with_selectivity(1, ScalarValue::Int32(10), 0.1),
    }
    .to_operator()
    .with_required(ordering(vec![1]));

    let mut tester = OptimizerTester::new();

    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 110);

    tester.optimize(
        query,
        r#"
04 Sort [04] ord=[1]
04 Select [02 03]
03 Expr col:1 > 10
02 HashJoin [00 01] using=[(1, 3)]
01 Scan B cols=[3]
00 Scan A cols=[1]
"#,
    );
}

#[test]
fn test_prefer_already_sorted_data() {
    let query = LogicalExpr::Get {
        source: "A".into(),
        columns: vec![1, 2],
    }
    .to_operator()
    .with_required(ordering(vec![1, 2]));

    let mut tester = OptimizerTester::new();
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

    tester.optimize(
        query,
        r#"
00 IndexScan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_merge_join_requires_sorted_inputs() {
    let query = LogicalExpr::Join {
        left: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        right: LogicalExpr::Get {
            source: "B".into(),
            columns: vec![3, 4],
        }
        .into(),
        condition: JoinCondition::using(vec![(1, 4)]),
    }
    .to_operator();

    let mut tester = OptimizerTester::new();
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
    );
}

#[test]
fn test_merge_join_satisfies_ordering_requirements() {
    let query = LogicalExpr::Join {
        left: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        right: LogicalExpr::Get {
            source: "B".into(),
            columns: vec![3, 4],
        }
        .into(),
        condition: JoinCondition::using(vec![(1, 4)]),
    }
    .to_operator()
    .with_required(ordering(vec![1]));

    let mut tester = OptimizerTester::new();
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
    );
}

#[test]
fn test_merge_join_does_no_satisfy_ordering_requirements() {
    let query = LogicalExpr::Join {
        left: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        right: LogicalExpr::Get {
            source: "B".into(),
            columns: vec![3, 4],
        }
        .into(),
        condition: JoinCondition::using(vec![(1, 4)]),
    }
    .to_operator()
    .with_required(ordering(vec![3]));

    let mut tester = OptimizerTester::new();
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
    );
}

#[test]
fn test_self_joins() {
    let query = LogicalExpr::Join {
        left: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        right: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        condition: JoinCondition::using(vec![(1, 1)]),
    }
    .to_operator()
    .with_required(ordering(vec![1]));

    let mut tester = OptimizerTester::new();
    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 50);

    tester.optimize(
        query.clone(),
        r#"
01 Sort [01] ord=[1]
01 HashJoin [00 00] using=[(1, 1)]
00 Scan A cols=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );

    tester.disable_rules(|r| r.name() == "HashJoinRule");

    tester.optimize(
        query,
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
    let inner = LogicalExpr::Join {
        left: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        right: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        condition: JoinCondition::using(vec![(1, 1)]),
    }
    .to_operator()
    .with_required(ordering(vec![1])); // this ordering requirement should be ignored

    let query = LogicalExpr::Join {
        left: inner.into(),
        right: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        condition: JoinCondition::using(vec![(1, 1)]),
    }
    .to_operator()
    .with_required(ordering(vec![1]));

    let mut tester = OptimizerTester::new();
    tester.add_rules(|_| vec![Box::new(HashJoinRule), Box::new(MergeSortJoinRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 50);

    tester.disable_rules(|r| r.name() == "HashJoinRule");

    tester.optimize(
        query.clone(),
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
        query,
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
    let query = LogicalExpr::Select {
        input: LogicalExpr::Join {
            left: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            right: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            condition: JoinCondition::using(vec![(1, 1)]),
        }
        .to_operator()
        .with_required(ordering(vec![1]))
        .into(),
        filter: filter(1, ScalarValue::Int32(10)),
    }
    .to_operator();

    let mut tester = OptimizerTester::new();
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
    );
}

#[test]
fn test_inner_sort_satisfied_by_ordering_providing_operator() {
    let query = LogicalExpr::Select {
        input: LogicalExpr::Join {
            left: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            right: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            condition: JoinCondition::using(vec![(1, 1)]),
        }
        .to_operator()
        .with_required(ordering(vec![1]))
        .into(),
        filter: filter_with_selectivity(1, ScalarValue::Int32(10), 0.1),
    }
    .to_operator();

    let mut tester = OptimizerTester::new();
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
    );
}

#[test]
fn test_join_associativity_ax_bxc() {
    let query = LogicalExpr::Join {
        left: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        right: LogicalExpr::Join {
            left: LogicalExpr::Get {
                source: "B".into(),
                columns: vec![3, 4],
            }
            .into(),
            right: LogicalExpr::Get {
                source: "C".into(),
                columns: vec![5, 6],
            }
            .into(),
            condition: JoinCondition::using(vec![(3, 6)]),
        }
        .into(),
        condition: JoinCondition::using(vec![(1, 3)]),
    }
    .to_operator();

    let mut tester = OptimizerTester::new();
    tester
        .add_rules(|_| vec![Box::new(HashJoinRule), Box::new(JoinAssociativityRule), Box::new(JoinCommutativityRule)]);

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 200);
    tester.set_table_access_cost("C", 500);

    tester.disable_rules(|r| r.name() == "JoinCommutativityRule");

    tester.optimize(
        query.clone(),
        r#"query: Ax[BxC] => [AxB]xC
04 HashJoin [05 02] using=[(1, 6)]
02 Scan C cols=[5, 6]
05 HashJoin [00 01] using=[(1, 3)]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    );
    tester.reset_rule_filters();

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 150);
    tester.set_table_access_cost("C", 250);

    tester.optimize(
        query.clone(),
        r#"query: Ax[BxC] => Ax[BxC]
04 HashJoin [00 03] using=[(1, 3)]
03 HashJoin [01 02] using=[(3, 6)]
02 Scan C cols=[5, 6]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    );

    tester.set_table_access_cost("A", 250);
    tester.set_table_access_cost("B", 100);
    tester.set_table_access_cost("C", 150);

    tester.optimize(
        query,
        r#"query: Ax[BxC] => [BxC]xA
04 HashJoin [03 00] using=[(3, 1)]
00 Scan A cols=[1, 2]
03 HashJoin [01 02] using=[(3, 6)]
02 Scan C cols=[5, 6]
01 Scan B cols=[3, 4]
"#,
    );
}

#[test]
fn test_join_associativity_axb_xc() {
    let query = LogicalExpr::Join {
        left: LogicalExpr::Join {
            left: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            right: LogicalExpr::Get {
                source: "B".into(),
                columns: vec![3, 4],
            }
            .into(),
            condition: JoinCondition::using(vec![(1, 4)]),
        }
        .into(),
        right: LogicalExpr::Get {
            source: "C".into(),
            columns: vec![5, 6],
        }
        .into(),
        condition: JoinCondition::using(vec![(1, 6)]),
    }
    .to_operator();

    let mut tester = OptimizerTester::new();
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
    );
}

#[test]
fn test_enforce_grouping() {
    let query = LogicalExpr::Aggregate {
        input: LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .into(),
        aggr_exprs: vec![ScalarNode::from(Expr::Aggregate {
            func: AggregateFunction::Count,
            args: vec![Expr::Column(1)],
            filter: None,
        })],
        group_exprs: vec![ScalarNode::from(Expr::Column(2))],
    }
    .to_operator();

    let mut tester = OptimizerTester::new();
    tester.add_rules(|_| vec![Box::new(HashAggregateRule)]);

    tester.set_table_access_cost("A", 100);

    tester.optimize(
        query,
        r#"
03 HashAggregate [00 02 01]
01 Expr col:2
02 Expr count(col:1)
00 Scan A cols=[1, 2]
"#,
    );
}

#[test]
fn test_union() {
    fn union_op(all: bool) -> Operator {
        let left = LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        };
        let right = LogicalExpr::Get {
            source: "B".into(),
            columns: vec![3, 4],
        };
        let union = LogicalExpr::Union {
            left: left.into(),
            right: right.into(),
            all,
        };
        Operator::from(OperatorExpr::from(union))
    }

    let mut tester = OptimizerTester::new();

    tester.set_table_access_cost("A", 100);
    tester.set_table_access_cost("B", 100);

    tester.add_rules(|_| vec![Box::new(UnionRule)]);

    let union = union_op(false);

    tester.optimize(
        union,
        r#"query: union all=false -> unique
02 Unique [ord:[1, 2]=00 ord:[3, 4]=01]
01 Sort [01] ord=[3, 4]
01 Scan B cols=[3, 4]
00 Sort [00] ord=[1, 2]
00 Scan A cols=[1, 2]
"#,
    );

    let union = union_op(true);
    tester.optimize(
        union,
        r#"query: union all=true -> append
02 Append [00 01]
01 Scan B cols=[3, 4]
00 Scan A cols=[1, 2]
"#,
    );
}
