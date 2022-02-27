#![allow(dead_code)]
use crate::operators::relational::join::JoinCondition;
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalExpr, LogicalGet, LogicalJoin, LogicalProjection,
};
use crate::operators::relational::RelNode;
use crate::rules::rewrite::rewrite_rel_inputs;

/// A basic implementation of a redundant projection removal rule.
///
/// # Limitations
///
/// * Does not remove projections that contain nested sub queries.
/// * Does not remove projection if its input operator is either UNION, INTERSECT or EXCEPT.
/// * Currently column positions in projection list are taken into account. So the second projection in
/// `[a1, b1] -> [a1, b1, c1]` will be removed.  But when the same columns in the parent projection are arranged
/// in different order this rule preserve both projections (eg. `[b2, a1] -> [a1, b1, c1]`).
/// * Does not support ON condition in joins.
///
/// # Note
///
/// The implementation does not rely on logical properties because logical properties are not set
/// for new expressions created by this rule.
pub fn remove_redundant_projections(expr: &RelNode) -> Option<RelNode> {
    Some(rewrite(expr))
}

fn rewrite(expr: &RelNode) -> RelNode {
    if let LogicalExpr::Projection(projection) = expr.expr().logical() {
        // We exploit the fact that output columns of a projection ([col:1, col:2, col:2 as c3]) are [col:1, col:2, col:3]
        // Because of that we can simply compare output columns of a projection operator and its child operator
        // and if those columns are equal then the projection is redundant and can be removed.
        let input = &projection.input;
        let columns = &projection.columns;

        match input.expr().logical() {
            LogicalExpr::Projection(LogicalProjection {
                columns: child_columns,
                input: child_input,
                exprs: child_exprs,
                ..
            }) => {
                if columns.iter().zip(child_columns.iter()).all(|(a, b)| a == b) {
                    // Use the expression list from the child projection to replace
                    // aliased references in the parent projection with corresponding alias expressions.
                    //
                    // For example:
                    //
                    // Projection: cols=[1, 2] exprs: [col1: 1, col:2] # Where col:2 = Expr col:1 as c2
                    //   Projection: cols=[1, 2] exprs: [col1: 1, col:1 as c2]
                    //
                    // gets rewritten into:
                    //   Projection: cols=[1, 2] exprs: [col1: 1, col:1 as c2]
                    //
                    let mut new_projection = projection.clone();
                    new_projection.input = child_input.clone();
                    new_projection.exprs = child_exprs[0..projection.exprs.len()].to_vec();

                    let new_projection = RelNode::from(LogicalExpr::Projection(new_projection));
                    rewrite(&new_projection)
                } else {
                    rewrite_inputs(expr)
                }
            }
            LogicalExpr::Join(LogicalJoin { condition, .. }) => {
                match condition {
                    JoinCondition::Using(using) => {
                        let (left, right) = using.get_columns_pair();
                        let join_columns: Vec<_> = left.into_iter().chain(right.into_iter()).collect();
                        if join_columns.len() == columns.len() && columns.iter().all(|c| join_columns.contains(c)) {
                            // Projection contains only columns from the join condition -> projection is redundant.
                            rewrite_inputs(input)
                        } else {
                            rewrite_inputs(expr)
                        }
                    }
                    JoinCondition::On(_) => {
                        //TODO: support ON condition
                        rewrite_inputs(expr)
                    }
                }
            }
            LogicalExpr::Aggregate(LogicalAggregate {
                columns: child_columns, ..
            }) => {
                if child_columns == columns {
                    // Projection contains only columns produced by the aggregation -> projection is redundant.
                    rewrite_inputs(input)
                } else {
                    rewrite_inputs(expr)
                }
            }
            LogicalExpr::Get(LogicalGet {
                columns: child_columns, ..
            }) => {
                if child_columns == columns {
                    // Projection contains only columns from the scan -> projection is redundant.
                    input.clone()
                } else {
                    rewrite_inputs(expr)
                }
            }
            _ => rewrite_inputs(expr),
        }
    } else {
        rewrite_inputs(expr)
    }
}

fn rewrite_inputs(expr: &RelNode) -> RelNode {
    rewrite_rel_inputs(expr, rewrite)
}

#[cfg(test)]
mod test {
    use crate::error::OptimizerError;
    use crate::operators::builder::OperatorBuilder;
    use crate::operators::relational::join::JoinType;
    use crate::operators::scalar::{col, scalar, ScalarExpr};
    use crate::rules::rewrite::testing::build_and_rewrite_expr;

    use super::*;

    #[test]
    fn test_remove_redundant_projection() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1"), col("a2"), col("a2").alias("c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 3] exprs: [col:1, col:2, col:2 AS c3]
  input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_replace_redundant_projection_after_get() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1"), col("a2")])?;
                let project = project.project(vec![col("a1"), col("a2")])?;

                Ok(project)
            },
            r#"
LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_keep_projection_with_aliases_after_get() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1"), col("a2").alias("c2")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 3] exprs: [col:1, col:2 AS c2]
  input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_replace_redundant_intermediate_projections() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1"), col("a2"), col("a2").alias("c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 3] exprs: [col:1, col:2, col:2 AS c3]
  input: LogicalGet A cols=[1, 2]
"#,
        );

        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a2"), col("a1"), col("a2").alias("c3")])?;
                // 2 projections below must be removed
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 3] exprs: [col:1, col:2, col:3]
  input: LogicalProjection cols=[2, 1, 3] exprs: [col:2, col:1, col:2 AS c3]
    input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_remove_input_projection_if_it_has_more_columns() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2", "a3"])?;
                let project = from_a.project(vec![col("a1"), col("a2"), scalar(1)])?;
                let project = project.project(vec![col("a1"), col("a2")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2] exprs: [col:1, col:2]
  input: LogicalGet A cols=[1, 2, 3]
"#,
        );
    }

    #[test]
    fn test_preserve_expressions_when_remove_projections() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2", "a3"])?;
                let project = from_a.project(vec![col("a1"), col("a2").alias("c2"), col("a3")])?;
                let project = project.project(vec![col("a1"), col("c2"), col("a3")])?;
                let project = project.project(vec![col("a1"), col("c2")])?;

                Ok(project)
            },
            //
            r#"
LogicalProjection cols=[1, 4] exprs: [col:1, col:2 AS c2]
  input: LogicalGet A cols=[1, 2, 3]
"#,
        )
    }

    #[test]
    fn test_remove_redundant_projection_when_projection_has_more_columns() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2", "a3"])?;
                let project = from_a.project(vec![col("a1"), col("a2"), col("a3")])?;
                let project =
                    project.project(vec![col("a1"), col("a2"), col("a2").alias("c3"), col("a1") + col("a2")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 4, 5] exprs: [col:1, col:2, col:2 AS c3, col:1 + col:2]
  input: LogicalGet A cols=[1, 2, 3]
"#,
        )
    }

    #[test]
    fn test_keep_redundant_projection_that_uses_aliases() {
        // Currently we do not replace alias expressions in projections.
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2", "a3"])?;
                let project = from_a.project(vec![col("a1"), col("a2"), (col("a1") + col("a2")).alias("sum")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("sum").alias("c3")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 5] exprs: [col:1, col:2, col:4 AS c3]
  input: LogicalProjection cols=[1, 2, 4] exprs: [col:1, col:2, col:1 + col:2 AS sum]
    input: LogicalGet A cols=[1, 2, 3]
"#,
        )
    }

    #[test]
    fn test_retain_projections_with_sub_queries() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2"])?;

                let mut from_b = builder.sub_query_builder().get("B", vec!["b1"])?;
                let count = from_b.aggregate_builder().add_func("count", "b1")?.build()?.build()?;
                let count = ScalarExpr::SubQuery(RelNode::from(count));

                let project = from_a.project(vec![col("a1"), col("a2"), count.clone()])?;
                let project = project.project(vec![col("a1"), col("a2"), count])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 6] exprs: [col:1, col:2, SubQuery 02]
  input: LogicalProjection cols=[1, 2, 5] exprs: [col:1, col:2, SubQuery 02]
    input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_remove_redundant_projection_before_aggregate() {
        rewrite_expr(
            |builder| {
                let mut from_a = builder.get("A", vec!["a1", "a2"])?;
                let aggr = from_a
                    .aggregate_builder()
                    .add_column("a1")?
                    .add_func("sum", "a1")?
                    .group_by("a1")?
                    .build()?;
                let project = aggr.project(vec![col("a1"), col("sum")])?;

                Ok(project)
            },
            r#"
LogicalAggregate
  aggr_exprs: [col:1, sum(col:1)]
  group_exprs: [col:1]
  input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_remove_redundant_projection_aggregate_remove_nested_projections() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1"), col("a2")])?;
                let mut project = project.project(vec![col("a1"), col("a2")])?;

                let aggr = project
                    .aggregate_builder()
                    .add_column("a1")?
                    .add_func("sum", "a1")?
                    .group_by("a1")?
                    .build()?;
                let project = aggr.project(vec![col("a1"), col("sum")])?;

                Ok(project)
            },
            r#"
LogicalAggregate
  aggr_exprs: [col:1, sum(col:1)]
  group_exprs: [col:1]
  input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_keep_redundant_projection_with_aliases_before_aggregate() {
        rewrite_expr(
            |builder| {
                let mut from_a = builder.get("A", vec!["a1", "a2"])?;
                let aggr = from_a
                    .aggregate_builder()
                    .add_column("a1")?
                    .add_func("sum", "a1")?
                    .group_by("a1")?
                    .build()?;

                let s = col("sum").alias("s");
                let s2 = col("a1") + (col("a1")).alias("s2");
                let project = aggr.project(vec![s, s2])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[4, 5] exprs: [col:3 AS s, col:1 + col:1 AS s2]
  input: LogicalAggregate
    aggr_exprs: [col:1, sum(col:1)]
    group_exprs: [col:1]
    input: LogicalGet A cols=[1, 2]
"#,
        );

        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1"), col("a2")])?;
                let mut project = project.project(vec![col("a1"), col("a2")])?;
                let aggr = project
                    .aggregate_builder()
                    .add_column("a1")?
                    .add_func("sum", "a1")?
                    .group_by("a1")?
                    .build()?;
                let project = aggr.project(vec![col("sum").alias("s"), col("a1") + (col("a1")).alias("s2")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[4, 5] exprs: [col:3 AS s, col:1 + col:1 AS s2]
  input: LogicalAggregate
    aggr_exprs: [col:1, sum(col:1)]
    group_exprs: [col:1]
    input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_remove_projection_before_join_if_projection_uses_only_columns_from_join_condition() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
                let from_b = builder.get("B", vec!["b1", "b2"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;
                let projection = join.project(vec![col("a1"), col("b1")])?;

                Ok(projection)
            },
            r#"
LogicalJoin type=Inner using=[(1, 3)]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
"#,
        );
    }

    #[test]
    fn test_do_not_remove_projection_before_join_if_projection_introduces_additional_expressions() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
                let from_b = builder.get("B", vec!["b1", "b2"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;
                let projection = join.project(vec![col("a1"), col("b1"), col("b2")])?;

                Ok(projection)
            },
            r#"
LogicalProjection cols=[1, 3, 4] exprs: [col:1, col:3, col:4]
  input: LogicalJoin type=Inner using=[(1, 3)]
    left: LogicalGet A cols=[1, 2]
    right: LogicalGet B cols=[3, 4]
"#,
        );

        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
                let from_b = builder.get("B", vec!["b1", "b2"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;
                let projection = join.project(vec![col("a1"), col("b1"), col("a1").alias("c1")])?;

                Ok(projection)
            },
            r#"
LogicalProjection cols=[1, 3, 5] exprs: [col:1, col:3, col:1 AS c1]
  input: LogicalJoin type=Inner using=[(1, 3)]
    left: LogicalGet A cols=[1, 2]
    right: LogicalGet B cols=[3, 4]
"#,
        );

        // keep projection if it uses aliases
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
                let from_b = builder.get("B", vec!["b1", "b2"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;
                let projection = join.project(vec![col("a1"), col("b1").alias("c1")])?;

                Ok(projection)
            },
            r#"
LogicalProjection cols=[1, 5] exprs: [col:1, col:3 AS c1]
  input: LogicalJoin type=Inner using=[(1, 3)]
    left: LogicalGet A cols=[1, 2]
    right: LogicalGet B cols=[3, 4]
"#,
        );
    }

    #[test]
    fn test_do_not_remove_projection_before_join_if_projection_uses_aliases() {
        // keep projection if it uses aliases
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
                let from_b = builder.get("B", vec!["b1", "b2"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;
                let projection = join.project(vec![col("a1"), col("b1").alias("c1")])?;

                Ok(projection)
            },
            r#"
LogicalProjection cols=[1, 5] exprs: [col:1, col:3 AS c1]
  input: LogicalJoin type=Inner using=[(1, 3)]
    left: LogicalGet A cols=[1, 2]
    right: LogicalGet B cols=[3, 4]
"#,
        );
    }

    #[test]
    fn test_do_not_move_projection_before_select() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let select = from_a.select(Some(col("a1").eq(col("a2"))))?;
                let project = select.project(vec![col("a1"), col("a2")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2] exprs: [col:1, col:2]
  input: LogicalSelect
    input: LogicalGet A cols=[1, 2]
    filter: Expr col:1 = col:2
"#,
        )
    }

    #[test]
    fn test_do_not_move_projection_before_empty() {
        rewrite_expr(
            |builder| {
                let empty = builder.empty(true)?;
                let project = empty.project(vec![scalar(1)])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1] exprs: [1]
  input: LogicalEmpty return_one_row=true
"#,
        )
    }

    fn rewrite_expr<F>(f: F, expected: &str)
    where
        F: FnOnce(OperatorBuilder) -> Result<OperatorBuilder, OptimizerError>,
    {
        build_and_rewrite_expr(f, remove_redundant_projections, expected)
    }
}
