use crate::memo::MemoExpr;
use crate::meta::ColumnId;
use crate::operators::relational::join::JoinCondition;
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin, LogicalProjection,
    LogicalUnion,
};
use crate::operators::relational::RelNode;
use crate::rules::rewrite::filter_push_down::new_inputs;
use std::collections::HashSet;

/// A basic implementation of a redundant projection removal rule.
///
/// # Limitations
///
/// * Does not remove projection if that projection adds new aliases.
/// * Does not remove projections that contain nested sub queries.
/// * Does not remove projection if its input operator is either UNION, INTERSECT or EXCEPT.
/// * Currently column positions in projection list are taken into account. So the second projection in
/// `[a1, b1] -> [a1, b1, c1]` will be removed.  But when the same columns in the parent projection are arranged
/// in different order this rule preserve both projections (eg. `[b2, a1] -> [a1, b1, c1]`).
/// * Join operators are not supported because they depend on output columns which must be computed for newly created operators.
///
pub fn remove_redundant_projections(expr: &RelNode) -> Option<RelNode> {
    if let LogicalExpr::Projection(projection) = expr.expr().as_logical() {
        Some(rewrite(expr, &projection.input, &projection.columns))
    } else {
        None
    }
}

fn rewrite(parent: &RelNode, expr: &RelNode, columns: &[ColumnId]) -> RelNode {
    // We exploit the fact that output columns of projection ([col:1, col:2, col:2 as c3]) are [col:1, col:2, col:3]
    // Because of that we can simply compare output columns of a projection operator and its child operator
    // and if those columns are equal then the projection is redundant and can be removed.
    match expr.expr().as_logical() {
        LogicalExpr::Projection(LogicalProjection {
            input,
            columns: child_columns,
            ..
        }) => {
            // Replace this projection with its parent if this projection contains all parent columns.
            if columns.iter().zip(child_columns.iter()).all(|(a, b)| a == b) {
                rewrite(expr, &input, &child_columns)
            } else {
                rewrite_inputs(parent)
            }
        }
        LogicalExpr::Aggregate(LogicalAggregate {
            columns: child_columns, ..
        }) => {
            if columns == child_columns {
                // Remove the projection operator if it produces the same output
                rewrite_inputs(expr)
            } else {
                // Retain the projection if it produces different columns.
                rewrite_inputs(parent)
            }
        }
        LogicalExpr::Get(LogicalGet {
            columns: child_columns, ..
        }) => {
            // The projection contains the same columns as input operator => remove it
            if child_columns == columns {
                expr.clone()
            } else {
                rewrite_inputs(parent)
            }
        }
        LogicalExpr::Join(LogicalJoin { left, right, .. }) => {
            let left = rewrite_inputs(left);
            let right = rewrite_inputs(right);
            let join = new_inputs(expr, vec![left, right]);
            new_inputs(parent, vec![join])
            // TODO: In order to rewrite joins we must compute logical properties for expressions created by this rule.
            // match condition {
            //     JoinCondition::Using(using) => {
            //         let (left_cols, right_cols) = using.get_columns_pair();
            //         let using_columns: HashSet<_> = left_cols.into_iter().chain(right_cols.into_iter()).collect();
            //
            //         if using_columns.len() == columns.len() && columns.iter().all(|c| using_columns.contains(c)) {
            //             let left = rewrite_inputs(left);
            //             let right = rewrite_inputs(right);
            //             new_inputs(expr, vec![left, right])
            //         } else {
            //             let left = rewrite_inputs(left);
            //             let right = rewrite_inputs(right);
            //             let join = new_inputs(expr, vec![left, right]);
            //             new_inputs(parent, vec![join])
            //         }
            //     }
            //     JoinCondition::On(_) => {
            //         // TODO: support ON condition.
            //         rewrite_inputs(parent)
            //     }
            // }
        }
        // Rewrite child expressions of set operators
        LogicalExpr::Union(LogicalUnion { left, right, .. })
        | LogicalExpr::Except(LogicalExcept { left, right, .. })
        | LogicalExpr::Intersect(LogicalIntersect { left, right, .. }) => {
            let left = rewrite_inputs(left);
            let right = rewrite_inputs(right);
            let join = new_inputs(expr, vec![left, right]);
            new_inputs(parent, vec![join])
        }
        // Recursively apply this rule to the child expressions.
        _ => rewrite_inputs(parent),
    }
}

fn rewrite_inputs(expr: &RelNode) -> RelNode {
    let mut children = vec![];

    for i in 0..expr.mexpr().num_children() {
        let child_expr = expr.mexpr().get_child(i).unwrap();
        if child_expr.expr().is_relational() {
            let child_expr = RelNode::from(child_expr.clone());
            let output_columns = child_expr.props().logical.output_columns();
            let result = rewrite(expr, &child_expr, output_columns);

            children.push(result);
        }
    }

    new_inputs(expr, children)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::error::OptimizerError;
    use crate::operators::builder::OperatorBuilder;
    use crate::rules::rewrite::testing::{build_and_rewrite_expr, col, cols_add, expr_alias, scalar};

    #[test]
    fn test_remove_redundant_projection() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1"), col("a2"), expr_alias(col("a2"), "c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 3] exprs=[col:1, col:2, col:2 AS c3]
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
                let project = from_a.project(vec![col("a1"), expr_alias(col("a2"), "c2")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 3] exprs=[col:1, col:2 AS c2]
  input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_replace_redundant_intermediate_projections() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1"), col("a2"), expr_alias(col("a2"), "c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 3] exprs=[col:1, col:2, col:2 AS c3]
  input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_remove_input_projection_if_it_has_more_columns() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2", "a3"])?;
                // add a constant column so the projection bellow won't be removed
                let project = from_a.project(vec![col("a1"), col("a2"), scalar(1)])?;
                let project = project.project(vec![col("a1"), col("a2")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 4] exprs=[col:1, col:2, 1]
  input: LogicalGet A cols=[1, 2, 3]
"#,
        );

        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2", "a3"])?;
                // add alias expr so the projection bellow won't be removed
                let project = from_a.project(vec![col("a1"), col("a2"), expr_alias(col("a3"), "c3")])?;
                let project = project.project(vec![col("a1"), col("a2"), col("c3")])?;
                let project = project.project(vec![col("a1"), col("a2")])?;

                Ok(project)
            },
            // Only the first projection remains.
            r#"
LogicalProjection cols=[1, 2, 4] exprs=[col:1, col:2, col:3 AS c3]
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
                    project.project(vec![col("a1"), col("a2"), expr_alias(col("a2"), "c3"), cols_add("a1", "a2")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 4, 5] exprs=[col:1, col:2, col:2 AS c3, col:1 + col:2]
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
                let project = from_a.project(vec![col("a1"), col("a2"), expr_alias(cols_add("a1", "a2"), "sum")])?;
                let project = project.project(vec![col("a1"), col("a2"), expr_alias(col("sum"), "c3")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 5] exprs=[col:1, col:2, col:4 AS c3]
  input: LogicalProjection cols=[1, 2, 4] exprs=[col:1, col:2, col:1 + col:2 AS sum]
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
                let count = from_b.aggregate_builder().add_func("count", "b1")?.build()?.to_sub_query()?;

                let project = from_a.project(vec![col("a1"), col("a2"), count.clone()])?;
                let project = project.project(vec![col("a1"), col("a2"), count])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[1, 2, 6] exprs=[col:1, col:2, SubQuery 02]
  input: LogicalProjection cols=[1, 2, 5] exprs=[col:1, col:2, SubQuery 02]
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
  input: LogicalGet A cols=[1, 2]
  : Expr col:1
  : Expr sum(col:1)
  : Expr col:1
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
  input: LogicalGet A cols=[1, 2]
  : Expr col:1
  : Expr sum(col:1)
  : Expr col:1
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
                let project =
                    aggr.project(vec![expr_alias(col("sum"), "s"), expr_alias(cols_add("a1", "a1"), "s2")])?;

                Ok(project)
            },
            r#"
LogicalProjection cols=[4, 5] exprs=[col:3 AS s, col:1 + col:1 AS s2]
  input: LogicalAggregate
      input: LogicalGet A cols=[1, 2]
      : Expr col:1
      : Expr sum(col:1)
      : Expr col:1
"#,
        )
    }

    #[test]
    fn test_do_not_remove_projection_before_join() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
                let from_b = builder.get("B", vec!["b1", "b2"])?;
                let join = from_a.join_using(from_b, vec![("a1", "b1")])?;
                let projection = join.project(vec![col("a1"), col("b1")])?;

                Ok(projection)
            },
            r#"
LogicalProjection cols=[1, 3] exprs=[col:1, col:3]
  input: LogicalJoin using=[(1, 3)]
      left: LogicalGet A cols=[1, 2]
      right: LogicalGet B cols=[3, 4]
"#,
        );
    }

    fn rewrite_expr<F>(f: F, expected: &str)
    where
        F: FnOnce(OperatorBuilder) -> Result<OperatorBuilder, OptimizerError>,
    {
        build_and_rewrite_expr(f, remove_redundant_projections, expected)
    }
}
