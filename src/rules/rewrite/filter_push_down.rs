#![allow(dead_code)]
use crate::error::OptimizerError;
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;

use crate::meta::ColumnId;
use crate::operators::relational::join::{JoinCondition, JoinUsing};
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin, LogicalProjection,
    LogicalSelect, LogicalUnion,
};
use crate::operators::relational::{RelNode, SetOperator};
use crate::operators::scalar::expr::{BinaryOp, ExprRewriter};
use crate::operators::scalar::{exprs, ScalarExpr, ScalarNode};
use crate::rules::rewrite::{rewrite_rel_inputs, with_new_rel_inputs};

/// Rewrites the operator tree rooted at the given expression.
/// This rewrite rule attempts to push filters in an operator tree bellow other operators
/// so those filters run as early as possible.
///
/// If the given expression is not a select expression this function immediately returns [`None`](std::option::Option::None).
///
/// # Limitations:
/// * Currently it does not support ON condition in joins.
/// * Does not support SemiJoin and AntiJoin.
/// * When this rule passes filter through the join it adds redundant filter expressions.
///
/// # Note
///
/// The implementation does not rely on logical properties because logical properties are not set
/// for new expressions created by this rule.
pub fn predicate_push_down(expr: &RelNode) -> Result<Option<RelNode>, OptimizerError> {
    if let LogicalExpr::Select(_) = expr.expr().logical() {
        let state = State { filters: Vec::new() };
        let expr = rewrite(state, expr)?;
        Ok(Some(expr))
    } else {
        Ok(None)
    }
}

fn rewrite(mut state: State, expr: &RelNode) -> Result<RelNode, OptimizerError> {
    match expr.expr().logical() {
        LogicalExpr::Select(LogicalSelect { input, filter }) => {
            if let Some(filter) = filter {
                let mut filters = Vec::new();
                split_filter(filter.expr(), &mut filters);

                // Collect filters without columns and do not pass them down the operator tree.
                let mut constant_filters = Vec::new();

                for filter in filters {
                    let columns = exprs::collect_columns(&filter);
                    if columns.is_empty() {
                        constant_filters.push(filter);
                    } else if !state.filters.iter().any(|f| f.0.expr() == &filter) {
                        state.filters.push((ScalarNode::from(filter), columns.into_iter().collect()))
                    }
                }

                if constant_filters.is_empty() {
                    rewrite(state, input)
                } else {
                    let rel_node = rewrite(state, input)?;
                    // Add select with `constant_filters` such node should be later removed by another rule.
                    let expr = add_filter_node(&rel_node, constant_filters);
                    Ok(expr)
                }
            } else {
                // Keep select without filter it should be removed by another rule.
                let result = rewrite(state, input)?;
                with_new_rel_inputs(expr, vec![result])
            }
        }
        LogicalExpr::Projection(LogicalProjection {
            input, exprs, columns, ..
        }) => {
            // Rewrite filter expressions so they do not contain aliases.
            let projections = prepare_projections(columns, exprs);

            for (filter, columns) in state.filters.iter_mut() {
                let filter_expr = remove_projections(filter.expr(), &projections)?;
                let cols = exprs::collect_columns(&filter_expr);

                *columns = cols.into_iter().collect();
                *filter = ScalarNode::from(filter_expr);
            }

            let result = rewrite(state, input)?;
            with_new_rel_inputs(expr, vec![result])
        }
        LogicalExpr::Aggregate(LogicalAggregate {
            aggr_exprs, columns, ..
        }) => {
            // Add filters if the state contains filters that use columns containing aggregate expressions.
            // Pass remaining filters past this aggregate expression.

            // Collect columns used in group by clause
            let group_columns: Vec<ColumnId> = aggr_exprs
                .iter()
                .filter_map(|expr| match expr.expr() {
                    ScalarExpr::Column(id) => Some(*id),
                    _ => None,
                })
                .collect();

            // Exclude columns used by group by clause so only columns that contain aggregate expressions remain.
            let aggr_columns: Vec<_> = columns.iter().filter(|c| !group_columns.contains(c)).copied().collect();

            add_filters(state, expr, &aggr_columns)
        }
        LogicalExpr::WindowAggregate(_) => {
            // Window aggregate operator does not introduce new columns
            // We can safely push the filters past it.
            rewrite_inputs(state, expr)
        }
        LogicalExpr::Join(LogicalJoin {
            left, right, condition, ..
        }) => rewrite_join(state, expr, left, right, condition),
        LogicalExpr::Get(LogicalGet { columns, .. }) => add_filters(state, expr, columns),
        LogicalExpr::Union(LogicalUnion {
            left,
            right,
            all,
            columns,
        }) => rewrite_set_op(state, SetOperator::Union, *all, left, right, columns),
        LogicalExpr::Intersect(LogicalIntersect {
            left,
            right,
            all,
            columns,
        }) => rewrite_set_op(state, SetOperator::Intersect, *all, left, right, columns),
        LogicalExpr::Except(LogicalExcept {
            left,
            right,
            all,
            columns,
        }) => rewrite_set_op(state, SetOperator::Except, *all, left, right, columns),
        LogicalExpr::Distinct(_) => rewrite_inputs(state, expr),
        LogicalExpr::Limit(_) => rewrite_inputs(state, expr),
        LogicalExpr::Offset(_) => rewrite_inputs(state, expr),
        LogicalExpr::Values(_) => rewrite_inputs(state, expr),
        LogicalExpr::Empty(_) => rewrite_inputs(state, expr),
    }
}

#[derive(Clone)]
struct State {
    filters: Vec<(ScalarNode, HashSet<ColumnId>)>,
}

#[derive(Debug)]
struct Filter<'a>(&'a ScalarNode, &'a HashSet<ColumnId>);

impl State {
    fn from_filters(filters: Vec<Filter>) -> Self {
        let filters = filters.into_iter().map(|f| (f.0.clone(), f.1.clone())).collect();
        State { filters }
    }

    fn get_filters(&self, columns: &[ColumnId]) -> Vec<Filter> {
        self.filters
            .iter()
            .filter(|(_, cols)| cols.iter().all(|c| columns.contains(c)))
            .map(|(a, b)| Filter(a, b))
            .collect()
    }

    fn exclude_filters(&self, filters: &[Filter]) -> Vec<(ScalarNode, HashSet<ColumnId>)> {
        self.filters
            .iter()
            .filter(|(_, cols)| filters.iter().filter(|f| cols == f.1).count() == 0)
            .cloned()
            .collect()
    }
}

fn add_filter_node(expr: &RelNode, mut filters: Vec<ScalarExpr>) -> RelNode {
    let first_filter = filters.swap_remove(0);
    let filter = filters.into_iter().fold(first_filter, |acc, f| acc.and(f));

    let select = LogicalSelect {
        input: expr.clone(),
        filter: Some(ScalarNode::from(filter)),
    };

    RelNode::from(LogicalExpr::Select(select))
}

fn add_filters(mut state: State, expr: &RelNode, used_columns: &[ColumnId]) -> Result<RelNode, OptimizerError> {
    let applicable_filters = state.get_filters(used_columns);

    if applicable_filters.is_empty() {
        rewrite_inputs(state, expr)
    } else {
        let remaining_filters = state.exclude_filters(&applicable_filters);
        let rel_node = add_filter_node(expr, applicable_filters.into_iter().map(|f| f.0.expr().clone()).collect());

        state.filters = remaining_filters;
        rewrite_inputs(state, &rel_node)
    }
}

fn prepare_projections(columns: &[ColumnId], exprs: &[ScalarNode]) -> HashMap<ColumnId, ScalarExpr> {
    struct RewriteAliasExpr<'a> {
        projections: &'a HashMap<ColumnId, ScalarExpr>,
    }
    impl<'a> ExprRewriter<RelNode> for RewriteAliasExpr<'a> {
        type Error = Infallible;
        fn rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
            if let ScalarExpr::Column(id) = expr {
                // We must to replace aliased columns with their origins,
                // otherwise nested alias expressions (eg. "a + b2" where b2 = alias b1 and b1 = alias b)
                // would not be rewritten correctly.
                let projection = self.projections.get(&id).expect("Unexpected column in alias expr");
                if let ScalarExpr::Column(a_id) = projection {
                    return Ok(ScalarExpr::Column(*a_id));
                }
            }
            Ok(expr)
        }
    }

    let mut projections = HashMap::new();
    for (expr, col_id) in exprs.iter().zip(columns) {
        match expr.expr() {
            ScalarExpr::Alias(expr, _) => {
                let expr = expr.as_ref().clone();
                let mut rewriter = RewriteAliasExpr {
                    projections: &projections,
                };
                let expr = expr.rewrite(&mut rewriter).unwrap();
                projections.insert(*col_id, expr);
            }
            _ => {
                projections.insert(*col_id, expr.expr().clone());
            }
        }
    }
    projections
}

fn remove_projections(
    expr: &ScalarExpr,
    projection: &HashMap<ColumnId, ScalarExpr>,
) -> Result<ScalarExpr, OptimizerError> {
    let child_exprs = expr.get_children();
    let child_exprs: Result<Vec<ScalarExpr>, _> =
        child_exprs.into_iter().map(|e| remove_projections(&e, projection)).collect();

    // Replace references to projection items with expressions that do not use other projection items.
    if let ScalarExpr::Column(id) = expr {
        if let Some(expr) = projection.get(id) {
            return Ok(expr.clone());
        }
    }

    expr.with_children(child_exprs?)
}

fn rewrite_join(
    mut state: State,
    expr: &RelNode,
    left: &RelNode,
    right: &RelNode,
    condition: &JoinCondition,
) -> Result<RelNode, OptimizerError> {
    let join_using = match condition {
        JoinCondition::Using(using) => using,
        JoinCondition::On(_) => {
            //TODO: support ON condition
            return rewrite_inputs(state, expr);
        }
    };

    let additional_filters = duplicate_filters(&state, join_using);
    for (add_filter, cols) in additional_filters {
        if state.filters.iter().any(|f| f.0.expr() == add_filter.expr()) {
            continue;
        }
        state.filters.push((add_filter, cols));
    }

    let (left_filters, right_filters, remaining_filters) = get_join_filters(&state, join_using);
    let left_state = State::from_filters(left_filters);
    let new_left = rewrite(left_state, left)?;

    let right_state = State::from_filters(right_filters);
    let new_right = rewrite(right_state, right)?;

    let expr = with_new_rel_inputs(expr, vec![new_left, new_right])?;

    if remaining_filters.is_empty() {
        Ok(expr)
    } else {
        let mut state = state.clone();
        state.filters = state.exclude_filters(&remaining_filters);

        let rel_node = add_filter_node(&expr, remaining_filters.into_iter().map(|f| f.0.expr().clone()).collect());
        rewrite_inputs(state, &rel_node)
    }
}

fn duplicate_filters(state: &State, condition: &JoinUsing) -> Vec<(ScalarNode, HashSet<ColumnId>)> {
    let current_filters: HashSet<_> = state.filters.iter().map(|f| f.0.expr()).collect();
    state
        .filters
        .iter()
        .filter_map(|(filter, columns)| {
            let mut columns_to_replace = HashMap::new();
            for col_id in columns {
                for (l, r) in condition.columns() {
                    if col_id == l {
                        columns_to_replace.insert(*col_id, *r);
                        break;
                    } else if col_id == r {
                        columns_to_replace.insert(*col_id, *l);
                        break;
                    }
                }
            }
            if columns_to_replace.is_empty() {
                None
            } else {
                let filter = replace_columns(filter.expr(), &columns_to_replace);
                if current_filters.contains(&filter) {
                    None
                } else {
                    let filer_columns = columns_to_replace.into_values().collect();
                    Some((ScalarNode::from(filter), filer_columns))
                }
            }
        })
        .collect()
}

fn get_join_filters<'a>(
    state: &'a State,
    condition: &JoinUsing,
) -> (Vec<Filter<'a>>, Vec<Filter<'a>>, Vec<Filter<'a>>) {
    let (left_columns, right_columns) = condition.get_columns_pair();
    let left_filters = state.get_filters(&left_columns);
    let right_filters = state.get_filters(&right_columns);

    // Remaining filters must only include filters which use columns from both sides of the join.
    let remaining_filters: Vec<_> = state
        .filters
        .iter()
        .filter(|(_, cols)| {
            let has_left = cols.iter().any(|c| left_columns.contains(c));
            let has_right = cols.iter().any(|c| right_columns.contains(c));
            has_left && has_right
        })
        .filter(|(_, cols)| {
            // Filters which use columns from one side of the join must be ignored
            // (covers the case when a relation is joined with itself).
            let only_left = cols.iter().all(|c| left_columns.contains(c));
            let only_right = cols.iter().all(|c| right_columns.contains(c));
            !only_left && !only_right
        })
        .map(|(expr, cols)| Filter(expr, cols))
        .collect();

    (left_filters, right_filters, remaining_filters)
}

fn rewrite_set_op(
    state: State,
    set_op: SetOperator,
    all: bool,
    left: &RelNode,
    right: &RelNode,
    columns: &[ColumnId],
) -> Result<RelNode, OptimizerError> {
    let left_mapping: HashMap<ColumnId, ColumnId> = columns
        .iter()
        .zip(left.props().logical().output_columns().iter())
        .map(|(l, r)| (*l, *r))
        .collect();

    let right_mapping: HashMap<ColumnId, ColumnId> = columns
        .iter()
        .zip(right.props().logical().output_columns().iter())
        .map(|(l, r)| (*l, *r))
        .collect();

    let mut left_filters = vec![];
    let mut right_filters = vec![];

    for (filter_expr, _) in state.filters {
        let left_filter = replace_columns(filter_expr.expr(), &left_mapping);
        let left_filter = ScalarNode::from(left_filter);
        let left_columns: HashSet<ColumnId> = left_mapping.iter().map(|(_, v)| *v).collect();
        left_filters.push((left_filter, left_columns));

        let right_filter = replace_columns(filter_expr.expr(), &right_mapping);
        let right_filter = ScalarNode::from(right_filter);
        let right_columns: HashSet<ColumnId> = right_mapping.iter().map(|(_, v)| *v).collect();
        right_filters.push((right_filter, right_columns));
    }

    let left_state = State { filters: left_filters };
    let left = rewrite(left_state, left)?;

    let right_state = State { filters: right_filters };
    let right = rewrite(right_state, right)?;
    let columns = columns.to_vec();

    let expr = match set_op {
        SetOperator::Union => LogicalExpr::Union(LogicalUnion {
            left,
            right,
            all,
            columns,
        }),
        SetOperator::Intersect => LogicalExpr::Intersect(LogicalIntersect {
            left,
            right,
            all,
            columns,
        }),
        SetOperator::Except => LogicalExpr::Except(LogicalExcept {
            left,
            right,
            all,
            columns,
        }),
    };

    Ok(RelNode::from(expr))
}

fn rewrite_inputs(state: State, expr: &RelNode) -> Result<RelNode, OptimizerError> {
    rewrite_rel_inputs(expr, |child_expr| rewrite(state.clone(), child_expr))
}

fn replace_columns(expr: &ScalarExpr, columns: &HashMap<ColumnId, ColumnId>) -> ScalarExpr {
    struct ReplaceColumns<'a> {
        columns: &'a HashMap<ColumnId, ColumnId>,
    }
    impl<'a> ExprRewriter<RelNode> for ReplaceColumns<'a> {
        type Error = Infallible;
        fn rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
            if let ScalarExpr::Column(id) = expr {
                match self.columns.get(&id) {
                    Some(new_col) => Ok(ScalarExpr::Column(*new_col)),
                    None => Ok(expr),
                }
            } else {
                Ok(expr)
            }
        }
    }
    let expr = expr.clone();
    // Rewriter never returns an error
    let mut rewriter = ReplaceColumns { columns };
    expr.rewrite(&mut rewriter).unwrap()
}

fn split_filter(filter: &ScalarExpr, filters: &mut Vec<ScalarExpr>) {
    match filter {
        ScalarExpr::BinaryExpr {
            lhs,
            op: BinaryOp::And,
            rhs,
        } => {
            split_filter(lhs.as_ref(), filters);
            split_filter(rhs.as_ref(), filters);
        }
        expr => filters.push(expr.clone()),
    }
}

#[cfg(test)]
mod test {
    use crate::error::OptimizerError;
    use crate::operators::builder::OperatorBuilder;
    use crate::operators::relational::join::JoinType;
    use crate::operators::scalar::aggregates::{AggregateFunction, WindowOrAggregateFunction};
    use crate::operators::scalar::{col, scalar};
    use crate::rules::rewrite::testing::build_and_rewrite_expr;

    use super::*;

    #[test]
    fn test_push_past_project() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1")])?;
                let filter = col("a1").gt(scalar(100));
                let filter_a1 = project.select(Some(filter))?;
                Ok(filter_a1)
            },
            r#"
LogicalProjection cols=[1] exprs: [col:1]
  input: LogicalSelect
    input: LogicalGet A cols=[1, 2]
    filter: Expr col:1 > 100
"#,
        );
    }

    #[test]
    fn test_push_past_project_with_aliases() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let col_a1 = col("a1");
                let col_a2 = col("a2");
                let col_a3 = (col("a1") + col("a2")).alias("a3");

                let project = from_a.project(vec![col_a1, col_a2, col_a3])?;
                let filter = col("a3").gt(scalar(100)).and(col("a1").gt(scalar(10)));
                let filter_a1 = project.select(Some(filter))?;

                Ok(filter_a1)
            },
            r#"
LogicalProjection cols=[1, 2, 3] exprs: [col:1, col:2, col:1 + col:2 AS a3]
  input: LogicalSelect
    input: LogicalGet A cols=[1, 2]
    filter: Expr col:1 + col:2 > 100 AND col:1 > 10
"#,
        );
    }

    #[test]
    fn test_push_past_multiple_projects() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project1 = from_a.project(vec![col("a1")])?;
                let project2 = project1.project(vec![col("a1")])?;
                let project3 = project2.project(vec![col("a1")])?;
                let filter = col("a1").gt(scalar(100));
                let filter_a1 = project3.select(Some(filter))?;
                Ok(filter_a1)
            },
            r#"
LogicalProjection cols=[1] exprs: [col:1]
  input: LogicalProjection cols=[1] exprs: [col:1]
    input: LogicalProjection cols=[1] exprs: [col:1]
      input: LogicalSelect
        input: LogicalGet A cols=[1, 2]
        filter: Expr col:1 > 100
"#,
        );
    }

    #[test]
    fn test_push_past_project_and_combine_filters() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let project = from_a.project(vec![col("a1"), col("a2")])?;

                let filter = col("a2").gt(scalar(100));
                let filter_a1 = project.select(Some(filter))?;
                let project = filter_a1.project(vec![col("a1")])?;

                let filter = col("a1").gt(scalar(100));
                let filter_a1 = project.select(Some(filter))?;
                Ok(filter_a1)
            },
            r#"
LogicalProjection cols=[1] exprs: [col:1]
  input: LogicalProjection cols=[1, 2] exprs: [col:1, col:2]
    input: LogicalSelect
      input: LogicalGet A cols=[1, 2]
      filter: Expr col:1 > 100 AND col:2 > 100
"#,
        );
    }

    #[test]
    fn test_push_select_combine_filters() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let filter = col("a2").gt(scalar(100));
                let filter_a2 = from_a.select(Some(filter))?;

                let filter = col("a1").gt(scalar(100));
                let filter_a1 = filter_a2.select(Some(filter))?;
                Ok(filter_a1)
            },
            r#"
LogicalSelect
  input: LogicalGet A cols=[1, 2]
  filter: Expr col:1 > 100 AND col:2 > 100
"#,
        );
    }

    #[test]
    fn test_push_past_select_with_no_filters() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let no_filter = from_a.select(None)?;
                let filter = col("a1").gt(scalar(100));
                let filter_a1 = no_filter.select(Some(filter))?;
                Ok(filter_a1)
            },
            r#"
LogicalSelect
  input: LogicalSelect
    input: LogicalGet A cols=[1, 2]
    filter: Expr col:1 > 100
"#,
        );
    }

    #[test]
    fn test_push_past_select_with_constant_filters() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let filter = scalar(1).eq(scalar(1));
                let filter_a2 = from_a.select(Some(filter))?;
                let filter = col("a1").gt(scalar(100));
                let filter_a1 = filter_a2.select(Some(filter))?;
                Ok(filter_a1)
            },
            r#"
LogicalSelect
  input: LogicalSelect
    input: LogicalGet A cols=[1, 2]
    filter: Expr col:1 > 100
  filter: Expr 1 = 1
"#,
        );
    }

    #[test]
    fn test_push_past_join_left_side() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2", "a3"])?;
                let from_b = builder.get("B", vec!["b1", "b2", "b3"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;
                let filter = col("a1").gt(scalar(100));
                let select = join.select(Some(filter))?;
                Ok(select)
            },
            r#"
LogicalJoin type=Inner using=[(1, 4)]
  left: LogicalSelect
    input: LogicalGet A cols=[1, 2, 3]
    filter: Expr col:1 > 100
  right: LogicalSelect
    input: LogicalGet B cols=[4, 5, 6]
    filter: Expr col:4 > 100
"#,
        );

        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2", "a3"])?;
                let project_a = from_a.project(vec![col("a1"), col("a2")])?;
                let from_b = builder.get("B", vec!["b1", "b2", "b3"])?;
                let join = project_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;
                let filter = col("a1").gt(scalar(100));
                let select = join.select(Some(filter))?;
                Ok(select)
            },
            r#"
LogicalJoin type=Inner using=[(1, 4)]
  left: LogicalProjection cols=[1, 2] exprs: [col:1, col:2]
    input: LogicalSelect
      input: LogicalGet A cols=[1, 2, 3]
      filter: Expr col:1 > 100
  right: LogicalSelect
    input: LogicalGet B cols=[4, 5, 6]
    filter: Expr col:4 > 100
"#,
        );
    }

    #[test]
    fn test_push_past_join_right_side() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2", "a3"])?;
                let from_b = builder.get("B", vec!["b1", "b2", "b3"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;
                let filter = col("a1").gt(scalar(100));
                let select = join.select(Some(filter))?;
                Ok(select)
            },
            r#"
LogicalJoin type=Inner using=[(1, 4)]
  left: LogicalSelect
    input: LogicalGet A cols=[1, 2, 3]
    filter: Expr col:1 > 100
  right: LogicalSelect
    input: LogicalGet B cols=[4, 5, 6]
    filter: Expr col:4 > 100
"#,
        );

        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2", "a3"])?;
                let from_b = builder.get("B", vec!["b1", "b2", "b3"])?;
                let project_b = from_b.project(vec![col("b1"), col("b2")])?;
                let join = from_a.join_using(project_b, JoinType::Inner, vec![("a1", "b1")])?;
                let filter = col("a1").gt(scalar(100));
                let select = join.select(Some(filter))?;
                Ok(select)
            },
            r#"
LogicalJoin type=Inner using=[(1, 4)]
  left: LogicalSelect
    input: LogicalGet A cols=[1, 2, 3]
    filter: Expr col:1 > 100
  right: LogicalProjection cols=[4, 5] exprs: [col:4, col:5]
    input: LogicalSelect
      input: LogicalGet B cols=[4, 5, 6]
      filter: Expr col:4 > 100
"#,
        );
    }

    #[test]
    fn test_push_past_self_join() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2", "a3"])?;
                let from_b = builder.get("A", vec!["a1", "a2", "a3"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "a1")])?;
                let filter = col("a1").gt(scalar(100));
                let select = join.select(Some(filter))?;
                Ok(select)
            },
            r#"
LogicalJoin type=Inner using=[(1, 1)]
  left: LogicalSelect
    input: LogicalGet A cols=[1, 2, 3]
    filter: Expr col:1 > 100
  right: LogicalSelect
    input: LogicalGet A cols=[1, 2, 3]
    filter: Expr col:1 > 100
"#,
        );
    }

    #[test]
    fn test_push_filters_past_join_and_keep() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2", "a3"])?;
                let from_b = builder.get("B", vec!["b1", "b2", "b3"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;

                // col:a1 = col:b1 AND a1 > 10
                let a1_eq_b1 = col("a1").eq(col("b1"));
                let a1_gt_10 = col("a1").gt(scalar(10));
                let filter = a1_eq_b1.and(a1_gt_10);

                let select = join.select(Some(filter))?;
                Ok(select)
            },
            r#"
LogicalSelect
  input: LogicalJoin type=Inner using=[(1, 4)]
    left: LogicalSelect
      input: LogicalGet A cols=[1, 2, 3]
      filter: Expr col:1 > 10
    right: LogicalSelect
      input: LogicalGet B cols=[4, 5, 6]
      filter: Expr col:4 > 10
  filter: Expr col:1 = col:4 AND col:4 = col:1
"#,
        );
    }

    #[test]
    fn test_keep_complex_filter_before_join() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2", "a3"])?;
                let from_b = builder.get("B", vec!["b1", "b2", "b3"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;

                // col:a1 + col:b1 = 10
                let filter = (col("a1") + col("b1")).eq(scalar(10));

                let select = join.select(Some(filter))?;
                Ok(select)
            },
            r#"
LogicalSelect
  input: LogicalJoin type=Inner using=[(1, 4)]
    left: LogicalGet A cols=[1, 2, 3]
    right: LogicalGet B cols=[4, 5, 6]
  filter: Expr col:1 + col:4 = 10 AND col:4 + col:1 = 10
"#,
        );
    }

    #[test]
    fn test_push_past_join_partilally() {
        rewrite_expr(
            |builder| {
                let from_a = builder.clone().get("A", vec!["a1", "a2", "a3"])?;
                let from_b = builder.get("B", vec!["b1", "b2", "b3"])?;
                let join = from_a.join_using(from_b, JoinType::Inner, vec![("a1", "b1")])?;

                // col:a1 + col:b1 = 10 AND col:a1 > 5
                let a1_plus_b1 = col("a1") + col("b1");
                let a1_plus_b1_eq_10 = a1_plus_b1.eq(scalar(10));
                let a1_gt_5 = col("a1").gt(scalar(5));
                let filter = a1_plus_b1_eq_10.and(a1_gt_5);

                let select = join.select(Some(filter))?;
                Ok(select)
            },
            r#"
LogicalSelect
  input: LogicalJoin type=Inner using=[(1, 4)]
    left: LogicalSelect
      input: LogicalGet A cols=[1, 2, 3]
      filter: Expr col:1 > 5
    right: LogicalSelect
      input: LogicalGet B cols=[4, 5, 6]
      filter: Expr col:4 > 5
  filter: Expr col:1 + col:4 = 10 AND col:4 + col:1 = 10
"#,
        );
    }

    #[test]
    fn test_push_past_aggregate() {
        rewrite_expr(
            |builder| {
                let mut from_a = builder.get("A", vec!["a1"])?;
                let sum_a1 = from_a
                    .aggregate_builder()
                    .add_column("a1")?
                    .add_func("sum", "a1")?
                    .group_by("a1")?
                    .build()?;

                let filter = col("a1").gt(scalar(100));
                let filter_a1 = sum_a1.select(Some(filter))?;

                Ok(filter_a1)
            },
            r#"
LogicalAggregate cols=[1, 2]
  aggr_exprs: [col:1, sum(col:1)]
  group_exprs: [col:1]
  input: LogicalSelect
    input: LogicalGet A cols=[1]
    filter: Expr col:1 > 100
"#,
        )
    }

    #[test]
    fn test_do_not_push_past_aggregate() {
        rewrite_expr(
            |builder| {
                let mut from_a = builder.get("A", vec!["a1"])?;
                let sum_a1 = from_a
                    .aggregate_builder()
                    .add_column("a1")?
                    .add_func("sum", "a1")?
                    .group_by("a1")?
                    .build()?;

                let filter = col("sum").gt(scalar(100));
                let filter_a1 = sum_a1.select(Some(filter))?;

                Ok(filter_a1)
            },
            r#"
LogicalSelect
  input: LogicalAggregate cols=[1, 2]
    aggr_exprs: [col:1, sum(col:1)]
    group_exprs: [col:1]
    input: LogicalGet A cols=[1]
  filter: Expr col:2 > 100
"#,
        )
    }

    #[test]
    fn test_push_past_aggregate_and_projection() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1"])?;
                let sum_a1 = from_a
                    .project(vec![col("a1")])?
                    .aggregate_builder()
                    .add_column("a1")?
                    .add_func("sum", "a1")?
                    .group_by("a1")?
                    .build()?;

                let filter = col("a1").gt(scalar(100));
                let filter_a1 = sum_a1.select(Some(filter))?;

                Ok(filter_a1)
            },
            r#"
LogicalAggregate cols=[1, 2]
  aggr_exprs: [col:1, sum(col:1)]
  group_exprs: [col:1]
  input: LogicalProjection cols=[1] exprs: [col:1]
    input: LogicalSelect
      input: LogicalGet A cols=[1]
      filter: Expr col:1 > 100
"#,
        )
    }

    #[test]
    fn test_push_past_distinct() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let distinct = from_a.distinct(None)?;
                let filter = col("a1").gt(scalar(100));
                let filter_a1 = distinct.select(Some(filter))?;
                Ok(filter_a1)
            },
            r#"
LogicalDistinct cols=[1, 2]
  input: LogicalSelect
    input: LogicalGet A cols=[1, 2]
    filter: Expr col:1 > 100
"#,
        );
    }

    #[test]
    fn test_push_past_limit_offset() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let limit_offset = from_a.offset(10)?.limit(5)?;
                let filter = col("a1").gt(scalar(100));
                let filter_a1 = limit_offset.select(Some(filter))?;
                Ok(filter_a1)
            },
            r#"
LogicalLimit rows=5
  input: LogicalOffset rows=10
    input: LogicalSelect
      input: LogicalGet A cols=[1, 2]
      filter: Expr col:1 > 100
"#,
        );
    }

    #[test]
    fn test_push_past_union() {
        fn rewrite_union(all: bool) {
            rewrite_expr(
                |builder| {
                    let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
                    let from_b = builder.get("B", vec!["b1", "b2"])?;
                    let intersect = if all {
                        from_a.union_all(from_b)?
                    } else {
                        from_a.union(from_b)?
                    };
                    let filter = col("a1").gt(scalar(100)).and(col("a2").lt(scalar(75)));
                    let filter_a1 = intersect.select(Some(filter))?;
                    Ok(filter_a1)
                },
                r#"
LogicalUnion all=:all cols=[5, 6]
  left: LogicalSelect
    input: LogicalGet A cols=[1, 2]
    filter: Expr col:1 > 100 AND col:2 < 75
  right: LogicalSelect
    input: LogicalGet B cols=[3, 4]
    filter: Expr col:3 > 100 AND col:4 < 75
"#
                .replace(":all", format!("{}", all).as_str())
                .as_str(),
            );
        }

        rewrite_union(false);
        rewrite_union(true);
    }

    #[test]
    fn test_push_past_intersect() {
        fn rewrite_intersect(all: bool) {
            rewrite_expr(
                |builder| {
                    let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
                    let from_b = builder.get("B", vec!["b1", "b2"])?;
                    let intersect = if all {
                        from_a.intersect_all(from_b)?
                    } else {
                        from_a.intersect(from_b)?
                    };
                    let filter = col("a1").gt(scalar(100)).and(col("a2").lt(scalar(75)));
                    let filter_a1 = intersect.select(Some(filter))?;
                    Ok(filter_a1)
                },
                r#"
LogicalIntersect all=:all cols=[5, 6]
  left: LogicalSelect
    input: LogicalGet A cols=[1, 2]
    filter: Expr col:1 > 100 AND col:2 < 75
  right: LogicalSelect
    input: LogicalGet B cols=[3, 4]
    filter: Expr col:3 > 100 AND col:4 < 75
"#
                .replace(":all", format!("{}", all).as_str())
                .as_str(),
            );
        }

        rewrite_intersect(false);
        rewrite_intersect(true);
    }

    #[test]
    fn test_push_past_except() {
        fn rewrite_except(all: bool) {
            rewrite_expr(
                |builder| {
                    let from_a = builder.clone().get("A", vec!["a1", "a2"])?;
                    let from_b = builder.get("B", vec!["b1", "b2"])?;
                    let intersect = if all {
                        from_a.except_all(from_b)?
                    } else {
                        from_a.except(from_b)?
                    };
                    let filter = col("a1").gt(scalar(100)).and(col("a2").lt(scalar(75)));
                    let filter_a1 = intersect.select(Some(filter))?;
                    Ok(filter_a1)
                },
                r#"
LogicalExcept all=:all cols=[5, 6]
  left: LogicalSelect
    input: LogicalGet A cols=[1, 2]
    filter: Expr col:1 > 100 AND col:2 < 75
  right: LogicalSelect
    input: LogicalGet B cols=[3, 4]
    filter: Expr col:3 > 100 AND col:4 < 75
"#
                .replace(":all", format!("{}", all).as_str())
                .as_str(),
            );
        }

        rewrite_except(false);
        rewrite_except(true);
    }

    #[test]
    fn test_push_past_window_aggregate() {
        rewrite_expr(
            |builder| {
                let from_a = builder.get("A", vec!["a1", "a2"])?;
                let window_aggr = ScalarExpr::WindowAggregate {
                    func: WindowOrAggregateFunction::Aggregate(AggregateFunction::Count),
                    args: vec![col("a1")],
                    partition_by: vec![],
                    order_by: vec![],
                };
                let project = from_a.project(vec![col("a1"), window_aggr])?;
                let filter = col("a1").gt(scalar(100));
                let filter_a1 = project.select(Some(filter))?;
                Ok(filter_a1)
            },
            r#"
LogicalProjection cols=[1, 3] exprs: [col:1, col:3]
  input: LogicalWindowAggregate cols=[1, 3]
    input: LogicalSelect
      input: LogicalGet A cols=[1, 2]
      filter: Expr col:1 > 100
    window_expr: Expr count(col:1) OVER()
"#,
        );
    }

    fn rewrite_expr<F>(f: F, expected: &str)
    where
        F: FnOnce(OperatorBuilder) -> Result<OperatorBuilder, OptimizerError>,
    {
        build_and_rewrite_expr(f, predicate_push_down, expected)
    }
}
