use crate::error::OptimizerError;
use crate::meta::MetadataRef;
use crate::operators::relational::logical::{LogicalExpr, LogicalProjection, LogicalSelect};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::BinaryOp;
use crate::operators::scalar::{exprs, ScalarExpr, ScalarNode};
use crate::operators::Operator;

/// Provides methods to build relational operators.
pub trait BuildOperators {
    /// A reference to the metadata.
    fn metadata(&self) -> MetadataRef;

    /// Adds the given relational operator to the operator tree.
    fn add_operator(&mut self, expr: Operator) -> Result<RelNode, OptimizerError>;

    /// Creates an instance of a select operator.
    fn select(&mut self, input: RelNode, filter: Option<BuildFilter>) -> Result<Operator, OptimizerError>;

    /// Creates an instance of a left semi-join operator.
    fn left_semi_join(
        &mut self,
        left: RelNode,
        right: RelNode,
        condition: BuildFilter,
    ) -> Result<Operator, OptimizerError>;

    /// Creates an instance of a left join operator.
    fn left_join(&mut self, left: RelNode, right: RelNode, condition: BuildFilter) -> Result<Operator, OptimizerError>;
}

/// A filter expression that can be fully constructed or not.
//TODO: Find a better name.
pub enum BuildFilter {
    /// A new expression.
    New(ScalarExpr),
    /// A fully constructed expression.
    Constructed(ScalarNode),
}

impl From<ScalarNode> for BuildFilter {
    fn from(node: ScalarNode) -> Self {
        BuildFilter::Constructed(node)
    }
}

impl From<ScalarExpr> for BuildFilter {
    fn from(expr: ScalarExpr) -> Self {
        BuildFilter::New(expr)
    }
}

/// Checks whether the given filter expression can possibly contain subqueries that can be decorrelated.
pub fn possibly_has_correlated_subqueries(filter: &ScalarExpr) -> bool {
    for filter in split_conjunction(filter) {
        match &filter {
            ScalarExpr::Exists { query: subquery, .. } => {
                let mut expr = Some(subquery);
                let mut condition = None;
                while let Some(input_expr) = expr {
                    match input_expr.expr().logical() {
                        LogicalExpr::Projection(projection) => {
                            // We can safely ignore projections that contain only columns and scalar expressions
                            // in EXISTS (SELECT [projection]).
                            let can_ignore_projection = projection
                                .exprs
                                .iter()
                                .all(|expr| matches!(expr.expr(), ScalarExpr::Column(_) | ScalarExpr::Scalar(_)));
                            if !can_ignore_projection {
                                return false;
                            }
                            expr = Some(&projection.input)
                        }
                        LogicalExpr::Select(LogicalSelect {
                            filter: Some(filter_expr),
                            ..
                        }) => {
                            condition = Some((filter_expr, input_expr));
                            break;
                        }
                        _ => return false,
                    }
                }

                return if let Some((condition, input_expr)) = condition {
                    let cols = exprs::collect_columns(condition.expr());
                    let outer_columns = &input_expr.props().logical().outer_columns;

                    cols.iter().any(|id| outer_columns.contains(id))
                } else {
                    false
                };
            }
            ScalarExpr::InSubQuery { query: subquery, .. } => {
                //
                return matches!(subquery.expr().logical(), LogicalExpr::Projection(_));
            }
            _ => {}
        }
    }

    false
}

/// Attempts to decorrelate subqueries from the given filter expression.
pub fn decorrelate_subqueries<T>(
    input: RelNode,
    filter: ScalarNode,
    builder: &mut T,
) -> Result<Option<Operator>, OptimizerError>
where
    T: BuildOperators,
{
    let filter_exprs = split_conjunction(filter.expr());
    let exists_predicates = filter_exprs.iter().filter(|expr| match expr {
        ScalarExpr::Exists { .. } => true,
        ScalarExpr::InSubQuery { .. } => false,
        _ => true,
    });

    let in_subqueries_predicates = filter_exprs.iter().filter(|expr| match expr {
        ScalarExpr::Exists { .. } => false,
        ScalarExpr::InSubQuery { .. } => true,
        _ => true,
    });

    let (exists, exists_predicates): (Vec<_>, Vec<_>) =
        exists_predicates.partition(|expr| matches!(expr, ScalarExpr::Exists { .. }));

    let (in_subquery, in_subquery_predicates): (Vec<_>, Vec<_>) =
        in_subqueries_predicates.partition(|expr| matches!(expr, ScalarExpr::InSubQuery { .. }));

    // Filter contains both EXISTS and IN ignore it.
    if !exists.is_empty() && !in_subquery.is_empty() || exists.len() > 1 || in_subquery.len() > 1 {
        return Ok(None);
    }

    if !exists.is_empty() {
        if let ScalarExpr::Exists { not, query: subquery } = exists[0] {
            let exists_subquery = ExistsSubQuery { not: *not, subquery };
            let mut expr = Some(subquery);
            while let Some(input_expr) = expr {
                match input_expr.expr().logical() {
                    // We can ignore the projections (See has_correlated_subqueries)
                    LogicalExpr::Projection(projection) => expr = Some(&projection.input),
                    LogicalExpr::Select(select @ LogicalSelect { filter: Some(_), .. }) => {
                        return rewrite_exists_subquery(input, exists_subquery, select, exists_predicates, builder)
                    }
                    // TODO: Rewrite join conditions?
                    _ => return Ok(None),
                }
            }
        }
    } else if !in_subquery.is_empty() {
        if let ScalarExpr::InSubQuery {
            not,
            expr,
            query: subquery,
        } = in_subquery[0]
        {
            let in_subquery = InSubquery {
                expr,
                not: *not,
                subquery,
            };
            if let LogicalExpr::Projection(projection) = subquery.expr().logical() {
                return rewrite_in_subquery(input, in_subquery, projection, in_subquery_predicates, builder);
            }
        }
    }

    Ok(None)
}

struct ExistsSubQuery<'a> {
    not: bool,
    subquery: &'a RelNode,
}

struct InSubquery<'a> {
    expr: &'a ScalarExpr,
    not: bool,
    subquery: &'a RelNode,
}

fn rewrite_exists_subquery<T>(
    input: RelNode,
    exists_subquery: ExistsSubQuery,
    select: &LogicalSelect,
    mut remaining_predicates: Vec<&ScalarExpr>,
    builder: &mut T,
) -> Result<Option<Operator>, OptimizerError>
where
    T: BuildOperators,
{
    let LogicalSelect {
        input: select_input,
        filter,
    } = select;
    let filter = filter.as_ref().expect("Select must have a filter");

    if !exists_subquery.not {
        // Rewrite:
        //  SELECT a1 FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b1 = a2)
        // >>>>:
        //   SELECT a1 FROM a LEFT SEMI-JOIN b ON b1 = a2
        //

        // SELECT a1 FROM a WHERE EXISTS (SELECT 1 FROM b WHERE EXISTS (SELECT 1 FROM c WHERE c1 = b2))
        // >>>>>
        // SELECT a1 FROM a WHERE EXISTS (SELECT 1 FROM b LEFT SEMI-JOIN c ON c1 = b2)
        //

        // SELECT a1 FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b1 = a2 AND EXISTS (SELECT 1 FROM c WHERE c1 = b2))
        // >>>>>
        // SELECT a1 FROM a WHERE EXISTS (SELECT 1 FROM b SEMI-JOIN c ON c1 = b2 WHERE b1 = a2)
        // >>>>>
        // SELECT a1 FROM a LEFT SEMI-JOIN b ON b1 = a2 LEFT SEMI-JOIN c ON c1 = b2

        let join = builder.left_semi_join(input, select_input.clone(), filter.clone().into())?;

        if remaining_predicates.is_empty() {
            Ok(Some(join))
        } else {
            let join = builder.add_operator(join)?;

            let first = remaining_predicates.swap_remove(0).clone();
            let filter = remaining_predicates.into_iter().fold(first, |init, expr| init.and(expr.clone()));
            let select = builder.select(join, Some(filter.into()))?;

            Ok(Some(select))
        }
    } else {
        // Rewrite:
        //  SELECT a1 FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b1 = a2)
        // >>>>:
        //  SELECT a1 FROM a LEFT JOIN b ON b1 = a2 WHERE b1 is NULL
        //

        // NOT EXISTS (... FROM b WHERE b1 = a2)
        // >>>> + FILTER:
        // b1 is NULL

        // NOT EXISTS (... FROM b WHERE b1 + b2 = a2)
        // >>>>> + FILTER:
        // b1 is NULL AND b2 is NULL

        let conjunctions = split_conjunction(filter.expr());
        let mut is_null_columns = vec![];

        for expr in conjunctions {
            for col_id in exprs::collect_columns(&expr) {
                if exists_subquery.subquery.props().logical().outer_columns.contains(&col_id) {
                    continue;
                }
                let is_null = ScalarExpr::IsNull {
                    not: false,
                    expr: Box::new(ScalarExpr::Column(col_id)),
                };
                is_null_columns.push(is_null);
            }
        }

        if is_null_columns.is_empty() {
            Ok(None)
        } else {
            let first = is_null_columns.swap_remove(0);
            let is_null_filter = is_null_columns.into_iter().fold(first, |rest, expr| rest.and(expr));

            let join = builder.left_join(input, select_input.clone(), filter.clone().into())?;
            let join = builder.add_operator(join)?;

            let filter = remaining_predicates.into_iter().fold(is_null_filter, |init, expr| init.and(expr.clone()));
            let select = builder.select(join, Some(filter.into()))?;

            Ok(Some(select))
        }
    }
}

fn rewrite_in_subquery<T>(
    input: RelNode,
    in_subquery: InSubquery,
    projection: &LogicalProjection,
    mut remaining_predicates: Vec<&ScalarExpr>,
    builder: &mut T,
) -> Result<Option<Operator>, OptimizerError>
where
    T: BuildOperators,
{
    let (filter, select_input) = if let LogicalExpr::Select(LogicalSelect {
        filter: Some(filter),
        input,
    }) = projection.input.expr().logical()
    {
        (Some(filter.expr().clone()), input.clone())
    } else {
        (None, in_subquery.subquery.clone())
    };

    if !in_subquery.not {
        // SELECT a1 FROM a WHERE a1 IN (SELECT b1 FROM b)
        //>>>>
        // SELECT a1 FROM a LEFT SEMI-JOIN b ON b1 = a1 [+ WHERE remaining_predicates]

        // SELECT a1 FROM a WHERE a1 IN (SELECT b1 FROM b WHERE b2 > 3)
        //>>>>
        // SELECT a1 FROM a LEFT SEMI-JOIN b ON b1 = a1 AND b2 > 3 [+ WHERE remaining_predicates]

        let col_id = in_subquery.subquery.props().logical().output_columns()[0];
        let col_expr = builder
            .metadata()
            .get_column(&col_id)
            .expr()
            .cloned()
            .unwrap_or(ScalarExpr::Column(col_id));

        let join_condition = in_subquery.expr.clone().eq(col_expr);
        let join_condition = match filter {
            Some(expr) => join_condition.and(expr),
            _ => join_condition,
        };
        let join = builder.left_semi_join(input, select_input, join_condition.into())?;

        if remaining_predicates.is_empty() {
            Ok(Some(join))
        } else {
            let join = builder.add_operator(join)?;

            let first = remaining_predicates.swap_remove(0).clone();
            let filter = remaining_predicates.into_iter().fold(first, |init, expr| init.and(expr.clone()));
            let select = builder.select(join, Some(filter.into()))?;

            Ok(Some(select))
        }
    } else {
        // SELECT a1 FROM a WHERE a1 NOT IN (SELECT b1 FROM b [WHERE remaining_predicates])
        //>>>>
        // SELECT a1 FROM a LEFT JOIN b ON b1 = a1 WHERE b1 IS NULL [+ remaining_predicates]

        let col_id = in_subquery.subquery.props().logical().output_columns()[0];
        let input_produces_no_columns = projection.input.props().logical().output_columns().is_empty();

        if builder.metadata().get_column(&col_id).expr().is_some() && !input_produces_no_columns {
            //TODO: support a1 NOT IN (SELECT <non_column> FROM ...)
            return Ok(None);
        }
        let col_expr = ScalarExpr::Column(col_id);

        let join_condition = in_subquery.expr.clone().eq(col_expr.clone());
        let join_condition = match filter {
            Some(expr) => join_condition.and(expr),
            _ => join_condition,
        };
        let join = builder.left_join(input, select_input, join_condition.into())?;
        let join = builder.add_operator(join)?;

        let is_null_filter = ScalarExpr::IsNull {
            not: false,
            expr: Box::new(col_expr),
        };
        let select_filter = remaining_predicates.into_iter().fold(is_null_filter, |init, expr| init.and(expr.clone()));
        let select = builder.select(join, Some(select_filter.into()))?;

        Ok(Some(select))
    }
}

fn split_conjunction(filter: &ScalarExpr) -> Vec<ScalarExpr> {
    fn split(filter: &ScalarExpr, out: &mut Vec<ScalarExpr>) {
        match filter {
            ScalarExpr::BinaryExpr {
                lhs,
                op: BinaryOp::And,
                rhs,
            } => {
                split(lhs, out);
                split(rhs, out);
            }
            expr => out.push(expr.clone()),
        }
    }

    let mut out = vec![];
    split(filter, &mut out);
    out
}
