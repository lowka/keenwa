use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::not_implemented;
use crate::operators::builder::projection::{ProjectionList, ProjectionListBuilder};
use crate::operators::builder::scope::OperatorScope;
use crate::operators::builder::{OperatorBuilder, RewriteExprs, ValidateFilterExpr, ValidateProjectionExpr};
use crate::operators::relational::logical::{LogicalAggregate, LogicalExpr, LogicalWindowAggregate};
use crate::operators::relational::RelNode;
use crate::operators::scalar::aggregates::AggregateFunction;
use crate::operators::scalar::expr::{Expr, ExprVisitor};
use crate::operators::scalar::exprs::collect_columns;
use crate::operators::scalar::value::ScalarValue;
use crate::operators::scalar::{get_subquery, ScalarExpr, ScalarNode};
use crate::operators::{Operator, OperatorExpr, OperatorMetadata};
use itertools::Itertools;
use std::collections::HashSet;
use std::convert::{Infallible, TryFrom};

/// A builder for aggregate operators.  
pub struct AggregateBuilder<'a> {
    pub(super) builder: &'a mut OperatorBuilder,
    pub(super) aggr_exprs: Vec<AggrExpr>,
    pub(super) group_exprs: Vec<ScalarExpr>,
    pub(super) having: Option<ScalarExpr>,
}

pub(super) enum AggrExpr {
    Function {
        func: AggregateFunction,
        distinct: bool,
        column: String,
    },
    Column(String),
    Expr(ScalarExpr),
}

impl AggregateBuilder<'_> {
    /// Adds aggregate function `func` with the given column as an argument.
    pub fn add_func(mut self, func: &str, column_name: &str) -> Result<Self, OptimizerError> {
        let func = AggregateFunction::try_from(func)
            .map_err(|_| OptimizerError::argument(format!("Unknown aggregate function {}", func)))?;
        let aggr_expr = AggrExpr::Function {
            func,
            distinct: false,
            column: column_name.into(),
        };
        self.aggr_exprs.push(aggr_expr);
        Ok(self)
    }

    /// Adds column to the aggregate operator. The given column must appear in group_by clause.
    pub fn add_column(mut self, column_name: &str) -> Result<Self, OptimizerError> {
        self.aggr_exprs.push(AggrExpr::Column(column_name.into()));
        Ok(self)
    }

    /// Adds an expression to the aggregate operator.
    pub fn add_expr(mut self, expr: ScalarExpr) -> Result<Self, OptimizerError> {
        self.aggr_exprs.push(AggrExpr::Expr(expr));
        Ok(self)
    }

    /// Adds grouping by the given column.
    pub fn group_by(mut self, column_name: &str) -> Result<Self, OptimizerError> {
        self.group_exprs.push(ScalarExpr::ColumnName(column_name.into()));
        Ok(self)
    }

    /// Adds grouping by the given expression.
    pub fn group_by_expr(mut self, expr: ScalarExpr) -> Result<Self, OptimizerError> {
        self.group_exprs.push(expr);
        Ok(self)
    }

    /// Adds HAVING clause.
    pub fn having(mut self, expr: ScalarExpr) -> Result<Self, OptimizerError> {
        self.having = Some(expr);
        Ok(self)
    }

    /// Builds an aggregate operator and adds to an operator tree.
    pub fn build(self) -> Result<OperatorBuilder, OptimizerError> {
        let (input, scope) = self.builder.rel_node()?;

        let metadata = self.builder.metadata.clone();
        let has_window_functions = has_window_aggregates(self.aggr_exprs.iter().filter_map(|expr| match expr {
            AggrExpr::Expr(expr) => Some(expr),
            _ => None,
        }));

        enum ProjectionItem {
            // Positional items will be taken from the projection list
            Positional(usize),
            // Window expressions will be constructed using the arguments from the given positions.
            WindowExpr {
                expr: Option<ScalarExpr>,
                arg_positions: Vec<usize>,
            },
        }
        let mut original_projection: Vec<ProjectionItem> = vec![];

        let (projection_list, non_aggregate_columns) = if has_window_functions {
            let mut new_aggr_exprs = vec![];

            for expr in self.aggr_exprs {
                match expr {
                    AggrExpr::Function { .. } => {
                        original_projection.push(ProjectionItem::Positional(new_aggr_exprs.len()));
                        new_aggr_exprs.push(expr);
                    }
                    AggrExpr::Column(name) => {
                        original_projection.push(ProjectionItem::Positional(new_aggr_exprs.len()));
                        new_aggr_exprs.push(AggrExpr::Column(name));
                    }
                    AggrExpr::Expr(ScalarExpr::WindowAggregate {
                        func,
                        args,
                        partition_by,
                        order_by,
                    }) => {
                        let item = ScalarExpr::WindowAggregate {
                            func,
                            args: args.clone(),
                            partition_by: partition_by.clone(),
                            order_by: order_by.clone(),
                        };
                        let mut window_expr_args = vec![];
                        args.into_iter().for_each(|expr| {
                            window_expr_args.push(new_aggr_exprs.len());
                            new_aggr_exprs.push(AggrExpr::Expr(expr));
                        });
                        original_projection.push(ProjectionItem::WindowExpr {
                            expr: Some(item),
                            arg_positions: window_expr_args,
                        });

                        new_aggr_exprs.extend(partition_by.into_iter().map(AggrExpr::Expr));
                        new_aggr_exprs.extend(order_by.into_iter().map(AggrExpr::Expr));
                    }
                    AggrExpr::Expr(expr) => {
                        original_projection.push(ProjectionItem::Positional(new_aggr_exprs.len()));
                        new_aggr_exprs.push(AggrExpr::Expr(expr));
                    }
                }
            }

            build_aggregates(self.builder, &scope, metadata.clone(), new_aggr_exprs)?
        } else {
            build_aggregates(self.builder, &scope, metadata.clone(), self.aggr_exprs)?
        };

        let ProjectionList {
            column_ids,
            output_columns,
            projection: aggr_exprs,
        } = projection_list;

        let (group_exprs, group_by_columns) = build_group_by_exprs(
            self.builder,
            &scope,
            metadata.clone(),
            self.group_exprs,
            &aggr_exprs,
            &non_aggregate_columns,
        )?;

        let having_expr =
            build_having(self.builder, &scope, metadata, self.having, &non_aggregate_columns, &group_by_columns)?;

        let aggregate = LogicalExpr::Aggregate(LogicalAggregate {
            input,
            aggr_exprs,
            group_exprs,
            columns: column_ids.clone(),
            having: having_expr,
        });

        let operator = Operator::from(OperatorExpr::from(aggregate));
        let builder = OperatorBuilder::from_builder(self.builder, operator, &scope, output_columns);

        if has_window_functions {
            // original_projection should contain column ids of the columns
            // produced by an aggregate operator + window functions with replaced arguments
            // all at correct positions.
            let original_projection = original_projection
                .into_iter()
                .map(|expr| match expr {
                    ProjectionItem::Positional(position) => {
                        let col_id = column_ids[position];
                        ScalarExpr::Column(col_id)
                    }
                    ProjectionItem::WindowExpr {
                        mut expr,
                        arg_positions,
                    } => {
                        let mut expr = expr.take().expect("No window aggr expression");
                        if let ScalarExpr::WindowAggregate {
                            partition_by, order_by, ..
                        } = &mut expr
                        {
                            let mut new_children: Vec<_> = arg_positions
                                .into_iter()
                                .map(|i| {
                                    let col_id = column_ids[i];
                                    ScalarExpr::Column(col_id)
                                })
                                .collect();
                            new_children.extend_from_slice(partition_by);
                            new_children.extend_from_slice(order_by);
                            expr.with_children(new_children)
                        } else {
                            unreachable!("Expected a WindowAggregate expression but got {}", expr)
                        }
                    }
                })
                .collect();

            builder.projection_with_window_functions(original_projection, true, false, true)
        } else {
            Ok(builder)
        }
    }
}

fn disallow_nested_subqueries(expr: &ScalarExpr, clause: &str, location: &str) -> Result<(), OptimizerError> {
    struct DisallowNestedSubqueries<'a> {
        clause: &'a str,
        location: &'a str,
    }
    impl<'a> ExprVisitor<RelNode> for DisallowNestedSubqueries<'a> {
        type Error = OptimizerError;

        fn post_visit(&mut self, expr: &ScalarExpr) -> Result<(), Self::Error> {
            match expr {
                ScalarExpr::Column(_) => {}
                ScalarExpr::ColumnName(_) => {}
                ScalarExpr::Scalar(_) => {}
                ScalarExpr::BinaryExpr { .. } => {}
                ScalarExpr::Cast { .. } => {}
                ScalarExpr::Not(_) => {}
                ScalarExpr::IsNull { .. } => {}
                ScalarExpr::Negation(_) => {}
                ScalarExpr::Alias(_, _) => {}
                ScalarExpr::Case { .. } => {}
                ScalarExpr::Aggregate { .. } => {}
                ScalarExpr::Wildcard(_) => {
                    let message = format!("{}: Wildcard expressions are not allowed", self.clause);
                    return Err(OptimizerError::argument(message));
                }
                _ if get_subquery(expr).is_some() => {
                    let message = format!("{}: Subqueries in {} are not implemented", self.clause, self.location);
                    not_implemented!(message)
                }
                _ => {}
            }
            Ok(())
        }
    }
    let mut visitor = DisallowNestedSubqueries { clause, location };
    expr.accept(&mut visitor)
}

fn build_aggregates(
    builder: &mut OperatorBuilder,
    scope: &OperatorScope,
    metadata: OperatorMetadata,
    aggr_exprs: Vec<AggrExpr>,
) -> Result<(ProjectionList, HashSet<ColumnId>), OptimizerError> {
    let mut projection_builder = ProjectionListBuilder::new(builder, scope, true);
    let mut non_aggregate_columns = HashSet::new();

    for expr in aggr_exprs {
        match expr {
            AggrExpr::Function { func, distinct, column } => {
                let mut rewriter =
                    RewriteExprs::new(scope, metadata.clone(), ValidateProjectionExpr::projection_expr());
                let expr = ScalarExpr::ColumnName(column);
                let expr = expr.rewrite(&mut rewriter)?;
                let aggr_expr = ScalarExpr::Aggregate {
                    func,
                    distinct,
                    args: vec![expr],
                    filter: None,
                };
                projection_builder.add_expr(aggr_expr)?;
            }
            AggrExpr::Column(name) => {
                let expr = ScalarExpr::ColumnName(name);
                let (column_id, _) = projection_builder.add_expr(expr)?;

                non_aggregate_columns.insert(column_id);
            }
            AggrExpr::Expr(expr) => {
                let (_, expr) = projection_builder.add_expr(expr)?;

                // AggregateBuilder::disallow_nested_subqueries(expr, "AGGREGATE", "aggregate function/expression")?;

                let used_columns = match &expr {
                    ScalarExpr::Aggregate { .. } => vec![],
                    ScalarExpr::Alias(expr, _) if matches!(expr.as_ref(), ScalarExpr::Aggregate { .. }) => {
                        vec![]
                    }
                    _ => collect_columns(expr),
                };
                non_aggregate_columns.extend(used_columns);
            }
        }
    }

    let projection_list = projection_builder.build();
    Ok((projection_list, non_aggregate_columns))
}

fn build_group_by_exprs(
    builder: &mut OperatorBuilder,
    scope: &OperatorScope,
    metadata: OperatorMetadata,
    exprs: Vec<ScalarExpr>,
    aggr_exprs: &[ScalarNode],
    non_aggregate_columns: &HashSet<ColumnId>,
) -> Result<(Vec<ScalarNode>, HashSet<ColumnId>), OptimizerError> {
    let mut group_exprs = vec![];
    let mut group_by_columns = HashSet::new();

    for expr in exprs {
        let expr = if let ScalarExpr::Scalar(ScalarValue::Int32(Some(pos))) = expr {
            let pos = pos - 1;
            if pos >= 0 && (pos as usize) < aggr_exprs.len() {
                aggr_exprs[pos as usize].expr().clone()
            } else {
                // query error
                return Err(OptimizerError::argument(format!("GROUP BY: position {} is not in select list", pos)));
            }
        } else {
            expr
        };

        // AggregateBuilder::disallow_nested_subqueries(&expr, "GROUP BY", "expressions")?;

        let expr = match expr {
            ScalarExpr::Column(_) | ScalarExpr::ColumnName(_) => {
                let mut rewriter =
                    RewriteExprs::new(scope, metadata.clone(), ValidateProjectionExpr::projection_expr());
                let expr = expr.rewrite(&mut rewriter)?;
                let expr = builder.add_scalar_node(expr, scope)?;
                if let ScalarExpr::Column(id) = expr.expr() {
                    group_by_columns.insert(*id);
                } else {
                    unreachable!("GROUP BY: unexpected column expression: {}", expr.expr())
                }
                expr
            }
            ScalarExpr::SubQuery(_) => ScalarNode::from(expr),
            ScalarExpr::Scalar(ScalarValue::Int32(_)) => unreachable!("GROUP BY: positional argument"),
            ScalarExpr::Scalar(_) => {
                // query error
                return Err(OptimizerError::argument("GROUP BY: not integer constant argument"));
            }
            ScalarExpr::Aggregate { .. } => {
                // query error
                return Err(OptimizerError::argument("GROUP BY: Aggregate functions are not allowed"));
            }
            _ => {
                // query error
                let message = format!("GROUP BY: unsupported expression: {}", expr);
                return Err(OptimizerError::argument(message));
            }
        };
        group_exprs.push(expr);
    }

    for col_id in non_aggregate_columns.iter() {
        if !group_by_columns.contains(col_id) {
            // query error. Add table/table alias
            let column = metadata.get_column(col_id);
            return Err(OptimizerError::argument(format!(
                "AGGREGATE column {} must appear in GROUP BY clause",
                column.name()
            )));
        }
    }

    Ok((group_exprs, group_by_columns))
}

fn build_having(
    builder: &mut OperatorBuilder,
    scope: &OperatorScope,
    metadata: OperatorMetadata,
    mut having: Option<ScalarExpr>,
    non_aggregate_columns: &HashSet<ColumnId>,
    group_by_columns: &HashSet<ColumnId>,
) -> Result<Option<ScalarNode>, OptimizerError> {
    if let Some(expr) = having.take() {
        disallow_nested_subqueries(&expr, "AGGREGATE", "HAVING clause")?;

        let mut rewriter = RewriteExprs::new(scope, metadata, ValidateFilterExpr::where_clause());
        let expr = expr.rewrite(&mut rewriter)?;
        let mut validate = ValidateHavingClause {
            non_aggregate_columns,
            group_by_columns,
        };

        expr.accept(&mut validate)?;

        struct ValidateHavingClause<'a> {
            non_aggregate_columns: &'a HashSet<ColumnId>,
            group_by_columns: &'a HashSet<ColumnId>,
        }

        impl<'a> ExprVisitor<RelNode> for ValidateHavingClause<'a> {
            type Error = OptimizerError;

            fn pre_visit(&mut self, expr: &ScalarExpr) -> Result<bool, Self::Error> {
                match expr {
                    ScalarExpr::Aggregate { .. } => Ok(false),
                    ScalarExpr::Column(id) => {
                        if self.non_aggregate_columns.contains(id) || self.group_by_columns.contains(id) {
                            Ok(false)
                        } else {
                            Err(OptimizerError::argument(format!(
                                "AGGREGATE: Column {} must appear in GROUP BY clause or an aggregate function",
                                id
                            )))
                        }
                    }
                    _ => Ok(true),
                }
            }

            fn post_visit(&mut self, _expr: &ScalarExpr) -> Result<(), Self::Error> {
                Ok(())
            }
        }

        Ok(Some(builder.add_scalar_node(expr, scope)?))
    } else {
        Ok(None)
    }
}

pub(super) fn has_window_aggregates<'a, T>(mut exprs: T) -> bool
where
    T: Iterator<Item = &'a ScalarExpr>,
{
    #[derive(Default)]
    struct HasWindowAggregates {
        result: bool,
    }
    impl ExprVisitor<RelNode> for HasWindowAggregates {
        type Error = Infallible;

        fn pre_visit(&mut self, expr: &Expr<RelNode>) -> Result<bool, Self::Error> {
            if matches!(expr, ScalarExpr::WindowAggregate { .. }) {
                self.result = true;
            }
            Ok(!self.result)
        }

        fn post_visit(&mut self, _expr: &ScalarExpr) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    exprs.any(|expr| {
        let mut visitor = HasWindowAggregates::default();
        // HasWindowAggregates never returns an error.
        expr.accept(&mut visitor).unwrap();
        visitor.result
    })
}

pub(super) fn add_window_aggregates(
    builder: &mut OperatorBuilder,
    input: RelNode,
    scope: OperatorScope,
    column_ids: &[ColumnId],
    new_exprs: Vec<ScalarNode>,
) -> Result<(RelNode, OperatorScope, Vec<ScalarNode>), OptimizerError> {
    //
    // Collect Window Aggregate expressions sorted by the number of partition options.
    // So that expressions with the less partitioning options are placed closer to the root node of an operator tree:
    //
    // SELECT rank OVER (col:1, col:2), rank OVER (), rank OVER (col:1, col:2, col:3)
    //
    // Translated into:
    //
    // WindowAggregate expr=rank() partition=[]
    //   input: WindowAggregate expr=rank() partition=[col:1, col:2]
    //     input: WindowAggregate expr=rank() partition=[col:1, col:2, col:3]
    //       input: Scan A
    //
    // The equivalent window aggregate expressions across a single projection should not be translated into
    // different WindowAggregate operators and must be reused.
    //
    struct WindowExprKey {
        expr: ScalarExpr,
        expr_str: String,
        partition_keys: Vec<ColumnId>,
    }

    impl PartialEq for WindowExprKey {
        fn eq(&self, other: &Self) -> bool {
            self.expr_str == other.expr_str
        }
    }

    let window_aggregates = collect_window_aggregates(&new_exprs);
    let mut window_aggregates: Vec<ScalarExpr> = window_aggregates
        .into_iter()
        .filter_map(|expr| {
            if let ScalarExpr::WindowAggregate { partition_by, .. } = &expr {
                Some(WindowExprKey {
                    expr_str: format!("{}", expr),
                    partition_keys: partition_by
                        .iter()
                        .map(|expr| match expr {
                            ScalarExpr::Column(id) => id,
                            _ => unreachable!(),
                        })
                        .copied()
                        .collect(),
                    expr,
                })
            } else {
                None
            }
        })
        .sorted_by(|a, b| a.partition_keys.cmp(&b.partition_keys))
        .dedup()
        .map(|expr| expr.expr)
        .collect();

    let mut input = input;
    let mut scope = scope;
    let mut window_idx = 0;

    // Add a logical window aggregate to the root of an operator tree
    // starting with those that have the most specific partitioning options
    while let Some(window_expr) = window_aggregates.pop() {
        let mut window_expr_output_cols = vec![];
        let mut num_windows = 0;

        for (col_id, expr) in column_ids.iter().zip(new_exprs.iter()) {
            match expr.expr() {
                ScalarExpr::WindowAggregate { .. } => {
                    if num_windows <= window_idx {
                        window_expr_output_cols.push(*col_id);
                        num_windows += 1;
                    }
                }
                _ => window_expr_output_cols.push(*col_id),
            }
        }

        let window_aggregate = LogicalExpr::WindowAggregate(LogicalWindowAggregate {
            input,
            window_expr: builder.add_scalar_node(window_expr, &scope)?,
            columns: window_expr_output_cols,
        });
        builder.add_operator_and_scope(window_aggregate, scope);
        window_idx += 1;

        let (rel_node, rel_expr_scope) = builder.rel_node()?;
        input = rel_node;
        scope = rel_expr_scope;
    }

    let new_exprs: Result<Vec<_>, _> = new_exprs
        .into_iter()
        .zip(column_ids.iter())
        .map(|(expr, id)| match expr.expr() {
            ScalarExpr::WindowAggregate { .. } => builder.add_scalar_node(ScalarExpr::Column(*id), &scope),
            _ => Ok(expr),
        })
        .collect();

    Ok((input, scope, new_exprs?))
}

pub(super) fn collect_window_aggregates(exprs: &[ScalarNode]) -> Vec<ScalarExpr> {
    struct CollectWindowAggregates {
        output: Vec<ScalarExpr>,
    }
    impl ExprVisitor<RelNode> for CollectWindowAggregates {
        type Error = Infallible;

        fn pre_visit(&mut self, expr: &ScalarExpr) -> Result<bool, Self::Error> {
            if matches!(expr, ScalarExpr::WindowAggregate { .. }) {
                self.output.push(expr.clone());
                Ok(false)
            } else {
                Ok(true)
            }
        }

        fn post_visit(&mut self, _expr: &ScalarExpr) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    let mut out = vec![];
    for expr in exprs {
        let mut visitor = CollectWindowAggregates { output: vec![] };
        // CollectWindowAggregates never returns an error.
        expr.expr().accept(&mut visitor).unwrap();
        out.extend(visitor.output.into_iter());
    }
    out
}
