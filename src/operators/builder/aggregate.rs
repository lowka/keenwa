use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::builder::projection::{ProjectionList, ProjectionListBuilder};
use crate::operators::builder::{OperatorBuilder, RewriteExprs, ValidateFilterExpr, ValidateProjectionExpr};
use crate::operators::relational::logical::{LogicalAggregate, LogicalExpr};
use crate::operators::relational::RelNode;
use crate::operators::scalar::aggregates::AggregateFunction;
use crate::operators::scalar::expr::ExprVisitor;
use crate::operators::scalar::exprs::collect_columns;
use crate::operators::scalar::value::ScalarValue;
use crate::operators::scalar::{get_subquery, ScalarExpr, ScalarNode};
use crate::operators::{Operator, OperatorExpr};
use std::collections::HashSet;
use std::convert::TryFrom;

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
            .map_err(|_| OptimizerError::Argument(format!("Unknown aggregate function {}", func)))?;
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
    pub fn build(mut self) -> Result<OperatorBuilder, OptimizerError> {
        let (input, scope) = self.builder.rel_node()?;

        let aggr_exprs = std::mem::take(&mut self.aggr_exprs);
        let metadata = self.builder.metadata.clone();
        let mut projection_builder = ProjectionListBuilder::new(self.builder, &scope);
        let mut non_aggregate_columns = HashSet::new();

        for expr in aggr_exprs {
            match expr {
                AggrExpr::Function { func, distinct, column } => {
                    let mut rewriter =
                        RewriteExprs::new(&scope, metadata.clone(), ValidateProjectionExpr::projection_expr());
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

        let ProjectionList {
            column_ids,
            output_columns,
            projection: aggr_exprs,
        } = projection_builder.build();

        let mut group_exprs = vec![];
        let mut group_by_columns = HashSet::new();

        for expr in std::mem::take(&mut self.group_exprs) {
            let expr = if let ScalarExpr::Scalar(ScalarValue::Int32(pos)) = expr {
                let pos = pos - 1;
                if pos >= 0 && (pos as usize) < aggr_exprs.len() {
                    aggr_exprs[pos as usize].expr().clone()
                } else {
                    // query error
                    return Err(OptimizerError::Internal(format!("GROUP BY: position {} is not in select list", pos)));
                }
            } else {
                expr
            };

            // AggregateBuilder::disallow_nested_subqueries(&expr, "GROUP BY", "expressions")?;

            let expr = match expr {
                ScalarExpr::Column(_) | ScalarExpr::ColumnName(_) => {
                    let mut rewriter =
                        RewriteExprs::new(&scope, metadata.clone(), ValidateProjectionExpr::projection_expr());
                    let expr = expr.rewrite(&mut rewriter)?;
                    let expr = self.builder.add_scalar_node(expr, &scope);
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
                    return Err(OptimizerError::Internal("GROUP BY: not integer constant argument".to_string()));
                }
                ScalarExpr::Aggregate { .. } => {
                    // query error
                    return Err(OptimizerError::Internal("GROUP BY: Aggregate functions are not allowed".to_string()));
                }
                _ => {
                    // query error
                    return Err(OptimizerError::Internal(format!("GROUP BY: unsupported expression: {}", expr)));
                }
            };
            group_exprs.push(expr);
        }

        if !group_by_columns.is_empty() {
            for col_id in non_aggregate_columns.iter() {
                if !group_by_columns.contains(col_id) {
                    // query error. Add table/table alias
                    let column = self.builder.metadata.get_column(col_id);
                    return Err(OptimizerError::Internal(format!(
                        "AGGREGATE column {} must appear in GROUP BY clause",
                        column.name()
                    )));
                }
            }
        }

        let having_expr = if let Some(expr) = self.having.take() {
            disallow_nested_subqueries(&expr, "AGGREGATE", "HAVING clause")?;

            let mut rewriter = RewriteExprs::new(&scope, metadata, ValidateFilterExpr::where_clause());
            let expr = expr.rewrite(&mut rewriter)?;
            let mut validate = ValidateHavingClause {
                projection_columns: &non_aggregate_columns,
                group_by_columns: &group_by_columns,
            };

            expr.accept(&mut validate)?;

            struct ValidateHavingClause<'a> {
                projection_columns: &'a HashSet<ColumnId>,
                group_by_columns: &'a HashSet<ColumnId>,
            }

            impl<'a> ExprVisitor<RelNode> for ValidateHavingClause<'a> {
                type Error = OptimizerError;

                fn pre_visit(&mut self, expr: &ScalarExpr) -> Result<bool, Self::Error> {
                    match expr {
                        ScalarExpr::Aggregate { .. } => Ok(false),
                        ScalarExpr::Column(id) => {
                            if self.projection_columns.contains(id) || self.group_by_columns.contains(id) {
                                Ok(false)
                            } else {
                                Err(OptimizerError::Internal(format!(
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

            Some(self.builder.add_scalar_node(expr, &scope))
        } else {
            None
        };

        let aggregate = LogicalExpr::Aggregate(LogicalAggregate {
            input,
            aggr_exprs,
            group_exprs,
            columns: column_ids,
            having: having_expr,
        });
        let operator = Operator::from(OperatorExpr::from(aggregate));

        Ok(OperatorBuilder::from_builder(self.builder, operator, &scope, output_columns))
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
                    return Err(OptimizerError::Internal(format!(
                        "{}: Wildcard expressions are not allowed",
                        self.clause
                    )))
                }
                _ if get_subquery(expr).is_some() => {
                    return Err(OptimizerError::NotImplemented(format!(
                        "{}: Subqueries in {} are not implemented",
                        self.clause, self.location
                    )));
                }
                _ => {}
            }
            Ok(())
        }
    }
    let mut visitor = DisallowNestedSubqueries { clause, location };
    expr.accept(&mut visitor)
}
