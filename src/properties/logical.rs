use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::expr::Expr;
use crate::operators::join::JoinCondition;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
use crate::operators::{OperatorExpr, RelExpr, RelNode, ScalarNode};
use crate::properties::statistics::{Statistics, StatisticsBuilder};
use std::fmt::Debug;

/// Properties that are identical across all expressions within a memo group.
#[derive(Debug, Clone)]
pub struct LogicalProperties {
    // FIXME: use a bit set instead of a vec.
    output_columns: Vec<ColumnId>,
    //FIXME: Make non-optional?
    statistics: Option<Statistics>,
}

impl LogicalProperties {
    /// Creates a new instance of `LogicalProperties` with the specified properties.
    pub fn new(output_columns: Vec<ColumnId>, statistics: Option<Statistics>) -> Self {
        LogicalProperties {
            output_columns,
            statistics,
        }
    }

    /// Creates an empty `LogicalProperties` object.
    ///
    /// Right now this method is used for physical expressions because physical expressions do not use logical properties
    /// of other expressions directly. They instead use logical properties of the memo groups of their child expressions.
    pub fn empty() -> Self {
        LogicalProperties {
            output_columns: Vec::with_capacity(0),
            statistics: None,
        }
    }

    /// Returns the columns produced by the expression.
    pub fn output_columns(&self) -> &[ColumnId] {
        &self.output_columns
    }

    /// Returns statistics for the expression.
    pub fn statistics(&self) -> Option<&Statistics> {
        self.statistics.as_ref()
    }
}

impl Default for LogicalProperties {
    fn default() -> Self {
        LogicalProperties::empty()
    }
}

/// Provides logical properties for memo expressions.
pub trait PropertiesProvider: Debug {
    fn build_properties(
        &self,
        expr: &OperatorExpr,
        statistics: Option<&Statistics>,
    ) -> Result<LogicalProperties, OptimizerError>;
}

#[derive(Debug)]
pub struct LogicalPropertiesBuilder {
    statistics: Box<dyn StatisticsBuilder>,
}

impl PropertiesProvider for LogicalPropertiesBuilder {
    fn build_properties(
        &self,
        expr: &OperatorExpr,
        statistics: Option<&Statistics>,
    ) -> Result<LogicalProperties, OptimizerError> {
        match expr {
            OperatorExpr::Relational(rel_expr) => match rel_expr {
                RelExpr::Logical(expr) => {
                    let statistics = self.statistics.build_statistics(expr, statistics)?;
                    Ok(self.build_for_logical(expr, statistics))
                }
                RelExpr::Physical(expr) => {
                    self.build_for_physical(expr);
                    // Properties are not used by physical expressions
                    Ok(LogicalProperties::empty())
                }
            },
            OperatorExpr::Scalar(_) => {
                let statistics = statistics.cloned().unwrap_or_default();
                let props = LogicalProperties::new(Vec::new(), Some(statistics));
                Ok(props)
            }
        }
    }
}

impl LogicalPropertiesBuilder {
    pub fn new(statistics: Box<dyn StatisticsBuilder>) -> Self {
        LogicalPropertiesBuilder { statistics }
    }

    fn build_for_logical(&self, expr: &LogicalExpr, statistics: Option<Statistics>) -> LogicalProperties {
        let columns = match expr {
            LogicalExpr::Projection { input, columns } => collect_columns_from_input(input, columns),
            LogicalExpr::Select { input, .. } => {
                let empty = Vec::with_capacity(0);
                collect_columns_from_input(input, &empty)
            }
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                let mut required = Vec::new();
                collect_columns_from_scalar_exprs(aggr_exprs, &mut required);
                // TODO: validate group exprs
                collect_columns_from_scalar_exprs(group_exprs, &mut required);
                collect_columns_from_input(input, &required)
            }
            LogicalExpr::Join { left, right, condition } => collect_columns_from_join_condition(left, right, condition),
            LogicalExpr::Get { columns, .. } => columns.clone(),
        };
        LogicalProperties::new(columns, statistics)
    }

    fn build_for_physical(&self, expr: &PhysicalExpr) -> LogicalProperties {
        match expr {
            PhysicalExpr::Projection { input, columns } => {
                collect_columns_from_input(input, columns);
            }
            PhysicalExpr::Select { input, .. } => {
                let empty = Vec::with_capacity(0);
                collect_columns_from_input(input, &empty);
            }
            PhysicalExpr::HashAggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                let mut required = Vec::new();
                // TODO: validate group exprs
                collect_columns_from_scalar_exprs(aggr_exprs, &mut required);
                collect_columns_from_scalar_exprs(group_exprs, &mut required);
                collect_columns_from_input(input, &required);
            }
            PhysicalExpr::HashJoin { left, right, condition } => {
                collect_columns_from_join_condition(left, right, condition);
            }
            PhysicalExpr::MergeSortJoin { left, right, condition } => {
                collect_columns_from_join_condition(left, right, condition);
            }
            PhysicalExpr::Scan { .. } => {}
            PhysicalExpr::IndexScan { .. } => {}
            PhysicalExpr::Sort { input, ordering } => {
                let required = ordering.columns();
                collect_columns_from_input(input, required);
            }
        };
        //FIXME:
        LogicalProperties::empty()
    }
}

fn collect_columns_from_input(input: &RelNode, used: &[ColumnId]) -> Vec<ColumnId> {
    let logical = input.props().logical();
    let output_columns = logical.output_columns();
    let columns: Vec<ColumnId> = output_columns.iter().copied().collect();

    assert!(
        used.iter().all(|c| columns.contains(c)),
        "Unexpected columns. Inputs {:?}. required: {:?}",
        columns,
        used
    );

    columns
}

fn collect_columns_from_join_condition(left: &RelNode, right: &RelNode, condition: &JoinCondition) -> Vec<ColumnId> {
    match condition {
        JoinCondition::Using(using) => {
            let mut left_columns = collect_columns_from_input(left, using.left_columns());
            let right_columns = collect_columns_from_input(right, using.right_columns());
            left_columns.extend(right_columns.into_iter());
            left_columns
        }
    }
}

fn collect_columns_from_exprs(exprs: &[Expr], columns: &mut Vec<ColumnId>) {
    for expr in exprs {
        collect_columns_from_expr(expr, columns);
    }
}

fn collect_columns_from_scalar_exprs(exprs: &[ScalarNode], columns: &mut Vec<ColumnId>) {
    for expr in exprs {
        collect_columns_from_expr(expr.expr(), columns);
    }
}

fn collect_columns_from_expr(expr: &Expr, columns: &mut Vec<ColumnId>) {
    match expr {
        Expr::Column(id) => columns.push(*id),
        Expr::Scalar(_) => {}
        Expr::BinaryExpr { lhs, rhs, .. } => {
            collect_columns_from_expr(lhs, columns);
            collect_columns_from_expr(rhs, columns);
        }
        Expr::Not(expr) => {
            collect_columns_from_expr(expr, columns);
        }
        Expr::Aggregate { args, .. } => {
            collect_columns_from_exprs(args, columns);
        }
        Expr::SubQuery(expr) => {
            collect_columns_from_input(expr, columns);
        }
    }
}
