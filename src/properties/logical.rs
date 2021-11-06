use crate::meta::ColumnId;
use crate::operators::expressions::Expr;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
use crate::operators::{InputExpr, OperatorExpr};
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
    /// Creates a new instance of `LogicalProperties` with the specified attributes.
    pub fn new(output_columns: Vec<ColumnId>, statistics: Option<Statistics>) -> Self {
        LogicalProperties {
            output_columns,
            statistics,
        }
    }

    /// Creates an empty `LogicalProperties` object.
    ///
    /// Right now this method is used for physical expressions because physical expressions do not use logical properties
    /// of other expressions directly. They instead use logical properties of the memo groups of their input expressions.
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
    fn build_properties(&self, expr: &OperatorExpr, statistics: Option<&Statistics>) -> LogicalProperties;
}

#[derive(Debug)]
pub struct LogicalPropertiesBuilder {
    statistics: Box<dyn StatisticsBuilder>,
}

impl PropertiesProvider for LogicalPropertiesBuilder {
    fn build_properties(&self, expr: &OperatorExpr, statistics: Option<&Statistics>) -> LogicalProperties {
        match expr {
            OperatorExpr::Logical(expr) => {
                let statistics = self.statistics.build_statistics(expr, statistics);
                self.build_for_logical(expr, statistics)
            }
            OperatorExpr::Physical(expr) => {
                self.build_for_physical(expr);
                // Attributes are not used by physical expressions
                LogicalProperties::empty()
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
            LogicalExpr::Projection { input, columns } => collect_columns_from_inputs(vec![input], columns),
            LogicalExpr::Select { input, .. } => {
                let empty = Vec::with_capacity(0);
                collect_columns_from_inputs(vec![input], &empty)
            }
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                let mut required = Vec::new();
                collect_columns_from_exprs(aggr_exprs, &mut required);
                // TODO: validate group exprs
                collect_columns_from_exprs(group_exprs, &mut required);
                collect_columns_from_inputs(vec![input], &required)
            }
            LogicalExpr::Join { left, right, condition } => {
                let required = condition.columns();
                collect_columns_from_inputs(vec![left, right], required)
            }
            LogicalExpr::Get { columns, .. } => columns.clone(),
            LogicalExpr::Expr { .. } => Vec::with_capacity(0),
        };
        LogicalProperties::new(columns, statistics)
    }

    fn build_for_physical(&self, expr: &PhysicalExpr) -> LogicalProperties {
        match expr {
            PhysicalExpr::Projection { input, columns } => {
                collect_columns_from_inputs(vec![input], columns);
            }
            PhysicalExpr::Select { input, .. } => {
                let empty = Vec::with_capacity(0);
                collect_columns_from_inputs(vec![input], &empty);
            }
            PhysicalExpr::HashAggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                let mut required = Vec::new();
                collect_columns_from_exprs(aggr_exprs, &mut required);
                // TODO: validate group exprs
                collect_columns_from_exprs(group_exprs, &mut required);
                collect_columns_from_inputs(vec![input], &required);
            }
            PhysicalExpr::HashJoin { left, right, condition } => {
                let required = condition.columns();
                collect_columns_from_inputs(vec![left, right], required);
            }
            PhysicalExpr::MergeSortJoin { left, right, condition } => {
                let required = condition.columns();
                collect_columns_from_inputs(vec![left, right], required);
            }
            PhysicalExpr::Scan { .. } => {}
            PhysicalExpr::IndexScan { .. } => {}
            PhysicalExpr::Sort { input, ordering } => {
                let required = ordering.columns();
                collect_columns_from_inputs(vec![input], required);
            }
            PhysicalExpr::Expr { .. } => {}
        };
        //FIXME:
        LogicalProperties::empty()
    }
}

fn collect_columns_from_inputs(inputs: Vec<&InputExpr>, used: &[ColumnId]) -> Vec<ColumnId> {
    let columns: Vec<ColumnId> = inputs
        .iter()
        .flat_map(|node| {
            let logical = node.attrs().logical();
            let output_columns = logical.output_columns();
            output_columns.iter().copied()
        })
        .collect();

    assert!(
        used.iter().all(|c| columns.contains(c)),
        "Unexpected columns. Inputs {:?}. required: {:?}",
        columns,
        used
    );

    columns
}

fn collect_columns_from_exprs(exprs: &[Expr], columns: &mut Vec<ColumnId>) {
    for expr in exprs {
        collect_columns_from_expr(expr, columns);
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
    }
}
