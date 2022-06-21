use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};

/// Logical aggregate operator. Aggregate operator consumes input row and produces aggregate results.
#[derive(Debug, Clone)]
pub struct LogicalAggregate {
    /// The input operator.
    pub input: RelNode,
    /// The list of aggregate expressions. The projection list of this aggregate operator.
    pub aggr_exprs: Vec<ScalarNode>,
    /// The list of grouping expressions (`GROUP BY` clause).
    pub group_exprs: Vec<ScalarNode>,
    /// An optional filter expression (`HAVING` clause).
    pub having: Option<ScalarNode>,
    /// The output columns produced by this operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalAggregate {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.input)?;
        for expr in self.aggr_exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr)?;
        }
        for expr in self.group_exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr)?;
        }
        visitor.visit_opt_scalar(expr_ctx, self.having.as_ref())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "LogicalAggregate")?;

        Ok(LogicalAggregate {
            input: inputs.rel_node()?,
            aggr_exprs: inputs.scalar_nodes(self.aggr_exprs.len())?,
            group_exprs: inputs.scalar_nodes(self.group_exprs.len())?,
            having: inputs.scalar_opt_node(self.having.as_ref())?,
            columns: self.columns.clone(),
        })
    }

    pub(super) fn num_children(&self) -> usize {
        1 + self.aggr_exprs.len() + self.group_exprs.len() + self.having.as_ref().map(|_| 1).unwrap_or_default()
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        let num_aggr_exprs = self.aggr_exprs.len();
        let num_group_exprs = self.group_exprs.len();

        match i {
            0 => Some(self.input.mexpr()),
            _ if i >= 1 && i < num_aggr_exprs + 1 => {
                let expr = &self.aggr_exprs[i - 1];
                Some(expr.mexpr())
            }
            _ if i > num_aggr_exprs && i < 1 + num_aggr_exprs + num_group_exprs => {
                let expr = &self.group_exprs[i - 1 - num_aggr_exprs];
                Some(expr.mexpr())
            }
            _ if i > num_aggr_exprs + num_group_exprs => self.having.as_ref().map(|e| e.mexpr()),
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalAggregate");
        f.write_expr("input", &self.input);
        f.write_exprs("aggr_exprs", self.aggr_exprs.iter());
        f.write_exprs("group_exprs", self.group_exprs.iter());
        f.write_expr_if_present("having", self.having.as_ref());
        f.write_values("cols", &self.columns);
    }
}

/// Logical window aggregate operator. Window aggregate computes an aggregate over a range of values.
#[derive(Debug, Clone)]
pub struct LogicalWindowAggregate {
    /// The input operator.
    pub input: RelNode,
    /// The window function expression.
    pub window_expr: ScalarNode,
    /// The output columns produced by this operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalWindowAggregate {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.input)?;
        visitor.visit_scalar(expr_ctx, &self.window_expr)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "LogicalWindowAggregate")?;

        Ok(LogicalWindowAggregate {
            input: inputs.rel_node()?,
            window_expr: inputs.scalar_node()?,
            columns: self.columns.clone(),
        })
    }

    pub(super) fn num_children(&self) -> usize {
        2
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.input.mexpr()),
            1 => Some(self.window_expr.mexpr()),
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalWindowAggregate");
        f.write_expr("input", &self.input);
        f.write_expr("window_expr", &self.window_expr);
        f.write_values("cols", &self.columns);
    }
}
