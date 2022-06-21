use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::relational::RelNode;
use crate::operators::{Operator, OperatorCopyIn};

/// Logical limit operator. Limit operator reduces the number of rows produced
/// by its input operator to the number that is no greater than the specified constant value.
#[derive(Debug, Clone)]
pub struct LogicalLimit {
    /// The input operator.
    pub input: RelNode,
    /// The maximum number of rows to return.
    pub rows: usize,
}

impl LogicalLimit {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.input)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "LogicalLimit")?;

        Ok(LogicalLimit {
            input: inputs.rel_node()?,
            rows: self.rows,
        })
    }

    pub(super) fn num_children(&self) -> usize {
        1
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.input.mexpr()),
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalLimit");
        f.write_expr("input", &self.input);
        f.write_value("rows", self.rows);
    }
}
