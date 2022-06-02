use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::relational::RelNode;
use crate::operators::{Operator, OperatorCopyIn};

/// Logical offset operator skips the constant number of rows from its input.
#[derive(Debug, Clone)]
pub struct LogicalOffset {
    /// The input operator.
    pub input: RelNode,
    /// The number of rows to skip.
    pub rows: usize,
}

impl LogicalOffset {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.input)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "LogicalOffset");

        LogicalOffset {
            input: inputs.rel_node(),
            rows: self.rows,
        }
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
        f.write_name("LogicalOffset");
        f.write_expr("input", &self.input);
        f.write_value("rows", self.rows);
    }
}
