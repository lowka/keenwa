use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::{Operator, OperatorCopyIn};

/// Logical operator that produces no results.
///
/// This operator allows to implement input operators for queries such as `SELECT 1`
/// or to implement subtrees of logical plans that can be eliminated.
#[derive(Debug, Clone)]
pub struct LogicalEmpty {
    /// Whether this operator must return one row or not.
    pub return_one_row: bool,
}

impl LogicalEmpty {
    pub(super) fn copy_in<T>(
        &self,
        _visitor: &mut OperatorCopyIn<T>,
        _expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        Ok(())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "LogicalEmpty");

        LogicalEmpty {
            return_one_row: self.return_one_row,
        }
    }

    pub(super) fn num_children(&self) -> usize {
        0
    }

    pub(super) fn get_child(&self, _i: usize) -> Option<&Operator> {
        None
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalEmpty");
        f.write_value("return_one_row", &self.return_one_row)
    }
}
