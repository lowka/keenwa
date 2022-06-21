use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::{Operator, OperatorCopyIn};

/// Logical operator that returns data from a source table.
#[derive(Debug, Clone)]
pub struct LogicalGet {
    /// The identifier of the source table.
    pub source: String,
    /// Output columns.
    pub columns: Vec<ColumnId>,
}

impl LogicalGet {
    pub(super) fn copy_in<T>(
        &self,
        _visitor: &mut OperatorCopyIn<T>,
        _expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        Ok(())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "LogicalGet")?;

        Ok(LogicalGet {
            source: self.source.clone(),
            columns: self.columns.clone(),
        })
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
        f.write_name("LogicalGet");
        f.write_source(&self.source);
        f.write_values("cols", &self.columns);
    }
}
