use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;

/// Empty operator.
#[derive(Debug, Clone)]
pub struct Empty {
    /// Whether or not this operator produces a row.
    pub return_one_row: bool,
}

impl Empty {
    pub(super) fn copy_in<T>(
        &self,
        _visitor: &mut OperatorCopyIn<T>,
        _expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        Ok(())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "Empty");
        Empty {
            return_one_row: self.return_one_row,
        }
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
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
        f.write_name("Empty");
        f.write_value("return_one_row", &self.return_one_row)
    }
}
