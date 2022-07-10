use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;

/// Scan operator returns data from a source table.
#[derive(Debug, Clone)]
pub struct Scan {
    /// The identifier of the source table.
    pub source: String,
    /// The columns produced by this operator.
    pub columns: Vec<ColumnId>,
}

impl Scan {
    pub(super) fn copy_in<T>(
        &self,
        _visitor: &mut OperatorCopyIn<T>,
        _expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        Ok(())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "Scan")?;

        Ok(Scan {
            source: self.source.clone(),
            columns: self.columns.clone(),
        })
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
        f.write_name("Scan");
        f.write_source(self.source.as_ref());
        f.write_values("cols", self.columns.iter())
    }
}
