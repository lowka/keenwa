use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;
use crate::properties::OrderingChoice;

//FIXME: Rename IndexScan
#[derive(Debug, Clone)]
pub struct IndexScan {
    /// The identifier of the source table.
    pub source: String,
    /// The output columns produced by this operator.
    pub columns: Vec<ColumnId>,
    /// An optional ordering. If present this operator allows access to rows in that ordering.
    pub ordering: Option<OrderingChoice>,
}

impl IndexScan {
    pub(super) fn copy_in<T>(
        &self,
        _visitor: &mut OperatorCopyIn<T>,
        _expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        Ok(())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "IndexScan");

        IndexScan {
            source: self.source.clone(),
            columns: self.columns.clone(),
            ordering: self.ordering.clone(),
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
        f.write_name("IndexScan");
        f.write_source(self.source.as_ref());
        f.write_values("cols", self.columns.as_slice())
    }
}
