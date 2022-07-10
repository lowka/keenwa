use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::relational::RelNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;
use crate::properties::OrderingChoice;

/// Sort operator orders the rows produced by its input operator.
///
/// This operator is used to apply ordering in cases when an operator does not produce ordered rows
/// or the ordering of those rows does not match the requirements.
#[derive(Debug, Clone)]
pub struct Sort {
    /// The input operator.
    pub input: RelNode,
    /// The ordering.
    pub ordering: OrderingChoice,
}

impl Sort {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.input)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "Sort")?;

        Ok(Sort {
            input: inputs.rel_node()?,
            ordering: self.ordering.clone(),
        })
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    pub(super) fn num_children(&self) -> usize {
        1
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        if i == 0 {
            Some(self.input.mexpr())
        } else {
            None
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Sort");
        f.write_expr("input", &self.input);
        f.write_values("ord", self.ordering.columns().iter())
    }
}
