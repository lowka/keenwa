use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;

/// Append operator implements UNION operator by combining rows produced by input operators.
#[derive(Debug, Clone)]
pub struct Append {
    /// The left input operator.
    pub left: RelNode,
    /// The right input operator.
    pub right: RelNode,
    /// The output columns produced by this operator.
    pub columns: Vec<ColumnId>,
}

impl Append {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.left)?;
        visitor.visit_rel(expr_ctx, &self.right)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "Append")?;

        Ok(Append {
            left: inputs.rel_node()?,
            right: inputs.rel_node()?,
            columns: self.columns.clone(),
        })
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    pub(super) fn num_children(&self) -> usize {
        2
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.left.mexpr()),
            1 => Some(self.right.mexpr()),
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Append");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_values("cols", self.columns.iter());
    }
}
