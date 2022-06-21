use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;

/// Values operator produces a specified collection of rows.
#[derive(Debug, Clone)]
pub struct Values {
    /// The list of [tuples](crate::operators::scalar::expr::Expr::Tuple) where each tuple is a row.
    pub values: Vec<ScalarNode>,
    /// The output columns produces by the operator.
    pub columns: Vec<ColumnId>,
}

impl Values {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        for expr in self.values.iter() {
            visitor.visit_scalar(expr_ctx, expr)?;
        }
        Ok(())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "Values")?;

        Ok(Values {
            values: inputs.scalar_nodes(self.values.len())?,
            columns: self.columns.clone(),
        })
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    pub(super) fn num_children(&self) -> usize {
        self.values.len()
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        self.values.get(i).map(|row| row.mexpr())
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Values");
        f.write_exprs("values", self.values.iter());
        f.write_values("cols", &self.columns);
    }
}
