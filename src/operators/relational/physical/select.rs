use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;

/// Select operator returns only the rows that match its filter.
#[derive(Debug, Clone)]
pub struct Select {
    /// The input operator.
    pub input: RelNode,
    /// An optional filter expression.
    pub filter: Option<ScalarNode>,
}

impl Select {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.input)?;
        visitor.visit_opt_scalar(expr_ctx, self.filter.as_ref())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "Select");

        Select {
            input: inputs.rel_node(),
            filter: self.filter.as_ref().map(|_| inputs.scalar_node()),
        }
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    pub(super) fn num_children(&self) -> usize {
        1 + self.filter.as_ref().map(|_| 1).unwrap_or_default()
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.input.mexpr()),
            1 if self.filter.is_some() => {
                let filter = self.filter.as_ref().unwrap();
                Some(filter.mexpr())
            }
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Select");
        f.write_expr("input", &self.input);
        f.write_expr_if_present("filter", self.filter.as_ref())
    }
}
