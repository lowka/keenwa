use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::relational::RelNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::partitioning::Partitioning;
use crate::properties::physical::RequiredProperties;

/// Repartitions the input stream of data into possible several output streams.
#[derive(Debug, Clone)]
pub struct Exchanger {
    pub input: RelNode,
    pub partitioning: Partitioning,
}

impl Exchanger {
    pub(crate) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.input)
    }

    pub(crate) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "Exchanger")?;

        Ok(Exchanger {
            input: inputs.rel_node()?,
            partitioning: self.partitioning.clone(),
        })
    }

    pub(crate) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    pub(crate) fn num_children(&self) -> usize {
        1
    }

    pub(crate) fn get_child(&self, i: usize) -> Option<&Operator> {
        if i == 0 {
            Some(self.input.mexpr())
        } else {
            None
        }
    }

    pub(crate) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Exchanger");
        f.write_expr("input", &self.input);
        match &self.partitioning {
            Partitioning::Singleton => {
                f.write_value("partitioning", "()");
            }
            Partitioning::Partitioned(columns) => {
                f.write_values("partitioning", columns.iter());
            }
            Partitioning::OrderedPartitioning(columns) => {
                f.write_values("ord-partitioning", columns.iter());
            }
            Partitioning::HashPartitioning(columns) => {
                f.write_values("hash-partitioning", columns.iter());
            }
        }
    }
}
