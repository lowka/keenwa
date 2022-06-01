use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};

/// Logical projection operator.
#[derive(Debug, Clone)]
pub struct LogicalProjection {
    /// The input operator.
    pub input: RelNode,
    /// The list of scalar expressions this projection consists of.
    pub exprs: Vec<ScalarNode>,
    /// The identifiers of projection items produced by this operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalProjection {
    pub(super) fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
        for expr in self.exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr);
        }
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "LogicalProjection");

        LogicalProjection {
            input: inputs.rel_node(),
            exprs: inputs.scalar_nodes(self.exprs.len()),
            columns: self.columns.clone(),
        }
    }

    pub(super) fn num_children(&self) -> usize {
        1 + self.exprs.len()
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        let num_exprs = self.exprs.len();
        match i {
            0 => Some(self.input.mexpr()),
            _ if i >= 1 && i < num_exprs + 1 => {
                let expr = &self.exprs[i - 1];
                Some(expr.mexpr())
            }
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalProjection");
        f.write_expr("input", &self.input);
        f.write_exprs("exprs", self.exprs.iter());
        f.write_values("cols", &self.columns);
    }
}