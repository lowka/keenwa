use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};

/// Logical distinct operator. Distinct operator consumes the input rows and
/// produces only those which are distinct/unique.
#[derive(Debug, Clone)]
pub struct LogicalDistinct {
    /// The input operator.
    pub input: RelNode,
    /// An optional expression that determines whether a rows is distinct or not.
    /// If absent a row is distinct iff there is no other row that have the same values of its columns.
    pub on_expr: Option<ScalarNode>,
    /// The columns produced by this operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalDistinct {
    pub(super) fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
        visitor.visit_opt_scalar(expr_ctx, self.on_expr.as_ref());
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "LogicalDistinct");

        LogicalDistinct {
            input: inputs.rel_node(),
            on_expr: self.on_expr.as_ref().map(|_| inputs.scalar_node()),
            columns: self.columns.clone(),
        }
    }

    pub(super) fn num_children(&self) -> usize {
        1 + self.on_expr.as_ref().map(|_| 1).unwrap_or_default()
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.input.mexpr()),
            1 => self.on_expr.as_ref().map(|expr| expr.mexpr()),
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalDistinct");
        f.write_expr("input", &self.input);
        f.write_expr_if_present("on", self.on_expr.as_ref());
        f.write_values("cols", &self.columns);
    }
}
