use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;

/// HashSetOp implements INTERSECT/EXCEPT/INTERSECT ALL/EXCEPT ALL operators using a hashtable.
#[derive(Debug, Clone)]
pub struct HashedSetOp {
    /// The left input operator.
    pub left: RelNode,
    /// The right input operator.
    pub right: RelNode,
    /// Whether this is an INTERSECT or EXCEPT operator.
    pub intersect: bool,
    /// If `true` this an INTERSECT ALL/EXCEPT ALL operator.
    pub all: bool,
    /// The output columns produced by a set operator.
    pub columns: Vec<ColumnId>,
}

impl HashedSetOp {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.left)?;
        visitor.visit_rel(expr_ctx, &self.right)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "HashedSetOp")?;

        Ok(HashedSetOp {
            left: inputs.rel_node()?,
            right: inputs.rel_node()?,
            intersect: self.intersect,
            all: self.all,
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
        f.write_name("HashedSetOp");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("intersect", &self.intersect);
        f.write_value("all", self.all);
        f.write_values("cols", &self.columns)
    }
}
