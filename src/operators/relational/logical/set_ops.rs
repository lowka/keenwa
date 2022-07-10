use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::{Operator, OperatorCopyIn};

/// Logical union operator. Union operator combines rows produced by its inputs.
#[derive(Debug, Clone)]
pub struct LogicalUnion {
    /// The left input operator.
    pub left: RelNode,
    /// The right input operator.
    pub right: RelNode,
    /// If `all` is `false` duplicate rows are eliminated.
    pub all: bool,
    /// The output columns produced by this operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalUnion {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.left)?;
        visitor.visit_rel(expr_ctx, &self.right)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "LogicalUnion")?;

        Ok(LogicalUnion {
            left: inputs.rel_node()?,
            right: inputs.rel_node()?,
            all: self.all,
            columns: self.columns.clone(),
        })
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
        f.write_name("LogicalUnion");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("all", self.all);
        f.write_values("cols", self.columns.iter());
    }
}

/// Logical intersect operator. Intersect operator produces rows that present in both of its inputs.
#[derive(Debug, Clone)]
pub struct LogicalIntersect {
    pub left: RelNode,
    pub right: RelNode,
    /// If `all` is `false` duplicate rows are eliminated.
    pub all: bool,
    /// The output columns produced by a set operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalIntersect {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.left)?;
        visitor.visit_rel(expr_ctx, &self.right)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "LogicalIntersect")?;

        Ok(LogicalIntersect {
            left: inputs.rel_node()?,
            right: inputs.rel_node()?,
            all: self.all,
            columns: self.columns.clone(),
        })
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
        f.write_name("LogicalIntersect");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("all", self.all);
        f.write_values("cols", self.columns.iter());
    }
}

/// Logical except operator. Except operator produces rows that present in its left input
/// but absent from its right input.
#[derive(Debug, Clone)]
pub struct LogicalExcept {
    /// The left input operator.
    pub left: RelNode,
    /// The right input operator.
    pub right: RelNode,
    /// If `all` is `false` duplicate rows are eliminated.
    pub all: bool,
    /// The output columns produced by a set operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalExcept {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.left)?;
        visitor.visit_rel(expr_ctx, &self.right)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "LogicalExcept")?;

        Ok(LogicalExcept {
            left: inputs.rel_node()?,
            right: inputs.rel_node()?,
            all: self.all,
            columns: self.columns.clone(),
        })
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
        f.write_name("LogicalExcept");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("all", self.all);
        f.write_values("cols", self.columns.iter());
    }
}
