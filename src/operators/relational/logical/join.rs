use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::relational::join::{JoinCondition, JoinOn, JoinType};
use crate::operators::relational::RelNode;
use crate::operators::{Operator, OperatorCopyIn};

/// Logical join operator.
#[derive(Debug, Clone)]
pub struct LogicalJoin {
    /// The type of this join operator (e.g. Inner, Left, etc.)
    pub join_type: JoinType,
    /// The left side of the join operator.
    pub left: RelNode,
    /// The right side of the join operator.
    pub right: RelNode,
    /// The join condition.
    pub condition: JoinCondition,
}

impl LogicalJoin {
    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.left)?;
        visitor.visit_rel(expr_ctx, &self.right)?;
        visitor.visit_join_condition(expr_ctx, &self.condition)
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "LogicalJoin")?;

        Ok(LogicalJoin {
            join_type: self.join_type.clone(),
            left: inputs.rel_node()?,
            right: inputs.rel_node()?,
            condition: match &self.condition {
                JoinCondition::Using(_) => self.condition.clone(),
                JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node()?)),
            },
        })
    }

    pub(super) fn num_children(&self) -> usize {
        let num_opt = match &self.condition {
            JoinCondition::Using(_) => 0,
            JoinCondition::On(_) => 1,
        };
        2 + num_opt
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.left.mexpr()),
            1 => Some(self.right.mexpr()),
            2 => match &self.condition {
                JoinCondition::Using(_) => None,
                JoinCondition::On(on) => Some(on.expr.mexpr()),
            },
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalJoin");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("type", &self.join_type);
        match &self.condition {
            JoinCondition::Using(using) => f.write_value("using", using),
            JoinCondition::On(on) => f.write_value("on", on),
        }
    }
}
