use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::relational::join::JoinType;
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;

/// NestedLoop operator implements a join by iterating over all rows produced by the right operator
/// for each row produced by the left operator and emitting rows that match the join condition.
#[derive(Debug, Clone)]
pub struct NestedLoopJoin {
    /// The type of this join.
    pub join_type: JoinType,
    /// The left input operator.
    pub left: RelNode,
    /// The right input operator.
    pub right: RelNode,
    /// The join condition. Can be an arbitrary scalar expression.
    pub condition: Option<ScalarNode>,
}

impl NestedLoopJoin {
    pub(super) fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
        visitor.visit_opt_scalar(expr_ctx, self.condition.as_ref());
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "NestedLoopJoin");

        NestedLoopJoin {
            join_type: self.join_type.clone(),
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            condition: self.condition.as_ref().map(|_| inputs.scalar_node()),
        }
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    pub(super) fn num_children(&self) -> usize {
        2 + self.condition.as_ref().map(|_| 1).unwrap_or_default()
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.left.mexpr()),
            1 => Some(self.right.mexpr()),
            2 if self.condition.is_some() => {
                let condition = self.condition.as_ref().unwrap();
                Some(condition.mexpr())
            }
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("NestedLoopJoin");
        f.write_value("type", &self.join_type);
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        if let Some(condition) = self.condition.as_ref() {
            f.write_expr("condition", condition);
        }
    }
}
