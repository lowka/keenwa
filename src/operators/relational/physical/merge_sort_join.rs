use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::relational::join::{get_join_columns_pair, JoinCondition, JoinOn, JoinType};
use crate::operators::relational::RelNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;
use crate::properties::{derive_input_orderings, OrderingChoice};

/// MergeSortJoin operator implement a join by ordering its inputs.
#[derive(Debug, Clone)]
pub struct MergeSortJoin {
    /// The type of this join. MergeSortJoin can not be used to implement a [cross join][JoinType::Cross].
    pub join_type: JoinType,
    /// The left input operator.
    pub left: RelNode,
    /// The right input operator.
    pub right: RelNode,
    /// The join condition.
    pub condition: JoinCondition,
    /// The ordering in which data produced by the left operator is sorted.
    pub left_ordering: OrderingChoice,
    /// The ordering in which data produced by the right operator is sorted.
    pub right_ordering: OrderingChoice,
}

impl MergeSortJoin {
    pub(crate) fn derive_input_orderings(
        required_props: Option<&RequiredProperties>,
        left: &RelNode,
        right: &RelNode,
        condition: &JoinCondition,
    ) -> Option<(OrderingChoice, OrderingChoice)> {
        let (left, right) = match get_join_columns_pair(left, right, condition) {
            Some((left, right)) if !left.is_empty() && !right.is_empty() => (left, right),
            // MergeSortJoin can not be applied because the given join condition
            // can not be converted into [(left_col1 = right_col1), .., (left_col_n, right_col_n)].
            _ => return None,
        };

        match required_props
            .and_then(|r| r.ordering())
            .and_then(|r| derive_input_orderings(r, &left, &right))
        {
            Some(inputs) => Some(inputs),
            None => {
                // MergeSortJoin requires its inputs to be ordered ->
                // Add ordering to the columns used in the join condition.
                let left_ordering = OrderingChoice::from_columns(left);
                let right_ordering = OrderingChoice::from_columns(right);
                Some((left_ordering, right_ordering))
            }
        }
    }

    pub(super) fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
        visitor.visit_join_condition(expr_ctx, &self.condition);
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "MergeSortJoin");

        MergeSortJoin {
            join_type: self.join_type.clone(),
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            condition: match &self.condition {
                JoinCondition::Using(_) => self.condition.clone(),
                JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node())),
            },
            left_ordering: self.left_ordering.clone(),
            right_ordering: self.right_ordering.clone(),
        }
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        let left_ordering = RequiredProperties::new_with_ordering(self.left_ordering.clone());
        let right_ordering = RequiredProperties::new_with_ordering(self.right_ordering.clone());

        Some(vec![Some(left_ordering), Some(right_ordering)])
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
        f.write_name("MergeSortJoin");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("type", &self.join_type);
        match &self.condition {
            JoinCondition::Using(using) => f.write_value("using", using),
            JoinCondition::On(on) => f.write_value("on", on),
        }
        f.write_value("left_ord", &self.left_ordering);
        f.write_value("right_ord", &self.right_ordering);
    }
}
