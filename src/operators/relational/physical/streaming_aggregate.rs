use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::scalar::{ScalarExpr, ScalarNode};
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;
use crate::properties::{OrderingChoice, OrderingColumn};

/// StreamingAggregate operator performs aggregations on a sorted sequence of data.
#[derive(Debug, Clone)]
pub struct StreamingAggregate {
    /// The input operator.
    pub input: RelNode,
    /// The list of aggregate expressions.
    pub aggr_exprs: Vec<ScalarNode>,
    /// The list of grouping expressions (`GROUP BY` clause).
    pub group_exprs: Vec<ScalarNode>,
    /// An optional filter expression (`HAVING` clause).
    pub having: Option<ScalarNode>,
    /// The output columns.
    pub columns: Vec<ColumnId>,
    /// The ordering in which data produced by the input operator is sorted.
    pub ordering: OrderingChoice,
}

impl StreamingAggregate {
    pub(crate) fn derive_input_ordering(group_exprs: &[ScalarNode]) -> Option<OrderingChoice> {
        let grouping_columns: Vec<_> = group_exprs
            .iter()
            .filter_map(|e| match e.expr() {
                ScalarExpr::Column(id) => Some(OrderingColumn::asc(*id)),
                _ => None,
            })
            .collect();

        if grouping_columns.len() < group_exprs.len() {
            None
        } else {
            Some(OrderingChoice::new(grouping_columns))
        }
    }

    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        visitor.visit_rel(expr_ctx, &self.input)?;
        for expr in self.aggr_exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr)?;
        }
        for expr in self.group_exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr)?;
        }
        visitor.visit_opt_scalar(expr_ctx, self.having.as_ref())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "StreamingAggregate")?;

        Ok(StreamingAggregate {
            input: inputs.rel_node()?,
            aggr_exprs: inputs.scalar_nodes(self.aggr_exprs.len())?,
            group_exprs: inputs.scalar_nodes(self.group_exprs.len())?,
            having: inputs.scalar_opt_node(self.having.as_ref())?,
            columns: self.columns.clone(),
            ordering: self.ordering.clone(),
        })
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        let input_ordering = RequiredProperties::new_with_ordering(self.ordering.clone());
        let iter = std::iter::once(Some(input_ordering));
        let input_properties = iter.chain(std::iter::repeat(None).take(self.num_children() - 1)).collect();
        Some(input_properties)
    }

    pub(super) fn num_children(&self) -> usize {
        1 + self.aggr_exprs.len() + self.group_exprs.len() + self.having.as_ref().map(|_| 1).unwrap_or_default()
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        let num_aggr_exprs = self.aggr_exprs.len();
        let num_group_exprs = self.group_exprs.len();
        match i {
            0 => Some(self.input.mexpr()),
            _ if i >= 1 && i < 1 + num_aggr_exprs => {
                let expr = &self.aggr_exprs[i - 1];
                Some(expr.mexpr())
            }
            _ if i >= num_aggr_exprs && i < 1 + num_aggr_exprs + num_group_exprs => {
                let expr = &self.group_exprs[i - num_aggr_exprs - 1];
                Some(expr.mexpr())
            }
            _ if i > num_aggr_exprs + num_group_exprs => self.having.as_ref().map(|e| e.mexpr()),
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("StreamingAggregate");
        f.write_expr("input", &self.input);
        f.write_exprs("aggr_exprs", self.aggr_exprs.iter());
        f.write_exprs("group_exprs", self.group_exprs.iter());
        f.write_expr_if_present("having", self.having.as_ref());
        f.write_values("cols", self.columns.iter());
        f.write_value("ordering", &self.ordering);
    }
}

#[cfg(test)]
mod test {
    use crate::meta::testing::TestMetadata;
    use crate::operators::relational::physical::StreamingAggregate;
    use crate::operators::scalar::value::Scalar;
    use crate::operators::scalar::{ScalarExpr, ScalarNode};
    use crate::properties::testing::ordering_from_string;

    #[test]
    fn derive_ordering_from_group_by_exprs() {
        let mut metadata = TestMetadata::with_tables(vec!["A"]);
        metadata.add_columns("A", vec!["a1", "a2"]);

        let ordering = ordering_from_string(&metadata, &["A:+a1", "A:+a2"]);
        let group_by_a1 = ScalarNode::from(ScalarExpr::Column(metadata.find_column("A", "a1")));
        let group_by_a2 = ScalarNode::from(ScalarExpr::Column(metadata.find_column("A", "a2")));
        let group_by = vec![group_by_a1, group_by_a2];

        let actual_orderings = StreamingAggregate::derive_input_ordering(&group_by);
        assert_eq!(actual_orderings, Some(ordering));
    }

    #[test]
    fn derive_ordering_from_group_by_exprs_fails_when_group_expr_are_non_column_exprs() {
        let mut metadata = TestMetadata::with_tables(vec!["A"]);
        metadata.add_columns("A", vec!["a1", "a2"]);

        let a1 = ScalarExpr::Column(metadata.find_column("A", "a1"));
        let int = ScalarExpr::Scalar(1.get_value());

        let group_by_a1 = ScalarNode::from(a1.clone());
        let group_by_a1_plus_int = ScalarNode::from(a1 + int);
        let group_by = vec![group_by_a1, group_by_a1_plus_int];

        let actual_orderings = StreamingAggregate::derive_input_ordering(&group_by);
        assert_eq!(actual_orderings, None);
    }
}
