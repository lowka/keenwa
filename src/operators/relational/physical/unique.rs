use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;
use crate::properties::{derive_input_orderings, OrderingChoice};

/// Unique operator is implemented by sorting its inputs and producing only unique rows.
#[derive(Debug, Clone)]
pub struct Unique {
    /// The inputs operators.
    pub inputs: Vec<RelNode>,
    /// An optional expression that determines whether a rows is distinct or not.
    /// If absent a row is distinct iff there is no other row that have the same values of its columns.
    pub on_expr: Option<ScalarNode>,
    /// The output columns produces by this operator.
    pub columns: Vec<ColumnId>,
    /// The ordering in which data produced by the `i-th` input operator is sorted.
    pub ordering: Vec<OrderingChoice>,
}

impl Unique {
    pub(crate) fn derive_input_orderings(
        ordering: Option<&OrderingChoice>,
        inputs: &[RelNode],
        columns: &[ColumnId],
    ) -> Vec<OrderingChoice> {
        let ordering = match ordering {
            Some(ordering) => {
                // Unique operator is implemented by sorting its inputs.
                derive_input_orderings(ordering, columns, columns).map(|(l, _)| l)
            }
            None => None,
        };
        inputs
            .iter()
            .map(|input| {
                let output_columns = input.props().logical().output_columns();
                match &ordering {
                    Some(ordering) => ordering.with_mapping(columns, output_columns),
                    None => {
                        // Unique operator requires its inputs to be ordered.
                        OrderingChoice::from_columns(output_columns.to_vec())
                    }
                }
            })
            .collect()
    }

    pub(super) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        for input in self.inputs.iter() {
            visitor.visit_rel(expr_ctx, input)?;
        }
        visitor.visit_opt_scalar(expr_ctx, self.on_expr.as_ref())
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Result<Self, OptimizerError> {
        inputs.expect_len(self.num_children(), "Unique")?;

        Ok(Unique {
            inputs: inputs.rel_nodes(self.inputs.len())?,
            on_expr: inputs.scalar_opt_node(self.on_expr.as_ref())?,
            columns: self.columns.clone(),
            ordering: self.ordering.clone(),
        })
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        let requirements = self.ordering.iter().map(|ord| Some(RequiredProperties::new_with_ordering(ord.clone())));
        if self.on_expr.is_some() {
            Some(requirements.chain(std::iter::once(None)).collect())
        } else {
            Some(requirements.collect())
        }
    }

    pub(super) fn num_children(&self) -> usize {
        self.inputs.len() + self.on_expr.as_ref().map(|_| 1).unwrap_or_default()
    }

    pub(super) fn get_child(&self, i: usize) -> Option<&Operator> {
        let num_input = self.inputs.len();
        match num_input {
            _ if i < num_input => self.inputs.get(i).map(|input| input.mexpr()),
            _ if i >= num_input && i < self.num_children() => self.on_expr.as_ref().map(|expr| expr.mexpr()),
            _ => None,
        }
    }

    pub(super) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Unique");
        let num_input = self.inputs.len();
        match num_input {
            1 => f.write_expr("input", &self.inputs[0]),
            2 => {
                f.write_expr("left", &self.inputs[0]);
                f.write_expr("right", &self.inputs[1]);
            }
            _ => f.write_exprs("inputs", self.inputs.iter()),
        }
        f.write_expr_if_present("on", self.on_expr.as_ref());
        f.write_values("cols", self.columns.iter());
        match num_input {
            1 => f.write_value("ord", &self.ordering[0]),
            2 => {
                f.write_value("left_ord", &self.ordering[0]);
                f.write_value("right_ord", &self.ordering[1]);
            }
            _ => f.write_values("ord", self.ordering.iter()),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::meta::testing::TestMetadata;
    use crate::meta::ColumnId;
    use crate::operators::relational::logical::{LogicalEmpty, LogicalExpr};
    use crate::operators::relational::physical::Unique;
    use crate::operators::relational::{RelExpr, RelNode};
    use crate::operators::RelationalProperties;
    use crate::properties::testing::{ordering_from_string, ordering_to_string};
    use crate::properties::OrderingChoice;

    #[test]
    fn unique_derive_inputs() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B", "C", "U"]);
        let a_cols = metadata.add_columns("A", vec!["a1", "a2"]);
        let b_cols = metadata.add_columns("B", vec!["b1", "b2"]);
        let c_cols = metadata.add_columns("C", vec!["c1", "c2"]);
        let u_cols = metadata.add_columns("U", vec!["u1", "u2"]);

        let ordering = ordering_from_string(&metadata, &["U:+u1", "U:-u2"]);
        let result = unique_derive_input_orderings(Some(&ordering), vec![&a_cols, &b_cols, &c_cols], &u_cols);
        let orderings: Vec<_> = result.into_iter().map(|o| ordering_to_string(&metadata, &o)).collect();

        assert_eq!(
            orderings,
            vec![String::from("A:+a1, A:-a2"), String::from("B:+b1, B:-b2"), String::from("C:+c1, C:-c2")]
        )
    }

    #[test]
    fn unique_derive_inputs_extends_ordering() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B", "C", "U"]);
        let a_cols = metadata.add_columns("A", vec!["a1", "a2"]);
        let b_cols = metadata.add_columns("B", vec!["b1", "b2"]);
        let c_cols = metadata.add_columns("C", vec!["c1", "c2"]);
        let u_cols = metadata.add_columns("U", vec!["u1", "u2"]);

        let ordering = ordering_from_string(&metadata, &["U:+u1"]);
        let result = unique_derive_input_orderings(Some(&ordering), vec![&a_cols, &b_cols, &c_cols], &u_cols);
        let orderings: Vec<_> = result.into_iter().map(|o| ordering_to_string(&metadata, &o)).collect();

        assert_eq!(
            orderings,
            vec![String::from("A:+a1, A:+a2"), String::from("B:+b1, B:+b2"), String::from("C:+c1, C:+c2")]
        )
    }

    #[test]
    fn unique_derive_inputs_add_ordering_when_no_ordering_is_provided() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B", "C", "U"]);
        let a_cols = metadata.add_columns("A", vec!["a1", "a2"]);
        let b_cols = metadata.add_columns("B", vec!["b1", "b2"]);
        let c_cols = metadata.add_columns("C", vec!["c1", "c2"]);
        let u_cols = metadata.add_columns("U", vec!["u1", "u2"]);

        let result = unique_derive_input_orderings(None, vec![&a_cols, &b_cols, &c_cols], &u_cols);
        let orderings: Vec<_> = result.into_iter().map(|o| ordering_to_string(&metadata, &o)).collect();

        assert_eq!(
            orderings,
            vec![String::from("A:+a1, A:+a2"), String::from("B:+b1, B:+b2"), String::from("C:+c1, C:+c2")]
        )
    }

    fn unique_derive_input_orderings(
        ordering: Option<&OrderingChoice>,
        inputs: Vec<&[ColumnId]>,
        output_columns: &[ColumnId],
    ) -> Vec<OrderingChoice> {
        let rel_nodes: Vec<_> = inputs
            .into_iter()
            .map(|cols| {
                let dummy = LogicalExpr::Empty(LogicalEmpty { return_one_row: false });
                let mut props = RelationalProperties::default();
                props.logical.output_columns = cols.to_vec();

                RelNode::new(RelExpr::Logical(Box::new(dummy)), props)
            })
            .collect();

        Unique::derive_input_orderings(ordering, &rel_nodes, &output_columns)
    }
}
