use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;

/// HashAggregate implements an aggregate operator using a hashtable.
#[derive(Debug, Clone)]
pub struct HashAggregate {
    /// The input operator.
    pub input: RelNode,
    /// The list of aggregate expressions. The projection list of this aggregate operator.
    pub aggr_exprs: Vec<ScalarNode>,
    /// The list of grouping expressions (`GROUP BY` clause).
    pub group_exprs: Vec<ScalarNode>,
    /// An optional filter expression (`HAVING` clause).
    pub having: Option<ScalarNode>,
    /// The output columns produced by this operator.
    pub columns: Vec<ColumnId>,
}

impl HashAggregate {
    pub(super) fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
        for expr in self.aggr_exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr);
        }
        for expr in self.group_exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr);
        }
        visitor.visit_opt_scalar(expr_ctx, self.having.as_ref());
    }

    pub(super) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "HashAggregate");

        HashAggregate {
            input: inputs.rel_node(),
            aggr_exprs: inputs.scalar_nodes(self.aggr_exprs.len()),
            group_exprs: inputs.scalar_nodes(self.group_exprs.len()),
            having: self.having.as_ref().map(|_| inputs.scalar_node()),
            columns: self.columns.clone(),
        }
    }

    pub(super) fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
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
        f.write_name("HashAggregate");
        f.write_expr("input", &self.input);
        f.write_exprs("aggr_exprs", self.aggr_exprs.iter());
        f.write_exprs("group_exprs", self.group_exprs.iter());
        f.write_expr_if_present("having", self.having.as_ref());
        f.write_values("cols", &self.columns);
    }
}
