use crate::meta::ColumnId;
use crate::operators::expr::{Expr, ExprVisitor};
use crate::operators::RelNode;
use std::fmt::{Display, Formatter};

//TODO:
// - Add join types
// - Add On condition (Expr)

/// Join condition.
#[derive(Debug, Clone)]
pub enum JoinCondition {
    /// USING (column_list) condition.
    Using(JoinUsing),
    /// ON expr condition.
    On(JoinOn),
}

impl JoinCondition {
    pub fn using(columns: Vec<(ColumnId, ColumnId)>) -> Self {
        JoinCondition::Using(JoinUsing::new(columns))
    }
}

/// USING (`column_list`) condition.
#[derive(Debug, Clone)]
pub struct JoinUsing {
    columns: Vec<(ColumnId, ColumnId)>,
}

impl JoinUsing {
    /// Creates a new join condition from the given collection of (left, right) column pairs.
    pub fn new(columns: Vec<(ColumnId, ColumnId)>) -> Self {
        assert!(!columns.is_empty(), "No columns have been specified");

        JoinUsing { columns }
    }

    /// Returns columns used by this condition.
    pub fn columns(&self) -> &[(ColumnId, ColumnId)] {
        &self.columns
    }

    /// Returns this condition as a pair of (left columns, right columns).
    pub fn get_columns_pair(&self) -> (Vec<ColumnId>, Vec<ColumnId>) {
        let left = self.columns.iter().map(|(l, _)| l).copied().collect();
        let right = self.columns.iter().map(|(_, r)| r).copied().collect();

        (left, right)
    }
}

impl Display for JoinUsing {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.columns)
    }
}

#[derive(Debug, Clone)]
pub struct JoinOn {
    expr: Expr,
}

impl JoinOn {
    pub fn new(expr: Expr) -> Self {
        JoinOn { expr }
    }

    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    pub fn get_columns(&self) -> Vec<ColumnId> {
        let mut columns = Vec::new();
        struct CollectColumns<'a> {
            columns: &'a mut Vec<ColumnId>,
        }
        impl ExprVisitor for CollectColumns<'_> {
            fn post_visit(&mut self, expr: &Expr) {
                if let Expr::Column(id) = expr {
                    self.columns.push(*id);
                }
            }
        }
        let mut visitor = CollectColumns { columns: &mut columns };
        self.expr.accept(&mut visitor);
        columns
    }
}

impl Display for JoinOn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expr)
    }
}

pub fn get_non_empty_join_columns_pair(
    left: &RelNode,
    right: &RelNode,
    condition: &JoinCondition,
) -> Option<(Vec<ColumnId>, Vec<ColumnId>)> {
    match get_join_columns_pair(left, right, condition) {
        Some((l, r)) if !l.is_empty() && !r.is_empty() => Some((l, r)),
        _ => None,
    }
}

pub fn get_join_columns_pair(
    left: &RelNode,
    right: &RelNode,
    condition: &JoinCondition,
) -> Option<(Vec<ColumnId>, Vec<ColumnId>)> {
    match condition {
        JoinCondition::Using(using) => Some(using.get_columns_pair()),
        JoinCondition::On(on) => {
            let left_columns = left.props().logical().output_columns();
            let right_columns = right.props().logical().output_columns();
            let columns = on.get_columns();

            if !columns.is_empty() {
                let left: Vec<ColumnId> = columns.iter().filter(|c| left_columns.contains(*c)).copied().collect();
                let right: Vec<ColumnId> = columns.iter().filter(|c| right_columns.contains(*c)).copied().collect();

                if left.is_empty() && right.is_empty() {
                    return None;
                } else {
                    Some((left, right))
                }
            } else {
                None
            }
        }
    }
}
