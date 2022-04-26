//! Joins-related code.

use itertools::Itertools;
use std::fmt::{Display, Formatter};

use crate::meta::ColumnId;
use crate::operators::relational::RelNode;
use crate::operators::scalar::{exprs, ScalarExpr, ScalarNode};

/// Type of a join.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

impl Display for JoinType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            JoinType::Inner => write!(f, "Inner"),
            JoinType::Left => write!(f, "Left"),
            JoinType::Right => write!(f, "Right"),
            JoinType::Full => write!(f, "Full"),
            JoinType::Cross => write!(f, "Cross"),
        }
    }
}

/// Join condition.
#[derive(Debug, Clone)]
pub enum JoinCondition {
    /// USING (column_list) condition.
    Using(JoinUsing),
    /// ON <expr> condition.
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

    /// Returns this condition in the form of an expression (eg. col:1 = col2:2 AND col:3=col:4).
    pub fn get_expr(&self) -> ScalarNode {
        let mut result_expr: Option<ScalarExpr> = None;
        for (l, r) in self.columns.iter() {
            let col_eq = ScalarExpr::Column(*l).eq(ScalarExpr::Column(*r));
            match result_expr.as_mut() {
                None => result_expr = Some(col_eq),
                Some(e) => {
                    *e = e.clone().and(col_eq);
                }
            }
        }
        ScalarNode::from(result_expr.unwrap())
    }
}

impl Display for JoinUsing {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.columns.iter().map(|(l, r)| format!("({}, {})", l, r)).join(", "))
    }
}

/// ON <expr> condition.
#[derive(Debug, Clone)]
pub struct JoinOn {
    /// The expression.
    pub(crate) expr: ScalarNode,
}

impl JoinOn {
    /// Creates a new condition from the given expression.
    pub fn new(expr: ScalarNode) -> Self {
        JoinOn { expr }
    }

    /// Returns the expression.
    pub fn expr(&self) -> &ScalarNode {
        &self.expr
    }
}

impl Display for JoinOn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expr.expr())
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
            let columns = exprs::collect_columns(on.expr.expr());

            if !columns.is_empty() {
                let left: Vec<ColumnId> = columns.iter().filter(|c| left_columns.contains(*c)).copied().collect();
                let right: Vec<ColumnId> = columns.iter().filter(|c| right_columns.contains(*c)).copied().collect();

                if left.is_empty() && right.is_empty() {
                    None
                } else {
                    Some((left, right))
                }
            } else {
                None
            }
        }
    }
}
