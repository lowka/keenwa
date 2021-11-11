use crate::meta::ColumnId;
use std::fmt::{Display, Formatter};

//TODO:
// - Add join types
// - Add On condition (Expr)

/// Join condition.
#[derive(Debug, Clone)]
pub enum JoinCondition {
    /// USING (column_list) condition.
    Using(JoinUsing),
}

impl JoinCondition {
    pub fn using(columns: Vec<(ColumnId, ColumnId)>) -> Self {
        JoinCondition::Using(JoinUsing::new(columns))
    }
}

/// USING (`column_list`) condition.
#[derive(Debug, Clone)]
pub struct JoinUsing {
    left: Vec<ColumnId>,
    right: Vec<ColumnId>,
}

impl JoinUsing {
    /// Creates a new join condition from the given collection of (left, right) column pairs.
    pub fn new(columns: Vec<(ColumnId, ColumnId)>) -> Self {
        assert!(!columns.is_empty(), "No columns have been specified");
        let mut left = Vec::with_capacity(columns.len());
        let mut right = Vec::with_capacity(columns.len());

        for (l, r) in columns {
            left.push(l);
            right.push(r);
        }

        JoinUsing { left, right }
    }

    /// The columns used by the left side of a join.
    pub fn left_columns(&self) -> &[ColumnId] {
        &self.left
    }

    /// The columns used by the right side of a join.
    pub fn right_columns(&self) -> &[ColumnId] {
        &self.right
    }

    /// Returns this condition as a pair of (left columns, right columns).
    pub fn as_columns_pair(&self) -> (Vec<ColumnId>, Vec<ColumnId>) {
        let left = self.left.iter().copied().collect();
        let right = self.right.iter().copied().collect();
        (left, right)
    }
}

impl Display for JoinUsing {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let left = self.left.iter();
        let right = self.right.iter();
        let columns: Vec<_> = left.zip(right).collect();
        write!(f, "{:?}", columns)
    }
}
