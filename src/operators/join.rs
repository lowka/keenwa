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
