use crate::meta::ColumnId;
use std::collections::HashSet;

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
    pub fn new(columns: Vec<ColumnId>) -> Self {
        JoinCondition::Using(JoinUsing::new(columns))
    }

    pub fn using(left_columns: Vec<ColumnId>, right_columns: Vec<ColumnId>) -> Self {
        assert_eq!(left_columns.len(), right_columns.len(), "left.len != right.len");
        let mut columns = Vec::with_capacity(left_columns.len() + right_columns.len());
        for (l, r) in left_columns.into_iter().zip(right_columns) {
            columns.push(l);
            columns.push(r);
        }
        JoinCondition::Using(JoinUsing::new(columns))
    }

    /// The columns used by this condition.
    pub fn columns(&self) -> &[ColumnId] {
        match self {
            JoinCondition::Using(using) => using.columns(),
        }
    }

    pub fn filter_columns(
        condition: &JoinCondition,
        left: &[ColumnId],
        right: &[ColumnId],
    ) -> (Vec<ColumnId>, Vec<ColumnId>) {
        fn filter_non_duplicates(condition: &JoinCondition, output: &[ColumnId]) -> Vec<ColumnId> {
            let mut tmp = HashSet::new();
            let mut out = Vec::new();
            for id in condition.columns() {
                if output.contains(id) && tmp.insert(*id) {
                    out.push(*id);
                }
            }
            out
        }
        let left_join_columns = filter_non_duplicates(condition, left);
        let right_join_columns = filter_non_duplicates(condition, right);

        assert!(!left_join_columns.is_empty(), "left join columns are empty");
        assert!(!right_join_columns.is_empty(), "right join columns are empty");

        (left_join_columns, right_join_columns)
    }
}

/// USING (`column_list`) condition where the `column_list` is written in the following form:
/// ```text:
/// left_col1, right_col1, left_col2, right_col2 and so on.
/// ```
#[derive(Debug, Clone)]
pub struct JoinUsing {
    columns: Vec<ColumnId>,
}

impl JoinUsing {
    pub fn new(columns: Vec<ColumnId>) -> Self {
        assert_eq!(
            columns.len() % 2,
            0,
            "Expected an even number of columns but got {:?}: {:?}",
            columns.len(),
            columns
        );

        JoinUsing { columns }
    }

    /// The columns used by this condition.
    pub fn columns(&self) -> &[ColumnId] {
        &self.columns
    }
}
