//! Properties of an operator.
//!
//! Logical properties are shared by all expressions in a memo-group.
//! Physical properties are describe physical characteristics of the data (such as ordering).
//! These properties are required by some operators. For example MergeSortJoin requires its inputs to be ordered.

use std::fmt::{Display, Formatter};

use crate::meta::ColumnId;

pub mod logical;
pub mod physical;

/// Ordering. Describes how columns are sorted.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct OrderingChoice {
    columns: Vec<ColumnId>,
}

impl OrderingChoice {
    pub fn new(columns: Vec<ColumnId>) -> Self {
        assert!(!columns.is_empty(), "columns are not specified");
        OrderingChoice { columns }
    }

    pub fn columns(&self) -> &[ColumnId] {
        &self.columns
    }
}

impl Display for OrderingChoice {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ columns={:?} }}", self.columns)
    }
}
