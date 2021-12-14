use std::fmt::{Display, Formatter};

use crate::meta::ColumnId;

pub mod logical;
pub mod physical;

/// Ordering.
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
