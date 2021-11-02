use crate::catalog::ColumnRef;
use std::collections::HashMap;

/// Uniquely identifies a column within a query.
pub type ColumnId = usize;

/// Stores a mapping between databases objects and their identifiers that are globally unique within a query.
#[derive(Debug, Clone)]
pub struct Metadata {
    columns: HashMap<ColumnId, ColumnRef>,
}

impl Metadata {
    pub fn new(columns: HashMap<ColumnId, ColumnRef>) -> Self {
        Metadata { columns }
    }

    /// Retrieves a column by the given column id.
    pub fn get_column(&self, column_id: &ColumnId) -> &ColumnRef {
        self.columns
            .get(column_id)
            .unwrap_or_else(|| panic!("Unknown or unexpected column id: {:?}", column_id))
    }
}
