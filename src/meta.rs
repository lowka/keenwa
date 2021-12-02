use crate::catalog::{Column, ColumnRef};
use crate::operators::expr::Expr;
use std::collections::HashMap;

/// Uniquely identifies a column within a query.
pub type ColumnId = usize;

/// Stores a mapping between databases objects and their identifiers that are globally unique within a query.
#[derive(Debug, Clone)]
pub struct Metadata {
    columns: HashMap<ColumnId, ColumnMetadata>,
}

/// Column metadata stores a reference to a column of a particular database table or a reference to a synthetic
/// column produced by an operator. In later case the `expr` field stores an expression that used to compute values of the column.
#[derive(Debug, Clone)]
pub struct ColumnMetadata {
    /// Stores a reference to a column of some database table or attributes of a synthetic column.
    column: ColumnRef,
    /// In case of a synthetic column stores an expression that computes value for that column.
    expr: Option<Expr>,
}

impl ColumnMetadata {
    /// Creates a new instance of column metadata.
    pub fn new(column: ColumnRef, expr: Option<Expr>) -> Self {
        ColumnMetadata { column, expr }
    }

    /// Returns a column this metadata.
    pub fn column(&self) -> &Column {
        self.column.as_ref()
    }

    /// If this a synthetic column metadata return the expression used to compute values for that column.  
    pub fn expr(&self) -> Option<&Expr> {
        self.expr.as_ref()
    }
}

impl Metadata {
    pub fn new(columns: HashMap<ColumnId, ColumnMetadata>) -> Self {
        Metadata { columns }
    }

    /// Retrieves column metadata by the given column id.
    pub fn get_column(&self, column_id: &ColumnId) -> &ColumnMetadata {
        self.columns
            .get(column_id)
            .unwrap_or_else(|| panic!("Unknown or unexpected column id: {:?}", column_id))
    }

    pub fn columns(&self) -> impl Iterator<Item = (&ColumnId, &ColumnMetadata)> {
        self.columns.iter()
    }
}

pub struct MutableMetadata {
    columns: Vec<ColumnMetadata>,
}

impl MutableMetadata {
    pub fn new() -> Self {
        MutableMetadata { columns: Vec::new() }
    }

    pub fn add_column(&mut self, column: ColumnRef, expr: Option<Expr>) -> ColumnId {
        self.columns.push(ColumnMetadata::new(column, expr));
        self.columns.len()
    }

    pub fn get_column(&self, column_id: &ColumnId) -> &ColumnMetadata {
        self.columns
            .get(column_id - 1)
            .unwrap_or_else(|| panic!("Unknown or unexpected column id: {:?}", column_id))
    }

    pub fn get_metadata(self) -> Metadata {
        todo!()
    }
}
