use crate::datatypes::DataType;
use crate::operators::scalar::ScalarExpr;

/// Uniquely identifies a column within a query.
pub type ColumnId = usize;

/// Stores a mapping between databases objects and their identifiers that are globally unique within a query.
#[derive(Debug, Clone)]
pub struct Metadata {
    columns: Vec<ColumnMetadata>,
}

/// Column metadata stores information about some column. If the table property is set then this is information about
/// a column in that database table. Otherwise this is a metadata of a synthetic column derived
/// from a column in a projection list.
#[derive(Debug, Clone)]
pub struct ColumnMetadata {
    /// The name of this column.
    name: String,
    /// The type of this column.
    data_type: DataType,
    /// If present stores the name of the table this column belongs to.
    table: Option<String>,
    /// If present stores a copy of the expression this column is derived from.
    //FIXME: Currently it is only used in tests.
    expr: Option<ScalarExpr>,
}

impl ColumnMetadata {
    /// Creates column metadata for a column that belongs to the given table.
    pub fn new_table_column(name: String, data_type: DataType, table: String) -> Self {
        ColumnMetadata {
            name,
            data_type,
            table: Some(table),
            expr: None,
        }
    }

    /// Creates column metadata for a synthetic column.
    pub fn new_synthetic_column(name: String, data_type: DataType, expr: Option<ScalarExpr>) -> Self {
        ColumnMetadata {
            name,
            data_type,
            table: None,
            expr,
        }
    }

    /// Returns the name of this column.
    pub fn name(&self) -> &String {
        &self.name
    }

    /// Returns the type of this column.
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Returns the table this column belongs to. If the table is absent then this a synthetic column.
    pub fn table(&self) -> Option<&String> {
        self.table.as_ref()
    }

    /// Returns a copy of the expression this column is derived from.
    pub fn expr(&self) -> Option<&ScalarExpr> {
        self.expr.as_ref()
    }
}

impl Metadata {
    pub fn new(columns: Vec<ColumnMetadata>) -> Self {
        Metadata { columns }
    }

    /// Retrieves column metadata by the given column id.
    pub fn get_column(&self, column_id: &ColumnId) -> &ColumnMetadata {
        self.columns
            .get(column_id - 1)
            .unwrap_or_else(|| panic!("Unknown or unexpected column id: {:?}", column_id))
    }

    /// Returns an iterator over available column metadata.
    pub fn columns(&self) -> impl Iterator<Item = (ColumnId, &ColumnMetadata)> {
        self.columns.iter().enumerate().map(|(i, c)| (i + 1, c))
    }
}

/// A mutable variant of a [Metadata](self::Metadata).
pub struct MutableMetadata {
    columns: Vec<ColumnMetadata>,
}

impl MutableMetadata {
    /// Creates a new instance of a MutableMetadata.
    pub fn new() -> Self {
        MutableMetadata { columns: Vec::new() }
    }

    /// Adds a new column to this metadata.
    pub fn add_column(&mut self, column: ColumnMetadata) -> ColumnId {
        self.columns.push(column);
        self.columns.len()
    }

    /// Returns column metadata for the given column id.
    ///
    /// # Panics
    ///
    /// This method panics if there is no metadata for the given column.
    pub fn get_column(&self, column_id: &ColumnId) -> &ColumnMetadata {
        self.columns
            .get(column_id - 1)
            .unwrap_or_else(|| panic!("Unknown or unexpected column id: {:?}", column_id))
    }

    /// Converts this instance to a immutable [metadata](self::Metadata).
    pub fn to_metadata(self) -> Metadata {
        Metadata::new(self.columns)
    }
}
