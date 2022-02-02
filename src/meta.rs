use std::cell::{Ref, RefCell};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ops::Deref;

use crate::datatypes::DataType;
use crate::operators::scalar::ScalarExpr;

/// Uniquely identifies a column within a query.
pub type ColumnId = usize;

/// Stores a mapping between databases objects and their identifiers that are globally unique within a query.
#[derive(Debug, Clone)]
pub struct Metadata {
    columns: Vec<ColumnMetadata>,
}

/// Column metadata. If the table property is set then this is information about a column that belongs
/// to a database table. Otherwise this is a metadata of a synthetic column derived from an expression in a projection list.
#[derive(Debug, Clone)]
pub struct ColumnMetadata {
    id: ColumnId,
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
            id: 0,
            name,
            data_type,
            table: Some(table),
            expr: None,
        }
    }

    /// Creates column metadata for a synthetic column.
    pub fn new_synthetic_column(name: String, data_type: DataType, expr: Option<ScalarExpr>) -> Self {
        ColumnMetadata {
            id: 0,
            name,
            data_type,
            table: None,
            expr,
        }
    }

    /// Returns the identifier of this column.
    pub fn id(&self) -> &ColumnId {
        &self.id
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
#[derive(Debug, Clone)]
pub struct MutableMetadata {
    inner: RefCell<MutableMetadataInner>,
}

impl MutableMetadata {
    /// Creates a new instance of a MutableMetadata.
    pub fn new() -> Self {
        MutableMetadata {
            inner: RefCell::new(MutableMetadataInner::default()),
        }
    }

    /// Adds a new column to this metadata and returns its metadata identifier.
    /// When the given column belongs to a table this method first checks
    /// if such column already exists and if so returns its identifier.
    /// When the given column is synthetic this method always adds it as new column to this metadata.
    pub fn add_column(&self, mut column: ColumnMetadata) -> ColumnId {
        let mut inner = self.inner.borrow_mut();
        let id = inner.columns.len() + 1;
        column.id = id;

        if let Some(table) = column.table.as_ref() {
            let k = (column.name.clone(), table.clone());
            match inner.table_columns.entry(k) {
                Entry::Occupied(o) => {
                    return *o.get();
                }
                Entry::Vacant(v) => {
                    v.insert(id);
                }
            }
        }

        inner.columns.push(column);
        id
    }

    /// Returns column metadata for the given column metadata identifier.
    ///
    /// # Panics
    ///
    /// This method panics if there is no metadata for the given column.
    pub fn get_column(&self, column_id: &ColumnId) -> ColumnMetadataRef {
        let inner = self.inner.borrow();
        let r = Ref::map(inner, |inner| {
            inner
                .columns
                .get(column_id - 1)
                .unwrap_or_else(|| panic!("Unknown or unexpected column id: {:?}", column_id))
        });
        ColumnMetadataRef { inner: r }
    }

    /// Returns an iterator over available column metadata.
    pub fn get_columns(&self) -> Vec<ColumnMetadata> {
        let inner = self.inner.borrow();
        inner.columns.to_vec()
    }

    /// Returns a reference to a this metadata. A reference provides read-view into this metadata.
    pub fn get_ref(&self) -> MetadataRef<'_> {
        MetadataRef { metadata: self }
    }

    /// Converts this instance to a immutable [metadata](self::Metadata).
    pub fn to_metadata(self) -> Metadata {
        let inner = self.inner.borrow();
        Metadata::new(inner.columns.clone())
    }

    /// Creates an instances of a immutable [metadata](self::Metadata) from this metadata.
    pub fn build_metadata(&self) -> Metadata {
        let inner = self.inner.borrow();
        Metadata::new(inner.columns.clone())
    }
}

#[derive(Debug, Clone, Default)]
struct MutableMetadataInner {
    columns: Vec<ColumnMetadata>,
    table_columns: HashMap<(String, String), ColumnId>,
}

/// A reference to an instance of mutable metadata.
#[derive(Debug)]
pub struct MetadataRef<'a> {
    metadata: &'a MutableMetadata,
}

impl<'a> MetadataRef<'a> {
    /// Returns column metadata for the given column id.
    ///
    /// # Panics
    ///
    /// This method panics if there is no metadata for the given column.
    pub fn get_column(&self, column_id: &ColumnId) -> ColumnMetadataRef {
        self.metadata.get_column(column_id)
    }
}

/// A reference to a column metadata.
#[derive(Debug)]
pub struct ColumnMetadataRef<'a> {
    inner: Ref<'a, ColumnMetadata>,
}

impl Deref for ColumnMetadataRef<'_> {
    type Target = ColumnMetadata;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}
