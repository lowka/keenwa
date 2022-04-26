//! Query metadata (columns, relations, etc.)

use std::cell::{Ref, RefCell};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Deref;

use crate::datatypes::DataType;
use crate::operators::scalar::ScalarExpr;

/// Uniquely identifies a column within a query.
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash, Ord, PartialOrd)]
pub struct ColumnId(pub(crate) usize);

impl Display for ColumnId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Uniquely identifies a relation with a query.
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash, Ord, PartialOrd)]
pub struct RelationId(usize);

impl Display for RelationId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stores a mapping between databases objects and their identifiers that are globally unique within a query.
#[derive(Debug, Clone)]
pub struct Metadata {
    relations: Vec<RelationMetadata>,
    columns: Vec<ColumnMetadata>,
}

/// Relation metadata.
#[derive(Debug, Clone)]
pub struct RelationMetadata {
    /// The identifier.
    id: RelationId,
    /// The name of this relation.
    //TODO: Make optional
    name: String,
    /// The columns
    columns: Vec<ColumnId>,
}

impl RelationMetadata {
    fn new(name: String, columns: Vec<ColumnId>) -> Self {
        RelationMetadata {
            id: RelationId(0),
            name,
            columns,
        }
    }

    /// Returns the identifier of this relation.
    pub fn id(&self) -> &RelationId {
        &self.id
    }

    /// Returns the name of this relation.
    pub fn name(&self) -> &String {
        &self.name
    }

    /// Returns the columns of this relation.
    pub fn columns(&self) -> &[ColumnId] {
        &self.columns
    }
}

/// A reference to a relation metadata.
#[derive(Debug)]
pub struct RelationMetadataRef<'a> {
    inner: Ref<'a, RelationMetadata>,
}

impl Deref for RelationMetadataRef<'_> {
    type Target = RelationMetadata;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
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
    /// If presents store the identifier of the relation this column belongs to.
    relation_id: Option<RelationId>,
    /// If present stores the name of the table this column belongs to.
    //TODO: Redundant use relation_id instead.
    table: Option<String>,
    /// If present stores a copy of the expression this column is derived from.
    //FIXME: Currently it is only used in tests.
    expr: Option<ScalarExpr>,
}

const COLUMN_ID_NOT_SET: ColumnId = ColumnId(0);

impl ColumnMetadata {
    /// Creates column metadata for a column that belongs to the given table.
    pub fn new_table_column(name: String, data_type: DataType, relation_id: RelationId, table: String) -> Self {
        ColumnMetadata {
            id: COLUMN_ID_NOT_SET,
            name,
            data_type,
            relation_id: Some(relation_id),
            table: Some(table),
            expr: None,
        }
    }

    /// Creates column metadata for a synthetic column.
    pub fn new_synthetic_column(name: String, data_type: DataType, expr: Option<ScalarExpr>) -> Self {
        ColumnMetadata {
            id: COLUMN_ID_NOT_SET,
            name,
            data_type,
            relation_id: None,
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
    #[deprecated]
    pub fn table(&self) -> Option<&String> {
        self.table.as_ref()
    }

    /// Returns the identifier of the relation column belongs to. If the relation identifier
    /// is absent then this a synthetic column.
    pub fn relation_id(&self) -> Option<RelationId> {
        self.relation_id
    }

    /// Returns a copy of the expression this column is derived from.
    pub fn expr(&self) -> Option<&ScalarExpr> {
        self.expr.as_ref()
    }
}

impl Metadata {
    pub fn new(columns: Vec<ColumnMetadata>, relations: Vec<RelationMetadata>) -> Self {
        Metadata { columns, relations }
    }

    /// Retrieves relation metadata by the given relation id.
    pub fn get_relation(&self, relation_id: &RelationId) -> &RelationMetadata {
        self.relations
            .get(relation_id.0 - 1)
            .unwrap_or_else(|| panic!("Unknown or unexpected relation id: {}", relation_id))
    }

    /// Retrieves column metadata by the given column id.
    pub fn get_column(&self, column_id: &ColumnId) -> &ColumnMetadata {
        self.columns
            .get(column_id.0 - 1)
            .unwrap_or_else(|| panic!("Unknown or unexpected column id: {:?}", column_id))
    }

    /// Returns an iterator over available column metadata.
    pub fn get_columns(&self) -> &[ColumnMetadata] {
        &self.columns
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

    /// Adds a table with the given name and returns relation id assigned to it.
    /// This method does not check for duplicats and assigns new relation id for every table.
    pub fn add_table(&self, name: &str) -> RelationId {
        let mut inner = self.inner.borrow_mut();
        let id = RelationId(inner.relations.len() + 1);
        let mut relation = RelationMetadata::new(name.into(), vec![]);
        relation.id = id;
        inner.relations.push(relation);
        id
    }

    /// Adds a new column to this metadata and returns its metadata identifier.
    /// When the given column belongs to a table this method first checks
    /// if such column already exists and if so returns its identifier.
    /// When the given column is synthetic this method always adds it as new column to this metadata.
    pub fn add_column(&self, mut column: ColumnMetadata) -> ColumnId {
        let mut inner = self.inner.borrow_mut();
        let id = ColumnId(inner.columns.len() + 1);
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
        if let Some(relation_id) = column.relation_id {
            let relation = &mut inner.relations[relation_id.0 - 1];
            relation.columns.push(id);
        }

        inner.columns.push(column);
        id
    }

    /// Renames the column with the given identifier.
    ///
    /// # Panics
    ///
    /// This method panics if there is no metadata for the column with the given identifier.
    pub fn rename_column(&self, column_id: ColumnId, name: String) {
        let mut inner = self.inner.borrow_mut();
        let column = inner
            .columns
            .get_mut(column_id.0 - 1)
            .unwrap_or_else(|| panic!("Unknown or unexpected column id: {}", column_id));
        column.name = name;
    }

    /// Returns metadata for the relation with the given identifier.
    ///
    /// # Panics
    ///
    /// This method panics if relolation with the given identifier does not exist.
    pub fn get_relation(&self, relation_id: &RelationId) -> RelationMetadataRef {
        let inner = self.inner.borrow();
        let r = Ref::map(inner, |inner| {
            inner
                .relations
                .get(relation_id.0 - 1)
                .unwrap_or_else(|| panic!("Unknown or unexpected relation id: {}", relation_id))
        });
        RelationMetadataRef { inner: r }
    }

    /// Returns column metadata for the given column metadata identifier.
    ///
    /// # Panics
    ///
    /// This method panics if there is no metadata for the column with the given identifier.
    pub fn get_column(&self, column_id: &ColumnId) -> ColumnMetadataRef {
        let inner = self.inner.borrow();
        let r = Ref::map(inner, |inner| {
            inner
                .columns
                .get(column_id.0 - 1)
                .unwrap_or_else(|| panic!("Unknown or unexpected column id: {}", column_id))
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
        Metadata::new(inner.columns.clone(), inner.relations.clone())
    }

    /// Creates an instances of a immutable [metadata](self::Metadata) from this metadata.
    pub fn build_metadata(&self) -> Metadata {
        let inner = self.inner.borrow();
        Metadata::new(inner.columns.clone(), inner.relations.clone())
    }
}

#[derive(Debug, Clone, Default)]
struct MutableMetadataInner {
    relations: Vec<RelationMetadata>,
    columns: Vec<ColumnMetadata>,
    table_columns: HashMap<(String, String), ColumnId>,
}

/// A reference to an instance of mutable metadata.
#[derive(Debug, Clone)]
pub struct MetadataRef<'a> {
    metadata: &'a MutableMetadata,
}

impl<'a> MetadataRef<'a> {
    /// Returns relation metadata for the given relation id.
    ///
    /// # Panics
    ///
    /// This method panics if there is no metadata for the given relation.
    pub fn get_relation(&self, relation_id: &RelationId) -> RelationMetadataRef {
        self.metadata.get_relation(relation_id)
    }

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

#[cfg(test)]
pub mod testing {
    use crate::datatypes::DataType;
    use crate::meta::{ColumnId, ColumnMetadata, ColumnMetadataRef, MutableMetadata, RelationId};
    use crate::operators::scalar::ScalarExpr;
    use std::collections::HashMap;

    /// TestMetadata provides methods to build columns and relations used by tests.
    pub struct TestMetadata {
        inner: MutableMetadata,
        tables: HashMap<String, RelationId>,
    }

    impl TestMetadata {
        /// Creates an instance of TestMetadata.
        pub fn new() -> Self {
            TestMetadata {
                inner: MutableMetadata::new(),
                tables: HashMap::new(),
            }
        }

        /// Creates an instances of TestMetadata and pre-populates it with the given tables.
        pub fn with_tables(tables: Vec<impl AsRef<str>>) -> Self {
            let mut metadata = TestMetadata::new();
            for table in tables {
                metadata.add_table(table.as_ref())
            }
            metadata
        }

        /// Adds a table with the given name. See [MutableMetadata::add_table].
        pub fn add_table(&mut self, name: &str) {
            let relation_id = self.inner.add_table(name);
            self.tables.insert(name.into(), relation_id);
        }

        /// Returns a builder that creates [metadata for a table column](crate::meta::ColumnMetadata).
        ///
        /// # Panics
        ///
        /// This method panics if the given table does not exist.
        pub fn column(&mut self, table: &str) -> TestColumnMetadata {
            if let Some(relation_id) = self.tables.get(table).copied() {
                let num_cols = self.inner.get_relation(&relation_id).columns().len();
                let col_name = format!("{}{}", table.to_ascii_lowercase(), (num_cols + 1));

                TestColumnMetadata::new_table_column(self, col_name, relation_id, String::from(table))
            } else {
                panic!("Table {} does not exist. Existing: {:?}", table, self.tables.keys());
            }
        }

        /// Returns a builder that creates [metadata for a synthetic column](crate::meta::ColumnMetadata).
        pub fn synthetic_column(&mut self) -> TestColumnMetadata {
            let col_name = "?";

            TestColumnMetadata::new_synthetic_column(self, String::from(col_name))
        }

        /// Returns metadata id of the given column.
        ///
        /// # Panics
        ///
        /// This method panics if the given table or column does not exist.
        pub fn find_column(&self, table: &str, name: &str) -> ColumnId {
            if let Some(relation_id) = self.tables.get(table) {
                let table = self.inner.get_relation(relation_id);
                let col_id = table.columns().iter().find(|id| self.inner.get_column(id).name() == name).copied();
                col_id.unwrap_or_else(|| panic!("No column {} in table {}", name, table.name))
            } else {
                panic!("Table {} does not exist. Existing: {:?}", table, self.tables.keys());
            }
        }

        /// Returns metadata for the column with the given identifier.
        ///
        /// # Panics
        ///
        /// This method panics if there is no metadata for the column with the given identifier.
        pub fn get_column(&self, col_id: &ColumnId) -> ColumnMetadataRef {
            self.inner.get_column(col_id)
        }
    }

    /// A builder for [column metadata](ColumnMetadata).
    pub struct TestColumnMetadata<'a> {
        metadata: &'a mut TestMetadata,
        name: String,

        relation: Option<(String, RelationId)>,
        data_type: Option<DataType>,
        expr: Option<ScalarExpr>,
    }

    impl<'a> TestColumnMetadata<'a> {
        fn new_table_column(
            metadata: &'a mut TestMetadata,
            name: String,
            relation_id: RelationId,
            relation: String,
        ) -> Self {
            TestColumnMetadata {
                metadata,
                name,
                relation: Some((relation, relation_id)),
                data_type: None,
                expr: None,
            }
        }

        fn new_synthetic_column(metadata: &'a mut TestMetadata, name: String) -> Self {
            TestColumnMetadata {
                metadata,
                name,
                relation: None,
                data_type: None,
                expr: None,
            }
        }

        /// Specifies the name for the column.
        pub fn named(mut self, name: &str) -> TestColumnMetadata<'a> {
            self.name = String::from(name);
            self
        }

        /// Specifies the data type of the column.
        pub fn data_type(mut self, data_type: DataType) -> TestColumnMetadata<'a> {
            self.data_type = Some(data_type);
            self
        }

        /// Constructs a column metadata.
        pub fn build(mut self) -> ColumnId {
            let data_type = self.data_type.take().unwrap_or(DataType::Int32);
            let column = match self.relation.clone() {
                Some((relation, relation_id)) => {
                    ColumnMetadata::new_table_column(self.name, data_type, relation_id, relation)
                }
                None => ColumnMetadata::new_synthetic_column(self.name, data_type, self.expr),
            };
            self.metadata.inner.add_column(column)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::datatypes::DataType;
    use crate::meta::{ColumnId, ColumnMetadata, MutableMetadata, RelationId};

    #[test]
    fn test_add_table() {
        let metadata = MutableMetadata::new();
        let relation_id = metadata.add_table("A");
        assert_eq!(RelationId(1), relation_id, "the first relation id");
    }

    #[test]
    fn test_assign_new_relation_id_for_every_table() {
        let metadata = MutableMetadata::new();
        let relation_id1 = metadata.add_table("A");
        let relation_id2 = metadata.add_table("A");
        let relation_id3 = metadata.add_table("B");

        assert_ne!(relation_id1, relation_id2, "every relation must have the different id");
        assert_ne!(relation_id2, relation_id3, "every relation must have the different id");
    }

    #[test]
    fn test_columns_add_column() {
        let metadata = MutableMetadata::new();
        let relation_id = metadata.add_table("A");
        let col_a = ColumnMetadata::new_table_column("a1".into(), DataType::Int32, relation_id, "A".into());
        let col_id = metadata.add_column(col_a);

        assert_eq!(ColumnId(1), col_id, "the first column id");
    }

    #[test]
    fn test_columns_add_column_reuse_id() {
        let metadata = MutableMetadata::new();
        let relation_id = metadata.add_table("A");

        let col_a = ColumnMetadata::new_table_column("a1".into(), DataType::Int32, relation_id, "A".into());
        let col_b = ColumnMetadata::new_table_column("a1".into(), DataType::Int32, relation_id, "A".into());
        let col_a_copy = col_a.clone();

        let col_a_id = metadata.add_column(col_a);
        let col_b_id = metadata.add_column(col_b);
        let col_a_id2 = metadata.add_column(col_a_copy);

        assert_eq!(col_a_id, col_a_id2, "column ids must be the same");
        assert_eq!(col_a_id, col_b_id, "different columns must have different column ids");
    }
}
