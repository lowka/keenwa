//! Mutable implementation of a database catalog.

use std::any::Any;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use crate::catalog::{
    Catalog, Index, IndexRef, Schema, SchemaRef, Table, TableRef, __ensure_type_is_sync_send, DEFAULT_SCHEMA,
};
use crate::error::OptimizerError;

/// A [database catalog] that stores database objects in memory and provides operation to add/remove database objects.
///
/// # Error handling
///
/// Errors returned by methods of the `MutableCatalog` are recoverable.
///
/// [database catalog]: crate::catalog::Catalog
#[derive(Debug)]
pub struct MutableCatalog {
    schemas: RwLock<HashMap<ObjectId, SchemaRef>>,
}

impl MutableCatalog {
    /// Creates a instance of [MutableCatalog].
    pub fn new() -> Self {
        MutableCatalog {
            schemas: RwLock::new(HashMap::new()),
        }
    }

    /// Adds the given table to the specified schema.
    /// If such schema does not exists creates one.
    /// If the table already exists this method returns an error.
    pub fn add_table(&self, schema: &str, table: Table) -> Result<(), OptimizerError> {
        let mut schemas = self.schemas.write().unwrap();
        match schemas.entry(ObjectId::from(schema)) {
            Entry::Occupied(o) => {
                let schema = o.get();
                let schema = MutableSchema::from_ref(schema);
                schema.add_table(table)
            }
            Entry::Vacant(v) => {
                let schema = MutableSchema::new();
                let schema_ref = Arc::new(schema);
                v.insert(schema_ref.clone());
                let schema = schema_ref.as_ref();
                schema.add_table(table)
            }
        }
    }

    /// Adds the given index to the specified schema.
    /// If such schema does not exists creates one.
    /// If the index already exists this method returns an error.
    pub fn add_index(&self, schema: &str, index: Index) -> Result<(), OptimizerError> {
        let mut schemas = self.schemas.write().unwrap();
        match schemas.entry(ObjectId::from(schema)) {
            Entry::Occupied(o) => {
                let schema = o.get();
                let schema = MutableSchema::from_ref(schema);
                schema.add_index(index)
            }
            Entry::Vacant(v) => {
                let schema = MutableSchema::new();
                let schema_ref = Arc::new(schema);
                v.insert(schema_ref.clone());
                let schema = schema_ref.as_ref();
                schema.add_index(index)
            }
        }
    }

    /// Remove a database table with name `table` from the specified schema.
    /// If the schema or the table do not exist this method returns an error.  
    pub fn remove_table(&self, schema: &str, table: &str) -> Result<(), OptimizerError> {
        let mut schemas = self.schemas.write().unwrap();
        if let Some(schema) = schemas.get_mut(&ObjectId::from(schema)) {
            let schema = MutableSchema::from_ref(schema);
            schema.remove_table(table)
        } else {
            Err(OptimizerError::argument(format!("Schema does not exist. Schema: {}", schema)))
        }
    }

    /// Remove a database table with name `table` from the specified schema.
    /// If the schema or the index do not exist this method returns an error.  
    pub fn remove_index(&self, schema: &str, index: &str) -> Result<(), OptimizerError> {
        let mut schemas = self.schemas.write().unwrap();
        if let Some(schema) = schemas.get_mut(&ObjectId::from(schema)) {
            let schema = MutableSchema::from_ref(schema);
            schema.remove_index(index)
        } else {
            Err(OptimizerError::argument(format!("Schema does not exist. Schema: {}", schema)))
        }
    }
}

// see https://github.com/rust-lang/rust-clippy/issues/6066
#[allow(clippy::needless_collect)]
impl Catalog for MutableCatalog {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_schemas(&self) -> Vec<SchemaRef> {
        let schemas = self.schemas.read().unwrap();
        schemas.values().cloned().collect::<Vec<SchemaRef>>()
    }

    fn get_schema_by_name(&self, name: &str) -> Option<SchemaRef> {
        let schemas = self.schemas.read().unwrap();
        schemas.get(&ObjectId::from(name)).cloned()
    }

    fn get_table(&self, name: &str) -> Option<TableRef> {
        let schemas = self.schemas.read().unwrap();
        schemas.get(&ObjectId::from(DEFAULT_SCHEMA)).and_then(|s| s.get_table_by_name(name))
    }

    fn get_index(&self, name: &str) -> Option<IndexRef> {
        let schemas = self.schemas.read().unwrap();
        schemas.get(&ObjectId::from(DEFAULT_SCHEMA)).and_then(|s| s.get_index_by_name(name))
    }

    fn get_indexes(&self, table: &str) -> Vec<IndexRef> {
        let schemas = self.schemas.read().unwrap();
        let schema = schemas.get(&ObjectId::from(DEFAULT_SCHEMA));
        match schema {
            Some(schema) => schema.get_indexes(table),
            None => Vec::new(),
        }
    }
}

/// A [database schema](Schema) that stores object in memory and provides operation to add/remove database objects.
///
/// # Error handling
///
/// Errors returned by methods of the `MutableSchema` are recoverable.
#[derive(Debug)]
pub struct MutableSchema {
    inner: RwLock<Inner>,
}

impl MutableSchema {
    fn new() -> Self {
        MutableSchema {
            inner: RwLock::new(Inner::default()),
        }
    }

    fn from_ref(schema: &SchemaRef) -> &MutableSchema {
        schema
            .as_any()
            .downcast_ref::<MutableSchema>()
            .unwrap_or_else(|| panic!("Unable to downcast to MutableSchema: {:?}", schema))
    }

    /// Adds the given table to this schema. if a table with the same name already exists this method
    /// return an error.
    pub fn add_table(&self, table: Table) -> Result<(), OptimizerError> {
        let mut inner = self.inner.write().unwrap();
        let table_id = ObjectId::from(table.name.clone());
        match inner.tables.entry(table_id) {
            Entry::Occupied(_) => {
                Err(OptimizerError::argument(format!("Add table: Table already exists. Table: {}", table.name())))
            }
            Entry::Vacant(v) => {
                v.insert(Arc::new(table));
                Ok(())
            }
        }
    }

    /// Adds the given index to this schema. if an index with the same name already exists this method
    /// returns an error.
    pub fn add_index(&self, index: Index) -> Result<(), OptimizerError> {
        let mut inner = self.inner.write().unwrap();

        let table_id = ObjectId::from(index.table.clone());
        if inner.tables.get(&table_id).is_none() {
            let message = format!("Add index: table does not exist. Index: {}, table: {}", index.name(), index.table());
            return Err(OptimizerError::argument(message));
        }

        let index_id = ObjectId::from(index.name().to_string());
        match inner.indexes.entry(index_id) {
            Entry::Occupied(_) => {
                return Err(OptimizerError::argument(format!(
                    "Add index: Index already exists. Index: {}",
                    index.name()
                )))
            }
            Entry::Vacant(v) => {
                let index = Arc::new(index);
                v.insert(index.clone());
                match inner.table_indexes.entry(table_id) {
                    Entry::Occupied(mut o) => {
                        o.get_mut().push(index);
                    }
                    Entry::Vacant(v) => {
                        v.insert(vec![index]);
                    }
                };
            }
        };

        Ok(())
    }

    /// Remove a table with the give name. If the table does not exist this method returns an error.
    pub fn remove_table(&self, name: &str) -> Result<(), OptimizerError> {
        let mut inner = self.inner.write().unwrap();
        let table_name = ObjectId::from(name);

        if inner.tables.remove(&table_name).is_some() {
            inner.table_indexes.remove(&table_name);
            Ok(())
        } else {
            Err(OptimizerError::argument(format!("Remove table: Table does not exist. Table: {}", name)))
        }
    }

    /// Removes an index with the given name. if either index or table it belongs to does not exist
    /// this method returns an error.
    pub fn remove_index(&self, index: &str) -> Result<(), OptimizerError> {
        let mut inner = self.inner.write().unwrap();
        let index_name = ObjectId::from(index);

        if let Some(index) = inner.indexes.remove(&index_name) {
            let table_name = ObjectId::from(index.table.as_str());
            match inner.table_indexes.get_mut(&table_name) {
                Some(table_indexes) => {
                    table_indexes.retain(|i| i.name() != index.name());
                    Ok(())
                }
                None => Err(OptimizerError::argument("Remove index: Table does not exist")),
            }
        } else {
            Err(OptimizerError::argument("Remove index: Index does not exist"))
        }
    }
}

#[derive(Default, Debug)]
struct Inner {
    tables: HashMap<ObjectId, TableRef>,
    indexes: HashMap<ObjectId, IndexRef>,
    table_indexes: HashMap<ObjectId, Vec<IndexRef>>,
}

// see https://github.com/rust-lang/rust-clippy/issues/6066
#[allow(clippy::needless_collect)]
impl Schema for MutableSchema {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_tables(&self) -> Vec<TableRef> {
        let inner = self.inner.read().unwrap();
        inner.tables.values().cloned().collect::<Vec<TableRef>>()
    }

    fn get_table_by_name(&self, name: &str) -> Option<TableRef> {
        let inner = self.inner.read().unwrap();
        inner.tables.get(&ObjectId::from(name)).cloned()
    }

    fn get_indexes(&self, table: &str) -> Vec<IndexRef> {
        let inner = self.inner.read().unwrap();
        inner
            .table_indexes
            .get(&ObjectId::from(table))
            .cloned()
            .unwrap_or_else(|| Vec::with_capacity(0))
    }

    fn get_index_by_name(&self, name: &str) -> Option<IndexRef> {
        let inner = self.inner.read().unwrap();
        inner.indexes.get(&ObjectId::from(name)).cloned()
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct ObjectId {
    id: CaseInsensitiveString,
}

impl From<String> for ObjectId {
    fn from(id: String) -> Self {
        ObjectId {
            id: CaseInsensitiveString(id),
        }
    }
}

impl From<&str> for ObjectId {
    fn from(id: &str) -> Self {
        ObjectId {
            id: CaseInsensitiveString(String::from(id)),
        }
    }
}

#[derive(Debug)]
struct CaseInsensitiveString(String);

impl PartialEq for CaseInsensitiveString {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_ignore_ascii_case(&other.0)
    }
}

impl Eq for CaseInsensitiveString {}

impl Hash for CaseInsensitiveString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for c in self.0.as_bytes() {
            c.to_ascii_lowercase().hash(state)
        }
    }
}

#[allow(dead_code)]
fn __type_system_guarantees() {
    __ensure_type_is_sync_send::<MutableCatalog>();
    __ensure_type_is_sync_send::<MutableSchema>();
}

#[cfg(test)]
mod test {
    use crate::catalog::mutable::{MutableCatalog, MutableSchema};
    use crate::catalog::{Catalog, IndexBuilder, Schema, TableBuilder};
    use crate::datatypes::DataType;
    use crate::error::OptimizerError;

    #[test]
    fn test_tables() -> Result<(), OptimizerError> {
        let catalog = MutableCatalog::new();
        let table = TableBuilder::new("A").add_column("a1", DataType::Int32).build()?;

        catalog.add_table("s", table)?;

        let schema = catalog.get_schema_by_name("s").unwrap();
        let _ = schema
            .get_table_by_name("A")
            .unwrap_or_else(|| panic!("table A is missing from the schema s"));

        let schema: &MutableSchema = schema.as_any().downcast_ref::<MutableSchema>().unwrap();
        schema.remove_table("A")?;
        assert_eq!(schema.get_tables().len(), 0, "table has not been removed");
        Ok(())
    }

    #[test]
    fn test_indexes() -> Result<(), OptimizerError> {
        let catalog = MutableCatalog::new();
        let table1 = TableBuilder::new("A").add_column("a1", DataType::Int32).build()?;
        let table2 = TableBuilder::new("B").add_column("b1", DataType::Int32).build()?;

        catalog.add_table("s", table1)?;
        catalog.add_table("s", table2)?;

        let table1 = catalog.get_schema_by_name("s").unwrap().get_table_by_name("A").unwrap();
        let table2 = catalog.get_schema_by_name("s").unwrap().get_table_by_name("B").unwrap();

        let index1 = IndexBuilder::new(table1, "A_idx").add_column("a1").build()?;
        let index2 = IndexBuilder::new(table2, "B_idx").add_column("b1").build()?;

        catalog.add_index("s", index1)?;
        catalog.add_index("s", index2)?;

        let schema = catalog.get_schema_by_name("s").unwrap();
        let _ = schema
            .get_index_by_name("A_idx")
            .unwrap_or_else(|| panic!("index A_idx is missing from the schema s"));

        let schema: &MutableSchema = schema.as_any().downcast_ref::<MutableSchema>().unwrap();
        schema.remove_index("A_idx")?;

        assert_eq!(schema.get_indexes("A").len(), 0, "index has not been removed");
        assert_eq!(schema.get_indexes("B").len(), 1, "indexes from another table have been removed as well");

        schema.remove_table("B")?;
        assert_eq!(schema.get_indexes("B").len(), 0, "Table B has been removed but indexes are still available");

        Ok(())
    }
}
