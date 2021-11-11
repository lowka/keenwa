use crate::catalog::{Catalog, Index, IndexRef, Schema, SchemaRef, Table, TableRef, DEFAULT_SCHEMA};
use std::any::Any;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A [database catalog] that stores database objects in memory and provides operation to add/remove database objects.
///
/// [database catalog]: crate::catalog::Catalog
#[derive(Debug)]
pub struct MutableCatalog {
    schemas: RwLock<HashMap<String, SchemaRef>>,
}

impl MutableCatalog {
    pub fn new() -> Self {
        MutableCatalog {
            schemas: RwLock::new(HashMap::new()),
        }
    }

    pub fn add_table(&self, schema: &str, table: Table) {
        let mut schemas = self.schemas.write().unwrap();
        match schemas.entry(schema.to_string()) {
            Entry::Occupied(o) => {
                let schema = o.get();
                let schema = MutableSchema::from_ref(schema);
                schema.add_table(table);
            }
            Entry::Vacant(v) => {
                let schema = MutableSchema::new();
                schema.add_table(table);
                v.insert(Arc::new(schema));
            }
        }
    }

    pub fn add_index(&self, schema: &str, index: Index) {
        let mut schemas = self.schemas.write().unwrap();
        match schemas.entry(schema.to_string()) {
            Entry::Occupied(o) => {
                let schema = o.get();
                let schema = MutableSchema::from_ref(schema);
                schema.add_index(index);
            }
            Entry::Vacant(v) => {
                let schema = MutableSchema::new();
                schema.add_index(index);
                v.insert(Arc::new(schema));
            }
        }
    }

    pub fn remove_table(&self, schema: &str, table: &str) {
        let mut schemas = self.schemas.write().unwrap();
        if let Some(schema) = schemas.get_mut(schema) {
            let schema = MutableSchema::from_ref(schema);
            schema.remove_table(table);
        }
    }

    pub fn remove_index(&self, schema: &str, index: &str) {
        let mut schemas = self.schemas.write().unwrap();
        if let Some(schema) = schemas.get_mut(schema) {
            let schema = MutableSchema::from_ref(schema);
            schema.remove_index(index);
        }
    }
}

// see https://github.com/rust-lang/rust-clippy/issues/6066
#[allow(clippy::needless_collect)]
impl Catalog for MutableCatalog {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_schemas<'a>(&'a self) -> Box<dyn Iterator<Item = SchemaRef> + 'a> {
        let schemas = {
            let schemas = self.schemas.read().unwrap();
            schemas.values().cloned().collect::<Vec<SchemaRef>>()
        };
        Box::new(schemas.into_iter())
    }

    fn get_schema_by_name(&self, name: &str) -> Option<SchemaRef> {
        let schemas = self.schemas.read().unwrap();
        schemas.get(name).cloned()
    }

    fn get_table(&self, name: &str) -> Option<TableRef> {
        let schemas = self.schemas.read().unwrap();
        schemas.get(DEFAULT_SCHEMA).map(|s| s.get_table_by_name(name)).flatten()
    }

    fn get_index(&self, name: &str) -> Option<IndexRef> {
        let schemas = self.schemas.read().unwrap();
        schemas.get(DEFAULT_SCHEMA).map(|s| s.get_index_by_name(name)).flatten()
    }

    fn get_indexes<'a>(&'a self, table: &str) -> Box<dyn Iterator<Item = IndexRef> + 'a> {
        let schemas = self.schemas.read().unwrap();
        let schema = schemas.get(DEFAULT_SCHEMA);
        match schema {
            Some(schema) => {
                let indexes = schema.get_indexes(table).collect::<Vec<IndexRef>>();
                Box::new(indexes.into_iter())
            }
            None => Box::new(std::iter::empty()),
        }
    }
}

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

    pub fn add_table(&self, table: Table) {
        let mut inner = self.inner.write().unwrap();
        inner.tables.insert(table.name().into(), Arc::new(table));
    }

    pub fn add_index(&self, index: Index) {
        let mut inner = self.inner.write().unwrap();
        let _ = inner.tables.get(index.table()).unwrap_or_else(|| {
            panic!("Unable to add index {:?} - table {:?} does not exist", index.name(), index.table())
        });

        let existing = match inner.indexes.entry(index.name().into()) {
            Entry::Occupied(mut o) => {
                let index = Arc::new(index);
                let existing = o.get();
                let name = existing.name().clone();
                let table = existing.table().clone();

                o.insert(index);
                Some((name, table))
            }
            Entry::Vacant(v) => {
                let index = Arc::new(index);
                v.insert(index.clone());
                match inner.table_indexes.entry(index.table().into()) {
                    Entry::Occupied(mut o) => {
                        o.get_mut().push(index);
                    }
                    Entry::Vacant(v) => {
                        v.insert(vec![index]);
                    }
                };
                None
            }
        };
        if let Some((name, table)) = existing {
            let table_indexes = inner.table_indexes.get_mut(&table).unwrap();
            table_indexes.retain(|i| i.name() != &name);
        }
    }

    pub fn remove_table(&self, name: &str) {
        let mut inner = self.inner.write().unwrap();
        inner.tables.remove(name);
        inner.table_indexes.remove(name);
    }

    pub fn remove_index(&self, index: &str) {
        let mut inner = self.inner.write().unwrap();
        if let Some(index) = inner.indexes.remove(index) {
            let table_index = inner.table_indexes.get_mut(index.table()).unwrap();
            table_index.retain(|i| i.name() != index.name());
        }
    }
}

#[derive(Default, Debug)]
struct Inner {
    tables: HashMap<String, TableRef>,
    indexes: HashMap<String, IndexRef>,
    table_indexes: HashMap<String, Vec<IndexRef>>,
}

// see https://github.com/rust-lang/rust-clippy/issues/6066
#[allow(clippy::needless_collect)]
impl Schema for MutableSchema {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_tables<'a>(&'a self) -> Box<dyn Iterator<Item = TableRef> + 'a> {
        let inner = self.inner.write().unwrap();
        let tables = inner.tables.values().cloned().collect::<Vec<TableRef>>();
        Box::new(tables.into_iter())
    }

    fn get_table_by_name(&self, name: &str) -> Option<TableRef> {
        let inner = self.inner.write().unwrap();
        inner.tables.get(name).cloned()
    }

    fn get_indexes<'a>(&'a self, table: &str) -> Box<dyn Iterator<Item = IndexRef> + 'a> {
        let inner = self.inner.write().unwrap();
        let indexes = inner.table_indexes.get(table).cloned().unwrap_or_else(|| Vec::with_capacity(0));
        Box::new(indexes.into_iter())
    }

    fn get_index_by_name(&self, name: &str) -> Option<IndexRef> {
        let inner = self.inner.write().unwrap();
        inner.indexes.get(name).cloned()
    }
}

#[cfg(test)]
mod test {
    use crate::catalog::mutable::{MutableCatalog, MutableSchema};
    use crate::catalog::{Catalog, IndexBuilder, Schema, TableBuilder};
    use crate::datatypes::DataType;

    #[test]
    fn test_tables() {
        let catalog = MutableCatalog::new();
        let table = TableBuilder::new("A").add_column("a1", DataType::Int32).build();

        catalog.add_table("s", table);

        let schema = catalog.get_schema_by_name("s").unwrap();
        let _ = schema
            .get_table_by_name("A")
            .unwrap_or_else(|| panic!("table A is missing from the schema s"));

        let schema: &MutableSchema = schema.as_any().downcast_ref::<MutableSchema>().unwrap();
        schema.remove_table("A");
        assert_eq!(schema.get_tables().count(), 0, "table has not been removed")
    }

    #[test]
    fn test_indexes() {
        let catalog = MutableCatalog::new();
        let table1 = TableBuilder::new("A").add_column("a1", DataType::Int32).build();
        let index1 = IndexBuilder::new("A_idx").table("A").add_column(table1.get_column("a1").unwrap()).build();

        let table2 = TableBuilder::new("B").add_column("b1", DataType::Int32).build();
        let index2 = IndexBuilder::new("B_idx").table("B").add_column(table2.get_column("b1").unwrap()).build();

        catalog.add_table("s", table1);
        catalog.add_index("s", index1);

        catalog.add_table("s", table2);
        catalog.add_index("s", index2);

        let schema = catalog.get_schema_by_name("s").unwrap();
        let _ = schema
            .get_index_by_name("A_idx")
            .unwrap_or_else(|| panic!("index A_idx is missing from the schema s"));

        let schema: &MutableSchema = schema.as_any().downcast_ref::<MutableSchema>().unwrap();
        schema.remove_index("A_idx");

        assert_eq!(schema.get_indexes("A").count(), 0, "index has not been removed");
        assert_eq!(schema.get_indexes("B").count(), 1, "indexes from another table have been removed as well");

        schema.remove_table("B");
        assert_eq!(schema.get_indexes("B").count(), 0, "Table B has been removed but indexes are still available");
    }
}
