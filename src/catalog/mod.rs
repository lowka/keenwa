//! Database catalog.

use std::any::Any;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;

use crate::datatypes::DataType;
use crate::error::OptimizerError;

pub mod mutable;

pub type CatalogRef = Arc<dyn Catalog>;
pub type SchemaRef = Arc<dyn Schema>;
pub type TableRef = Arc<Table>;
pub type IndexRef = Arc<Index>;
pub type ColumnRef = Arc<Column>;

/// Provides access to database objects used by the optimizer.
//TODO: Add tests that ensure that this trait is object safe
pub trait Catalog: Debug + Sync + Send {
    /// Returns this catalog as [Any] in order it can be downcast to its implementation.
    fn as_any(&self) -> &dyn Any;

    /// Returns an iterator over schemas available in the catalog.
    fn get_schemas(&self) -> Vec<SchemaRef>;

    /// Returns a schema with the given name.
    fn get_schema_by_name(&self, name: &str) -> Option<SchemaRef>;

    /// Returns a table with the given name registered in the default schema.
    fn get_table(&self, name: &str) -> Option<TableRef>;

    /// Returns an index with the given name registered in default schema.
    fn get_index(&self, name: &str) -> Option<IndexRef>;

    /// Returns all indexes of of the given table in default schema.
    fn get_indexes(&self, table: &str) -> Vec<IndexRef>;
}

/// The name of default schema.
pub const DEFAULT_SCHEMA: &str = "default";

/// Represents a database schema.
//TODO: Add tests that ensure that this trait is object safe
pub trait Schema: Debug + Sync + Send {
    /// Returns this schema as [Any] in order it can be downcast to its implementation.
    fn as_any(&self) -> &dyn Any;

    /// Returns an iterator over tables registered in this schema.
    fn get_tables(&self) -> Vec<TableRef>;

    /// Returns a table with the given name.
    fn get_table_by_name(&self, name: &str) -> Option<TableRef>;

    /// Returns an iterator over indexes stored in this catalog.
    fn get_indexes(&self, table: &str) -> Vec<IndexRef>;

    /// Returns an index with the given name.
    fn get_index_by_name(&self, name: &str) -> Option<IndexRef>;
}

/// Represents a database table.
#[derive(Debug, Clone)]
pub struct Table {
    name: String,
    columns: Vec<ColumnRef>,
    statistics: Option<TableStatistics>,
}

impl Table {
    /// The name of this table.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The columns of this table.
    pub fn columns(&self) -> &[ColumnRef] {
        &self.columns
    }

    /// Returns a column with the given name.
    pub fn get_column(&self, name: &str) -> Option<ColumnRef> {
        self.columns.iter().find(|c| c.name == name).cloned()
    }

    /// Returns statistics available for this table.
    pub fn statistics(&self) -> Option<&TableStatistics> {
        self.statistics.as_ref()
    }
}

/// Statistics for a database table.
#[derive(Debug, Clone)]
pub struct TableStatistics {
    row_count: Option<usize>,
}

impl TableStatistics {
    /// Creates a new table statistics object.
    pub fn new(row_count: usize) -> Self {
        TableStatistics {
            row_count: Some(row_count),
        }
    }

    /// The total number of rows in a table.
    pub fn row_count(&self) -> Option<usize> {
        self.row_count
    }
}

/// A builder to create instances of a [table].
///
/// [table]: crate::catalog::Table
#[derive(Debug, Clone)]
pub struct TableBuilder {
    name: String,
    columns: Vec<ColumnRef>,
    statistics: Option<TableStatistics>,
}

impl TableBuilder {
    /// Creates a builder for a table the given name.
    pub fn new(name: &str) -> Self {
        TableBuilder {
            name: name.to_string(),
            columns: Vec::new(),
            statistics: None,
        }
    }

    /// Creates a builder from the given table.
    pub fn from_table(table: &Table) -> Self {
        let name = table.name.clone();
        let columns = table.columns.clone();
        let statistics = table.statistics.clone();

        let mut table_builder = TableBuilder::new(name.as_str());
        table_builder.columns = columns;
        table_builder.statistics = statistics;
        table_builder
    }

    /// Adds a column with the given name and data type to this table.
    pub fn add_column(mut self, name: &str, data_type: DataType) -> TableBuilder {
        let column = Column::new(name.to_string(), Some(self.name.clone()), data_type);
        self.columns.push(Arc::new(column));
        self
    }

    /// Sets row count statistics for this table.
    pub fn add_row_count(mut self, row_count: usize) -> TableBuilder {
        self.statistics = Some(TableStatistics::new(row_count));
        self
    }

    /// Creates an instance of a [table] with previously specified properties.
    ///
    /// [table]: crate::catalog::Table
    pub fn build(self) -> Result<Table, OptimizerError> {
        let mut names = HashSet::new();

        for col in self.columns.iter() {
            let col_name = col.name();
            if !names.insert(col_name) {
                let message = format!("Table: column already exists. Column: {} table: {}", col_name, self.name);
                return Err(OptimizerError::argument(message));
            }
        }

        Ok(Table {
            name: self.name,
            columns: self.columns,
            statistics: self.statistics,
        })
    }
}

/// Represents a database index.
#[derive(Debug, Clone)]
pub struct Index {
    name: String,
    table: String,
    // columns or expressions
    columns: Vec<ColumnRef>,
    ordering: Option<Ordering>,
}

impl Index {
    /// The name of this index.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The name of the table this index is defined for.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Returns the columns covered by this index.
    pub fn columns(&self) -> &[ColumnRef] {
        &self.columns
    }

    /// Specifies how values are sorted within this index.
    pub fn ordering(&self) -> Option<&Ordering> {
        self.ordering.as_ref()
    }
}

/// A builder to create instances of an [Index].
///
/// [Index]: crate::catalog::Index
#[derive(Debug, Clone)]
pub struct IndexBuilder {
    name: String,
    table: TableRef,
    columns: Vec<String>,
    ordering: Option<Ordering>,
}

impl IndexBuilder {
    /// Creates a builder for an index with the given name.
    pub fn new(table: TableRef, name: &str) -> Self {
        IndexBuilder {
            name: name.to_string(),
            table,
            columns: Vec::new(),
            ordering: None,
        }
    }

    /// Adds a column covered by this index.
    pub fn add_column(mut self, column: &str) -> IndexBuilder {
        self.columns.push(column.into());
        self
    }

    /// Specifies sorting options for this index.
    pub fn ordering(mut self, ordering: Ordering) -> IndexBuilder {
        self.ordering = Some(ordering);
        self
    }

    /// Creates an instance of an [index] with previously specified properties.
    ///
    /// [index]: crate::catalog::Index
    pub fn build(mut self) -> Result<Index, OptimizerError> {
        if self.columns.is_empty() {
            return Err(OptimizerError::argument("Index: no columns have been specified"));
        }

        if let Some(ordering) = self.ordering.as_ref() {
            for opt in ordering.options.iter() {
                if !self.columns.iter().any(|c| c == &opt.column.name) {
                    return Err(OptimizerError::argument(format!(
                        "Index: ordering option is not covered by index. Columns: {:?}. Ordering: {} {direction}",
                        self.columns,
                        opt.column.name(),
                        direction = if opt.descending { "DESC" } else { "ASC" },
                    )));
                }
            }
        }

        let mut columns = Vec::with_capacity(self.columns.len());

        for col_name in std::mem::take(&mut self.columns) {
            let col = match self.table.get_column(col_name.as_str()) {
                Some(col) if columns.contains(&col) => {
                    let message = format!("Index: column has been specified more than once. Column: {}", col_name);
                    return Err(OptimizerError::argument(message));
                }
                Some(col) => col,
                None => {
                    let message =
                        format!("Index: column does not exist. Table: {}, column: {}", &self.table.name, col_name);
                    return Err(OptimizerError::argument(message));
                }
            };

            columns.push(col);
        }

        Ok(Index {
            name: self.name,
            table: self.table.name.clone(),
            columns,
            ordering: self.ordering,
        })
    }
}

/// Specifies how data is sorted.
#[derive(Debug, Clone)]
pub struct Ordering {
    options: Vec<OrderingOption>,
}

/// A builder to create instances of [Ordering].
#[derive(Debug, Clone)]
pub struct OrderingBuilder {
    table: TableRef,
    columns: Vec<(String, bool)>,
}

impl OrderingBuilder {
    /// Creates a new instance of a builder to create an ordering from the columns of the given table.
    pub fn new(table: TableRef) -> Self {
        OrderingBuilder {
            table,
            columns: Vec::new(),
        }
    }

    /// Add a descending ordering for the given column.
    pub fn add_desc(self, column: &str) -> Self {
        self.add(column, true)
    }

    /// Add an ascending a ordering for the given column.
    pub fn add_asc(self, column: &str) -> Self {
        self.add(column, false)
    }

    /// Add an ordering for the given column.
    pub fn add(mut self, column: &str, descending: bool) -> Self {
        self.columns.push((String::from(column), descending));
        self
    }

    /// Creates an instance of [Ordering].
    ///
    /// * If the table does not contain a column specified in the ordering the method returns an error.
    /// * If some column has been specified more than once only the first ordering of that column
    /// is added to the resulting ordering (this behaviour is consistent with sorting produced by
    /// `ORDER BY` clause in `SQL`).
    pub fn build(self) -> Result<Ordering, OptimizerError> {
        if self.columns.is_empty() {
            return Err(OptimizerError::argument("Ordering: no columns have been specified"));
        }

        let mut options = vec![];
        let mut columns = HashSet::new();
        let table = self.table;

        for (name, descending) in self.columns {
            if !columns.insert(name.clone()) {
                continue;
            }
            match table.get_column(name.as_str()) {
                Some(column) => {
                    options.push(OrderingOption { column, descending });
                }
                None => {
                    return Err(OptimizerError::argument(format!(
                        "Ordering: column does not exist. Table: {}, column: {}",
                        table.name, name
                    )))
                }
            }
        }

        Ok(Ordering { options })
    }
}

impl Ordering {
    /// Creates a new instance of `Ordering`.
    pub fn new(options: Vec<OrderingOption>) -> Self {
        Ordering { options }
    }

    /// Options of this ordering.
    pub fn options(&self) -> &[OrderingOption] {
        &self.options
    }
}

/// An ordering option describe how values of a particular column are sorted.
///
/// [Ordering]: crate::catalog::schema::Ordering
#[derive(Debug, Clone)]
pub struct OrderingOption {
    column: ColumnRef,
    descending: bool,
}

impl OrderingOption {
    /// The column this ordering is applied to.
    pub fn column(&self) -> ColumnRef {
        self.column.clone()
    }

    /// Whether the values are sorted in descending order or not.
    pub fn descending(&self) -> bool {
        self.descending
    }
}

/// A column of a database table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    name: String,
    table: Option<String>,
    data_type: DataType,
}

impl Column {
    fn new(column_name: String, table_name: Option<String>, data_type: DataType) -> Self {
        Column {
            name: column_name,
            table: table_name,
            data_type,
        }
    }

    /// The name of this column.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The name of the table this column belongs to.
    /// If table is not specified then this column is derived from some expression.
    pub fn table(&self) -> Option<&str> {
        self.table.as_deref()
    }

    /// The data type of this column.
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

//

#[allow(dead_code)]
fn __type_system_guarantees() {
    __ensure_type_is_sync_send::<dyn Catalog>();
    __ensure_type_is_sync_send::<dyn Schema>();

    fn catalog_trait_is_object_safe(_arc: Arc<dyn Catalog>) {}
    fn schema_trait_is_object_safe(_arc: Arc<dyn Schema>) {}
}

#[allow(dead_code)]
fn __ensure_type_is_sync_send<T>()
where
    T: Sync + Send + ?Sized,
{
}

//

#[cfg(test)]
mod test {
    use crate::catalog::{IndexBuilder, Ordering, OrderingBuilder, TableBuilder};
    use crate::datatypes::DataType;
    use crate::error::OptimizerError;
    use std::fmt::Debug;
    use std::sync::Arc;

    // table

    #[test]
    fn test_table() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::String)
            .build();

        match table {
            Ok(table) => {
                assert_eq!(table.name(), "A", "name");
                assert_eq!(table.columns().len(), 2, "num columns");

                assert_eq!(table.columns()[0].name(), "a1", "column a1 name");
                assert_eq!(table.columns()[0].table(), Some("A"), "column a1 table");
                assert_eq!(table.columns()[0].data_type(), &DataType::Int32, "column a1 data type");

                assert_eq!(table.columns()[1].name(), "a2", "column a2 name");
                assert_eq!(table.columns()[1].table(), Some("A"), "column a2 table");
                assert_eq!(table.columns()[1].data_type(), &DataType::String, "column a2 data type");

                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    #[test]
    fn test_table_get_column() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::String)
            .add_column("a3", DataType::Bool)
            .build()?;

        assert_eq!(table.get_column("a1"), Some(table.columns[0].clone()));
        assert_eq!(table.get_column("a2"), Some(table.columns[1].clone()));
        assert_eq!(table.get_column("a3"), Some(table.columns[2].clone()));
        assert_eq!(table.get_column("a4"), None, "not existing column");

        Ok(())
    }

    #[test]
    fn test_table_adding_multiple_columns_with_the_same_name_is_not_allowed() {
        let res = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::String)
            .add_column("a1", DataType::Bool)
            .build();

        expect_error(res, "Table: column already exists")
    }

    #[test]
    fn test_table_allow_tables_without_columns() -> Result<(), OptimizerError> {
        let _ = TableBuilder::new("A").build()?;
        Ok(())
    }

    // index

    #[test]
    fn test_index() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::String)
            .build()?;
        let index = IndexBuilder::new(Arc::new(table.clone()), "A_a1_index").add_column("a1").build()?;

        assert_eq!(index.name(), "A_a1_index", "index name");
        assert_eq!(index.table(), "A", "table name");
        assert_eq!(index.columns().len(), 1, "columns num");

        assert_eq!(index.columns()[0], table.columns[0]);

        Ok(())
    }

    #[test]
    fn test_index_reject_index_that_uses_column_multiple_times() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::String)
            .build()?;

        let res = IndexBuilder::new(Arc::new(table.clone()), "A_a1_index")
            .add_column("a1")
            .add_column("a1")
            .build();

        expect_error(res, "Index: column has been specified more than once");

        Ok(())
    }

    #[test]
    fn test_index_reject_index_that_uses_unknown_column() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::String)
            .build()?;

        let res = IndexBuilder::new(Arc::new(table.clone()), "A_a1_index")
            .add_column("a1")
            .add_column("a4")
            .build();

        expect_error(res, "Index: column does not exist");

        Ok(())
    }

    #[test]
    fn test_index_reject_index_without_columns() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::String)
            .add_column("a3", DataType::Bool)
            .build()?;

        let table = Arc::new(table);
        let res = IndexBuilder::new(table, "A_a1_a2_idx").build();

        expect_error(res, "Index: no columns have been specified");

        Ok(())
    }

    #[test]
    fn test_index_reject_index_with_ordering_that_contains_unknown_fields() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::String)
            .add_column("a3", DataType::Bool)
            .build()?;

        let table = Arc::new(table);
        let ordering = OrderingBuilder::new(table.clone()).add_asc("a1").add_desc("a3").build()?;

        let res = IndexBuilder::new(table, "A_a1_a2_idx")
            .add_column("a1")
            .add_column("a2")
            .ordering(ordering)
            .build();

        expect_error(res, "Index: ordering option is not covered by index");

        Ok(())
    }

    // ordering

    #[test]
    fn test_ordering() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::Int32)
            .build()?;

        let table = Arc::new(table);
        {
            let ordering = OrderingBuilder::new(table.clone()).add("a1", true).add("a2", false).build()?;

            expect_col_ordering(&ordering, "a1", 0, true);
            expect_col_ordering(&ordering, "a2", 1, false);
        }

        {
            let ordering = OrderingBuilder::new(table.clone()).add_desc("a1").add_asc("a2").build()?;

            expect_col_ordering(&ordering, "a1", 0, true);
            expect_col_ordering(&ordering, "a2", 1, false);
        }

        Ok(())
    }

    #[test]
    fn test_ordering_reject_ordering_containing_unknown_errors() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::Int32)
            .build()?;
        let table = Arc::new(table);

        let res = OrderingBuilder::new(table.clone()).add("a1", true).add("a3", false).build();
        expect_error(res, "Ordering: column does not exist");

        Ok(())
    }

    #[test]
    fn test_ordering_ignore_subsequent_ordering_of_the_same_column() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::Int32)
            .build()?;
        let table = Arc::new(table);

        let ordering = OrderingBuilder::new(table).add_asc("a1").add_desc("a2").add_desc("a1").build()?;

        assert_eq!(ordering.options().len(), 2, "Expected 2 columns");

        expect_col_ordering(&ordering, "a1", 0, false);
        expect_col_ordering(&ordering, "a2", 1, true);

        Ok(())
    }

    #[test]
    fn test_ordering_reject_ordering_without_columns() -> Result<(), OptimizerError> {
        let table = TableBuilder::new("A").build()?;
        let err = OrderingBuilder::new(Arc::new(table)).build().err();

        assert!(matches!(err, Some(OptimizerError::Argument(_))), "error");

        Ok(())
    }

    fn expect_col_ordering(ordering: &Ordering, col: &str, pos: usize, desc: bool) {
        let options = ordering.options();
        let actual_pos = options.iter().position(|ord| ord.column.name() == col).unwrap();
        assert_eq!(actual_pos, pos, "Position does not match. Column: {}", col);
        assert_eq!(options[pos].descending(), desc, "Direction does not match. Column: {}", col)
    }

    fn expect_error<T>(result: Result<T, OptimizerError>, message: &str)
    where
        T: Debug,
    {
        match result {
            Ok(r) => assert!(false, "Unexpected result: {:?}", r),
            Err(OptimizerError::Argument(err)) => {
                assert!(err.message().contains(message), "Unexpected error: {}. Expected: {}", err.message(), message);
            }
            Err(err) => assert!(false, "Unexpected error: {}. Expected: {}", err, message),
        }
    }
}
