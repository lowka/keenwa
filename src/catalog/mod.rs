use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use crate::datatypes::DataType;

pub mod mutable;

pub type CatalogRef = Arc<dyn Catalog>;
pub type SchemaRef = Arc<dyn Schema>;
pub type TableRef = Arc<Table>;
pub type IndexRef = Arc<Index>;
pub type ColumnRef = Arc<Column>;

/// Provides access to database objects used by the optimizer.
//TODO: Add tests that ensure that this trait is object safe
pub trait Catalog: Debug {
    /// Returns this catalog as [`Any`](std::any::Any) in order it can be downcast to its implementation.
    fn as_any(&self) -> &dyn Any;

    /// Returns an iterator over schemas available in the catalog.
    fn get_schemas<'a>(&'a self) -> Box<dyn Iterator<Item = SchemaRef> + 'a>;

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
pub trait Schema: Debug {
    /// Returns this schema as [`Any`](std::any::Any) in order it can be downcast to its implementation.
    fn as_any(&self) -> &dyn Any;

    /// Returns an iterator over tables registered in this schema.
    fn get_tables<'a>(&'a self) -> Box<dyn Iterator<Item = TableRef> + 'a>;

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
    pub fn name(&self) -> &String {
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
    pub fn build(self) -> Table {
        // assert!(!self.columns.is_empty(), "No columns has been specified");
        Table {
            name: self.name,
            columns: self.columns,
            statistics: self.statistics,
        }
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
    pub fn name(&self) -> &String {
        &self.name
    }

    /// The name of the table this index is defined for.
    pub fn table(&self) -> &String {
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
    table: Option<String>,
    columns: Vec<ColumnRef>,
    ordering: Option<Ordering>,
}

impl IndexBuilder {
    /// Creates a builder for an index with the given name.
    pub fn new(name: &str) -> Self {
        IndexBuilder {
            name: name.to_string(),
            table: None,
            columns: Vec::new(),
            ordering: None,
        }
    }

    /// Specifies table this index belongs to.
    pub fn table(mut self, table: &str) -> IndexBuilder {
        self.table = Some(table.to_string());
        self
    }

    /// Adds a column covered by this index.
    pub fn add_column(mut self, column: ColumnRef) -> IndexBuilder {
        assert!(column.table.is_some(), "Index can not contain columns that do not belong to a table: {:?}", column);
        self.columns.push(column);
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
    pub fn build(self) -> Index {
        assert!(!self.columns.is_empty(), "No columns have been specified");

        if let Some(ordering) = self.ordering.as_ref() {
            for opt in ordering.options.iter() {
                assert!(
                    self.columns.iter().any(|c| c == &opt.column),
                    "ordering option {:?} is not covered by index. Columns: {:?}",
                    opt,
                    self.columns
                )
            }
        }

        Index {
            name: self.name,
            table: self.table.expect("table is not specified"),
            columns: self.columns,
            ordering: self.ordering,
        }
    }
}

/// Specifies how data is sorted.
#[derive(Debug, Clone)]
pub struct Ordering {
    options: Vec<OrderingOption>,
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
    fn column(&self) -> ColumnRef {
        self.column.clone()
    }

    /// Whether the values are sorted in descending order or not.
    fn descending(&self) -> bool {
        self.descending
    }
}

/// A column of a database table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    //FIXME Column names may be optional. For example: SELECT 1 as a, count(1).
    // The second column in the query above has no name and can not be referenced from the parent scope.
    name: String,
    table: Option<String>,
    data_type: DataType,
}

impl Column {
    pub(crate) fn new(column_name: String, table_name: Option<String>, data_type: DataType) -> Self {
        Column {
            name: column_name,
            table: table_name,
            data_type,
        }
    }

    /// The name of this column.
    pub fn name(&self) -> &String {
        &self.name
    }

    /// The name of the table this column belongs to.
    /// If table is not specified then this column is derived from some expression.
    pub fn table(&self) -> Option<&String> {
        self.table.as_ref()
    }

    /// The data type of this column.
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
}
