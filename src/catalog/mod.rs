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
    /// Returns this catalog as [`Any`](std::any::Any) in order it can be downcast to its implementation.
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
    /// Returns this schema as [`Any`](std::any::Any) in order it can be downcast to its implementation.
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
        if self.columns.is_empty() {
            return Err(OptimizerError::argument("No columns has been specified"));
        }

        let mut names = HashSet::new();
        for col in self.columns.iter() {
            let col_name = col.name();
            if !names.insert(col_name) {
                let message = format!("Column already exists. Column: {} table: {}", col_name, self.name);
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
            return Err(OptimizerError::argument("No columns have been specified"));
        }

        if let Some(ordering) = self.ordering.as_ref() {
            for opt in ordering.options.iter() {
                if !self.columns.iter().any(|c| c == &opt.column.name) {
                    return Err(OptimizerError::argument(format!(
                        "ordering option {:?} is not covered by index. Columns: {:?}",
                        opt, self.columns
                    )));
                }
            }
        }

        let columns = std::mem::take(&mut self.columns);

        let columns: Result<Vec<ColumnRef>, _> = columns
            .into_iter()
            .map(|name| match self.table.get_column(name.as_str()) {
                Some(col) => Ok(col),
                None => Err(OptimizerError::argument(format!(
                    "Column does not exist. Table: {}, column: {}",
                    &self.table.name, name
                ))),
            })
            .collect();

        Ok(Index {
            name: self.name,
            table: self.table.name.clone(),
            columns: columns?,
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
    pub fn build(self) -> Result<Ordering, OptimizerError> {
        let mut options = vec![];
        let table = self.table;

        for (name, descending) in self.columns {
            match table.get_column(name.as_str()) {
                Some(column) => {
                    options.push(OrderingOption { column, descending });
                }
                None => {
                    return Err(OptimizerError::argument(format!(
                        "Column does not exist. Table: {}, column: {}",
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
    pub(crate) fn new(column_name: String, table_name: Option<String>, data_type: DataType) -> Self {
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
    pub fn table(&self) -> Option<&String> {
        self.table.as_ref()
    }

    /// The data type of this column.
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
}
