//! Properties of an operator.
//!
//! Logical properties are shared by all expressions in a memo-group.
//! Physical properties are describe physical characteristics of the data (such as ordering).
//! These properties are required by some operators. For example MergeSortJoin requires its inputs to be ordered.

use itertools::Itertools;
use std::fmt::{Display, Formatter};

use crate::meta::ColumnId;

pub mod logical;
pub mod physical;

/// Ordering. Describes how columns are sorted.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct OrderingChoice {
    columns: Vec<OrderingColumn>,
}

impl OrderingChoice {
    /// Creates an ordering from the given ordering columns.
    ///
    /// # Panics
    ///
    /// This method panics if the ordering columns are empty.
    pub fn new(columns: Vec<OrderingColumn>) -> Self {
        assert!(!columns.is_empty(), "columns are not specified");
        OrderingChoice { columns }
    }

    /// Creates an ordering from the given columns.
    /// Returns an ordering where all columns are ordered in ascending order.
    pub fn from_columns(columns: Vec<ColumnId>) -> Self {
        OrderingChoice::new(columns.into_iter().map(OrderingColumn::asc).collect())
    }

    /// A reference to the ordering columns.
    pub fn columns(&self) -> &[OrderingColumn] {
        &self.columns
    }

    /// Returns true if the this ordering is the prefix of the given ordering.
    pub fn prefix_of(&self, other: &OrderingChoice) -> bool {
        if self.columns.len() > other.columns.len() {
            return false;
        }
        self.columns.iter().zip(other.columns.iter()).all(|(l, r)| l == r)
    }
}

impl Display for OrderingChoice {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.columns.iter().join(", "))
    }
}

/// Specifies how the column is sorted.
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash, Ord, PartialOrd)]
pub struct OrderingColumn(i32);

impl OrderingColumn {
    /// Creates an ordering for the given column.
    pub fn ord(id: ColumnId, descending: bool) -> Self {
        if descending {
            OrderingColumn::desc(id)
        } else {
            OrderingColumn::asc(id)
        }
    }

    /// Ordering in ascending order for the given column.
    pub fn asc(id: ColumnId) -> Self {
        OrderingColumn(id.0 as i32)
    }

    /// Ordering in descending order for the given column.
    pub fn desc(id: ColumnId) -> Self {
        OrderingColumn(id.0 as i32).to_desc()
    }

    /// Returns the column.
    pub fn column(&self) -> ColumnId {
        ColumnId(self.0.abs() as usize)
    }

    /// Returns `true` if the column is sorted in descending order.
    pub fn descending(&self) -> bool {
        self.0 < 0
    }

    /// Returns `true` if the column is sorted in ascending order.
    pub fn ascending(&self) -> bool {
        self.0 > 0
    }

    /// Converts this ordering to ascending ordering.
    pub fn to_asc(self) -> OrderingColumn {
        OrderingColumn(self.0.abs())
    }

    /// Converts this ordering to descending ordering.
    pub fn to_desc(self) -> OrderingColumn {
        OrderingColumn(-self.0.abs())
    }
}

impl Display for OrderingColumn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.descending() {
            // if ordering is descending than the column id has a negative sign.
            write!(f, "{}", self.0)
        } else {
            write!(f, "+{}", self.0)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::meta::testing::TestMetadata;
    use crate::meta::ColumnId;
    use crate::properties::{OrderingChoice, OrderingColumn};

    fn new_column() -> ColumnId {
        let mut metadata = TestMetadata::with_tables(vec!["A"]);
        metadata.column("A").build()
    }

    #[test]
    fn ordering_asc() {
        let col1 = new_column();
        let col1_asc = OrderingColumn::ord(col1, false);

        assert_eq!(col1_asc.column(), col1, "column");

        expect_format(&col1_asc, "+1", "OrderingColumn::ord(col, ASC)");

        assert!(col1_asc.ascending(), "asc");
        assert!(!col1_asc.descending(), "desc");

        let col1_asc = OrderingColumn::asc(col1);
        expect_format(&col1_asc, "+1", "OrderingColumn::asc");
    }

    #[test]
    fn ordering_desc() {
        let col1 = new_column();
        let col1_desc = OrderingColumn::ord(col1, true);

        assert_eq!(col1_desc.column(), col1, "column");

        expect_format(&col1_desc, "-1", "OrderingColumn::ord(col, DESC)");

        assert!(!col1_desc.ascending(), "asc");
        assert!(col1_desc.descending(), "desc");

        let col1_desc = OrderingColumn::desc(col1);
        expect_format(&col1_desc, "-1", "OrderingColumn::desc");
    }

    #[test]
    fn ordering_asc_to_desc() {
        let col1 = new_column();

        let col1_asc = OrderingColumn::asc(col1);
        let col1_desc = OrderingColumn::desc(col1);

        assert_eq!(col1_asc.to_desc(), col1_desc, "col:1 to_desc")
    }

    #[test]
    fn ordering_desc_to_asc() {
        let col1 = new_column();

        let col1_asc = OrderingColumn::asc(col1);
        let col1_desc = OrderingColumn::desc(col1);

        assert_eq!(col1_desc.to_asc(), col1_asc, "col:1 to_asc")
    }

    #[test]
    fn prefix_of() {
        let mut metadata = TestMetadata::with_tables(vec!["A"]);
        let _ = metadata.column("A").named("a1").build();
        let _ = metadata.column("A").named("a2").build();
        let _ = metadata.column("A").named("a3").build();

        let left = build_ordering(&metadata, "A", &["a1:asc"]);
        let right = build_ordering(&metadata, "A", &["a1:asc", "a2:asc"]);
        assert!(left.prefix_of(&right), "left: {} right {}", left, right);

        let left = build_ordering(&metadata, "A", &["a1:asc", "a2:asc"]);
        let right = build_ordering(&metadata, "A", &["a1:asc", "a2:asc"]);
        assert!(left.prefix_of(&right), "left: {} right {}", left, right);

        let left = build_ordering(&metadata, "A", &["a1:desc", "a2:asc"]);
        let right = build_ordering(&metadata, "A", &["a1:desc", "a2:asc"]);
        assert!(left.prefix_of(&right), "left: {} right {}", left, right);
    }

    #[test]
    fn not_prefix_of() {
        let mut metadata = TestMetadata::with_tables(vec!["A"]);
        let _ = metadata.column("A").named("a1").build();
        let _ = metadata.column("A").named("a2").build();
        let _ = metadata.column("A").named("a3").build();

        let left = build_ordering(&metadata, "A", &["a1:asc", "a2:asc", "a3:asc"]);
        let right = build_ordering(&metadata, "A", &["a1:asc", "a2:asc"]);
        assert!(!left.prefix_of(&right), "NOT prefix left: {} right {}", left, right);

        let left = build_ordering(&metadata, "A", &["a1:desc"]);
        let right = build_ordering(&metadata, "A", &["a1:asc", "a2:asc"]);
        assert!(!left.prefix_of(&right), "NOT prefix left: {} right {}", left, right);
    }

    fn expect_format(ord: &OrderingColumn, expected: &str, name: &str) {
        let actual = format!("{}", ord);
        assert_eq!(actual, expected, "{}", name);
    }

    fn build_ordering(metadata: &TestMetadata, table: &str, columns: &[&str]) -> OrderingChoice {
        let columns = columns
            .iter()
            .map(|name| {
                let (name, ord) = name.split_once(":").unwrap();
                let id = metadata.find_column(table, name);
                match ord {
                    "asc" => OrderingColumn::asc(id),
                    "desc" => OrderingColumn::desc(id),
                    _ => panic!("Unexpected column ordering: {}", name),
                }
            })
            .collect();
        OrderingChoice::new(columns)
    }
}
