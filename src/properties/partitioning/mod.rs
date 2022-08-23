use crate::meta::ColumnId;
use itertools::Itertools;
use std::fmt::{Display, Formatter};

/// Partitioning scheme.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub enum Partitioning {
    /// Singleton partitioning
    Singleton,
    /// Tuples are partitioned by the given columns.
    Partitioned(Vec<ColumnId>),
    // use ordering column(?)
    //TODO: Rename to range partitioning.
    OrderedPartitioning(Vec<ColumnId>),
    HashPartitioning(Vec<ColumnId>),
    //RandomPartitioning
}

impl Partitioning {
    /// Checks whether this partitioning is a subset of the `other` partitioning.
    pub fn subset_of(&self, other: &Partitioning) -> bool {
        use Partitioning::*;
        match (self, other) {
            (Singleton, Singleton) => true,
            (Partitioned(this), Partitioned(other)) => {
                if this.len() > other.len() {
                    false
                } else {
                    this.iter().all(|c| other.contains(c))
                }
            }
            (Singleton, Partitioned(_)) => false,
            (Partitioned(_), Singleton) => false,
            // Ordered Partitioning
            (OrderedPartitioning(this), OrderedPartitioning(other)) => {
                if this.len() > other.len() {
                    false
                } else {
                    this.iter().zip(other.iter()).all(|(l, r)| l == r)
                }
            }
            (OrderedPartitioning(_), _) => false,
            (_, OrderedPartitioning(_)) => false,
            // HashPartitioning
            (HashPartitioning(this), HashPartitioning(other)) => {
                if this.len() > other.len() {
                    false
                } else {
                    this.iter().all(|c| other.contains(c))
                }
            }
            (HashPartitioning(_), _) => false,
            (_, HashPartitioning(_)) => false,
        }
    }

    /// Creates a new partitioning in which all columns columns from the `source_columns` with the columns `output_columns`.
    pub fn with_mapping(&self, source_columns: &[ColumnId], output_columns: &[ColumnId]) -> Partitioning {
        match self {
            Partitioning::Singleton => Partitioning::Singleton,
            Partitioning::Partitioned(cols)
            | Partitioning::OrderedPartitioning(cols)
            | Partitioning::HashPartitioning(cols) => {
                let columns: Vec<_> = cols
                    .iter()
                    .copied()
                    .map(|col| {
                        if let Some(p) =
                            source_columns.iter().position(|s| *s == col).filter(|p| *p < output_columns.len())
                        {
                            output_columns[p]
                        } else {
                            col
                        }
                    })
                    .collect();
                Partitioning::Partitioned(columns)
            }
        }
    }

    /// Returns this partitioning is its canonical from.
    /// Partitioning is said to be in its canonical form iff:
    /// - The canonical form of a [singleton partitioning scheme](Partitioning::Singleton)
    /// is the same partitioning.
    /// - The canonical form of a [partitioned by columns scheme](Partitioning::Partitioned)
    /// is a partitioning scheme of type [Partitioning::Partitioned]
    /// where all its columns are ordered by their ids in the ascending order.
    /// - The canonical form of an [ordered partitioning scheme](Partitioning::OrderedPartitioning)
    /// is the same partitioning.
    /// - The canonical form of a [hash partitioning scheme](Partitioning::HashPartitioning)
    /// is a partitioning scheme of type [Partitioning::HashPartitioning]  
    /// where all its columns are ordered by their ids in the ascending order.
    pub fn canonical_form(self) -> Partitioning {
        match self {
            Partitioning::Singleton => self.clone(),
            Partitioning::Partitioned(mut cols) => {
                cols.sort();
                Partitioning::Partitioned(cols)
            }
            Partitioning::OrderedPartitioning(_) => self.clone(),
            Partitioning::HashPartitioning(mut cols) => {
                cols.sort();
                Partitioning::HashPartitioning(cols)
            }
        }
    }

    /// Returns partitioning in its normalized form.
    /// Partitioning is said to be in normalized form iff:
    /// - The normalized form of a [singleton partitioning scheme](Partitioning::Singleton)
    /// is the same partitioning.
    /// - The normalized form of a [partitioned by columns scheme](Partitioning::Partitioned)
    /// is partitioning scheme of type [Partitioning::Partitioned]
    /// where all its columns are ordered by their ids in the ascending order.
    /// - The normalized form of an [ordered partitioning scheme](Partitioning::OrderedPartitioning)
    /// partitioning scheme of type [Partitioning::Partitioned] in its normalized form.
    /// - The normalized form of a [hash partitioning scheme](Partitioning::HashPartitioning)
    /// partitioning scheme of type [Partitioning::Partitioned] in its normalized form.
    pub fn normalize(self) -> Partitioning {
        match self {
            Partitioning::Singleton => self.clone(),
            Partitioning::Partitioned(mut cols) => {
                cols.sort();
                Partitioning::Partitioned(cols)
            }
            Partitioning::OrderedPartitioning(mut cols) => {
                cols.sort();
                Partitioning::Partitioned(cols)
            }
            Partitioning::HashPartitioning(mut cols) => {
                cols.sort();
                Partitioning::Partitioned(cols)
            }
        }
    }

    /// Returns an iterator over all possible implementations of this partitioning.
    /// If this partitioning scheme is an implementation then this method returns `None`.
    pub fn get_implementations(&self) -> Option<impl Iterator<Item = Partitioning>> {
        match self {
            Partitioning::Singleton => None,
            Partitioning::Partitioned(cols) => Some(
                vec![
                    Partitioning::HashPartitioning(cols.clone()),
                    Partitioning::OrderedPartitioning(cols.clone()),
                ]
                .into_iter(),
            ),
            Partitioning::OrderedPartitioning(_) => None,
            Partitioning::HashPartitioning(_) => None,
        }
    }
}

impl Display for Partitioning {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Partitioning::Singleton => write!(f, "()"),
            Partitioning::Partitioned(columns) => write!(f, "[{}]", columns.iter().join(", ")),
            Partitioning::OrderedPartitioning(columns) => write!(f, "ord[{}]", columns.iter().join(", ")),
            Partitioning::HashPartitioning(columns) => write!(f, "hash({})", columns.iter().join(", ")),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::meta::testing::TestMetadata;

    #[test]
    fn subset_of() {
        fn expect_subset(left: &Partitioning, right: &Partitioning) {
            assert!(left.subset_of(right), "{} must be a subset of {}", left, right);
        }

        fn expect_not_subset(left: &Partitioning, right: &Partitioning) {
            assert!(!left.subset_of(right), "{} must not subset of {}", left, right);
        }

        let mut metadata = TestMetadata::with_tables(vec!["A"]);
        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();
        let col3 = metadata.column("A").build();

        let p12 = Partitioning::Partitioned(vec![col1, col2]);
        let p21 = Partitioning::Partitioned(vec![col2, col1]);
        let p123 = Partitioning::Partitioned(vec![col1, col2, col3]);

        expect_subset(&p12, &p21);
        expect_subset(&p21, &p12);

        expect_subset(&p12, &p123);
        expect_subset(&p21, &p123);

        expect_not_subset(&p123, &p12);
        expect_not_subset(&p123, &p21);

        //

        let s = Partitioning::Singleton;

        expect_not_subset(&p12, &s);
        expect_not_subset(&s, &p12);

        expect_subset(&s, &s);
    }

    #[test]
    fn with_mapping() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();

        let col3 = metadata.column("B").build();
        let col4 = metadata.column("B").build();

        let p1 = Partitioning::Partitioned(vec![col1, col2]);
        let p2 = Partitioning::Partitioned(vec![col3, col4]);

        let result = p1.with_mapping(&[col1, col2], &[col3, col4]);
        assert_eq!(result, p2, "p1 -> p2");

        let result = p2.with_mapping(&[col3, col4], &[col1, col2]);
        assert_eq!(result, p1, "p2 -> p1");
    }

    #[test]
    fn with_mapping_retains_other_columns() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B", "C"]);
        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();

        let col3 = metadata.column("B").build();
        let col4 = metadata.column("B").build();

        let col5 = metadata.column("C").build();
        let col6 = metadata.column("C").build();

        let p1 = Partitioning::Partitioned(vec![col1, col5, col2]);
        let p2 = Partitioning::Partitioned(vec![col3, col5, col4]);

        let result = p1.with_mapping(&[col1, col2], &[col3, col4]);
        assert_eq!(result, p2, "p1 -> p2");

        let result = p2.with_mapping(&[col3, col4], &[col1, col2]);
        assert_eq!(result, p1, "p2 -> p1");
    }

    #[test]
    fn with_mapping_output_contains_less_columns() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();
        let col3 = metadata.column("B").build();

        let p1 = Partitioning::Partitioned(vec![col1, col2]);
        let result = p1.with_mapping(&[col1, col2], &[col3]);

        assert_eq!(result, Partitioning::Partitioned(vec![col3, col2]));
    }

    #[test]
    fn canonical_form() {
        let mut metadata = TestMetadata::with_tables(vec!["A"]);
        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();

        let p12 = Partitioning::Partitioned(vec![col1, col2]);
        let p21 = Partitioning::Partitioned(vec![col2, col1]);

        assert_eq!(p21.canonical_form(), p12);
    }
}
