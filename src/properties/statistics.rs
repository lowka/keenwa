/// The number of rows returned by an operator in case when no statistics is available.
pub const UNKNOWN_ROW_COUNT: f64 = 1000f64;

/// Statistics associated with an operator.
#[derive(Debug, Clone)]
pub struct Statistics {
    row_count: f64,
    selectivity: f64,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            row_count: UNKNOWN_ROW_COUNT,
            selectivity: Statistics::DEFAULT_SELECTIVITY,
        }
    }
}

impl Statistics {
    /// The default value of selectivity statistics.
    pub const DEFAULT_SELECTIVITY: f64 = 1.0;

    /// Creates new statistics with the given row count and selectivity.
    ///
    /// # Panics
    ///
    /// This method panics if the `row_count` is negative or the `selectivity` lies outside `[0.0, 1.0]` bounds.
    pub fn new(row_count: f64, selectivity: f64) -> Self {
        assert!(row_count >= 0f64, "row_count must be non negative");
        assert!(
            (0f64..=Self::DEFAULT_SELECTIVITY).contains(&selectivity),
            "selectivity must be within [0.0, 1.0] range but got: {}",
            selectivity
        );
        Statistics { row_count, selectivity }
    }

    /// Creates a new statistics with selectivity set to the given value.
    pub fn from_selectivity(selectivity: f64) -> Self {
        Statistics::new(0.0, selectivity)
    }

    /// Creates a new statistics with row_count set to the given value.
    pub fn from_row_count(row_count: f64) -> Self {
        Statistics::new(row_count, Self::DEFAULT_SELECTIVITY)
    }

    /// The estimated number of rows returned by an operator.
    pub fn row_count(&self) -> f64 {
        self.row_count
    }

    /// The selectivity of a predicate.
    pub fn selectivity(&self) -> f64 {
        self.selectivity
    }
}
