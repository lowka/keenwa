//! Data type supported by the optimizer.

use itertools::Itertools;
use std::fmt::{Display, Formatter};

/// Data types supported in scalar expressions and column definitions.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum DataType {
    /// NULL data type.
    Null,
    /// Boolean.
    Bool,
    /// 32-bit signed integer.
    Int32,
    /// Utf-8 string.
    String,
    /// Date in days since January 1, year 1 in the Gregorian calendar.
    Date,
    /// Time (seconds and nanoseconds) since midnight.
    Time,
    /// Timestamp in milliseconds since UNIX epoch.
    /// The boolean argument indicates whether this timestamp type has a timezone or not.
    Timestamp(bool),
    /// Interval.
    Interval,
    /// Tuple data type.
    ///
    /// The arguments holds the types of each field of this tuple type.
    Tuple(Vec<DataType>),
    /// Array data type.
    ///
    /// The argument holds the element type of this array type.
    Array(Box<DataType>),
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Null => write!(f, "Null"),
            DataType::Bool => write!(f, "Bool"),
            DataType::Int32 => write!(f, "Int32"),
            DataType::String => write!(f, "String"),
            DataType::Date => write!(f, "Date"),
            DataType::Time => write!(f, "Time"),
            DataType::Timestamp(tz) => write!(f, "Timestamp{tz}", tz = if *tz { " with time zone" } else { "" }),
            DataType::Interval => write!(f, "Interval"),
            DataType::Tuple(values) => write!(f, "Tuple({})", values.iter().join(", ")),
            DataType::Array(element_type) => write!(f, "{}[]", element_type),
        }
    }
}
