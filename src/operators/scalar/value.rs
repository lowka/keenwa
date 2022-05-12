use itertools::Itertools;
use std::fmt::{Display, Formatter};

use crate::datatypes::DataType;
use crate::operators::scalar::value::Interval::{DaySecond, YearMonth};

/// Supported scalar values.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ScalarValue {
    Null,
    Bool(bool),
    Int32(i32),
    String(String),
    /// Date in days since UNIX epoch.
    Date(i64),
    /// Time in milliseconds since midnight.
    Time(i32),
    /// Time in milliseconds since UNIX epoch with an optional timezone.
    Timestamp(i64, Option<String>),
    /// Interval.
    Interval(Interval),
    /// Tuple.
    Tuple(Vec<ScalarValue>),
    /// Array.
    Array(Array),
}

/// Interval.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Interval {
    /// Year to month interval. When an interval is negative both values are negative.
    YearMonth(i32, i32),
    /// Day to second interval. Stores time as seconds.
    /// When an interval is negative both values are negative.
    DaySecond(i32, i32),
}

const MONTHS_IN_YEAR: u32 = 12;
const HOURS_IN_DAY: u32 = 24;
const SECONDS_IN_MINUTE: u32 = 60;
const MINUTE_IN_HOUR: u32 = 60;

impl Interval {
    /// Creates an instance of a YearMonth interval from the given values with the given sign.
    ///
    /// # Panics
    ///
    /// This method panics if the provided values are out of their valid range and the sign is neither `1` nor `-1`.
    pub fn from_year_month(sign: i32, years: u32, month: u32) -> Interval {
        assert!(sign == -1 || sign == 1, "sign must be -1 or 1");
        assert!(month < MONTHS_IN_YEAR, "months must be between 0 and 11 (inclusive): {}", month);

        YearMonth(sign * years as i32, sign * month as i32)
    }

    /// Creates an instance of a DaySecond interval from the given values with the given sign.
    ///
    /// # Panics
    ///
    /// This method panics if the provided values if are out of their range and the sign is neither `1` nor `-1`.
    pub fn from_days_seconds(sign: i32, days: u32, hours: u32, minutes: u32, seconds: u32) -> Interval {
        assert!(sign == -1 || sign == 1, "sign must be -1 or 1");
        assert!(hours < HOURS_IN_DAY, "hours must be between  0 and 23 (inclusive): {}", hours);
        assert!(minutes < MINUTE_IN_HOUR, "minutes must be between 0 and 59 (inclusive): {}", minutes);
        assert!(seconds < SECONDS_IN_MINUTE, "seconds must be between 0 and 59 (inclusive): {}", seconds);

        let time_in_seconds = hours * MINUTE_IN_HOUR * SECONDS_IN_MINUTE + minutes * SECONDS_IN_MINUTE + seconds;
        DaySecond(sign * days as i32, sign * time_in_seconds as i32)
    }
}

/// Array.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Array {
    /// The type of elements.
    pub element_type: DataType,
    /// Elements of the array.
    pub elements: Vec<ScalarValue>,
}

impl ScalarValue {
    /// Returns the type of this scalar value.
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Null => DataType::Null,
            ScalarValue::Bool(_) => DataType::Bool,
            ScalarValue::Int32(_) => DataType::Int32,
            ScalarValue::String(_) => DataType::String,
            ScalarValue::Date(_) => DataType::Date,
            ScalarValue::Time(_) => DataType::Time,
            ScalarValue::Timestamp(_, tz) => DataType::Timestamp(tz.is_some()),
            ScalarValue::Interval(_) => DataType::Interval,
            ScalarValue::Tuple(values) => DataType::Tuple(values.iter().map(|val| val.data_type()).collect()),
            ScalarValue::Array(array) => array.element_type.clone(),
        }
    }
}

impl Display for ScalarValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Null => write!(f, "NULL"),
            ScalarValue::Bool(value) => write!(f, "{}", value),
            ScalarValue::Int32(value) => write!(f, "{}", value),
            ScalarValue::String(value) => write!(f, "{}", value),
            ScalarValue::Date(value) => write!(f, "{}", value),
            ScalarValue::Time(value) => write!(f, "{}", value),
            ScalarValue::Timestamp(value, tz) => {
                write!(f, "{}", value)?;
                if let Some(tz) = tz {
                    write!(f, " {}", tz)?;
                }
                Ok(())
            }
            ScalarValue::Interval(Interval::YearMonth(years, months)) => {
                let sign = if *years < 0 || *months < 0 { "-" } else { "" };

                write!(f, "{sign}{} YEARS {} MONTHS", years.abs(), months.abs(), sign = sign)
            }
            ScalarValue::Interval(Interval::DaySecond(days, time_in_seconds)) => {
                let sign = if *days < 0 || *time_in_seconds < 0 { "-" } else { "" };

                let days = days.abs() as u32;
                let time_in_seconds = time_in_seconds.abs() as u32;
                let hours = time_in_seconds / MINUTE_IN_HOUR / SECONDS_IN_MINUTE;
                let hours_in_seconds = hours * MINUTE_IN_HOUR * SECONDS_IN_MINUTE;
                let minutes = (time_in_seconds - hours_in_seconds) / MINUTE_IN_HOUR;
                let seconds = time_in_seconds - hours_in_seconds - minutes * MINUTE_IN_HOUR;
                let parts = [(days, "DAYS"), (hours, "HOURS"), (minutes, "MINUTES"), (seconds, "SECONDS")];

                write!(f, "{}", sign)?;
                for (i, (value, label)) in parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{} {}", value, label)?;
                }
                Ok(())
            }
            ScalarValue::Tuple(values) => {
                write!(f, "({})", values.iter().join(", "))
            }
            ScalarValue::Array(array) => {
                write!(f, "[{}]", array.elements.iter().join(", "))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_scalar_value_data_types() {
        assert_eq!(ScalarValue::Null.data_type(), DataType::Null, "null value");
        assert_eq!(ScalarValue::Bool(true).data_type(), DataType::Bool, "bool value");
        assert_eq!(ScalarValue::Int32(1).data_type(), DataType::Int32, "i32 value");
        assert_eq!(ScalarValue::String(String::from("abc")).data_type(), DataType::String, "string value");
    }
}
