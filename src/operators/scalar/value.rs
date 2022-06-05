use chrono::{DateTime, Datelike, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime, Timelike, Utc};
use itertools::Itertools;
use std::fmt::{Display, Formatter};

use crate::datatypes::DataType;
use crate::error::OptimizerError;

/// Supported scalar values.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ScalarValue {
    /// Null.
    Null,
    /// Boolean.
    Bool(Option<bool>),
    /// 32-bit signed integer.
    Int32(Option<i32>),
    /// Utf-8 string.
    String(Option<String>),
    /// Date in days since January 1, year 1 in the Gregorian calendar.
    Date(Option<i32>),
    /// Time (seconds and nanoseconds) since midnight.
    Time(Option<(u32, u32)>),
    /// Time in milliseconds since UNIX epoch with an optional timezone.
    Timestamp(Option<i64>, Option<i32>),
    /// Interval.
    Interval(Option<Interval>),
    /// Tuple and its type [DataType::Tuple].
    Tuple(Option<Vec<ScalarValue>>, Box<DataType>),
    /// Array and its type [DataType::Array].
    Array(Option<Array>, Box<DataType>),
}

/// Converts a type to a [scalar value](ScalarValue).
pub trait Scalar {
    /// Convert this type to a scalar value.
    fn get_value(&self) -> ScalarValue;
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

        Interval::YearMonth(sign * years as i32, sign * month as i32)
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
        Interval::DaySecond(sign * days as i32, sign * time_in_seconds as i32)
    }
}

/// Array.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Array {
    /// Elements of the array.
    pub elements: Vec<ScalarValue>,
}

impl ScalarValue {
    /// Returns the type of this scalar value.
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Null => DataType::Null,
            ScalarValue::Bool(Some(_)) => DataType::Bool,
            ScalarValue::Bool(None) => DataType::Null,
            ScalarValue::Int32(_) => DataType::Int32,
            ScalarValue::String(_) => DataType::String,
            ScalarValue::Date(_) => DataType::Date,
            ScalarValue::Time(_) => DataType::Time,
            ScalarValue::Timestamp(_, tz) => DataType::Timestamp(tz.is_some()),
            ScalarValue::Interval(_) => DataType::Interval,
            ScalarValue::Tuple(_, tuple_type) => *tuple_type.clone(),
            ScalarValue::Array(_, array_type) => *array_type.clone(),
        }
    }
}

impl Display for ScalarValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Null => write!(f, "NULL"),
            ScalarValue::Bool(Some(value)) => write!(f, "{}", value),
            ScalarValue::Int32(Some(value)) => write!(f, "{}", value),
            ScalarValue::String(Some(value)) => write!(f, "{}", value),
            ScalarValue::Date(Some(value)) => match NaiveDate::from_num_days_from_ce_opt(*value) {
                Some(date) => write!(f, "{}", date),
                None => write!(f, "Invalid date: {} days", value),
            },
            ScalarValue::Time(Some((secs, nanos))) => {
                match NaiveTime::from_num_seconds_from_midnight_opt(*secs, *nanos) {
                    Some(time) => write!(f, "{}", time.format("%H:%M:%S%.3f")),
                    None => write!(f, "Invalid time: {} secs {} nanos", secs, nanos),
                }
            }
            ScalarValue::Timestamp(Some(value), tz) => {
                let secs = *value / 1000;
                let millis = *value - 1000 * secs;

                let timezone = tz.map(|tz| FixedOffset::east_opt(tz).map(Ok).unwrap_or(Err(tz)));
                let datetime = NaiveDateTime::from_timestamp_opt(secs, (millis * 1_000_000) as u32)
                    .map(Ok)
                    .unwrap_or(Err(millis));

                match (datetime, timezone) {
                    (Ok(timestamp), Some(Ok(offset))) => {
                        let timestamp: DateTime<FixedOffset> = DateTime::from_utc(timestamp, offset);
                        write!(f, "{}", timestamp.format("%Y-%m-%dT%H:%M:%S%.3f%Z"))?;
                    }
                    (Ok(timestamp), None) => {
                        let timestamp: DateTime<FixedOffset> = DateTime::from_utc(timestamp, FixedOffset::east(0));
                        write!(f, "{}", timestamp.format("%Y-%m-%dT%H:%M:%S%.3f"))?;
                    }
                    (_, offset) => {
                        write!(f, "Invalid timestamp: {} millis", millis)?;
                        match offset {
                            Some(Ok(offset)) => {
                                write!(f, " [offset {} seconds]", offset.local_minus_utc())?;
                            }
                            Some(Err(offset)) => {
                                write!(f, " [invalid offset: {} seconds]", offset)?;
                            }
                            None => {}
                        }
                    }
                }
                Ok(())
            }
            ScalarValue::Interval(Some(Interval::YearMonth(years, months))) => {
                let sign = if *years < 0 || *months < 0 { "-" } else { "" };

                write!(f, "{sign}{} YEARS {} MONTHS", years.abs(), months.abs(), sign = sign)
            }
            ScalarValue::Interval(Some(Interval::DaySecond(days, time_in_seconds))) => {
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
            ScalarValue::Tuple(Some(values), _) => {
                write!(f, "({})", values.iter().join(", "))
            }
            ScalarValue::Array(Some(array), _) => {
                write!(f, "[{}]", array.elements.iter().join(", "))
            }
            ScalarValue::Bool(None) => write!(f, "NULL"),
            ScalarValue::Int32(None) => write!(f, "NULL"),
            ScalarValue::String(None) => write!(f, "NULL"),
            ScalarValue::Date(None) => write!(f, "NULL"),
            ScalarValue::Time(None) => write!(f, "NULL"),
            ScalarValue::Timestamp(None, _) => write!(f, "NULL"),
            ScalarValue::Interval(None) => write!(f, "NULL"),
            ScalarValue::Tuple(None, _) => write!(f, "NULL"),
            ScalarValue::Array(None, _) => write!(f, "NULL"),
        }
    }
}

/// Parses the given string into [ScalarValue::Date].
pub fn parse_date(input: &str) -> Result<ScalarValue, OptimizerError> {
    match input.parse::<chrono::NaiveDate>() {
        Ok(val) => Ok(ScalarValue::Date(Some(val.num_days_from_ce()))),
        Err(err) => Err(OptimizerError::argument(format!("Invalid date: {}", err))),
    }
}

/// Parses the given string into [ScalarValue::Time].
pub fn parse_time(input: &str) -> Result<ScalarValue, OptimizerError> {
    match input.parse::<chrono::NaiveTime>() {
        Ok(val) => {
            let secs = val.num_seconds_from_midnight();
            let nanos = val.nanosecond();
            Ok(ScalarValue::Time(Some((secs, nanos))))
        }
        Err(err) => Err(OptimizerError::argument(format!("Invalid time: {}", err))),
    }
}

/// Parses the given RFC 3339 string into [ScalarValue::Timestamp] without timezone.
pub fn parse_timestamp(input: &str) -> Result<ScalarValue, OptimizerError> {
    // DATE`T`TIME[timezone]
    // If the right part contains `+` or `-` sign then a string probably include a timezone information.
    let may_include_timezone =
        matches!(input.split_once('T'), Some((_, right)) if right.contains('+') || right.contains('-'));

    if may_include_timezone {
        match DateTime::parse_from_rfc3339(input) {
            Ok(val) => Ok(ScalarValue::Timestamp(
                Some(val.with_timezone(&Utc).timestamp_millis()),
                Some(val.timezone().local_minus_utc()),
            )),
            Err(err) => Err(OptimizerError::argument(format!("Invalid timestamp: {}", err))),
        }
    } else {
        match NaiveDateTime::parse_from_str(input, "%Y-%m-%dT%H:%M:%S%.3f") {
            Ok(val) => Ok(ScalarValue::Timestamp(Some(val.timestamp_millis()), None)),
            Err(err) => Err(OptimizerError::argument(format!("Invalid timestamp: {}", err))),
        }
    }
}

impl Scalar for bool {
    fn get_value(&self) -> ScalarValue {
        ScalarValue::Bool(Some(*self))
    }
}

impl Scalar for i32 {
    fn get_value(&self) -> ScalarValue {
        ScalarValue::Int32(Some(*self))
    }
}

impl Scalar for &str {
    fn get_value(&self) -> ScalarValue {
        ScalarValue::String(Some((*self).to_owned()))
    }
}

impl Scalar for String {
    fn get_value(&self) -> ScalarValue {
        ScalarValue::String(Some(self.clone()))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use chrono::{Datelike, NaiveDate};

    #[test]
    fn test_scalar_value_data_types() {
        assert_eq!(ScalarValue::Null.data_type(), DataType::Null, "null value");
        assert_eq!(true.get_value().data_type(), DataType::Bool, "bool value");
        assert_eq!(1.get_value().data_type(), DataType::Int32, "i32 value");
        assert_eq!(ScalarValue::String(Some(String::from("abc"))).data_type(), DataType::String, "string value");

        assert_eq!(
            ScalarValue::Interval(Some(Interval::YearMonth(0, 0))).data_type(),
            DataType::Interval,
            "date value"
        );
        assert_eq!(
            ScalarValue::Interval(Some(Interval::DaySecond(0, 0))).data_type(),
            DataType::Interval,
            "date value"
        );

        assert_eq!(ScalarValue::Date(Some(1)).data_type(), DataType::Date, "date value");
        assert_eq!(ScalarValue::Time(Some((0, 0))).data_type(), DataType::Time, "time value");
        assert_eq!(ScalarValue::Timestamp(Some(0), None).data_type(), DataType::Timestamp(false), "timestamp value");
        assert_eq!(
            ScalarValue::Timestamp(Some(0), Some(0)).data_type(),
            DataType::Timestamp(true),
            "timestamp with time zone value"
        );

        let fields = vec![ScalarValue::Bool(Some(true)), ScalarValue::Int32(Some(1))];
        let tuple_type = Box::new(DataType::Tuple(fields.iter().map(|f| f.data_type()).collect()));
        assert_eq!(
            ScalarValue::Tuple(Some(fields), tuple_type).data_type(),
            DataType::Tuple(vec![DataType::Bool, DataType::Int32]),
            "tuple value"
        );

        let array = Array {
            elements: vec![ScalarValue::Int32(Some(1))],
        };
        let array_type = Box::new(DataType::Array(Box::new(array.elements[0].data_type())));

        assert_eq!(
            ScalarValue::Array(Some(array), array_type).data_type(),
            DataType::Array(Box::new(DataType::Int32)),
            "array value"
        );
    }

    #[test]
    pub fn test_parse_date() {
        let input = "2000-10-01";
        let result = parse_date(input).expect("Failed to parse date");
        let expected_date = NaiveDate::from_ymd(2000, 10, 1);
        let actual_date = ScalarValue::Date(Some(expected_date.num_days_from_ce()));

        assert_eq!(actual_date, result);
        assert_eq!(format!("{}", actual_date), input, "Display impl for Date");
    }

    #[test]
    pub fn test_parse_time() {
        let input = "01:18:53.502";
        let result = parse_time(input).expect("Failed to parse time");
        let expected_time = NaiveTime::from_hms_nano(1, 18, 53, 502_000_000);
        let actual_time =
            ScalarValue::Time(Some((expected_time.num_seconds_from_midnight(), expected_time.nanosecond())));

        assert_eq!(actual_time, result);
        assert_eq!(format!("{}", actual_time), input, "Display impl for Time");
    }

    #[test]
    pub fn test_parse_timestamp() {
        let input = "2000-10-01T01:18:53.100";
        let result = parse_timestamp(input).expect("Failed to parse timestamp");
        assert_eq!(format!("{}", result), input);

        let input = "2000-10-01T01:18:53";
        let result = parse_timestamp(input).expect("Failed to parse timestamp");
        assert_eq!(format!("{}", result), format!("{}.000", input));
    }

    #[test]
    pub fn test_parse_timestamp_with_timezone() {
        let input = "2000-10-01T01:18:53.100+10:15";
        let result = parse_timestamp(input).expect("Failed to parse timestamp");
        assert_eq!(format!("{}", result), input);

        let input = "2000-10-01T01:18:53.100-10:15";
        let result = parse_timestamp(input).expect("Failed to parse timestamp");
        assert_eq!(format!("{}", result), input);
    }

    #[test]
    pub fn test_format_date() {
        fn format_date(days: i32, expected: &str) {
            let ts = ScalarValue::Date(Some(days));
            assert_eq!(format!("{}", ts), expected, "days: {:?}", days);
        }
        format_date(-100000000, "Invalid date: -100000000 days");

        format_date(-1500, "-0004-11-22");
        format_date(0, "0000-12-31");
        format_date(1, "0001-01-01");
        format_date(700_000, "1917-07-15");

        format_date(100000000, "Invalid date: 100000000 days");
    }

    #[test]
    pub fn test_format_time() {
        fn format_time(seconds: u32, nanos: u32, expected: &str) {
            let ts = ScalarValue::Time(Some((seconds, nanos)));
            assert_eq!(format!("{}", ts), expected, "seconds: {:?} nanos: {:?}", seconds, nanos);
        }

        format_time(0, 0, "00:00:00.000");
        format_time(0, 10_000_000, "00:00:00.010");
        format_time(0, 1_000_000_000, "00:00:01.000");

        format_time(4 * 60 * 60 + 12 * 60, 0, "04:12:00.000");

        format_time(24 * 60 * 60 + 1, 0, "Invalid time: 86401 secs 0 nanos");

        format_time(1000000, 1_000_000_000, "Invalid time: 1000000 secs 1000000000 nanos");
        format_time(1000000, 0, "Invalid time: 1000000 secs 0 nanos");
    }

    #[test]
    pub fn test_format_timestamp() {
        fn format_ts(millis: i64, zone_offset: Option<i32>, expected: &str) {
            let ts = ScalarValue::Timestamp(Some(millis), zone_offset);
            assert_eq!(format!("{}", ts), expected, "millis: {:?} zone offset: {:?}", millis, zone_offset);
        }

        format_ts(0, None, "1970-01-01T00:00:00.000");
        format_ts(0, Some(60 + 15), "1970-01-01T00:01:15.000+00:01:15");

        format_ts(100000, None, "1970-01-01T00:01:40.000");
        format_ts(100000, Some(60 + 15), "1970-01-01T00:02:55.000+00:01:15");
        format_ts(100000, Some(-160 + 15), "1969-12-31T23:59:15.000-00:02:25");

        format_ts(-100, None, "Invalid timestamp: -100 millis");
        format_ts(-100, Some(60 + 10), "Invalid timestamp: -100 millis [offset 70 seconds]");
        format_ts(-100, Some(24 * 60 * 60 + 1), "Invalid timestamp: -100 millis [invalid offset: 86401 seconds]");
        format_ts(
            -100,
            Some((24 * 60 * 60 + 1) * -1),
            "Invalid timestamp: -100 millis [invalid offset: -86401 seconds]",
        );
    }
}
