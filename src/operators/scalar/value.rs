use crate::datatypes::DataType;
use std::fmt::{Display, Formatter};

/// Supported scalar values.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ScalarValue {
    Null,
    Bool(bool),
    Int32(i32),
    String(String),
}

impl ScalarValue {
    /// Returns the type of this scalar value.
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Null => DataType::Null,
            ScalarValue::Bool(_) => DataType::Bool,
            ScalarValue::Int32(_) => DataType::Int32,
            ScalarValue::String(_) => DataType::String,
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
