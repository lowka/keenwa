use std::fmt::{Display, Formatter};

/// Data types supported in scalar expressions and column definitions.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum DataType {
    Null,
    Bool,
    Int32,
    String,
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Null => write!(f, "Null"),
            DataType::Bool => write!(f, "Bool"),
            DataType::Int32 => write!(f, "Int32"),
            DataType::String => write!(f, "String"),
        }
    }
}
