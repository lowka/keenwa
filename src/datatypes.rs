/// Data types supported in scalar expressions and column definitions.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum DataType {
    Null,
    Bool,
    Int32,
    String,
}
