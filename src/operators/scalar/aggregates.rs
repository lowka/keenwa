use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::operators::scalar::func::{get_return_type, FunctionSignature, FunctionSignatureBuilder};
use std::convert::TryFrom;
use std::fmt::{Display, Formatter};
use std::num::NonZeroUsize;

/// Supported aggregate functions.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum AggregateFunction {
    Avg,
    Count,
    Max,
    Min,
    Sum,
}

impl AggregateFunction {
    /// Returns the return type of this aggregate function if it called with the given arguments.
    pub fn get_return_type(&self, arg_types: &[DataType]) -> Result<DataType, OptimizerError> {
        let signature = self.get_signature();
        get_return_type(self, &signature, arg_types)
    }

    fn get_signature(&self) -> FunctionSignature {
        fn func() -> FunctionSignatureBuilder {
            let arg_lists: Vec<_> = DataType::NUMERIC_TYPES.iter().map(|tpe| vec![tpe.clone()]).collect();
            FunctionSignatureBuilder::one_of(arg_lists)
        }

        fn any(usize: usize) -> FunctionSignatureBuilder {
            let num_args = NonZeroUsize::new(usize).unwrap();
            FunctionSignatureBuilder::any(num_args)
        }

        match self {
            AggregateFunction::Avg => func().return_mirrors(),
            AggregateFunction::Count => any(1).return_type(DataType::Int32),
            AggregateFunction::Max => func().return_mirrors(),
            AggregateFunction::Min => func().return_mirrors(),
            AggregateFunction::Sum => func().return_mirrors(),
        }
    }
}

impl TryFrom<&str> for AggregateFunction {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "avg" => Ok(AggregateFunction::Avg),
            "count" => Ok(AggregateFunction::Count),
            "max" => Ok(AggregateFunction::Max),
            "min" => Ok(AggregateFunction::Min),
            "sum" => Ok(AggregateFunction::Sum),
            _ => Err(()),
        }
    }
}

impl Display for AggregateFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateFunction::Avg => write!(f, "avg"),
            AggregateFunction::Count => write!(f, "count"),
            AggregateFunction::Max => write!(f, "max"),
            AggregateFunction::Min => write!(f, "min"),
            AggregateFunction::Sum => write!(f, "sum"),
        }
    }
}

/// Supported window functions.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum WindowFunction {
    RowNumber,
    Rank,
    FirstValue,
    LastValue,
}

impl WindowFunction {
    /// Returns the return type of this window function if is called with the given arguments.
    pub fn get_return_type(&self, arg_types: &[DataType]) -> Result<DataType, OptimizerError> {
        let signature = self.get_signature();
        get_return_type(self, &signature, arg_types)
    }

    fn get_signature(&self) -> FunctionSignature {
        match self {
            WindowFunction::RowNumber => FunctionSignatureBuilder::exact(vec![]).return_type(DataType::Int32),
            WindowFunction::Rank => FunctionSignatureBuilder::exact(vec![]).return_type(DataType::Int32),
            WindowFunction::FirstValue => FunctionSignatureBuilder::any(NonZeroUsize::new(1).unwrap()).return_mirrors(),
            WindowFunction::LastValue => FunctionSignatureBuilder::any(NonZeroUsize::new(1).unwrap()).return_mirrors(),
        }
    }
}

impl TryFrom<&str> for WindowFunction {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "row_number" => Ok(WindowFunction::RowNumber),
            "rank" => Ok(WindowFunction::Rank),
            "first_value" => Ok(WindowFunction::FirstValue),
            "last_value" => Ok(WindowFunction::LastValue),
            _ => Err(()),
        }
    }
}

impl Display for WindowFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowFunction::RowNumber => write!(f, "row_number"),
            WindowFunction::Rank => write!(f, "rank"),
            WindowFunction::FirstValue => write!(f, "first_value"),
            WindowFunction::LastValue => write!(f, "last_value"),
        }
    }
}

/// Function that can be used by window aggregate expressions.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum WindowOrAggregateFunction {
    Aggregate(AggregateFunction),
    Window(WindowFunction),
}

impl TryFrom<&str> for WindowOrAggregateFunction {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match AggregateFunction::try_from(value) {
            Ok(f) => Ok(WindowOrAggregateFunction::Aggregate(f)),
            Err(_) => WindowFunction::try_from(value).map(WindowOrAggregateFunction::Window),
        }
    }
}

impl WindowOrAggregateFunction {
    pub fn get_return_type(&self, args_types: &[DataType]) -> Result<DataType, OptimizerError> {
        match self {
            WindowOrAggregateFunction::Aggregate(f) => f.get_return_type(args_types),
            WindowOrAggregateFunction::Window(f) => f.get_return_type(args_types),
        }
    }
}

impl From<AggregateFunction> for WindowOrAggregateFunction {
    fn from(f: AggregateFunction) -> Self {
        WindowOrAggregateFunction::Aggregate(f)
    }
}

impl From<WindowFunction> for WindowOrAggregateFunction {
    fn from(f: WindowFunction) -> Self {
        WindowOrAggregateFunction::Window(f)
    }
}

impl Display for WindowOrAggregateFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowOrAggregateFunction::Aggregate(func) => write!(f, "{}", func),
            WindowOrAggregateFunction::Window(func) => write!(f, "{}", func),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::datatypes::DataType;
    use crate::operators::scalar::aggregates::{AggregateFunction, WindowFunction, WindowOrAggregateFunction};
    use crate::{test_from_str, test_function_signature};

    fn test_aggregate_function(f: AggregateFunction, expected_name: &str, expected_signature: &str) {
        test_function_signature!(AggregateFunction, &f, expected_name, expected_signature);
        test_from_str!(WindowOrAggregateFunction, &WindowOrAggregateFunction::Aggregate(f), expected_name);
    }

    fn test_window_aggr_function(f: WindowFunction, expected_name: &str, expected_signature: &str) {
        test_function_signature!(WindowFunction, &f, expected_name, expected_signature);
        test_from_str!(WindowOrAggregateFunction, &WindowOrAggregateFunction::Window(f), expected_name);
    }

    #[test]
    fn avg() {
        test_aggregate_function(AggregateFunction::Avg, "avg", "one of [int32], [float32] -> <mirrors> immutable");
    }

    #[test]
    fn count() {
        test_aggregate_function(AggregateFunction::Count, "count", "any <1> -> int32 immutable");

        for tpe in vec![DataType::Null, DataType::Int32, DataType::String, DataType::Bool] {
            let actual_return_type = AggregateFunction::Count.get_return_type(&[tpe.clone()]).unwrap();
            assert_eq!(actual_return_type, DataType::Int32, "count({}) return type", tpe)
        }
    }

    #[test]
    fn min() {
        test_aggregate_function(AggregateFunction::Min, "min", "one of [int32], [float32] -> <mirrors> immutable");
    }

    #[test]
    fn max() {
        test_aggregate_function(AggregateFunction::Max, "max", "one of [int32], [float32] -> <mirrors> immutable");
    }

    #[test]
    fn sum() {
        test_aggregate_function(AggregateFunction::Sum, "sum", "one of [int32], [float32] -> <mirrors> immutable");
    }

    #[test]
    fn rank() {
        test_window_aggr_function(WindowFunction::Rank, "rank", "exact -> int32 immutable");
    }

    #[test]
    fn row_number() {
        test_window_aggr_function(WindowFunction::RowNumber, "row_number", "exact -> int32 immutable");
    }

    #[test]
    fn first_value() {
        test_window_aggr_function(WindowFunction::FirstValue, "first_value", "any <1> -> <mirrors> immutable");
    }

    #[test]
    fn last_value() {
        test_window_aggr_function(WindowFunction::LastValue, "last_value", "any <1> -> <mirrors> immutable");
    }
}
