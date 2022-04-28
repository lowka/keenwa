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
            FunctionSignatureBuilder::one_of(vec![vec![DataType::Int32]])
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

#[cfg(test)]
mod test {
    use crate::datatypes::DataType;
    use crate::operators::scalar::aggregates::AggregateFunction;
    use crate::test_function;

    fn test_aggregate_function(f: AggregateFunction, expected_name: &str, expected_signature: &str) {
        test_function!(AggregateFunction, &f, expected_name, expected_signature);
    }

    #[test]
    fn avg() {
        test_aggregate_function(AggregateFunction::Avg, "avg", "one of [int32] -> <mirrors> immutable");
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
        test_aggregate_function(AggregateFunction::Min, "min", "one of [int32] -> <mirrors> immutable");
    }

    #[test]
    fn max() {
        test_aggregate_function(AggregateFunction::Max, "max", "one of [int32] -> <mirrors> immutable");
    }

    #[test]
    fn sum() {
        test_aggregate_function(AggregateFunction::Sum, "sum", "one of [int32] -> <mirrors> immutable");
    }
}
