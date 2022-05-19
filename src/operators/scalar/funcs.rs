use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::operators::scalar::func::{get_return_type, FunctionSignature, FunctionSignatureBuilder};
use std::convert::TryFrom;
use std::fmt::{Display, Formatter};

/// Build-in functions.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ScalarFunction {
    BitLength,
    CharacterLength,
    Concat,
    Lower,
    Position,
    ToHex,
    Replace,
    Upper,
}

impl ScalarFunction {
    /// Returns the return type of this function if it called with the given arguments.
    pub fn get_return_type(&self, arg_types: &[DataType]) -> Result<DataType, OptimizerError> {
        let signature = self.get_signature();
        get_return_type(self, &signature, arg_types)
    }

    fn get_signature(&self) -> FunctionSignature {
        fn arg(tpe: DataType) -> FunctionSignatureBuilder {
            FunctionSignatureBuilder::exact(vec![tpe])
        }
        fn args(args: Vec<DataType>) -> FunctionSignatureBuilder {
            FunctionSignatureBuilder::exact(args)
        }
        fn varargs(args: Vec<DataType>) -> FunctionSignatureBuilder {
            FunctionSignatureBuilder::variadic(args)
        }

        match self {
            ScalarFunction::BitLength => arg(DataType::String).return_type(DataType::Int32),
            ScalarFunction::CharacterLength => arg(DataType::String).return_type(DataType::Int32),
            ScalarFunction::Concat => varargs(vec![]).return_type(DataType::String),
            ScalarFunction::Lower => arg(DataType::String).return_type(DataType::String),
            ScalarFunction::Position => args(vec![DataType::String, DataType::String]).return_type(DataType::Int32),
            ScalarFunction::ToHex => arg(DataType::Int32).return_type(DataType::String),
            ScalarFunction::Replace => args(vec![DataType::String, DataType::String]).return_type(DataType::String),
            ScalarFunction::Upper => arg(DataType::String).return_type(DataType::String),
        }
    }
}

impl TryFrom<&str> for ScalarFunction {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "bit_length" => Ok(ScalarFunction::BitLength),
            "character_length" => Ok(ScalarFunction::CharacterLength),
            "concat" => Ok(ScalarFunction::Concat),
            "lower" => Ok(ScalarFunction::Lower),
            "position" => Ok(ScalarFunction::Position),
            "to_hex" => Ok(ScalarFunction::ToHex),
            "replace" => Ok(ScalarFunction::Replace),
            "upper" => Ok(ScalarFunction::Upper),
            _ => Err(()),
        }
    }
}

impl Display for ScalarFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarFunction::BitLength => write!(f, "bit_length"),
            ScalarFunction::CharacterLength => write!(f, "character_length"),
            ScalarFunction::Concat => write!(f, "concat"),
            ScalarFunction::Lower => write!(f, "lower"),
            ScalarFunction::Position => write!(f, "position"),
            ScalarFunction::ToHex => write!(f, "to_hex"),
            ScalarFunction::Replace => write!(f, "replace"),
            ScalarFunction::Upper => write!(f, "upper"),
        }
    }
}

#[cfg(test)]
pub mod testing {
    use crate::operators::scalar::func::{ArgumentList, FunctionReturnType, FunctionSignature, Volatility};
    use itertools::Itertools;

    #[macro_export]
    macro_rules! test_function {
        ($func_type:tt, $func_var:expr, $expected_name:expr, $expected_signature: expr) => {
            fn expect_from_str(expected: &$func_type, expected_str: &str) {
                use std::convert::TryFrom;

                let actual_str = format!("{}", expected);
                assert_eq!(actual_str, expected_str, "display and string representation do not match");
                assert_eq!(
                    Ok(expected),
                    $func_type::try_from(actual_str.as_str()).as_ref(),
                    "try_from conversion does not match"
                );
            }

            fn expect_signature(f: &$func_type, expected: &str) {
                use crate::operators::scalar::funcs::testing::signature_to_test_string;

                let signature = f.get_signature();
                let actual_signature = signature_to_test_string(&signature);
                assert_eq!(actual_signature, expected, "signature");
            }

            expect_from_str($func_var, $expected_name);
            expect_signature($func_var, $expected_signature);
        };
    }

    /// Converts the given function signature to a string representation that is used by tests.
    pub fn signature_to_test_string(signature: &FunctionSignature) -> String {
        let mut buf = String::new();

        match &signature.args {
            ArgumentList::Exact(args) => {
                buf.push_str("exact");
                if !args.is_empty() {
                    buf.push(' ');
                    buf.push_str(args.iter().join(", ").to_lowercase().as_str());
                }
            }
            ArgumentList::OneOf(args) => {
                buf.push_str("one of ");
                buf.push_str(
                    args.iter()
                        .map(|args| match args {
                            ArgumentList::Exact(args) => format!("[{}]", args.iter().join(", ").to_lowercase()),
                            ArgumentList::OneOf(_) => panic!("Nested argument lists of type OneOf are not supported"),
                            ArgumentList::Any(num) => format!("[any <{}>]", num),
                            ArgumentList::Variadic(args) => format!("[{}, ...]", args.iter().join(", ").to_lowercase()),
                        })
                        .join(", ")
                        .as_str(),
                );
            }
            ArgumentList::Any(num) => {
                buf.push_str(format!("any <{}>", num).as_str());
            }
            ArgumentList::Variadic(args) => {
                buf.push_str(format!("[{}, ...]", args.iter().join(", ").to_lowercase()).as_str());
            }
        }
        buf.push_str(" -> ");
        match &signature.return_type {
            FunctionReturnType::Concrete(tpe) => buf.push_str(format!("{}", tpe).to_lowercase().as_str()),
            FunctionReturnType::Mirror => buf.push_str("<mirrors>"),
            FunctionReturnType::Expr(_) => buf.push_str("<expr>"),
        }
        match &signature.volatility {
            Volatility::Immutable => buf.push_str(" immutable"),
            Volatility::Stable => buf.push_str(" stable"),
            Volatility::Volatile => buf.push_str(" volatile"),
        }
        buf
    }
}

#[cfg(test)]
mod test {
    use crate::operators::scalar::funcs::ScalarFunction;
    use crate::test_function;

    fn test_scalar_function(f: ScalarFunction, expected_name: &str, expected_signature: &str) {
        test_function!(ScalarFunction, &f, expected_name, expected_signature);
    }

    #[test]
    fn bit_length() {
        test_scalar_function(ScalarFunction::BitLength, "bit_length", "exact string -> int32 immutable");
    }

    #[test]
    fn character_length() {
        test_scalar_function(ScalarFunction::CharacterLength, "character_length", "exact string -> int32 immutable");
    }

    #[test]
    fn lower() {
        test_scalar_function(ScalarFunction::Lower, "lower", "exact string -> string immutable");
    }

    #[test]
    fn position() {
        test_scalar_function(ScalarFunction::Position, "position", "exact string, string -> int32 immutable");
    }

    #[test]
    fn to_hex() {
        test_scalar_function(ScalarFunction::ToHex, "to_hex", "exact int32 -> string immutable");
    }

    #[test]
    fn replace() {
        test_scalar_function(ScalarFunction::Replace, "replace", "exact string, string -> string immutable");
    }

    #[test]
    fn upper() {
        test_scalar_function(ScalarFunction::Upper, "upper", "exact string -> string immutable");
    }
}
