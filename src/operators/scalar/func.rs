use crate::datatypes::DataType;
use crate::error::OptimizerError;
use itertools::Itertools;
use std::fmt::{Debug, Display, Formatter};
use std::num::NonZeroUsize;
use std::sync::Arc;

/// A signature of a function.
#[derive(Debug)]
pub struct FunctionSignature {
    /// Arguments of the function.
    pub args: ArgumentList,
    /// A function that computes a return type.
    pub return_type: FunctionReturnType,
    /// [Volatility] of the function.
    pub volatility: Volatility,
}

/// ArgumentList describes with what arguments a function can be called.
#[derive(Debug)]
pub enum ArgumentList {
    /// A function can be called only with arguments of the given types.
    Exact(Vec<DataType>),
    /// A function can be called with one of the given argument lists.
    OneOf(Vec<ArgumentList>),
    /// A function with `n` arguments where each argument can be of an arbitrary type.
    Any(NonZeroUsize),
    /// A function can be called with arguments of the given types and any number of additional arguments of an arbitrary type.
    Variadic(Vec<DataType>),
}

/// Volatility describes whether evaluation of a function is dependent only on functions arguments
/// or depends on other factors as well.
#[derive(Debug)]
pub enum Volatility {
    /// Multiple evaluations of an immutable function with the same arguments always return the same result.
    Immutable,
    /// Multiple evaluation of a stable function with the same arguments return
    /// the same result within a query.
    Stable,
    /// Multiple evaluations of a volatile function with same arguments do not guarantee to produce the same result.
    Volatile,
}

/// Describes a return type of a function.
pub enum FunctionReturnType {
    /// A return type of a function is a concrete type.
    Concrete(DataType),
    /// A return type of a function mirrors the type of the first argument of that function.
    /// Useful for to define mathematical functions such as `min`, `max`, etc.
    Mirror,
    /// A return type of a function depends on arguments of that function.
    Expr(ReturnTypeExpr),
}

impl Debug for FunctionReturnType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionReturnType::Concrete(tpe) => f.debug_tuple("Concrete").field(tpe).finish(),
            FunctionReturnType::Mirror => f.debug_tuple("Mirror").finish(),
            FunctionReturnType::Expr(_) => f.debug_tuple("Expr").finish(),
        }
    }
}

/// Computes the return type of a function given the types of its arguments.
pub type ReturnTypeExpr = Arc<dyn Fn(&[DataType]) -> Result<DataType, OptimizerError> + Send + Sync>;

/// A builder to create [function signatures](FunctionSignature).
pub struct FunctionSignatureBuilder {
    args: ArgumentList,
    volatility: Volatility,
}

impl FunctionSignatureBuilder {
    /// Creates a builder for a function with arguments of the given types.
    pub fn exact(args: Vec<DataType>) -> FunctionSignatureBuilder {
        FunctionSignatureBuilder {
            args: ArgumentList::Exact(args),
            volatility: Volatility::Immutable,
        }
    }

    /// Creates a builder for a function with multiple argument lists. See [ArgumentList::OneOf].
    pub fn one_of(args: Vec<Vec<DataType>>) -> FunctionSignatureBuilder {
        FunctionSignatureBuilder {
            args: ArgumentList::OneOf(args.into_iter().map(ArgumentList::Exact).collect()),
            volatility: Volatility::Immutable,
        }
    }

    /// Creates a builder for a function with `num` arguments of an arbitrary type. See [ArgumentList::Any].
    pub fn any(num: NonZeroUsize) -> FunctionSignatureBuilder {
        FunctionSignatureBuilder {
            args: ArgumentList::Any(num),
            volatility: Volatility::Immutable,
        }
    }

    /// Creates a builder for a function with variable number of arguments. See [ArgumentList::Variadic].
    pub fn variadic(args: Vec<DataType>) -> FunctionSignatureBuilder {
        FunctionSignatureBuilder {
            args: ArgumentList::Variadic(args),
            volatility: Volatility::Immutable,
        }
    }

    /// Set volatility of this function to [stable](Volatility::Stable).
    pub fn stable(mut self) -> FunctionSignatureBuilder {
        self.volatility = Volatility::Stable;
        self
    }

    /// Set volatility of this function to [volatile](Volatility::Volatile).
    pub fn volatile(mut self) -> FunctionSignatureBuilder {
        self.volatility = Volatility::Volatile;
        self
    }

    /// Builds a function with the given return type. See [FunctionReturnType::Concrete].
    pub fn return_type(self, tpe: DataType) -> FunctionSignature {
        FunctionSignature {
            args: self.args,
            return_type: FunctionReturnType::Concrete(tpe),
            volatility: self.volatility,
        }
    }

    /// Builds a function with a return type that is the same as the type of its first argument.
    /// See [FunctionReturnType::Mirror].
    pub fn return_mirrors(self) -> FunctionSignature {
        FunctionSignature {
            args: self.args,
            return_type: FunctionReturnType::Mirror,
            volatility: self.volatility,
        }
    }

    /// Builds a function with a return type computed by the given expression.
    /// See [ReturnTypeExpr].
    pub fn return_expr(self, expr: ReturnTypeExpr) -> FunctionSignature {
        FunctionSignature {
            args: self.args,
            return_type: FunctionReturnType::Expr(expr),
            volatility: self.volatility,
        }
    }
}

/// Checks whether a function with the given signature can be called with the given arguments.
pub fn verify_function_arguments(
    signature: &FunctionSignature,
    arg_types: &[DataType],
) -> Result<bool, OptimizerError> {
    let matches = match &signature.args {
        ArgumentList::Exact(exact) => exact == arg_types,
        ArgumentList::OneOf(args) => args.iter().any(|a| matches!(a, ArgumentList::Exact(exact) if exact == arg_types)),
        ArgumentList::Any(num) => num.get() == arg_types.len(),
        ArgumentList::Variadic(args) => args.len() <= arg_types.len() && args == &arg_types[0..args.len()],
    };

    Ok(matches)
}

/// Returns the return type of this function if it is called with arguments of the given types.
pub fn get_return_type<T>(
    func: &T,
    signature: &FunctionSignature,
    arg_types: &[DataType],
) -> Result<DataType, OptimizerError>
where
    T: Display + ?Sized,
{
    let matches = verify_function_arguments(signature, arg_types)?;
    if matches {
        match &signature.return_type {
            FunctionReturnType::Concrete(tpe) => Ok(tpe.clone()),
            FunctionReturnType::Mirror => Ok(arg_types[0].clone()),
            FunctionReturnType::Expr(expr) => (expr)(arg_types),
        }
    } else {
        Err(OptimizerError::argument(format!("No function {}({})", func, arg_types.iter().join(", "))))
    }
}

#[cfg(test)]
mod test {
    use crate::datatypes::DataType;
    use crate::error::OptimizerError;
    use crate::operators::scalar::func::{
        get_return_type, verify_function_arguments, FunctionSignature, FunctionSignatureBuilder,
    };
    use crate::operators::scalar::funcs::testing::signature_to_test_string;
    use itertools::Itertools;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    #[test]
    fn test_verify_exact_empty_arg_list() {
        let signature = FunctionSignatureBuilder::exact(vec![]).return_type(DataType::String);
        expect_signature_accepts_args(&signature, &[]);
    }

    #[test]
    fn test_verify_empty_signature_rejects_non_empty_arg_list() {
        let signature = FunctionSignatureBuilder::exact(vec![]).return_type(DataType::String);

        expect_signature_does_not_accept_args(&signature, &[DataType::Int32]);
    }

    #[test]
    fn test_verify_exact_nary_arg_list() {
        let signature =
            FunctionSignatureBuilder::exact(vec![DataType::Int32, DataType::Int32]).return_type(DataType::String);

        expect_signature_accepts_args(&signature, &[DataType::Int32, DataType::Int32]);
    }

    #[test]
    fn test_nary_signature_rejects_not_matching_arg_types() {
        let signature =
            FunctionSignatureBuilder::exact(vec![DataType::String, DataType::Int32]).return_type(DataType::String);

        expect_signature_does_not_accept_args(&signature, &[DataType::Bool, DataType::Int32]);
    }

    #[test]
    fn test_nary_signature_rejects_not_matching_arg_list_length() {
        let signature =
            FunctionSignatureBuilder::exact(vec![DataType::String, DataType::Int32]).return_type(DataType::String);

        expect_signature_does_not_accept_args(&signature, &[DataType::String, DataType::Int32, DataType::Int32]);
    }

    #[test]
    fn test_verify_any_arg_signature_non_empty_arg_list() {
        let num_args = NonZeroUsize::new(1).unwrap();
        let signature = FunctionSignatureBuilder::any(num_args).return_type(DataType::Int32);
        expect_signature_accepts_args(&signature, &[DataType::Int32])
    }

    #[test]
    fn test_any_arg_signature_reject_arg_list_of_non_matching_len() {
        let num_args = NonZeroUsize::new(1).unwrap();
        let signature = FunctionSignatureBuilder::any(num_args).return_type(DataType::Int32);
        expect_signature_does_not_accept_args(&signature, &[]);
        expect_signature_does_not_accept_args(&signature, &[DataType::Int32, DataType::String]);
    }

    #[test]
    fn test_verify_vararg_signature_no_args() {
        let signature = FunctionSignatureBuilder::variadic(vec![]).return_type(DataType::Int32);

        expect_signature_accepts_args(&signature, &[]);
        expect_signature_accepts_args(&signature, &[DataType::String]);
        expect_signature_accepts_args(&signature, &[DataType::String, DataType::Int32]);
    }

    #[test]
    fn test_verify_vararg_signature_some_args() {
        let signature =
            FunctionSignatureBuilder::variadic(vec![DataType::String, DataType::Bool]).return_type(DataType::Int32);

        expect_signature_does_not_accept_args(&signature, &[DataType::Int32]);
        expect_signature_does_not_accept_args(&signature, &[DataType::String]);
        expect_signature_does_not_accept_args(&signature, &[DataType::Int32, DataType::Bool]);
        expect_signature_does_not_accept_args(&signature, &[DataType::String, DataType::String]);

        expect_signature_accepts_args(&signature, &[DataType::String, DataType::Bool]);
        expect_signature_accepts_args(&signature, &[DataType::String, DataType::Bool, DataType::Int32]);
    }

    #[test]
    fn test_return_type_fails_when_arguments_do_not_match() {
        let signature = FunctionSignatureBuilder::exact(vec![]).return_type(DataType::Int32);
        let result = get_return_type("test_func", &signature, &[DataType::String])
            .map_err(|e| format!("{}", e))
            .map(|r| format!("{}", r));

        assert_eq!(result, Err("Argument error: No function test_func(String)".to_string()));
    }

    #[test]
    fn test_return_type_for_a_function_with_concrete_return_type() {
        let signature = FunctionSignatureBuilder::exact(vec![]).return_type(DataType::Int32);
        let result = get_return_type("test_func", &signature, &[]).unwrap();
        assert_eq!(result, DataType::Int32, "return type");
    }

    #[test]
    fn test_return_type_for_a_function_with_mirror_return_type() {
        for tpe in [DataType::Bool, DataType::Int32, DataType::String] {
            let signature = FunctionSignatureBuilder::exact(vec![tpe.clone()]).return_mirrors();
            let result = get_return_type("test_func", &signature, &[tpe.clone()]).unwrap();
            assert_eq!(result, tpe, "return type");
        }
    }

    #[test]
    fn test_return_type_for_a_function_with_return_type_expr() {
        for tpe in [DataType::Bool, DataType::Int32, DataType::String] {
            let ret_expr = |_args: &[DataType]| Ok(DataType::Null);
            let signature = FunctionSignatureBuilder::exact(vec![tpe.clone()]).return_expr(Arc::new(ret_expr));
            let result = get_return_type("test_func", &signature, &[tpe.clone()]).unwrap();
            assert_eq!(result, DataType::Null, "return type");
        }
    }

    fn expect_signature_accepts_args(signature: &FunctionSignature, args: &[DataType]) {
        let r = verify_function_arguments(signature, args);

        assert_signature_matches(signature, args, true, r);
    }

    fn expect_signature_does_not_accept_args(signature: &FunctionSignature, args: &[DataType]) {
        let r = verify_function_arguments(signature, args);

        assert_signature_matches(signature, args, false, r);
    }

    fn assert_signature_matches(
        signature: &FunctionSignature,
        args: &[DataType],
        matches: bool,
        result: Result<bool, OptimizerError>,
    ) {
        let message = if matches { "must accept" } else { "must not accept" };
        assert_eq!(
            Some(&matches),
            result.as_ref().ok(),
            "<{}> {} the arguments: [{}]. Result: {:?}",
            signature_to_test_string(signature),
            message,
            args.iter().join(", "),
            result
        )
    }
}
