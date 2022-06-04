//! Error types.

use backtrace::Backtrace;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

pub mod macros;

/// The error type used by the optimizer and its components.
#[derive(Debug)]
pub enum OptimizerError {
    /// This error indicates that a function of the optimizer or its components has been called with an invalid argument.
    Argument(ArgumentError),
    /// This error indicates that one of internal invariants of the optimizer or its components has been violated.
    Internal(InternalError),
    /// This error indicates that a block of code has not been implemented.
    NotImplemented(String),
    /// This error indicates that the described feature is not supported.
    Unsupported(String),
}

impl OptimizerError {
    /// Creates an [argument error](OptimizerError::Argument).
    /// This method is a shorthand for `OptimizerError::Argument(ArgumentError::new(message))`.
    pub fn argument<T>(message: T) -> OptimizerError
    where
        T: Into<String>,
    {
        OptimizerError::Argument(ArgumentError::new(message))
    }

    /// Creates an [internal error](OptimizerError::Internal).
    /// This method is a shorthand for `OptimizerError::Internal(InternalError::new(message, None))`.
    pub fn internal<T>(message: T) -> OptimizerError
    where
        T: Into<String>,
    {
        OptimizerError::Internal(InternalError::new(message, None))
    }
}

impl Display for OptimizerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerError::Argument(err) => write!(f, "Argument error: {}", err),
            OptimizerError::Internal(err) => write!(f, "Internal error: {}", err),
            OptimizerError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            OptimizerError::Unsupported(msg) => write!(f, "Not supported: {}", msg),
        }
    }
}

impl Error for OptimizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            OptimizerError::Argument(_) => None,
            OptimizerError::Internal(InternalError { cause: Some(error), .. }) => Some(error),
            OptimizerError::Internal(_) => None,
            OptimizerError::NotImplemented(_) => None,
            OptimizerError::Unsupported(_) => None,
        }
    }
}

impl From<ArgumentError> for OptimizerError {
    fn from(err: ArgumentError) -> Self {
        OptimizerError::Argument(err)
    }
}

impl From<InternalError> for OptimizerError {
    fn from(err: InternalError) -> Self {
        OptimizerError::Internal(err)
    }
}

/// Argument error. See [OptimizerError::Argument].
#[derive(Debug)]
pub struct ArgumentError {
    message: String,
    backtrace: Backtrace,
}

impl ArgumentError {
    /// Creates a new instance of an [ArgumentError].
    pub fn new<T>(message: T) -> Self
    where
        T: Into<String>,
    {
        ArgumentError {
            message: message.into(),
            backtrace: Backtrace::new(),
        }
    }
}

impl Display for ArgumentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.message)
    }
}

/// Internal error. See [OptimizerError::Internal].
#[derive(Debug)]
pub struct InternalError {
    message: String,
    cause: Option<Box<OptimizerError>>,
    backtrace: Backtrace,
}

impl InternalError {
    /// Creates an instance of an [InternalError] with the given message and an optional cause.
    /// This method captures a backtrace.
    pub fn new<T>(message: T, err: Option<OptimizerError>) -> Self
    where
        T: Into<String>,
    {
        InternalError {
            message: message.into(),
            cause: err.map(Box::new),
            backtrace: Backtrace::new(),
        }
    }

    /// Creates an instance of an [InternalError] with the given message and cause.
    /// This method captures a backtrace.
    pub fn with_cause<T>(message: T, cause: OptimizerError) -> Self
    where
        T: Into<String>,
    {
        InternalError {
            message: message.into(),
            cause: Some(Box::new(cause)),
            backtrace: Backtrace::new(),
        }
    }
}

impl From<&str> for InternalError {
    fn from(message: &str) -> Self {
        InternalError::new(message, None)
    }
}

impl From<String> for InternalError {
    fn from(message: String) -> Self {
        InternalError::new(message, None)
    }
}

impl Display for InternalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(cause) = self.cause.as_ref() {
            write!(f, " caused by: {}", cause)?
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::error::{InternalError, OptimizerError};
    use std::error::Error;

    #[test]
    fn internal_error_source() {
        let source_error = OptimizerError::internal("err");
        let expected_source_error = format!("{}", source_error);

        let err = OptimizerError::Internal(InternalError::new("err", Some(source_error)));
        assert!(err.source().is_some(), "no source error");

        let actual_source_error = err.source().unwrap();
        assert_eq!(format!("{}", actual_source_error), expected_source_error, "source error")
    }

    #[test]
    fn internal_error_without_source() {
        let err = OptimizerError::Internal(InternalError::new("err", None));
        assert!(err.source().is_none())
    }
}
