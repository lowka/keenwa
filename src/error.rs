use std::error::Error;
use std::fmt::{Display, Formatter};

/// The error type used by the optimizer and its components.
#[derive(Debug)]
pub enum OptimizerError {
    /// This error indicates that a function of the optimizer or its components has been called with an invalid argument.
    Argument(String),
    /// This error indicates that one of internal invariants of the optimizer or its components has been violated.
    Internal(String),
    /// This error indicates that a block of code has not been implemented.
    NotImplemented(String),
    /// This error indicates that the described feature is not supported.
    Unsupported(String),
}

impl Display for OptimizerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerError::Argument(msg) => write!(f, "Argument error: {}", msg),
            OptimizerError::Internal(msg) => write!(f, "Internal error: {}", msg),
            OptimizerError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            OptimizerError::Unsupported(msg) => write!(f, "Not supported: {}", msg),
        }
    }
}

impl Error for OptimizerError {}
