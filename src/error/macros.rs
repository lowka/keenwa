//TODO: Reduce duplication in both docs and macro code.

//!A collection of helper macros to return errors.

/// This macro indicates that feature is not supported.
/// - Unconditional error:
/// ```
///   # use keenwa::error::OptimizerError;
///   # use keenwa::not_supported;
///
///   // Unconditional error
///   fn some_feature() -> Result<(), OptimizerError> {
///      not_supported!("This feature is not supported");
///      Ok(())
///   }
///   assert!(matches!(some_feature(), Err(OptimizerError::Unsupported { .. })));
/// ```
/// - Conditional error:
/// ```
///   # use keenwa::error::OptimizerError;
///   # use keenwa::not_supported;
///
///   fn partially_supported_feature(f: bool) -> Result<(), OptimizerError> {
///      not_supported!(f, "This feature is not supported when f is true");
///      Ok(())
///   }
///
///   assert!(matches!(partially_supported_feature(true), Err(OptimizerError::Unsupported { .. })));
///   assert!(partially_supported_feature(false).is_ok());
/// ```
/// - Conditional error (an error message contains arguments):
/// ```
///   # use keenwa::error::OptimizerError;
///   # use keenwa::not_supported;
///
///   fn with_error_message(f: bool, arg: i32) -> Result<(), OptimizerError> {
///     not_supported!(f, "This feature is not supported: f={}, arg={}", f, arg);
///     Ok(())
///   }
///
///   match with_error_message(true, 1) {
///     Err(OptimizerError::Unsupported(m)) => assert!(m.contains("f=true, arg=1")),
///     _ => assert!(false, "Test failure")
///   }
///   assert!(with_error_message(false, 1).is_ok());
/// ```
#[macro_export]
macro_rules! not_supported {
    //unconditional error
    ($message:tt) => {{
        return core::result::Result::Err($crate::error::OptimizerError::Unsupported(format!("{}", $message)));
    }};
    ($condition:expr, $message:tt) => {{
        if $condition {
            return core::result::Result::Err($crate::error::OptimizerError::Unsupported(format!("{}", $message)));
        }
    }};
    ($condition:expr, $message:tt, $($arg:expr),* $(,)?) => {{
        if $condition {
            let message = std::fmt::format(format_args!($message, $($arg),*));
            return core::result::Result::Err($crate::error::OptimizerError::Unsupported(message));
        }
    }};
}

/// This macro indicates that feature is not implemented.
/// - Unconditional error:
/// ```
///   # use keenwa::error::OptimizerError;
///   # use keenwa::not_implemented;
///
///   // Unconditional error
///   fn some_feature() -> Result<(), OptimizerError> {
///      not_implemented!("This feature is not implemented");
///      Ok(())
///   }
///   assert!(matches!(some_feature(), Err(OptimizerError::NotImplemented { .. })));
/// ```
/// - Conditional error:
/// ```
///   # use keenwa::error::OptimizerError;
///   # use keenwa::not_implemented;
///
///   fn partially_implemented_feature(f: bool) -> Result<(), OptimizerError> {
///      not_implemented!(f, "This feature is not implemented when f is true");
///      Ok(())
///   }
///
///   assert!(matches!(partially_implemented_feature(true), Err(OptimizerError::NotImplemented { .. })));
///   assert!(partially_implemented_feature(false).is_ok());
/// ```
/// - Conditional error (an error message contains arguments):
/// ```
///   # use keenwa::error::OptimizerError;
///   # use keenwa::not_implemented;
///
///   fn with_error_message(f: bool, arg: i32) -> Result<(), OptimizerError> {
///     not_implemented!(f, "This feature is not implemented: f={}, arg={}", f, arg);
///     Ok(())
///   }
///
///   match with_error_message(true, 1) {
///     Err(OptimizerError::NotImplemented(m)) => assert!(m.contains("f=true, arg=1")),
///     _ => assert!(false, "Test failure")
///   }
///   assert!(with_error_message(false, 1).is_ok());
/// ```
#[macro_export]
macro_rules! not_implemented {
    //unconditional error
    ($message:tt) => {{
        return core::result::Result::Err($crate::error::OptimizerError::NotImplemented(format!("{}", $message)));
    }};
    // condition -> message
    ($cond:expr, $message:tt) => {{
        if $cond {
            return core::result::Result::Err($crate::error::OptimizerError::NotImplemented(format!("{}", $message)));
        }
    }};
    ($condition:expr, $message:tt, $($arg:expr),+ $(,)?) => {{
        if $condition {
            let message = std::fmt::format(format_args!($message, $($arg),+));
            return core::result::Result::Err($crate::error::OptimizerError::NotImplemented(message));
        }
    }};
}

pub use not_implemented;
pub use not_supported;

#[allow(unused_braces)]
#[allow(unreachable_code)]
#[allow(unused)]
mod tests {
    use crate::error::OptimizerError;

    macro_rules! error_macro_tests {
        ($err_macro:tt, $test_name:ident) => {
            fn $test_name() -> core::result::Result<(), $crate::error::OptimizerError> {
                $err_macro!("Unconditional. Message");

                $err_macro!(false, "Condition. Message");
                $err_macro!(false, "Condition. Message with one placeholder {}", 1);
                $err_macro!(false, "Condition. Message with multiple placeholders {} {} {}", 1, 2, 3);

                $err_macro!({ false }, "{{Condition}}.  Message");
                $err_macro!({ false }, "{{Condition}}. Message with one placeholder {}", 1);
                $err_macro!(
                    { false },
                    "{{Condition}}. Message with placeholder multiple placeholders {} {} {}",
                    1,
                    2,
                    3
                );

                Ok(())
            }
        };
    }

    error_macro_tests!(not_supported, not_supported_test_cases);
    error_macro_tests!(not_implemented, not_implemented_test_cases);
}
