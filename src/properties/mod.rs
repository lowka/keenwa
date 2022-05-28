//! Properties of an operator.
//!
//! Logical properties are shared by all expressions in a memo-group.
//! Physical properties are describe physical characteristics of the data (such as ordering).
//! These properties are required by some operators. For example MergeSortJoin requires its inputs to be ordered.

pub mod logical;
mod ordering;
pub mod physical;

pub use ordering::{derive::derive_input_orderings, OrderingChoice, OrderingColumn};
