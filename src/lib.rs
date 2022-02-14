pub mod catalog;
pub mod cost;
pub mod datatypes;
pub mod error;
pub mod memo;
pub mod meta;
pub mod operators;
pub mod optimizer;
pub mod properties;
pub mod rules;
mod sql;
pub mod statistics;
#[cfg(test)]
pub mod testing;
#[cfg(test)]
mod tests;

// TODO: Document panics
