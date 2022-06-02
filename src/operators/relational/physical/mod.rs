//! Physical expressions supported by the optimizer.

use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;

use crate::error::OptimizerError;
pub use append::Append;
pub use empty::Empty;
pub use hash_aggregate::HashAggregate;
pub use hash_join::HashJoin;
pub use hash_set_op::HashedSetOp;
pub use index_scan::IndexScan;
pub use limit::Limit;
pub use merge_sort_join::MergeSortJoin;
pub use nested_loop_join::NestedLoopJoin;
pub use offset::Offset;
pub use projection::Projection;
pub use scan::Scan;
pub use select::Select;
pub use sort::Sort;
pub use streaming_aggregate::StreamingAggregate;
pub use unique::Unique;
pub use values::Values;

mod append;
mod empty;
mod hash_aggregate;
mod hash_join;
mod hash_set_op;
mod index_scan;
mod limit;
mod merge_sort_join;
mod nested_loop_join;
mod offset;
mod projection;
mod scan;
mod select;
mod sort;
mod streaming_aggregate;
mod unique;
mod values;

// TODO: Docs
/// A physical expression represents an algorithm that can be used to implement a [logical expression].
///
/// [logical expression]: crate::operators::relational::logical::LogicalExpr
#[derive(Debug, Clone)]
pub enum PhysicalExpr {
    /// Projection operator.
    Projection(Projection),
    /// Select operator.
    Select(Select),
    /// HashAggregate operator.
    HashAggregate(HashAggregate),
    /// HashJoin operator.
    HashJoin(HashJoin),
    /// MergeSortJoin operator.
    MergeSortJoin(MergeSortJoin),
    /// NestedLoopJoin operator.
    NestedLoopJoin(NestedLoopJoin),
    /// Streaming aggregate operator.
    StreamingAggregate(StreamingAggregate),
    /// Scan operator.
    Scan(Scan),
    /// IndexScan operator.
    IndexScan(IndexScan),
    /// Sort operator.
    Sort(Sort),
    /// Append operator.
    Append(Append),
    /// Unique operator.
    Unique(Unique),
    /// HashedSetOp operator.
    HashedSetOp(HashedSetOp),
    /// Limit operator.
    Limit(Limit),
    /// Offset operator.
    Offset(Offset),
    /// Values operator.
    Values(Values),
    /// Relation that produces no rows.
    Empty(Empty),
}

impl PhysicalExpr {
    pub(crate) fn copy_in<T>(
        &self,
        visitor: &mut OperatorCopyIn<T>,
        expr_ctx: &mut ExprContext<Operator>,
    ) -> Result<(), OptimizerError> {
        match self {
            PhysicalExpr::Projection(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Select(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::HashAggregate(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::HashJoin(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::MergeSortJoin(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::NestedLoopJoin(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::StreamingAggregate(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Scan(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::IndexScan(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Sort(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Unique(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Append(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::HashedSetOp(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Limit(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Offset(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Values(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Empty(expr) => expr.copy_in(visitor, expr_ctx),
        }
    }

    pub fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        match self {
            PhysicalExpr::Projection(expr) => PhysicalExpr::Projection(expr.with_new_inputs(inputs)),
            PhysicalExpr::Select(expr) => PhysicalExpr::Select(expr.with_new_inputs(inputs)),
            PhysicalExpr::HashAggregate(expr) => PhysicalExpr::HashAggregate(expr.with_new_inputs(inputs)),
            PhysicalExpr::HashJoin(expr) => PhysicalExpr::HashJoin(expr.with_new_inputs(inputs)),
            PhysicalExpr::MergeSortJoin(expr) => PhysicalExpr::MergeSortJoin(expr.with_new_inputs(inputs)),
            PhysicalExpr::NestedLoopJoin(expr) => PhysicalExpr::NestedLoopJoin(expr.with_new_inputs(inputs)),
            PhysicalExpr::StreamingAggregate(expr) => PhysicalExpr::StreamingAggregate(expr.with_new_inputs(inputs)),
            PhysicalExpr::Scan(expr) => PhysicalExpr::Scan(expr.with_new_inputs(inputs)),
            PhysicalExpr::IndexScan(expr) => PhysicalExpr::IndexScan(expr.with_new_inputs(inputs)),
            PhysicalExpr::Sort(expr) => PhysicalExpr::Sort(expr.with_new_inputs(inputs)),
            PhysicalExpr::Unique(expr) => PhysicalExpr::Unique(expr.with_new_inputs(inputs)),
            PhysicalExpr::Append(expr) => PhysicalExpr::Append(expr.with_new_inputs(inputs)),
            PhysicalExpr::HashedSetOp(expr) => PhysicalExpr::HashedSetOp(expr.with_new_inputs(inputs)),
            PhysicalExpr::Limit(expr) => PhysicalExpr::Limit(expr.with_new_inputs(inputs)),
            PhysicalExpr::Offset(expr) => PhysicalExpr::Offset(expr.with_new_inputs(inputs)),
            PhysicalExpr::Values(expr) => PhysicalExpr::Values(expr.with_new_inputs(inputs)),
            PhysicalExpr::Empty(expr) => PhysicalExpr::Empty(expr.with_new_inputs(inputs)),
        }
    }

    pub fn num_children(&self) -> usize {
        match self {
            PhysicalExpr::Projection(expr) => expr.num_children(),
            PhysicalExpr::Select(expr) => expr.num_children(),
            PhysicalExpr::HashAggregate(expr) => expr.num_children(),
            PhysicalExpr::HashJoin(expr) => expr.num_children(),
            PhysicalExpr::MergeSortJoin(expr) => expr.num_children(),
            PhysicalExpr::NestedLoopJoin(expr) => expr.num_children(),
            PhysicalExpr::StreamingAggregate(expr) => expr.num_children(),
            PhysicalExpr::Scan(expr) => expr.num_children(),
            PhysicalExpr::IndexScan(expr) => expr.num_children(),
            PhysicalExpr::Sort(expr) => expr.num_children(),
            PhysicalExpr::Unique(expr) => expr.num_children(),
            PhysicalExpr::Append(expr) => expr.num_children(),
            PhysicalExpr::HashedSetOp(expr) => expr.num_children(),
            PhysicalExpr::Limit(expr) => expr.num_children(),
            PhysicalExpr::Offset(expr) => expr.num_children(),
            PhysicalExpr::Values(expr) => expr.num_children(),
            PhysicalExpr::Empty(expr) => expr.num_children(),
        }
    }

    pub fn get_child(&self, i: usize) -> Option<&Operator> {
        match self {
            PhysicalExpr::Projection(expr) => expr.get_child(i),
            PhysicalExpr::Select(expr) => expr.get_child(i),
            PhysicalExpr::HashAggregate(expr) => expr.get_child(i),
            PhysicalExpr::HashJoin(expr) => expr.get_child(i),
            PhysicalExpr::MergeSortJoin(expr) => expr.get_child(i),
            PhysicalExpr::NestedLoopJoin(expr) => expr.get_child(i),
            PhysicalExpr::StreamingAggregate(expr) => expr.get_child(i),
            PhysicalExpr::Scan(expr) => expr.get_child(i),
            PhysicalExpr::IndexScan(expr) => expr.get_child(i),
            PhysicalExpr::Sort(expr) => expr.get_child(i),
            PhysicalExpr::Unique(expr) => expr.get_child(i),
            PhysicalExpr::Append(expr) => expr.get_child(i),
            PhysicalExpr::HashedSetOp(expr) => expr.get_child(i),
            PhysicalExpr::Limit(expr) => expr.get_child(i),
            PhysicalExpr::Offset(expr) => expr.get_child(i),
            PhysicalExpr::Values(expr) => expr.get_child(i),
            PhysicalExpr::Empty(expr) => expr.get_child(i),
        }
    }

    pub fn get_required_input_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        match self {
            PhysicalExpr::Projection(expr) => expr.get_required_input_properties(),
            PhysicalExpr::Select(expr) => expr.get_required_input_properties(),
            PhysicalExpr::HashAggregate(expr) => expr.get_required_input_properties(),
            PhysicalExpr::HashJoin(expr) => expr.get_required_input_properties(),
            PhysicalExpr::MergeSortJoin(expr) => expr.get_required_input_properties(),
            PhysicalExpr::NestedLoopJoin(expr) => expr.get_required_input_properties(),
            PhysicalExpr::StreamingAggregate(expr) => expr.get_required_input_properties(),
            PhysicalExpr::Scan(expr) => expr.get_required_input_properties(),
            PhysicalExpr::IndexScan(expr) => expr.get_required_input_properties(),
            PhysicalExpr::Sort(expr) => expr.get_required_input_properties(),
            PhysicalExpr::Unique(expr) => expr.get_required_input_properties(),
            PhysicalExpr::Append(expr) => expr.get_required_input_properties(),
            PhysicalExpr::HashedSetOp(expr) => expr.get_required_input_properties(),
            PhysicalExpr::Limit(expr) => expr.get_required_input_properties(),
            PhysicalExpr::Offset(expr) => expr.get_required_input_properties(),
            PhysicalExpr::Values(expr) => expr.get_required_input_properties(),
            PhysicalExpr::Empty(expr) => expr.get_required_input_properties(),
        }
    }

    pub fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            PhysicalExpr::Projection(expr) => expr.format_expr(f),
            PhysicalExpr::Select(expr) => expr.format_expr(f),
            PhysicalExpr::HashJoin(expr) => expr.format_expr(f),
            PhysicalExpr::MergeSortJoin(expr) => expr.format_expr(f),
            PhysicalExpr::NestedLoopJoin(expr) => expr.format_expr(f),
            PhysicalExpr::StreamingAggregate(expr) => expr.format_expr(f),
            PhysicalExpr::Scan(expr) => expr.format_expr(f),
            PhysicalExpr::IndexScan(expr) => expr.format_expr(f),
            PhysicalExpr::Sort(expr) => expr.format_expr(f),
            PhysicalExpr::HashAggregate(expr) => expr.format_expr(f),
            PhysicalExpr::Unique(expr) => expr.format_expr(f),
            PhysicalExpr::Append(expr) => expr.format_expr(f),
            PhysicalExpr::HashedSetOp(expr) => expr.format_expr(f),
            PhysicalExpr::Limit(expr) => expr.format_expr(f),
            PhysicalExpr::Offset(expr) => expr.format_expr(f),
            PhysicalExpr::Values(expr) => expr.format_expr(f),
            PhysicalExpr::Empty(expr) => expr.format_expr(f),
        }
    }
}
