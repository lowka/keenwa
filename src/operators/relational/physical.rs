//! Physical expressions supported by the optimizer.

use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::join::{get_join_columns_pair, JoinCondition, JoinOn, JoinType};
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::RequiredProperties;
use crate::properties::OrderingChoice;

// TODO: Docs
/// A physical expression represents an algorithm that can be used to implement a [logical expression].
///
/// [logical expression]: crate::operators::relational::logical::LogicalExpr
#[derive(Debug, Clone)]
pub enum PhysicalExpr {
    Projection(Projection),
    Select(Select),
    HashAggregate(HashAggregate),
    HashJoin(HashJoin),
    MergeSortJoin(MergeSortJoin),
    NestedLoopJoin(NestedLoopJoin),
    Scan(Scan),
    IndexScan(IndexScan),
    Sort(Sort),
    Append(Append),
    Unique(Unique),
    HashedSetOp(HashedSetOp),
    Limit(Limit),
    Offset(Offset),
    Values(Values),
    /// Relation that produces no rows.
    Empty(Empty),
}

impl PhysicalExpr {
    pub(crate) fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        match self {
            PhysicalExpr::Projection(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Select(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::HashAggregate(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::HashJoin(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::MergeSortJoin(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::NestedLoopJoin(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Scan(_) => {}
            PhysicalExpr::IndexScan(_) => {}
            PhysicalExpr::Sort(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Unique(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Append(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::HashedSetOp(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Limit(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Offset(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Values(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Empty(_) => {}
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

    pub fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        match self {
            PhysicalExpr::Projection(expr) => expr.build_required_properties(),
            PhysicalExpr::Select(expr) => expr.build_required_properties(),
            PhysicalExpr::HashAggregate(expr) => expr.build_required_properties(),
            PhysicalExpr::HashJoin(expr) => expr.build_required_properties(),
            PhysicalExpr::MergeSortJoin(expr) => expr.build_required_properties(),
            PhysicalExpr::NestedLoopJoin(expr) => expr.build_required_properties(),
            PhysicalExpr::Scan(expr) => expr.build_required_properties(),
            PhysicalExpr::IndexScan(expr) => expr.build_required_properties(),
            PhysicalExpr::Sort(expr) => expr.build_required_properties(),
            PhysicalExpr::Unique(expr) => expr.build_required_properties(),
            PhysicalExpr::Append(expr) => expr.build_required_properties(),
            PhysicalExpr::HashedSetOp(expr) => expr.build_required_properties(),
            PhysicalExpr::Limit(expr) => expr.build_required_properties(),
            PhysicalExpr::Offset(expr) => expr.build_required_properties(),
            PhysicalExpr::Values(expr) => expr.build_required_properties(),
            PhysicalExpr::Empty(expr) => expr.build_required_properties(),
        }
    }

    pub(crate) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            PhysicalExpr::Projection(expr) => expr.format_expr(f),
            PhysicalExpr::Select(expr) => expr.format_expr(f),
            PhysicalExpr::HashJoin(expr) => expr.format_expr(f),
            PhysicalExpr::MergeSortJoin(expr) => expr.format_expr(f),
            PhysicalExpr::NestedLoopJoin(expr) => expr.format_expr(f),
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

#[derive(Debug, Clone)]
pub struct Projection {
    pub input: RelNode,
    pub exprs: Vec<ScalarNode>,
    pub columns: Vec<ColumnId>,
}

impl Projection {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
        for expr in self.exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr);
        }
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(1 + self.exprs.len(), "Projection");
        Projection {
            input: inputs.rel_node(),
            exprs: inputs.scalar_nodes(self.exprs.len()),
            columns: self.columns.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        1 + self.exprs.len()
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        let num_exprs = self.exprs.len();
        match i {
            0 => Some(self.input.mexpr()),
            _ if i >= 1 && i < num_exprs + 1 => {
                let expr = &self.exprs[i - 1];
                Some(expr.mexpr())
            }
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Projection");
        f.write_expr("input", &self.input);
        f.write_exprs("exprs", self.exprs.iter());
        f.write_values("cols", &self.columns);
    }
}

#[derive(Debug, Clone)]
pub struct Select {
    pub input: RelNode,
    pub filter: Option<ScalarNode>,
}

impl Select {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
        visitor.visit_opt_scalar(expr_ctx, self.filter.as_ref());
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        let num_opt = self.filter.as_ref().map(|_| 1).unwrap_or_default();
        inputs.expect_len(1 + num_opt, "Select");

        Select {
            input: inputs.rel_node(),
            filter: self.filter.as_ref().map(|_| inputs.scalar_node()),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        1 + self.filter.as_ref().map(|_| 1).unwrap_or_default()
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.input.mexpr()),
            1 if self.filter.is_some() => {
                let filter = self.filter.as_ref().unwrap();
                Some(filter.mexpr())
            }
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Select");
        f.write_expr("input", &self.input);
        f.write_expr_if_present("filter", self.filter.as_ref())
    }
}

#[derive(Debug, Clone)]
pub struct HashAggregate {
    pub input: RelNode,
    pub aggr_exprs: Vec<ScalarNode>,
    pub group_exprs: Vec<ScalarNode>,
    pub having: Option<ScalarNode>,
    pub columns: Vec<ColumnId>,
}

impl HashAggregate {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
        for expr in self.aggr_exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr);
        }
        for expr in self.group_exprs.iter() {
            visitor.visit_scalar(expr_ctx, expr);
        }
        visitor.visit_opt_scalar(expr_ctx, self.having.as_ref());
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "HashAggregate");

        HashAggregate {
            input: inputs.rel_node(),
            aggr_exprs: inputs.scalar_nodes(self.aggr_exprs.len()),
            group_exprs: inputs.scalar_nodes(self.group_exprs.len()),
            having: self.having.as_ref().map(|_| inputs.scalar_node()),
            columns: self.columns.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        1 + self.aggr_exprs.len() + self.group_exprs.len() + self.having.as_ref().map(|_| 1).unwrap_or_default()
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        let num_aggr_exprs = self.aggr_exprs.len();
        let num_group_exprs = self.group_exprs.len();
        match i {
            0 => Some(self.input.mexpr()),
            _ if i >= 1 && i < 1 + num_aggr_exprs => {
                let expr = &self.aggr_exprs[i - 1];
                Some(expr.mexpr())
            }
            _ if i >= num_aggr_exprs && i < 1 + num_aggr_exprs + num_group_exprs => {
                let expr = &self.group_exprs[i - num_aggr_exprs - 1];
                Some(expr.mexpr())
            }
            _ if i > num_aggr_exprs + num_group_exprs => self.having.as_ref().map(|e| e.mexpr()),
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("HashAggregate");
        f.write_expr("input", &self.input);
        f.write_exprs("aggr_exprs", self.aggr_exprs.iter());
        f.write_exprs("group_exprs", self.group_exprs.iter());
        f.write_expr_if_present("having", self.having.as_ref());
        f.write_values("cols", &self.columns);
    }
}

#[derive(Debug, Clone)]
pub struct HashJoin {
    pub join_type: JoinType,
    pub left: RelNode,
    pub right: RelNode,
    pub condition: JoinCondition,
}

impl HashJoin {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
        visitor.visit_join_condition(expr_ctx, &self.condition);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        let num_opt = match &self.condition {
            JoinCondition::Using(_) => 0,
            JoinCondition::On(_) => 1,
        };
        inputs.expect_len(2 + num_opt, "HashJoin");

        HashJoin {
            join_type: self.join_type.clone(),
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            condition: match &self.condition {
                JoinCondition::Using(_) => self.condition.clone(),
                JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node())),
            },
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        let num_opt = match &self.condition {
            JoinCondition::Using(_) => 0,
            JoinCondition::On(_) => 1,
        };
        2 + num_opt
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.left.mexpr()),
            1 => Some(self.right.mexpr()),
            2 => match &self.condition {
                JoinCondition::Using(_) => None,
                JoinCondition::On(on) => Some(on.expr.mexpr()),
            },
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("HashJoin");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("type", &self.join_type);
        match &self.condition {
            JoinCondition::Using(using) => f.write_value("using", using),
            JoinCondition::On(on) => f.write_value("on", on),
        };
    }
}

#[derive(Debug, Clone)]
pub struct MergeSortJoin {
    pub join_type: JoinType,
    pub left: RelNode,
    pub right: RelNode,
    pub condition: JoinCondition,
}

impl MergeSortJoin {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
        visitor.visit_join_condition(expr_ctx, &self.condition);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        let num_opt = match &self.condition {
            JoinCondition::Using(_) => 0,
            JoinCondition::On(_) => 1,
        };
        inputs.expect_len(2 + num_opt, "MergeSortJoin");

        MergeSortJoin {
            join_type: self.join_type.clone(),
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            condition: match &self.condition {
                JoinCondition::Using(_) => self.condition.clone(),
                JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node())),
            },
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        match get_join_columns_pair(&self.left, &self.right, &self.condition) {
            Some((left, right)) if !left.is_empty() && !right.is_empty() => {
                let left_ordering = RequiredProperties::new_with_ordering(OrderingChoice::from_columns(left));
                let right_ordering = RequiredProperties::new_with_ordering(OrderingChoice::from_columns(right));

                Some(vec![Some(left_ordering), Some(right_ordering)])
            }
            _ => None,
        }
    }

    fn num_children(&self) -> usize {
        let num_opt = match &self.condition {
            JoinCondition::Using(_) => 0,
            JoinCondition::On(_) => 1,
        };
        2 + num_opt
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.left.mexpr()),
            1 => Some(self.right.mexpr()),
            2 => match &self.condition {
                JoinCondition::Using(_) => None,
                JoinCondition::On(on) => Some(on.expr.mexpr()),
            },
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("MergeSortJoin");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("type", &self.join_type);
        match &self.condition {
            JoinCondition::Using(using) => f.write_value("using", using),
            JoinCondition::On(on) => f.write_value("on", on),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NestedLoopJoin {
    pub join_type: JoinType,
    pub left: RelNode,
    pub right: RelNode,
    pub condition: Option<ScalarNode>,
}

impl NestedLoopJoin {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
        visitor.visit_opt_scalar(expr_ctx, self.condition.as_ref());
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        let num_opt = self.condition.as_ref().map(|_| 1).unwrap_or_default();
        inputs.expect_len(2 + num_opt, "NestedLoopJoin");

        NestedLoopJoin {
            join_type: self.join_type.clone(),
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            condition: self.condition.as_ref().map(|_| inputs.scalar_node()),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        2 + self.condition.as_ref().map(|_| 1).unwrap_or_default()
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.left.mexpr()),
            1 => Some(self.right.mexpr()),
            2 if self.condition.is_some() => {
                let condition = self.condition.as_ref().unwrap();
                Some(condition.mexpr())
            }
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("NestedLoopJoin");
        f.write_value("type", &self.join_type);
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        if let Some(condition) = self.condition.as_ref() {
            f.write_expr("condition", condition);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scan {
    pub source: String,
    pub columns: Vec<ColumnId>,
}

impl Scan {
    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(0, "Scan");

        Scan {
            source: self.source.clone(),
            columns: self.columns.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        0
    }

    fn get_child(&self, _i: usize) -> Option<&Operator> {
        None
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Scan");
        f.write_source(self.source.as_ref());
        f.write_values("cols", self.columns.as_slice())
    }
}

#[derive(Debug, Clone)]
pub struct IndexScan {
    pub source: String,
    pub columns: Vec<ColumnId>,
    pub ordering: Option<OrderingChoice>,
}

impl IndexScan {
    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(0, "IndexScan");

        IndexScan {
            source: self.source.clone(),
            columns: self.columns.clone(),
            ordering: self.ordering.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        0
    }

    fn get_child(&self, _i: usize) -> Option<&Operator> {
        None
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("IndexScan");
        f.write_source(self.source.as_ref());
        f.write_values("cols", self.columns.as_slice())
    }
}

#[derive(Debug, Clone)]
pub struct Sort {
    pub input: RelNode,
    pub ordering: OrderingChoice,
}

impl Sort {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(1, "Sort");

        Sort {
            input: inputs.rel_node(),
            ordering: self.ordering.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        1
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        if i == 0 {
            Some(self.input.mexpr())
        } else {
            None
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Sort");
        f.write_expr("input", &self.input);
        f.write_values("ord", self.ordering.columns())
    }
}

#[derive(Debug, Clone)]
pub struct Append {
    pub left: RelNode,
    pub right: RelNode,
    pub columns: Vec<ColumnId>,
}

impl Append {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(2, "Append");

        Append {
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            columns: self.columns.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        2
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.left.mexpr()),
            1 => Some(self.right.mexpr()),
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Append");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_values("cols", &self.columns);
    }
}

#[derive(Debug, Clone)]
pub struct Unique {
    pub inputs: Vec<RelNode>,
    pub on_expr: Option<ScalarNode>,
    pub columns: Vec<ColumnId>,
}

impl Unique {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        for input in self.inputs.iter() {
            visitor.visit_rel(expr_ctx, input);
        }
        visitor.visit_opt_scalar(expr_ctx, self.on_expr.as_ref());
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        let num_children = self.num_children();
        inputs.expect_len(num_children, "Unique");

        Unique {
            inputs: inputs.rel_nodes(self.inputs.len()),
            on_expr: self.on_expr.as_ref().map(|_| inputs.scalar_node()),
            columns: self.columns.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        let mut requirements: Vec<_> = self
            .inputs
            .iter()
            .map(|input| {
                let columns = input.props().logical().output_columns().to_vec();
                Some(RequiredProperties::new_with_ordering(OrderingChoice::from_columns(columns)))
            })
            .collect();
        if self.on_expr.is_some() {
            requirements.push(None);
        }
        Some(requirements)
    }

    fn num_children(&self) -> usize {
        self.inputs.len() + self.on_expr.as_ref().map(|_| 1).unwrap_or_default()
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        let num_input = self.inputs.len();
        match num_input {
            _ if i < num_input => self.inputs.get(i).map(|input| input.mexpr()),
            _ if i >= num_input && i < self.num_children() => self.on_expr.as_ref().map(|expr| expr.mexpr()),
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Unique");
        let num_input = self.inputs.len();
        match num_input {
            1 => f.write_expr("input", &self.inputs[0]),
            2 => {
                f.write_expr("left", &self.inputs[0]);
                f.write_expr("right", &self.inputs[1]);
            }
            _ => f.write_exprs("inputs", self.inputs.iter()),
        }
        f.write_expr_if_present("on", self.on_expr.as_ref());
        f.write_values("cols", &self.columns);
    }
}

#[derive(Debug, Clone)]
pub struct HashedSetOp {
    pub left: RelNode,
    pub right: RelNode,
    /// Whether this is an INTERSECT or EXCEPT operator.
    pub intersect: bool,
    /// If `true` this an INTERSECT ALL/EXCEPT ALL operator.
    pub all: bool,
    /// Output columns produced by a set operator.
    pub columns: Vec<ColumnId>,
}

impl HashedSetOp {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(2, "HashedSetOp");

        HashedSetOp {
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            intersect: self.intersect,
            all: self.all,
            columns: self.columns.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        2
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.left.mexpr()),
            1 => Some(self.right.mexpr()),
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("HashedSetOp");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("intersect", &self.intersect);
        f.write_value("all", self.all);
        f.write_values("cols", &self.columns)
    }
}

/// Limits the output of the input operator to produce no more than the specified number of rows.
#[derive(Debug, Clone)]
pub struct Limit {
    pub input: RelNode,
    pub rows: usize,
}

impl Limit {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "Limit");
        Limit {
            input: inputs.rel_node(),
            rows: self.rows,
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        1
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.input.mexpr()),
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Limit");
        f.write_expr("input", &self.input);
        f.write_value("rows", self.rows)
    }
}

/// Skips the first `rows` number of items produced by the input operator.
#[derive(Debug, Clone)]
pub struct Offset {
    pub input: RelNode,
    pub rows: usize,
}

impl Offset {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.num_children(), "Offset");
        Offset {
            input: inputs.rel_node(),
            rows: self.rows,
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        1
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        match i {
            0 => Some(self.input.mexpr()),
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Offset");
        f.write_expr("input", &self.input);
        f.write_value("rows", self.rows)
    }
}

#[derive(Debug, Clone)]
pub struct Values {
    /// A list of tuples where each tuple is a row.
    pub values: Vec<ScalarNode>,
    pub columns: Vec<ColumnId>,
}

impl Values {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        for expr in self.values.iter() {
            visitor.visit_scalar(expr_ctx, expr);
        }
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(self.values.len(), "Values");
        Values {
            values: inputs.scalar_nodes(self.values.len()),
            columns: self.columns.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        self.values.len()
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        self.values.get(i).map(|row| row.mexpr())
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Values");
        f.write_exprs("values", self.values.iter());
        f.write_values("cols", &self.columns);
    }
}

#[derive(Debug, Clone)]
pub struct Empty {
    pub return_one_row: bool,
}

impl Empty {
    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(0, "Empty");
        Empty {
            return_one_row: self.return_one_row,
        }
    }

    fn build_required_properties(&self) -> Option<Vec<Option<RequiredProperties>>> {
        None
    }

    fn num_children(&self) -> usize {
        0
    }

    fn get_child(&self, _i: usize) -> Option<&Operator> {
        None
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Empty");
        f.write_value("return_one_row", &self.return_one_row)
    }
}
