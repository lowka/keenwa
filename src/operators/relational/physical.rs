use crate::memo::{ChildNodeRef, ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::join::{get_join_columns_pair, JoinCondition, JoinOn};
use crate::operators::relational::RelNode;
use crate::operators::scalar::ScalarNode;
use crate::operators::{Operator, OperatorCopyIn};
use crate::properties::physical::PhysicalProperties;
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
    NestedLoop(NestedLoop),
    Scan(Scan),
    IndexScan(IndexScan),
    Sort(Sort),
    Append(Append),
    Unique(Unique),
    HashedSetOp(HashedSetOp),
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
            PhysicalExpr::NestedLoop(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Scan(_) => {}
            PhysicalExpr::IndexScan(_) => {}
            PhysicalExpr::Sort(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Unique(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::Append(expr) => expr.copy_in(visitor, expr_ctx),
            PhysicalExpr::HashedSetOp(expr) => expr.copy_in(visitor, expr_ctx),
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
            PhysicalExpr::NestedLoop(expr) => PhysicalExpr::NestedLoop(expr.with_new_inputs(inputs)),
            PhysicalExpr::Scan(expr) => PhysicalExpr::Scan(expr.with_new_inputs(inputs)),
            PhysicalExpr::IndexScan(expr) => PhysicalExpr::IndexScan(expr.with_new_inputs(inputs)),
            PhysicalExpr::Sort(expr) => PhysicalExpr::Sort(expr.with_new_inputs(inputs)),
            PhysicalExpr::Unique(expr) => PhysicalExpr::Unique(expr.with_new_inputs(inputs)),
            PhysicalExpr::Append(expr) => PhysicalExpr::Append(expr.with_new_inputs(inputs)),
            PhysicalExpr::HashedSetOp(expr) => PhysicalExpr::HashedSetOp(expr.with_new_inputs(inputs)),
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
            PhysicalExpr::NestedLoop(expr) => expr.num_children(),
            PhysicalExpr::Scan(expr) => expr.num_children(),
            PhysicalExpr::IndexScan(expr) => expr.num_children(),
            PhysicalExpr::Sort(expr) => expr.num_children(),
            PhysicalExpr::Unique(expr) => expr.num_children(),
            PhysicalExpr::Append(expr) => expr.num_children(),
            PhysicalExpr::HashedSetOp(expr) => expr.num_children(),
            PhysicalExpr::Empty(expr) => expr.num_children(),
        }
    }

    pub fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        match self {
            PhysicalExpr::Projection(expr) => expr.get_child(i),
            PhysicalExpr::Select(expr) => expr.get_child(i),
            PhysicalExpr::HashAggregate(expr) => expr.get_child(i),
            PhysicalExpr::HashJoin(expr) => expr.get_child(i),
            PhysicalExpr::MergeSortJoin(expr) => expr.get_child(i),
            PhysicalExpr::NestedLoop(expr) => expr.get_child(i),
            PhysicalExpr::Scan(expr) => expr.get_child(i),
            PhysicalExpr::IndexScan(expr) => expr.get_child(i),
            PhysicalExpr::Sort(expr) => expr.get_child(i),
            PhysicalExpr::Unique(expr) => expr.get_child(i),
            PhysicalExpr::Append(expr) => expr.get_child(i),
            PhysicalExpr::HashedSetOp(expr) => expr.get_child(i),
            PhysicalExpr::Empty(expr) => expr.get_child(i),
        }
    }

    pub fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        match self {
            PhysicalExpr::Projection(expr) => expr.build_required_properties(),
            PhysicalExpr::Select(expr) => expr.build_required_properties(),
            PhysicalExpr::HashAggregate(expr) => expr.build_required_properties(),
            PhysicalExpr::HashJoin(expr) => expr.build_required_properties(),
            PhysicalExpr::MergeSortJoin(expr) => expr.build_required_properties(),
            PhysicalExpr::NestedLoop(expr) => expr.build_required_properties(),
            PhysicalExpr::Scan(expr) => expr.build_required_properties(),
            PhysicalExpr::IndexScan(expr) => expr.build_required_properties(),
            PhysicalExpr::Sort(expr) => expr.build_required_properties(),
            PhysicalExpr::Unique(expr) => expr.build_required_properties(),
            PhysicalExpr::Append(expr) => expr.build_required_properties(),
            PhysicalExpr::HashedSetOp(expr) => expr.build_required_properties(),
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
            PhysicalExpr::NestedLoop(expr) => expr.format_expr(f),
            PhysicalExpr::Scan(expr) => expr.format_expr(f),
            PhysicalExpr::IndexScan(expr) => expr.format_expr(f),
            PhysicalExpr::Sort(expr) => expr.format_expr(f),
            PhysicalExpr::HashAggregate(expr) => expr.format_expr(f),
            PhysicalExpr::Unique(expr) => expr.format_expr(f),
            PhysicalExpr::Append(expr) => expr.format_expr(f),
            PhysicalExpr::HashedSetOp(expr) => expr.format_expr(f),
            PhysicalExpr::Empty(expr) => expr.format_expr(f),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Projection {
    pub input: RelNode,
    pub columns: Vec<ColumnId>,
}

impl Projection {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(1, "Projection");
        Projection {
            input: inputs.rel_node(),
            columns: self.columns.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        1
    }

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        if i == 0 {
            Some((&self.input).into())
        } else {
            None
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Projection");
        f.write_expr("input", &self.input);
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

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        1 + self.filter.as_ref().map(|_| 1).unwrap_or_default()
    }

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        match i {
            0 => Some((&self.input).into()),
            1 if self.filter.is_some() => {
                let filter = self.filter.as_ref().unwrap();
                Some(filter.into())
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
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(1 + self.aggr_exprs.len() + self.group_exprs.len(), "HashAggregate");

        HashAggregate {
            input: inputs.rel_node(),
            aggr_exprs: inputs.scalar_nodes(self.aggr_exprs.len()),
            group_exprs: inputs.scalar_nodes(self.group_exprs.len()),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        1 + self.aggr_exprs.len() + self.group_exprs.len()
    }

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        let num_aggr_exprs = self.aggr_exprs.len();
        let num_group_exprs = self.group_exprs.len();
        match i {
            0 => Some((&self.input).into()),
            _ if i >= 1 && i < 1 + num_aggr_exprs => {
                let expr = &self.aggr_exprs[i - 1];
                Some(expr.into())
            }
            _ if i >= num_aggr_exprs && i < 1 + num_aggr_exprs + num_group_exprs => {
                let expr = &self.group_exprs[i - num_aggr_exprs - 1];
                Some(expr.into())
            }
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("HashAggregate");
        f.write_expr("input", &self.input);
        for aggr_expr in self.aggr_exprs.iter() {
            f.write_expr("", aggr_expr);
        }
        for group_expr in self.group_exprs.iter() {
            f.write_expr("", group_expr);
        }
    }
}

#[derive(Debug, Clone)]
pub struct HashJoin {
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
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            condition: match &self.condition {
                JoinCondition::Using(_) => self.condition.clone(),
                JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node())),
            },
        }
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        let num_opt = match &self.condition {
            JoinCondition::Using(_) => 0,
            JoinCondition::On(_) => 1,
        };
        2 + num_opt
    }

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        match i {
            0 => Some((&self.left).into()),
            1 => Some((&self.right).into()),
            2 => match &self.condition {
                JoinCondition::Using(_) => unreachable!(),
                JoinCondition::On(on) => Some((&on.expr).into()),
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
        match &self.condition {
            JoinCondition::Using(using) => f.write_value("using", using),
            JoinCondition::On(on) => f.write_value("on", on),
        };
    }
}

#[derive(Debug, Clone)]
pub struct MergeSortJoin {
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
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            condition: match &self.condition {
                JoinCondition::Using(_) => self.condition.clone(),
                JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node())),
            },
        }
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        match get_join_columns_pair(&self.left, &self.right, &self.condition) {
            Some((left, right)) if !left.is_empty() && !right.is_empty() => {
                let left_ordering = PhysicalProperties::new(OrderingChoice::new(left));
                let right_ordering = PhysicalProperties::new(OrderingChoice::new(right));

                Some(vec![left_ordering, right_ordering])
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

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        match i {
            0 => Some((&self.left).into()),
            1 => Some((&self.right).into()),
            2 => match &self.condition {
                JoinCondition::Using(_) => None,
                JoinCondition::On(on) => Some((&on.expr).into()),
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
        match &self.condition {
            JoinCondition::Using(using) => f.write_value("using", using),
            JoinCondition::On(on) => f.write_value("on", on),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NestedLoop {
    pub left: RelNode,
    pub right: RelNode,
    pub condition: Option<ScalarNode>,
}

impl NestedLoop {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
        visitor.visit_opt_scalar(expr_ctx, self.condition.as_ref());
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        let num_opt = self.condition.as_ref().map(|_| 1).unwrap_or_default();
        inputs.expect_len(2 + num_opt, "NestedLoop");

        NestedLoop {
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            condition: self.condition.as_ref().map(|_| inputs.scalar_node()),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        2 + self.condition.as_ref().map(|_| 1).unwrap_or_default()
    }

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        match i {
            0 => Some((&self.left).into()),
            1 => Some((&self.right).into()),
            2 if self.condition.is_some() => {
                let condition = self.condition.as_ref().unwrap();
                Some(condition.into())
            }
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("NestedLoopJoin");
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

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        0
    }

    fn get_child(&self, _i: usize) -> Option<ChildNodeRef<Operator>> {
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
}

impl IndexScan {
    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(0, "IndexScan");

        IndexScan {
            source: self.source.clone(),
            columns: self.columns.clone(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        0
    }

    fn get_child(&self, _i: usize) -> Option<ChildNodeRef<Operator>> {
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

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        1
    }

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        if i == 0 {
            Some((&self.input).into())
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
        f.write_value("ord", format!("{:?}", self.ordering.columns()).as_str())
    }
}

#[derive(Debug, Clone)]
pub struct Append {
    pub left: RelNode,
    pub right: RelNode,
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
        }
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        2
    }

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        match i {
            0 => Some((&self.left).into()),
            1 => Some((&self.right).into()),
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
    }
}

#[derive(Debug, Clone)]
pub struct Unique {
    pub left: RelNode,
    pub right: RelNode,
}

impl Unique {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(2, "Unique");

        Unique {
            left: inputs.rel_node(),
            right: inputs.rel_node(),
        }
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        let left = self.left.props().logical().output_columns().iter().copied().collect();
        let right = self.right.props().logical().output_columns().iter().copied().collect();
        let left_ordering = PhysicalProperties::new(OrderingChoice::new(left));
        let right_ordering = PhysicalProperties::new(OrderingChoice::new(right));

        let requirements = vec![left_ordering, right_ordering];
        Some(requirements)
    }

    fn num_children(&self) -> usize {
        2
    }

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        match i {
            0 => Some((&self.left).into()),
            1 => Some((&self.right).into()),
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Unique");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
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
}

impl HashedSetOp {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(2, "Unique");

        HashedSetOp {
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            intersect: self.intersect,
            all: self.all,
        }
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        2
    }

    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Operator>> {
        match i {
            0 => Some((&self.left).into()),
            1 => Some((&self.right).into()),
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
    }
}

#[derive(Debug, Clone)]
pub struct Empty {}

impl Empty {
    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(0, "Empty");
        Empty {}
    }

    fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        None
    }

    fn num_children(&self) -> usize {
        0
    }

    fn get_child(&self, _i: usize) -> Option<ChildNodeRef<Operator>> {
        None
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("Empty");
    }
}
