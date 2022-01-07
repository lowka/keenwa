use crate::error::OptimizerError;
use crate::memo::{ExprContext, MemoExprFormatter, NewChildExprs};
use crate::meta::ColumnId;
use crate::operators::relational::join::{JoinCondition, JoinOn};
use crate::operators::relational::{RelExpr, RelNode};
use crate::operators::scalar::expr::ExprVisitor;
use crate::operators::scalar::{ScalarExpr, ScalarNode};
use crate::operators::{Operator, OperatorCopyIn, OperatorExpr};

// TODO: Docs
/// A logical expression describes a high-level operator without specifying an implementation algorithm to be used.
#[derive(Debug, Clone)]
pub enum LogicalExpr {
    Projection(LogicalProjection),
    Select(LogicalSelect),
    Aggregate(LogicalAggregate),
    Join(LogicalJoin),
    Get(LogicalGet),
    Union(LogicalUnion),
    Intersect(LogicalIntersect),
    Except(LogicalExcept),
    /// Relation that produces no rows.
    Empty(LogicalEmpty),
}

impl LogicalExpr {
    pub(crate) fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        match self {
            LogicalExpr::Projection(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Select(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Join(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Get(_) => {}
            LogicalExpr::Aggregate(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Union(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Intersect(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Except(expr) => expr.copy_in(visitor, expr_ctx),
            LogicalExpr::Empty(_) => {}
        }
    }

    pub(crate) fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        match self {
            LogicalExpr::Projection(expr) => LogicalExpr::Projection(expr.with_new_inputs(inputs)),
            LogicalExpr::Select(expr) => LogicalExpr::Select(expr.with_new_inputs(inputs)),
            LogicalExpr::Join(expr) => LogicalExpr::Join(expr.with_new_inputs(inputs)),
            LogicalExpr::Get(expr) => LogicalExpr::Get(expr.with_new_inputs(inputs)),
            LogicalExpr::Aggregate(expr) => LogicalExpr::Aggregate(expr.with_new_inputs(inputs)),
            LogicalExpr::Union(expr) => LogicalExpr::Union(expr.with_new_inputs(inputs)),
            LogicalExpr::Intersect(expr) => LogicalExpr::Intersect(expr.with_new_inputs(inputs)),
            LogicalExpr::Except(expr) => LogicalExpr::Except(expr.with_new_inputs(inputs)),
            LogicalExpr::Empty(expr) => LogicalExpr::Empty(expr.with_new_inputs(inputs)),
        }
    }

    pub(crate) fn num_children(&self) -> usize {
        match self {
            LogicalExpr::Projection(expr) => expr.num_children(),
            LogicalExpr::Select(expr) => expr.num_children(),
            LogicalExpr::Join(expr) => expr.num_children(),
            LogicalExpr::Get(expr) => expr.num_children(),
            LogicalExpr::Aggregate(expr) => expr.num_children(),
            LogicalExpr::Union(expr) => expr.num_children(),
            LogicalExpr::Intersect(expr) => expr.num_children(),
            LogicalExpr::Except(expr) => expr.num_children(),
            LogicalExpr::Empty(expr) => expr.num_children(),
        }
    }

    pub(crate) fn get_child(&self, i: usize) -> Option<&Operator> {
        match self {
            LogicalExpr::Projection(expr) => expr.get_child(i),
            LogicalExpr::Select(expr) => expr.get_child(i),
            LogicalExpr::Aggregate(expr) => expr.get_child(i),
            LogicalExpr::Join(expr) => expr.get_child(i),
            LogicalExpr::Get(expr) => expr.get_child(i),
            LogicalExpr::Union(expr) => expr.get_child(i),
            LogicalExpr::Intersect(expr) => expr.get_child(i),
            LogicalExpr::Except(expr) => expr.get_child(i),
            LogicalExpr::Empty(expr) => expr.get_child(i),
        }
    }

    pub(crate) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            LogicalExpr::Projection(expr) => expr.format_expr(f),
            LogicalExpr::Select(expr) => expr.format_expr(f),
            LogicalExpr::Join(expr) => expr.format_expr(f),
            LogicalExpr::Get(expr) => expr.format_expr(f),
            LogicalExpr::Aggregate(expr) => expr.format_expr(f),
            LogicalExpr::Union(expr) => expr.format_expr(f),
            LogicalExpr::Intersect(expr) => expr.format_expr(f),
            LogicalExpr::Except(expr) => expr.format_expr(f),
            LogicalExpr::Empty(expr) => expr.format_expr(f),
        }
    }

    // FIXME: ??? ToOperator trait
    pub fn to_operator(self) -> Operator {
        let expr = RelExpr::Logical(Box::new(self));
        let expr = OperatorExpr::Relational(expr);
        Operator::from(expr)
    }

    /// Performs a depth-first traversal of this expression tree calling methods of the given `visitor`.
    ///
    /// If [LogicalExprVisitor::pre_visit] returns `Ok(false)` then child expression of the expression are not visited.
    ///
    /// If an error is returned then traversal terminates.
    pub fn accept<T>(&self, visitor: &mut T) -> Result<(), OptimizerError>
    where
        T: LogicalExprVisitor,
    {
        if !visitor.pre_visit(self)? {
            return Ok(());
        }

        struct VisitNestedLogicalExprs<'a, T> {
            visitor: &'a mut T,
        }
        impl<T> ExprVisitor<RelNode> for VisitNestedLogicalExprs<'_, T>
        where
            T: LogicalExprVisitor,
        {
            type Error = OptimizerError;

            fn post_visit(&mut self, expr: &ScalarExpr) -> Result<(), Self::Error> {
                if let ScalarExpr::SubQuery(rel_node) = expr {
                    rel_node.expr().logical().accept(self.visitor)?;
                }
                Ok(())
            }
        }
        let mut expr_visitor = VisitNestedLogicalExprs { visitor };

        match self {
            LogicalExpr::Projection(LogicalProjection { input, exprs, .. }) => {
                for expr in exprs {
                    expr.accept(&mut expr_visitor)?;
                }
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Select(LogicalSelect { input, filter }) => {
                if let Some(expr) = filter.as_ref() {
                    expr.expr().accept(&mut expr_visitor)?;
                }
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Aggregate(LogicalAggregate {
                input,
                aggr_exprs,
                group_exprs,
                ..
            }) => {
                for aggr_expr in aggr_exprs {
                    aggr_expr.expr().accept(&mut expr_visitor)?;
                }
                for group_expr in group_exprs {
                    group_expr.expr().accept(&mut expr_visitor)?;
                }
                input.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Join(LogicalJoin { left, right, .. }) => {
                left.expr().logical().accept(visitor)?;
                right.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Get(_) => {}
            LogicalExpr::Union(LogicalUnion { left, right, .. })
            | LogicalExpr::Intersect(LogicalIntersect { left, right, .. })
            | LogicalExpr::Except(LogicalExcept { left, right, .. }) => {
                left.expr().logical().accept(visitor)?;
                right.expr().logical().accept(visitor)?;
            }
            LogicalExpr::Empty(_) => {}
        }
        visitor.post_visit(self)
    }
}

/// Called by [LogicalExpr::accept] during a traversal of an expression tree.
pub trait LogicalExprVisitor {
    /// Called before all child expressions of `expr` are visited.
    /// Default implementation always returns`Ok(true)`.
    fn pre_visit(&mut self, _expr: &LogicalExpr) -> Result<bool, OptimizerError> {
        Ok(true)
    }

    /// Called after all child expressions of `expr` are visited.
    fn post_visit(&mut self, expr: &LogicalExpr) -> Result<(), OptimizerError>;
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SetOperator {
    Union,
    Intersect,
    Except,
}

#[derive(Debug, Clone)]
pub struct LogicalProjection {
    pub input: RelNode,
    // This list of expressions is converted into a list of columns.
    pub exprs: Vec<ScalarExpr>,
    pub columns: Vec<ColumnId>,
}

impl LogicalProjection {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(1, "LogicalProjection");
        LogicalProjection {
            input: inputs.rel_node(),
            columns: self.columns.clone(),
            exprs: self.exprs.clone(),
        }
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
        f.write_name("LogicalProjection");
        f.write_expr("input", &self.input);
        f.write_values("cols", &self.columns);
        f.write_values("exprs", &self.exprs);
    }
}

#[derive(Debug, Clone)]
pub struct LogicalSelect {
    pub input: RelNode,
    pub filter: Option<ScalarNode>,
}

impl LogicalSelect {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.input);
        visitor.visit_opt_scalar(expr_ctx, self.filter.as_ref());
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        let num_opt = self.filter.as_ref().map(|_| 1).unwrap_or_default();
        inputs.expect_len(1 + num_opt, "LogicalSelect");

        LogicalSelect {
            input: inputs.rel_node(),
            filter: self.filter.as_ref().map(|_| inputs.scalar_node()),
        }
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
        f.write_name("LogicalSelect");
        f.write_expr("input", &self.input);
        f.write_expr_if_present("filter", self.filter.as_ref())
    }
}

#[derive(Debug, Clone)]
pub struct LogicalAggregate {
    pub input: RelNode,
    pub aggr_exprs: Vec<ScalarNode>,
    pub group_exprs: Vec<ScalarNode>,
    /// Output columns produced by an aggregate operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalAggregate {
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
        inputs.expect_len(1 + self.aggr_exprs.len() + self.group_exprs.len(), "LogicalAggregate");

        LogicalAggregate {
            input: inputs.rel_node(),
            aggr_exprs: inputs.scalar_nodes(self.aggr_exprs.len()),
            group_exprs: inputs.scalar_nodes(self.group_exprs.len()),
            columns: self.columns.clone(),
        }
    }

    fn num_children(&self) -> usize {
        1 + self.aggr_exprs.len() + self.group_exprs.len()
    }

    fn get_child(&self, i: usize) -> Option<&Operator> {
        let num_aggr_exprs = self.aggr_exprs.len();
        let num_group_exprs = self.group_exprs.len();
        match i {
            0 => Some(self.input.mexpr()),
            _ if i >= 1 && i < num_aggr_exprs + 1 => {
                let expr = &self.aggr_exprs[i - 1];
                Some(expr.mexpr())
            }
            _ if i > num_aggr_exprs && i < 1 + num_aggr_exprs + num_group_exprs => {
                let expr = &self.group_exprs[i - 1 - num_aggr_exprs];
                Some(expr.mexpr())
            }
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalAggregate");
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
pub struct LogicalJoin {
    pub left: RelNode,
    pub right: RelNode,
    pub condition: JoinCondition,
}

impl LogicalJoin {
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
        inputs.expect_len(2 + num_opt, "LogicalJoin");

        LogicalJoin {
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            condition: match &self.condition {
                JoinCondition::Using(_) => self.condition.clone(),
                JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node())),
            },
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
                JoinCondition::Using(_) => unreachable!(),
                JoinCondition::On(on) => Some((&on.expr).mexpr()),
            },
            _ => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalJoin");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        match &self.condition {
            JoinCondition::Using(using) => f.write_value("using", using),
            JoinCondition::On(on) => f.write_value("on", on),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogicalGet {
    pub source: String,
    pub columns: Vec<ColumnId>,
}

impl LogicalGet {
    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(0, "LogicalGet");

        LogicalGet {
            source: self.source.clone(),
            columns: self.columns.clone(),
        }
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
        f.write_name("LogicalGet");
        f.write_source(&self.source);
        f.write_values("cols", &self.columns);
    }
}

#[derive(Debug, Clone)]
pub struct LogicalUnion {
    pub left: RelNode,
    pub right: RelNode,
    pub all: bool,
    /// Output columns produced by a set operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalUnion {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(2, "LogicalUnion");
        LogicalUnion {
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            all: self.all,
            columns: self.columns.clone(),
        }
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
        f.write_name("LogicalUnion");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("all", self.all);
    }
}

#[derive(Debug, Clone)]
pub struct LogicalIntersect {
    pub left: RelNode,
    pub right: RelNode,
    pub all: bool,
    /// Output columns produced by a set operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalIntersect {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(2, "LogicalUnion");
        LogicalIntersect {
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            all: self.all,
            columns: self.columns.clone(),
        }
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
        f.write_name("LogicalIntersect");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("all", self.all);
    }
}

#[derive(Debug, Clone)]
pub struct LogicalExcept {
    pub left: RelNode,
    pub right: RelNode,
    pub all: bool,
    /// Output columns produced by a set operator.
    pub columns: Vec<ColumnId>,
}

impl LogicalExcept {
    fn copy_in<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        visitor.visit_rel(expr_ctx, &self.left);
        visitor.visit_rel(expr_ctx, &self.right);
    }

    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(2, "LogicalExcept");
        LogicalExcept {
            left: inputs.rel_node(),
            right: inputs.rel_node(),
            all: self.all,
            columns: self.columns.clone(),
        }
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
        f.write_name("LogicalExcept");
        f.write_expr("left", &self.left);
        f.write_expr("right", &self.right);
        f.write_value("all", self.all);
    }
}

#[derive(Debug, Clone)]
pub struct LogicalEmpty {
    pub return_one_row: bool,
}

impl LogicalEmpty {
    fn with_new_inputs(&self, inputs: &mut NewChildExprs<Operator>) -> Self {
        inputs.expect_len(0, "LogicalEmpty");
        LogicalEmpty {
            return_one_row: self.return_one_row,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        f.write_name("LogicalEmpty");
        f.write_value("return_one_row", &self.return_one_row)
    }

    fn num_children(&self) -> usize {
        0
    }

    fn get_child(&self, _i: usize) -> Option<&Operator> {
        None
    }
}
