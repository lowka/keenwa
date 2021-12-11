use crate::memo::{ExprContext, MemoExprFormatter};
use crate::meta::ColumnId;
use crate::operators::relational::join::{JoinCondition, JoinOn};
use crate::operators::relational::{RelExpr, RelNode};
use crate::operators::scalar::expr::ExprVisitor;
use crate::operators::scalar::{ScalarExpr, ScalarNode};
use crate::operators::{Operator, OperatorExpr};
use crate::operators::{OperatorCopyIn, OperatorInputs};

// TODO: Docs
/// A logical expression describes a high-level operator without specifying an implementation algorithm to be used.
#[derive(Debug, Clone)]
pub enum LogicalExpr {
    Projection {
        input: RelNode,
        // This list of expressions is converted into a list of columns.
        exprs: Vec<ScalarExpr>,
        columns: Vec<ColumnId>,
    },
    Select {
        input: RelNode,
        filter: Option<ScalarNode>,
    },
    Aggregate {
        input: RelNode,
        aggr_exprs: Vec<ScalarNode>,
        group_exprs: Vec<ScalarNode>,
        /// Output columns produced by an aggregate operator.
        columns: Vec<ColumnId>,
    },
    Join {
        left: RelNode,
        right: RelNode,
        condition: JoinCondition,
    },
    Get {
        source: String,
        columns: Vec<ColumnId>,
    },
    Union {
        left: RelNode,
        right: RelNode,
        all: bool,
        /// Output columns produced by a set operator.
        columns: Vec<ColumnId>,
    },
    Intersect {
        left: RelNode,
        right: RelNode,
        all: bool,
        /// Output columns produced by a set operator.
        columns: Vec<ColumnId>,
    },
    Except {
        left: RelNode,
        right: RelNode,
        all: bool,
        /// Output columns produced by a set operator.
        columns: Vec<ColumnId>,
    },
}

impl LogicalExpr {
    pub(crate) fn traverse<T>(&self, visitor: &mut OperatorCopyIn<T>, expr_ctx: &mut ExprContext<Operator>) {
        match self {
            LogicalExpr::Projection { input, .. } => {
                visitor.visit_rel(expr_ctx, input);
            }
            LogicalExpr::Select { input, filter } => {
                visitor.visit_rel(expr_ctx, input);
                visitor.visit_opt_scalar(expr_ctx, filter.as_ref());
            }
            LogicalExpr::Join {
                left, right, condition, ..
            } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
                visitor.visit_join_condition(expr_ctx, condition);
            }
            LogicalExpr::Get { .. } => {}
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
                ..
            } => {
                visitor.visit_rel(expr_ctx, input);
                for expr in aggr_exprs {
                    visitor.visit_scalar(expr_ctx, expr);
                }
                for group_expr in group_exprs {
                    visitor.visit_scalar(expr_ctx, group_expr);
                }
            }
            LogicalExpr::Union { left, right, .. } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
            }
            LogicalExpr::Intersect { left, right, .. } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
            }
            LogicalExpr::Except { left, right, .. } => {
                visitor.visit_rel(expr_ctx, left);
                visitor.visit_rel(expr_ctx, right);
            }
        }
    }

    pub fn with_new_inputs(&self, inputs: &mut OperatorInputs) -> Self {
        match self {
            LogicalExpr::Projection { columns, exprs, .. } => {
                inputs.expect_len(1, "LogicalProjection");
                LogicalExpr::Projection {
                    input: inputs.rel_node(),
                    columns: columns.clone(),
                    exprs: exprs.clone(),
                }
            }
            LogicalExpr::Select { filter, .. } => {
                let num_opt = filter.as_ref().map(|_| 1).unwrap_or_default();
                inputs.expect_len(1 + num_opt, "LogicalSelect");
                LogicalExpr::Select {
                    input: inputs.rel_node(),
                    filter: filter.as_ref().map(|_| inputs.scalar_node()),
                }
            }
            LogicalExpr::Join { condition, .. } => {
                //inputs.expect_len(2, "LogicalJoin");
                LogicalExpr::Join {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    condition: match condition {
                        JoinCondition::Using(_) => condition.clone(),
                        JoinCondition::On(_) => JoinCondition::On(JoinOn::new(inputs.scalar_node())),
                    },
                }
            }
            LogicalExpr::Get { source, columns } => {
                inputs.expect_len(0, "LogicalGet");
                LogicalExpr::Get {
                    source: source.clone(),
                    columns: columns.clone(),
                }
            }
            LogicalExpr::Aggregate {
                aggr_exprs,
                group_exprs,
                columns,
                ..
            } => {
                inputs.expect_len(1 + aggr_exprs.len() + group_exprs.len(), "LogicalAggregate");
                LogicalExpr::Aggregate {
                    input: inputs.rel_node(),
                    aggr_exprs: inputs.scalar_nodes(aggr_exprs.len()),
                    group_exprs: inputs.scalar_nodes(group_exprs.len()),
                    columns: columns.clone(),
                }
            }
            LogicalExpr::Union { all, columns, .. } => {
                inputs.expect_len(2, "LogicalUnion");
                LogicalExpr::Union {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    all: *all,
                    columns: columns.clone(),
                }
            }
            LogicalExpr::Intersect { all, columns, .. } => {
                inputs.expect_len(2, "LogicalIntersect");
                LogicalExpr::Intersect {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    all: *all,
                    columns: columns.clone(),
                }
            }
            LogicalExpr::Except { all, columns, .. } => {
                inputs.expect_len(2, "LogicalExcept");
                LogicalExpr::Except {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                    all: *all,
                    columns: columns.clone(),
                }
            }
        }
    }

    pub(crate) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            LogicalExpr::Projection {
                input, columns, exprs, ..
            } => {
                f.write_name("LogicalProjection");
                f.write_expr("input", input);
                f.write_values("cols", columns);
                f.write_values("exprs", exprs);
            }
            LogicalExpr::Select { input, filter } => {
                f.write_name("LogicalSelect");
                f.write_expr("input", input);
                f.write_expr_if_present("filter", filter.as_ref());
            }
            LogicalExpr::Join { left, right, condition } => {
                f.write_name("LogicalJoin");
                f.write_expr("left", left);
                f.write_expr("right", right);
                match condition {
                    JoinCondition::Using(using) => f.write_value("using", using),
                    JoinCondition::On(on) => f.write_value("on", on),
                }
            }
            LogicalExpr::Get { source, columns } => {
                f.write_name("LogicalGet");
                f.write_source(source);
                f.write_values("cols", columns);
            }
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
                ..
            } => {
                f.write_name("LogicalAggregate");
                f.write_expr("input", input);
                for aggr_expr in aggr_exprs {
                    f.write_expr("", aggr_expr);
                }
                for group_expr in group_exprs {
                    f.write_expr("", group_expr);
                }
            }
            LogicalExpr::Union {
                left,
                right,
                all,
                columns: _columns,
            } => {
                f.write_name("LogicalUnion");
                f.write_expr("left", left);
                f.write_expr("right", right);
                f.write_value("all", all);
            }
            LogicalExpr::Intersect {
                left,
                right,
                all,
                columns: _columns,
            } => {
                f.write_name("LogicalIntersect");
                f.write_expr("left", left);
                f.write_expr("right", right);
                f.write_value("all", all);
            }
            LogicalExpr::Except {
                left,
                right,
                all,
                columns: _columns,
            } => {
                f.write_name("LogicalExcept");
                f.write_expr("left", left);
                f.write_expr("right", right);
                f.write_value("all", all);
            }
        }
    }

    // FIXME: ??? ToOperator trait
    pub fn to_operator(self) -> Operator {
        let expr = RelExpr::Logical(Box::new(self));
        let expr = OperatorExpr::Relational(expr);
        Operator::from(expr)
    }

    /// Performs depth-first traversal of this expression tree calling methods of the given `visitor`.
    pub fn accept<T>(&self, visitor: &mut T)
    where
        T: LogicalExprVisitor,
    {
        if !visitor.pre_visit(self) {
            return;
        }

        struct VisitNestedLogicalExprs<'a, T> {
            visitor: &'a mut T,
        }
        impl<T> ExprVisitor<RelNode> for VisitNestedLogicalExprs<'_, T>
        where
            T: LogicalExprVisitor,
        {
            fn post_visit(&mut self, expr: &ScalarExpr) {
                if let ScalarExpr::SubQuery(rel_node) = expr {
                    rel_node.expr().as_logical().accept(self.visitor);
                }
            }
        }
        let mut expr_visitor = VisitNestedLogicalExprs { visitor };

        match self {
            LogicalExpr::Projection { input, exprs, .. } => {
                for expr in exprs {
                    expr.accept(&mut expr_visitor);
                }
                input.expr().as_logical().accept(visitor);
            }
            LogicalExpr::Select { input, filter } => {
                if let Some(expr) = filter.as_ref() {
                    expr.expr().accept(&mut expr_visitor);
                }
                input.expr().as_logical().accept(visitor);
            }
            LogicalExpr::Aggregate {
                input,
                aggr_exprs,
                group_exprs,
                ..
            } => {
                for aggr_expr in aggr_exprs {
                    aggr_expr.expr().accept(&mut expr_visitor);
                }
                for group_expr in group_exprs {
                    group_expr.expr().accept(&mut expr_visitor);
                }
                input.expr().as_logical().accept(visitor);
            }
            LogicalExpr::Join { left, right, .. } => {
                left.expr().as_logical().accept(visitor);
                right.expr().as_logical().accept(visitor);
            }
            LogicalExpr::Get { .. } => {}
            LogicalExpr::Union { left, right, .. }
            | LogicalExpr::Intersect { left, right, .. }
            | LogicalExpr::Except { left, right, .. } => {
                left.expr().as_logical().accept(visitor);
                right.expr().as_logical().accept(visitor);
            }
        }
        visitor.post_visit(self);
    }
}

/// Used by [`LogicalExpr::accept`](self::LogicalExpr::accept) during a traversal of an expression tree.
pub trait LogicalExprVisitor {
    /// Called before all child expressions of `expr` are visited.  
    fn pre_visit(&mut self, _expr: &LogicalExpr) -> bool {
        true
    }

    /// Called after all child expressions of `expr` are visited.
    fn post_visit(&mut self, expr: &LogicalExpr);
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SetOperator {
    Union,
    Intersect,
    Except,
}