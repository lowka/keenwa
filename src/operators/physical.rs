use crate::memo::{ExprContext, MemoExprDigest, MemoExprFormatter, TraversalContext};
use crate::meta::ColumnId;
use crate::operators::expressions::Expr;
use crate::operators::join::JoinCondition;
use crate::operators::{InputExpr, Operator};
use crate::properties::physical::PhysicalProperties;
use crate::properties::OrderingChoice;

// TODO: Docs
/// A physical expression represents an algorithm that can be used to implement a [logical expression].
///
/// [logical expression]: crate::operators::logical::LogicalExpr
#[derive(Debug, Clone)]
pub enum PhysicalExpr {
    Projection {
        input: InputExpr,
        columns: Vec<ColumnId>,
    },
    Select {
        input: InputExpr,
        filter: Expr,
    },
    HashAggregate {
        input: InputExpr,
        aggr_exprs: Vec<Expr>,
        group_exprs: Vec<Expr>,
    },
    HashJoin {
        left: InputExpr,
        right: InputExpr,
        condition: JoinCondition,
    },
    MergeSortJoin {
        left: InputExpr,
        right: InputExpr,
        condition: JoinCondition,
    },
    Scan {
        source: String,
        columns: Vec<ColumnId>,
    },
    IndexScan {
        source: String,
        columns: Vec<ColumnId>,
    },
    Sort {
        input: InputExpr,
        ordering: OrderingChoice,
    },
    Expr {
        expr: Expr,
    },
}

impl PhysicalExpr {
    pub(crate) fn traverse(&self, expr_ctx: &mut ExprContext<Operator>, ctx: &mut TraversalContext<Operator>) {
        match self {
            PhysicalExpr::Projection { input, .. } => {
                ctx.visit_input(expr_ctx, input);
            }
            PhysicalExpr::Select { input, .. } => {
                ctx.visit_input(expr_ctx, input);
            }
            PhysicalExpr::HashAggregate { input, .. } => {
                ctx.visit_input(expr_ctx, input);
            }
            PhysicalExpr::HashJoin { left, right, .. } => {
                ctx.visit_input(expr_ctx, left);
                ctx.visit_input(expr_ctx, right);
            }
            PhysicalExpr::MergeSortJoin { left, right, .. } => {
                ctx.visit_input(expr_ctx, left);
                ctx.visit_input(expr_ctx, right);
            }
            PhysicalExpr::Scan { .. } => {}
            PhysicalExpr::IndexScan { .. } => {}
            PhysicalExpr::Sort { input, .. } => {
                ctx.visit_input(expr_ctx, input);
            }
            PhysicalExpr::Expr { .. } => {}
        }
    }

    pub fn with_new_inputs(&self, mut inputs: Vec<InputExpr>) -> Self {
        match self {
            PhysicalExpr::Projection { columns, .. } => {
                assert_eq!(1, inputs.len(), "Projection operator expects 1 input but got {:?}", inputs);
                PhysicalExpr::Projection {
                    input: inputs.swap_remove(0),
                    columns: columns.clone(),
                }
            }
            PhysicalExpr::Select { filter, .. } => {
                assert_eq!(1, inputs.len(), "Select operator expects 1 input but got {:?}", inputs);
                PhysicalExpr::Select {
                    input: inputs.swap_remove(0),
                    filter: filter.clone(),
                }
            }
            PhysicalExpr::HashAggregate {
                aggr_exprs,
                group_exprs,
                ..
            } => {
                assert_eq!(1, inputs.len(), "HashAggregate operator expects 1 input but got {:?}", inputs);
                PhysicalExpr::HashAggregate {
                    input: inputs.swap_remove(0),
                    aggr_exprs: aggr_exprs.clone(),
                    group_exprs: group_exprs.clone(),
                }
            }
            PhysicalExpr::HashJoin { condition, .. } => {
                assert_eq!(2, inputs.len(), "HashJoin operator expects 2 inputs but got {:?}", inputs);
                PhysicalExpr::HashJoin {
                    left: inputs.swap_remove(0),
                    right: inputs.swap_remove(0),
                    condition: condition.clone(),
                }
            }
            PhysicalExpr::MergeSortJoin { condition, .. } => {
                assert_eq!(2, inputs.len(), "MergeSortJoin operator expects 2 inputs but got {:?}", inputs);
                PhysicalExpr::MergeSortJoin {
                    left: inputs.swap_remove(0),
                    right: inputs.swap_remove(0),
                    condition: condition.clone(),
                }
            }
            PhysicalExpr::Scan { source, columns } => {
                assert_eq!(0, inputs.len(), "Scan operator expects 0 inputs but got {:?}", inputs);
                PhysicalExpr::Scan {
                    source: source.clone(),
                    columns: columns.clone(),
                }
            }
            PhysicalExpr::IndexScan { source, columns } => {
                assert_eq!(0, inputs.len(), "IndexScan operator expects 0 inputs but got {:?}", inputs);
                PhysicalExpr::IndexScan {
                    source: source.clone(),
                    columns: columns.clone(),
                }
            }
            PhysicalExpr::Sort { ordering, .. } => {
                assert_eq!(1, inputs.len(), "Sort operator expects 1 input but got {:?}", inputs);
                PhysicalExpr::Sort {
                    input: inputs[0].clone(),
                    ordering: ordering.clone(),
                }
            }
            PhysicalExpr::Expr { expr } => {
                assert_eq!(0, inputs.len(), "Expr operator expects 0 input but got {:?}", inputs);
                PhysicalExpr::Expr { expr: expr.clone() }
            }
        }
    }

    pub fn build_required_properties(&self) -> Option<Vec<PhysicalProperties>> {
        match self {
            PhysicalExpr::Projection { .. } => None,
            PhysicalExpr::Select { .. } => None,
            PhysicalExpr::HashJoin { .. } => None,
            PhysicalExpr::MergeSortJoin { condition, .. } => {
                let (left, right) = match condition {
                    JoinCondition::Using(using) => using.as_columns_pair(),
                };
                let left_ordering = PhysicalProperties::new(OrderingChoice::new(left));
                let right_ordering = PhysicalProperties::new(OrderingChoice::new(right));

                let requirements = vec![left_ordering, right_ordering];
                Some(requirements)
            }
            PhysicalExpr::Scan { .. } => None,
            PhysicalExpr::IndexScan { .. } => None,
            PhysicalExpr::Sort { .. } => None,
            PhysicalExpr::Expr { .. } => None,
            PhysicalExpr::HashAggregate { .. } => None,
        }
    }

    pub(crate) fn make_digest<D>(&self, digest: &mut D)
    where
        D: MemoExprDigest,
    {
        match self {
            PhysicalExpr::Projection { input, columns } => {
                digest.append_expr_type("p:Projection");
                digest.append_input("input", input);
                digest.append_property("cols", columns);
            }
            PhysicalExpr::Select { input, filter } => {
                digest.append_expr_type("p:Select");
                digest.append_input("input", input);
                digest.append_value(format!("filter={}", filter).as_str());
            }
            PhysicalExpr::HashAggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                digest.append_expr_type("p:HashAggregate");
                digest.append_input("input", input);
                digest.append_property("aggrs", aggr_exprs);
                digest.append_property("groups", group_exprs);
            }
            PhysicalExpr::HashJoin { left, right, condition } => {
                digest.append_expr_type("p:HashJoin");
                digest.append_input("left", left);
                digest.append_input("right", right);
                digest.append_value(format!("condition={:?}", condition).as_str());
            }
            PhysicalExpr::MergeSortJoin { left, right, condition } => {
                digest.append_expr_type("p:MergeSortJoin");
                digest.append_input("left", left);
                digest.append_input("right", right);
                digest.append_value(format!("condition={:?}", condition).as_str());
            }
            PhysicalExpr::Scan { source, columns } => {
                digest.append_expr_type("p:Scan");
                digest.append_value(source);
                digest.append_property("cols", columns);
            }
            PhysicalExpr::IndexScan { source, columns } => {
                digest.append_expr_type("p:IndexScan");
                digest.append_value(source);
                digest.append_property("cols", columns);
            }
            PhysicalExpr::Sort { input, ordering } => {
                digest.append_expr_type("p:Sort");
                digest.append_input("input", input);
                digest.append_value(format!("ordering={}", ordering).as_str());
            }
            PhysicalExpr::Expr { expr } => {
                digest.append_expr_type("p:Expr");
                digest.append_value(format!("{}", expr).as_str());
            }
        }
    }

    pub(crate) fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self {
            PhysicalExpr::Projection { input, columns } => {
                f.write_name("Projection");
                f.write_input("input", input);
                f.write_value("cols", format!("{:?}", columns))
            }
            PhysicalExpr::Select { input, filter } => {
                f.write_name("Select");
                f.write_input("input", input);
                f.write_value("filter", format!("{}", filter))
            }
            PhysicalExpr::HashJoin { left, right, condition } => {
                f.write_name("HashJoin");
                f.write_input("left", left);
                f.write_input("right", right);
                match condition {
                    JoinCondition::Using(using) => f.write_value("using", format!("{}", using)),
                };
            }
            PhysicalExpr::MergeSortJoin { left, right, condition } => {
                f.write_name("MergeSortJoin");
                f.write_input("left", left);
                f.write_input("right", right);
                match condition {
                    JoinCondition::Using(using) => f.write_value("using", format!("{}", using)),
                }
            }
            PhysicalExpr::Scan { source, columns } => {
                f.write_name("Scan");
                f.write_source(source);
                f.write_value("cols", format!("{:?}", columns))
            }
            PhysicalExpr::IndexScan { source, columns } => {
                f.write_name("IndexScan");
                f.write_source(source);
                f.write_value("cols", format!("{:?}", columns))
            }
            PhysicalExpr::Sort { input, ordering } => {
                f.write_name("Sort");
                f.write_input("input", input);
                f.write_value("ord", format!("{:?}", ordering.columns()))
            }
            PhysicalExpr::Expr { expr } => {
                f.write_name(format!("Expr {}", expr).as_str());
            }
            PhysicalExpr::HashAggregate {
                input,
                aggr_exprs,
                group_exprs,
            } => {
                f.write_name("HashAggregate");
                f.write_input("input", input);
                f.write_values("aggrs", &aggr_exprs);
                f.write_values("groups", &group_exprs);
            }
        }
    }
}
