use crate::catalog::{CatalogRef, IndexRef};
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::relational::join::{JoinCondition, JoinType};
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalEmpty, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin,
    LogicalProjection, LogicalSelect, LogicalUnion,
};
use crate::operators::relational::physical::{
    Append, Empty, HashAggregate, HashJoin, HashedSetOp, IndexScan, MergeSortJoin, NestedLoopJoin, PhysicalExpr,
    Projection, Scan, Select, Unique,
};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::{BinaryOp, Expr, ExprVisitor};
use crate::operators::scalar::ScalarExpr;
use crate::rules::{Rule, RuleContext, RuleMatch, RuleResult, RuleType};
use std::convert::Infallible;

#[derive(Debug)]
pub struct GetToScanRule {
    catalog: CatalogRef,
}

impl GetToScanRule {
    pub fn new(catalog: CatalogRef) -> Self {
        GetToScanRule { catalog }
    }
}

impl Rule for GetToScanRule {
    fn name(&self) -> String {
        "GetToScanRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, operator: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(operator, LogicalExpr::Get { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Get(LogicalGet { source, columns }) => {
                let table = self.catalog.get_table(source);
                match table {
                    Some(_) => {
                        let expr = PhysicalExpr::Scan(Scan {
                            source: source.into(),
                            columns: columns.clone(),
                        });
                        Ok(Some(RuleResult::Implementation(expr)))
                    }
                    None => {
                        Err(OptimizerError::Internal(format!("Table is not found or does not exists: {:?}", source)))
                    }
                }
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
pub struct SelectRule;

impl Rule for SelectRule {
    fn name(&self) -> String {
        "SelectRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, operator: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(operator, LogicalExpr::Select { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Select(LogicalSelect { input, filter }) => {
                let expr = PhysicalExpr::Select(Select {
                    input: input.clone(),
                    filter: filter.clone(),
                });
                Ok(Some(RuleResult::Implementation(expr)))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
pub struct ProjectionRule;

impl Rule for ProjectionRule {
    fn name(&self) -> String {
        "ProjectionRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, operator: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(operator, LogicalExpr::Projection { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Projection(LogicalProjection { input, columns, exprs }) => {
                let expr = PhysicalExpr::Projection(Projection {
                    input: input.clone(),
                    exprs: exprs.clone(),
                    columns: columns.clone(),
                });
                Ok(Some(RuleResult::Implementation(expr)))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug, PartialEq)]
enum JoinExprOpType {
    Equality,
    Comparison,
    Other,
}

struct ResolveExprOpType<'a> {
    left_columns: &'a [ColumnId],
    right_columns: &'a [ColumnId],
    result: Option<JoinExprOpType>,
}

impl<'a> ResolveExprOpType<'a> {
    fn new(left: &'a RelNode, right: &'a RelNode) -> Self {
        ResolveExprOpType {
            left_columns: left.props().logical().output_columns(),
            right_columns: right.props().logical().output_columns(),
            result: None,
        }
    }

    fn set_result(&mut self, next_op: JoinExprOpType) {
        // if all operators in an expression are equality operators then the result must be JoinExprOpType::Equality.
        // if an expression consists of both comparison and equality operators then the result must be a JoinExprOpType::Comparison.
        // If at least one operator in an expression is neither equality nor comparison then the result must be JoinExprOpType::Other.
        match self.result.as_mut() {
            None => self.result = Some(next_op),
            Some(current @ JoinExprOpType::Equality) if next_op == JoinExprOpType::Comparison => {
                *current = JoinExprOpType::Comparison;
            }
            Some(current @ JoinExprOpType::Comparison) if next_op != JoinExprOpType::Other => {
                *current = JoinExprOpType::Comparison;
            }
            Some(current) if current != &next_op => *current = JoinExprOpType::Other,
            Some(current) => *current = next_op,
        }
    }
}

fn resolve_join_expr_type(left: &RelNode, right: &RelNode, condition: &JoinCondition) -> JoinExprOpType {
    match condition {
        JoinCondition::Using(_) => JoinExprOpType::Equality,
        JoinCondition::On(join_on) => {
            let mut visitor = ResolveExprOpType::new(left, right);
            // Never returns an error.
            join_on.expr().expr().accept(&mut visitor).unwrap();
            // The result is never None
            visitor.result.unwrap()
        }
    }
}

impl<'a> ExprVisitor<RelNode> for ResolveExprOpType<'a> {
    type Error = Infallible;
    fn pre_visit(&mut self, expr: &ScalarExpr) -> Result<bool, Self::Error> {
        match expr {
            ScalarExpr::BinaryExpr {
                lhs,
                op:
                    bin_op @ BinaryOp::Eq
                    | bin_op @ BinaryOp::Lt
                    | bin_op @ BinaryOp::Gt
                    | bin_op @ BinaryOp::LtEq
                    | bin_op @ BinaryOp::GtEq,
                rhs,
            } => match (&**lhs, &**rhs) {
                (ScalarExpr::Column(lhs), ScalarExpr::Column(rhs))
                    if (self.left_columns.contains(lhs) && self.right_columns.contains(rhs))
                        || (self.right_columns.contains(lhs) && self.left_columns.contains(rhs)) =>
                {
                    let join_expr_type = match bin_op {
                        BinaryOp::Eq => JoinExprOpType::Equality,
                        BinaryOp::Gt | BinaryOp::Lt | BinaryOp::GtEq | BinaryOp::LtEq => JoinExprOpType::Comparison,
                        _ => unreachable!(),
                    };
                    self.set_result(join_expr_type);
                    Ok(false)
                }
                (_, _) => {
                    self.set_result(JoinExprOpType::Other);
                    Ok(false)
                }
            },
            ScalarExpr::BinaryExpr { op: BinaryOp::And, .. } => Ok(true),
            _ => {
                self.set_result(JoinExprOpType::Other);
                Ok(false)
            }
        }
    }

    fn post_visit(&mut self, _expr: &ScalarExpr) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct HashJoinRule;

impl Rule for HashJoinRule {
    fn name(&self) -> String {
        "HashJoinRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        match expr {
            LogicalExpr::Join(LogicalJoin {
                join_type,
                left,
                right,
                condition,
            }) if join_type != &JoinType::Cross => {
                let join_condition_type = resolve_join_expr_type(left, right, condition);
                if join_condition_type == JoinExprOpType::Equality {
                    Some(RuleMatch::Expr)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join(LogicalJoin {
                join_type,
                left,
                right,
                condition,
            }) if join_type != &JoinType::Cross => {
                let expr = PhysicalExpr::HashJoin(HashJoin {
                    join_type: join_type.clone(),
                    left: left.clone(),
                    right: right.clone(),
                    condition: condition.clone(),
                });
                Ok(Some(RuleResult::Implementation(expr)))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
pub struct MergeSortJoinRule;

impl Rule for MergeSortJoinRule {
    fn name(&self) -> String {
        "MergeSortJoinRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        match expr {
            LogicalExpr::Join(LogicalJoin {
                join_type,
                left,
                right,
                condition,
            }) if join_type != &JoinType::Cross => {
                let join_expr_type = resolve_join_expr_type(left, right, condition);
                if join_expr_type == JoinExprOpType::Equality || join_expr_type == JoinExprOpType::Comparison {
                    Some(RuleMatch::Expr)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join(LogicalJoin {
                join_type,
                left,
                right,
                condition,
            }) if join_type != &JoinType::Cross => {
                let expr = PhysicalExpr::MergeSortJoin(MergeSortJoin {
                    join_type: join_type.clone(),
                    left: left.clone(),
                    right: right.clone(),
                    condition: condition.clone(),
                });
                Ok(Some(RuleResult::Implementation(expr)))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
pub struct NestedLoopJoinRule;

impl Rule for NestedLoopJoinRule {
    fn name(&self) -> String {
        "NestedLoopJoinRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Join(_)) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join(LogicalJoin {
                join_type,
                left,
                right,
                condition,
                ..
            }) => {
                let condition = match condition {
                    JoinCondition::Using(using) => using.get_expr(),
                    JoinCondition::On(on) => on.expr().clone(),
                };

                let expr = PhysicalExpr::NestedLoopJoin(NestedLoopJoin {
                    join_type: join_type.clone(),
                    left: left.clone(),
                    right: right.clone(),
                    condition: Some(condition),
                });
                Ok(Some(RuleResult::Implementation(expr)))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
pub struct IndexOnlyScanRule {
    catalog: CatalogRef,
}

impl IndexOnlyScanRule {
    pub fn new(catalog: CatalogRef) -> Self {
        IndexOnlyScanRule { catalog }
    }

    fn find_index(&self, ctx: &RuleContext, source: &str, columns: &[ColumnId]) -> Option<IndexRef> {
        for index in self.catalog.get_indexes(source).filter(|i| i.table() == source) {
            if index.columns().len() > columns.len() {
                continue;
            }
            for (i, idx_column) in index.columns().iter().enumerate() {
                let column_id = columns[i];
                let metadata = ctx.metadata();
                let column = metadata.get_column(&column_id);
                if idx_column.name() != column.name() {
                    continue;
                }
            }
            return Some(index);
        }
        None
    }
}

impl Rule for IndexOnlyScanRule {
    fn name(&self) -> String {
        "IndexOnlyScanRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Get { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Get(LogicalGet { source, columns }) => {
                let index = self.find_index(ctx, source, columns);
                Ok(index.map(|_| {
                    let expr = PhysicalExpr::IndexScan(IndexScan {
                        source: source.clone(),
                        columns: columns.clone(),
                    });
                    RuleResult::Implementation(expr)
                }))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
pub struct HashAggregateRule;

impl Rule for HashAggregateRule {
    fn name(&self) -> String {
        "HashAggregateRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Aggregate { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        if let LogicalExpr::Aggregate(LogicalAggregate {
            input,
            aggr_exprs,
            group_exprs,
            ..
        }) = expr
        {
            let expr = PhysicalExpr::HashAggregate(HashAggregate {
                input: input.clone(),
                aggr_exprs: aggr_exprs.clone(),
                group_exprs: group_exprs.clone(),
            });
            Ok(Some(RuleResult::Implementation(expr)))
        } else {
            Ok(None)
        }
    }
}

pub struct UnionRule;

impl Rule for UnionRule {
    fn name(&self) -> String {
        "UnionRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Union { .. } | LogicalExpr::Intersect { .. } | LogicalExpr::Except { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        if let LogicalExpr::Union(LogicalUnion { left, right, all, .. }) = expr {
            let expr = if *all {
                PhysicalExpr::Append(Append {
                    left: left.clone(),
                    right: right.clone(),
                })
            } else {
                PhysicalExpr::Unique(Unique {
                    left: left.clone(),
                    right: right.clone(),
                })
            };
            Ok(Some(RuleResult::Implementation(expr)))
        } else {
            Ok(None)
        }
    }
}

pub struct HashSetOpRule;

impl Rule for HashSetOpRule {
    fn name(&self) -> String {
        "HashSetOpRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Intersect { .. } | LogicalExpr::Except { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        let expr = match expr {
            LogicalExpr::Intersect(LogicalIntersect { left, right, all, .. }) => {
                PhysicalExpr::HashedSetOp(HashedSetOp {
                    left: left.clone(),
                    right: right.clone(),
                    intersect: true,
                    all: *all,
                })
            }
            LogicalExpr::Except(LogicalExcept { left, right, all, .. }) => PhysicalExpr::HashedSetOp(HashedSetOp {
                left: left.clone(),
                right: right.clone(),
                intersect: false,
                all: *all,
            }),
            _ => return Ok(None),
        };
        Ok(Some(RuleResult::Implementation(expr)))
    }
}

pub struct EmptyRule;

impl Rule for EmptyRule {
    fn name(&self) -> String {
        "Empty".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Empty(_)) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        if let LogicalExpr::Empty(LogicalEmpty { return_one_row }) = expr {
            Ok(Some(RuleResult::Implementation(PhysicalExpr::Empty(Empty {
                return_one_row: *return_one_row,
            }))))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::meta::ColumnId;
    use crate::operators::relational::join::JoinType::Inner;
    use crate::operators::relational::join::{JoinOn, JoinUsing};
    use crate::operators::relational::logical::LogicalEmpty;
    use crate::operators::relational::logical::LogicalExpr::Join;
    use crate::operators::relational::{RelExpr, RelNode};
    use crate::operators::scalar::value::ScalarValue;
    use crate::operators::scalar::{col, ScalarExpr, ScalarNode};
    use crate::operators::{Properties, RelationalProperties, ScalarProperties};
    use crate::properties::logical::LogicalProperties;
    use crate::rules::testing::{expect_apply, expect_no_match, RuleTester};

    use super::*;

    fn join_expr(join_type: JoinType) -> LogicalExpr {
        let condition = JoinCondition::using(vec![(1, 3)]);
        join_expr_on(join_type, condition)
    }

    fn join_expr_on(join_type: JoinType, condition: JoinCondition) -> LogicalExpr {
        let join = LogicalJoin {
            join_type,
            left: new_get("A", vec![1, 2]),
            right: new_get("B", vec![3, 4]),
            condition,
        };
        LogicalExpr::Join(join)
    }

    #[test]
    fn test_join_condition_for_on_expr() {
        fn expect_condition_type(on_expr: ScalarExpr, expected: JoinExprOpType) {
            let condition = JoinCondition::On(JoinOn::new(on_expr.clone().into()));
            let left = new_get("A", vec![1, 2]);
            let right = new_get("B", vec![3, 4]);

            let cond_type = resolve_join_expr_type(&left, &right, &condition);
            assert_eq!(expected, cond_type, "expr: {}", on_expr)
        }

        let expr = col_id(1).eq(col_id(3));
        expect_condition_type(expr, JoinExprOpType::Equality);

        let expr = col_id(3).eq(col_id(1));
        expect_condition_type(expr, JoinExprOpType::Equality);

        let expr = col_id(3).eq(col_id(1));
        let expr1 = col_id(2).eq(col_id(4));
        expect_condition_type(expr.and(expr1), JoinExprOpType::Equality);

        let expr = col_id(3).eq(col_id(1));
        let expr1 = col_id(4).eq(col_id(1));
        let expr2 = col_id(1).eq(col_id(3));
        expect_condition_type(expr.and(expr1).and(expr2), JoinExprOpType::Equality);

        // col:3 = col: 1 AND col:4 > col:1 AND col:1 = col:3 => Compare
        let expr = col_id(3).eq(col_id(1));
        let expr1 = col_id(4).gt(col_id(1));
        let expr2 = col_id(1).eq(col_id(3));
        expect_condition_type(expr.and(expr1).and(expr2), JoinExprOpType::Comparison);

        // col:3 > col: 1 AND col:4 > col:1 AND col:1 = col:3 => Compare
        let expr = col_id(3).gt(col_id(1));
        let expr1 = col_id(4).gt(col_id(1));
        let expr2 = col_id(1).eq(col_id(3));
        expect_condition_type(expr.and(expr1).and(expr2), JoinExprOpType::Comparison);

        for op in vec![BinaryOp::Gt, BinaryOp::Lt, BinaryOp::GtEq, BinaryOp::LtEq] {
            let expr = col_id(1).binary_expr(op, col_id(3));
            expect_condition_type(expr, JoinExprOpType::Comparison);
        }

        let expr = col_id(1).or(col_id(3));
        expect_condition_type(expr, JoinExprOpType::Other);

        let expr = col_id(1).eq(col_id(1));
        expect_condition_type(expr, JoinExprOpType::Other);

        let expr = col_id(100).eq(col_id(200));
        expect_condition_type(expr, JoinExprOpType::Other);
    }

    #[test]
    fn test_nested_loop_join() {
        let expr = r#"
NestedLoopJoin type=:type
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  condition: Expr col:1 = col:3
"#;
        expect_apply(NestedLoopJoinRule, &join_expr(JoinType::Inner), expr.replace(":type", "Inner"));
        expect_apply(NestedLoopJoinRule, &join_expr(JoinType::Cross), expr.replace(":type", "Cross"));
        expect_apply(NestedLoopJoinRule, &join_expr(JoinType::Left), expr.replace(":type", "Left"));
        expect_apply(NestedLoopJoinRule, &join_expr(JoinType::Right), expr.replace(":type", "Right"));
        expect_apply(NestedLoopJoinRule, &join_expr(JoinType::Full), expr.replace(":type", "Full"));
    }

    #[test]
    fn test_hash_join_using() {
        let expr = r#"
HashJoin:type using=[(1, 3)]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
"#;
        expect_apply(HashJoinRule, &join_expr(JoinType::Inner), expr.replace(":type", ""));
        expect_no_match(HashJoinRule, &join_expr(JoinType::Cross));
        expect_apply(HashJoinRule, &join_expr(JoinType::Left), expr.replace(":type", " type=Left"));
        expect_apply(HashJoinRule, &join_expr(JoinType::Right), expr.replace(":type", " type=Right"));
        expect_apply(HashJoinRule, &join_expr(JoinType::Full), expr.replace(":type", " type=Full"));
    }

    #[test]
    fn test_hash_join_on_expr() {
        fn condition_matches(on_expr: ScalarExpr) {
            let expr = r#"
HashJoin on=:expr
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
"#;
            let cond_str = format!("{}", on_expr);
            let condition = JoinCondition::On(JoinOn::new(on_expr.into()));
            expect_apply(
                HashJoinRule,
                &join_expr_on(JoinType::Inner, condition),
                expr.replace(":expr", cond_str.as_str()),
            );
        }

        let on_expr = col_id(1).eq(col_id(3));
        condition_matches(on_expr);

        let on_expr = col_id(3).eq(col_id(1));
        condition_matches(on_expr);

        let on_expr = (col_id(1).eq(col_id(3))).and(col_id(2).eq(col_id(4)));
        condition_matches(on_expr);
    }

    #[test]
    fn test_hash_join_can_not_be_used_with_non_eq_conditions() {
        for op in vec![BinaryOp::Or, BinaryOp::Gt, BinaryOp::Lt, BinaryOp::GtEq, BinaryOp::LtEq] {
            let on_expr = col_id(1).binary_expr(op, col_id(3));
            let condition = JoinCondition::On(JoinOn::new(on_expr.into()));

            expect_no_match(HashJoinRule, &join_expr_on(JoinType::Inner, condition));
        }
    }

    #[test]
    fn test_merge_join() {
        let expr = r#"
MergeSortJoin:type using=[(1, 3)]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
"#;
        expect_apply(MergeSortJoinRule, &join_expr(JoinType::Inner), expr.replace(":type", ""));
        expect_no_match(MergeSortJoinRule, &join_expr(JoinType::Cross));
        expect_apply(MergeSortJoinRule, &join_expr(JoinType::Left), expr.replace(":type", " type=Left"));
        expect_apply(MergeSortJoinRule, &join_expr(JoinType::Right), expr.replace(":type", " type=Right"));
        expect_apply(MergeSortJoinRule, &join_expr(JoinType::Full), expr.replace(":type", " type=Full"));
    }

    #[test]
    fn test_merge_join_on_expr() {
        fn condition_matches(on_expr: ScalarExpr) {
            let expr = r#"
MergeSortJoin on=:expr
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
"#;
            let condition = JoinCondition::On(JoinOn::new(on_expr.clone().into()));
            let on_expr_str = format!("{}", on_expr);

            expect_apply(
                MergeSortJoinRule,
                &join_expr_on(JoinType::Inner, condition),
                expr.replace(":expr", on_expr_str.as_str()),
            );
        }

        for op in vec![BinaryOp::Eq, BinaryOp::Gt, BinaryOp::Lt, BinaryOp::GtEq, BinaryOp::LtEq] {
            let on_expr = col_id(1).binary_expr(op, col_id(3));
            condition_matches(on_expr);
        }
    }

    #[test]
    fn test_sorted_inputs_union() {
        let mut tester = RuleTester::new(UnionRule);

        fn union_expr(all: bool) -> LogicalExpr {
            LogicalExpr::Union(LogicalUnion {
                left: new_get("A", vec![1, 2]),
                right: new_get("B", vec![3, 4]),
                all,
                columns: vec![],
            })
        }

        let union = union_expr(false);
        tester.apply(
            &union,
            r#"
Unique
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]    
"#,
        );

        let union_all = union_expr(true);
        tester.apply(
            &union_all,
            r#"
Append
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]    
"#,
        );
    }

    #[test]
    fn test_intercept_hashset_op() {
        let mut tester = RuleTester::new(HashSetOpRule);

        fn intersect_expr(all: bool) -> LogicalExpr {
            LogicalExpr::Intersect(LogicalIntersect {
                left: new_get("A", vec![1, 2]),
                right: new_get("B", vec![3, 4]),
                all,
                columns: vec![],
            })
        }

        let intersect = intersect_expr(false);
        tester.apply(
            &intersect,
            r#"
HashedSetOp intersect=true all=false
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
    "#,
        );

        let intersect_all = intersect_expr(true);
        tester.apply(
            &intersect_all,
            r#"
HashedSetOp intersect=true all=true
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
    "#,
        );
    }

    #[test]
    fn test_except_hashset_op() {
        let mut tester = RuleTester::new(HashSetOpRule);

        fn except_expr(all: bool) -> LogicalExpr {
            LogicalExpr::Except(LogicalExcept {
                left: new_get("A", vec![1, 2]),
                right: new_get("B", vec![3, 4]),
                all,
                columns: vec![],
            })
        }

        let expect = except_expr(false);
        tester.apply(
            &expect,
            r#"
HashedSetOp intersect=false all=false
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
    "#,
        );

        let expect_all = except_expr(true);
        tester.apply(
            &expect_all,
            r#"
HashedSetOp intersect=false all=true
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
    "#,
        );
    }

    #[test]
    fn test_empty() {
        let mut tester = RuleTester::new(EmptyRule);
        let expr = LogicalExpr::Empty(LogicalEmpty { return_one_row: true });
        tester.apply(
            &expr,
            r#"
Empty return_one_row=true
        "#,
        )
    }

    fn new_get(src: &str, columns: Vec<ColumnId>) -> RelNode {
        let expr = LogicalExpr::Get(LogicalGet {
            source: src.into(),
            columns: columns.clone(),
        });
        let props = RelationalProperties {
            logical: LogicalProperties::new(columns, None),
            required: Default::default(),
        };
        RelNode::new(RelExpr::Logical(Box::new(expr)), props)
    }

    fn col_id(id: ColumnId) -> ScalarExpr {
        ScalarExpr::Column(id)
    }
}
