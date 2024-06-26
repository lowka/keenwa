use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::relational::join::{JoinCondition, JoinType};
use crate::operators::relational::logical::{LogicalExpr, LogicalJoin};
use crate::operators::relational::physical::{HashJoin, MergeSortJoin, NestedLoopJoin, PhysicalExpr};
use crate::operators::relational::RelNode;
use crate::operators::scalar::expr::{BinaryOp, ExprVisitor};
use crate::operators::scalar::ScalarExpr;
use crate::rules::{Rule, RuleContext, RuleMatch, RuleResult, RuleType};
use std::convert::Infallible;

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

    fn apply(&self, ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join(LogicalJoin {
                join_type,
                left,
                right,
                condition,
            }) if join_type != &JoinType::Cross => {
                let required_props = ctx.required_properties();
                if let Some((left_ordering, right_ordering)) =
                    MergeSortJoin::derive_input_orderings(required_props, left, right, condition)
                {
                    let expr = PhysicalExpr::MergeSortJoin(MergeSortJoin {
                        join_type: join_type.clone(),
                        left: left.clone(),
                        right: right.clone(),
                        condition: condition.clone(),
                        left_ordering,
                        right_ordering,
                    });
                    Ok(Some(RuleResult::Implementation(expr)))
                } else {
                    Ok(None)
                }
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

#[derive(Debug, PartialEq)]
enum JoinExprOpType {
    /// All expressions use equality operator between columns.
    /// Hash join requires equality expressions.
    Equality,
    /// Expression use either equality or comparison operators between columns.
    /// MergeSortJoin can use both equality and comparison expressions.
    Comparison,
    /// Other operators.
    /// Nested loop join can use any expression.
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::meta::testing::TestMetadata;
    use crate::operators::relational::join::JoinOn;
    use crate::operators::scalar::col;
    use crate::operators::scalar::expr::ExprRewriter;
    use crate::rules::testing::{expect_apply, expect_no_match, new_src};

    fn test_metadata() -> TestMetadata {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B", "C"]);
        // a1, a2
        let _ = metadata.column("A").build();
        let _ = metadata.column("A").build();
        // b1, b2
        let _ = metadata.column("B").build();
        let _ = metadata.column("B").build();
        // c1, c2
        let _ = metadata.column("C").build();
        let _ = metadata.column("C").build();
        //
        metadata
    }

    fn join_expr(metadata: &TestMetadata, join_type: JoinType) -> LogicalExpr {
        let col1 = metadata.find_column("A", "a1");
        let col3 = metadata.find_column("B", "b1");

        let condition = JoinCondition::using(vec![(col1, col3)]);
        join_expr_on(metadata, join_type, condition)
    }

    fn join_expr_on(metadata: &TestMetadata, join_type: JoinType, condition: JoinCondition) -> LogicalExpr {
        let col1 = metadata.find_column("A", "a1");
        let col2 = metadata.find_column("A", "a2");

        let col3 = metadata.find_column("B", "b1");
        let col4 = metadata.find_column("B", "b2");

        let join = LogicalJoin {
            join_type,
            left: new_src("A", vec![col1, col2]),
            right: new_src("B", vec![col3, col4]),
            condition,
        };
        LogicalExpr::Join(join)
    }

    fn expect_condition_type(metadata: &TestMetadata, on_expr: ScalarExpr, expected: JoinExprOpType) {
        let col1 = metadata.find_column("A", "a1");
        let col2 = metadata.find_column("A", "a2");

        let col3 = metadata.find_column("B", "b1");
        let col4 = metadata.find_column("B", "b2");

        let mut rewriter = ReplaceColumnNames { metadata };
        let on_expr = on_expr.rewrite(&mut rewriter).unwrap();
        let condition = JoinCondition::On(JoinOn::new(on_expr.clone().into()));

        let left = new_src("A", vec![col1, col2]);
        let right = new_src("B", vec![col3, col4]);

        let cond_type = resolve_join_expr_type(&left, &right, &condition);
        assert_eq!(expected, cond_type, "expr: {}", on_expr)
    }

    #[test]
    fn test_join_expr_eq_type() {
        let metadata = test_metadata();

        let expr = col("A:a1").eq(col("B:b1"));
        expect_condition_type(&metadata, expr, JoinExprOpType::Equality);

        let expr = col("B:b1").eq(col("A:a1"));
        expect_condition_type(&metadata, expr, JoinExprOpType::Equality);

        let expr = col("B:b1").eq(col("A:a1"));
        let expr1 = col("A:a2").eq(col("B:b2"));
        expect_condition_type(&metadata, expr.and(expr1), JoinExprOpType::Equality);

        let expr = col("B:b1").eq(col("A:a1"));
        let expr1 = col("B:b2").eq(col("A:a1"));
        let expr2 = col("A:a1").eq(col("B:b1"));
        expect_condition_type(&metadata, expr.and(expr1).and(expr2), JoinExprOpType::Equality);
    }

    #[test]
    fn test_join_expr_comparison_type() {
        let metadata = test_metadata();

        for op in [BinaryOp::Gt, BinaryOp::Lt, BinaryOp::GtEq, BinaryOp::LtEq] {
            let expr = col("A:a1").binary_expr(op, col("B:b1"));
            expect_condition_type(&metadata, expr, JoinExprOpType::Comparison);
        }

        // col:b1 = col:a1 AND col:b2 > col:a1 AND col:a1 = col:b1 => Compare
        let expr = col("B:b1").eq(col("A:a1"));
        let expr1 = col("B:b2").gt(col("A:a1"));
        let expr2 = col("A:a1").eq(col("B:b1"));
        expect_condition_type(&metadata, expr.and(expr1).and(expr2), JoinExprOpType::Comparison);

        // col:b1 > col:a1 AND col:b2 > col:a1 AND col:a1 = col:b1 => Compare
        let expr = col("B:b1").gt(col("A:a1"));
        let expr1 = col("B:b2").gt(col("A:a1"));
        let expr2 = col("A:a1").eq(col("B:b1"));
        expect_condition_type(&metadata, expr.and(expr1).and(expr2), JoinExprOpType::Comparison);
    }

    #[test]
    fn test_join_expr_other_type() {
        let metadata = test_metadata();

        let expr = col("A:a1").or(col("B:b1"));
        expect_condition_type(&metadata, expr, JoinExprOpType::Other);

        let expr = col("A:a1").eq(col("A:a1"));
        expect_condition_type(&metadata, expr, JoinExprOpType::Other);

        let expr = col("C:c1").eq(col("C:c2"));
        expect_condition_type(&metadata, expr, JoinExprOpType::Other);

        let expr = col("A:a2").gt(col("A:a1"));
        expect_condition_type(&metadata, expr, JoinExprOpType::Other);
    }

    #[test]
    fn test_nested_loop_join() {
        let metadata = test_metadata();

        let expr = r#"
NestedLoopJoin type=:type
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  condition: Expr col:1 = col:3
"#;
        expect_apply(NestedLoopJoinRule, &join_expr(&metadata, JoinType::Inner), expr.replace(":type", "Inner"));
        expect_apply(NestedLoopJoinRule, &join_expr(&metadata, JoinType::Cross), expr.replace(":type", "Cross"));
        expect_apply(NestedLoopJoinRule, &join_expr(&metadata, JoinType::Left), expr.replace(":type", "Left"));
        expect_apply(NestedLoopJoinRule, &join_expr(&metadata, JoinType::Right), expr.replace(":type", "Right"));
        expect_apply(NestedLoopJoinRule, &join_expr(&metadata, JoinType::Full), expr.replace(":type", "Full"));
    }

    #[test]
    fn test_hash_join_using() {
        let metadata = test_metadata();

        let expr = r#"
HashJoin type=:type using=[(1, 3)]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
"#;
        expect_apply(HashJoinRule, &join_expr(&metadata, JoinType::Inner), expr.replace(":type", "Inner"));
        expect_no_match(HashJoinRule, &join_expr(&metadata, JoinType::Cross));
        expect_apply(HashJoinRule, &join_expr(&metadata, JoinType::Left), expr.replace(":type", "Left"));
        expect_apply(HashJoinRule, &join_expr(&metadata, JoinType::Right), expr.replace(":type", "Right"));
        expect_apply(HashJoinRule, &join_expr(&metadata, JoinType::Full), expr.replace(":type", "Full"));
    }

    #[test]
    fn test_hash_join_on_expr() {
        fn condition_matches(metadata: &TestMetadata, on_expr: ScalarExpr) {
            let expr = r#"
HashJoin type=Inner on=:expr
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
"#;
            let mut rewriter = ReplaceColumnNames { metadata };
            let on_expr = on_expr.rewrite(&mut rewriter).unwrap();
            let cond_str = format!("{}", on_expr);
            let condition = JoinCondition::On(JoinOn::new(on_expr.into()));
            expect_apply(
                HashJoinRule,
                &join_expr_on(metadata, JoinType::Inner, condition),
                expr.replace(":expr", cond_str.as_str()),
            );
        }

        let metadata = test_metadata();

        let on_expr = col("A:a1").eq(col("B:b1"));
        condition_matches(&metadata, on_expr);

        let on_expr = col("B:b1").eq(col("A:a1"));
        condition_matches(&metadata, on_expr);

        let on_expr = col("A:a1").eq(col("B:b1")).and(col("A:a2").eq(col("B:b2")));
        condition_matches(&metadata, on_expr);
    }

    #[test]
    fn test_hash_join_can_not_be_used_with_non_eq_conditions() {
        let metadata = test_metadata();

        for op in [BinaryOp::Or, BinaryOp::Gt, BinaryOp::Lt, BinaryOp::GtEq, BinaryOp::LtEq] {
            let on_expr = col("A:a1").binary_expr(op, col("B:b2"));
            let condition = JoinCondition::On(JoinOn::new(on_expr.into()));

            expect_no_match(HashJoinRule, &join_expr_on(&metadata, JoinType::Inner, condition));
        }
    }

    #[test]
    fn test_merge_join() {
        let metadata = test_metadata();

        let expr = r#"
MergeSortJoin type=:type using=[(1, 3)] left_ord=[+1] right_ord=[+3]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
"#;
        expect_apply(MergeSortJoinRule, &join_expr(&metadata, JoinType::Inner), expr.replace(":type", "Inner"));
        expect_no_match(MergeSortJoinRule, &join_expr(&metadata, JoinType::Cross));
        expect_apply(MergeSortJoinRule, &join_expr(&metadata, JoinType::Left), expr.replace(":type", "Left"));
        expect_apply(MergeSortJoinRule, &join_expr(&metadata, JoinType::Right), expr.replace(":type", "Right"));
        expect_apply(MergeSortJoinRule, &join_expr(&metadata, JoinType::Full), expr.replace(":type", "Full"));
    }

    #[test]
    fn test_merge_join_on_expr() {
        fn condition_matches(metadata: &TestMetadata, on_expr: ScalarExpr) {
            let expr = r#"
MergeSortJoin type=Inner on=:expr left_ord=[+1] right_ord=[+3]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
"#;
            let mut rewriter = ReplaceColumnNames { metadata };
            let on_expr = on_expr.rewrite(&mut rewriter).unwrap();
            let condition = JoinCondition::On(JoinOn::new(on_expr.clone().into()));
            let on_expr_str = format!("{}", on_expr);

            expect_apply(
                MergeSortJoinRule,
                &join_expr_on(metadata, JoinType::Inner, condition),
                expr.replace(":expr", on_expr_str.as_str()),
            );
        }

        let metadata = test_metadata();

        for op in [BinaryOp::Eq, BinaryOp::Gt, BinaryOp::Lt, BinaryOp::GtEq, BinaryOp::LtEq] {
            let on_expr = col("A:a1").binary_expr(op, col("B:b1"));
            condition_matches(&metadata, on_expr);
        }
    }

    struct ReplaceColumnNames<'a> {
        metadata: &'a TestMetadata,
    }

    impl ExprRewriter<RelNode> for ReplaceColumnNames<'_> {
        type Error = Infallible;

        fn rewrite(&mut self, expr: ScalarExpr) -> Result<ScalarExpr, Self::Error> {
            if let ScalarExpr::ColumnName(name) = expr {
                let (table, col) = name.split_once(':').expect("Name should be the table:col format");
                let col_id = self.metadata.find_column(table, col);
                Ok(ScalarExpr::Column(col_id))
            } else {
                Ok(expr)
            }
        }
    }
}
