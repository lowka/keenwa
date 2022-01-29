use crate::catalog::{CatalogRef, IndexRef};
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::relational::join::{get_non_empty_join_columns_pair, JoinCondition, JoinType};
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalEmpty, LogicalExcept, LogicalExpr, LogicalGet, LogicalIntersect, LogicalJoin,
    LogicalProjection, LogicalSelect, LogicalUnion,
};
use crate::operators::relational::physical::{
    Append, Empty, HashAggregate, HashJoin, HashedSetOp, IndexScan, MergeSortJoin, NestedLoop, PhysicalExpr,
    Projection, Scan, Select, Unique,
};
use crate::rules::{Rule, RuleContext, RuleMatch, RuleResult, RuleType};

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
            LogicalExpr::Join(LogicalJoin { join_type, .. }) if join_type != &JoinType::Cross => Some(RuleMatch::Expr),
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
                if get_non_empty_join_columns_pair(left, right, condition).is_some() {
                    let expr = PhysicalExpr::HashJoin(HashJoin {
                        join_type: join_type.clone(),
                        left: left.clone(),
                        right: right.clone(),
                        condition: condition.clone(),
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
            LogicalExpr::Join(LogicalJoin { join_type, .. }) if join_type != &JoinType::Cross => Some(RuleMatch::Expr),
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
                if get_non_empty_join_columns_pair(left, right, condition).is_some() {
                    let expr = PhysicalExpr::MergeSortJoin(MergeSortJoin {
                        join_type: join_type.clone(),
                        left: left.clone(),
                        right: right.clone(),
                        condition: condition.clone(),
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
        if matches!(
            expr,
            LogicalExpr::Join(LogicalJoin {
                join_type: JoinType::Inner | JoinType::Cross,
                ..
            })
        ) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join(LogicalJoin {
                join_type: JoinType::Inner | JoinType::Cross,
                left,
                right,
                condition,
                ..
            }) => {
                let condition = match condition {
                    JoinCondition::Using(using) => using.get_expr(),
                    JoinCondition::On(on) => on.expr().clone(),
                };

                let expr = PhysicalExpr::NestedLoop(NestedLoop {
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
    use super::*;
    use crate::meta::ColumnId;
    use crate::operators::relational::join::JoinOn;
    use crate::operators::relational::join::JoinType::Inner;
    use crate::operators::relational::logical::LogicalEmpty;
    use crate::operators::relational::RelNode;
    use crate::operators::scalar::value::ScalarValue;
    use crate::operators::scalar::{ScalarExpr, ScalarNode};
    use crate::rules::testing::{expect_apply, expect_no_match, RuleTester};

    fn join_expr(join_type: JoinType) -> LogicalExpr {
        let condition = if join_type == JoinType::Cross {
            let expr = ScalarExpr::Scalar(ScalarValue::Bool(true));
            JoinCondition::On(JoinOn::new(ScalarNode::from(expr)))
        } else {
            JoinCondition::using(vec![(1, 3)])
        };
        let join = LogicalJoin {
            join_type,
            left: new_get("A", vec![1, 2]),
            right: new_get("B", vec![3, 4]),
            condition,
        };
        LogicalExpr::Join(join)
    }

    #[test]
    fn test_nested_loop_join() {
        let expr = r#"
NestedLoopJoin
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  condition: Expr :condition
"#;
        expect_apply(NestedLoopJoinRule, &join_expr(JoinType::Inner), expr.replace(":condition", "col:1 = col:3"));
        expect_apply(NestedLoopJoinRule, &join_expr(JoinType::Cross), expr.replace(":condition", "true"));
        expect_no_match(NestedLoopJoinRule, &join_expr(JoinType::Left));
        expect_no_match(NestedLoopJoinRule, &join_expr(JoinType::Right));
        expect_no_match(NestedLoopJoinRule, &join_expr(JoinType::Full));
    }

    #[test]
    fn test_hash_join() {
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
            columns,
        });
        RelNode::from(expr)
    }
}
