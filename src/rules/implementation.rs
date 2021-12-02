use crate::catalog::{CatalogRef, IndexRef};
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
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
            LogicalExpr::Get { source, columns } => {
                let table = self.catalog.get_table(source);
                match table {
                    Some(_) => {
                        let expr = PhysicalExpr::Scan {
                            source: source.into(),
                            columns: columns.clone(),
                        };
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
            LogicalExpr::Select { input, filter } => {
                let expr = PhysicalExpr::Select {
                    input: input.clone(),
                    filter: filter.clone(),
                };
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
            LogicalExpr::Projection {
                input,
                columns,
                exprs: _exprs,
            } => {
                let expr = PhysicalExpr::Projection {
                    input: input.clone(),
                    columns: columns.clone(),
                };
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

    fn matches(&self, _ctx: &RuleContext, operator: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(operator, LogicalExpr::Join { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join { left, right, condition } => {
                let expr = PhysicalExpr::HashJoin {
                    left: left.clone(),
                    right: right.clone(),
                    condition: condition.clone(),
                };
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

    fn matches(&self, _ctx: &RuleContext, operator: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(operator, LogicalExpr::Join { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join { left, right, condition } => {
                let expr = PhysicalExpr::MergeSortJoin {
                    left: left.clone(),
                    right: right.clone(),
                    condition: condition.clone(),
                };
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
                let column = ctx.metadata().get_column(&column_id);
                if idx_column.as_ref() != column.column() {
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
            LogicalExpr::Get { source, columns } => {
                let index = self.find_index(ctx, source, columns);
                Ok(index.map(|_| {
                    let expr = PhysicalExpr::IndexScan {
                        source: source.clone(),
                        columns: columns.clone(),
                    };
                    RuleResult::Implementation(expr)
                }))
            }
            _ => Ok(None),
        }
    }
}

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
        if let LogicalExpr::Aggregate {
            input,
            aggr_exprs,
            group_exprs,
        } = expr
        {
            let expr = PhysicalExpr::HashAggregate {
                input: input.clone(),
                aggr_exprs: aggr_exprs.clone(),
                group_exprs: group_exprs.clone(),
            };
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
        if let LogicalExpr::Union { left, right, all } = expr {
            let expr = if *all {
                PhysicalExpr::Append {
                    left: left.clone(),
                    right: right.clone(),
                }
            } else {
                PhysicalExpr::Unique {
                    left: left.clone(),
                    right: right.clone(),
                }
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
            LogicalExpr::Intersect { left, right, all } => PhysicalExpr::HashedSetOp {
                left: left.clone(),
                right: right.clone(),
                intersect: true,
                all: *all,
            },
            LogicalExpr::Except { left, right, all } => PhysicalExpr::HashedSetOp {
                left: left.clone(),
                right: right.clone(),
                intersect: false,
                all: *all,
            },
            _ => return Ok(None),
        };
        Ok(Some(RuleResult::Implementation(expr)))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::meta::ColumnId;
    use crate::operators::RelNode;
    use crate::rules::testing::RuleTester;

    #[test]
    fn test_sorted_inputs_union() {
        let mut tester = RuleTester::new(UnionRule);

        fn union_expr(all: bool) -> LogicalExpr {
            LogicalExpr::Union {
                left: new_get("A", vec![1, 2]),
                right: new_get("B", vec![3, 4]),
                all,
            }
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
            LogicalExpr::Intersect {
                left: new_get("A", vec![1, 2]),
                right: new_get("B", vec![3, 4]),
                all,
            }
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
            LogicalExpr::Except {
                left: new_get("A", vec![1, 2]),
                right: new_get("B", vec![3, 4]),
                all,
            }
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

    fn new_get(src: &str, columns: Vec<ColumnId>) -> RelNode {
        let expr = LogicalExpr::Get {
            source: src.into(),
            columns,
        };
        RelNode::from(expr)
    }
}
