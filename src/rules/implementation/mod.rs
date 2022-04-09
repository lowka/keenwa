//! Implementation rules. See [rules module](super).

pub mod aggregate;
pub mod index;
pub mod join;
pub mod set;

pub use aggregate::HashAggregateRule;
pub use index::IndexOnlyScanRule;
pub use join::{HashJoinRule, MergeSortJoinRule, NestedLoopJoinRule};
pub use set::{HashSetOpRule, UnionRule};

use crate::catalog::CatalogRef;
use crate::error::OptimizerError;
use crate::operators::relational::logical::{
    LogicalDistinct, LogicalEmpty, LogicalExpr, LogicalGet, LogicalProjection, LogicalSelect,
};
use crate::operators::relational::physical::{Empty, HashAggregate, PhysicalExpr, Projection, Scan, Select, Unique};
use crate::operators::scalar::{ScalarExpr, ScalarNode};
use crate::operators::ScalarProperties;
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

#[derive(Debug)]
pub struct DistinctRule;

impl Rule for DistinctRule {
    fn name(&self) -> String {
        "DistinctRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Distinct(_)) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Distinct(LogicalDistinct {
                input,
                on_expr,
                columns,
            }) => {
                let expr = if let Some(on_expr) = on_expr {
                    PhysicalExpr::Unique(Unique {
                        inputs: vec![input.clone()],
                        on_expr: Some(on_expr.clone()),
                        columns: columns.clone(),
                    })
                } else {
                    let group_exprs: Vec<ScalarNode> = columns
                        .iter()
                        .map(|id| {
                            let expr = ScalarNode::new(ScalarExpr::Column(*id), ScalarProperties::default());
                            expr
                        })
                        .collect();

                    PhysicalExpr::HashAggregate(HashAggregate {
                        input: input.clone(),
                        aggr_exprs: group_exprs.clone(),
                        group_exprs,
                        having: None,
                        columns: columns.clone(),
                    })
                };
                Ok(Some(RuleResult::Implementation(expr)))
            }
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::rules::testing::RuleTester;

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
}
