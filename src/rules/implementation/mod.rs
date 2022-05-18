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
    LogicalDistinct, LogicalEmpty, LogicalExpr, LogicalGet, LogicalLimit, LogicalOffset, LogicalProjection,
    LogicalSelect, LogicalValues,
};
use crate::operators::relational::physical::{
    Empty, HashAggregate, Limit, Offset, PhysicalExpr, Projection, Scan, Select, Unique, Values,
};
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
pub struct ValuesRule;

impl Rule for ValuesRule {
    fn name(&self) -> String {
        "ValuesRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Values(_)) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        if let LogicalExpr::Values(LogicalValues { values, columns }) = expr {
            let expr = PhysicalExpr::Values(Values {
                values: values.clone(),
                columns: columns.clone(),
            });
            Ok(Some(RuleResult::Implementation(expr)))
        } else {
            Ok(None)
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
                        .map(|id| ScalarNode::new(ScalarExpr::Column(*id), ScalarProperties::default()))
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

#[derive(Debug)]
pub struct LimitOffsetRule;

impl Rule for LimitOffsetRule {
    fn name(&self) -> String {
        "LimitOffsetRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Limit(_) | LogicalExpr::Offset(_)) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Limit(LogicalLimit { input, rows }) => {
                let expr = PhysicalExpr::Limit(Limit {
                    input: input.clone(),
                    rows: *rows,
                });
                Ok(Some(RuleResult::Implementation(expr)))
            }
            LogicalExpr::Offset(LogicalOffset { input, rows }) => {
                let expr = PhysicalExpr::Offset(Offset {
                    input: input.clone(),
                    rows: *rows,
                });
                Ok(Some(RuleResult::Implementation(expr)))
            }
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::meta::testing::TestMetadata;
    use crate::operators::scalar::value::ScalarValue;
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

    #[test]
    fn test_values() {
        let mut tester = RuleTester::new(ValuesRule);
        let mut metadata = TestMetadata::with_tables(vec!["A"]);
        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();

        let val1 = ScalarExpr::Scalar(ScalarValue::Int32(1));
        let val2 = ScalarExpr::Scalar(ScalarValue::Bool(true));

        let tuple = ScalarNode::from(ScalarExpr::Tuple(vec![val1, val2]));
        let expr = LogicalExpr::Values(LogicalValues {
            values: vec![tuple],
            columns: vec![col1, col2],
        });
        tester.apply(
            &expr,
            r#"
Values cols=[1, 2] values: [(1, true)]
        "#,
        )
    }
}
