use crate::catalog::{CatalogRef, IndexRef};
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
use crate::rules::{unexpected_logical_operator, Rule, RuleContext, RuleMatch, RuleResult, RuleType};

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
            LogicalExpr::Projection { input, columns } => {
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
                if idx_column != column {
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
