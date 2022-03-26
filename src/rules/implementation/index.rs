use crate::catalog::{CatalogRef, IndexRef};
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::relational::logical::{LogicalExpr, LogicalGet};
use crate::operators::relational::physical::{IndexScan, PhysicalExpr};
use crate::rules::{Rule, RuleContext, RuleMatch, RuleResult, RuleType};

#[derive(Debug)]
pub struct IndexOnlyScanRule {
    catalog: CatalogRef,
}

impl IndexOnlyScanRule {
    pub fn new(catalog: CatalogRef) -> Self {
        IndexOnlyScanRule { catalog }
    }

    fn find_index(&self, ctx: &RuleContext, source: &str, columns: &[ColumnId]) -> Option<IndexRef> {
        for index in self.catalog.get_indexes(source).iter().filter(|i| i.table() == source) {
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
            return Some(index.clone());
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
