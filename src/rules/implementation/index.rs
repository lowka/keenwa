use crate::catalog::{CatalogRef, IndexRef, Ordering};
use crate::error::OptimizerError;
use crate::meta::{ColumnId, MetadataRef};
use crate::operators::relational::logical::{LogicalExpr, LogicalGet};
use crate::operators::relational::physical::{IndexScan, PhysicalExpr};
use crate::properties::{OrderingChoice, OrderingColumn};
use crate::rules::{Rule, RuleContext, RuleMatch, RuleResult, RuleType};

#[derive(Debug)]
pub struct IndexOnlyScanRule {
    catalog: CatalogRef,
}

impl IndexOnlyScanRule {
    pub fn new(catalog: CatalogRef) -> Self {
        IndexOnlyScanRule { catalog }
    }

    fn find_covering_index(&self, ctx: &RuleContext, source: &str, columns: &[ColumnId]) -> Option<IndexRef> {
        for index in self.catalog.get_indexes(source).iter().filter(|i| i.table() == source) {
            let num_columns = index
                .columns()
                .iter()
                .zip(columns.iter())
                .filter(|(idx_column, column_id)| {
                    let metadata = ctx.metadata();
                    let column = metadata.get_column(column_id);
                    idx_column.name() == column.name()
                })
                .count();

            if num_columns >= columns.len() {
                return Some(index.clone());
            }
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
                let index = self.find_covering_index(ctx, source, columns);
                match index.as_ref().and_then(|i| i.ordering()) {
                    Some(ordering) => {
                        let ordering = ordering_choice_from_catalog_ordering(ctx.metadata(), ordering);
                        let expr = PhysicalExpr::IndexScan(IndexScan {
                            source: source.clone(),
                            columns: columns.clone(),
                            ordering: Some(ordering?),
                        });
                        Ok(Some(RuleResult::Implementation(expr)))
                    }
                    None => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }
}

fn ordering_choice_from_catalog_ordering(
    metadata: &MetadataRef,
    ordering: &Ordering,
) -> Result<OrderingChoice, OptimizerError> {
    let ordering_columns: Result<Vec<_>, OptimizerError> = ordering
        .options()
        .iter()
        .map(|ord| {
            let column = ord.column();
            // Column::table is always present for Index's columns.
            let table = column.table().expect("Index column without table");
            let name = column.name();
            match metadata.get_column_id(table, name) {
                Some(col_id) => Ok(OrderingColumn::ord(col_id, ord.descending())),
                None => {
                    Err(OptimizerError::internal(format!("Column does not exist. Table: {}, column: {}", table, name)))
                }
            }
        })
        .collect();
    Ok(OrderingChoice::new(ordering_columns?))
}

#[cfg(test)]
mod test {
    use crate::catalog::mutable::MutableCatalog;
    use crate::catalog::{Catalog, IndexBuilder, IndexRef, TableBuilder, DEFAULT_SCHEMA};
    use crate::datatypes::DataType;
    use crate::error::OptimizerError;
    use crate::meta::testing::TestMetadata;
    use crate::meta::{ColumnId, MetadataRef};
    use crate::rules::implementation::IndexOnlyScanRule;
    use crate::rules::RuleContext;
    use std::rc::Rc;
    use std::sync::Arc;

    #[test]
    fn test() -> Result<(), OptimizerError> {
        let catalog = MutableCatalog::new();

        let table = TableBuilder::new("a")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::Int32)
            .add_column("a3", DataType::Int32)
            .build()?;
        catalog.add_table(DEFAULT_SCHEMA, table);

        let table = catalog.get_table("a").unwrap();

        let index = IndexBuilder::new(table, "a_a1_a3_idx").add_column("a1").add_column("a3").build()?;
        catalog.add_index(DEFAULT_SCHEMA, index);

        let mut metadata = TestMetadata::with_tables(vec!["a"]);
        let col1 = metadata.column("a").named("a1").data_type(DataType::Int32).build();
        let col2 = metadata.column("a").named("a2").data_type(DataType::Int32).build();
        let col3 = metadata.column("a").named("a3").data_type(DataType::Int32).build();

        let rule = IndexOnlyScanRule::new(Arc::new(catalog));

        let index = find_covering_index(&rule, metadata.get_ref(), "a", &[col1, col3]);
        assert!(index.is_some(), "index covers [1, 3]");

        let index = find_covering_index(&rule, metadata.get_ref(), "a", &[col1]);
        assert!(index.is_some(), "index covers [1]");

        let index = find_covering_index(&rule, metadata.get_ref(), "a", &[col1, col2]);
        assert!(index.is_none(), "index does not cover [1, 2]");

        let index = find_covering_index(&rule, metadata.get_ref(), "a", &[col2]);
        assert!(index.is_none(), "index does not cover [2]");

        Ok(())
    }

    fn find_covering_index(
        rule: &IndexOnlyScanRule,
        metadata: MetadataRef,
        source: &str,
        columns: &[ColumnId],
    ) -> Option<IndexRef> {
        let props = Rc::new(None);
        rule.find_covering_index(&RuleContext::new(props.clone(), metadata), source, columns)
    }
}
