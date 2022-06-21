use std::rc::Rc;
use std::sync::Arc;

use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::meta::MutableMetadata;
use crate::operators::builder::{MemoizeOperators, OperatorBuilder};
use crate::operators::format::format_operator_tree;
use crate::operators::relational::RelNode;
use crate::operators::OperatorMemoBuilder;
use crate::statistics::simple::{DefaultSelectivityStatistics, SimpleCatalogStatisticsBuilder};

pub fn build_and_rewrite_expr<F, R>(f: F, rule: R, expected: &str)
where
    F: FnOnce(OperatorBuilder) -> Result<OperatorBuilder, OptimizerError>,
    R: Fn(&RelNode) -> Result<Option<RelNode>, OptimizerError>,
{
    let catalog = Arc::new(MutableCatalog::new());
    let selectivity_provider = Rc::new(DefaultSelectivityStatistics);
    let metadata = Rc::new(MutableMetadata::new());
    let statistics_builder = SimpleCatalogStatisticsBuilder::new(catalog.clone(), selectivity_provider);

    let memo = OperatorMemoBuilder::new(metadata.clone()).build_with_statistics(statistics_builder);
    let mut memoization = MemoizeOperators::new(memo);

    catalog.add_table(
        DEFAULT_SCHEMA,
        TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::Int32)
            .add_column("a3", DataType::Int32)
            .add_column("a4", DataType::Int32)
            .add_row_count(100)
            .build()
            .expect("A table"),
    );
    catalog.add_table(
        DEFAULT_SCHEMA,
        TableBuilder::new("B")
            .add_column("b1", DataType::Int32)
            .add_column("b2", DataType::Int32)
            .add_column("b3", DataType::Int32)
            .add_column("b4", DataType::Int32)
            .add_row_count(50)
            .build()
            .expect("B table"),
    );

    let builder = OperatorBuilder::new(memoization.take_callback(), catalog, metadata);

    let result = (f)(builder).expect("Operator setup function failed");
    let expr = result.build().expect("Failed to build an operator");
    let rel_node = RelNode::try_from(expr).expect("Failed to create rel node");
    let rewritten = (rule)(&rel_node).expect("Failed to rewrite expression");
    let expr = rewritten.expect("Rewrite rule has not been applied");

    let mut buf = String::new();
    buf.push('\n');
    buf.push_str(format_operator_tree(expr.mexpr()).as_str());
    buf.push('\n');
    assert_eq!(buf, expected, "expected expr");
}
