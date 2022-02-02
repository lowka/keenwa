use std::rc::Rc;
use std::sync::Arc;

use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::memo::MemoBuilder;
use crate::meta::MutableMetadata;
use crate::operators::builder::{MemoizeOperatorCallback, OperatorBuilder};
use crate::operators::properties::LogicalPropertiesBuilder;
use crate::operators::relational::RelNode;
use crate::optimizer::SetPropertiesCallback;
use crate::rules::testing::format_operator_tree;
use crate::statistics::simple::{DefaultSelectivityStatistics, SimpleCatalogStatisticsBuilder};

pub fn build_and_rewrite_expr<F, R>(f: F, rule: R, expected: &str)
where
    F: FnOnce(OperatorBuilder) -> Result<OperatorBuilder, OptimizerError>,
    R: Fn(&RelNode) -> Option<RelNode>,
{
    let catalog = Arc::new(MutableCatalog::new());
    let selectivity_provider = Rc::new(DefaultSelectivityStatistics);
    let metadata = Rc::new(MutableMetadata::new());
    let statistics_builder = SimpleCatalogStatisticsBuilder::new(catalog.clone(), selectivity_provider);
    let properties_builder = Rc::new(LogicalPropertiesBuilder::new(statistics_builder));

    let memo = MemoBuilder::new(metadata.clone())
        .set_callback(Rc::new(SetPropertiesCallback::new(properties_builder)))
        .build();
    let memoization = Rc::new(MemoizeOperatorCallback::new(memo));

    catalog.add_table(
        DEFAULT_SCHEMA,
        TableBuilder::new("A")
            .add_column("a1", DataType::Int32)
            .add_column("a2", DataType::Int32)
            .add_column("a3", DataType::Int32)
            .add_column("a4", DataType::Int32)
            .add_row_count(100)
            .build(),
    );
    catalog.add_table(
        DEFAULT_SCHEMA,
        TableBuilder::new("B")
            .add_column("b1", DataType::Int32)
            .add_column("b2", DataType::Int32)
            .add_column("b3", DataType::Int32)
            .add_column("b4", DataType::Int32)
            .add_row_count(50)
            .build(),
    );

    let builder = OperatorBuilder::new(memoization.clone(), catalog, metadata);

    let result = (f)(builder).expect("Operator setup function failed");
    let expr = result.build().expect("Failed to build an operator");

    println!("{}", format_operator_tree(&expr).as_str());
    println!(">>>");

    let expr = (rule)(&RelNode::from(expr)).expect("Rewrite rule has not been applied");

    let mut buf = String::new();
    buf.push('\n');
    buf.push_str(format_operator_tree(expr.mexpr()).as_str());
    buf.push('\n');
    assert_eq!(buf, expected, "expected expr");

    // Expressions should not outlive the memo.
    let _ = Rc::try_unwrap(memoization).unwrap();
}
