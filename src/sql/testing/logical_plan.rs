//! TODO: Move to another module.
use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
use crate::memo::MemoBuilder;
use crate::meta::MutableMetadata;
use crate::operators::builder::{MemoizeOperators, OperatorBuilder};
use crate::operators::format::{OperatorTreeFormatter, SubQueriesFormatter};
use crate::operators::properties::LogicalPropertiesBuilder;
use crate::operators::OperatorMemoBuilder;
use crate::sql::build_from_sql;
use crate::sql::testing::parser::parse_test_cases;
use crate::sql::testing::{
    SqlTestCase, SqlTestCaseRunner, SqlTestCaseSet, TestCaseRunResult, TestCaseRunner, TestCaseRunnerFactory,
    TestRunnerError, TestTable,
};
use crate::statistics::NoStatisticsBuilder;
use std::rc::Rc;
use std::sync::Arc;

pub fn run_sql_parser_tests(str: &str, catalog_str: &str) {
    let mut buf = String::from(catalog_str);
    buf.push_str("---\n");
    buf.push_str(str);

    let SqlTestCaseSet { catalog, test_cases } = parse_test_cases(&buf).unwrap();
    let runner = SqlTestCaseRunner::new(SqlToOperatorTreeRunner, catalog, test_cases);

    runner.run()
}

struct SqlToOperatorTreeRunner;
struct SqlToOperatorTreeInstance {
    metadata: Rc<MutableMetadata>,
    builder: OperatorBuilder,
}

impl TestCaseRunner for SqlToOperatorTreeInstance {
    fn run_test_case(self, query: &str) -> Result<TestCaseRunResult, TestRunnerError> {
        match build_from_sql(self.builder, query) {
            Ok(op) => {
                let sub_queries_formatter = SubQueriesFormatter::new(self.metadata);
                let formatter = OperatorTreeFormatter::new()
                    // .properties_last()
                    .add_formatter(Box::new(sub_queries_formatter));
                let plan_str = formatter.format(&op);
                Ok(TestCaseRunResult::Success(plan_str))
            }
            Err(error) => Ok(TestCaseRunResult::from(error)),
        }
    }
}

impl TestCaseRunnerFactory for SqlToOperatorTreeRunner {
    type Runner = SqlToOperatorTreeInstance;

    fn new_runner(&self, _test_case: &SqlTestCase, tables: Vec<&TestTable>) -> Result<Self::Runner, TestRunnerError> {
        let metadata = Rc::new(MutableMetadata::new());
        let builder = new_operator_builder(metadata.clone(), &tables);
        Ok(SqlToOperatorTreeInstance { builder, metadata })
    }
}

fn new_operator_builder(metadata: Rc<MutableMetadata>, tables: &[&TestTable]) -> OperatorBuilder {
    let properties_builder = LogicalPropertiesBuilder::new(NoStatisticsBuilder);
    let memo = OperatorMemoBuilder::new(metadata.clone()).build_with_properties(properties_builder);
    let mut memoization = MemoizeOperators::new(memo);

    let catalog = Arc::new(MutableCatalog::new());

    for table in tables {
        let mut builder = TableBuilder::new(&table.name);
        for (name, data_type) in table.columns.iter() {
            builder = builder.add_column(name, data_type.clone());
        }
        catalog.add_table(DEFAULT_SCHEMA, builder.build());
    }

    OperatorBuilder::new(memoization.take_callback(), catalog, metadata)
}
