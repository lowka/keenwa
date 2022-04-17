use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
use crate::memo::format_memo;
use crate::operators::format::{OperatorFormatter, OperatorTreeFormatter, SubQueriesFormatter};
use crate::operators::{ExprMemo, Operator};
use crate::sql::testing::parser::parse_test_cases;
use crate::sql::testing::{
    SqlTestCase, SqlTestCaseRunner, SqlTestCaseSet, TestCaseRunResult, TestCaseRunner, TestCaseRunnerFactory,
    TestRunnerError, TestTable,
};
use crate::sql::OperatorFromSqlBuilder;
use crate::statistics::{NoStatisticsBuilder, StatisticsBuilder};
use itertools::Itertools;
use std::sync::Arc;

/// Runs the given SQL-test cases from the given string using the given catalog definition.
///
/// See [SqlTestCaseRunner] for examples for test cases format.
///
/// Catalog defined in YAML format:
/// ```text
/// catalog:
///  - table: A
///     columns: a1:int32, a2:bool,
///  - table: B
///     columns: b1:int32,b2:bool, b3:string
/// ```
///
pub fn run_sql_tests(sql_test_cases_str: &str, catalog_str: &str) {
    let mut buf = String::from(catalog_str);
    buf.push_str("---\n");
    buf.push_str(sql_test_cases_str);

    let SqlTestCaseSet { catalog, test_cases } = parse_test_cases(&buf).unwrap();
    let runner = SqlTestCaseRunner::new(SqlToOperatorTreeRunner, catalog, test_cases);

    runner.run()
}

struct SqlToOperatorTreeRunner;
struct SqlToOperatorTreeInstance<T> {
    builder: OperatorFromSqlBuilder<T>,
}

impl<T> TestCaseRunner for SqlToOperatorTreeInstance<T>
where
    T: StatisticsBuilder + 'static,
{
    fn run_test_case(self, query: &str) -> Result<TestCaseRunResult, TestRunnerError> {
        match self.builder.build(query) {
            Ok((op, _, metadata)) => {
                let sub_queries_formatter = SubQueriesFormatter::new(metadata);
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
    type Runner = SqlToOperatorTreeInstance<NoStatisticsBuilder>;

    fn new_runner(&self, _test_case: &SqlTestCase, tables: Vec<&TestTable>) -> Result<Self::Runner, TestRunnerError> {
        let builder = new_operator_builder(&tables);
        Ok(SqlToOperatorTreeInstance { builder })
    }
}

fn new_operator_builder(tables: &[&TestTable]) -> OperatorFromSqlBuilder<NoStatisticsBuilder> {
    let catalog = Arc::new(MutableCatalog::new());

    for table in tables {
        let mut builder = TableBuilder::new(&table.name);
        for (name, data_type) in table.columns.iter() {
            builder = builder.add_column(name, data_type.clone());
        }
        catalog.add_table(DEFAULT_SCHEMA, builder.build());
    }

    OperatorFromSqlBuilder::new(catalog, NoStatisticsBuilder)
}
