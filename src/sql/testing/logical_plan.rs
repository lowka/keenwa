use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
use crate::operators::format::{OperatorTreeFormatter, SubQueriesFormatter};
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::RelExpr;
use crate::operators::OperatorExpr;
use crate::sql::testing::parser::parse_test_cases;
use crate::sql::testing::{
    SqlTestCase, SqlTestCaseRunner, SqlTestCaseSet, TestCaseFailure, TestCaseRunResult, TestCaseRunner,
    TestCaseRunnerFactory, TestRunnerError, TestTable,
};
use crate::sql::OperatorFromSqlBuilder;
use crate::statistics::{NoStatisticsBuilder, StatisticsBuilder};
use std::sync::Arc;

/// Runs SQL-test cases from the given string using the given catalog definition.
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
    if !catalog_str.is_empty() {
        buf.push_str("---\n");
    }
    buf.push_str(sql_test_cases_str);

    let SqlTestCaseSet { catalog, test_cases } = parse_test_cases(&buf).unwrap();
    let runner = SqlTestCaseRunner::new(RelExprTestCaseRunnerFactory, catalog, test_cases);

    runner.run()
}

/// Runs SQL-test cases from the given string using the given catalog definition.
/// Unlike [run_sql_tests] this function should be used to test scalar expressions.
/// Internally it wraps a query string from a test case definition into a `SELECT` operator
/// (to build a projection) and matches the expected result with
/// a first expression of a resulting projection operator.
///
/// ```text
/// query: |  
///  cast('111' as int32)
/// ok:
///  CAST('111' AS int32)
///
///  ===>
///
/// query: |  
///  SELECT cast('111' as int32)
/// ok: |
///  LogicalProjection cols=[1] exprs: [CAST('111' AS Int32)]
///     input: LogicalEmpty return_one_row=true
///  ===>
/// ```
/// Variants with multiple queries and error matching are also supported.
///
pub fn run_sql_expression_tests(sql_test_cases_str: &str, catalog_str: &str) {
    let mut buf = String::from(catalog_str);
    if !catalog_str.is_empty() {
        buf.push_str("---\n");
    }
    buf.push_str(sql_test_cases_str);

    let SqlTestCaseSet { catalog, test_cases } = parse_test_cases(&buf).unwrap();
    let runner = SqlTestCaseRunner::new(ScalarExprTestCaseRunnerFactory, catalog, test_cases);

    runner.run()
}

/// Creates instances of [RelExprTestCaseRunner].
struct RelExprTestCaseRunnerFactory;
struct RelExprTestCaseRunner<T> {
    builder: OperatorFromSqlBuilder<T>,
}

impl<T> TestCaseRunner for RelExprTestCaseRunner<T>
where
    T: StatisticsBuilder + 'static,
{
    fn run_test_case(self, query: &str) -> Result<TestCaseRunResult, TestRunnerError> {
        match self.builder.build(query) {
            Ok((op, _, metadata)) => {
                let sub_queries_formatter = SubQueriesFormatter::new(metadata);
                let formatter = OperatorTreeFormatter::new().add_formatter(Box::new(sub_queries_formatter));
                let plan_str = formatter.format(&op);
                Ok(TestCaseRunResult::Success(plan_str))
            }
            Err(error) => Ok(TestCaseRunResult::from(error)),
        }
    }
}

impl TestCaseRunnerFactory for RelExprTestCaseRunnerFactory {
    type Runner = RelExprTestCaseRunner<NoStatisticsBuilder>;

    fn new_runner(&self, _test_case: &SqlTestCase, tables: Vec<&TestTable>) -> Result<Self::Runner, TestRunnerError> {
        let builder = new_operator_builder(&tables);
        Ok(RelExprTestCaseRunner { builder })
    }
}

/// Creates instances of [ScalarExprTestCaseRunner].
struct ScalarExprTestCaseRunnerFactory;
struct ScalarExprTestCaseRunner<T> {
    builder: OperatorFromSqlBuilder<T>,
}

impl TestCaseRunnerFactory for ScalarExprTestCaseRunnerFactory {
    type Runner = ScalarExprTestCaseRunner<NoStatisticsBuilder>;

    fn new_runner(&self, _test_case: &SqlTestCase, tables: Vec<&TestTable>) -> Result<Self::Runner, TestRunnerError> {
        let builder = new_operator_builder(&tables);
        Ok(ScalarExprTestCaseRunner { builder })
    }
}

impl<T> TestCaseRunner for ScalarExprTestCaseRunner<T>
where
    T: StatisticsBuilder + 'static,
{
    fn run_test_case(self, query: &str) -> Result<TestCaseRunResult, TestRunnerError> {
        // Wrap an expression into a SELECT list.
        let query = format!("SELECT {}", query);
        //
        match self.builder.build(query.as_str()) {
            Ok((op, _, _)) => {
                match op.expr() {
                    OperatorExpr::Relational(rel) => match rel {
                        RelExpr::Logical(expr) => {
                            if let LogicalExpr::Projection(projection) = expr.as_ref() {
                                let first_expr = projection.exprs[0].expr();
                                return Ok(TestCaseRunResult::Success(format!("{}", first_expr)));
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
                let formatter = OperatorTreeFormatter::new();
                let unexpected = formatter.format(&op);
                let message = format!("Root operator must be a logical projection operator but got: {}", unexpected);
                let test_failure = TestCaseFailure::new(message);
                Ok(TestCaseRunResult::Failure(test_failure))
            }
            Err(error) => Ok(TestCaseRunResult::from(error)),
        }
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
