use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{TableBuilder, DEFAULT_SCHEMA};
use crate::operators::builder::OperatorBuilderConfig;
use crate::operators::format::{AppendMemo, AppendMetadata, OperatorTreeFormatter, SubQueriesFormatter};
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::RelExpr;
use crate::operators::OperatorExpr;
use crate::sql::testing::parser::parse_test_cases;
use crate::sql::testing::{
    DefaultTestOptionsParser, SqlTestCase, SqlTestCaseRunner, SqlTestCaseSet, TestCaseFailure, TestCaseRunResult,
    TestCaseRunner, TestCaseRunnerFactory, TestCatalog, TestOptions, TestRunnerError,
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
    parse_and_run(sql_test_cases_str, catalog_str, RelExprTestCaseRunnerFactory)
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
    parse_and_run(sql_test_cases_str, catalog_str, ScalarExprTestCaseRunnerFactory)
}

fn parse_and_run<T>(sql_test_cases_str: &str, catalog_str: &str, test_case_runner_factory: T)
where
    T: TestCaseRunnerFactory,
{
    let mut buf = String::from(catalog_str);
    if !catalog_str.is_empty() {
        buf.push_str("---\n");
    }
    buf.push_str(sql_test_cases_str);

    match parse_test_cases(buf.as_str(), &DefaultTestOptionsParser) {
        Ok(SqlTestCaseSet {
            catalog,
            options,
            test_cases,
        }) => {
            let runner = SqlTestCaseRunner::new(test_case_runner_factory, catalog, options, test_cases);
            runner.run()
        }
        Err(error) => panic!("Failed to parse test cases: {}", error),
    }
}

/// Creates instances of [RelExprTestCaseRunner].
struct RelExprTestCaseRunnerFactory;
struct RelExprTestCaseRunner<T> {
    inner: InnerRunner<T>,
}

impl<T> TestCaseRunner for RelExprTestCaseRunner<T>
where
    T: StatisticsBuilder + 'static,
{
    fn run_test_case(self, query: &str) -> Result<TestCaseRunResult, TestRunnerError> {
        self.inner.run_and_check_root(query)
    }
}

impl TestCaseRunnerFactory for RelExprTestCaseRunnerFactory {
    type Runner = RelExprTestCaseRunner<NoStatisticsBuilder>;

    fn new_runner(
        &self,
        _test_case: &SqlTestCase,
        catalog: &TestCatalog,
        options: &TestOptions,
    ) -> Result<Self::Runner, TestRunnerError> {
        let builder = new_operator_builder(catalog, options);
        Ok(RelExprTestCaseRunner {
            inner: InnerRunner::new(options, builder),
        })
    }
}

/// Creates instances of [ScalarExprTestCaseRunner].
struct ScalarExprTestCaseRunnerFactory;
struct ScalarExprTestCaseRunner<T> {
    inner: InnerRunner<T>,
}

impl TestCaseRunnerFactory for ScalarExprTestCaseRunnerFactory {
    type Runner = ScalarExprTestCaseRunner<NoStatisticsBuilder>;

    fn new_runner(
        &self,
        _test_case: &SqlTestCase,
        catalog: &TestCatalog,
        options: &TestOptions,
    ) -> Result<Self::Runner, TestRunnerError> {
        let builder = new_operator_builder(catalog, options);
        Ok(ScalarExprTestCaseRunner {
            inner: InnerRunner::new(options, builder),
        })
    }
}

impl<T> TestCaseRunner for ScalarExprTestCaseRunner<T>
where
    T: StatisticsBuilder + 'static,
{
    fn run_test_case(self, query: &str) -> Result<TestCaseRunResult, TestRunnerError> {
        self.inner.run_and_check_projection(query)
    }
}

struct InnerRunner<T> {
    builder: OperatorFromSqlBuilder<T>,
    append_memo: bool,
    append_metadata: bool,
}

impl<T> InnerRunner<T>
where
    T: StatisticsBuilder + 'static,
{
    fn new(options: &TestOptions, builder: OperatorFromSqlBuilder<T>) -> Self {
        InnerRunner {
            builder,
            append_memo: options.append_memo.expect("no default value for append_memo option"),
            append_metadata: options.append_memo.expect("no default value for append_metadata option"),
        }
    }

    fn run_and_check_root(self, query: &str) -> Result<TestCaseRunResult, TestRunnerError> {
        match self.builder.build(query) {
            Ok((op, memo, metadata)) => {
                let sub_queries_formatter = SubQueriesFormatter::new(metadata.clone());
                let mut formatter = OperatorTreeFormatter::new().add_formatter(Box::new(sub_queries_formatter));
                if self.append_metadata {
                    formatter = formatter.add_formatter(Box::new(AppendMetadata::new(metadata)));
                }
                if self.append_memo {
                    formatter = formatter.add_formatter(Box::new(AppendMemo::new(memo)))
                }
                let plan_str = formatter.format(&op);
                Ok(TestCaseRunResult::Success(plan_str))
            }
            Err(error) => Ok(TestCaseRunResult::from(error)),
        }
    }

    fn run_and_check_projection(self, query: &str) -> Result<TestCaseRunResult, TestRunnerError> {
        // Wrap an expression into a select list.
        let query = format!("SELECT {}", query);
        match self.builder.build(query.as_str()) {
            Ok((op, _, _)) => {
                if let OperatorExpr::Relational(RelExpr::Logical(expr)) = op.expr() {
                    if let LogicalExpr::Projection(projection) = expr.as_ref() {
                        let first_expr = projection.exprs[0].expr();
                        return Ok(TestCaseRunResult::Success(format!("{}", first_expr)));
                    }
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

fn new_operator_builder(
    test_catalog: &TestCatalog,
    options: &TestOptions,
) -> OperatorFromSqlBuilder<NoStatisticsBuilder> {
    let catalog = Arc::new(MutableCatalog::new());

    for table in test_catalog.tables.iter() {
        let mut builder = TableBuilder::new(&table.name);
        for (name, data_type) in table.columns.iter() {
            builder = builder.add_column(name, data_type.clone());
        }
        catalog.add_table(DEFAULT_SCHEMA, builder.build());
    }

    let mut sql_builder = OperatorFromSqlBuilder::new(catalog, NoStatisticsBuilder);
    let operator_config = OperatorBuilderConfig {
        decorrelate_subqueries: options.decorrelate_subqueries.expect("no default value for decorrelate_subqueries"),
    };
    sql_builder.operator_builder_config(operator_config);
    sql_builder
}
