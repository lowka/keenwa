pub mod logical_plan;
pub mod parser;

use crate::datatypes::DataType;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Test cases are defined in following YAML structure:
///
/// ```text
/// query: |
///   Multiline
///     Query 1  
/// # or queries
/// queries:
///   - a1
///   - q2
/// ok: result
/// error: err
/// ```
/// Successful execution.
///
/// ```text
/// query: |
///   Multiline
///     Query 1  
/// Ok: |
///   Result
///```
///
/// Error (exact match). Passes if the query fails with the given error.
/// ```text
/// query: |
/// Multiline
///   Query 1  
/// error: |
///   Error
/// ```
/// Any error. Passes if the query fails with any error.
/// ```text
/// query: |
/// Multiline
///   Query 1  
/// error: ~
/// ```
///
/// Multiple queries - Passes if all queries return the same result.
/// ```text
/// queries:
///   - Query 1
///   - Query 2
/// ok: |
///   Some result
/// ```
///
/// Multiple queries - Passes if all queries fail with the same error.
/// ```text
/// queries:
///   - Query 1
///   - Query 2
/// ok: |
///   Some result
/// ```
pub struct SqlTestCaseRunner<T> {
    factory: T,
    catalog: Option<TestCatalog>,
    test_cases: Option<Vec<SqlTestCase>>,
    print_runs: bool,
}

impl<T> SqlTestCaseRunner<T>
where
    T: TestCaseRunnerFactory,
{
    /// Creates an environment to run [SqlTestCase]s.
    pub fn new(factory: T, catalog: Option<TestCatalog>, test_cases: Vec<SqlTestCase>) -> Self {
        SqlTestCaseRunner {
            factory,
            catalog,
            test_cases: Some(test_cases),
            print_runs: false,
        }
    }

    /// Runs all the available test cases, collects the results. If there are test failures this method panics
    /// with a message that describes test failures.
    ///
    /// # Panics
    ///
    /// This method panics if there are test failures.
    pub fn run(mut self) {
        let tables = self.prepare_catalog();
        let (test_cases, run_results) = self.run_and_collect_results(tables);
        let error_report = self.build_error_report(test_cases, run_results);

        if let Some(error_report) = error_report {
            panic!("There are test failures:\n{}", error_report);
        }
    }

    fn prepare_catalog(&mut self) -> Vec<TestTable> {
        let catalog = self.catalog.take();
        match catalog {
            Some(catalog) => catalog.tables,
            None => vec![],
        }
    }

    fn run_and_collect_results(&mut self, tables: Vec<TestTable>) -> (Vec<SqlTestCase>, Vec<TestCaseResult>) {
        let mut run_results = vec![];
        let test_cases: Vec<_> = self.test_cases.take().unwrap().into_iter().collect();

        for (test_case_id, test_case) in test_cases.iter().enumerate() {
            for (query_id, query) in test_case.queries.iter().enumerate() {
                let mut actual_tables = HashMap::new();
                tables.iter().for_each(|t| {
                    actual_tables.insert(&t.name, t);
                });

                if let Some(catalog) = &test_case.catalog {
                    actual_tables.extend(catalog.tables.iter().map(|t| (&t.name, t)))
                }
                let tables: Vec<_> = actual_tables.into_iter().map(|(_, t)| t).collect();

                match self.factory.new_runner(test_case, tables) {
                    Ok(instance) => {
                        let test_result = instance.run_test_case(query.as_str());
                        let test_result = match test_result {
                            Ok(r) => r.to_result(),
                            Err(error) => {
                                let message = format!("Error: test case returned an error.\n{}", error);
                                let error = TestCaseResult {
                                    test_case_id,
                                    query_id,
                                    result: Err(message),
                                };
                                run_results.push(error);
                                continue;
                            }
                        };

                        self.add_test_case_result(test_case, test_case_id, query_id, test_result, &mut run_results);
                    }
                    Err(error) => {
                        let message = format!("Error: failed to create a test case.\n{}", error);
                        let error = TestCaseResult {
                            test_case_id,
                            query_id,
                            result: Err(message),
                        };
                        run_results.push(error)
                    }
                }
            }
        }

        (test_cases, run_results)
    }

    fn build_error_report(self, test_cases: Vec<SqlTestCase>, run_results: Vec<TestCaseResult>) -> Option<String> {
        let mut errors = vec![];

        struct TestErrorInfo<'a> {
            test_case_id: usize,
            query: &'a str,
            message: &'a str,
            actual: Option<&'a str>,
        }

        for (i, test_case) in test_cases.iter().enumerate() {
            for test_case_result in run_results.iter().filter(|r| r.test_case_id == i) {
                let test_case_id = test_case_result.test_case_id;
                let query_id = test_case_result.query_id;
                let query = test_case.queries[query_id].trim();

                match &test_case_result.result {
                    Ok(result) => {
                        log::debug!(
                            "Test case: {}. Expected result: {:?}. Actual result: {:?}",
                            query,
                            test_case.expected_result,
                            result.result
                        );

                        if self.print_runs {
                            println!("Test case#{} query#{}:\n{}", test_case_id, query_id, query);
                        }

                        if let Some(Mismatch { message, actual }) = &result.mismatch {
                            if self.print_runs {
                                println!("Test case#{} query#{} failed.", test_case_id, query_id);
                                println!();
                            }

                            errors.push(TestErrorInfo {
                                test_case_id,
                                query,
                                message,
                                actual: Some(actual),
                            });
                        } else if self.print_runs {
                            println!("Test case#{} query#{} passed.", test_case_id, query_id);
                            println!();
                        }
                    }
                    Err(error) => {
                        let error_info = TestErrorInfo {
                            test_case_id,
                            query,
                            message: error,
                            actual: None,
                        };

                        errors.push(error_info);
                    }
                }
            }
        }

        if errors.is_empty() {
            return None;
        }

        let mut error_report = String::new();
        error_report.push('\n');

        let error_num = errors.len();
        for (i, error) in errors.into_iter().enumerate() {
            error_report.push_str(format!("Test case#{} failed:", error.test_case_id).as_str());
            error_report.push('\n');
            error_report.push_str("Query:\n");
            error_report.push_str(error.query);
            error_report.push('\n');
            error_report.push('\n');

            error_report.push_str(error.message.trim_end());
            error_report.push('\n');

            if let Some(actual) = error.actual {
                error_report.push_str(actual.trim_end());
                error_report.push('\n');
            }

            if i < error_num - 1 {
                error_report.push_str("------------");
                error_report.push('\n');
            }
        }

        Some(error_report)
    }

    fn add_test_case_result(
        &self,
        test_case: &SqlTestCase,
        test_case_id: usize,
        query_id: usize,
        test_case_result: Result<String, TestCaseFailure>,
        run_results: &mut Vec<TestCaseResult>,
    ) {
        let mismatch = match (&test_case.expected_result, &test_case_result) {
            (Ok(expected), Ok(actual)) if expected.trim_end() == actual.trim_end() => None,
            (Ok(expected), Ok(actual)) => {
                Some((format!("Expected:\n\"{}\"", expected), format!("But got:\n\"{}\"", actual)))
            }
            (Err(expected), Err(actual)) if expected.matches(actual) => None,
            (Err(expected), Err(actual)) => {
                Some((format!("Expected error:\n\"{}\"", expected), format!("But got another error:\n\"{}\"", actual)))
            }
            (Ok(expected), Err(actual)) => {
                Some((format!("Expected OK:\n{}", expected), format!("But got a test failure:\n{}", actual)))
            }
            (Err(_), Ok(actual)) => Some(("Expected a test failure but got:".to_string(), format!("\n{}", actual))),
        };

        let result = Ok(TestCaseResultData {
            result: test_case_result.map_err(|_| ()),
            mismatch: mismatch.map(|(msg, actual)| Mismatch { message: msg, actual }),
        });
        run_results.push(TestCaseResult {
            test_case_id,
            query_id,
            result,
        })
    }
}

/// Factory that creates execution environments to run [SqlTestCase]s.
pub trait TestCaseRunnerFactory {
    /// The type of a test case runner.
    type Runner: TestCaseRunner;

    /// Creates an instance of a [TestCaseRunner] to execute the given test case.
    fn new_runner(&self, test_case: &SqlTestCase, tables: Vec<&TestTable>) -> Result<Self::Runner, TestRunnerError>;
}

/// Result of a test case run.
#[derive(Debug, Clone)]
pub enum TestCaseRunResult {
    /// Contains the success value.
    Success(String),
    /// Contains the failure.
    Failure(TestCaseFailure),
}

impl TestCaseRunResult {
    fn from_result(r: Result<String, TestCaseFailure>) -> Self {
        match r {
            Ok(message) => TestCaseRunResult::Success(message),
            Err(error) => TestCaseRunResult::Failure(error),
        }
    }

    fn to_result(self) -> Result<String, TestCaseFailure> {
        match self {
            TestCaseRunResult::Success(r) => Ok(r),
            TestCaseRunResult::Failure(err) => Err(err),
        }
    }
}

impl<T> From<T> for TestCaseRunResult
where
    T: Error,
{
    fn from(error: T) -> Self {
        TestCaseRunResult::Failure(TestCaseFailure::new(error.to_string()))
    }
}

/// Executes a test case and returns a result.
pub trait TestCaseRunner {
    /// Executes the given query.
    fn run_test_case(self, query: &str) -> Result<TestCaseRunResult, TestRunnerError>;
}

/// A collection of test cases with an optional catalog.
#[derive(Debug)]
pub struct SqlTestCaseSet {
    /// Database catalog. Contains objects used by the test cases.
    pub catalog: Option<TestCatalog>,
    /// Test cases.
    pub test_cases: Vec<SqlTestCase>,
}

/// A test case consists of a query (or several queries) and an expected result.
/// The result can be either text or on error.
#[derive(Debug)]
pub struct SqlTestCase {
    /// Identifier.
    id: usize,
    /// SQL query string.
    queries: Vec<String>,
    /// Contains database objects specific for this test case.
    catalog: Option<TestCatalog>,
    /// Expected logical plan as string or an error.
    expected_result: Result<String, TestCaseFailure>,
}

/// A test database catalog.
#[derive(Debug)]
pub struct TestCatalog {
    /// A collection of table definitions.
    tables: Vec<TestTable>,
}

/// A test table definition.
#[derive(Debug)]
pub struct TestTable {
    /// Name.
    name: String,
    /// Column definitions.
    columns: Vec<(String, DataType)>,
}

/// An error returned by the test environment.
#[derive(Debug)]
pub struct TestRunnerError {
    /// Contains a description of the error.
    pub message: String,
    /// Contains the cause of the error.
    pub cause: Option<Box<dyn Error>>,
}

impl Display for TestRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            None => write!(f, "TestRunner error: {}", self.message),
            Some(error) => write!(f, "TestRunner error: {} cause by {}", self.message, error),
        }
    }
}

impl Error for TestRunnerError {
    fn cause(&self) -> Option<&dyn Error> {
        self.cause.as_ref().map(|e| e.as_ref())
    }
}

/// Describes a failure test failure.
#[derive(Debug, Clone, PartialEq)]
pub struct TestCaseFailure {
    /// An optional error message.
    pub message: Option<String>,
}

impl TestCaseFailure {
    /// Creates a test case failure with the given message.
    pub fn new<T>(message: T) -> Self
    where
        T: Into<String>,
    {
        TestCaseFailure {
            message: Some(message.into()),
        }
    }

    fn no_message() -> Self {
        TestCaseFailure { message: None }
    }

    fn matches(&self, other: &TestCaseFailure) -> bool {
        match &self.message {
            None => true,
            Some(_) => self.to_string().trim_end() == other.to_string().trim_end(),
        }
    }
}

impl Display for TestCaseFailure {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(message) = &self.message {
            write!(f, "{}", message)
        } else {
            Ok(())
        }
    }
}

impl Error for TestCaseFailure {}

#[derive(Debug)]
struct TestCaseResult {
    test_case_id: usize,
    query_id: usize,
    result: Result<TestCaseResultData, String>,
}

#[derive(Debug)]
struct TestCaseResultData {
    result: Result<String, ()>,
    mismatch: Option<Mismatch>,
}

#[derive(Debug)]
struct Mismatch {
    message: String,
    actual: String,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::sql::testing::parser::parse_test_cases;
    use std::collections::HashMap;

    #[test]
    fn test_run_test_cases() {
        let runner = new_test_case_runner(DummyRunnerFactory::default());
        runner.run();
    }

    #[test]
    fn test_case_from_files_no_errors() {
        let runner = new_test_case_runner(DummyRunnerFactory::default());
        let report = run_tests(runner);

        assert_eq!(report, None);
    }

    #[test]
    fn test_case_from_files_test_creation_error() {
        let mut factory = DummyRunnerFactory::default();
        factory.fail_to_start = Some(1);

        let runner = new_test_case_runner(factory);
        let report = run_tests(runner);

        assert_eq!(
            report,
            Some(
                r#"
Test case#1 failed:
Query:
SELECT 2

Error: failed to create a test case.
TestRunner error: Can't run
"#
                .to_owned()
            ),
        );
    }

    #[test]
    fn test_case_from_files_test_case_mismatch_expected_ok_but_got_error() {
        let mut errors = HashMap::new();
        errors.insert(2, Ok(TestCaseRunResult::Failure(TestCaseFailure::new("Test case failed"))));

        let mut factory = DummyRunnerFactory::default();
        factory.responses = Some(errors);

        let runner = new_test_case_runner(factory);
        let report = run_tests(runner);

        assert_eq!(
            report,
            Some(
                r#"
Test case#2 failed:
Query:
SELECT 3

Expected OK:
Step3
Step4
But got a test failure:
Test case failed
"#
                .to_owned()
            )
        );
    }

    #[test]
    fn test_case_from_files_test_case_mismatch_expected_error_but_got_ok() {
        let mut errors = HashMap::new();
        errors.insert(1, Ok(TestCaseRunResult::Success("Valid\n    plan".to_string())));

        let mut factory = DummyRunnerFactory::default();
        factory.responses = Some(errors);

        let runner = new_test_case_runner(factory);
        let report = run_tests(runner);

        assert_eq!(
            report,
            Some(
                r#"
Test case#1 failed:
Query:
SELECT 2

Expected a test failure but got:

Valid
    plan
"#
                .to_owned()
            )
        );
    }

    #[test]
    fn test_case_from_files_test_case_no_mismatch_when_test_case_has_no_error_description() {
        let mut errors = HashMap::new();
        errors.insert(1, Ok(TestCaseRunResult::Failure(TestCaseFailure::new("Can't do"))));

        let mut factory = DummyRunnerFactory::default();
        factory.responses = Some(errors);

        let runner = new_test_case_runner(factory);
        let report = run_tests(runner);

        assert_eq!(report, None);
    }

    #[test]
    fn test_case_from_files_test_case_mismatch_text_error() {
        let mut errors = HashMap::new();
        errors.insert(3, Ok(TestCaseRunResult::Failure(TestCaseFailure::new("Can't do"))));

        let mut factory = DummyRunnerFactory::default();
        factory.responses = Some(errors);

        let runner = new_test_case_runner(factory);
        let report = run_tests(runner);

        assert_eq!(
            report,
            Some(
                r#"
Test case#3 failed:
Query:
SELECT 4

Expected error:
"Some error
"
But got another error:
"Can't do"
"#
                .to_owned()
            )
        );
    }

    #[test]
    fn test_case_from_files_test_case_mismatch() {
        let mut errors = HashMap::new();
        errors.insert(0, Ok(TestCaseRunResult::Success("Invalid\n  plan".to_string())));

        let mut factory = DummyRunnerFactory::default();
        factory.responses = Some(errors);

        let runner = new_test_case_runner(factory);
        let report = run_tests(runner);

        assert_eq!(
            report,
            Some(
                r#"
Test case#0 failed:
Query:
SELECT 1

Expected:
"Step1
Step2
"
But got:
"Invalid
  plan"
"#
                .to_owned()
            )
        );
    }

    #[test]
    fn test_case_from_files_test_case_runner_error() {
        let mut errors = HashMap::new();
        errors.insert(0, Err(()));

        let mut factory = DummyRunnerFactory::default();
        factory.responses = Some(errors);

        let runner = new_test_case_runner(factory);
        let report = run_tests(runner);

        assert_eq!(
            report,
            Some(
                r#"
Test case#0 failed:
Query:
SELECT 1

Error: test case returned an error.
TestRunner error: Dummy runner error!
"#
                .to_owned()
            )
        );
    }

    #[test]
    fn test_case_with_catalog() {
        let factory = DummyRunnerFactory::default();

        let mut runner = new_test_case_runner(factory);
        let tables = runner.prepare_catalog();

        let (cases, results) = runner.run_and_collect_results(tables);
        let report = runner.build_error_report(cases, results);

        assert_eq!(None, report);
    }

    fn new_test_case_runner<T>(factory: T) -> SqlTestCaseRunner<T>
    where
        T: TestCaseRunnerFactory,
    {
        let s = include_str!("test_runner.yaml");
        let SqlTestCaseSet { catalog, test_cases } = parse_test_cases(s).unwrap();
        SqlTestCaseRunner::new(factory, catalog, test_cases)
    }

    fn run_tests(mut runner: SqlTestCaseRunner<DummyRunnerFactory>) -> Option<String> {
        let tables = runner.prepare_catalog();
        let (cases, results) = runner.run_and_collect_results(tables);
        runner.build_error_report(cases, results)
    }

    #[derive(Default)]
    struct DummyRunnerFactory {
        fail_to_start: Option<usize>,
        responses: Option<HashMap<usize, Result<TestCaseRunResult, ()>>>,
    }

    struct DummyRunnerInstance {
        result: Result<TestCaseRunResult, TestRunnerError>,
    }

    impl TestCaseRunnerFactory for DummyRunnerFactory {
        type Runner = DummyRunnerInstance;

        fn new_runner(
            &self,
            test_case: &SqlTestCase,
            _tables: Vec<&TestTable>,
        ) -> Result<Self::Runner, TestRunnerError> {
            if self.fail_to_start == Some(test_case.id) {
                Err(TestRunnerError {
                    message: "Can't run".to_string(),
                    cause: None,
                })
            } else {
                let result = if let Some(responses) = &self.responses {
                    responses
                        .get(&test_case.id)
                        .cloned()
                        .unwrap_or_else(|| Ok(TestCaseRunResult::from_result(test_case.expected_result.clone())))
                } else {
                    Ok(TestCaseRunResult::from_result(test_case.expected_result.clone()))
                };
                Ok(DummyRunnerInstance {
                    result: match result {
                        Ok(result) => Ok(result),
                        Err(_) => Err(TestRunnerError {
                            message: "Dummy runner error!".to_string(),
                            cause: None,
                        }),
                    },
                })
            }
        }
    }

    impl TestCaseRunner for DummyRunnerInstance {
        fn run_test_case(self, _query: &str) -> Result<TestCaseRunResult, TestRunnerError> {
            self.result
        }
    }
}
