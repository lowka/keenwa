use crate::datatypes::DataType;
use crate::sql::testing::{SqlTestCase, SqlTestCaseSet, TestCaseFailure, TestCatalog, TestTable};
use serde::{Deserialize, Serialize};

/// An item in a database catalog.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum CatalogItem {
    /// Database table
    Table {
        /// table name
        table: String,
        /// columns: <col1>:<col1-type>, <col1>:<col1-type> ...
        columns: String,
    },
}

/// A test case.
#[derive(Debug, Deserialize, Serialize)]
struct TestCase {
    /// Catalog, If present modifies the existing catalog.
    catalog: Option<Vec<CatalogItem>>,
    /// Test case that contains a single query. If present `queries` must be not be set.
    query: Option<String>,
    /// Test case contains multiple queries. If present `query` must be not be set.
    queries: Option<Vec<String>>,
    /// Successful result. If present `error` must be not be set.
    ok: Option<String>,
    /// Unsuccessful result. If present `ok` must be not be set.
    error: Option<String>,
}

/// Reads a YAML-document from the given string that contains multiple test cases and converts it into a [SqlTestCaseSet].
pub fn parse_test_cases(str: &str) -> Result<SqlTestCaseSet, String> {
    let mut buf = String::new();
    let mut catalog: Option<TestCatalog> = None;
    let mut test_cases: Vec<SqlTestCase> = vec![];

    for line in str.lines() {
        if line.trim_end() == "---" {
            let test_case = parse_test_case(&buf)?;

            if let Some(test_case) = test_case {
                if test_cases.is_empty() && catalog.is_none() {
                    match try_validate_if_catalogue(&test_case) {
                        Ok(true) => {
                            catalog = yaml_to_catalog(test_case.catalog)?;
                            buf.clear();
                            continue;
                        }
                        Ok(_) => {}
                        Err(err) => return Err(format!("Invalid catalog: {}\n\nError:\n{}", buf, err)),
                    }
                }

                parse_sql_test_case(test_case, &buf, &mut test_cases)?;

                buf.clear();
            }
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }

    let buf = buf.trim_end();
    if !buf.is_empty() {
        let test_case = parse_test_case(buf)?;
        if let Some(test_case) = test_case {
            parse_sql_test_case(test_case, buf, &mut test_cases)?;
        }
    }

    if test_cases.is_empty() {
        Err(format!("No test cases"))
    } else {
        Ok(SqlTestCaseSet { catalog, test_cases })
    }
}

fn parse_test_case(str: &str) -> Result<Option<TestCase>, String> {
    let mut buf = String::new();
    for line in str.lines() {
        if line.starts_with("#") {
            // remove comments
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    if buf.is_empty() {
        Ok(None)
    } else {
        let test_case = serde_yaml::from_str::<TestCase>(&buf);
        match test_case {
            Ok(test_case) => Ok(Some(test_case)),
            Err(e) => Err(format!("{}", e)),
        }
    }
}

fn parse_sql_test_case(test_case: TestCase, str: &str, test_cases: &mut Vec<SqlTestCase>) -> Result<(), String> {
    match validate_test_case(&test_case) {
        Ok(_) => {}
        Err(err) => return Err(format!("Invalid test case:\n{}\n\nError:\n{}", str, err)),
    }

    let test_case = yaml_to_test_case(test_case, test_cases.len());
    match test_case {
        Ok(test_case) => {
            test_cases.push(test_case);
        }
        Err(err) => return Err(format!("Failed to parse a test case:\n{}\n\nError:\n{}", str, err)),
    }
    Ok(())
}

fn try_validate_if_catalogue(test_case: &TestCase) -> Result<bool, String> {
    if test_case.catalog.is_none() || test_case.query.is_some() && test_case.queries.is_some() {
        Ok(false)
    } else {
        if test_case.ok.is_some() || test_case.error.is_some() {
            Err(format!("Catalog: neither `ok` nor `error` can be specified"))
        } else {
            Ok(true)
        }
    }
}

fn validate_test_case(test_case: &TestCase) -> Result<(), String> {
    if test_case.query.is_none() && test_case.queries.is_none() {
        return Err(format!("Test case: Neither `query` nor `queries` has been specified"));
    }

    if test_case.query.is_some() && test_case.queries.is_some() {
        return Err(format!("Test case: Both `query` and `queries` have been specified"));
    }

    if test_case.ok.is_some() && test_case.error.is_some() {
        return Err(format!("Test case: Both `ok` and `error` have been specified"));
    }

    // if both ok and error are absent then it is an error.

    Ok(())
}

fn yaml_to_test_case(test_case: TestCase, id: usize) -> Result<SqlTestCase, String> {
    let TestCase {
        catalog,
        query,
        queries,
        ok,
        error,
    } = test_case;

    let queries = queries.unwrap_or_else(|| vec![query.unwrap()]);
    let catalog = yaml_to_catalog(catalog)?;

    Ok(SqlTestCase {
        id,
        queries,
        catalog,
        expected_result: yaml_to_result(ok, error)?,
    })
}

fn parse_table_columns(str: &str) -> Result<Vec<(String, DataType)>, String> {
    let str = str.trim();
    let mut columns = vec![];

    for column in str.split(',') {
        match column.trim().split_once(":") {
            Some((column_name, data_type)) => {
                let data_type = match data_type {
                    _ if data_type.eq_ignore_ascii_case("bool") => DataType::Bool,
                    _ if data_type.eq_ignore_ascii_case("int32") => DataType::Int32,
                    _ if data_type.eq_ignore_ascii_case("string") => DataType::String,
                    _ => {
                        return Err(format!(
                            "Unable to parse column definition. Column: {} Unexpected data type: {}",
                            column_name, data_type
                        ))
                    }
                };
                columns.push((String::from(column_name), data_type));
            }
            _ => return Err(format!("Invalid column definition. Expected <name>:<type> but got: {}", column)),
        }
    }

    Ok(columns)
}

fn yaml_to_catalog(catalog_items: Option<Vec<CatalogItem>>) -> Result<Option<TestCatalog>, String> {
    if let Some(items) = catalog_items {
        let tables: Result<Vec<TestTable>, String> = items
            .into_iter()
            .map(|item| match item {
                CatalogItem::Table { table, columns } => {
                    let columns = parse_table_columns(&columns)?;
                    Ok(TestTable { name: table, columns })
                }
            })
            .collect();
        Ok(Some(TestCatalog { tables: tables? }))
    } else {
        Ok(None)
    }
}

fn yaml_to_result(ok: Option<String>, error: Option<String>) -> Result<Result<String, TestCaseFailure>, String> {
    if let Some(ok) = ok {
        Ok(Ok(ok))
    } else if let Some(error) = error {
        Ok(Err(TestCaseFailure::new(error)))
    } else {
        Ok(Err(TestCaseFailure::no_message()))
    }
}

#[cfg(test)]
mod test {
    use crate::sql::testing::parser::{parse_test_case, parse_test_cases};
    use crate::sql::testing::TestCaseFailure;

    #[test]
    fn test_parse_test_case() {
        let test_case = r#"
query: |
  SELECT a
  FROM t
ok: |
  Plan o=1
    inner: Plan o=2
"#;
        let test_case = parse_test_case(test_case).unwrap().unwrap();

        assert_eq!(test_case.catalog.map(|c| format!("{:?}", c)), None, "catalog");
        assert_eq!(test_case.query, Some(String::from("SELECT a\nFROM t\n")), "query");
        assert_eq!(test_case.queries, None, "queries");
        assert_eq!(test_case.ok, Some(String::from("Plan o=1\n  inner: Plan o=2\n")), "ok");
        assert_eq!(test_case.error, None, "error");
    }

    #[test]
    fn test_case_with_catalog() {
        let test_case = r#"
catalog:
  - table: a
    columns: a1:int32, a2:string        
query: |
  SELECT a
  FROM t
ok: |
  Plan o=1
    inner: Plan o=2
"#;
        let test_case = parse_test_case(test_case).unwrap().unwrap();
        let catalog = String::from("[Table { table: \"a\", columns: \"a1:int32, a2:string\" }]");

        assert_eq!(test_case.catalog.map(|c| format!("{:?}", c)), Some(catalog), "catalog");
        assert_eq!(test_case.query, Some(String::from("SELECT a\nFROM t\n")), "query");
        assert_eq!(test_case.queries, None, "queries");
        assert_eq!(test_case.ok, Some(String::from("Plan o=1\n  inner: Plan o=2\n")), "ok");
        assert_eq!(test_case.error, None, "error");
    }

    #[test]
    fn test_with_initial_catalog() {
        let test_case = r#"
catalog:
  - table: a1
    columns: a11:int32        
---        
query: q1
ok: ok
"#;
        let test_case_set = parse_test_cases(test_case).unwrap();
        assert!(test_case_set.catalog.is_some(), "catalog");
        assert_eq!(1, test_case_set.test_cases.len(), "test cases");
    }

    #[test]
    fn test_multiple_test_cases() {
        let test_case = r#"
query: q2
error: err   
---        
query: q1
ok: ok
"#;
        let test_case_set = parse_test_cases(test_case).unwrap();
        assert_eq!(2, test_case_set.test_cases.len(), "test cases");
    }

    #[test]
    fn test_assume_an_error_when_both_ok_and_error_are_not_set() {
        let test_case = r#"
query: |
  SELECT a
  FROM t
error: ~
"#;
        let test_case = &parse_test_cases(test_case).unwrap().test_cases[0];

        assert_eq!(test_case.queries, vec![String::from("SELECT a\nFROM t\n")], "queries");
        assert_eq!(test_case.expected_result, Err(TestCaseFailure::no_message()), "error");
    }

    #[test]
    fn test_reject_empty_test_case() {
        let test_case = r#"

"#;
        expect_error(test_case, "No test cases");
    }

    #[test]
    fn test_reject_test_case_when_both_query_and_queries_are_present() {
        let test_case = r#"
query: q1
queries:
 - q1
 - q2
ok: ok
"#;
        expect_error(test_case, "Test case: Both `query` and `queries` have been specified");
    }

    #[test]
    fn test_reject_test_case_when_neither_query_nor_queries_is_present() {
        let test_case = r#"
ok: 1
"#;
        expect_error(test_case, "Test case: Neither `query` nor `queries` has been specified");
    }

    #[test]
    fn test_reject_test_case_when_both_ok_error_are_present() {
        let test_case = r#"
query: q1
ok: ok
error: Err    
"#;
        expect_error(test_case, "Test case: Both `ok` and `error` have been specified");
    }

    fn expect_error(test_cases: &str, error_message: &str) {
        let err = parse_test_cases(test_cases).expect_err("Should err");
        assert!(err.ends_with(error_message), "Expected error: {} but got:\n{}", error_message, err);
    }
}
