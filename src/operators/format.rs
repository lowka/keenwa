use crate::error::OptimizerError;
use crate::memo::{format_memo, Memo, MemoExpr, MemoExprFormatter, MemoFormatterFlags, StringMemoFormatter};
use crate::meta::MutableMetadata;
use crate::operators::relational::join::JoinCondition;
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalExpr, LogicalExprVisitor, LogicalJoin, LogicalSelect,
};
use crate::operators::relational::{RelExpr, RelNode};
use crate::operators::scalar::get_subquery;
use crate::operators::{Operator, OperatorExpr, Properties};
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use itertools::Itertools;
use std::fmt::Display;
use std::rc::Rc;

/// Builds the following textual representation of the given operator tree:
///
/// ```text:
///  RootExpr [root-expr-properties]
///    Expr_0 [expr_0-properties]
///      ...
///        LeafExpr_0 [leaf-expr_0_properties]
///        ...
///    ...
///    Expr_n [expr_n-properties]
/// ```
pub fn format_operator_tree(expr: &Operator) -> String {
    // Formatting is done using to formatters:
    // 1. FormatHeader writes name, source, values and buffers exprs (written by write_exprs).
    // 2. then FormatExprs writes the child expressions and appends expressions buffered by FormatHeader.

    let mut buf = String::new();
    let flags = MemoFormatterFlags::All;
    let fmt = StringMemoFormatter::new(&mut buf, flags);
    let mut header = FormatHeader::new(fmt, flags);
    Operator::format_expr(expr.expr(), expr.props(), &mut header);

    let written_exprs = header.written_exprs;
    let mut fmt = FormatExprs::new(&mut buf, flags);
    fmt.write_exprs(written_exprs);

    Operator::format_expr(expr.expr(), expr.props(), &mut fmt);

    buf
}

/// Build a textual representation of an operator tree.
pub struct OperatorTreeFormatter {
    // If Some: Properties first/last
    // If None: do not write
    write_properties: Option<bool>,
    formatters: Vec<Box<dyn OperatorFormatter>>,
}

/// Allows to write additional information to [OperatorTreeFormatter].
pub trait OperatorFormatter {
    /// Writes additional information to the given buffer.
    fn write_operator(&self, operator: &Operator, buf: &mut String);
}

impl OperatorTreeFormatter {
    /// Creates a new instance of `OperatorTreeFormatter`.
    pub fn new() -> OperatorTreeFormatter {
        OperatorTreeFormatter {
            write_properties: None,
            formatters: vec![],
        }
    }

    /// Specifies how to write output columns. Formatter write properties then an operator tree.
    pub fn properties_first(mut self) -> Self {
        self.write_properties = Some(true);
        self
    }

    /// Specifies how to write output columns. Formatter write an operator then write properties.
    pub fn properties_last(mut self) -> Self {
        self.write_properties = Some(false);
        self
    }

    /// Adds additional formatter.
    pub fn add_formatter(mut self, f: Box<dyn OperatorFormatter>) -> Self {
        self.formatters.push(f);
        self
    }

    /// Produces a string representation of the given operator tree.
    pub fn format(self, operator: &Operator) -> String {
        fn write_properties(buf: &mut String, props: &Properties, padding: &str) {
            if let Properties::Relational(props) = props {
                let logical = props.logical();
                let output_columns = logical.output_columns().iter().map(|id| format!("{}", id)).join(", ");
                buf.push_str(format!("{}output cols: [{}]", padding, output_columns).as_str());
            }
        }

        fn add_new_line(buf: &mut String, condition: bool) {
            if condition {
                buf.push('\n');
            }
        }

        let mut buf = String::new();

        // properties first
        if let Some(true) = &self.write_properties {
            write_properties(&mut buf, operator.props(), "");
            add_new_line(&mut buf, true);
        }

        buf.push_str(format_operator_tree(operator).as_str());

        // properties last
        if let Some(false) = self.write_properties {
            add_new_line(&mut buf, true);
            write_properties(&mut buf, operator.props(), "  ");
        }

        add_new_line(&mut buf, !self.formatters.is_empty());

        for formatter in self.formatters {
            formatter.write_operator(operator, &mut buf);
        }

        buf
    }
}

struct FormatHeader<'a> {
    fmt: StringMemoFormatter<'a>,
    written_exprs: Vec<String>,
    flags: MemoFormatterFlags,
}

impl<'a> FormatHeader<'a> {
    fn new(fmt: StringMemoFormatter<'a>, flags: MemoFormatterFlags) -> Self {
        FormatHeader {
            fmt,
            written_exprs: vec![],
            flags,
        }
    }
}

impl MemoExprFormatter for FormatHeader<'_> {
    fn write_name(&mut self, name: &str) {
        self.fmt.write_name(name);
    }

    fn write_source(&mut self, source: &str) {
        self.fmt.write_source(source);
    }

    fn write_expr<T>(&mut self, _name: &str, _input: impl AsRef<T>)
    where
        T: MemoExpr,
    {
        //exprs are written by FormatExprs
    }

    fn write_exprs<T>(&mut self, name: &str, input: impl ExactSizeIterator<Item = impl AsRef<T>>)
    where
        T: MemoExpr,
    {
        // Formats expression list to add it to a buffer later (see FormatExprs::write_expr)
        if input.len() == 0 {
            return;
        }
        let mut buf = String::new();
        let mut fmt = StringMemoFormatter::new(&mut buf, self.flags);

        fmt.push_str(name);
        fmt.push_str(": [");
        for (i, expr) in input.enumerate() {
            if i > 0 {
                fmt.push_str(", ");
            }
            fmt.write_expr_name(false);
            let expr = expr.as_ref();
            T::format_expr(expr.expr(), expr.props(), &mut fmt);
            fmt.write_expr_name(true);
        }
        fmt.push(']');

        self.written_exprs.push(buf);
    }

    fn write_value<D>(&mut self, name: &str, value: D)
    where
        D: Display,
    {
        self.fmt.write_value(name, value);
    }

    fn write_values<D>(&mut self, name: &str, values: &[D])
    where
        D: Display,
    {
        self.fmt.write_values(name, values);
    }

    fn flags(&self) -> &MemoFormatterFlags {
        self.fmt.flags()
    }
}

struct FormatExprs<'b> {
    buf: &'b mut String,
    depth: usize,
    flags: MemoFormatterFlags,
}

impl<'a> FormatExprs<'a> {
    fn new(buf: &'a mut String, flags: MemoFormatterFlags) -> Self {
        FormatExprs { buf, depth: 0, flags }
    }

    fn pad_depth(&mut self, c: char) {
        for _ in 1..=self.depth * 2 {
            self.buf.push(c);
        }
    }

    fn write_exprs(&mut self, written_exprs: Vec<String>) {
        // Writes expressions collected by FormatHeader.
        // if there are multiple expression lists writes them at different lines (Aggregate).
        // If there is only one expression list writes it at the same line (Projection)
        let len = written_exprs.len();
        for written_expr in written_exprs {
            if len > 1 {
                self.buf.push('\n');
                self.depth += 1;
                self.pad_depth(' ');
                self.depth -= 1;
            } else {
                self.buf.push(' ');
            }
            self.buf.push_str(written_expr.as_str());
        }
    }
}

impl MemoExprFormatter for FormatExprs<'_> {
    fn write_name(&mut self, _name: &str) {
        // name is written by FormatHeader
    }

    fn write_source(&mut self, _source: &str) {
        // source is written by FormatHeader
    }

    fn write_expr<T>(&mut self, name: &str, input: impl AsRef<T>)
    where
        T: MemoExpr,
    {
        self.depth += 1;
        if self.depth >= 1 {
            self.buf.push('\n');
        }
        self.pad_depth(' ');
        self.buf.push_str(name);
        self.buf.push_str(": ");

        let expr = input.as_ref();
        let fmt = StringMemoFormatter::new(self.buf, self.flags);
        let mut header = FormatHeader {
            fmt,
            written_exprs: vec![],
            flags: self.flags,
        };

        T::format_expr(expr.expr(), expr.props(), &mut header);

        let written_exprs = header.written_exprs;
        self.write_exprs(written_exprs);

        T::format_expr(expr.expr(), expr.props(), self);

        self.depth -= 1;
    }

    fn write_exprs<T>(&mut self, _name: &str, _input: impl ExactSizeIterator<Item = impl AsRef<T>>)
    where
        T: MemoExpr,
    {
        // exprs are written by FormatHeader
    }

    fn write_value<D>(&mut self, _name: &str, _value: D)
    where
        D: Display,
    {
        // values are written by FormatHeader
    }

    fn write_values<D>(&mut self, _name: &str, _values: &[D])
    where
        D: Display,
    {
        // values are written by FormatHeader
    }

    fn flags(&self) -> &MemoFormatterFlags {
        &MemoFormatterFlags::None
    }
}

/// [OperatorFormatter] that writes plans of sub queries.
pub struct SubQueriesFormatter {
    metadata: Rc<MutableMetadata>,
}

impl SubQueriesFormatter {
    /// Creates an instance of [SubQueriesFormatter].
    pub fn new(metadata: Rc<MutableMetadata>) -> Self {
        Self { metadata }
    }
}

impl OperatorFormatter for SubQueriesFormatter {
    fn write_operator(&self, operator: &Operator, buf: &mut String) {
        fn add_subquery(buf: &mut String, subquery: &RelNode, description: &str) {
            let operator_str = format_operator_tree(subquery.mexpr());
            buf.push_str(format!("Sub query from {}:\n{}\n", description, operator_str).as_str());
        }

        let mut new_line = true;
        // Sub queries from columns:
        for column in self.metadata.get_columns() {
            if let Some(query) = column.expr().and_then(get_subquery) {
                if new_line {
                    new_line = false;
                    buf.push('\n');
                }
                add_subquery(buf, query, format!("column {}", column.id()).as_str());
            }
        }

        // Sub queries from other expressions:
        match operator.expr() {
            OperatorExpr::Relational(RelExpr::Logical(expr)) => {
                let mut visitor = CollectSubQueries {
                    buf,
                    new_line: &mut new_line,
                };
                // CollectSubQueries never returns an error.
                expr.accept(&mut visitor).unwrap();
            }
            OperatorExpr::Scalar(expr) => {
                if let Some(query) = get_subquery(expr) {
                    add_subquery(buf, query, "scalar operator");
                }
            }
            _ => {}
        }

        struct CollectSubQueries<'a> {
            buf: &'a mut String,
            new_line: &'a mut bool,
        }

        impl<'a> CollectSubQueries<'a> {
            fn add_new_line(&mut self) {
                if *self.new_line {
                    self.buf.push('\n');
                    *self.new_line = false;
                }
            }
        }

        impl<'a> LogicalExprVisitor for CollectSubQueries<'a> {
            fn pre_visit_subquery(&mut self, expr: &LogicalExpr, subquery: &RelNode) -> Result<bool, OptimizerError> {
                match expr {
                    LogicalExpr::Select(LogicalSelect { filter, .. }) => {
                        if let Some(filter) = filter {
                            self.add_new_line();
                            add_subquery(self.buf, subquery, format!("filter {}", filter.expr()).as_str());
                            Ok(true)
                        } else {
                            Ok(false)
                        }
                    }
                    LogicalExpr::Aggregate(LogicalAggregate { aggr_exprs, .. }) => {
                        //TODO: Exclude aggregate exprs and print only group by and having.
                        // Because aggregate exprs has already been written from the metadata.
                        self.add_new_line();
                        let aggr_str: String = aggr_exprs.iter().map(|s| s.expr()).join(", ");
                        add_subquery(self.buf, subquery, format!("aggregate {}", aggr_str).as_str());

                        Ok(true)
                    }
                    LogicalExpr::Join(LogicalJoin {
                        condition: JoinCondition::On(expr),
                        ..
                    }) => {
                        self.add_new_line();
                        add_subquery(self.buf, subquery, format!("join condition: {}", expr.expr().expr()).as_str());
                        Ok(true)
                    }
                    // subqueries from projection list are taken from the metadata.
                    _ => Ok(false),
                }
            }

            fn post_visit(&mut self, _expr: &LogicalExpr) -> Result<(), OptimizerError> {
                Ok(())
            }
        }
    }
}

pub(crate) struct PropertiesFormatter<'a, T> {
    fmt: &'a mut T,
}

impl<'a, T> PropertiesFormatter<'a, T>
where
    T: MemoExprFormatter,
{
    pub fn new(fmt: &'a mut T) -> Self {
        PropertiesFormatter { fmt }
    }

    pub fn format(mut self, props: &Properties) {
        match props {
            Properties::Relational(props) => {
                self.format_logical(props.logical());
                self.format_physical(props.physical());
            }
            Properties::Scalar(props) => {
                if props.has_correlated_sub_queries {
                    self.fmt.write_value("", "<correlated>");
                }
            }
        }
    }

    fn format_logical(&mut self, logical: &LogicalProperties) {
        let outer_columns = &logical.outer_columns;
        if !outer_columns.is_empty() {
            self.fmt.write_values("outer_cols", outer_columns)
        }
    }

    fn format_physical(&mut self, physical: &PhysicalProperties) {
        let required = physical.required.as_ref();
        if let Some(ordering) = required.and_then(|r| r.ordering()) {
            self.fmt.write_values("ordering", ordering.columns());
        }
    }
}

/// [OperatorFormatter] that writes metadata.
pub struct AppendMetadata {
    metadata: Rc<MutableMetadata>,
}

impl AppendMetadata {
    /// Creates a new instance of [AppendMetadata] formatter.
    pub fn new(metadata: Rc<MutableMetadata>) -> Self {
        AppendMetadata { metadata }
    }
}

impl OperatorFormatter for AppendMetadata {
    fn write_operator(&self, _operator: &Operator, buf: &mut String) {
        buf.push_str("Metadata:\n");

        for column in self.metadata.get_columns() {
            let id = column.id();
            let expr = column.expr();
            let relation_id = column.relation_id();
            let table_name = relation_id.map(|id| String::from(self.metadata.get_relation(&id).name()));
            let column_name = column.name();
            let column_info = match (expr, table_name) {
                (None, None) => format!("  col:{} {} {:?}", id, column_name, column.data_type()),
                (None, Some(table)) => {
                    format!("  col:{} {}.{} {:?}", id, table, column_name, column.data_type())
                }
                (Some(expr), _) => {
                    format!("  col:{} {} {:?}, expr: {}", id, column_name, column.data_type(), expr)
                }
            };
            buf.push_str(column_info.as_str());
            buf.push('\n');
        }
    }
}

/// [OperatorFormatter] that writes contents of a memo.
pub struct AppendMemo<T> {
    memo: Memo<Operator, T>,
}

impl<T> AppendMemo<T> {
    /// Creates a new instance of [AppendMemo] formatter.
    pub fn new(memo: Memo<Operator, T>) -> Self {
        AppendMemo { memo }
    }
}

impl<T> OperatorFormatter for AppendMemo<T> {
    fn write_operator(&self, _operator: &Operator, buf: &mut String) {
        buf.push_str("Memo:\n");
        // TODO: Do not include properties.
        let memo_as_string = format_memo(&self.memo);
        let lines = memo_as_string
            .split('\n')
            .map(|l| {
                if !l.is_empty() {
                    let mut s = String::new();
                    s.push_str("  ");
                    s.push_str(l);
                    s
                } else {
                    l.to_string()
                }
            })
            .join("\n");

        buf.push_str(lines.as_str());
    }
}

#[cfg(test)]
mod test {
    use crate::memo::ScalarNode;
    use crate::meta::testing::TestMetadata;
    use crate::operators::format::{format_operator_tree, OperatorTreeFormatter};
    use crate::operators::relational::logical::{
        LogicalAggregate, LogicalEmpty, LogicalExpr, LogicalGet, LogicalProjection,
    };
    use crate::operators::relational::RelNode;
    use crate::operators::scalar::value::{Scalar, ScalarValue};
    use crate::operators::scalar::ScalarExpr;
    use crate::operators::{Operator, OperatorExpr, Properties, RelationalProperties};
    use crate::properties::logical::LogicalProperties;
    use crate::properties::physical::{PhysicalProperties, RequiredProperties};
    use crate::properties::OrderingChoice;

    #[test]
    fn test_single_line_fmt() {
        let mut metadata = TestMetadata::with_tables(vec!["A"]);

        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();
        let col3 = metadata.column("A").build();

        let from_a = LogicalExpr::Get(LogicalGet {
            source: "A".to_string(),
            columns: vec![col1, col2, col3],
        });
        let p1 = LogicalProjection {
            input: RelNode::from(from_a),
            exprs: vec![ScalarNode::from(ScalarExpr::Column(col1)), ScalarNode::from(ScalarExpr::Column(col2))],
            columns: vec![col1, col2],
        };
        let p2 = LogicalProjection {
            input: RelNode::from(LogicalExpr::Projection(p1)),
            exprs: vec![ScalarNode::from(ScalarExpr::Column(col1))],
            columns: vec![col1],
        };
        let projection = LogicalExpr::Projection(p2);
        expect_formatted(
            &projection,
            r#"
LogicalProjection cols=[1] exprs: [col:1]
  input: LogicalProjection cols=[1, 2] exprs: [col:1, col:2]
    input: LogicalGet A cols=[1, 2, 3]
"#,
        )
    }

    #[test]
    fn test_multiline_fmt() {
        let mut metadata = TestMetadata::with_tables(vec!["A"]);

        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();
        let col3 = metadata.synthetic_column().build();
        let col4 = metadata.synthetic_column().build();

        let from_a = LogicalExpr::Get(LogicalGet {
            source: "A".to_string(),
            columns: vec![col1, col2],
        });
        let aggr = LogicalExpr::Aggregate(LogicalAggregate {
            input: RelNode::from(from_a),
            aggr_exprs: vec![ScalarNode::from(ScalarExpr::Scalar(10.get_value()))],
            group_exprs: vec![ScalarNode::from(ScalarExpr::Column(col1))],
            having: None,
            columns: vec![col3, col4],
        });

        expect_formatted(
            &aggr,
            r#"
LogicalAggregate cols=[3, 4]
  aggr_exprs: [10]
  group_exprs: [col:1]
  input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_format_expr_with_properties() {
        let mut metadata = TestMetadata::with_tables(vec!["A"]);

        let col1 = metadata.column("A").build();
        let col2 = metadata.synthetic_column().build();

        let empty = LogicalExpr::Empty(LogicalEmpty { return_one_row: true });

        let project_expr = LogicalExpr::Projection(LogicalProjection {
            input: RelNode::from(empty),
            exprs: vec![ScalarNode::from(ScalarExpr::Column(col1)), ScalarNode::from(ScalarExpr::Column(col2))],
            columns: vec![col1, col2],
        });

        let project_props = Properties::Relational(RelationalProperties {
            logical: LogicalProperties {
                output_columns: vec![col1, col2],
                outer_columns: vec![col1],
                has_correlated_subqueries: false,
                statistics: None,
            },
            physical: PhysicalProperties {
                required: Some(RequiredProperties::new_with_ordering(OrderingChoice::from_columns(vec![col1]))),
                presentation: None,
            },
        });

        let fmt = OperatorTreeFormatter::new().properties_last();
        let str = fmt.format(&Operator::new(OperatorExpr::from(project_expr), project_props));

        expect_formatted_tree(
            r#"
LogicalProjection cols=[1, 2] outer_cols=[1] ordering=[+1] exprs: [col:1, col:2]
  input: LogicalEmpty return_one_row=true
  output cols: [1, 2]
"#,
            str.as_str(),
        );
    }

    fn expect_formatted(expr: &LogicalExpr, expected: &str) {
        let expr = Operator::from(OperatorExpr::from(expr.clone()));
        let str = format_operator_tree(&expr);
        expect_formatted_tree(expected, str.as_str())
    }

    fn expect_formatted_tree(expected: &str, actual: &str) {
        assert_eq!(actual, expected.trim(), "expected format");
    }
}
