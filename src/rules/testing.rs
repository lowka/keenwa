use itertools::Itertools;
use std::fmt::Display;
use std::rc::Rc;

use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

use crate::error::OptimizerError;
use crate::memo::{MemoBuilder, MemoExpr, MemoExprFormatter, MemoGroupCallback, StringMemoFormatter};
use crate::meta::{ColumnId, MutableMetadata};
use crate::operators::relational::join::JoinCondition;
use crate::operators::relational::logical::{
    LogicalAggregate, LogicalExpr, LogicalExprVisitor, LogicalGet, LogicalJoin, LogicalSelect,
};
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::relational::{RelExpr, RelNode};
use crate::operators::scalar::ScalarExpr;
use crate::operators::{ExprMemo, Operator, OperatorExpr, OperatorMetadata, Properties, RelationalProperties};
use crate::properties::logical::LogicalProperties;
use crate::properties::physical::PhysicalProperties;
use crate::rules::{EvaluationResponse, Rule, RuleContext, RuleId, RuleIterator, RuleResult, RuleSet};

/// Expects that given expression does not match the given rule.
///
/// It calls [RuleTester::no_match] with `can_apply` flag set to `false`.
pub fn expect_no_match<T>(rule: T, expr: &LogicalExpr)
where
    T: Rule + 'static,
{
    let mut tester = RuleTester::new(rule);
    tester.no_match(expr, false)
}

/// Expects that given rule can be applied to the given expression and compares the result
/// expression with the expected value. See [RuleTester::apply].
pub fn expect_apply<T, S>(rule: T, expr: &LogicalExpr, expected: S)
where
    T: Rule + 'static,
    S: AsRef<str>,
{
    let mut tester = RuleTester::new(rule);
    tester.apply(expr, expected.as_ref())
}

/// Provides methods to test [optimization rules].
///
/// [optimization rules]: crate::rules::Rule
pub struct RuleTester {
    rule: Box<dyn Rule>,
    memo: ExprMemo,
    metadata: MutableMetadata,
}

impl RuleTester {
    /// Creates a tester for the given rule.
    pub fn new<T>(rule: T) -> Self
    where
        T: Rule + 'static,
    {
        struct Callback;
        impl MemoGroupCallback for Callback {
            type Expr = OperatorExpr;
            type Props = Properties;
            type Metadata = OperatorMetadata;

            fn new_group(
                &self,
                _expr: &Self::Expr,
                provided_props: &Self::Props,
                _metadata: &Self::Metadata,
            ) -> Result<Self::Props, OptimizerError> {
                Ok(provided_props.clone())
            }
        }

        let memo = MemoBuilder::new(Rc::new(MutableMetadata::new())).set_callback(Rc::new(Callback)).build();
        RuleTester {
            rule: Box::new(rule),
            memo,
            metadata: MutableMetadata::new(),
        }
    }

    /// Attempts to apply the rule to the given expression and then compares the result with the expected value.
    pub fn apply(&mut self, expr: &LogicalExpr, expected: &str) {
        let memo_expr = self.memo.insert_group(Operator::from(OperatorExpr::from(expr.clone())));

        let ctx = RuleContext::new(Rc::new(PhysicalProperties::none()), self.metadata.get_ref());
        let rule_match = self.rule.matches(&ctx, expr);
        assert!(rule_match.is_some(), "Rule does not match: {:?}", self.rule);

        let expr = memo_expr.expr();
        let result = self.rule.apply(&ctx, expr.relational().logical());
        let expr = match result {
            Ok(Some(RuleResult::Substitute(expr))) => Operator::from(OperatorExpr::from(expr)),
            Ok(Some(RuleResult::Implementation(expr))) => Operator::from(OperatorExpr::from(expr)),
            Ok(None) => panic!("Rule matched but not applied: {:?}", self.rule),
            Err(e) => panic!("Failed to apply a rule. Rule: {:?}. Error: {}", self.rule, e),
        };
        let new_expr = self.memo.insert_group(expr);
        let actual_expr = format_operator_tree(&new_expr);

        assert_eq!(actual_expr.trim_end(), expected.trim());
    }

    /// Matches the given expression to the rule and expects it not to match. If the match is unsuccessful
    /// and `can_apply` is `true` this method then calls [Rule::apply](crate::rules::Rule::apply)
    /// on the given expression. If that call does not return `Ok(None)` this methods fails.
    pub fn no_match(&mut self, expr: &LogicalExpr, can_apply: bool) {
        let memo_expr = self.memo.insert_group(Operator::from(OperatorExpr::from(expr.clone())));
        let expr_str = format_operator_tree(&memo_expr);

        let ctx = RuleContext::new(Rc::new(PhysicalProperties::none()), self.metadata.get_ref());
        let rule_match = self.rule.matches(&ctx, expr);
        assert!(
            rule_match.is_none(),
            "Rule should not have matched. Rule: {:?} match: {:?} expr:\n{}",
            self.rule,
            rule_match,
            expr_str
        );

        if can_apply {
            let expr = memo_expr.expr();
            let result = self.rule.apply(&ctx, expr.relational().logical());
            assert!(
                matches!(result, Ok(None)),
                "Rule should have not been applied. Rule: {:?} result: {:?} expr:\n{}",
                self.rule,
                result,
                expr_str
            )
        }
    }
}

/// A wrapper for a [RuleSet] that changes the behaviour of the underlying rule set.
///
/// [RuleSet]: crate::rules::RuleSet
#[derive(Debug)]
pub struct TestRuleSet<S> {
    rule_set: S,
    shuffle_rules: bool,
    explore_with_enforcer: bool,
}

impl<S> TestRuleSet<S> {
    pub fn new(rule_set: S, shuffle_rules: bool, explore_with_enforcer: bool) -> Self {
        TestRuleSet {
            rule_set,
            shuffle_rules,
            explore_with_enforcer,
        }
    }
}

impl<S> RuleSet for TestRuleSet<S>
where
    S: RuleSet,
{
    fn get_rules(&self) -> RuleIterator {
        if self.shuffle_rules {
            let mut rules: Vec<(&RuleId, &dyn Rule)> = self.rule_set.get_rules().collect();
            let mut rng = ThreadRng::default();
            rules.shuffle(&mut rng);

            RuleIterator::new(rules)
        } else {
            self.rule_set.get_rules()
        }
    }

    fn apply_rule(
        &self,
        rule_id: &RuleId,
        ctx: &RuleContext,
        expr: &LogicalExpr,
    ) -> Result<Option<RuleResult>, OptimizerError> {
        self.rule_set.apply_rule(rule_id, ctx, expr)
    }

    fn evaluate_properties(
        &self,
        expr: &PhysicalExpr,
        required_properties: &PhysicalProperties,
    ) -> Result<EvaluationResponse, OptimizerError> {
        self.rule_set.evaluate_properties(expr, required_properties)
    }

    fn create_enforcer(
        &self,
        required_properties: &PhysicalProperties,
        input: RelNode,
    ) -> Result<(PhysicalExpr, PhysicalProperties), OptimizerError> {
        self.rule_set.create_enforcer(required_properties, input)
    }

    fn can_explore_with_enforcer(&self, expr: &LogicalExpr, required_properties: &PhysicalProperties) -> bool {
        if self.explore_with_enforcer {
            self.rule_set.can_explore_with_enforcer(expr, required_properties)
        } else {
            false
        }
    }
}

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
    let fmt = StringMemoFormatter::new(&mut buf);
    let mut header = FormatHeader {
        fmt,
        written_exprs: vec![],
    };
    Operator::format_expr(expr.expr(), expr.props(), &mut header);
    let written_exprs = header.written_exprs;

    let mut fmt = FormatExprs {
        buf: &mut buf,
        depth: 0,
    };
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
                buf.push_str(format!("{}output cols: {:?}\n", padding, logical.output_columns()).as_str());
                if !props.required.is_empty() {
                    if let Some(ordering) = props.required().ordering() {
                        buf.push_str(format!("{}ordering: {:?}\n", padding, ordering.columns()).as_str());
                    }
                }
            }
        }

        let mut buf = String::new();

        // properties first
        if let Some(true) = &self.write_properties {
            write_properties(&mut buf, operator.props(), "");
        }

        buf.push_str(format_operator_tree(operator).as_str());
        buf.push('\n');

        // properties last
        if let Some(false) = self.write_properties {
            write_properties(&mut buf, operator.props(), "  ");
        }

        for formatter in self.formatters {
            formatter.write_operator(operator, &mut buf);
        }

        buf
    }
}

struct FormatHeader<'b> {
    fmt: StringMemoFormatter<'b>,
    written_exprs: Vec<String>,
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
        let mut fmt = StringMemoFormatter::new(&mut buf);

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
}

struct FormatExprs<'b> {
    buf: &'b mut String,
    depth: usize,
}

impl FormatExprs<'_> {
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
        self.buf.push('\n');
        self.pad_depth(' ');
        self.buf.push_str(name);
        self.buf.push_str(": ");

        let expr = input.as_ref();
        let fmt = StringMemoFormatter::new(self.buf);
        let mut header = FormatHeader {
            fmt,
            written_exprs: vec![],
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
}

/// [ExtraFormatter] that writes plans of sub queries.
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
        let mut new_line = true;
        // Sub queries from columns:
        for column in self.metadata.get_columns() {
            if let Some(ScalarExpr::SubQuery(query)) = column.expr() {
                if new_line {
                    new_line = false;
                    buf.push('\n');
                }
                buf.push_str(format!("Sub query from column {}:\n", column.id()).as_str());
                buf.push_str(format_operator_tree(query.mexpr()).as_str());
                buf.push('\n');
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
            // Alias(SubQuery(_), _) is not covered ?
            OperatorExpr::Scalar(ScalarExpr::SubQuery(query)) => {
                buf.push_str("Sub query from scalar operator:\n".to_string().as_str());
                buf.push_str(format_operator_tree(query.mexpr()).as_str());
                buf.push('\n');
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

            fn write_subquery(&mut self, description: &str, subquery: &RelNode) {
                self.buf.push_str(format!("Sub query from {}:\n", description).as_str());
                self.buf.push_str(format_operator_tree(subquery.mexpr()).as_str());
                self.buf.push('\n');
            }
        }

        impl<'a> LogicalExprVisitor for CollectSubQueries<'a> {
            fn pre_visit_subquery(&mut self, expr: &LogicalExpr, subquery: &RelNode) -> Result<bool, OptimizerError> {
                match expr {
                    LogicalExpr::Select(LogicalSelect { filter, .. }) => {
                        if let Some(filter) = filter {
                            self.add_new_line();
                            self.write_subquery(format!("filter {}", filter.expr()).as_str(), subquery);
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
                        self.write_subquery(format!("aggregate {}", aggr_str).as_str(), subquery);

                        Ok(true)
                    }
                    LogicalExpr::Join(LogicalJoin {
                        condition: JoinCondition::On(expr),
                        ..
                    }) => {
                        self.add_new_line();
                        self.write_subquery(format!("join condition: {}", expr.expr().expr()).as_str(), subquery);
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

/// Creates an instance of [LogicalExpr::Get] with the given columns used as output columns.
/// `columns` are also used to populate `output_columns` of [LogicalProperties] of the created expression.
pub fn new_src(src: &str, columns: Vec<ColumnId>) -> RelNode {
    let expr = LogicalExpr::Get(LogicalGet {
        source: src.into(),
        columns: columns.clone(),
    });
    let props = RelationalProperties {
        logical: LogicalProperties::new(columns, None),
        required: Default::default(),
    };
    RelNode::new(RelExpr::Logical(Box::new(expr)), props)
}

#[cfg(test)]
mod test {
    use crate::memo::ScalarNode;
    use crate::operators::relational::logical::{LogicalAggregate, LogicalExpr, LogicalGet, LogicalProjection};
    use crate::operators::relational::RelNode;
    use crate::operators::scalar::value::ScalarValue;
    use crate::operators::scalar::ScalarExpr;
    use crate::operators::{Operator, OperatorExpr};
    use crate::rules::testing::format_operator_tree;

    #[test]
    fn test_single_line_fmt() {
        let from_a = LogicalExpr::Get(LogicalGet {
            source: "A".to_string(),
            columns: vec![1, 2, 3],
        });
        let p1 = LogicalProjection {
            input: RelNode::from(from_a),
            exprs: vec![ScalarNode::from(ScalarExpr::Column(1)), ScalarNode::from(ScalarExpr::Column(2))],
            columns: vec![1, 2],
        };
        let p2 = LogicalProjection {
            input: RelNode::from(LogicalExpr::Projection(p1)),
            exprs: vec![ScalarNode::from(ScalarExpr::Column(1))],
            columns: vec![1],
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
        let from_a = LogicalExpr::Get(LogicalGet {
            source: "A".to_string(),
            columns: vec![1, 2],
        });
        let aggr = LogicalExpr::Aggregate(LogicalAggregate {
            input: Operator::from(OperatorExpr::from(from_a)).into(),
            aggr_exprs: vec![ScalarNode::from(ScalarExpr::Scalar(ScalarValue::Int32(10)))],
            group_exprs: vec![ScalarNode::from(ScalarExpr::Column(1))],
            having: None,
            columns: vec![3, 4],
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

    fn expect_formatted(expr: &LogicalExpr, expected: &str) {
        let expr = Operator::from(OperatorExpr::from(expr.clone()));
        let str = format_operator_tree(&expr);
        let str = format!("\n{}\n", str);
        assert_eq!(str, expected, "expected format");
    }
}
