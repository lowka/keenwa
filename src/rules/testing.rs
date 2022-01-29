use crate::error::OptimizerError;
use crate::memo::{MemoBuilder, MemoExpr, MemoExprFormatter, MemoGroupCallback, StringMemoFormatter};
use crate::meta::MutableMetadata;
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::relational::RelNode;
use crate::operators::{ExprMemo, Operator, OperatorExpr, OperatorMetadata, Properties};
use crate::properties::physical::PhysicalProperties;
use crate::rules::{Rule, RuleContext, RuleId, RuleIterator, RuleResult, RuleSet};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use std::fmt::Display;
use std::rc::Rc;

/// Expects that given expression does not match the given rule. See [RuleTester::no_match].
pub fn expect_no_match<T>(rule: T, expr: &LogicalExpr)
where
    T: Rule + 'static,
{
    let mut tester = RuleTester::new(rule);
    tester.no_match(&expr)
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
            ) -> Self::Props {
                provided_props.clone()
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

    /// Matches the given expression to the rule and expects it not to match. If the match is unsuccessful this method
    /// then calls [Rule::apply](crate::rules::Rule::apply) on the given expression.
    /// If that call does not return `Ok(None)` this methods fails.
    pub fn no_match(&mut self, expr: &LogicalExpr) {
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
            let mut rules: Vec<(&RuleId, &Box<dyn Rule>)> = self.rule_set.get_rules().collect();
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
    ) -> Result<(bool, bool), OptimizerError> {
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
            fmt.set_write_name(false);
            let expr = expr.as_ref();
            T::format_expr(expr.expr(), expr.props(), &mut fmt);
            fmt.set_write_name(true);
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

#[cfg(test)]
mod test {
    use crate::memo::ScalarNode;
    use crate::operators::relational::logical::{LogicalAggregate, LogicalExpr, LogicalGet, LogicalProjection};
    use crate::operators::relational::{RelExpr, RelNode};
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
            columns: vec![3, 4],
        });

        expect_formatted(
            &aggr,
            r#"
LogicalAggregate
  aggr_exprs: [10]
  group_exprs: [col:1]
  input: LogicalGet A cols=[1, 2]
"#,
        )
    }

    fn expect_formatted(expr: &LogicalExpr, expected: &str) {
        let expr = Operator::from(OperatorExpr::from(expr.clone()));
        let mut str = format_operator_tree(&expr);
        let str = format!("\n{}\n", str);
        assert_eq!(str, expected, "expected format");
    }
}
