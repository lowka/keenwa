use crate::error::OptimizerError;
use crate::memo::{InputNode, MemoExpr, MemoExprCallback, MemoExprFormatter, StringMemoFormatter};
use crate::meta::Metadata;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
use crate::operators::{ExprMemo, InputExpr, Operator, OperatorExpr, Properties};
use crate::properties::logical::{LogicalPropertiesBuilder, PropertiesProvider};
use crate::properties::physical::PhysicalProperties;
use crate::properties::statistics::{Statistics, StatisticsBuilder};
use crate::rules::{Rule, RuleContext, RuleId, RuleIterator, RuleResult, RuleSet};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fmt::Display;
use std::rc::Rc;

/// Provides methods to test [optimization rules].
///
/// [optimization rules]: crate::rules::Rule
pub struct RuleTester {
    rule: Box<dyn Rule>,
    memo: ExprMemo,
    metadata: Metadata,
}

impl RuleTester {
    /// Creates a tester for the given rule.
    pub fn new<T>(rule: T) -> Self
    where
        T: Rule + 'static,
    {
        let props_builder = LogicalPropertiesBuilder::new(Box::new(NoStatsBuilder));
        RuleTester {
            rule: Box::new(rule),
            memo: ExprMemo::with_callback(Rc::new(props_builder)),
            metadata: Metadata::new(HashMap::new()),
        }
    }

    pub fn set_metadata(&mut self, metadata: Metadata) {
        self.metadata = metadata;
    }

    /// Attempts to apply the rule to the given expression and then compares the result with the expected value.
    pub fn apply(&mut self, expr: &LogicalExpr, expected: &str) {
        let (_, expr_ref) = self.memo.insert(Operator::from(OperatorExpr::from(expr.clone())));

        let ctx = RuleContext::new(PhysicalProperties::none(), &self.metadata);
        let rule_match = self.rule.matches(&ctx, expr);
        assert!(rule_match.is_some(), "Rule does not match: {:?}", self.rule);

        let result = self.rule.apply(&ctx, expr_ref.mexpr().expr().as_logical());
        match result {
            Ok(RuleResult::Substitute(new_expr)) => {
                let (_, new_expr) = self.memo.insert(Operator::from(OperatorExpr::from(new_expr)));
                let actual_expr = format_expr(new_expr.mexpr());

                assert_eq!(actual_expr.trim_end(), expected.trim());
            }
            Ok(_) => panic!("Unexpected result: {:?}", result),
            Err(e) => panic!("Failed to apply a rule. Rule: {:?}. Error: {}", self.rule, e),
        };
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
    ) -> Result<RuleResult, OptimizerError> {
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
        input: InputExpr,
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

/// Builds the following textual representation of the given expression:
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
fn format_expr(expr: &Operator) -> String {
    let mut buf = String::new();
    let fmt = StringMemoFormatter::new(&mut buf);
    let mut fmt = FormatHeader { fmt };
    expr.format_expr(&mut fmt);

    let mut fmt = FormatInputs {
        buf: &mut buf,
        depth: 0,
    };
    expr.format_expr(&mut fmt);

    buf
}

struct FormatHeader<'b> {
    fmt: StringMemoFormatter<'b>,
}

impl MemoExprFormatter for FormatHeader<'_> {
    fn write_name(&mut self, name: &str) {
        self.fmt.write_name(name);
    }

    fn write_source(&mut self, source: &str) {
        self.fmt.write_source(source);
    }

    fn write_input<E>(&mut self, _name: &str, _input: &InputNode<E>)
    where
        E: MemoExpr,
    {
        //inputs are written by another formatter
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

struct FormatInputs<'b> {
    buf: &'b mut String,
    depth: usize,
}

impl FormatInputs<'_> {
    fn pad_depth(&mut self, c: char) {
        for i in 1..=self.depth {
            for _ in 1..=i * 2 {
                self.buf.push(c);
            }
        }
    }
}

impl MemoExprFormatter for FormatInputs<'_> {
    fn write_name(&mut self, _name: &str) {
        // name is written by another formatter
    }

    fn write_source(&mut self, _source: &str) {
        // source is written by another formatter
    }

    fn write_input<E>(&mut self, name: &str, input: &InputNode<E>)
    where
        E: MemoExpr,
    {
        self.depth += 1;
        self.buf.push('\n');
        self.pad_depth(' ');
        self.buf.push_str(name);
        self.buf.push_str(": ");
        match input {
            InputNode::Expr(expr) => {
                let fmt = StringMemoFormatter::new(self.buf);
                let mut header = FormatHeader { fmt };
                expr.format_expr(&mut header);
                expr.format_expr(self);
            }
            InputNode::Group(group) => {
                let expr = group.mexpr().mexpr();
                let fmt = StringMemoFormatter::new(self.buf);
                let mut header = FormatHeader { fmt };
                expr.format_expr(&mut header);
                expr.format_expr(self);
            }
        }
        self.depth -= 1;
    }

    fn write_value<D>(&mut self, _name: &str, _value: D)
    where
        D: Display,
    {
        // values are written by another formatter
    }

    fn write_values<D>(&mut self, _name: &str, _values: &[D])
    where
        D: Display,
    {
        // values are written by another formatter
    }
}

#[derive(Debug)]
struct NoStatsBuilder;

impl StatisticsBuilder for NoStatsBuilder {
    fn build_statistics(
        &self,
        _expr: &LogicalExpr,
        _statistics: Option<&Statistics>,
    ) -> Result<Option<Statistics>, OptimizerError> {
        Ok(None)
    }
}

impl MemoExprCallback for LogicalPropertiesBuilder {
    type Expr = Operator;
    type Attrs = Properties;

    fn new_expr(&self, expr: &Self::Expr, attrs: Self::Attrs) -> Self::Attrs {
        let logical = self
            .build_properties(expr.expr(), attrs.logical().statistics())
            .expect("Failed to build logical properties");
        Properties::new(logical, attrs.required)
    }
}
