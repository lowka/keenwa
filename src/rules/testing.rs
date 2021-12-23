use crate::error::OptimizerError;
use crate::memo::{MemoExpr, MemoExprFormatter, MemoGroupCallback, StringMemoFormatter};
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

        RuleTester {
            rule: Box::new(rule),
            memo: ExprMemo::with_callback(Rc::new(MutableMetadata::new()), Rc::new(Callback)),
            metadata: MutableMetadata::new(),
        }
    }

    /// Attempts to apply the rule to the given expression and then compares the result with the expected value.
    pub fn apply(&mut self, expr: &LogicalExpr, expected: &str) {
        let (_, expr_ref) = self.memo.insert_group(Operator::from(OperatorExpr::from(expr.clone())));

        let ctx = RuleContext::new(Rc::new(PhysicalProperties::none()), self.metadata.get_ref());
        let rule_match = self.rule.matches(&ctx, expr);
        assert!(rule_match.is_some(), "Rule does not match: {:?}", self.rule);

        let expr = expr_ref.mexpr().expr();
        let result = self.rule.apply(&ctx, expr.as_relational().as_logical());
        let expr = match result {
            Ok(Some(RuleResult::Substitute(expr))) => Operator::from(OperatorExpr::from(expr)),
            Ok(Some(RuleResult::Implementation(expr))) => Operator::from(OperatorExpr::from(expr)),
            Ok(None) => panic!("Rule matched but not applied: {:?}", self.rule),
            Err(e) => panic!("Failed to apply a rule. Rule: {:?}. Error: {}", self.rule, e),
        };
        let (_, new_expr) = self.memo.insert_group(expr);
        let actual_expr = format_operator_tree(new_expr.mexpr());

        assert_eq!(actual_expr.trim_end(), expected.trim());
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
    let mut buf = String::new();
    let fmt = StringMemoFormatter::new(&mut buf);
    let mut fmt = FormatHeader { fmt };
    Operator::format_expr(expr.expr(), expr.props(), &mut fmt);

    let mut fmt = FormatExprs {
        buf: &mut buf,
        depth: 0,
    };
    Operator::format_expr(expr.expr(), expr.props(), &mut fmt);

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

    fn write_expr<T>(&mut self, _name: &str, _input: impl AsRef<T>)
    where
        T: MemoExpr,
    {
        //exprs are written by another formatter
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
        for i in 1..=self.depth {
            for _ in 1..=i * 2 {
                self.buf.push(c);
            }
        }
    }
}

impl MemoExprFormatter for FormatExprs<'_> {
    fn write_name(&mut self, _name: &str) {
        // name is written by another formatter
    }

    fn write_source(&mut self, _source: &str) {
        // source is written by another formatter
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
        let mut header = FormatHeader { fmt };

        T::format_expr(expr.expr(), expr.props(), &mut header);
        T::format_expr(expr.expr(), expr.props(), self);

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
