use std::rc::Rc;

use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

use crate::error::OptimizerError;
use crate::memo::{MemoBuilder, MemoGroupCallback};
use crate::meta::{ColumnId, MutableMetadata};
use crate::operators::format::format_operator_tree;
use crate::operators::relational::logical::{LogicalExpr, LogicalGet};
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::relational::{RelExpr, RelNode};
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
