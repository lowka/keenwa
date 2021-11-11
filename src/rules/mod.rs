use crate::error::OptimizerError;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::iter::FromIterator;

use crate::meta::Metadata;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
use crate::operators::InputExpr;
use crate::properties::physical::PhysicalProperties;
use crate::rules::enforcers::{DefaultEnforcers, EnforcerRules};

mod enforcers;
pub mod implementation;
#[cfg(test)]
pub mod testing;
pub mod transformation;

/// An optimization rule used by the optimizer. An optimization rule either transform one logical expression into another or
/// provides an implementation of the given logical expression.
/// TODO: Require Debug trait.
pub trait Rule {
    /// The name of this rule.
    fn name(&self) -> String;

    /// Returns type type of this rule.
    fn rule_type(&self) -> RuleType;

    /// Whether this rule applies to the entire group or not.
    /// If `true` this rule is applied only to the first expression in a group it matches.
    fn group_rule(&self) -> bool {
        false
    }

    /// Checks whether this rule can be applied to the given expression `expr`.
    //TODO: Hide logical expression inside the RuleContext and provide access to via Operation::as_logical
    fn matches(&self, ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch>;

    /// Tries to apply this rule to the given expression `expr`.
    /// If this rule can not be applied to the given expression method must return `Ok(None)`.
    fn apply(&self, ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError>;
}

impl Debug for dyn Rule {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ name={}, type={:?}, group_rule={:?} }}", self.name(), self.rule_type(), self.group_rule())
    }
}

/// Rule type specifies which expressions a rule produces.
/// A transformation rule produce [logical expressions].
/// An implementation rule produce [physical expressions].
///
/// [logical expressions]: crate::operators::logical::LogicalExpr
/// [physical expressions]: crate::operators::physical::PhysicalExpr
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RuleType {
    /// Transformation rules produce equivalent logical expressions.
    Transformation,
    /// Implementation rules produce physical expressions.
    /// Physical expressions are used to compute cost of a query plan.
    Implementation,
}

/// Specifies how a rule matches an expression.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RuleMatch {
    /// A rule matches the expression.
    /// Meaning that the rule will be applied to all expressions it matched.
    /// If another expression in the same memo group matches this rule then the rule will also be applied to that expression.
    Expr,
    /// A rule matches the entire memo group.
    /// Meaning that the rule will be applied only to the first expression it matched.
    /// If another expression in the same memo group matches this rule the rule will not be applied to that expression.
    Group,
}

/// A result of a rule application.
#[derive(Debug)]
pub enum RuleResult {
    /// A alternative logical expression.
    Substitute(LogicalExpr),
    /// An implementation.
    Implementation(PhysicalExpr),
}

#[derive(Debug)]
pub struct RuleContext<'m> {
    required_properties: PhysicalProperties,
    metadata: &'m Metadata,
}

impl<'m> RuleContext<'m> {
    pub fn new(required_properties: PhysicalProperties, metadata: &'m Metadata) -> Self {
        RuleContext {
            required_properties,
            metadata,
        }
    }

    pub fn required_properties(&self) -> &PhysicalProperties {
        &self.required_properties
    }

    pub fn metadata(&self) -> &Metadata {
        self.metadata
    }
}

/// An opaque identifier of an optimization rule.
pub type RuleId = usize;

/// Provides access to optimization rules used by the optimizer.
pub trait RuleSet: Debug {
    /// Returns an iterator over available optimization rules.
    fn get_rules(&self) -> RuleIterator;

    /// Applies a rule with the given identifier to the expression `expr`.
    fn apply_rule(
        &self,
        rule_id: &RuleId,
        ctx: &RuleContext,
        expr: &LogicalExpr,
    ) -> Result<Option<RuleResult>, OptimizerError>;

    /// Checks whether the given physical expression satisfies the required physical properties.
    //FIXME: return a struct - rename method.
    fn evaluate_properties(
        &self,
        expr: &PhysicalExpr,
        required_properties: &PhysicalProperties,
    ) -> Result<(bool, bool), OptimizerError>;

    /// Creates an enforcer operator for the specified physical properties.
    /// If enforcer can not be created this method must return an error.
    fn create_enforcer(
        &self,
        required_properties: &PhysicalProperties,
        input: InputExpr,
    ) -> Result<(PhysicalExpr, PhysicalProperties), OptimizerError>;

    /// Provides a hint to the optimizer whether or not to explore an alternative plan for the given expression
    /// that can use an enforcer as a parent expression. For example:
    /// ```text
    ///   Select filter: x > 10
    ///     [ordering: y desc] # required physical properties
    ///     > Scan 'C'
    /// ```
    ///  There are two ways to achieve the ordering requirements for that query if 'C' is not ordered by 'x':
    ///  1) Retrieve tuples from 'C', sort them, and then apply the filter.
    /// ``` text
    ///   Select filter: x > 10
    ///     > Sort ordering: y desc
    ///        > Scan 'C'
    ///  ```
    ///  2) Retrieve tuples from 'C', filter them, and only after that apply the required ordering.
    /// ``` text
    ///   Sort ordering: y desc
    ///     > Select filter: x > 10
    ///        > Scan 'C'
    /// ```
    /// Plan #2 is more beneficial because in this case a sort operation will have less tuples to sort.
    ///
    fn can_explore_with_enforcer(&self, expr: &LogicalExpr, required_properties: &PhysicalProperties) -> bool;
}

/// An iterator over available optimization rules.
pub struct RuleIterator<'r> {
    rules: std::vec::IntoIter<(&'r RuleId, &'r Box<dyn Rule>)>,
}

impl<'r> RuleIterator<'r> {
    pub fn new(rules: Vec<(&'r RuleId, &'r Box<dyn Rule>)>) -> Self {
        RuleIterator {
            rules: rules.into_iter(),
        }
    }
}

impl<'r> Iterator for RuleIterator<'r> {
    type Item = (&'r RuleId, &'r Box<dyn Rule>);

    fn next(&mut self) -> Option<Self::Item> {
        self.rules.next()
    }
}

impl Debug for RuleIterator<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RuleIterator({:?})", self.rules.as_slice())
    }
}

/// An implementation of [RuleSet] trait that uses a predefined set optimization rules.
///
/// [RuleSet]: crate::rules::RuleSet
#[derive(Debug)]
pub struct StaticRuleSet {
    rules: HashMap<RuleId, Box<dyn Rule>>,
    enforcers: DefaultEnforcers,
}

impl StaticRuleSet {
    /// Creates a new [RuleSet] from the given collection of rules.
    pub fn new(rules: Vec<Box<dyn Rule>>) -> Self {
        let rules = HashMap::from_iter(rules.into_iter().enumerate());
        StaticRuleSet {
            rules,
            enforcers: DefaultEnforcers,
        }
    }
}

impl RuleSet for StaticRuleSet {
    fn get_rules(&self) -> RuleIterator {
        let rules: Vec<(&RuleId, &Box<dyn Rule>)> = self.rules.iter().collect();
        RuleIterator::new(rules)
    }

    fn apply_rule(
        &self,
        rule_id: &RuleId,
        ctx: &RuleContext,
        expr: &LogicalExpr,
    ) -> Result<Option<RuleResult>, OptimizerError> {
        match self.rules.get(rule_id) {
            Some(rule) => rule.apply(ctx, expr),
            None => Err(OptimizerError::Internal(format!("Rule#{} does not exist", rule_id))),
        }
    }

    fn evaluate_properties(
        &self,
        expr: &PhysicalExpr,
        required_properties: &PhysicalProperties,
    ) -> Result<(bool, bool), OptimizerError> {
        if required_properties.is_empty() {
            // provides_property, retains_property
            Ok((true, false))
        } else {
            self.enforcers.evaluate_properties(expr, required_properties)
        }
    }

    fn create_enforcer(
        &self,
        required_properties: &PhysicalProperties,
        input: InputExpr,
    ) -> Result<(PhysicalExpr, PhysicalProperties), OptimizerError> {
        assert!(
            !required_properties.is_empty(),
            "Unable to create an enforcer - no required properties: {:?}",
            required_properties
        );

        self.enforcers.create_enforcer(required_properties, input)
    }

    fn can_explore_with_enforcer(&self, expr: &LogicalExpr, required_properties: &PhysicalProperties) -> bool {
        self.enforcers.can_explore_with_enforcer(expr, required_properties)
    }
}
