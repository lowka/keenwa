//! Rules used by the optimizer.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

use crate::error::OptimizerError;
use crate::meta::MetadataRef;
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::relational::RelExpr;
use crate::properties::physical::RequiredProperties;
use crate::rules::enforcers::BuiltinPhysicalPropertiesProvider;

mod enforcers;
pub mod implementation;
pub mod pattern;
mod rewrite;
#[cfg(test)]
pub mod testing;
pub mod transformation;

pub use enforcers::SupportedPartitioningSchemes;
pub use enforcers::{run_enforce_properties, run_explore_alternatives};

/// An optimization rule used by the optimizer. An optimization rule either transform one logical expression into another or
/// provides an implementation of the given logical expression.
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
        rule_debug_format(self, f)
    }
}

/// Rule type specifies which expressions a rule produces.
/// A transformation rule produce [logical expressions].
/// An implementation rule produce [physical expressions].
///
/// [logical expressions]: crate::operators::relational::logical::LogicalExpr
/// [physical expressions]: crate::operators::relational::physical::PhysicalExpr
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
    required_properties: Rc<Option<RequiredProperties>>,
    metadata: MetadataRef<'m>,
}

impl<'m> RuleContext<'m> {
    pub fn new(required_properties: Rc<Option<RequiredProperties>>, metadata: MetadataRef<'m>) -> Self {
        RuleContext {
            required_properties,
            metadata,
        }
    }

    pub fn required_properties(&self) -> Option<&RequiredProperties> {
        self.required_properties.as_ref().as_ref()
    }

    pub fn metadata(&self) -> &MetadataRef {
        &self.metadata
    }
}

/// An opaque identifier of an optimization rule.
pub type RuleId = usize;

/// Provides access to optimization rules used by the optimizer.
pub trait RuleSet {
    /// The type of [PhysicalPropertiesProvider].
    type PhysicalPropsProvider: PhysicalPropertiesProvider;

    /// Returns an iterator over available optimization rules.
    fn get_rules(&self) -> RuleIterator;

    /// Applies a rule with the given identifier to the expression `expr`.
    fn apply_rule(
        &self,
        rule_id: &RuleId,
        ctx: &RuleContext,
        expr: &LogicalExpr,
    ) -> Result<Option<RuleResult>, OptimizerError>;

    /// Returns a reference to [PhysicalPropertiesProvider].
    fn get_physical_properties_provider(&self) -> &Self::PhysicalPropsProvider;
}

/// PhysicalPropertiesProvider is a component of the optimizer that provides hints how
/// to optimize expressions when physical properties are present.
//FIXME: Remove in favour of traits for PhysicalExprs?
pub trait PhysicalPropertiesProvider {
    /// Returns a [strategy](DerivePropertyMode) that should be used in order to derive the required physical properties
    /// for the given [physical expression](PhysicalExpr).
    ///
    /// Method must return an error if required properties are empty.
    fn derive_properties(
        &self,
        expr: &PhysicalExpr,
        required_properties: &RequiredProperties,
    ) -> Result<DerivePropertyMode, OptimizerError>;

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
    fn can_explore_with_enforcer(&self, expr: &RelExpr, required_properties: &RequiredProperties) -> bool;

    fn support_implementation(&self, required_properties: &RequiredProperties) -> bool;
}

/// Describes how physical properties should be derived from a physical expression.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum DerivePropertyMode {
    /// This strategy should be used when a physical expression
    /// expression retains the required physical property.
    PropertyIsRetained,
    /// This strategy should be used when a physical expression
    /// provides the required physical property.
    PropertyIsProvided,
    /// This strategy should be used when a physical expression can neither retain nor provide
    /// the required physical property. In such case an enforcer operator must be used
    /// in order to provide the required physical property.
    ApplyEnforcer,
}

/// An iterator over available optimization rules.
pub struct RuleIterator<'r> {
    rules: std::vec::IntoIter<(&'r RuleId, &'r dyn Rule)>,
}

impl<'r> RuleIterator<'r> {
    /// Create an iterator over the given rules.
    pub fn new(rules: Vec<(&'r RuleId, &'r dyn Rule)>) -> Self {
        RuleIterator {
            rules: rules.into_iter(),
        }
    }
}

impl<'r> Iterator for RuleIterator<'r> {
    type Item = (&'r RuleId, &'r dyn Rule);

    fn next(&mut self) -> Option<Self::Item> {
        self.rules.next()
    }
}

impl<'a> Debug for RuleIterator<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        struct DebugRule<'a> {
            rule: &'a dyn Rule,
        }
        impl Debug for DebugRule<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                rule_debug_format(self.rule, f)
            }
        }
        f.debug_list()
            .entries(self.rules.as_slice().iter().map(|(id, rule)| {
                let rule = DebugRule { rule: *rule };
                (id, rule)
            }))
            .finish()
    }
}

/// Result of a call to [PhysicalPropertiesProvider::evaluate_properties].
#[derive(Debug)]
//TODO: Find a better name. Replace with enum.
pub struct EvaluationResponse {
    /// If `true` a physical expression provides physical properties.
    pub provides_property: bool,
    /// If `true` a physical expression retains physical properties.
    pub retains_property: bool,
}

fn rule_debug_format(rule: &dyn Rule, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Rule")
        .field("name", &rule.name())
        .field("type", &rule.rule_type())
        .field("group_rule", &rule.group_rule())
        .finish()
}

/// An implementation of [RuleSet] trait that uses a predefined set optimization rules.
#[derive(Debug)]
pub struct StaticRuleSet {
    rules: HashMap<RuleId, Box<dyn Rule>>,
    properties_provider: BuiltinPhysicalPropertiesProvider,
}

impl RuleSet for StaticRuleSet {
    type PhysicalPropsProvider = BuiltinPhysicalPropertiesProvider;

    fn get_rules(&self) -> RuleIterator {
        let rules: Vec<(&RuleId, &dyn Rule)> = self.rules.iter().map(|(id, rule)| (id, rule.as_ref())).collect();
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
            None => Err(OptimizerError::internal(format!("Rule#{} does not exist", rule_id))),
        }
    }

    fn get_physical_properties_provider(&self) -> &Self::PhysicalPropsProvider {
        &self.properties_provider
    }
}

/// Builder for a [StaticRuleSet].
pub struct StaticRuleSetBuilder {
    rules: HashMap<RuleId, Box<dyn Rule>>,
    explore_with_enforcer: bool,
    supported_partitioning: SupportedPartitioningSchemes,
}

impl StaticRuleSetBuilder {
    /// Creates an instance of a [StaticRuleSetBuilder].
    pub fn new() -> Self {
        StaticRuleSetBuilder {
            rules: Default::default(),
            explore_with_enforcer: true,
            supported_partitioning: SupportedPartitioningSchemes::all(),
        }
    }

    /// Adds the given rules.
    pub fn add_rules<T>(mut self, rules: T) -> Self
    where
        T: IntoIterator<Item = Box<dyn Rule>>,
    {
        let iter = rules.into_iter();
        iter.enumerate().for_each(|(i, rule)| {
            self.rules.insert(i, rule);
        });
        self
    }

    /// See [PhysicalPropertiesProvider::can_explore_with_enforcer].
    pub fn explore_with_enforcer(mut self, value: bool) -> Self {
        self.explore_with_enforcer = value;
        self
    }

    pub fn supported_partitioning(mut self, schemes: SupportedPartitioningSchemes) -> Self {
        self.supported_partitioning = schemes;
        self
    }

    /// Builds an instance of a [StaticRuleSet].
    pub fn build(self) -> StaticRuleSet {
        let properties_provider = BuiltinPhysicalPropertiesProvider::explore_alternatives(self.explore_with_enforcer);
        StaticRuleSet {
            rules: self.rules,
            properties_provider: properties_provider.with_supported_partitioning_schemes(self.supported_partitioning),
        }
    }
}
