mod ordering;
mod partitioning;

use crate::error::OptimizerError;
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::relational::{RelExpr, RelNode};
use crate::properties::physical::RequiredProperties;
use crate::rules::{DerivePropertyMode, PhysicalPropertiesProvider};
use std::collections::VecDeque;
use std::fmt::{Display, Formatter};

/// Default implementation of [PhysicalPropertiesProvider].
/// Provides an enforcer of a sorted physical property.
#[derive(Debug)]
pub struct BuiltinPhysicalPropertiesProvider {
    // TODO: Move this flag to the optimizer?
    explore_with_enforcer: bool,
}

impl BuiltinPhysicalPropertiesProvider {
    /// Creates an instance of [BuiltinPhysicalPropertiesProvider].
    pub fn new() -> Self {
        BuiltinPhysicalPropertiesProvider {
            explore_with_enforcer: true,
        }
    }

    /// See [PhysicalPropertiesProvider::can_explore_with_enforcer].
    pub(crate) fn explore_alternatives(value: bool) -> Self {
        BuiltinPhysicalPropertiesProvider {
            explore_with_enforcer: value,
        }
    }
}

impl PhysicalPropertiesProvider for BuiltinPhysicalPropertiesProvider {
    fn derive_properties(
        &self,
        expr: &PhysicalExpr,
        required_properties: &RequiredProperties,
    ) -> Result<DerivePropertyMode, OptimizerError> {
        if required_properties.has_several() {
            // Both ordering and partitioning are present
            // Properties must be optimized with enforcer.
            Ok(DerivePropertyMode::ApplyEnforcer)
        } else if let Some(ordering) = required_properties.ordering() {
            Ok(ordering.derive_from_expr(expr))
        } else if let Some(partitioning) = required_properties.partitioning() {
            Ok(partitioning.derive_from_expr(expr))
        } else {
            Err(OptimizerError::internal("No required properties"))
        }
    }

    fn get_enforcer_order(
        &self,
        required_properties: &RequiredProperties,
    ) -> Result<RequiredPropertiesQueue, OptimizerError> {
        let ordering = required_properties.ordering();
        let partitioning = required_properties.partitioning();
        match (ordering, partitioning) {
            (Some(ord), Some(p)) => {
                let ord = RequiredProperties::new_with_ordering(ord.clone());
                let p = RequiredProperties::new_with_partitioning(p.clone());
                // When we have both ordering and partitioning (requirements are added
                // in a reverse order because we remove requirements from a tail of a queue):
                //
                // First optimize by partitioning:
                // - next: required properties are None
                // - enforced: contains only partitioning
                //
                // Then optimize by ordering. At this step parent ctx must contain both
                // ordering and partitioning.
                // - next: partitioning
                // - enforced: ordering and partitioning.
                //
                let mut requirements = RequiredPropertiesQueue::new();
                requirements.push(ord, Some(p.clone()), required_properties.clone());
                requirements.push(p.clone(), None, p);
                Ok(requirements)
            }
            (Some(ord), None) => {
                let ord = RequiredProperties::new_with_ordering(ord.clone());
                let mut requirements = RequiredPropertiesQueue::new();
                requirements.push(ord.clone(), None, ord);
                Ok(requirements)
            }
            (None, Some(p)) => {
                let p = RequiredProperties::new_with_partitioning(p.clone());
                let mut requirements = RequiredPropertiesQueue::new();
                requirements.push(p.clone(), None, p);
                Ok(requirements)
            }
            (None, None) => Err(OptimizerError::internal("No required properties")),
        }
    }

    fn create_enforcer(
        &self,
        required_properties: &RequiredProperties,
        input: RelNode,
    ) -> Result<(PhysicalExpr, Option<RequiredProperties>), OptimizerError> {
        if required_properties.has_several() {
            // Should be kept in sync with `evaluate_properties`/`can_explore_with_enforcer`
            // because we do not enforce multiple properties at once.
            Err(OptimizerError::internal(format!("Can not create enforcer: {}", required_properties)))
        } else if let Some(ordering) = required_properties.ordering() {
            let ordering_enforcer = ordering.create_enforcer_expr(input);
            Ok((ordering_enforcer, None))
        } else if let Some(partitioning) = required_properties.partitioning() {
            let partitioning_enforcer = partitioning.create_enforcer_expr(input);
            Ok((partitioning_enforcer, None))
        } else {
            let message = format!("Unexpected required properties: {:?}", required_properties);
            Err(OptimizerError::internal(message))
        }
    }

    fn can_explore_with_enforcer(&self, expr: &RelExpr, required_properties: &RequiredProperties) -> bool {
        match expr {
            RelExpr::Logical(expr) if matches!(**expr, LogicalExpr::Select { .. }) => {
                // Do not explore with an enforcer when there are multiple requirements.
                // see `evaluate_properties`/`create_enforcer`
                // or when `explore_with_enforcer` flag is not set.
                if !self.explore_with_enforcer || required_properties.has_several() {
                    false
                } else {
                    required_properties.has_any()
                }
            }
            RelExpr::Logical(_) => false,
            // expr is a physical expression when we optimize an enforcer expression group.
            RelExpr::Physical(_) => false,
        }
    }
}

/// Trait that should be implemented by a physical property that can be enforced by an operator.
trait PhysicalProperty: Sized {
    /// Creates an instance of a [physical expression](PhysicalExpr) that enforces this property.
    fn create_enforcer_expr(&self, input: RelNode) -> PhysicalExpr;

    /// Returns a strategy that should be used to derive this property from
    /// the given [physical expression](PhysicalExpr).
    fn derive_from_expr(&self, expr: &PhysicalExpr) -> DerivePropertyMode;
}

/// Contains an order in which required properties should be applied.
#[derive(Debug)]
pub struct RequiredPropertiesQueue {
    inner: VecDeque<(RequiredProperties, Option<RequiredProperties>, RequiredProperties)>,
}

impl RequiredPropertiesQueue {
    pub(crate) fn new() -> Self {
        RequiredPropertiesQueue { inner: VecDeque::new() }
    }

    /// Adds requirements.
    pub(crate) fn push(
        &mut self,
        required: RequiredProperties,
        next: Option<RequiredProperties>,
        enforced: RequiredProperties,
    ) {
        self.inner.push_back((required, next, enforced))
    }

    /// Retrieves next the required properties to be applied.
    pub fn pop(&mut self) -> (RequiredProperties, Option<RequiredProperties>, RequiredProperties) {
        self.inner.pop_back().expect("No required properties")
    }

    /// Returns `true` this queue is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Display for RequiredPropertiesQueue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, (r, _, p)) in self.inner.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}/{}", r, p)?;
        }
        write!(f, "]")
    }
}
