mod ordering;
mod partitioning;

use crate::error::OptimizerError;
use crate::memo::GroupId;
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::exchanger::{Exchanger, ExchangerType};
use crate::operators::relational::physical::{PhysicalExpr, Sort};
use crate::operators::relational::RelExpr;
use crate::operators::{ExprRef, OuterScope};
use crate::optimizer::OptimizerContext;
use crate::properties::partitioning::Partitioning;
use crate::properties::physical::RequiredProperties;
use crate::properties::OrderingChoice;
use crate::rules::{DerivePropertyMode, PhysicalPropertiesProvider};

#[derive(Debug)]
pub struct SupportedPartitioningSchemes {
    schemes: Option<Vec<std::mem::Discriminant<Partitioning>>>,
}

impl SupportedPartitioningSchemes {
    pub fn all() -> Self {
        SupportedPartitioningSchemes { schemes: None }
    }

    pub fn empty() -> Self {
        SupportedPartitioningSchemes { schemes: Some(vec![]) }
    }

    pub fn add(&mut self, p: &Partitioning) {
        let d = std::mem::discriminant(p);
        match self.schemes.as_mut() {
            Some(schemes) => schemes.push(d),
            None => self.schemes = Some(vec![d]),
        }
    }

    pub fn supports(&self, p: &Partitioning) -> bool {
        self.schemes
            .as_ref()
            .map(|s| {
                let d = std::mem::discriminant(p);
                s.contains(&d)
            })
            .unwrap_or(true)
    }
}

/// Default implementation of [PhysicalPropertiesProvider].
/// Provides an enforcer of a sorted physical property.
#[derive(Debug)]
pub struct BuiltinPhysicalPropertiesProvider {
    // TODO: Move this flag to the optimizer?
    explore_with_enforcer: bool,
    supported_partitioning_schemes: SupportedPartitioningSchemes,
}

impl BuiltinPhysicalPropertiesProvider {
    /// Creates an instance of [BuiltinPhysicalPropertiesProvider].
    pub fn new() -> Self {
        BuiltinPhysicalPropertiesProvider {
            explore_with_enforcer: true,
            supported_partitioning_schemes: SupportedPartitioningSchemes::all(),
        }
    }

    /// See [PhysicalPropertiesProvider::can_explore_with_enforcer].
    pub(crate) fn explore_alternatives(value: bool) -> Self {
        BuiltinPhysicalPropertiesProvider {
            explore_with_enforcer: value,
            supported_partitioning_schemes: SupportedPartitioningSchemes::all(),
        }
    }

    pub fn with_supported_partitioning_schemes(mut self, value: SupportedPartitioningSchemes) -> Self {
        self.supported_partitioning_schemes = value;
        self
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

    fn can_explore_with_enforcer(&self, expr: &RelExpr, required_properties: &RequiredProperties) -> bool {
        match expr {
            RelExpr::Logical(expr) if matches!(**expr, LogicalExpr::Select { .. }) => {
                // ENFORCERS: Do not explore with an enforcer when there are multiple requirements
                // when `explore_with_enforcer` flag is not set.
                // (see `run_explore_alternatives')
                if !self.explore_with_enforcer || required_properties.has_several() {
                    false
                } else {
                    // Only ordering is supported in exploration of alternative plans.
                    required_properties.ordering().is_some()
                }
            }
            RelExpr::Logical(_) => false,
            // expr is a physical expression when we optimize an enforcer expression group.
            RelExpr::Physical(_) => false,
        }
    }

    fn support_implementation(&self, required_properties: &RequiredProperties) -> bool {
        if let Some(p) = required_properties.partitioning() {
            self.supported_partitioning_schemes.supports(p)
        } else {
            true
        }
    }
}

/// Trait that should be implemented by a physical property that can be enforced by an operator.
trait PhysicalProperty: Sized {
    /// Returns a strategy that should be used to derive this property from
    /// the given [physical expression](PhysicalExpr).
    fn derive_from_expr(&self, expr: &PhysicalExpr) -> DerivePropertyMode;
}

pub fn run_enforce_properties<T>(
    optimizer: &mut T,
    group_id: GroupId,
    required_properties: RequiredProperties,
    input_expr: ExprRef,
) -> Result<(), OptimizerError>
where
    T: OptimizerContext,
{
    /*
    Partitioning:
    Check input reqs if RetainsProperty branch of get_optimize_rel_task
    if partitioning != None -> return EnforcerTask or require_properties is inputs req are the same

    Check required ctx partitioning vs partitioning required by expression:


    1. Build inferred partitioning bottom-up - known partitioning (kP)/ LogicalProperties  inferred_partitioning/interesting_partitioning
    2. During optimization check ctx.req.partitioning against kP in order to decide
    whether to apply EnforceDataExchange or not.

    if partitioning is serial
       if expr has parallel req -> insert merge operator, optimize with parallel req,
       if expr has serial req -> do nothing, optimize under serial req

       [if expr has no partitioning requirements -> assume it requires serial output]

    else if partitioning is parallel
       if expr has serial req -> add initial partitioning operator, optimize inputs with serial reqs
       if expr has other partitioning requirements -> insert repartitioning operator for such input expr, optimize with expr.partitioning
       if expr has the same partitioning -> do nothing output is partitioned, optimize under the same partitioning scheme

       [if expr no partitioning requirements -> assume it requires parallel output under the same partitioning scheme]
     */

    let f = format!("{}: {} expr: {}", group_id, required_properties, input_expr);

    if required_properties.has_several() {
        let partitioning = required_properties.partitioning().unwrap().clone();
        let ordering = required_properties.ordering();
        let (exchanger_type, partitioning_schemes) =
            generate_ordered_input_partitioning_schemes(&partitioning, ordering);
        let ordering = required_properties.ordering().clone().expect("Expected ordering");

        for (input_partitioning, should_be_ordered) in partitioning_schemes {
            if should_be_ordered {
                let part_enforcer_expr =
                    add_partitioning_enforcer(optimizer, exchanger_type, partitioning.clone(), group_id)?;
                let partitioning_remaining_properties =
                    Some(RequiredProperties::new_with_partitioning(input_partitioning));

                let ordering_enforcer_expr =
                    add_ordering_enforcer(optimizer, ordering.clone(), part_enforcer_expr.group_id())?;
                let ordering_remaining_properties = required_properties.clone().without_ordering().into_option();

                let ordering_fmt = format!("{}", ordering_enforcer_expr);
                let partitioning_fmt = format!("{}", part_enforcer_expr);

                // Enforce ordering (uses partitioning enforcer as an input)
                optimizer.optimize_expr(
                    group_id,
                    ordering_enforcer_expr.clone(),
                    part_enforcer_expr.clone(),
                    required_properties.clone(),
                    ordering_remaining_properties,
                )?;

                // Enforce partitioning (uses original expression as an input)
                optimizer.optimize_expr(
                    ordering_enforcer_expr.group_id(),
                    part_enforcer_expr,
                    input_expr.clone(),
                    RequiredProperties::new_with_partitioning(partitioning.clone()),
                    partitioning_remaining_properties,
                )?;
            } else {
                let enforcer_expr =
                    add_partitioning_enforcer(optimizer, exchanger_type, partitioning.clone(), group_id)?;
                let remaining_properties = Some(required_properties.clone().with_partitioning(input_partitioning));

                let enforcer_fmt = format!("{}", enforcer_expr);

                optimizer.optimize_expr(
                    group_id,
                    enforcer_expr,
                    input_expr.clone(),
                    required_properties.clone(),
                    remaining_properties,
                )?;
            }
        }

        Ok(())
    } else if let Some(ordering) = required_properties.ordering() {
        let enforcer_expr = add_ordering_enforcer(optimizer, ordering.clone(), input_expr.group_id())?;
        let remaining_properties = required_properties.clone().without_ordering().into_option();

        optimizer.optimize_expr(group_id, enforcer_expr, input_expr, required_properties, remaining_properties)?;

        Ok(())
    } else if let Some(partitioning) = required_properties.partitioning() {
        // insert repartitioning operator when p != input partitioning.

        let (exchanger_type, partitioning_schemes) = generate_input_partitioning_schemes(partitioning);

        for input_partitioning in partitioning_schemes {
            let enforcer_expr = add_partitioning_enforcer(optimizer, exchanger_type, partitioning.clone(), group_id)?;
            let remaining_properties = Some(RequiredProperties::new_with_partitioning(input_partitioning));

            optimizer.optimize_expr(
                group_id,
                enforcer_expr,
                input_expr.clone(),
                RequiredProperties::new_with_partitioning(partitioning.clone()),
                remaining_properties,
            )?;
        }

        Ok(())
    } else {
        let message = format!("Unexpected required properties: {}", required_properties);
        Err(OptimizerError::internal(message))
    }
}

pub fn run_explore_alternatives<T>(
    optimizer: &mut T,
    group_id: GroupId,
    required_properties: RequiredProperties,
) -> Result<(), OptimizerError>
where
    T: OptimizerContext,
{
    // ENFORCERS: An enforcer operator is used to explore alternative plans.
    // We should optimize its inputs w/o properties provided by that enforcer.

    if let Some(ordering) = required_properties.ordering() {
        // At the moment only ordering is supported (See [BuiltinPhysicalPropertiesProvider::can_explore_with_enforcer].)
        let enforcer_expr = add_ordering_enforcer(optimizer, ordering.clone(), group_id)?;
        let remaining_properties = required_properties.clone().without_ordering().into_option();

        optimizer.optimize_enforcer(group_id, enforcer_expr, required_properties, remaining_properties)
    } else {
        Err(OptimizerError::internal(format!("Unexpected required properties: {}", required_properties)))
    }
}

fn enforce_ordering<T>(
    optimizer: &mut T,
    group_id: GroupId,
    input_expr: ExprRef,
    ordering: OrderingChoice,
    required_properties: RequiredProperties,
) -> Result<ExprRef, OptimizerError>
where
    T: OptimizerContext,
{
    let enforcer_expr = add_ordering_enforcer(optimizer, ordering.clone(), input_expr.group_id())?;
    let remaining_properties = required_properties.clone().without_ordering().into_option();

    optimizer.optimize_expr(group_id, enforcer_expr, input_expr, required_properties, remaining_properties)
}

fn enforce_partitioning<T>(
    optimizer: &mut T,
    group_id: GroupId,
    input_expr: ExprRef,
    exchanger_type: ExchangerType,
    partitioning: Partitioning,
    input_partitioning: Partitioning,
    required_properties: RequiredProperties,
) -> Result<ExprRef, OptimizerError>
where
    T: OptimizerContext,
{
    let enforcer_expr = add_partitioning_enforcer(optimizer, exchanger_type, partitioning.clone(), group_id)?;
    let remaining_properties = Some(RequiredProperties::new_with_partitioning(input_partitioning));

    optimizer.optimize_expr(
        group_id,
        enforcer_expr,
        input_expr.clone(),
        required_properties.clone(),
        remaining_properties,
    )
}

fn add_ordering_enforcer<T>(
    optimizer: &mut T,
    ordering: OrderingChoice,
    input_group_id: GroupId,
) -> Result<ExprRef, OptimizerError>
where
    T: OptimizerContext,
{
    let input = optimizer.get_rel_node(input_group_id)?;
    let scope = OuterScope::from(input.props().clone());

    let enforcer_expr = PhysicalExpr::Sort(Sort { input, ordering });
    optimizer.add_enforcer(enforcer_expr, &scope)
}

fn add_partitioning_enforcer<T>(
    optimizer: &mut T,
    exchanger_type: ExchangerType,
    partitioning: Partitioning,
    input_group_id: GroupId,
) -> Result<ExprRef, OptimizerError>
where
    T: OptimizerContext,
{
    let input = optimizer.get_rel_node(input_group_id)?;
    let scope = OuterScope::from(input.props().clone());

    let enforcer_expr = PhysicalExpr::Exchanger(Exchanger {
        input,
        exchanger_type,
        partitioning,
    });
    optimizer.add_enforcer(enforcer_expr, &scope)
}

fn generate_input_partitioning_schemes(partitioning: &Partitioning) -> (ExchangerType, Vec<Partitioning>) {
    use Partitioning::*;

    // We can only return concrete partitioning scheme because
    // the optimizer can only operate on specific partitioning schemes

    match partitioning {
        Singleton => (ExchangerType::FullMerge, vec![OrderedPartitioning(vec![]), HashPartitioning(vec![])]),
        Partitioned(_) => (ExchangerType::Init, vec![Singleton]),
        OrderedPartitioning(_) => (ExchangerType::Init, vec![Singleton]),
        HashPartitioning(_) => (ExchangerType::Init, vec![Singleton]),
    }
}

fn generate_ordered_input_partitioning_schemes(
    partitioning: &Partitioning,
    ordering: Option<&OrderingChoice>,
) -> (ExchangerType, Vec<(Partitioning, bool)>) {
    use Partitioning::*;

    // We can only return concrete partitioning scheme because
    // the optimizer can only operate on specific partitioning schemes

    match (partitioning, ordering) {
        (Singleton, Some(ordering)) => (
            ExchangerType::FullMerge,
            vec![
                (OrderedPartitioning(ordering.clone().into_columns()), false),
                (HashPartitioning(vec![]), true),
            ],
        ),
        (Singleton, None) => (ExchangerType::FullMerge, vec![(HashPartitioning(vec![]), true)]),
        (Partitioned(_), _) => (ExchangerType::Init, vec![(Singleton, true)]),
        (OrderedPartitioning(cols), Some(ordering)) if cols == &ordering.clone().into_columns() => {
            (ExchangerType::Init, vec![(Singleton, false)])
        }
        (OrderedPartitioning(_), _) => (ExchangerType::Init, vec![(Singleton, true)]),
        (HashPartitioning(_), _) => (ExchangerType::Init, vec![(Singleton, true)]),
    }
}
