use crate::error::OptimizerError;
use crate::operators::relational::join::{get_join_columns_pair, JoinCondition};
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::{IndexScan, MergeSortJoin, PhysicalExpr, Select, Sort};
use crate::operators::relational::RelNode;
use crate::properties::physical::RequiredProperties;
use crate::properties::OrderingChoice;
use crate::rules::EvaluationResponse;

/// Enforcer-related methods of the [RuleSet](super::RuleSet) trait.
pub trait EnforcerRules {
    /// See [RuleSet::evaluate_properties](super::RuleSet::evaluate_properties).
    fn evaluate_properties(
        &self,
        expr: &PhysicalExpr,
        properties: &RequiredProperties,
    ) -> Result<EvaluationResponse, OptimizerError> {
        evaluate_properties(expr, properties)
    }

    /// See [RuleSet::can_explore_with_enforcer](super::RuleSet::can_explore_with_enforcer).
    fn can_explore_with_enforcer(&self, expr: &LogicalExpr, properties: &RequiredProperties) -> bool {
        if properties.ordering().is_some() {
            matches!(expr, LogicalExpr::Select { .. } | LogicalExpr::Projection { .. })
        } else {
            false
        }
    }

    /// See [RuleSet::create_enforcer](super::RuleSet::create_enforcer).
    fn create_enforcer(
        &self,
        properties: &RequiredProperties,
        input: RelNode,
    ) -> Result<(PhysicalExpr, Option<RequiredProperties>), OptimizerError>;
}

/// Default implementation of [EnforcerRules].
/// Provides an enforcer of a sorted physical property.
#[derive(Debug)]
pub struct DefaultEnforcers;

impl EnforcerRules for DefaultEnforcers {
    fn create_enforcer(
        &self,
        properties: &RequiredProperties,
        input: RelNode,
    ) -> Result<(PhysicalExpr, Option<RequiredProperties>), OptimizerError> {
        create_enforcer(properties, input)
    }
}

fn create_enforcer(
    properties: &RequiredProperties,
    input: RelNode,
) -> Result<(PhysicalExpr, Option<RequiredProperties>), OptimizerError> {
    if let Some(ordering) = properties.ordering() {
        let sort_enforcer = PhysicalExpr::Sort(Sort {
            input,
            ordering: ordering.clone(),
        });
        Ok((sort_enforcer, None))
    } else {
        let message = format!("Unexpected physical property. Only ordering is supported: {:?}", properties);
        Err(OptimizerError::NotImplemented(message))
    }
}

fn evaluate_properties(
    expr: &PhysicalExpr,
    required_properties: &RequiredProperties,
) -> Result<EvaluationResponse, OptimizerError> {
    let provides_property = expr_provides_property(expr, required_properties)?;
    let retains_property = if !provides_property {
        expr_retains_property(expr, required_properties)?
    } else {
        false
    };
    Ok(EvaluationResponse {
        provides_property,
        retains_property,
    })
}

pub fn expr_retains_property(expr: &PhysicalExpr, required: &RequiredProperties) -> Result<bool, OptimizerError> {
    let retains = match (expr, required.ordering()) {
        (_, None) => true,
        (PhysicalExpr::Select(Select { .. }), Some(_)) => true,
        (
            PhysicalExpr::MergeSortJoin(MergeSortJoin {
                left, right, condition, ..
            }),
            Some(ordering),
        ) => join_provides_ordering(left, right, condition, Some(ordering)),
        (
            PhysicalExpr::IndexScan(IndexScan {
                ordering: column_ordering,
                ..
            }),
            Some(ordering),
        ) => column_ordering
            .as_ref()
            .map(|ord| ordering_is_preserved(ord, Some(ordering)))
            .unwrap_or_default(),
        (
            PhysicalExpr::Sort(Sort {
                ordering: sort_ordering,
                ..
            }),
            Some(ordering),
        ) => ordering_is_preserved(sort_ordering, Some(ordering)),
        // ???: projection w/o expressions always retains required physical properties
        (_, _) => false,
    };
    Ok(retains)
}

pub fn expr_provides_property(expr: &PhysicalExpr, required: &RequiredProperties) -> Result<bool, OptimizerError> {
    let preserved = match (expr, required.ordering()) {
        (_, None) => true,
        (
            PhysicalExpr::MergeSortJoin(MergeSortJoin {
                left, right, condition, ..
            }),
            Some(ordering),
        ) => join_provides_ordering(left, right, condition, Some(ordering)),
        (
            PhysicalExpr::IndexScan(IndexScan {
                ordering: column_ordering,
                ..
            }),
            Some(ordering),
        ) => column_ordering
            .as_ref()
            .map(|ord| ordering_is_preserved(ord, Some(ordering)))
            .unwrap_or_default(),
        (
            PhysicalExpr::Sort(Sort {
                ordering: sort_ordering,
                ..
            }),
            Some(ordering),
        ) => ordering_is_preserved(sort_ordering, Some(ordering)),
        (_, Some(_)) => false,
    };
    Ok(preserved)
}

fn join_provides_ordering(
    left: &RelNode,
    right: &RelNode,
    condition: &JoinCondition,
    ordering: Option<&OrderingChoice>,
) -> bool {
    assert!(ordering.is_some(), "ordering must be present");

    if let Some((left, right)) = get_join_columns_pair(left, right, condition) {
        let left_side_ordered = if !left.is_empty() {
            let left_ordering = OrderingChoice::from_columns(left);
            ordering_is_preserved(&left_ordering, ordering)
        } else {
            false
        };
        let right_side_ordered = if !right.is_empty() {
            let right_ordering = OrderingChoice::from_columns(right);
            ordering_is_preserved(&right_ordering, ordering)
        } else {
            false
        };

        left_side_ordered || right_side_ordered
    } else {
        false
    }
}

fn ordering_is_preserved(ord: &OrderingChoice, required: Option<&OrderingChoice>) -> bool {
    match required {
        Some(required) => required.prefix_of(ord),
        None => false,
    }
}
