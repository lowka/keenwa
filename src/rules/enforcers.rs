use crate::error::OptimizerError;
use crate::operators::relational::join::{get_join_columns_pair, JoinCondition};
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::relational::RelNode;
use crate::properties::physical::PhysicalProperties;
use crate::properties::OrderingChoice;
use std::fmt::Debug;

pub trait EnforcerRules: Debug {
    fn evaluate_properties(
        &self,
        expr: &PhysicalExpr,
        properties: &PhysicalProperties,
    ) -> Result<(bool, bool), OptimizerError> {
        evaluate_properties(expr, properties)
    }

    fn can_explore_with_enforcer(&self, expr: &LogicalExpr, properties: &PhysicalProperties) -> bool {
        if !properties.is_empty() {
            matches!(expr, LogicalExpr::Select { .. } | LogicalExpr::Projection { .. })
        } else {
            false
        }
    }

    fn create_enforcer(
        &self,
        properties: &PhysicalProperties,
        input: RelNode,
    ) -> Result<(PhysicalExpr, PhysicalProperties), OptimizerError>;
}

#[derive(Debug)]
pub struct DefaultEnforcers;

impl EnforcerRules for DefaultEnforcers {
    fn create_enforcer(
        &self,
        properties: &PhysicalProperties,
        input: RelNode,
    ) -> Result<(PhysicalExpr, PhysicalProperties), OptimizerError> {
        create_enforcer(properties, input)
    }
}

fn create_enforcer(
    properties: &PhysicalProperties,
    input: RelNode,
) -> Result<(PhysicalExpr, PhysicalProperties), OptimizerError> {
    if let Some(ordering) = properties.ordering() {
        let sort_enforcer = PhysicalExpr::Sort {
            input,
            ordering: ordering.clone(),
        };
        Ok((sort_enforcer, PhysicalProperties::none()))
    } else {
        let message = format!("Unexpected physical property. Only ordering is supported: {:?}", properties);
        Err(OptimizerError::NotImplemented(message))
    }
}

fn evaluate_properties(
    expr: &PhysicalExpr,
    required_properties: &PhysicalProperties,
) -> Result<(bool, bool), OptimizerError> {
    let provides_property = expr_provides_property(expr, required_properties)?;
    let retains_property = if !provides_property {
        expr_retains_property(expr, required_properties)?
    } else {
        false
    };
    Ok((provides_property, retains_property))
}

pub fn expr_retains_property(expr: &PhysicalExpr, required: &PhysicalProperties) -> Result<bool, OptimizerError> {
    let retains = match (expr, required.as_option()) {
        (_, None) => true,
        (PhysicalExpr::Select { .. }, Some(_)) => true,
        (
            PhysicalExpr::MergeSortJoin {
                left, right, condition, ..
            },
            Some(ordering),
        ) => join_provides_ordering(left, right, condition, ordering),
        (PhysicalExpr::IndexScan { columns, .. }, Some(ordering)) => {
            let idx_ordering = OrderingChoice::new(columns.clone());
            ordering_is_preserved(&idx_ordering, ordering)
        }
        (
            PhysicalExpr::Sort {
                ordering: sort_ordering,
                ..
            },
            Some(ordering),
        ) => ordering_is_preserved(sort_ordering, ordering),
        // ???: projection w/o expressions always retains required physical properties
        (_, _) => false,
    };
    Ok(retains)
}

pub fn expr_provides_property(expr: &PhysicalExpr, required: &PhysicalProperties) -> Result<bool, OptimizerError> {
    let preserved = match (expr, required.as_option()) {
        (_, None) => true,
        (
            PhysicalExpr::MergeSortJoin {
                left, right, condition, ..
            },
            Some(ordering),
        ) => join_provides_ordering(left, right, condition, ordering),
        (PhysicalExpr::IndexScan { columns, .. }, Some(ordering)) => {
            let idx_ordering = OrderingChoice::new(columns.clone());
            ordering_is_preserved(&idx_ordering, ordering)
        }
        (
            PhysicalExpr::Sort {
                ordering: sort_ordering,
                ..
            },
            Some(ordering),
        ) => ordering_is_preserved(sort_ordering, ordering),
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
            let left_ordering = OrderingChoice::new(left);
            ordering_is_preserved(&left_ordering, ordering)
        } else {
            false
        };
        let right_side_ordered = if !right.is_empty() {
            let right_ordering = OrderingChoice::new(right);
            ordering_is_preserved(&right_ordering, ordering)
        } else {
            false
        };

        left_side_ordered || right_side_ordered
    } else {
        false
    }
}

fn ordering_is_preserved(ord: &OrderingChoice, other: Option<&OrderingChoice>) -> bool {
    match other {
        Some(other) => {
            let columns = ord.columns();
            let other_columns = other.columns();
            let ord_match_num = columns.iter().zip(other_columns).filter(|(s, r)| s == r).count();
            columns.len() == ord_match_num
        }
        None => false,
    }
}
