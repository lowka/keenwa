use crate::error::OptimizerError;
use crate::operators::join::JoinCondition;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
use crate::operators::InputExpr;
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
        input: InputExpr,
    ) -> Result<(PhysicalExpr, PhysicalProperties), OptimizerError>;
}

#[derive(Debug)]
pub struct DefaultEnforcers;

impl EnforcerRules for DefaultEnforcers {
    fn create_enforcer(
        &self,
        properties: &PhysicalProperties,
        input: InputExpr,
    ) -> Result<(PhysicalExpr, PhysicalProperties), OptimizerError> {
        create_enforcer(properties, input)
    }
}

fn create_enforcer(
    properties: &PhysicalProperties,
    input: InputExpr,
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
        (PhysicalExpr::MergeSortJoin { condition, .. }, Some(ordering)) => {
            let (left, right) = match condition {
                JoinCondition::Using(using) => using.as_columns_pair(),
            };
            let left_ordering = OrderingChoice::new(left);
            let right_ordering = OrderingChoice::new(right);

            ordering_is_preserved(&left_ordering, ordering) || ordering_is_preserved(&right_ordering, ordering)
        }
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
        (PhysicalExpr::Expr { expr: _ }, Some(_)) => {
            return operator_does_not_support_properties(expr, required);
        }
        // ???: projection w/o expressions always retains required physical properties
        (_, _) => false,
    };
    Ok(retains)
}

pub fn expr_provides_property(expr: &PhysicalExpr, required: &PhysicalProperties) -> Result<bool, OptimizerError> {
    let preserved = match (expr, required.as_option()) {
        (_, None) => true,
        (PhysicalExpr::MergeSortJoin { condition, .. }, Some(ordering)) => {
            let (left, right) = match condition {
                JoinCondition::Using(using) => using.as_columns_pair(),
            };
            let left_ordering = OrderingChoice::new(left);
            let right_ordering = OrderingChoice::new(right);

            ordering_is_preserved(&left_ordering, ordering) || ordering_is_preserved(&right_ordering, ordering)
        }
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
        (PhysicalExpr::Expr { expr: _ }, Some(_)) => {
            return operator_does_not_support_properties(expr, required);
        }
        (_, Some(_)) => false,
    };
    Ok(preserved)
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

fn operator_does_not_support_properties(expr: &PhysicalExpr, p: &PhysicalProperties) -> Result<bool, OptimizerError> {
    let message = format!("Operator does not support physical properties. Operator: {:?}. Properties: {:?}", expr, p);
    Err(OptimizerError::Internal(message))
}
