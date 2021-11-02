use crate::operators::join::JoinCondition;
use crate::operators::logical::LogicalExpr;
use crate::operators::physical::PhysicalExpr;
use crate::operators::InputExpr;
use crate::properties::physical::PhysicalProperties;
use crate::properties::OrderingChoice;
use std::fmt::Debug;

pub trait EnforcerRules: Debug {
    fn evaluate_properties(&self, expr: &PhysicalExpr, properties: &PhysicalProperties) -> (bool, bool) {
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
    ) -> Result<(PhysicalExpr, PhysicalProperties), String>;
}

#[derive(Debug)]
pub struct DefaultEnforcers;

impl EnforcerRules for DefaultEnforcers {
    fn create_enforcer(
        &self,
        properties: &PhysicalProperties,
        input: InputExpr,
    ) -> Result<(PhysicalExpr, PhysicalProperties), String> {
        create_enforcer(properties, input)
    }
}

fn create_enforcer(
    properties: &PhysicalProperties,
    input: InputExpr,
) -> Result<(PhysicalExpr, PhysicalProperties), String> {
    if let Some(ordering) = properties.ordering() {
        let sort_enforcer = PhysicalExpr::Sort {
            input,
            ordering: ordering.clone(),
        };
        Ok((sort_enforcer, PhysicalProperties::none()))
    } else {
        panic!("Unexpected physical property. Only ordering is supported: {:?}", properties)
    }
}

fn evaluate_properties(expr: &PhysicalExpr, required_properties: &PhysicalProperties) -> (bool, bool) {
    let provides_property = expr_provides_property(expr, required_properties);
    let retains_property = if !provides_property {
        expr_retains_property(expr, required_properties)
    } else {
        false
    };
    (provides_property, retains_property)
}

pub fn expr_retains_property(expr: &PhysicalExpr, required: &PhysicalProperties) -> bool {
    match (expr, required.as_option()) {
        (_, None) => true,
        (PhysicalExpr::Select { .. }, Some(_)) => true,
        (
            PhysicalExpr::MergeSortJoin {
                left, right, condition, ..
            },
            Some(ordering),
        ) => {
            let left_columns = left.attrs().logical().output_columns();
            let right_columns = right.attrs().logical().output_columns();
            let (left, right) = JoinCondition::filter_columns(condition, left_columns, right_columns);
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
        (PhysicalExpr::Expr { expr }, Some(p)) => {
            panic!("Expr operator does not support physical properties. Expr: {:?}. Properties: {:?}", expr, p)
        }
        // ???: projection w/o expressions always retains required physical properties
        (_, _) => false,
    }
}

pub fn expr_provides_property(expr: &PhysicalExpr, required: &PhysicalProperties) -> bool {
    match (expr, required.as_option()) {
        (_, None) => true,
        (
            PhysicalExpr::MergeSortJoin {
                left, right, condition, ..
            },
            Some(ordering),
        ) => {
            let left_columns = left.attrs().logical().output_columns();
            let right_columns = right.attrs().logical().output_columns();
            let (left, right) = JoinCondition::filter_columns(condition, left_columns, right_columns);
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
        (PhysicalExpr::Expr { expr }, Some(p)) => {
            panic!("Expr operator does not support physical properties. Expr: {:?}. Properties: {:?}", expr, p)
        }
        (_, Some(_)) => false,
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
