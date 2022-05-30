use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::relational::logical::LogicalExpr;
use crate::operators::relational::physical::{IndexScan, MergeSortJoin, PhysicalExpr, Select, Sort, Unique};
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
            // FIXME: We can only pass the ordering past the projection if that ordering does not use synthentic columns
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
                left_ordering,
                right_ordering,
                ..
            }),
            Some(ordering),
        ) => join_provides_ordering(ordering, left_ordering, right_ordering),
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
        (
            PhysicalExpr::Unique(Unique {
                inputs,
                columns: output_columns,
                ordering: output_ordering,
                ..
            }),
            Some(ordering),
        ) => unique_provides_ordering(ordering, inputs, output_ordering, output_columns),
        // ???: projection w/o expressions always retains required physical properties
        (_, _) => false,
    };
    Ok(retains)
}

pub fn expr_provides_property(expr: &PhysicalExpr, required: &RequiredProperties) -> Result<bool, OptimizerError> {
    let provides = match (expr, required.ordering()) {
        (_, None) => true,
        (
            PhysicalExpr::MergeSortJoin(MergeSortJoin {
                left_ordering,
                right_ordering,
                ..
            }),
            Some(ordering),
        ) => join_provides_ordering(ordering, left_ordering, right_ordering),
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
    Ok(provides)
}

fn join_provides_ordering(
    ordering: &OrderingChoice,
    left_ordering: &OrderingChoice,
    right_ordering: &OrderingChoice,
) -> bool {
    // Convert the required ordering to the ordering where all columns from the right side
    // of the join are replaced with columns from the left side of the join.
    let left_columns: Vec<_> = left_ordering.clone().into_columns();
    let right_columns: Vec<_> = right_ordering.clone().into_columns();

    let normalized = ordering.with_mapping(&right_columns, &left_columns);
    // the resulting ordering must be a prefix of the left ordering.
    normalized.prefix_of(left_ordering)
}

fn unique_provides_ordering(
    ordering: &OrderingChoice,
    inputs: &[RelNode],
    output_ordering: &[OrderingChoice],
    output_columns: &[ColumnId],
) -> bool {
    // All inputs of a unique operator are ordered in the same way
    // and the required ordering uses columns from its output. E.g.:
    //
    // Inputs:
    //  - a1 asc, a2 asc, a3 desc
    //  - b2 asc, b2 asc, b3 desc
    // Output:
    //    o1 asc, o2 asc, o3 desc
    //
    // Ordering:
    //    o1 asc, o2 asc
    //
    // Replace output columns from the required ordering with
    // columns of the first input and check whether the required
    // ordering if the prefix of the resulting one.
    //
    let input = &inputs[0];
    let input_columns = input.props().logical().output_columns();
    let input_ordering = &output_ordering[0];
    let ord = input_ordering.with_mapping(input_columns, output_columns);

    ordering.prefix_of(&ord)
}

fn ordering_is_preserved(ord: &OrderingChoice, required: Option<&OrderingChoice>) -> bool {
    match required {
        Some(required) => required.prefix_of(ord),
        None => false,
    }
}
