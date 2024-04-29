use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::not_supported;
use crate::operators::scalar::expr::{BinaryOp, Expr, NestedExpr};
use itertools::Itertools;

/// A helper trait that provides the type of the given column.
pub trait ColumnTypeRegistry {
    /// Returns the type of the given column.
    fn get_column_type(&self, col_id: &ColumnId) -> DataType;
}

/// Resolves the type of the given expression.
///
/// Because the optimizer uses column identifiers
/// instead of column references this method returns an error if the given expression is
/// a [column reference expression](super::expr::Expr::ColumnName).
///
pub fn resolve_expr_type<T, R>(expr: &Expr<T>, column_registry: &R) -> Result<DataType, OptimizerError>
where
    T: NestedExpr,
    R: ColumnTypeRegistry + ?Sized,
{
    match expr {
        Expr::Column(column_id) => Ok(column_registry.get_column_type(column_id)),
        Expr::ColumnName(_) => {
            let message = format!("Expr {} should have been replaced with Column(id)", expr);
            Err(OptimizerError::internal(message))
        }
        Expr::Scalar(value) => Ok(value.data_type()),
        Expr::BinaryExpr { lhs, op, rhs } => {
            let left_tpe = resolve_expr_type(lhs, column_registry)?;
            let right_tpe = resolve_expr_type(rhs, column_registry)?;
            resolve_binary_expr_type(expr, op, left_tpe, right_tpe)
        }
        Expr::Cast { data_type, .. } => Ok(data_type.clone()),
        Expr::Not(expr) => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            expect_type_or_null(&tpe, &DataType::Bool, expr)?;
            Ok(tpe)
        }
        Expr::Negation(expr) => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            match tpe {
                DataType::Int32 => Ok(tpe),
                DataType::Float32 => Ok(tpe),
                _ => {
                    let message = format!("Expr: {}. No operator for type {}", expr, tpe);
                    Err(type_error(message))
                }
            }
        }
        Expr::Alias(expr, _) => resolve_expr_type(expr, column_registry),
        Expr::InList { expr, exprs, .. } => {
            let _ = resolve_expr_type(expr, column_registry)?;
            for expr in exprs {
                let _ = resolve_expr_type(expr, column_registry)?;
            }
            Ok(DataType::Bool)
        }
        Expr::IsNull { expr, .. } => {
            let _ = resolve_expr_type(expr, column_registry)?;
            Ok(DataType::Bool)
        }
        Expr::IsFalse { expr, .. } => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            expect_type_or_null(&tpe, &DataType::Bool, &expr)?;
            Ok(DataType::Bool)
        }
        Expr::IsTrue { expr, .. } => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            expect_type_or_null(&tpe, &DataType::Bool, &expr)?;
            Ok(DataType::Bool)
        }
        Expr::IsUnknown { expr, .. } => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            expect_type_or_null(&tpe, &DataType::Bool, &expr)?;
            Ok(DataType::Bool)
        }
        Expr::Between { expr, low, high, .. } => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            let low_type = resolve_expr_type(low, column_registry)?;
            let high_type = resolve_expr_type(high, column_registry)?;
            expect_type_or_null(&low_type, &tpe, low)?;
            expect_type_or_null(&high_type, &tpe, high)?;
            Ok(tpe)
        }
        Expr::Case {
            expr: base_expr,
            when_then_exprs,
            else_expr,
        } => {
            // There are two forms of CASE expressions:
            //
            // - The basic form (1):
            //   CASE WHEN col:1 == 1 THEN 'one'
            //        WHEN col:1 == 2 THEN 'two'
            //        ELSE 'three'
            //
            // - The "simple" form (2):
            //   CASE col:1 WHEN 1 THEN 'one'
            //              WHEN 2 THEN 'two'
            //              ELSE 'three'
            //
            // (1) In the basic form WHEN expressions are boolean expressions.
            //
            // (2) In the "simple" form WHEN expressions must be of the same
            // type as the expression that follows the CASE keyword
            // (col:1 in the example 2 above).
            //
            let when_expr_type = if let Some(expr) = base_expr {
                resolve_expr_type(expr, column_registry)?
            } else {
                DataType::Bool
            };

            let mut expr_result_type: Option<DataType> = None;

            for (when, then) in when_then_exprs {
                let condition_type = resolve_expr_type(when, column_registry)?;
                expect_type_or_null(&condition_type, &when_expr_type, when)?;
                let result_type = resolve_expr_type(then, column_registry)?;

                match &mut expr_result_type {
                    Some(tpe) if tpe != &result_type => {
                        return Err(OptimizerError::internal(format!(
                            "Unexpected result type in a THEN clause of a CASE expression. Expected {} but got {}",
                            tpe, result_type
                        )));
                    }
                    _ => expr_result_type = Some(result_type),
                }
            }

            if let Some(else_expr) = else_expr {
                let result_type = resolve_expr_type(else_expr, column_registry)?;
                match &mut expr_result_type {
                    Some(tpe) if tpe != &result_type => {
                        let message = format!(
                            "Unexpected result type in an ELSE clause of a CASE expression. Expected {} but got {}",
                            tpe, result_type
                        );
                        return Err(type_error(message));
                    }
                    _ => expr_result_type = Some(result_type),
                }
            }

            match expr_result_type {
                Some(result_type) => Ok(result_type),
                None => Err(OptimizerError::argument("Invalid CASE expression")),
            }
        }
        Expr::Tuple(exprs) => {
            let expr_types: Result<Vec<DataType>, _> =
                exprs.iter().map(|expr| resolve_expr_type(expr, column_registry)).collect();
            Ok(DataType::Tuple(expr_types?))
        }
        Expr::Array(exprs) => {
            let expr_types: Result<Vec<DataType>, _> =
                exprs.iter().map(|expr| resolve_expr_type(expr, column_registry)).collect();
            let expr_types = expr_types?;

            if expr_types.is_empty() {
                //TODO: Untyped array case.
                not_supported!("Empty array expression")
            }

            let mut element_type = None;
            for expr_type in expr_types.iter() {
                if expr_type != &DataType::Null {
                    if element_type.is_none() {
                        element_type = Some(expr_type.clone())
                    } else if Some(expr_type) != element_type.as_ref() {
                        return Err(OptimizerError::argument("Array with elements of different types"));
                    }
                }
            }

            match element_type {
                Some(DataType::Array(element_type)) => Ok(DataType::Array(element_type)),
                Some(element_type) => Ok(DataType::Array(Box::new(element_type))),
                None => {
                    Err(type_error(format!("Unable to resolve array element type: [{}]", expr_types.iter().join(", "))))
                }
            }
        }
        Expr::ArrayIndex { array, indexes } => {
            let arr_type = resolve_expr_type(array, column_registry)?;
            if let DataType::Array(element_type) = arr_type {
                for expr in indexes {
                    let index_type = resolve_expr_type(expr, column_registry)?;
                    if index_type != DataType::Int32 {
                        return Err(type_error(format!("Invalid array index type {}", index_type)));
                    }
                }
                Ok(*element_type)
            } else {
                Err(type_error(format!("Expected array but got {}", arr_type)))
            }
        }
        Expr::ScalarFunction { func, args } => {
            let arg_types: Result<Vec<DataType>, _> =
                args.iter().map(|arg| resolve_expr_type(arg, column_registry)).collect();

            let arg_types = arg_types?;
            func.get_return_type(&arg_types)
        }
        Expr::Like { expr, pattern, .. } => {
            let expr_type = resolve_expr_type(expr, column_registry)?;
            expect_type_or_null(&expr_type, &DataType::String, &expr)?;

            let pattern_type = resolve_expr_type(pattern, column_registry)?;
            expect_type_or_null(&pattern_type, &DataType::String, &expr)?;

            Ok(DataType::Bool)
        }
        Expr::Aggregate { func, args, .. } => {
            let arg_types: Result<Vec<DataType>, _> =
                args.iter().map(|arg| resolve_expr_type(arg, column_registry)).collect();

            let arg_types = arg_types?;
            func.get_return_type(&arg_types)
        }
        Expr::WindowAggregate { func, args, .. } => {
            let arg_types: Result<Vec<DataType>, _> =
                args.iter().map(|arg| resolve_expr_type(arg, column_registry)).collect();

            let arg_types = arg_types?;
            func.get_return_type(&arg_types)
        }
        Expr::Ordering { expr, .. } => {
            let message = format!("Ordering expression {} does not support type resolution", expr);
            Err(OptimizerError::argument(message))
        }
        Expr::Exists { .. } | Expr::InSubQuery { .. } => {
            // query must be a valid subquery.
            Ok(DataType::Bool)
        }
        Expr::SubQuery(query) => {
            // query must be a valid subquery.
            let cols = query.output_columns();
            let col_id = cols[0];
            Ok(column_registry.get_column_type(&col_id))
        }
        Expr::Wildcard(_) => {
            let message = format!("Expr {} should have been replaced with Column(id)", expr);
            Err(OptimizerError::argument(message))
        }
    }
}

fn resolve_binary_expr_type<T>(
    expr: &Expr<T>,
    op: &BinaryOp,
    lhs: DataType,
    rhs: DataType,
) -> Result<DataType, OptimizerError>
where
    T: NestedExpr,
{
    use BinaryOp::*;
    use DataType::*;

    fn is_comparable_type(tpe: &DataType) -> bool {
        match tpe {
            Null => true,
            Bool => false,
            Int32 => true,
            Float32 => true,
            String => false,
            Date => true,
            Time => true,
            Timestamp(_) => true,
            Interval => true,
            Tuple(_) => false,
            Array(_) => false,
        }
    }

    fn support_comparison_ops(lhs: &DataType, rhs: &DataType) -> bool {
        match (lhs, rhs) {
            _ if is_comparable_type(lhs) && is_comparable_type(rhs) && lhs == rhs => true,
            _ if is_comparable_type(lhs) && rhs == &Null => true,
            _ if lhs == &Null && is_comparable_type(rhs) => true,
            _ if are_numeric_types(lhs, rhs) => true,
            _ => false,
        }
    }

    fn support_arithmetic_ops(lhs: &DataType, rhs: &DataType, types: &[DataType]) -> bool {
        match (lhs, rhs) {
            (lhs, rhs) if types.contains(lhs) && types.contains(rhs) => true,
            (lhs, Null) if types.contains(lhs) => true,
            (Null, rhs) if types.contains(rhs) => true,
            _ => false,
        }
    }

    fn support_string_ops(lhs: &DataType, rhs: &DataType) -> bool {
        matches!((lhs, rhs), (String, String) | (String, Null) | (Null, String) | (Null, Null))
    }

    fn support_equality(lhs: &DataType, rhs: &DataType) -> bool {
        match (lhs, rhs) {
            _ if lhs == rhs => true,
            _ if lhs == &Null || rhs == &Null => true,
            _ if are_numeric_types(lhs, rhs) => true,
            _ => false,
        }
    }

    match (&lhs, &op, &rhs) {
        // Logical
        (Bool, And, Bool) => Ok(Bool),
        (Bool, Or, Bool) => Ok(Bool),
        (Bool, And, Null) => Ok(Bool),
        (Bool, Or, Null) => Ok(Bool),

        (Null, And, Bool) => Ok(Bool),
        (Null, Or, Bool) => Ok(Bool),
        (Null, And, Null) => Ok(Bool),
        (Null, Or, Null) => Ok(Bool),
        // Logical

        // Arithmetic
        (lhs, Plus | Minus | Multiply | Divide | Modulo, rhs) if support_arithmetic_ops(lhs, rhs, &[Int32]) => {
            Ok(Int32)
        }
        // int32 <op> float32 = float32
        // float32 <op> int32 = float32
        (lhs, Plus | Minus | Multiply | Divide, rhs) if support_arithmetic_ops(lhs, rhs, &[Int32, Float32]) => {
            Ok(Float32)
        }

        // Comparison
        (lhs, Lt | LtEq | Gt | GtEq, rhs) if support_comparison_ops(lhs, rhs) => Ok(Bool),

        // Date
        (Date, Minus, Date) => Ok(Interval),
        (Date, Plus, Interval) => Ok(Date),
        (Date, Minus, Interval) => Ok(Date),

        // Time
        (Time, Minus, Time) => Ok(Interval),
        (Time, Plus, Interval) => Ok(Time),
        (Time, Minus, Interval) => Ok(Interval),
        (Interval, Minus, Time) => Ok(Interval),
        (Time, Minus, Null) => Ok(Interval),
        (Null, Minus, Time) => Ok(Interval),

        // Timestamp
        (Timestamp(tz), Plus | Minus, Interval) => Ok(Timestamp(*tz)),
        (Timestamp(tz), Plus | Minus, Null) => Ok(Timestamp(*tz)),
        (Null, Plus | Minus, Timestamp(tz)) => Ok(Timestamp(*tz)),

        // Interval
        (Interval, Plus | Minus, Interval) => Ok(Interval),
        (Interval, Plus, Date) => Ok(Date),
        (Interval, Plus, Time) => Ok(Time),
        (Interval, Plus | Minus, Null) => Ok(Interval),
        (Null, Plus | Minus, Interval) => Ok(Interval),

        // <interval> mul/div <num>
        (Interval, Multiply | Divide, Int32) => Ok(Interval),
        (Interval, Multiply | Divide, Null) => Ok(Interval),
        (Int32, Multiply, Interval) => Ok(Interval),
        (Null, Multiply, Interval) => Ok(Interval),

        // String Concat
        (String, Concat, _) => Ok(String),
        (_, Concat, String) => Ok(String),
        (Null, Concat, Null) => Ok(String),

        // Equality
        (lhs, Eq | NotEq, rhs) if support_equality(lhs, rhs) => Ok(Bool),

        _ => {
            let message = format!("Expr: {} No operator for {} {} {}", expr, lhs, op, rhs);
            Err(type_error(message))
        }
    }
}

fn are_numeric_types(lhs: &DataType, rhs: &DataType) -> bool {
    DataType::NUMERIC_TYPES.contains(lhs) && DataType::NUMERIC_TYPES.contains(rhs)
}

fn expect_type_or_null<T>(tpe: &DataType, expected: &DataType, expr: &Expr<T>) -> Result<(), OptimizerError>
where
    T: NestedExpr,
{
    if tpe != &DataType::Null && tpe != expected {
        let message = format!("Expr: {}. Expected type {} but got {}", expr, expected, tpe);
        Err(type_error(message))
    } else {
        Ok(())
    }
}

fn type_error<T>(message: T) -> OptimizerError
where
    T: Into<String>,
{
    OptimizerError::internal(format!("Type error: {}", message.into()))
}

#[cfg(test)]
mod test {
    use crate::datatypes::DataType;
    use crate::meta::testing::TestMetadata;
    use crate::meta::ColumnId;
    use crate::operators::scalar::aggregates::AggregateFunction;
    use crate::operators::scalar::expr::{BinaryOp, NestedExpr};
    use crate::operators::scalar::types::{resolve_binary_expr_type, resolve_expr_type, ColumnTypeRegistry};
    use crate::operators::scalar::value::{Array, Interval, ScalarValue};
    use itertools::Itertools;
    use std::convert::TryFrom;
    use std::fmt::Formatter;
    use std::hash::{Hash, Hasher};
    use std::ops::{Add, Div, Mul, Rem, Sub};

    #[derive(Debug, Clone)]
    struct DummyRelExpr(Vec<ColumnId>);

    impl Eq for DummyRelExpr {}

    impl PartialEq<Self> for DummyRelExpr {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }

    impl Hash for DummyRelExpr {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_usize(0)
        }
    }

    impl NestedExpr for DummyRelExpr {
        fn output_columns(&self) -> &[ColumnId] {
            &self.0
        }

        fn outer_columns(&self) -> &[ColumnId] {
            &[]
        }

        fn write_to_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    type Expr = super::Expr<DummyRelExpr>;

    fn null_value() -> Expr {
        Expr::Scalar(ScalarValue::Null)
    }

    fn bool_value() -> Expr {
        Expr::Scalar(ScalarValue::Bool(Some(true)))
    }

    fn int_value() -> Expr {
        Expr::Scalar(ScalarValue::Int32(Some(1)))
    }

    fn fp_value() -> Expr {
        Expr::Scalar(ScalarValue::Float32(Some(1.0)))
    }

    fn str_value() -> Expr {
        Expr::Scalar(ScalarValue::String(Some("s".to_string())))
    }

    fn date_value() -> Expr {
        Expr::Scalar(ScalarValue::Date(Some(10000)))
    }

    fn time_value() -> Expr {
        Expr::Scalar(ScalarValue::Time(Some((100, 0))))
    }

    fn timestamp_value(tz: bool) -> Expr {
        let tz = if tz { Some(1000) } else { None };
        Expr::Scalar(ScalarValue::Timestamp(Some(1000), tz))
    }

    fn interval_value(years: bool) -> Expr {
        let interval = if years {
            Interval::YearMonth(100, 1)
        } else {
            Interval::DaySecond(10, 0)
        };
        Expr::Scalar(ScalarValue::Interval(Some(interval)))
    }

    fn col_name(name: &str) -> Expr {
        Expr::ColumnName(name.into())
    }

    struct MockColumnTypeRegistry {
        metadata: TestMetadata,
    }

    impl MockColumnTypeRegistry {
        fn new() -> Self {
            MockColumnTypeRegistry {
                metadata: TestMetadata::new(),
            }
        }

        fn add_column(&mut self, data_type: DataType) -> ColumnId {
            self.metadata.synthetic_column().data_type(data_type).build()
        }
    }

    impl ColumnTypeRegistry for MockColumnTypeRegistry {
        fn get_column_type(&self, col_id: &ColumnId) -> DataType {
            self.metadata.get_column(col_id).data_type().clone()
        }
    }

    fn new_subquery_from_column(registry: &mut MockColumnTypeRegistry, col_type: DataType) -> DummyRelExpr {
        let id = registry.add_column(col_type);
        DummyRelExpr(vec![id])
    }

    #[test]
    fn test_column_name_type_should_not_be_resolved() {
        let col = Expr::ColumnName("a".into());
        expect_not_resolved(&col)
    }

    #[test]
    fn test_column_id_type() {
        fn expect_column_type(col_type: DataType) {
            let mut registry = MockColumnTypeRegistry::new();
            let id = registry.add_column(col_type.clone());
            let expr = Expr::Column(id);

            expect_resolved(&expr, &registry, &col_type);
        }

        for tpe in get_data_types() {
            expect_column_type(tpe);
        }
    }

    #[test]
    fn test_scalar_type() {
        for tpe in get_data_types() {
            let value = get_value_expr(tpe.clone());
            expect_type(&value, &tpe);
        }
    }

    #[test]
    fn test_subquery_type() {
        fn expect_subquery_type(col_type: DataType) {
            let mut registry = MockColumnTypeRegistry::new();
            let subquery = new_subquery_from_column(&mut registry, col_type.clone());
            let expr = Expr::SubQuery(subquery);

            expect_resolved(&expr, &registry, &col_type);
        }

        expect_subquery_type(DataType::Bool);
        expect_subquery_type(DataType::Int32);
        expect_subquery_type(DataType::String);
    }

    #[test]
    fn test_exists_subquery() {
        let mut registry = MockColumnTypeRegistry::new();
        let query = new_subquery_from_column(&mut registry, DataType::Int32);
        let expr = Expr::Exists { not: false, query };

        expect_resolved(&expr, &registry, &DataType::Bool);
    }

    #[test]
    fn test_in_subquery() {
        let mut registry = MockColumnTypeRegistry::new();
        let query = new_subquery_from_column(&mut registry, DataType::Int32);
        let expr = Expr::InSubQuery {
            expr: Box::new(col_name("a")),
            not: false,
            query,
        };

        expect_resolved(&expr, &registry, &DataType::Bool);
    }

    #[test]
    fn test_cast_type() {
        // cast any type to any other type.
        for tpe in get_data_types() {
            for another_type in get_data_types() {
                let value = get_value_expr(tpe.clone());
                expect_type(&value.cast(another_type.clone()), &another_type);
            }
        }
    }

    #[test]
    fn test_not_type() {
        expect_type(&!bool_value(), &DataType::Bool);

        expect_not_resolved(&!int_value());
        expect_not_resolved(&!str_value());
    }

    #[test]
    fn test_not_trait() {
        let expr = !col_name("a");
        expect_expr(&expr, "NOT col:a")
    }

    #[test]
    fn test_negation_type() {
        for tpe in get_data_types().into_iter() {
            let value_expr = get_value_expr(tpe.clone());
            if DataType::NUMERIC_TYPES.contains(&tpe) {
                expect_type(&value_expr.negate(), &tpe);
            } else {
                expect_not_resolved(&value_expr.negate());
            }
        }
    }

    #[test]
    fn test_alias() {
        for tpe in get_data_types() {
            let value = get_value_expr(tpe.clone());
            expect_type(&value.alias("a"), &tpe);
        }
    }

    #[test]
    fn test_is_null() {
        for tpe in get_data_types() {
            let value = get_value_expr(tpe.clone());
            expect_type(
                &Expr::IsNull {
                    not: true,
                    expr: Box::new(value),
                },
                &DataType::Bool,
            );
        }
    }

    #[test]
    fn in_list() {
        expect_type(
            &Expr::InList {
                not: true,
                expr: Box::new(str_value()),
                exprs: vec![str_value(), bool_value()],
            },
            &DataType::Bool,
        );
    }

    #[test]
    fn test_array() {
        let expr = Expr::Array(vec![int_value(), int_value()]);
        expect_type(&expr, &DataType::Array(Box::new(DataType::Int32)));

        let expr = Expr::Array(vec![int_value(), null_value()]);
        expect_type(&expr, &DataType::Array(Box::new(DataType::Int32)));

        let expr = Expr::Array(vec![null_value(), int_value()]);
        expect_type(&expr, &DataType::Array(Box::new(DataType::Int32)));

        // multidimensional array
        let expr = Expr::Array(vec![Expr::Array(vec![int_value()]), Expr::Array(vec![int_value()])]);
        expect_type(&expr, &DataType::Array(Box::new(DataType::Int32)));

        let expr = Expr::Array(vec![
            Expr::Array(vec![Expr::Array(vec![int_value()])]),
            Expr::Array(vec![Expr::Array(vec![int_value()])]),
        ]);
        expect_type(&expr, &DataType::Array(Box::new(DataType::Int32)));

        // array elements must be of the same type
        let expr = Expr::Array(vec![int_value(), bool_value()]);
        expect_not_resolved(&expr);

        let expr = Expr::Array(vec![null_value(), int_value(), bool_value()]);
        expect_not_resolved(&expr);

        // FIXME: array with all elements of null type.
        let expr = Expr::Array(vec![null_value(), null_value()]);
        expect_not_resolved(&expr);

        // multidimensional array
        let expr = Expr::Array(vec![Expr::Array(vec![int_value()]), Expr::Array(vec![bool_value()])]);
        expect_not_resolved(&expr);
    }

    #[test]
    fn test_array_index() {
        let arr_expr = Expr::Array(vec![bool_value(), bool_value()]);
        let expr = Expr::ArrayIndex {
            array: Box::new(arr_expr),
            indexes: vec![int_value()],
        };
        expect_type(&expr, &DataType::Bool);

        let arr_expr = Expr::Array(vec![Expr::Array(vec![bool_value()]), Expr::Array(vec![bool_value()])]);
        let expr = Expr::ArrayIndex {
            array: Box::new(arr_expr),
            indexes: vec![int_value()],
        };
        expect_type(&expr, &DataType::Bool);

        // invalid index type
        let arr_expr = Expr::Array(vec![int_value(), int_value()]);
        let expr = Expr::ArrayIndex {
            array: Box::new(arr_expr),
            indexes: vec![bool_value()],
        };
        expect_not_resolved(&expr)
    }

    #[test]
    fn aggregate() {
        fn aggr(f: &str, args: Vec<Expr>) -> Expr {
            Expr::Aggregate {
                func: AggregateFunction::try_from(f).unwrap(),
                distinct: false,
                args,
                filter: None,
            }
        }

        fn expect_aggr_invalid_args(expr: Expr) {
            expect_not_resolved(&aggr("min", vec![expr.clone()]));
            expect_not_resolved(&aggr("max", vec![expr.clone()]));
            expect_not_resolved(&aggr("avg", vec![expr.clone()]));
            expect_not_resolved(&aggr("sum", vec![expr.clone()]));
        }

        let types = vec![(int_value(), &DataType::Int32), (fp_value(), &DataType::Float32)];
        for (val, tpe) in types {
            expect_type(&aggr("min", vec![val.clone()]), tpe);
            expect_type(&aggr("max", vec![val.clone()]), tpe);
            expect_type(&aggr("avg", vec![val.clone()]), tpe);
            expect_type(&aggr("sum", vec![val.clone()]), tpe);
        }

        expect_type(&aggr("count", vec![int_value()]), &DataType::Int32);
        expect_type(&aggr("count", vec![fp_value()]), &DataType::Int32);

        expect_aggr_invalid_args(str_value());
        expect_aggr_invalid_args(bool_value());
    }

    // binary expressions

    #[test]
    fn test_logical_exprs() {
        use BinaryOp::*;
        use DataType::*;

        let ops = vec![And, Or];

        // logical operators are supported for bools and nulls
        for op in ops {
            valid_bin_expr(Bool, op.clone(), Bool, Bool);
            valid_bin_expr(Bool, op.clone(), Null, Bool);
            valid_bin_expr(Null, op.clone(), Bool, Bool);
            valid_bin_expr(Null, op.clone(), Null, Bool);
        }
    }

    #[test]
    fn test_arithmetic() {
        use BinaryOp::*;
        use DataType::*;

        let arithmetic_ops = vec![Plus, Minus, Multiply, Divide];

        // integer numeric types <op> another numeric type | null
        for mut types in vec![Int32, Null].into_iter().combinations_with_replacement(2) {
            let lhs = types.swap_remove(0);
            let rhs = types.swap_remove(0);

            for op in arithmetic_ops.clone().into_iter().chain(std::iter::once(Modulo)) {
                if lhs != Null && rhs != Null {
                    valid_bin_expr(lhs.clone(), op.clone(), rhs.clone(), Int32);
                }
            }
        }

        // floating point types <op> another floating point type | null
        // floating point types do not support modulo operator
        for mut types in vec![Float32, Null].into_iter().combinations_with_replacement(2) {
            let lhs = types.swap_remove(0);
            let rhs = types.swap_remove(0);
            for op in arithmetic_ops.clone() {
                if lhs != Null && rhs != Null {
                    valid_bin_expr(lhs.clone(), op.clone(), rhs.clone(), Float32);
                }
            }
        }

        // integer types and floating point types can be used in arithmetic operators
        // (except for the modulo operator)
        for mut types in vec![Int32, Float32].into_iter().combinations_with_replacement(2) {
            let lhs = types.swap_remove(0);
            let rhs = types.swap_remove(0);
            // ignore int32 <op> int32
            if lhs == Int32 && rhs == Int32 {
                continue;
            }
            for op in arithmetic_ops.clone().into_iter() {
                valid_bin_expr(lhs.clone(), op.clone(), rhs.clone(), Float32);
            }

            invalid_bin_expr(lhs.clone(), Modulo, rhs.clone());
        }

        // no arithmetic operators when both operands are Null
        for op in arithmetic_ops {
            invalid_bin_expr(Null, op, Null);
        }
    }

    // comparison

    #[test]
    fn test_comparison() {
        use BinaryOp::*;
        use DataType::*;

        let comparable_types = vec![Int32, Float32, Date, Time, Interval, Timestamp(true), Timestamp(false)];
        let ops = vec![Lt, Gt, LtEq, GtEq];

        // comparable types can be compared with the same type or with null
        for tpe in comparable_types.clone() {
            for op in ops.clone() {
                valid_bin_expr(tpe.clone(), op.clone(), tpe.clone(), Bool);
                // rhs is NULL
                valid_bin_expr(tpe.clone(), op.clone(), Null, Bool);
                // lhs is NULL
                valid_bin_expr(Null, op.clone(), tpe.clone(), Bool);
                // NULL v NULL
                valid_bin_expr(Null, op.clone(), Null, Bool);
            }
        }

        // numeric types can be compared itself and with other numeric types
        let numeric_types: Vec<_> = DataType::NUMERIC_TYPES.iter().clone().collect();
        let num = DataType::NUMERIC_TYPES.len();

        for mut vals in numeric_types.into_iter().combinations_with_replacement(num) {
            let lhs = vals.swap_remove(0);
            let rhs = vals.swap_remove(0);

            for op in ops.clone() {
                valid_bin_expr(lhs.clone(), op, rhs.clone(), Bool);
            }
        }

        // not supported
        for tpe in [DataType::Tuple(vec![String]), DataType::Array(Box::new(String))] {
            for op in ops.clone() {
                invalid_bin_expr(tpe.clone(), op, tpe.clone())
            }
        }
    }

    #[test]
    fn test_date_time() {
        use BinaryOp::*;
        use DataType::*;

        valid_bin_expr(Date, Plus, Interval, Date);
        valid_bin_expr(Date, Minus, Interval, Date);
        valid_bin_expr(Interval, Plus, Date, Date);

        invalid_bin_expr(Date, Plus, Null);
        invalid_bin_expr(Date, Minus, Null);
        invalid_bin_expr(Null, Plus, Date);

        valid_bin_expr(Time, Minus, Time, Interval);
        valid_bin_expr(Time, Plus, Interval, Time);
        valid_bin_expr(Time, Minus, Interval, Interval);

        valid_bin_expr(Time, Minus, Null, Interval);
        valid_bin_expr(Null, Minus, Time, Interval);

        invalid_bin_expr(Time, Plus, Null);
        invalid_bin_expr(Null, Plus, Time);

        valid_bin_expr(Interval, Plus, Interval, Interval);
        valid_bin_expr(Interval, Minus, Interval, Interval);

        valid_bin_expr(Interval, Multiply, Int32, Interval);
        valid_bin_expr(Interval, Divide, Int32, Interval);
        valid_bin_expr(Interval, Multiply, Null, Interval);
        valid_bin_expr(Interval, Divide, Null, Interval);

        valid_bin_expr(Interval, Minus, Null, Interval);
        valid_bin_expr(Interval, Plus, Null, Interval);
    }

    #[test]
    fn test_string_ops() {
        use DataType::*;

        fn make_like_expr(expr: DataType, pattern_type: DataType, escape_char: Option<char>, like: bool) -> Expr {
            let expr = get_value_expr(expr);
            let pattern = get_value_expr(pattern_type);

            if like {
                expr.like(pattern, escape_char)
            } else {
                expr.ilike(pattern, escape_char)
            }
        }

        // let ops = vec![Like, NotLike];
        let str_tpes = vec![String];
        // [like/ilike, escape]
        let ops = vec![(false, None), (false, Some('c')), (true, None), (true, Some('c'))];

        // every type must be either string or null
        for tpe in str_tpes.clone() {
            for (like, escape_char) in ops.clone() {
                let expr = make_like_expr(String, String, escape_char, like);
                expect_type(&expr, &Bool);

                let expr = make_like_expr(String, Null, escape_char, like);
                expect_type(&expr, &Bool);

                let expr = make_like_expr(Null, String, escape_char, like);
                expect_type(&expr, &Bool);

                let expr = make_like_expr(Null, Null, escape_char, like);
                expect_type(&expr, &Bool);
            }
        }

        // no operator for non string types
        for tpe in get_data_types()
            .into_iter()
            .filter(|tpe| tpe != &Null)
            .filter(|tpe| !str_tpes.contains(tpe))
        {
            for (like, escape_char) in ops.clone() {
                let expr = make_like_expr(String, tpe.clone(), escape_char, like);
                expect_not_resolved(&expr);

                let expr = make_like_expr(tpe.clone(), String, escape_char, like);
                expect_not_resolved(&expr);

                let expr = make_like_expr(tpe.clone(), tpe.clone(), escape_char, like);
                expect_not_resolved(&expr);
            }
        }
    }

    #[test]
    fn test_concat() {
        use BinaryOp::*;
        use DataType::*;

        // every type can be concatenated with a string
        for tpe in get_data_types() {
            valid_bin_expr(String, Concat, tpe.clone(), String);
            valid_bin_expr(tpe.clone(), Concat, String, String);
        }

        // no concat operator for non-string types (except for the case when both types are null)
        for mut types in get_data_types().into_iter().filter(|tpe| tpe != &String).combinations_with_replacement(2) {
            let lhs = types.swap_remove(0);
            let rhs = types.swap_remove(0);

            if lhs == Null && rhs == Null {
                valid_bin_expr(Null, Concat, Null, String);
            } else {
                invalid_bin_expr(lhs, Concat, rhs);
            }
        }
    }

    #[test]
    fn test_equality() {
        use BinaryOp::*;
        use DataType::*;

        let ops = vec![Eq, NotEq];

        // every type can be test for equality with itself
        for tpe in get_data_types() {
            for op in ops.clone() {
                valid_bin_expr(tpe.clone(), op, tpe.clone(), Bool);
            }
        }

        // every type can be tested for equality with Null
        for tpe in get_data_types() {
            for op in ops.clone() {
                valid_bin_expr(tpe.clone(), op.clone(), Null, Bool);
                valid_bin_expr(Null, op, tpe.clone(), Bool);
            }
        }

        // Nulls can be tested for equality
        for op in ops.clone() {
            valid_bin_expr(Null, op, Null, Bool);
        }
    }

    #[test]
    fn add_trait() {
        let expr = col_name("a").add(col_name("b"));
        expect_expr(&expr, "col:a + col:b");
    }

    #[test]
    fn sub_trait() {
        let expr = col_name("a").sub(col_name("b"));
        expect_expr(&expr, "col:a - col:b");
    }

    #[test]
    fn mul_trait() {
        let expr = col_name("a").mul(col_name("b"));
        expect_expr(&expr, "col:a * col:b");
    }

    #[test]
    fn div_trait() {
        let expr = col_name("a").div(col_name("b"));
        expect_expr(&expr, "col:a / col:b");
    }

    #[test]
    fn rem_trait() {
        let expr = col_name("a").rem(col_name("b"));
        expect_expr(&expr, "col:a % col:b");
    }

    #[test]
    fn case_expr_simple_form() {
        let expr = Expr::Case {
            expr: Some(Box::new(str_value())),
            when_then_exprs: vec![(str_value(), int_value()), (str_value(), int_value())],
            else_expr: None,
        };
        expect_type(&expr, &DataType::Int32);
    }

    #[test]
    fn case_expr_simple_form_when_exprs_must_be_of_the_same_type_as_base_expr() {
        let expr = Expr::Case {
            expr: Some(Box::new(str_value())),
            when_then_exprs: vec![(str_value(), int_value()), (str_value(), int_value())],
            else_expr: None,
        };
        expect_type(&expr, &DataType::Int32);

        let expr = Expr::Case {
            expr: Some(Box::new(str_value())),
            when_then_exprs: vec![(bool_value(), int_value()), (bool_value(), int_value())],
            else_expr: None,
        };
        expect_not_resolved(&expr);
    }

    #[test]
    fn case_expr() {
        let expr = Expr::Case {
            expr: None,
            when_then_exprs: vec![(bool_value(), str_value()), (bool_value(), str_value())],
            else_expr: None,
        };
        expect_type(&expr, &DataType::String);
    }

    #[test]
    fn case_expr_when_clause_must_be_boolean_exprs() {
        let expr = Expr::Case {
            expr: None,
            when_then_exprs: vec![(bool_value(), str_value()), (bool_value(), str_value())],
            else_expr: None,
        };
        expect_type(&expr, &DataType::String);
    }

    fn expect_type(expr: &Expr, expected_type: &DataType) {
        let registry = MockColumnTypeRegistry::new();
        expect_resolved(expr, &registry, expected_type)
    }

    fn expect_resolved(expr: &Expr, metadata: &MockColumnTypeRegistry, expected_type: &DataType) {
        match resolve_expr_type(expr, metadata) {
            Ok(actual) => assert_eq!(&actual, expected_type),
            Err(err) => panic!("Failed to resolve expression type: {}", err),
        }
    }

    fn expect_not_resolved(expr: &Expr) {
        let registry = MockColumnTypeRegistry::new();
        let expr_str = format!("{}", expr);
        let result = resolve_expr_type(expr, &registry);
        assert!(result.is_err(), "Expected an error. Expr: {}. Result: {:?}", expr_str, result)
    }

    fn expect_expr(expr: &Expr, expected_str: &str) {
        assert_eq!(expected_str, format!("{}", expr))
    }

    fn get_value_expr(tpe: DataType) -> Expr {
        match tpe {
            DataType::Null => null_value(),
            DataType::Int32 => int_value(),
            DataType::Bool => bool_value(),
            DataType::Float32 => fp_value(),
            DataType::String => str_value(),
            DataType::Date => date_value(),
            DataType::Time => time_value(),
            DataType::Interval => {
                interval_value(true /*years*/)
            }
            DataType::Timestamp(tz) => timestamp_value(tz),
            DataType::Tuple(ref types) => {
                let first_field = &types[0];
                if let Expr::Scalar(value) = get_value_expr(first_field.clone()) {
                    let tuple = ScalarValue::Tuple(Some(vec![value]), Box::new(tpe.clone()));
                    Expr::Scalar(tuple)
                } else {
                    unreachable!()
                }
            }
            DataType::Array(ref element_type) => {
                let element_type = (**element_type).clone();
                if let Expr::Scalar(value) = get_value_expr(element_type.clone()) {
                    let array_data_type = DataType::Array(Box::new(element_type));
                    let array = ScalarValue::Array(Some(Array { elements: vec![value] }), Box::new(array_data_type));
                    Expr::Scalar(array)
                } else {
                    unreachable!()
                }
            }
        }
    }

    fn get_data_types() -> Vec<DataType> {
        vec![
            DataType::Null,
            DataType::Bool,
            DataType::Int32,
            DataType::Float32,
            DataType::String,
            DataType::Date,
            DataType::Time,
            DataType::Timestamp(true),
            DataType::Timestamp(false),
            DataType::Tuple(vec![DataType::String]),
            DataType::Array(Box::new(DataType::String)),
        ]
    }

    fn valid_bin_expr(lhs_type: DataType, op: BinaryOp, rhs_type: DataType, expected_type: DataType) {
        let lhs = get_value_expr(lhs_type.clone());
        let rhs = get_value_expr(rhs_type.clone());
        let bin_expr = lhs.binary_expr(op.clone(), rhs);
        let expr_type = resolve_binary_expr_type(&bin_expr, &op, lhs_type, rhs_type);

        match expr_type {
            Ok(actual_type) => assert_eq!(actual_type, expected_type, "type does not match. Expr: {}", bin_expr),
            Err(e) => panic!("Failed to resolve the type of a binary expression: {}. Error: {:?}", bin_expr, e),
        }
    }

    fn invalid_bin_expr(lhs_type: DataType, op: BinaryOp, rhs_type: DataType) {
        let lhs = get_value_expr(lhs_type.clone());
        let rhs = get_value_expr(rhs_type.clone());
        let bin_expr = lhs.binary_expr(op.clone(), rhs);
        let expr_type = resolve_binary_expr_type(&bin_expr, &op, lhs_type.clone(), rhs_type.clone());

        match expr_type {
            Ok(actual_type) => {
                panic!(
                    "Should have failed to resolve the type of a binary expression: {}. Result: {:?}",
                    bin_expr, actual_type
                );
            }
            Err(err) => {
                let err = err.to_string();
                let message = format!("No operator for {} {} {}", lhs_type, op, rhs_type);
                assert!(err.contains(&message), "Unexpected error message: {}", err);
            }
        }
    }
}
