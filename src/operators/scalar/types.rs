use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::operators::scalar::expr::{BinaryOp, Expr, NestedExpr};

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
            resolve_binary_expr_type(expr, op, &left_tpe, &right_tpe)
        }
        Expr::Cast { data_type, .. } => Ok(data_type.clone()),
        Expr::Not(expr) => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            expect_type_or_null(&tpe, &DataType::Bool, expr)?;
            Ok(DataType::Bool)
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
            let lhs_type = resolve_expr_type(expr, column_registry)?;
            let skip_num;
            let first_not_null;

            if &lhs_type == &DataType::Null {
                if let Some((i, first_type)) = first_not_null_type(exprs, column_registry)? {
                    skip_num = i + 1;
                    first_not_null = first_type;
                } else {
                    // All items are NULL, there is nothing to type check.
                    return Ok(DataType::Bool);
                }
            } else {
                // lhs is not null, we should check all elements.
                skip_num = 0;
                first_not_null = lhs_type;
            }

            for expr in exprs.iter().skip(skip_num) {
                let item_tpe = resolve_expr_type(expr, column_registry)?;
                expect_type_or_null(&item_tpe, &first_not_null, expr)?;
            }

            Ok(DataType::Bool)
        }
        Expr::IsNull { expr, .. } => {
            let _ = resolve_expr_type(expr, column_registry)?;
            Ok(DataType::Bool)
        }
        Expr::IsFalse { expr, .. } => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            expect_type_or_null(&tpe, &DataType::Bool, expr)?;
            Ok(DataType::Bool)
        }
        Expr::IsTrue { expr, .. } => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            expect_type_or_null(&tpe, &DataType::Bool, expr)?;
            Ok(DataType::Bool)
        }
        Expr::IsUnknown { expr, .. } => {
            let tpe = resolve_expr_type(expr, column_registry)?;
            expect_type_or_null(&tpe, &DataType::Bool, expr)?;
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
                return Err(type_error("Cannot determine type of empty array"));
            }

            if let Some((i, first_type)) = first_not_null_type(exprs, column_registry)? {
                for item_expr in exprs.iter().skip(i + 1) {
                    let item_type = resolve_expr_type(item_expr, column_registry)?;

                    expect_type_or_null_with_error(&item_type, &first_type, || {
                        OptimizerError::argument("Array with elements of different types")
                    })?;
                }

                Ok(DataType::array(first_type))
            } else {
                Ok(DataType::array(DataType::Null))
            }
        }
        Expr::ArrayIndex { expr, index } => {
            let arr_type = resolve_expr_type(expr, column_registry)?;
            if let DataType::Array(element_type) = arr_type {
                let index_type = resolve_expr_type(index, column_registry)?;
                if index_type != DataType::Int32 {
                    return Err(type_error(format!("Invalid array index type {}", index_type)));
                }
                Ok(element_type.as_ref().clone())
            } else {
                Err(type_error(format!("Expected array but got {}", arr_type)))
            }
        }
        Expr::ArraySlice {
            expr,
            lower_bound,
            upper_bound,
            stride,
        } => {
            let arr_type = resolve_expr_type(expr, column_registry)?;
            if let DataType::Array(element_type) = arr_type {
                if let Some(expr) = lower_bound {
                    let tpe = resolve_expr_type(expr, column_registry)?;
                    if tpe != DataType::Int32 {
                        return Err(type_error(format!("Invalid array slice index type {}", tpe)));
                    }
                }

                if let Some(expr) = upper_bound {
                    let tpe = resolve_expr_type(expr, column_registry)?;
                    if tpe != DataType::Int32 {
                        return Err(type_error(format!("Invalid array slice index type {}", tpe)));
                    }
                }

                if let Some(expr) = stride {
                    let tpe = resolve_expr_type(expr, column_registry)?;
                    if tpe != DataType::Int32 {
                        return Err(type_error(format!("Invalid array slice stride type {}", tpe)));
                    }
                }

                Ok(DataType::array(element_type.as_ref().clone()))
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
            expect_type_or_null(&expr_type, &DataType::String, expr)?;

            let pattern_type = resolve_expr_type(pattern, column_registry)?;
            expect_type_or_null(&pattern_type, &DataType::String, expr)?;

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
    lhs: &DataType,
    rhs: &DataType,
) -> Result<DataType, OptimizerError>
where
    T: NestedExpr,
{
    use BinaryOp::*;
    use DataType::*;

    fn support_comparison_ops(lhs: &DataType, rhs: &DataType) -> bool {
        match (lhs, rhs) {
            _ if lhs == &Null || rhs == &Null || lhs == rhs => true,
            _ if are_numeric_types(lhs, rhs) => true,
            (Timestamp(_), Timestamp(_)) => true,
            (Array(l), Array(r)) if l == r => true,
            (Tuple(l), Tuple(r)) if l == r => true,
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

        // Timestamp
        (Timestamp(tz), Plus | Minus, Interval) => Ok(Timestamp(*tz)),

        // Interval
        (Interval, Plus | Minus, Interval) => Ok(Interval),
        (Interval, Plus, Date) => Ok(Date),
        (Interval, Plus, Time) => Ok(Time),

        // <interval> mul/div <num>
        (Interval, Multiply | Divide, Int32) => Ok(Interval),
        (Int32, Multiply, Interval) => Ok(Interval),

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

fn expect_type_or_null<T>(actual: &DataType, expected: &DataType, expr: &Expr<T>) -> Result<(), OptimizerError>
where
    T: NestedExpr,
{
    expect_type_or_null_with_error(actual, expected, || {
        let message = format!("Expr: {}. Expected type {} but got {}", expr, expected, actual);
        type_error(message)
    })
}

fn expect_type_or_null_with_error<F>(
    actual: &DataType,
    expected: &DataType,
    error_reporter: F,
) -> Result<(), OptimizerError>
where
    F: Fn() -> OptimizerError,
{
    if actual != &DataType::Null && actual != expected {
        let error = error_reporter();
        Err(error)
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

fn first_not_null_type<T, R>(
    exprs: &[Expr<T>],
    column_registry: &R,
) -> Result<Option<(usize, DataType)>, OptimizerError>
where
    T: NestedExpr,
    R: ColumnTypeRegistry + ?Sized,
{
    for (i, expr) in exprs.iter().enumerate() {
        let expr_type = resolve_expr_type(expr, column_registry)?;
        if &expr_type != &DataType::Null {
            return Ok(Some((i, expr_type)));
        }
    }

    Ok(None)
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
        expect_not_resolved(&col, "Internal error: Expr col:a should have been replaced with Column(id)")
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
        for tpe in get_data_types().iter() {
            let value = get_value_expr(tpe);
            expect_type(&value, tpe);
        }
    }

    #[test]
    fn test_subquery_type() {
        let mut registry = MockColumnTypeRegistry::new();

        for tpe in get_data_types().iter() {
            let subquery = new_subquery_from_column(&mut registry, tpe.clone());
            let expr = Expr::SubQuery(subquery);

            expect_resolved(&expr, &registry, tpe);
        }
    }

    #[test]
    fn test_exists_subquery() {
        use DataType::*;

        let mut registry = MockColumnTypeRegistry::new();

        for tpe in get_data_types().iter() {
            let query = new_subquery_from_column(&mut registry, tpe.clone());
            let expr = Expr::Exists { not: false, query };

            expect_resolved(&expr, &registry, &Bool);
        }
    }

    #[test]
    fn test_in_subquery() {
        use DataType::*;

        let mut registry = MockColumnTypeRegistry::new();

        for tpe in get_data_types().iter() {
            let query = new_subquery_from_column(&mut registry, tpe.clone());
            let expr = Expr::InSubQuery {
                expr: Box::new(col_name("a")),
                not: false,
                query,
            };

            expect_resolved(&expr, &registry, &Bool);
        }
    }

    #[test]
    fn test_cast_type() {
        // cast any type to any other type.
        let data_types = get_data_types();

        for tpe in data_types.iter() {
            for another_type in data_types.iter() {
                let value = get_value_expr(tpe);
                expect_type(&value.cast(another_type.clone()), &another_type);
            }
        }
    }

    #[test]
    fn test_not_type() {
        use DataType::*;

        expect_type(&!bool_value(), &Bool);
        expect_type(&!null_value(), &Bool);

        for tpe in get_data_types_expect_null().iter().filter(|t| *t != &Bool) {
            let mismatch_err = mismatch_error(&Bool, tpe);
            let expr = get_value_expr(tpe);
            expect_not_resolved(&!expr, &mismatch_err)
        }
    }

    #[test]
    fn test_not_trait() {
        let expr = !col_name("a");
        expect_expr(&expr, "NOT col:a")
    }

    #[test]
    fn test_negation_type() {
        for tpe in get_data_types().iter() {
            let value_expr = get_value_expr(tpe);
            if DataType::NUMERIC_TYPES.contains(tpe) {
                expect_type(&value_expr.negate(), tpe);
            } else {
                let mismatch_err = format!("No operator for type {}", tpe);
                expect_not_resolved(&value_expr.negate(), &mismatch_err);
            }
        }
    }

    #[test]
    fn test_alias() {
        for tpe in get_data_types().iter() {
            let value = get_value_expr(tpe);
            expect_type(&value.alias("a"), tpe);
        }
    }

    #[test]
    fn test_is_null() {
        use DataType::*;

        for tpe in get_data_types().iter() {
            let value = get_value_expr(tpe);
            expect_type(&value.is_null(), &Bool);
        }
    }

    #[test]
    fn test_is_false() {
        use DataType::*;

        expect_type(&get_value_expr(&Bool).is_false(), &Bool);
        expect_type(&get_value_expr(&Null).is_false(), &Bool);

        for tpe in get_data_types_expect_null().iter().filter(|t| *t != &Bool) {
            let value = get_value_expr(tpe).is_false();
            let mismatch_err = mismatch_error(&Bool, tpe);

            expect_not_resolved(&value, &mismatch_err)
        }
    }

    #[test]
    fn test_is_true() {
        use DataType::*;

        expect_type(&get_value_expr(&Bool).is_true(), &Bool);
        expect_type(&get_value_expr(&Null).is_true(), &Bool);

        for tpe in get_data_types_expect_null().iter().filter(|t| *t != &Bool) {
            let value = get_value_expr(tpe).is_true();
            let mismatch_err = mismatch_error(&Bool, tpe);

            expect_not_resolved(&value, &mismatch_err)
        }
    }

    #[test]
    fn test_is_unknown() {
        use DataType::*;

        expect_type(&get_value_expr(&Bool).is_unknown(), &Bool);
        expect_type(&get_value_expr(&Null).is_unknown(), &Bool);

        for tpe in get_data_types_expect_null().iter().filter(|t| *t != &Bool) {
            let value = get_value_expr(tpe).is_unknown();
            let mismatch_err = mismatch_error(&Bool, tpe);

            expect_not_resolved(&value, &mismatch_err)
        }
    }

    #[test]
    fn test_in_list() {
        use DataType::*;

        let data_types_expect_null = get_data_types_expect_null();

        for tpe in data_types_expect_null.iter() {
            // Allow TYPE IN [TYPE]
            expect_type(
                &Expr::InList {
                    not: false,
                    expr: Box::new(get_value_expr(tpe)),
                    exprs: vec![get_value_expr(tpe)],
                },
                &Bool,
            );

            // Allow TYPE IN [TYPE, NULL]
            expect_type(
                &Expr::InList {
                    not: false,
                    expr: Box::new(get_value_expr(tpe)),
                    exprs: vec![get_value_expr(tpe), null_value()],
                },
                &Bool,
            );

            // Allow TYPE IN [NULL, TYPE]
            expect_type(
                &Expr::InList {
                    not: false,
                    expr: Box::new(get_value_expr(tpe)),
                    exprs: vec![null_value(), get_value_expr(tpe)],
                },
                &Bool,
            );
        }

        for tpe in data_types_expect_null.iter() {
            // Allow NULL IN [TYPE, ... ]
            expect_type(
                &Expr::InList {
                    not: false,
                    expr: Box::new(null_value()),
                    exprs: vec![get_value_expr(tpe)],
                },
                &Bool,
            );

            // Allow NULL IN [TYPE, NULL, ... ]
            expect_type(
                &Expr::InList {
                    not: false,
                    expr: Box::new(null_value()),
                    exprs: vec![get_value_expr(tpe), null_value()],
                },
                &Bool,
            );

            // Allow NULL IN [NULL, TYPE, ... ]
            expect_type(
                &Expr::InList {
                    not: false,
                    expr: Box::new(null_value()),
                    exprs: vec![null_value(), get_value_expr(tpe)],
                },
                &Bool,
            );
        }

        // Allow NULL IN [NULL, NULL, ...]
        expect_type(
            &Expr::InList {
                not: false,
                expr: Box::new(null_value()),
                exprs: vec![null_value(), null_value()],
            },
            &Bool,
        );

        for expr_tpe in data_types_expect_null.iter() {
            for item_tpe in data_types_expect_null.iter().filter(|t| *t != expr_tpe) {
                // Reject if types do not match TYPE1 IN [TYPE2, ... ]
                let mismatch_err = mismatch_error(item_tpe, expr_tpe);

                expect_not_resolved(
                    &Expr::InList {
                        not: false,
                        expr: Box::new(get_value_expr(item_tpe)),
                        exprs: vec![get_value_expr(expr_tpe)],
                    },
                    &mismatch_err,
                );

                // Reject NULL IN [TYPE1, TYPE2, ... ]
                expect_not_resolved(
                    &Expr::InList {
                        not: false,
                        expr: Box::new(null_value()),
                        exprs: vec![get_value_expr(item_tpe), get_value_expr(expr_tpe)],
                    },
                    &mismatch_err,
                );

                // Reject TYPE1 IN [TYPE2, NULL, ... ]
                expect_not_resolved(
                    &Expr::InList {
                        not: false,
                        expr: Box::new(get_value_expr(item_tpe)),
                        exprs: vec![get_value_expr(expr_tpe), null_value()],
                    },
                    &mismatch_err,
                );
            }
        }
    }

    #[test]
    fn test_array() {
        use DataType::*;

        let expr = Expr::Array(vec![int_value(), int_value()]);
        expect_type(&expr, &DataType::array(Int32));

        let expr = Expr::Array(vec![int_value(), null_value()]);
        expect_type(&expr, &DataType::array(Int32));

        let expr = Expr::Array(vec![null_value(), int_value()]);
        expect_type(&expr, &DataType::array(Int32));

        // multidimensional arrays
        // array of arrays of ints
        let expr = Expr::Array(vec![Expr::Array(vec![int_value()]), Expr::Array(vec![int_value()])]);
        expect_type(&expr, &DataType::array(DataType::array(Int32)));

        // array of arrays of arrays ints
        let expr = Expr::Array(vec![
            Expr::Array(vec![Expr::Array(vec![int_value()])]),
            Expr::Array(vec![Expr::Array(vec![int_value()])]),
        ]);
        expect_type(&expr, &DataType::array(DataType::array(DataType::array(Int32))));

        let expr = Expr::Array(vec![null_value(), null_value()]);
        expect_type(&expr, &DataType::array(Null));

        // array elements must be of the same type
        let diff_elem_type_error = "Array with elements of different types";

        let expr = Expr::Array(vec![int_value(), bool_value()]);
        expect_not_resolved(&expr, diff_elem_type_error);

        let expr = Expr::Array(vec![null_value(), int_value(), bool_value()]);
        expect_not_resolved(&expr, diff_elem_type_error);

        // multidimensional array
        let expr = Expr::Array(vec![Expr::Array(vec![int_value()]), Expr::Array(vec![bool_value()])]);
        expect_not_resolved(&expr, diff_elem_type_error);

        // empty array
        let expr = Expr::Array(vec![]);
        expect_not_resolved(&expr, "Cannot determine type of empty array");
    }

    #[test]
    fn test_array_index() {
        use DataType::*;

        let data_types = get_data_types();

        for element_type in data_types.iter() {
            let arr_expr = Expr::Array(vec![get_value_expr(&element_type)]);
            let expr = Expr::ArrayIndex {
                expr: Box::new(arr_expr),
                index: Box::new(int_value()),
            };
            expect_type(&expr, element_type);
        }

        // Reject non-int index type
        for index_type in data_types.iter().filter(|t| *t != &Int32) {
            let arr_expr = Expr::Array(vec![int_value()]);
            let expr = Expr::ArrayIndex {
                expr: Box::new(arr_expr),
                index: Box::new(get_value_expr(index_type)),
            };
            let mismatch_error = format!("Invalid array index type {}", index_type);
            expect_not_resolved(&expr, &mismatch_error);
        }
    }

    #[test]
    pub fn array_slice() {
        use DataType::*;

        let data_types = get_data_types();

        // Valid slice types
        // lower, upper, stride
        let slices = vec![
            (Some(int_value()), None, None),
            (Some(int_value()), Some(int_value()), None),
            (Some(int_value()), Some(int_value()), Some(int_value())),
        ];
        for slice in slices {
            for element_type in data_types.iter() {
                let arr_expr = Expr::Array(vec![get_value_expr(&element_type)]);
                let (lower_bound, upper_bound, stride) = slice.clone();
                let expr = Expr::ArraySlice {
                    expr: Box::new(arr_expr),
                    lower_bound: lower_bound.map(Box::new),
                    upper_bound: upper_bound.map(Box::new),
                    stride: stride.map(Box::new),
                };
                expect_type(&expr, &DataType::array(element_type.clone()));
            }
        }

        //

        for index_type in data_types.iter().filter(|t| *t != &Int32) {
            let arr_expr = Expr::Array(vec![get_value_expr(&Bool)]);
            let lower_bound = get_value_expr(&index_type);
            let upper_bound = get_value_expr(&index_type);

            // Reject non-int lower bound
            let expr = Expr::ArraySlice {
                expr: Box::new(arr_expr.clone()),
                lower_bound: Some(Box::new(lower_bound)),
                upper_bound: None,
                stride: None,
            };
            let mismatch_error = format!("Invalid array slice index type {}", index_type);
            expect_not_resolved(&expr, &mismatch_error);

            // Reject non int-upper bound
            let expr = Expr::ArraySlice {
                expr: Box::new(arr_expr.clone()),
                lower_bound: Some(Box::new(int_value())),
                upper_bound: Some(Box::new(upper_bound)),
                stride: None,
            };
            let mismatch_error = format!("Invalid array slice index type {}", index_type);
            expect_not_resolved(&expr, &mismatch_error);

            // Reject non-int stride
            let stride = get_value_expr(&index_type);
            let expr = Expr::ArraySlice {
                expr: Box::new(arr_expr),
                lower_bound: Some(Box::new(int_value())),
                upper_bound: Some(Box::new(int_value())),
                stride: Some(Box::new(stride)),
            };
            let mismatch_error = format!("Invalid array slice stride type {}", index_type);
            expect_not_resolved(&expr, &mismatch_error);
        }
    }

    #[test]
    fn aggregate() {
        use DataType::*;

        fn aggr(f: &str, args: Vec<Expr>) -> Expr {
            Expr::Aggregate {
                func: AggregateFunction::try_from(f).unwrap(),
                distinct: false,
                args,
                filter: None,
            }
        }

        let data_types = get_data_types();

        for tpe in data_types.iter() {
            let val = get_value_expr(tpe);
            expect_type(&aggr("min", vec![val.clone()]), tpe);
            expect_type(&aggr("max", vec![val.clone()]), tpe);
            expect_type(&aggr("count", vec![val]), &Int32);
        }

        let allow_sum = vec![Int32, Float32];

        for tpe in allow_sum.iter() {
            let val = get_value_expr(tpe);
            expect_type(&aggr("sum", vec![val]), &tpe);
        }

        for tpe in data_types.iter().filter(|t| !allow_sum.contains(t)) {
            let val = get_value_expr(tpe);
            let mismatch_err = format!("No function sum({})", tpe);
            expect_not_resolved(&aggr("sum", vec![val]), &mismatch_err)
        }

        let allow_avg: Vec<_> = data_types.iter().filter(|t| matches!(t, Int32 | Float32)).collect();

        for tpe in allow_avg.iter() {
            let val = get_value_expr(tpe);
            expect_type(&aggr("avg", vec![val]), tpe);
        }

        for tpe in data_types.iter().filter(|t| !allow_avg.contains(t)) {
            let val = get_value_expr(tpe);
            let mismatch_err = format!("No function avg({})", tpe);
            expect_not_resolved(&aggr("avg", vec![val]), &mismatch_err)
        }
    }

    // binary expressions

    #[test]
    fn test_logical_exprs() {
        use BinaryOp::*;
        use DataType::*;

        let ops = vec![And, Or];

        // logical operators are supported for bools and nulls
        for op in ops {
            valid_bin_expr(&Bool, op.clone(), &Bool, &Bool);
            valid_bin_expr(&Bool, op.clone(), &Null, &Bool);
            valid_bin_expr(&Null, op.clone(), &Bool, &Bool);
            valid_bin_expr(&Null, op.clone(), &Null, &Bool);
        }
    }

    #[test]
    fn test_arithmetic() {
        use BinaryOp::*;
        use DataType::*;

        let arithmetic_ops = vec![Plus, Minus, Multiply, Divide];

        // integer numeric types <op> another numeric type | null
        for mut types in vec![Int32, Null].iter().combinations_with_replacement(2) {
            let lhs = types.swap_remove(0);
            let rhs = types.swap_remove(0);

            for op in arithmetic_ops.iter().chain(std::iter::once(&Modulo)) {
                if *lhs != Null && *rhs != Null {
                    valid_bin_expr(&lhs, op.clone(), &rhs, &Int32);
                }
            }
        }

        // floating point types <op> another floating point type | null
        // floating point types do not support modulo operator
        for mut types in vec![Float32, Null].iter().combinations_with_replacement(2) {
            let lhs = types.swap_remove(0);
            let rhs = types.swap_remove(0);
            for op in arithmetic_ops.iter() {
                if *lhs != Null && *rhs != Null {
                    valid_bin_expr(&lhs, op.clone(), &rhs, &Float32);
                }
            }
        }

        // integer types and floating point types can be used in arithmetic operators
        // (except for the modulo operator)
        for mut types in DataType::NUMERIC_TYPES.iter().combinations_with_replacement(2) {
            let lhs = types.swap_remove(0);
            let rhs = types.swap_remove(0);
            // ignore int32 <op> int32
            if *lhs == Int32 && *rhs == Int32 {
                continue;
            }
            for op in arithmetic_ops.iter() {
                valid_bin_expr(&lhs, op.clone(), &rhs, &Float32);
            }

            invalid_bin_expr(&lhs, Modulo, &rhs);
        }

        // no arithmetic operators when both operands are Null
        for op in arithmetic_ops.iter() {
            invalid_bin_expr(&Null, op.clone(), &Null);
        }
    }

    // comparison

    #[test]
    fn test_comparison() {
        use DataType::*;

        let simple_types: Vec<_> = get_data_types_expect_null()
            .into_iter()
            .filter(|t| !matches!(t, Tuple(_) | Array(_)))
            .collect();

        // comparable types can be compared with the same type or with null
        for tpe in simple_types.iter() {
            for op in get_cmp_ops() {
                valid_bin_expr(tpe, op.clone(), tpe, &Bool);
                // rhs is NULL
                valid_bin_expr(tpe, op.clone(), &Null, &Bool);
                // lhs is NULL
                valid_bin_expr(&Null, op.clone(), tpe, &Bool);
                // NULL v NULL
                valid_bin_expr(&Null, op.clone(), &Null, &Bool);
            }
        }

        // numeric types can be compared itself and with other numeric types
        let numeric_types: Vec<_> = DataType::NUMERIC_TYPES.iter().clone().collect();
        let num = DataType::NUMERIC_TYPES.len();

        for mut vals in numeric_types.iter().combinations_with_replacement(num) {
            let lhs = vals.swap_remove(0);
            let rhs = vals.swap_remove(0);

            for op in get_cmp_ops() {
                valid_bin_expr(lhs, op, rhs, &Bool);
            }
        }
    }

    #[test]
    fn test_array_comparison() {
        use DataType::*;

        // Allow array[type1] <cmp> array[type2] if type1 = type2
        let data_types = get_data_types();

        for element_type in data_types.iter() {
            let arr_type = DataType::array(element_type.clone());

            for op in get_cmp_ops() {
                valid_bin_expr(&arr_type, op, &arr_type, &Bool)
            }
        }

        // Reject array[type] <cmp> type and vice versa
        for element_type in get_data_types_expect_null().iter().filter(|t| !matches!(t, Array(_))) {
            let arr_type = DataType::array(element_type.clone());

            for op in get_cmp_ops() {
                invalid_bin_expr(&arr_type, op.clone(), element_type);
                invalid_bin_expr(element_type, op.clone(), &arr_type);
            }
        }

        // Reject array[type1] <cmp> array[type2] if type != type2
        for element_type in data_types.iter() {
            for another_type in data_types.iter().filter(|t| *t != element_type) {
                let arr_type1 = DataType::array(element_type.clone());
                let arr_type2 = DataType::array(another_type.clone());

                for op in get_cmp_ops() {
                    invalid_bin_expr(&arr_type1, op, &arr_type2)
                }
            }
        }

        // Reject multidimensional arrays with different dimensions
        for element_type in data_types.iter() {
            let arr_type1 = DataType::array(element_type.clone());
            let arr_type2 = DataType::array(DataType::array(element_type.clone()));
            let arr_type3 = DataType::array(DataType::array(DataType::array(element_type.clone())));

            for op in get_cmp_ops() {
                invalid_bin_expr(&arr_type1, op.clone(), &arr_type2);
                invalid_bin_expr(&arr_type1, op.clone(), &arr_type3);
                invalid_bin_expr(&arr_type2, op.clone(), &arr_type3);
            }
        }
    }

    #[test]
    fn test_tuple_comparison() {
        use DataType::*;

        let data_types = get_data_types();
        let data_types_expect_null = get_data_types_expect_null();

        // Allow tuple[type1] <cmp> tuple[type2] if type1 = type2
        for element_type in data_types.iter() {
            let tuple_type = Tuple(vec![element_type.clone()]);

            for op in get_cmp_ops() {
                valid_bin_expr(&tuple_type, op, &tuple_type, &Bool)
            }
        }

        // Reject tuple[type] <cmp> type and vice versa
        for element_type in data_types_expect_null.iter().filter(|t| !matches!(t, Tuple(_))) {
            let tuple_type = Tuple(vec![element_type.clone()]);

            for op in get_cmp_ops() {
                invalid_bin_expr(&tuple_type, op.clone(), element_type);
                invalid_bin_expr(element_type, op.clone(), &tuple_type);
            }
        }

        // Reject tuple[type1] <cmp> tuple[type2] if type1 != type2
        for element_type in data_types.iter() {
            for another_type in data_types.iter().filter(|t| t != &element_type) {
                let tuple_type1 = Tuple(vec![element_type.clone()]);
                let tuple_type2 = Tuple(vec![another_type.clone()]);

                for op in get_cmp_ops() {
                    invalid_bin_expr(&tuple_type1, op, &tuple_type2)
                }
            }
        }

        // Reject tuple1 <cmp> tuple2 if tuple types have different number of fields
        for element_type in data_types.iter() {
            let tuple_type1 = Tuple(vec![element_type.clone()]);
            let tuple_type2 = Tuple(vec![element_type.clone(), element_type.clone()]);

            for op in get_cmp_ops() {
                invalid_bin_expr(&tuple_type1, op, &tuple_type2)
            }
        }
    }

    #[test]
    fn test_date_time() {
        use BinaryOp::*;
        use DataType::*;

        valid_bin_expr(&Date, Plus, &Interval, &Date);
        valid_bin_expr(&Date, Minus, &Interval, &Date);
        valid_bin_expr(&Interval, Plus, &Date, &Date);

        valid_bin_expr(&Time, Minus, &Time, &Interval);
        valid_bin_expr(&Time, Plus, &Interval, &Time);
        valid_bin_expr(&Time, Minus, &Interval, &Interval);

        valid_bin_expr(&Interval, Plus, &Interval, &Interval);
        valid_bin_expr(&Interval, Minus, &Interval, &Interval);

        valid_bin_expr(&Interval, Multiply, &Int32, &Interval);
        valid_bin_expr(&Interval, Divide, &Int32, &Interval);

        invalid_bin_expr(&Date, Plus, &Null);
        invalid_bin_expr(&Null, Plus, &Date);

        invalid_bin_expr(&Date, Minus, &Null);
        invalid_bin_expr(&Date, Plus, &Null);

        invalid_bin_expr(&Time, Minus, &Null);
        invalid_bin_expr(&Null, Minus, &Time);

        invalid_bin_expr(&Time, Plus, &Null);
        invalid_bin_expr(&Null, Plus, &Time);

        invalid_bin_expr(&Interval, Minus, &Null);
        invalid_bin_expr(&Null, Minus, &Interval);

        invalid_bin_expr(&Interval, Plus, &Null);
        invalid_bin_expr(&Null, Plus, &Interval);

        invalid_bin_expr(&Interval, Multiply, &Null);
        invalid_bin_expr(&Null, Multiply, &Interval);

        invalid_bin_expr(&Interval, Divide, &Null);
        invalid_bin_expr(&Null, Divide, &Interval);
    }

    #[test]
    fn test_string_ops() {
        use DataType::*;

        fn make_like_expr(expr: &DataType, pattern_type: &DataType, escape_char: Option<char>, like: bool) -> Expr {
            let expr = get_value_expr(expr);
            let pattern = get_value_expr(pattern_type);

            if like {
                expr.like(pattern, escape_char)
            } else {
                expr.ilike(pattern, escape_char)
            }
        }
        let data_types_expect_null = get_data_types_expect_null();

        let str_types = vec![String];
        // [like/ilike, escape]
        let ops = vec![(false, None), (false, Some('c')), (true, None), (true, Some('c'))];

        // every type must be either string or null
        for str_type in str_types.iter() {
            for (like, escape_char) in ops.clone() {
                let expr = make_like_expr(str_type, str_type, escape_char, like);
                expect_type(&expr, &Bool);

                let expr = make_like_expr(str_type, &Null, escape_char, like);
                expect_type(&expr, &Bool);

                let expr = make_like_expr(&Null, &str_type, escape_char, like);
                expect_type(&expr, &Bool);

                let expr = make_like_expr(&Null, &Null, escape_char, like);
                expect_type(&expr, &Bool);
            }
        }

        // no operator for non string types
        for str_type in str_types.iter() {
            for tpe in data_types_expect_null.iter().filter(|t| !str_types.contains(t)) {
                for (like, escape_char) in ops.clone() {
                    let expr = make_like_expr(str_type, tpe, escape_char, like);
                    let mismatch_err = mismatch_error(str_type, &tpe);
                    expect_not_resolved(&expr, &mismatch_err);

                    let expr = make_like_expr(tpe, str_type, escape_char, like);
                    expect_not_resolved(&expr, &mismatch_err);

                    let expr = make_like_expr(tpe, tpe, escape_char, like);
                    expect_not_resolved(&expr, &mismatch_err);
                }
            }
        }
    }

    #[test]
    fn test_concat() {
        use BinaryOp::*;
        use DataType::*;

        // every type can be concatenated with a string
        for tpe in get_data_types().iter() {
            valid_bin_expr(&String, Concat, tpe, &String);
            valid_bin_expr(tpe, Concat, &String, &String);
        }

        valid_bin_expr(&Null, Concat, &Null, &String);

        // no concat operator for non-string types (except for the case when both types are null)
        for mut types in get_data_types_expect_null()
            .iter()
            .filter(|t| *t != &String)
            .combinations_with_replacement(2)
        {
            let lhs = types.swap_remove(0);
            let rhs = types.swap_remove(0);

            invalid_bin_expr(&lhs, Concat, &rhs);
        }
    }

    #[test]
    fn test_equality() {
        use BinaryOp::*;
        use DataType::*;

        let ops = vec![Eq, NotEq];

        // every type can be test for equality with itself
        for tpe in get_data_types().iter() {
            for op in ops.clone() {
                valid_bin_expr(tpe, op, tpe, &Bool);
            }
        }

        // every type can be tested for equality with Null
        for tpe in get_data_types().iter() {
            for op in ops.clone() {
                valid_bin_expr(tpe, op.clone(), &Null, &Bool);
                valid_bin_expr(&Null, op, tpe, &Bool);
            }
        }

        // Nulls can be tested for equality
        for op in ops.clone() {
            valid_bin_expr(&Null, op, &Null, &Bool);
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
        for tpe1 in get_data_types().iter() {
            let val1 = get_value_expr(tpe1);

            for tpe2 in get_data_types().iter() {
                let val2 = get_value_expr(tpe2);

                let expr = Expr::Case {
                    expr: Some(Box::new(val1.clone())),
                    when_then_exprs: vec![(val1.clone(), val2.clone()), (val1.clone(), val2.clone())],
                    else_expr: None,
                };
                expect_type(&expr, &tpe2);
            }
        }
    }

    #[test]
    fn case_expr_simple_form_when_exprs_must_be_of_the_same_type_as_base_expr() {
        for tpe1 in get_data_types_expect_null().iter() {
            let val1 = get_value_expr(tpe1);

            for tpe2 in get_data_types_expect_null().iter().filter(|t| *t != tpe1) {
                let val2 = get_value_expr(tpe2);

                let when_expr1 = val1.clone();
                let when_expr2 = val2.clone();
                let then_expr = int_value();

                // Type mismatch in WHEN clause of CASE <expr> WHEN <val> THEN <result>
                let expr = Expr::Case {
                    expr: Some(Box::new(val1.clone())),
                    when_then_exprs: vec![(when_expr1, then_expr.clone()), (when_expr2, then_expr)],
                    else_expr: None,
                };

                let mismatch_err = mismatch_error(&tpe1, &tpe2);
                expect_not_resolved(&expr, &mismatch_err);

                // Type mismatch in THEN clause of CASE <expr> WHEN <val> THEN <result>
                let when_expr = val1.clone();
                let then_expr1 = val1.clone();
                let then_expr2 = val2.clone();

                let expr = Expr::Case {
                    expr: Some(Box::new(val1.clone())),
                    when_then_exprs: vec![(when_expr.clone(), then_expr1), (when_expr, then_expr2)],
                    else_expr: None,
                };

                let mismatch_err = format!(
                    "Unexpected result type in a THEN clause of a CASE expression. Expected {} but got {}",
                    tpe1, tpe2
                );
                expect_not_resolved(&expr, &mismatch_err);
            }
        }
    }

    #[test]
    fn case_expr() {
        use DataType::*;

        for tpe in get_data_types().iter() {
            let then_expr = get_value_expr(tpe);

            let expr = Expr::Case {
                expr: None,
                when_then_exprs: vec![(bool_value(), then_expr.clone()), (bool_value(), then_expr.clone())],
                else_expr: None,
            };
            expect_type(&expr, &tpe);
        }

        // expr in CASE WHEN <expr> THEN ... should have <boolean> type.
        for tpe in get_data_types_expect_null().iter().filter(|t| *t != &Bool) {
            let when_expr = get_value_expr(tpe);
            let then_expr = str_value();

            let expr = Expr::Case {
                expr: None,
                when_then_exprs: vec![(when_expr.clone(), then_expr.clone()), (when_expr.clone(), then_expr)],
                else_expr: None,
            };

            let mismatch_err = mismatch_error(&Bool, &tpe);
            expect_not_resolved(&expr, &mismatch_err);
        }
    }

    fn expect_type(expr: &Expr, expected_type: &DataType) {
        let registry = MockColumnTypeRegistry::new();
        expect_resolved(expr, &registry, expected_type)
    }

    fn expect_resolved(expr: &Expr, metadata: &MockColumnTypeRegistry, expected_type: &DataType) {
        match resolve_expr_type(expr, metadata) {
            Ok(actual) => assert_eq!(&actual, expected_type, "Expected type does not match"),
            Err(err) => panic!("Failed to resolve expression type: {}", err),
        }
    }

    fn mismatch_error(expected: &DataType, actual: &DataType) -> String {
        format!("Expected type {} but got {}", expected, actual)
    }

    fn expect_not_resolved(expr: &Expr, error_message: &str) {
        let registry = MockColumnTypeRegistry::new();

        let result = resolve_expr_type(expr, &registry);
        assert!(result.is_err(), "Expected a type error. Expr: {}. Result: {:?}", expr, result);

        let error = result.unwrap_err();
        let actual_error = format!("{}", error);
        assert!(
            actual_error.contains(error_message),
            "Type error message mismatch. Expected a message that includes:\n{}\nBut got:\n{}",
            error_message,
            actual_error
        )
    }

    fn expect_expr(expr: &Expr, expected_str: &str) {
        assert_eq!(expected_str, format!("{}", expr))
    }

    fn get_value_expr(tpe: &DataType) -> Expr {
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
            DataType::Timestamp(tz) => timestamp_value(*tz),
            DataType::Tuple(types) => {
                let values: Vec<_> = types
                    .iter()
                    .cloned()
                    .map(|t| match get_value_expr(&t) {
                        Expr::Scalar(value) => value,
                        _ => unreachable!(),
                    })
                    .collect();
                let tuple = ScalarValue::Tuple(Some(values), Box::new(tpe.clone()));
                Expr::Scalar(tuple)
            }
            DataType::Array(element_type) => {
                if let Expr::Scalar(value) = get_value_expr(element_type) {
                    let array = ScalarValue::Array(Some(Array { elements: vec![value] }), Box::new(tpe.clone()));
                    Expr::Scalar(array)
                } else {
                    unreachable!()
                }
            }
        }
    }

    fn get_data_types() -> Vec<DataType> {
        use DataType::*;

        let mut types = vec![Null, Bool, Int32, Float32, String, Date, Time, Timestamp(true), Timestamp(false)];

        types.extend(types.clone().into_iter().map(|t| Array(Box::new(t))));
        types.extend(types.clone().into_iter().map(|t| Tuple(vec![t])));

        types
    }

    fn get_data_types_expect_null() -> Vec<DataType> {
        get_data_types().into_iter().filter(|t| t != &DataType::Null).collect()
    }

    fn get_cmp_ops() -> Vec<BinaryOp> {
        vec![BinaryOp::Eq, BinaryOp::NotEq, BinaryOp::Lt, BinaryOp::Gt, BinaryOp::LtEq, BinaryOp::GtEq]
    }

    fn valid_bin_expr(lhs_type: &DataType, op: BinaryOp, rhs_type: &DataType, expected_type: &DataType) {
        let lhs = get_value_expr(lhs_type);
        let rhs = get_value_expr(rhs_type);
        let bin_expr = lhs.binary_expr(op.clone(), rhs);
        let expr_type = resolve_binary_expr_type(&bin_expr, &op, &lhs_type, &rhs_type);

        match expr_type {
            Ok(actual_type) => assert_eq!(&actual_type, expected_type, "type does not match. Expr: {}", bin_expr),
            Err(e) => panic!("Failed to resolve the type of a binary expression: {}. Error: {:?}", bin_expr, e),
        }
    }

    fn invalid_bin_expr(lhs_type: &DataType, op: BinaryOp, rhs_type: &DataType) {
        let lhs = get_value_expr(lhs_type);
        let rhs = get_value_expr(rhs_type);
        let bin_expr = lhs.binary_expr(op.clone(), rhs);
        let expr_type = resolve_binary_expr_type(&bin_expr, &op, &lhs_type, &rhs_type);

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
