use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::meta::ColumnId;
use crate::not_supported;
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
            expect_type_or_null(&tpe, &DataType::Int32, expr)?;
            Ok(tpe)
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
            if let Some(expr_type) = expr_types.first() {
                let same_type = expr_types[1..].iter().all(|tpe| expr_type == tpe);
                if same_type {
                    Ok(expr_type.clone())
                } else {
                    Err(type_error("Array with elements of different type"))
                }
            } else {
                not_supported!("Empty array expression")
            }
        }
        Expr::ScalarFunction { func, args } => {
            let arg_types: Result<Vec<DataType>, _> =
                args.iter().map(|arg| resolve_expr_type(arg, column_registry)).collect();

            let arg_types = arg_types?;
            func.get_return_type(&arg_types)
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
        match (lhs, rhs) {
            (String, String) => true,
            (String, Null) => true,
            (Null, String) => true,
            _ => false,
        }
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

        // <interval> mul/div <numb>
        (Interval, Multiply | Divide, Int32) => Ok(Interval),
        (Interval, Multiply | Divide, Null) => Ok(Interval),
        (Int32, Multiply, Interval) => Ok(Interval),
        (Null, Multiply, Interval) => Ok(Interval),

        // String ops
        (lhs, Like, rhs) if support_string_ops(lhs, rhs) => Ok(String),
        (lhs, NotLike, rhs) if support_string_ops(lhs, rhs) => Ok(String),

        // String Concat
        (String, Concat, _) => Ok(String),
        (_, Concat, String) => Ok(String),

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
    use crate::operators::scalar::types::{resolve_expr_type, ColumnTypeRegistry};
    use crate::operators::scalar::value::ScalarValue;
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

    fn str_value() -> Expr {
        Expr::Scalar(ScalarValue::String(Some("s".to_string())))
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
    fn column_name_type_should_not_be_resolved() {
        let col = Expr::ColumnName("a".into());
        expect_not_resolved(&col)
    }

    #[test]
    fn column_id_type() {
        fn expect_column_type(col_type: DataType) {
            let mut registry = MockColumnTypeRegistry::new();
            let id = registry.add_column(col_type.clone());
            let expr = Expr::Column(id);

            expect_resolved(&expr, &registry, &col_type);
        }

        expect_column_type(DataType::Bool);
        expect_column_type(DataType::Int32);
        expect_column_type(DataType::String);
    }

    #[test]
    fn scalar_type() {
        expect_type(&bool_value(), &DataType::Bool);
        expect_type(&int_value(), &DataType::Int32);
        expect_type(&str_value(), &DataType::String);
    }

    #[test]
    fn subquery_type() {
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
    fn exists_subquery() {
        let mut registry = MockColumnTypeRegistry::new();
        let query = new_subquery_from_column(&mut registry, DataType::Int32);
        let expr = Expr::Exists { not: false, query };

        expect_resolved(&expr, &registry, &DataType::Bool);
    }

    #[test]
    fn in_subquery() {
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
    fn cast_type() {
        expect_type(&int_value().cast(DataType::Bool), &DataType::Bool);
        expect_type(&int_value().cast(DataType::Int32), &DataType::Int32);
        expect_type(&int_value().cast(DataType::String), &DataType::String);
        expect_type(&int_value().cast(DataType::Float32), &DataType::Float32);
    }

    #[test]
    fn not_type() {
        expect_type(&!bool_value(), &DataType::Bool);

        expect_not_resolved(&!int_value());
        expect_not_resolved(&!str_value());
    }

    #[test]
    fn not_trait() {
        let expr = !col_name("a");
        expect_expr(&expr, "NOT col:a")
    }

    #[test]
    fn negation_type() {
        expect_type(&int_value().negate(), &DataType::Int32);

        expect_not_resolved(&bool_value().negate());
        expect_not_resolved(&str_value().negate());
    }

    #[test]
    fn alias() {
        expect_type(&bool_value().alias("a"), &DataType::Bool);
        expect_type(&int_value().alias("a"), &DataType::Int32);
        expect_type(&str_value().alias("a"), &DataType::String);
    }

    #[test]
    fn is_null() {
        expect_type(
            &Expr::IsNull {
                not: true,
                expr: Box::new(str_value()),
            },
            &DataType::Bool,
        );
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

        expect_type(&aggr("min", vec![int_value()]), &DataType::Int32);
        expect_type(&aggr("max", vec![int_value()]), &DataType::Int32);
        expect_type(&aggr("avg", vec![int_value()]), &DataType::Int32);
        expect_type(&aggr("sum", vec![int_value()]), &DataType::Int32);
        expect_type(&aggr("count", vec![int_value()]), &DataType::Int32);

        expect_aggr_invalid_args(str_value());
        expect_aggr_invalid_args(bool_value());
    }

    // binary expressions

    #[test]
    fn and_type() {
        expect_logical_expr_type(BinaryOp::And);
    }

    #[test]
    fn or_type() {
        expect_logical_expr_type(BinaryOp::Or);
    }

    #[test]
    fn eq_type() {
        expect_equality_expr_type(BinaryOp::Eq);
    }

    #[test]
    fn ne_type() {
        expect_equality_expr_type(BinaryOp::NotEq);
    }

    // arithmetic

    #[test]
    fn lt_type() {
        expect_arithmetic_cmp_expr_type(BinaryOp::Lt);
    }

    #[test]
    fn lte_type() {
        expect_arithmetic_cmp_expr_type(BinaryOp::LtEq);
    }

    #[test]
    fn gt_type() {
        expect_arithmetic_cmp_expr_type(BinaryOp::Gt);
    }

    #[test]
    fn gte_type() {
        expect_arithmetic_cmp_expr_type(BinaryOp::GtEq);
    }

    #[test]
    fn add_type() {
        expect_arithmetic_expr_type(BinaryOp::Plus);
    }

    #[test]
    fn add_trait() {
        let expr = col_name("a").add(col_name("b"));
        expect_expr(&expr, "col:a + col:b");
    }

    #[test]
    fn sub_type() {
        expect_arithmetic_expr_type(BinaryOp::Minus);
    }

    #[test]
    fn sub_trait() {
        let expr = col_name("a").sub(col_name("b"));
        expect_expr(&expr, "col:a - col:b");
    }

    #[test]
    fn mul_type() {
        expect_arithmetic_expr_type(BinaryOp::Multiply);
    }

    #[test]
    fn mul_trait() {
        let expr = col_name("a").mul(col_name("b"));
        expect_expr(&expr, "col:a * col:b");
    }

    #[test]
    fn div_type() {
        expect_arithmetic_expr_type(BinaryOp::Divide);
    }

    #[test]
    fn div_trait() {
        let expr = col_name("a").div(col_name("b"));
        expect_expr(&expr, "col:a / col:b");
    }

    #[test]
    fn mod_type() {
        expect_arithmetic_expr_type(BinaryOp::Modulo);
    }

    #[test]
    fn rem_trait() {
        let expr = col_name("a").rem(col_name("b"));
        expect_expr(&expr, "col:a % col:b");
    }

    #[test]
    fn string_concat() {
        expect_type(&str_value().binary_expr(BinaryOp::Concat, str_value()), &DataType::String);
        expect_type(&str_value().binary_expr(BinaryOp::Concat, int_value()), &DataType::String);
        expect_type(&str_value().binary_expr(BinaryOp::Concat, bool_value()), &DataType::String);
        expect_type(&int_value().binary_expr(BinaryOp::Concat, str_value()), &DataType::String);
        expect_type(&bool_value().binary_expr(BinaryOp::Concat, str_value()), &DataType::String);

        expect_not_resolved(&int_value().binary_expr(BinaryOp::Concat, bool_value()));
    }

    #[test]
    fn string_like() {
        expect_string_expr_type(BinaryOp::Like);
    }

    #[test]
    fn string_not_like() {
        expect_string_expr_type(BinaryOp::NotLike);
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

    // =, !=
    fn expect_equality_expr_type(op: BinaryOp) {
        expect_type(&bool_value().binary_expr(op.clone(), bool_value()), &DataType::Bool);
        expect_type(&int_value().binary_expr(op.clone(), int_value()), &DataType::Bool);
        expect_type(&str_value().binary_expr(op.clone(), str_value()), &DataType::Bool);

        expect_type(&null_value().binary_expr(op.clone(), null_value()), &DataType::Bool);
        expect_type(&null_value().binary_expr(op.clone(), bool_value()), &DataType::Bool);
        expect_type(&bool_value().binary_expr(op.clone(), null_value()), &DataType::Bool);

        expect_not_resolved(&int_value().binary_expr(op, bool_value()));
    }

    // AND, OR
    fn expect_logical_expr_type(op: BinaryOp) {
        expect_type(&bool_value().binary_expr(op.clone(), bool_value()), &DataType::Bool);

        expect_type(&null_value().binary_expr(op.clone(), null_value()), &DataType::Bool);
        expect_type(&null_value().binary_expr(op.clone(), bool_value()), &DataType::Bool);
        expect_type(&bool_value().binary_expr(op.clone(), null_value()), &DataType::Bool);

        expect_not_resolved(&bool_value().binary_expr(op.clone(), int_value()));
        expect_not_resolved(&int_value().binary_expr(op.clone(), bool_value()));

        expect_not_resolved(&int_value().binary_expr(op.clone(), int_value()));
        expect_not_resolved(&str_value().binary_expr(op, str_value()));
    }

    // Comparison operators: >, <=, etc.
    fn expect_arithmetic_cmp_expr_type(op: BinaryOp) {
        expect_type(&int_value().binary_expr(op.clone(), int_value()), &DataType::Bool);

        expect_type(&null_value().binary_expr(op.clone(), null_value()), &DataType::Bool);
        expect_type(&null_value().binary_expr(op.clone(), int_value()), &DataType::Bool);
        expect_type(&int_value().binary_expr(op.clone(), null_value()), &DataType::Bool);

        expect_not_resolved(&bool_value().binary_expr(op.clone(), bool_value()));
        expect_not_resolved(&bool_value().binary_expr(op.clone(), int_value()));
        expect_not_resolved(&int_value().binary_expr(op.clone(), bool_value()));

        expect_not_resolved(&str_value().binary_expr(op.clone(), str_value()));
        expect_not_resolved(&str_value().binary_expr(op.clone(), int_value()));
        expect_not_resolved(&int_value().binary_expr(op, str_value()));
    }

    // Arithmetic operators
    fn expect_arithmetic_expr_type(op: BinaryOp) {
        expect_type(&int_value().binary_expr(op.clone(), int_value()), &DataType::Int32);

        expect_not_resolved(&null_value().binary_expr(op.clone(), null_value()));
        expect_type(&null_value().binary_expr(op.clone(), int_value()), &DataType::Int32);
        expect_type(&int_value().binary_expr(op.clone(), null_value()), &DataType::Int32);

        expect_not_resolved(&bool_value().binary_expr(op.clone(), bool_value()));
        expect_not_resolved(&bool_value().binary_expr(op.clone(), int_value()));
        expect_not_resolved(&int_value().binary_expr(op.clone(), bool_value()));

        expect_not_resolved(&str_value().binary_expr(op.clone(), str_value()));
        expect_not_resolved(&str_value().binary_expr(op.clone(), int_value()));
        expect_not_resolved(&int_value().binary_expr(op, str_value()));
    }

    fn expect_string_expr_type(op: BinaryOp) {
        expect_type(&str_value().binary_expr(op.clone(), str_value()), &DataType::String);

        expect_not_resolved(&str_value().binary_expr(op.clone(), int_value()));
        expect_not_resolved(&int_value().binary_expr(op.clone(), str_value()));
        expect_not_resolved(&bool_value().binary_expr(op.clone(), int_value()));
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
}
