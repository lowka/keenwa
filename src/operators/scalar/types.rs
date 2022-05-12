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
            Err(OptimizerError::Internal(message))
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
            if let Some(expr) = base_expr {
                let tpe = resolve_expr_type(expr, column_registry)?;
                expect_type_or_null(&tpe, &DataType::Bool, expr)?
            }

            let mut expr_result_type: Option<DataType> = None;

            for (when, then) in when_then_exprs {
                let condition_type = resolve_expr_type(when, column_registry)?;
                expect_type_or_null(&condition_type, &DataType::Bool, when)?;
                let result_type = resolve_expr_type(then, column_registry)?;

                match &mut expr_result_type {
                    Some(tpe) if tpe != &result_type => {
                        return Err(OptimizerError::Internal(format!(
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
                        return Err(OptimizerError::Internal(format!(
                            "Unexpected result type in an ELSE clause of a CASE expression. Expected {} but got {}",
                            tpe, result_type
                        )));
                    }
                    _ => expr_result_type = Some(result_type),
                }
            }

            Ok(expr_result_type.expect("Invalid case expression"))
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
                    Err(OptimizerError::Internal(format!("Array with elements of different type")))
                }
            } else {
                Err(OptimizerError::Unsupported(format!("Empty array expression")))
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
            Err(OptimizerError::Internal(message))
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
    match op {
        BinaryOp::And | BinaryOp::Or => {
            expect_type_or_null(&lhs, &DataType::Bool, expr)?;
            expect_type_or_null(&rhs, &DataType::Bool, expr)?;
            Ok(DataType::Bool)
        }
        BinaryOp::Eq | BinaryOp::NotEq => {
            if lhs != rhs && lhs != DataType::Null && rhs != DataType::Null {
                let msg = format!("Expr: {} Types does not match. lhs: {}. rhs: {}", expr, lhs, rhs);
                Err(OptimizerError::Internal(msg))
            } else {
                Ok(DataType::Bool)
            }
        }
        // Comparison
        BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
            expect_type_or_null(&lhs, &DataType::Int32, expr)?;
            expect_type_or_null(&rhs, &DataType::Int32, expr)?;
            Ok(DataType::Bool)
        }
        // Arithmetic
        BinaryOp::Plus | BinaryOp::Minus | BinaryOp::Multiply | BinaryOp::Divide | BinaryOp::Modulo => {
            expect_type_or_null(&lhs, &DataType::Int32, expr)?;
            expect_type_or_null(&rhs, &DataType::Int32, expr)?;
            if lhs != DataType::Null {
                Ok(lhs)
            } else if rhs != DataType::Null {
                Ok(rhs)
            } else {
                let message = format!("Expr: {} No operator for NULL {} NULL", expr, op);
                Err(OptimizerError::Internal(message))
            }
        }
        // String operators
        BinaryOp::Concat | BinaryOp::Like | BinaryOp::NotLike => {
            expect_type_or_null(&lhs, &DataType::String, expr)?;
            expect_type_or_null(&rhs, &DataType::String, expr)?;
            Ok(DataType::String)
        }
    }
}

fn expect_type_or_null<T>(tpe: &DataType, expected: &DataType, expr: &Expr<T>) -> Result<(), OptimizerError>
where
    T: NestedExpr,
{
    if tpe != &DataType::Null && tpe != expected {
        let msg = format!("Expr: {}. Expected type {} but got {}", expr, expected, tpe);
        Err(OptimizerError::Internal(msg))
    } else {
        Ok(())
    }
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
        Expr::Scalar(ScalarValue::Bool(true))
    }

    fn int_value() -> Expr {
        Expr::Scalar(ScalarValue::Int32(1))
    }

    fn str_value() -> Expr {
        Expr::Scalar(ScalarValue::String("s".into()))
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
        expect_string_expr_type(BinaryOp::Concat);
    }

    #[test]
    fn string_like() {
        expect_string_expr_type(BinaryOp::Like);
    }

    #[test]
    fn string_not_like() {
        expect_string_expr_type(BinaryOp::NotLike);
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
