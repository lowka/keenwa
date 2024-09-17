#[cfg(test)]
mod testing;

use crate::catalog::CatalogRef;
use crate::datatypes::DataType;
use crate::error::macros::{not_implemented, not_supported};
use crate::error::OptimizerError;
use crate::meta::MutableMetadata;
use crate::operators::builder::{
    MemoizeOperators, OperatorBuilder, OperatorBuilderConfig, OrderingOption, OrderingOptions, TableAlias,
};
use crate::operators::properties::LogicalPropertiesBuilder;
use crate::operators::relational::join::JoinType;
use crate::operators::relational::RelNode;
use crate::operators::scalar::aggregates::WindowOrAggregateFunction;
use crate::operators::scalar::expr::{BinaryOp, ExprVisitor};
use crate::operators::scalar::funcs::ScalarFunction;
use crate::operators::scalar::value::{parse_date, parse_time, parse_timestamp, Interval, ScalarValue};
use crate::operators::scalar::{scalar, ScalarExpr};
use crate::operators::{ExprMemo, Operator, OperatorMemoBuilder, OperatorMetadata};
use crate::statistics::StatisticsBuilder;
use itertools::Itertools;
use sqlparser::ast::{
    BinaryOperator, CastKind, Cte, DataType as SqlDataType, DateTimeField, Distinct, DuplicateTreatment, Expr,
    Function, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident, Interval as SqlInterval, Join,
    JoinConstraint, JoinOperator, ObjectName, OrderBy, OrderByExpr, Query, Select, SelectItem, SetExpr, SetOperator,
    SetQuantifier, Statement, Subscript, TableAlias as SqlTableAlias, TableFactor, TableWithJoins, TimezoneInfo,
    UnaryOperator, Value, Values, WildcardAdditionalOptions, WindowSpec, WindowType, With,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserError};
use std::convert::{Infallible, TryFrom};
use std::ops::Not;
use std::rc::Rc;
use std::str::FromStr;

/// OperatorFromSqlBuilder creates an [operator tree](Operator) given SQL query.
pub struct OperatorFromSqlBuilder<T> {
    catalog: CatalogRef,
    statistics_builder: T,
    operator_builder_config: OperatorBuilderConfig,
}

impl<T> OperatorFromSqlBuilder<T>
where
    T: StatisticsBuilder + 'static,
{
    /// Creates a new instance of a [OperatorFromSqlBuilder].
    pub fn new(catalog: CatalogRef, statistics_builder: T) -> Self {
        OperatorFromSqlBuilder {
            catalog,
            statistics_builder,
            operator_builder_config: OperatorBuilderConfig::default(),
        }
    }

    /// Specifies [OperatorBuilderConfig] to use.
    pub fn operator_builder_config(&mut self, value: OperatorBuilderConfig) {
        self.operator_builder_config = value;
    }

    /// Parses the given SQL string and returns an [operator tree](Operator) that can then passed to the [query optimizer].
    ///
    /// [query optimizer]: crate::optimizer::Optimizer
    pub fn build(self, query: &str) -> Result<(Operator, ExprMemo, OperatorMetadata), OptimizerError> {
        let metadata = Rc::new(MutableMetadata::new());
        let properties_builder = LogicalPropertiesBuilder::new(self.statistics_builder);
        let memo = OperatorMemoBuilder::new(metadata.clone()).build_with_properties(properties_builder);
        let mut memoization = MemoizeOperators::new(memo);

        let operator_builder = OperatorBuilder::with_config(
            self.operator_builder_config,
            memoization.take_callback(),
            self.catalog.clone(),
            metadata.clone(),
        );

        let operator = build_from_sql(operator_builder, query)?;
        let memo = memoization.into_memo();

        Ok((operator, memo, metadata))
    }
}

fn build_from_sql(builder: OperatorBuilder, query_str: &str) -> Result<Operator, OptimizerError> {
    //TODO: Support multiple dialects.
    let dialect = GenericDialect {};

    let ast: Vec<Statement> = Parser::parse_sql(&dialect, query_str)?;
    not_implemented!(ast.len() > 1, "Multiple statements in a single query");

    let mut builder = builder;
    for stmt in ast {
        builder = build_statement(builder, stmt)?;
        // Add presentation property to the top level operator.
        builder = builder.with_presentation()?;
    }

    builder.build()
}

fn build_statement(builder: OperatorBuilder, stmt: Statement) -> Result<OperatorBuilder, OptimizerError> {
    match stmt {
        Statement::Analyze { .. } => not_implemented!("ANALYZE"),
        Statement::Truncate { .. } => not_implemented!("TRUNCATE"),
        Statement::Msck { .. } => not_supported!("MSCK"),
        Statement::Query(query) => build_query(builder, *query),
        Statement::Insert { .. } => not_implemented!("INSERT"),
        Statement::Directory { .. } => not_supported!("DIRECTORY"),
        Statement::Copy { .. } => not_implemented!("COPY"),
        Statement::Update { .. } => not_implemented!("UPDATE"),
        Statement::Delete { .. } => not_implemented!("DELETE"),
        Statement::CreateView { .. } => not_implemented!("CREATE VIEW"),
        Statement::CreateTable { .. } => not_implemented!("CREATE TABLE"),
        Statement::CreateVirtualTable { .. } => not_supported!("CREATE VIRTUAL TABLE"),
        Statement::CreateIndex { .. } => not_implemented!("CREATE INDEX"),
        Statement::AlterTable { .. } => not_implemented!("ALTER TABLE"),
        Statement::Drop { .. } => not_implemented!("DROP"),
        Statement::SetVariable { .. } => not_supported!("SET VARIABLE"),
        Statement::SetRole { .. } => not_supported!("SET ROLE"),
        Statement::ShowVariable { .. } => not_supported!("SHOW VARIABLE"),
        Statement::ShowCreate { .. } => not_supported!("SHOW CREATE"),
        Statement::ShowColumns { .. } => not_supported!("SHOW COLUMNS"),
        Statement::StartTransaction { .. } => not_implemented!("START TRANSACTION"),
        Statement::SetTransaction { .. } => not_supported!("SET TRANSACTION"),
        Statement::Comment { .. } => Ok(builder),
        Statement::Commit { .. } => not_implemented!("COMMIT"),
        Statement::Rollback { .. } => not_implemented!("ROLLBACK"),
        Statement::CreateSchema { .. } => not_implemented!("CREATE SCHEMA"),
        Statement::CreateDatabase { .. } => not_implemented!("CREATE DATABASE"),
        Statement::Assert { .. } => not_implemented!("ASSERT"),
        Statement::Grant { .. } => not_implemented!("GRANT"),
        Statement::Revoke { .. } => not_implemented!("REVOKE"),
        Statement::Deallocate { .. } => not_supported!("DEALLOCATE"),
        Statement::Execute { .. } => not_supported!("EXECUTE"),
        Statement::Prepare { .. } => not_supported!("PREPARE"),
        Statement::ExplainTable { .. } => not_supported!("EXPLAIN TABLE"),
        Statement::Kill { .. } => not_supported!("KILL"),
        Statement::Explain { .. } => not_implemented!("EXPLAIN"),
        Statement::Savepoint { .. } => not_implemented!("SAVEPOINT"),
        Statement::Merge { .. } => not_implemented!("MERGE"),
        _ => not_supported!("Not supported statement"),
    }
}

fn build_query(builder: OperatorBuilder, query: Query) -> Result<OperatorBuilder, OptimizerError> {
    let builder = if let Some(with) = query.with {
        build_cte(builder, with)?
    } else {
        builder
    };

    let builder = build_set_expr(builder, *query.body)?;
    let builder = if let Some(order_by) = query.order_by {
        build_order_by(builder, order_by)?
    } else {
        builder
    };

    let builder = if let Some(offset) = query.offset {
        let rows = build_limit_offset_argument("OFFSET", offset.value, builder.clone())?;
        builder.offset(rows)?
    } else {
        builder
    };

    let builder = if let Some(limit) = query.limit {
        let rows = build_limit_offset_argument("LIMIT", limit, builder.clone())?;
        builder.limit(rows)?
    } else {
        builder
    };

    not_implemented!(query.fetch.is_some(), "FETCH");

    Ok(builder)
}

fn build_limit_offset_argument(operator: &str, expr: Expr, builder: OperatorBuilder) -> Result<usize, OptimizerError> {
    let expr = build_scalar_expr(expr, builder)?;
    if let ScalarExpr::Scalar(ScalarValue::Int32(Some(value))) = &expr {
        if *value < 0 {
            Err(OptimizerError::argument(format!("{}: number of rows must be non negative: {}", operator, value)))
        } else {
            Ok(*value as usize)
        }
    } else {
        Err(OptimizerError::argument(format!("{}: invalid value expression: {}", operator, expr)))
    }
}

fn build_set_expr(builder: OperatorBuilder, set_expr: SetExpr) -> Result<OperatorBuilder, OptimizerError> {
    match set_expr {
        SetExpr::Select(select) => build_select(builder, *select),
        SetExpr::Query(query) => build_query(builder, *query),
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } => build_set_operation(builder, op, set_quantifier, *left, *right),
        SetExpr::Values(values) => build_values(builder, values),
        SetExpr::Insert(_) => not_implemented!("INSERT"),
        SetExpr::Update(_) => not_implemented!("UPDATE"),
        SetExpr::Table(_) => not_implemented!("TABLE"),
    }
}

fn build_cte(builder: OperatorBuilder, with: With) -> Result<OperatorBuilder, OptimizerError> {
    not_implemented!(with.recursive, "Recursive CTE");

    let mut cte_builder = builder.sub_query_builder();

    for cte_table in with.cte_tables {
        let cte_table: Cte = cte_table;
        let table_expr = build_query(cte_builder.clone(), *cte_table.query)?;
        let table_alias = TableAlias {
            name: cte_table.alias.name.to_string(),
            columns: cte_table.alias.columns.into_iter().map(|c| c.to_string()).collect(),
        };
        cte_builder = cte_builder.with_table_expr(table_alias, table_expr)?;
    }

    Ok(cte_builder)
}

fn build_select(builder: OperatorBuilder, select: Select) -> Result<OperatorBuilder, OptimizerError> {
    not_supported!(!select.cluster_by.is_empty(), "SELECT ... CLUSTER BY");
    not_supported!(!select.sort_by.is_empty(), "SELECT ... SORT BY");
    not_supported!(select.top.is_some(), "SELECT TOP");
    not_supported!(!select.distribute_by.is_empty(), "SELECT ... DISTRIBUTE BY");

    not_implemented!(!select.lateral_views.is_empty(), "Lateral views IN SELECT");

    let mut projection = vec![];
    // We should be able to reference tables and columns accessible
    // to the topmost operator in the operator tree.
    let builder = if select.from.is_empty() {
        builder.empty(true)?
    } else {
        let mut builder = builder;
        for (i, from) in select.from.into_iter().enumerate() {
            // Disallow reference to columns from the left side of a join in projections
            if i == 0 {
                builder = build_from(builder, from)?;
            } else {
                let right = build_from(builder.clone(), from)?;
                builder = builder.join_on(right, JoinType::Cross, scalar(true))?;
            }
        }
        builder
    };
    let mut is_aggregate = false;

    let (select_distinct, distinct_on) = match select.distinct {
        None => (false, None),
        Some(Distinct::Distinct) => (true, None),
        Some(Distinct::On(mut exprs)) => {
            not_implemented!(exprs.len() > 1, "SELECT DISTINCT <multiple columns>");

            let expr = exprs.remove(0);
            let distinct_expr = build_scalar_expr(expr, builder.clone())?;

            (true, Some(distinct_expr))
        }
    };

    for item in select.projection.into_iter() {
        let subquery_builder = builder.clone();
        let mut expr = match item {
            SelectItem::UnnamedExpr(expr) => build_scalar_expr(expr, subquery_builder)?,
            SelectItem::ExprWithAlias { expr, alias } => {
                let expr = build_scalar_expr(expr, subquery_builder)?;
                expr.alias(alias.value.as_str())
            }
            SelectItem::QualifiedWildcard(object_name, opts) => {
                not_implemented!(object_name.0.len() > 1, "Projection: object names other than <name>.*");
                let name = object_name.to_string();

                validate_wildcard_options(&opts)?;

                ScalarExpr::Wildcard(Some(name))
            }
            SelectItem::Wildcard(opts) => {
                validate_wildcard_options(&opts)?;

                ScalarExpr::Wildcard(None)
            }
        };
        #[derive(Default)]
        struct IsAggregate {
            is_aggr: bool,
        }
        impl ExprVisitor<RelNode> for IsAggregate {
            type Error = Infallible;

            fn pre_visit(&mut self, expr: &ScalarExpr) -> Result<bool, Self::Error> {
                if matches!(expr, ScalarExpr::Aggregate { .. }) {
                    self.is_aggr = true;
                }

                Ok(!self.is_aggr)
            }

            fn post_visit(&mut self, _expr: &ScalarExpr) -> Result<(), Self::Error> {
                Ok(())
            }
        }
        if !is_aggregate {
            let mut visitor = IsAggregate::default();
            // IsAggregate never returns an error.
            expr.accept(&mut visitor).unwrap();
            if visitor.is_aggr {
                is_aggregate = true;
            }
        }
        //
        // DISTINCT ON (expr) workaround:
        //
        // SELECT DISTINCT ON (a1 + a2) a1, a2, a3 FROM table_a is parsed as
        // Select(distinct=true)
        //   Projection exprs = [ Alias (Function {name=ON, args=a1+a2}, a1), a2, a3 ]
        //     From table_a
        //
        // if select_distinct && i == 0 {
        //     // build_scalar_expr converts Function {name=ON, args=[arg]} expr into arg expr.
        //     if let ScalarExpr::Alias(expr, col) = &mut expr {
        //         let expr = expr.as_ref();
        //         distinct_on = Some(expr.clone());
        //         projection.push(ScalarExpr::ColumnName(col.clone()));
        //         continue;
        //     }
        // }

        projection.push(expr);
    }

    if !is_aggregate {
        is_aggregate = match &select.group_by {
            GroupByExpr::All(_) => not_supported!("GROUP BY ALL"),
            GroupByExpr::Expressions(_, with_modifiers) if !with_modifiers.is_empty() => {
                not_supported!("GROUP BY expressions with modifiers")
            }
            GroupByExpr::Expressions(exprs, _) => !exprs.is_empty(),
        }
    }

    let builder = if let Some(expr) = select.selection {
        let expr = build_scalar_expr(expr, builder.clone())?;
        builder.select(Some(expr))?
    } else {
        builder
    };

    let operator = if is_aggregate {
        let scalar_expr_builder = builder.clone();
        let mut builder = builder;
        let mut aggregate_builder = builder.aggregate_builder();

        for item in projection {
            aggregate_builder = aggregate_builder.add_expr(item)?;
        }

        match select.group_by {
            GroupByExpr::All(_) => not_supported!("GROUP BY ALL"),
            GroupByExpr::Expressions(_, with_modifiers) if !with_modifiers.is_empty() => {
                not_supported!("GROUP BY expressions with modifiers")
            }
            GroupByExpr::Expressions(exprs, _) => {
                for group_by_expr in exprs {
                    let group_by_expr: Expr = group_by_expr;
                    let expr = build_scalar_expr(group_by_expr, scalar_expr_builder.clone())?;
                    aggregate_builder = aggregate_builder.group_by_expr(expr)?;
                }
            }
        }

        if let Some(having_expr) = select.having {
            let having_expr: Expr = having_expr;
            let expr = build_scalar_expr(having_expr, scalar_expr_builder)?;
            aggregate_builder = aggregate_builder.having(expr)?;
        }

        aggregate_builder.build()?
    } else {
        if select.having.is_some() {
            return Err(OptimizerError::argument("HAVING clause is not allowed in non-aggregate queries".to_string()));
        }

        builder.project(projection)?
    };
    if select_distinct {
        operator.distinct(distinct_on)
    } else {
        Ok(operator)
    }
}

fn validate_wildcard_options(opts: &WildcardAdditionalOptions) -> Result<(), OptimizerError> {
    let WildcardAdditionalOptions {
        opt_ilike,
        opt_exclude,
        opt_except,
        opt_rename,
        opt_replace,
    } = opts;

    not_supported!(opt_ilike.is_some(), "Wildcard: ILIKE is not supported.");
    not_supported!(opt_exclude.is_some(), "Wildcard: EXCLUDE is not supported.");
    not_supported!(opt_replace.is_some(), "Wildcard: REPLACE is not supported.");
    not_supported!(opt_rename.is_some(), "Wildcard: RENAME is not supported.");
    not_supported!(opt_except.is_some(), "Wildcard: EXCEPT is not supported.");

    Ok(())
}

fn build_set_operation(
    builder: OperatorBuilder,
    op: sqlparser::ast::SetOperator,
    set_qualifier: SetQuantifier,
    left: SetExpr,
    right: SetExpr,
) -> Result<OperatorBuilder, OptimizerError> {
    let left = build_set_expr(builder.clone(), left)?;
    let right = build_set_expr(builder, right)?;
    let all = match set_qualifier {
        SetQuantifier::All => true,
        SetQuantifier::Distinct => not_supported!("Set qualifier: DISTINCT"),
        SetQuantifier::ByName => not_supported!("Set qualifier: BY NAME"),
        SetQuantifier::AllByName => not_supported!("Set qualifier: ALL BY NAME"),
        SetQuantifier::DistinctByName => not_supported!("Set qualifier: DISTINCT BY NAME"),
        SetQuantifier::None => false,
    };

    match op {
        SetOperator::Union if all => left.union_all(right),
        SetOperator::Union => left.union(right),
        SetOperator::Except if all => left.except_all(right),
        SetOperator::Except => left.except(right),
        SetOperator::Intersect if all => left.intersect_all(right),
        SetOperator::Intersect => left.intersect(right),
    }
}

fn build_values(builder: OperatorBuilder, values: Values) -> Result<OperatorBuilder, OptimizerError> {
    let value_lists: Vec<Vec<Expr>> = values.rows;
    let mut value_list_result = Vec::with_capacity(value_lists.len());

    for value_list in value_lists {
        let value_list: Result<Vec<ScalarExpr>, OptimizerError> =
            value_list.into_iter().map(|expr| build_scalar_expr(expr, builder.clone())).collect();
        value_list_result.push(value_list?);
    }

    builder.values(value_list_result)
}

fn build_from(builder: OperatorBuilder, from: TableWithJoins) -> Result<OperatorBuilder, OptimizerError> {
    let join_builder = builder.new_relation_builder();
    let mut builder = build_relation(join_builder.clone(), from.relation)?;

    for join in from.joins {
        let join: Join = join;
        let (join_type, constraint) = match join.join_operator {
            JoinOperator::Inner(constraint) => (JoinType::Inner, Some(constraint)),
            JoinOperator::LeftOuter(constraint) => (JoinType::Left, Some(constraint)),
            JoinOperator::RightOuter(constraint) => (JoinType::Right, Some(constraint)),
            JoinOperator::FullOuter(constraint) => (JoinType::Full, Some(constraint)),
            JoinOperator::CrossJoin => (JoinType::Cross, None),
            JoinOperator::CrossApply => not_supported!("CROSS APPLY JOIN"),
            JoinOperator::OuterApply => not_supported!("OUTER APPLY JOIN"),
            JoinOperator::LeftSemi(constraint) => (JoinType::LeftSemi, Some(constraint)),
            JoinOperator::RightSemi(constraint) => (JoinType::RightSemi, Some(constraint)),
            JoinOperator::LeftAnti(_) => not_supported!("LEFT ANTI"),
            JoinOperator::RightAnti(_) => not_supported!("RIGHT ANTI"),
            JoinOperator::AsOf { .. } => not_supported!("AS OF"),
        };
        let right = build_relation(join_builder.clone(), join.relation)?;
        match constraint {
            Some(JoinConstraint::On(expr)) => {
                let expr = build_scalar_expr(expr, builder.clone())?;
                builder = builder.join_on(right, join_type, expr)?;
            }
            Some(JoinConstraint::Using(idents)) => {
                let idents: Vec<Ident> = idents;
                let columns = idents.into_iter().map(|i| i.to_string()).map(|i| (i.clone(), i)).collect();
                builder = builder.join_using(right, join_type, columns)?;
            }
            Some(JoinConstraint::Natural) => {
                builder = builder.natural_join(right, join_type)?;
            }
            None | Some(JoinConstraint::None) => {
                // TODO: move to OperatorBuilder?
                not_supported!(
                    join_type != JoinType::Cross,
                    "JOIN without condition is allowed only for CROSS join types"
                );

                builder = builder.join_on(right, join_type, scalar(true))?;
            }
        }
    }

    Ok(builder)
}

fn build_relation(builder: OperatorBuilder, relation: TableFactor) -> Result<OperatorBuilder, OptimizerError> {
    // Make outer columns, table etc. accessible.
    let builder = match relation {
        TableFactor::Table {
            name,
            alias,
            args,
            with_hints,
            version,
            with_ordinality,
            partitions,
        } => {
            not_implemented!(args.is_some(), "FROM table. Table valued-function arguments");
            not_implemented!(!with_hints.is_empty(), "FROM table WITH hints");
            not_implemented!(version.is_some(), "FROM table WITH version");
            not_implemented!(with_ordinality, "FROM table WITH ordinality");
            not_implemented!(!partitions.is_empty(), "FROM table WITH partitions");

            let alias = build_table_alias(alias)?;
            let builder = builder.from(name.to_string().as_str())?;

            if let Some(alias) = alias {
                builder.with_alias(alias)?
            } else {
                builder
            }
        }
        TableFactor::Derived {
            lateral,
            subquery,
            alias,
        } => {
            not_implemented!(lateral, "FROM lateral derived table");

            let alias = build_table_alias(alias)?;
            let builder = build_query(builder, *subquery)?;

            if let Some(alias) = alias {
                builder.with_alias(alias)?
            } else {
                builder
            }
        }
        TableFactor::TableFunction { .. } => not_supported!("FROM TABLE function"),
        TableFactor::NestedJoin {
            table_with_joins,
            alias,
        } => {
            let table: TableWithJoins = *table_with_joins;

            not_supported!(alias.is_some(), "NESTED JOIN with alias");

            build_from(builder, table)?
        }
        TableFactor::UNNEST { .. } => not_supported!("FROM UNNEST"),
        TableFactor::JsonTable { .. } => not_supported!("FROM JSON_TABLE"),
        TableFactor::Pivot { .. } => not_supported!("FROM table PIVOT(..,)"),
        TableFactor::Unpivot { .. } => not_supported!("FROM table UNPIVOT(...)"),
        TableFactor::Function { .. } => not_supported!("FROM function(...)"),
        TableFactor::MatchRecognize { .. } => not_supported!("FROM .. MATCH_RECOGNIZE"),
    };
    Ok(builder)
}

fn build_table_alias(table_alias: Option<SqlTableAlias>) -> Result<Option<TableAlias>, OptimizerError> {
    let alias = if let Some(alias) = table_alias {
        let columns: Vec<Ident> = alias.columns;
        let name = alias.name.to_string();
        let columns = columns.into_iter().map(|c| c.to_string()).collect();

        Some(TableAlias { name, columns })
    } else {
        None
    };
    Ok(alias)
}

fn build_order_by(builder: OperatorBuilder, order_by: OrderBy) -> Result<OperatorBuilder, OptimizerError> {
    not_supported!(order_by.interpolate.is_some(), "ORDER BY ... INTERPOLATE. Clickhouse specific");

    let order_by_exprs = order_by.exprs;
    if order_by_exprs.is_empty() {
        Ok(builder)
    } else {
        let order_options: Result<Vec<OrderingOption>, OptimizerError> = order_by_exprs
            .into_iter()
            .map(|order_by_expr| {
                not_supported!(order_by_expr.nulls_first.is_some(), "ORDER BY: NULLS FIRST | LAST option");

                let expr = build_scalar_expr(order_by_expr.expr, builder.clone())?;
                let descending = order_by_expr.asc.map(|v| !v).unwrap_or_default();

                Ok(OrderingOption::new(expr, descending))
            })
            .collect();
        let options = OrderingOptions::new(order_options?);
        Ok(builder.order_by(options)?)
    }
}

fn build_scalar_expr(expr: Expr, builder: OperatorBuilder) -> Result<ScalarExpr, OptimizerError> {
    let expr = match expr {
        Expr::Identifier(ident) => ScalarExpr::ColumnName(ident.value),
        Expr::CompoundIdentifier(indents) if indents.len() > 2 => not_implemented!("compound identifier expression"),
        Expr::CompoundIdentifier(indents) => ScalarExpr::ColumnName(indents.into_iter().join(".")),
        Expr::JsonAccess { .. } => not_implemented!("JSON access expression"),
        Expr::CompositeAccess { .. } => not_implemented!("Composite access expression"),
        Expr::IsNull(expr) => {
            let expr = build_scalar_expr(*expr, builder)?;
            ScalarExpr::IsNull {
                not: false,
                expr: Box::new(expr),
            }
        }
        Expr::IsNotNull(expr) => {
            let expr = build_scalar_expr(*expr, builder)?;
            ScalarExpr::IsNull {
                not: true,
                expr: Box::new(expr),
            }
        }
        Expr::IsDistinctFrom(_, _) => not_supported!("IS DISTINCT FROM expression"),
        Expr::IsNotDistinctFrom(_, _) => not_supported!("IS NOT DISTINCT FROM expression"),
        Expr::InList { expr, list, negated } => {
            let expr = build_scalar_expr(*expr, builder.clone())?;
            let exprs: Result<Vec<ScalarExpr>, OptimizerError> =
                list.into_iter().map(|expr| build_scalar_expr(expr, builder.clone())).collect();
            ScalarExpr::InList {
                not: negated,
                expr: Box::new(expr),
                exprs: exprs?,
            }
        }
        Expr::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            let expr = build_scalar_expr(*expr, builder.clone())?;
            let builder = build_query(builder.sub_query_builder(), *subquery)?;
            let subquery = builder.build()?;
            ScalarExpr::InSubQuery {
                not: negated,
                expr: Box::new(expr),
                query: RelNode::try_from(subquery)?,
            }
        }
        Expr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let expr = build_scalar_expr(*expr, builder.clone())?;
            let low = build_scalar_expr(*low, builder.clone())?;
            let high = build_scalar_expr(*high, builder)?;
            ScalarExpr::Between {
                not: negated,
                expr: Box::new(expr),
                low: Box::new(low),
                high: Box::new(high),
            }
        }
        Expr::BinaryOp { op, left, right } => build_binary_expr(op, *left, *right, builder)?,
        Expr::AnyOp { .. } => not_implemented!("ANY expression"),
        Expr::AllOp { .. } => not_implemented!("ALL expression"),
        Expr::UnaryOp { op, expr } => build_unary_expr(op, *expr, builder)?,
        Expr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => {
            match kind {
                CastKind::Cast => {}
                CastKind::TryCast => not_implemented!("TRY CAST / SAFE CAST expression"),
                CastKind::SafeCast => not_implemented!("SAFE CAST expression"),
                CastKind::DoubleColon => {}
            }
            let expr = build_scalar_expr(*expr, builder)?;
            let data_type = convert_data_type(data_type)?;

            not_implemented!(format.is_some(), "CAST ... AS type FORMAT format_string_expression");

            ScalarExpr::Cast {
                expr: Box::new(expr),
                data_type,
            }
        }
        Expr::Extract { .. } => not_supported!("EXTRACT expression"),
        Expr::Position { .. } => not_supported!("POSITION expression"),
        Expr::Substring { .. } => not_implemented!("SUBSTRING expression"),
        Expr::Trim { .. } => not_implemented!("TRIM expression"),
        Expr::Collate { .. } => not_supported!("COLLATE expression"),
        Expr::Nested(expr) => build_scalar_expr(*expr, builder)?,
        Expr::Value(value) => build_value_expr(value)?,
        Expr::TypedString { data_type, value } => match data_type {
            SqlDataType::Date => ScalarExpr::Scalar(parse_date(value.as_str())?),
            SqlDataType::Timestamp(precision, tz) => {
                not_implemented!(precision.is_some(), "Timestamp: precision");
                not_implemented!(tz != TimezoneInfo::None, "Timestamp: timezone");
                ScalarExpr::Scalar(parse_timestamp(value.as_str())?)
            }
            SqlDataType::Time(precision, tz) => {
                not_implemented!(precision.is_some(), "Time: precision");
                not_implemented!(tz != TimezoneInfo::None, "Time: timezone");
                ScalarExpr::Scalar(parse_time(value.as_str())?)
            }
            _ => {
                let message = format!("Typed string expression for data type: {}", data_type);
                not_supported!(message)
            }
        },
        Expr::MapAccess { .. } => not_supported!("Map access expression"),
        Expr::Function(f) => build_function_expr(f, builder)?,
        Expr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            let base_expr = match operand {
                Some(expr) => Some(build_scalar_expr(*expr, builder.clone())?),
                None => None,
            };
            let when_then_exprs: Result<Vec<(ScalarExpr, ScalarExpr)>, OptimizerError> = conditions
                .into_iter()
                .zip(results)
                .map(|(when, then)| {
                    let when = build_scalar_expr(when, builder.clone())?;
                    let then = build_scalar_expr(then, builder.clone())?;
                    Ok((when, then))
                })
                .collect();
            let else_expr = match else_result {
                Some(expr) => Some(build_scalar_expr(*expr, builder)?),
                None => None,
            };
            ScalarExpr::Case {
                expr: base_expr.map(Box::new),
                when_then_exprs: when_then_exprs?,
                else_expr: else_expr.map(Box::new),
            }
        }
        Expr::Exists { subquery, negated } => {
            let builder = build_query(builder.sub_query_builder(), *subquery)?;
            let subquery = builder.build()?;
            ScalarExpr::Exists {
                not: negated,
                query: RelNode::try_from(subquery)?,
            }
        }
        Expr::Subquery(query) => {
            let builder = build_query(builder.sub_query_builder(), *query)?;
            // Add presentation to a subquery to preserve the names of its columns.
            let builder = builder.with_presentation()?;
            let subquery = builder.build()?;
            ScalarExpr::SubQuery(RelNode::try_from(subquery)?)
        }
        Expr::GroupingSets(_) => not_supported!("GROUPING SETS expression"),
        Expr::Cube(_) => not_supported!("CUBE expression"),
        Expr::Rollup(_) => not_supported!("ROLLUP expression"),
        Expr::Tuple(values) => {
            let values: Result<Vec<ScalarExpr>, OptimizerError> =
                values.into_iter().map(|expr| build_scalar_expr(expr, builder.clone())).collect();
            ScalarExpr::Tuple(values?)
        }
        Expr::InUnnest { .. } => not_supported!("InUnnest expression"),
        Expr::Subscript { expr, subscript } => {
            let expr = build_scalar_expr(*expr, builder.clone())?;
            match *subscript {
                Subscript::Index { index } => {
                    let index = build_scalar_expr(index, builder.clone())?;
                    ScalarExpr::ArrayIndex {
                        expr: Box::new(expr),
                        index: Box::new(index),
                    }
                }
                Subscript::Slice {
                    lower_bound,
                    upper_bound,
                    stride,
                } => {
                    let lower_bound = match lower_bound {
                        Some(e) => Some(build_scalar_expr(e, builder.clone())?),
                        None => None,
                    };
                    let upper_bound = match upper_bound {
                        Some(e) => Some(build_scalar_expr(e, builder.clone())?),
                        None => None,
                    };
                    let stride = match stride {
                        Some(e) => Some(build_scalar_expr(e, builder.clone())?),
                        None => None,
                    };
                    ScalarExpr::ArraySlice {
                        expr: Box::new(expr),
                        lower_bound: lower_bound.map(Box::new),
                        upper_bound: upper_bound.map(Box::new),
                        stride: stride.map(Box::new),
                    }
                }
            }
        }
        Expr::Array(array) => {
            let values: Result<Vec<ScalarExpr>, OptimizerError> =
                array.elem.into_iter().map(|expr| build_scalar_expr(expr, builder.clone())).collect();
            ScalarExpr::Array(values?)
        }
        Expr::Interval(SqlInterval {
            value,
            leading_field,
            leading_precision,
            last_field,
            fractional_seconds_precision,
        }) => {
            let value = build_scalar_expr(*value, builder.clone())?;

            if let Some(leading_field) = leading_field {
                if leading_precision.is_some() {
                    not_supported!("Interval: Leading field precision is not supported")
                }
                if fractional_seconds_precision.is_some() {
                    not_supported!("Interval: Fractional second precision is not supported")
                }

                match value {
                    ScalarExpr::Scalar(ScalarValue::String(Some(str))) => {
                        build_interval_literal(str.as_str(), leading_field, last_field)?
                    }
                    other => {
                        let msg = format!("Interval value: {}", other);
                        not_implemented!(msg)
                    }
                }
            } else {
                not_supported!("Interval: literals without leading field")
            }
        }
        Expr::Like {
            negated,
            expr,
            pattern,
            escape_char,
        } => {
            let expr = Box::new(build_scalar_expr(*expr, builder.clone())?);
            let pattern = Box::new(build_scalar_expr(*pattern, builder.clone())?);
            ScalarExpr::Like {
                not: negated,
                expr,
                pattern,
                escape_char,
                case_insensitive: false,
            }
        }
        Expr::ILike {
            negated,
            expr,
            pattern,
            escape_char,
        } => {
            let expr = Box::new(build_scalar_expr(*expr, builder.clone())?);
            let pattern = Box::new(build_scalar_expr(*pattern, builder.clone())?);
            ScalarExpr::Like {
                not: negated,
                expr,
                pattern,
                escape_char,
                case_insensitive: true,
            }
        }
        Expr::IsFalse(expr) => {
            let expr = Box::new(build_scalar_expr(*expr, builder.clone())?);
            ScalarExpr::IsFalse { not: false, expr }
        }
        Expr::IsNotFalse(expr) => {
            let expr = Box::new(build_scalar_expr(*expr, builder.clone())?);
            ScalarExpr::IsFalse { not: true, expr }
        }
        Expr::IsTrue(expr) => {
            let expr = Box::new(build_scalar_expr(*expr, builder.clone())?);
            ScalarExpr::IsTrue { not: false, expr }
        }
        Expr::IsNotTrue(expr) => {
            let expr = Box::new(build_scalar_expr(*expr, builder.clone())?);
            ScalarExpr::IsTrue { not: true, expr }
        }
        Expr::IsUnknown(expr) => {
            let expr = Box::new(build_scalar_expr(*expr, builder.clone())?);
            ScalarExpr::IsUnknown { not: false, expr }
        }
        Expr::IsNotUnknown(expr) => {
            let expr = Box::new(build_scalar_expr(*expr, builder.clone())?);
            ScalarExpr::IsUnknown { not: true, expr }
        }
        Expr::SimilarTo {
            negated,
            expr,
            pattern,
            escape_char,
        } => {
            let expr = Box::new(build_scalar_expr(*expr, builder.clone())?);
            let pattern = Box::new(build_scalar_expr(*pattern, builder.clone())?);
            ScalarExpr::Like {
                not: negated,
                expr,
                pattern,
                escape_char,
                case_insensitive: false,
            }
        }
        Expr::RLike { .. } => not_supported!("RLike function. MySQL specific"),
        Expr::Convert { .. } => not_supported!("CONVERT function"),
        Expr::AtTimeZone { .. } => not_supported!("AT TIME ZONE expression"),
        Expr::Ceil { .. } => not_supported!("CEIL expression"),
        Expr::Floor { .. } => not_supported!("FLOOR expression"),
        Expr::Overlay { .. } => not_supported!("OVERLAY expression"),
        Expr::IntroducedString { .. } => not_supported!("Introduced string expression. MySQL specific"),
        Expr::Struct { .. } => not_supported!("Struct expression. BigQuery specific"),
        Expr::Named { .. } => not_supported!("Named expression in a typeless struct. BigQuery specific"),
        Expr::Dictionary(_) => not_supported!("Literal struct expression. Duckdb specific"),
        Expr::Map(_) => not_supported!("Literal struct expression. Duckdb specific"),
        Expr::MatchAgainst { .. } => not_implemented!("MATCH (<col>, <col>, ...) AGAINST (<expr> [<search modifier>])"),
        Expr::Wildcard => ScalarExpr::Wildcard(None),
        Expr::QualifiedWildcard(name) => {
            not_implemented!(name.0.len() > 1, "Wildcard: object names other than <name>.*");
            let name = name.to_string();

            ScalarExpr::Wildcard(Some(name))
        }
        Expr::OuterJoin(_) => not_supported!("Outer join syntax (+)"),
        Expr::Prior(_) => not_supported!("A reference to the prior level in a CONNECT BY clause"),
        Expr::Lambda(_) => not_supported!("A lambda function"),
    };
    Ok(expr)
}

fn build_binary_expr(
    op: BinaryOperator,
    lhs: Expr,
    rhs: Expr,
    builder: OperatorBuilder,
) -> Result<ScalarExpr, OptimizerError> {
    let op = match op {
        BinaryOperator::Plus => BinaryOp::Plus,
        BinaryOperator::Minus => BinaryOp::Minus,
        BinaryOperator::Multiply => BinaryOp::Multiply,
        BinaryOperator::Divide => BinaryOp::Divide,
        BinaryOperator::Modulo => BinaryOp::Modulo,
        BinaryOperator::StringConcat => BinaryOp::Concat,
        BinaryOperator::Gt => BinaryOp::Gt,
        BinaryOperator::Lt => BinaryOp::Lt,
        BinaryOperator::GtEq => BinaryOp::GtEq,
        BinaryOperator::LtEq => BinaryOp::LtEq,
        BinaryOperator::Spaceship => not_supported!("Binary operator: Spaceship"),
        BinaryOperator::Eq => BinaryOp::Eq,
        BinaryOperator::NotEq => BinaryOp::NotEq,
        BinaryOperator::And => BinaryOp::And,
        BinaryOperator::Or => BinaryOp::Or,
        BinaryOperator::Xor => not_implemented!("Binary operator: Xor"),
        BinaryOperator::BitwiseOr => not_implemented!("Binary operator: Bitwise Or"),
        BinaryOperator::BitwiseAnd => not_implemented!("Binary operator: Bitwise And"),
        BinaryOperator::BitwiseXor => not_implemented!("Binary operator: Bitwise Xor"),
        BinaryOperator::PGBitwiseXor => not_supported!("Binary operator: Bitwise Xor (Postgres)"),
        BinaryOperator::PGBitwiseShiftLeft => not_supported!("Binary operator: Bitwise Shift Left (Postgres)"),
        BinaryOperator::PGBitwiseShiftRight => not_supported!("Binary operator: Bitwise Shift Right (Postgres)"),
        BinaryOperator::PGRegexMatch => not_supported!("Binary operator: MATCH (Postgres)"),
        BinaryOperator::PGRegexIMatch => not_supported!("Binary operator: IMATCH (Postgres)"),
        BinaryOperator::PGRegexNotMatch => not_supported!("Binary operator: Regex Match (Postgres)"),
        BinaryOperator::PGRegexNotIMatch => not_supported!("Binary operator: Regex not IMatch (Postgres)"),
        BinaryOperator::DuckIntegerDivide => not_supported!("Binary operator: // division operator (DuckDb)"),
        BinaryOperator::MyIntegerDivide => not_supported!("Binary operator: DIV operator (MySql)"),
        BinaryOperator::Custom(_) => not_supported!("Binary operator: custom operator"),
        BinaryOperator::PGExp => not_supported!("Binary operator: a ^ b (Postgres)"),
        BinaryOperator::PGOverlap => not_supported!("Binary operator: a && b (Postgres)"),
        BinaryOperator::PGLikeMatch => not_supported!("Binary operator: a ~~ b (Postgres)"),
        BinaryOperator::PGILikeMatch => not_supported!("Binary operator: a ~~* b (Postgres)"),
        BinaryOperator::PGNotLikeMatch => not_supported!("Binary operator: a !~~ b (Postgres)"),
        BinaryOperator::PGNotILikeMatch => not_supported!("Binary operator: a !~~* b (Postgres)"),
        BinaryOperator::PGStartsWith => not_supported!("Binary operator: a ^@ b (Postgres)"),
        BinaryOperator::PGCustomBinaryOperator(_) => not_supported!("Binary operator: custom operator (Postgres)"),
        BinaryOperator::Arrow => not_supported!("Binary operator: -> (Postgres)"),
        BinaryOperator::LongArrow => not_supported!("Binary operator: ->> (Postgres)"),
        BinaryOperator::HashArrow => not_supported!("Binary operator: #> (Postgres)"),
        BinaryOperator::HashLongArrow => not_supported!("Binary operator: #>> (Postgres)"),
        BinaryOperator::AtAt => not_supported!("Binary operator: @@ (Postgres)"),
        BinaryOperator::AtArrow => not_supported!("Binary operator: @> (Postgres)"),
        BinaryOperator::ArrowAt => not_supported!("Binary operator: <@ (Postgres)"),
        BinaryOperator::HashMinus => not_supported!("Binary operator: #- (Postgres)"),
        BinaryOperator::AtQuestion => not_supported!("Binary operator: @? (Postgres)"),
        BinaryOperator::Question => not_supported!("Binary operator: ? (Postgres)"),
        BinaryOperator::QuestionAnd => not_supported!("Binary operator: ?& (Postgres)"),
        BinaryOperator::QuestionPipe => not_supported!("Binary operator: ?| (Postgres)"),
    };
    let lhs = build_scalar_expr(lhs, builder.clone())?;
    let rhs = build_scalar_expr(rhs, builder)?;
    Ok(lhs.binary_expr(op, rhs))
}

fn build_unary_expr(op: UnaryOperator, expr: Expr, builder: OperatorBuilder) -> Result<ScalarExpr, OptimizerError> {
    let expr = build_scalar_expr(expr, builder)?;
    let expr = match op {
        UnaryOperator::Plus => expr,
        UnaryOperator::Minus => expr.negate(),
        UnaryOperator::Not => {
            // NOT EXISTS workaround
            if let ScalarExpr::Exists { query, .. } = expr {
                ScalarExpr::Exists { not: true, query }
            } else {
                expr.not()
            }
        }
        UnaryOperator::PGBitwiseNot => not_supported!("Unary operator: Bitwise not (Postgres)"),
        UnaryOperator::PGSquareRoot => not_supported!("Unary operator: Square root (Postgres)"),
        UnaryOperator::PGCubeRoot => not_supported!("Unary operator: Cube root (Postgres)"),
        UnaryOperator::PGPostfixFactorial => not_supported!("Unary operator: Postfix factorial (Postgres)"),
        UnaryOperator::PGPrefixFactorial => not_supported!("Unary operator: Prefix factorial (Postgres)"),
        UnaryOperator::PGAbs => not_supported!("Unary operator: ABS (Postgres)"),
    };
    Ok(expr)
}

fn build_value_expr(value: Value) -> Result<ScalarExpr, OptimizerError> {
    //IDE: "Rust Plugin may show "Match is not exhaustive" error
    //because sqlparser's bigdecimal feature is not enabled.
    let value = match value {
        Value::Number(value, _) => {
            let value = if value.contains('.') {
                f32::from_str(value.as_str())
                    .map(|v| ScalarValue::Float32(Some(v)))
                    .map_err(|e| e.to_string())
            } else {
                i32::from_str(value.as_str())
                    .map(|v| ScalarValue::Int32(Some(v)))
                    .map_err(|e| e.to_string())
            };
            value.map_err(|e| OptimizerError::argument(format!("Unable to parse a numeric literal: {}", e)))?
        }
        Value::SingleQuotedString(value) => ScalarValue::String(Some(value)),
        Value::NationalStringLiteral(_) => not_supported!("National string literal"),
        Value::HexStringLiteral(_) => not_supported!("HEX value literal"),
        Value::DoubleQuotedString(value) => ScalarValue::String(Some(value)),
        Value::Boolean(value) => ScalarValue::Bool(Some(value)),
        Value::Null => ScalarValue::Null,
        Value::DollarQuotedString(_) => not_supported!("Dollar quoted string"),
        Value::EscapedStringLiteral(_) => not_supported!("Escaped string literal (e'string value')"),
        Value::SingleQuotedRawStringLiteral(_) => {
            not_supported!("Single quoted bytes string literal (R'abc')")
        }
        Value::SingleQuotedByteStringLiteral(_) => {
            not_supported!("Single quoted bytes string literal (B'string value')")
        }
        Value::DoubleQuotedRawStringLiteral(_) => {
            not_supported!("Double quoted bytes string literal (R\"abc\")")
        }
        Value::DoubleQuotedByteStringLiteral(_) => {
            not_supported!("Double quoted bytes string literal (B\"string value\")")
        }
        Value::TripleSingleQuotedString(_) => {
            not_supported!("Triple quoted string literal ('''abc''')")
        }
        Value::TripleSingleQuotedRawStringLiteral(_) => {
            not_implemented!("Triple quoted raw string literal (R'''abc''')")
        }
        Value::TripleSingleQuotedByteStringLiteral(_) => {
            not_supported!("Triple quoted bytes string literal (B'''abc'')")
        }
        Value::TripleDoubleQuotedString(_) => {
            not_supported!("Triple double quoted string literal (\"\"\"abc\"\"\")")
        }
        Value::TripleDoubleQuotedRawStringLiteral(_) => {
            not_supported!("Triple double quoted string literal (R\"\"\"abc\"\"\"\"")
        }
        Value::TripleDoubleQuotedByteStringLiteral(_) => {
            not_supported!("Triple double quoted string literal (B\"\"\"abc\"\"\")")
        }
        Value::UnicodeStringLiteral(_) => not_supported!("Unicode string literal u&'string value'"),
        Value::Placeholder(_) => not_implemented!("Placeholder"),
    };
    Ok(ScalarExpr::Scalar(value))
}

fn build_interval_literal(
    value: &str,
    leading_field: DateTimeField,
    last_field: Option<DateTimeField>,
) -> Result<ScalarExpr, OptimizerError> {
    fn interval_field_out_of_range(value: &str) -> OptimizerError {
        OptimizerError::argument(format!("Interval field is out of range: {}", value))
    }

    fn invalid_interval_literal(value: &str) -> OptimizerError {
        OptimizerError::argument(format!("Invalid interval literal: {}", value))
    }

    fn parse_int_value(part: &str, whole_value: &str, bounds: Option<(u32, u32)>) -> Result<u32, OptimizerError> {
        match (u32::from_str(part), bounds) {
            (Ok(v), None) => Ok(v),
            (Ok(v), Some((min, max))) if v >= min && v <= max => Ok(v),
            (Ok(_), _) => Err(interval_field_out_of_range(whole_value)),
            _ => Err(invalid_interval_literal(whole_value)),
        }
    }

    fn parse_year_month(
        sign: i32,
        value: &str,
        original_value: &str,
        from_field: usize,
        to_field: usize,
    ) -> Result<Interval, OptimizerError> {
        let mut fields = [0u32; 2];
        let bounds = [(0, 9999), (0, 11)];

        if to_field - from_field == 0 {
            fields[to_field] = parse_int_value(value, original_value, Some(bounds[to_field]))?;
        } else {
            let mut parts = value.split('-');
            let year = parts.next();
            let month = parts.next();
            let (year, month) = match (year, month) {
                (Some(year), Some(month)) => (
                    parse_int_value(year, value, Some(bounds[0])),
                    parse_int_value(month, original_value, Some(bounds[1])),
                ),
                (Some(year), None) => (parse_int_value(year, original_value, Some(bounds[0])), Ok(0)),
                _ => return Err(invalid_interval_literal(original_value)),
            };
            fields[0] = year?;
            fields[1] = month?;
        }

        Interval::try_from_year_month(sign, fields[0], fields[1])
    }

    fn parse_day_second(
        sign: i32,
        parse_value: &str,
        original_value: &str,
        from_field: usize,
        to_field: usize,
    ) -> Result<Interval, OptimizerError> {
        let mut fields = [0u32; 4];
        let bounds = [(0, 999_999), (0, 23), (0, 59), (0, 59)];

        if to_field - from_field == 0 {
            fields[to_field] = parse_int_value(parse_value, original_value, Some(bounds[to_field]))?;
        } else {
            let mut index = 1;
            let mut parts = parse_value.split(':');

            while index <= to_field {
                if index == 1 {
                    if let Some((day, hour)) = parts.next().and_then(|p| p.trim().split_once(' ')) {
                        let day = parse_int_value(day.trim(), original_value, Some(bounds[0]))?;
                        let hour = parse_int_value(hour.trim(), original_value, Some(bounds[1]))?;
                        fields[0] = day;
                        fields[1] = hour;
                    } else {
                        return Err(invalid_interval_literal(original_value));
                    }
                } else {
                    let field_bounds = bounds[index];
                    // minutes and seconds must be 2 digits
                    if let Some(part) = parts.next().and_then(|p| if p.len() == 2 { Some(p) } else { None }) {
                        let int_part = parse_int_value(part.trim(), original_value, Some(bounds[index]))?;
                        if int_part >= field_bounds.0 && int_part <= field_bounds.1 {
                            fields[index] = int_part;
                        } else {
                            return Err(interval_field_out_of_range(original_value));
                        }
                    } else {
                        return Err(invalid_interval_literal(original_value));
                    }
                }
                index += 1;
            }
            if parts.next().is_some() {
                return Err(invalid_interval_literal(original_value));
            }
        }

        Interval::try_from_days_seconds(sign, fields[0], fields[1], fields[2], fields[3])
    }

    static SUPPORTED_INTERVALS: [(DateTimeField, Option<DateTimeField>, usize, usize); 7] = [
        // Year - 0, Month - 1
        (DateTimeField::Year, None, 0, 0),
        (DateTimeField::Year, Some(DateTimeField::Month), 0, 1),
        (DateTimeField::Month, None, 1, 1),
        // Day - 0, Hour - 1, Minute - 2, Second - 3
        (DateTimeField::Day, None, 0, 0),
        (DateTimeField::Day, Some(DateTimeField::Hour), 0, 1),
        (DateTimeField::Day, Some(DateTimeField::Minute), 0, 2),
        (DateTimeField::Day, Some(DateTimeField::Second), 0, 3),
    ];

    let interval_type = SUPPORTED_INTERVALS
        .iter()
        .position(|(leading, last, ..)| leading == &leading_field && last.as_ref() == last_field.as_ref());

    if let Some(interval_type) = interval_type {
        let (field, .., from, to) = &SUPPORTED_INTERVALS[interval_type];
        let signed_value = value.trim();
        let (unsigned_value, sign) = if signed_value.starts_with('-') {
            (&value[1..], -1)
        } else {
            (value, 1)
        };
        let interval = if field == &DateTimeField::Year || field == &DateTimeField::Month {
            parse_year_month(sign, unsigned_value, value, *from, *to)?
        } else {
            parse_day_second(sign, unsigned_value, value, *from, *to)?
        };
        Ok(ScalarExpr::Scalar(ScalarValue::Interval(Some(interval))))
    } else {
        Err(invalid_interval_literal(value))
    }
}

fn build_function_expr(func: Function, builder: OperatorBuilder) -> Result<ScalarExpr, OptimizerError> {
    let name = func.name.to_string().to_lowercase();

    not_implemented!(!func.within_group.is_empty(), "Aggregate function WITHIN GROUP ordering");

    let window_spec = match func.over {
        None => None,
        Some(WindowType::WindowSpec(spec)) => Some(build_window_spec(builder.clone(), spec)?),
        Some(WindowType::NamedWindow(_)) => {
            not_implemented!("Named window")
        }
    };

    let mut args = vec![];
    let mut count_all = false;
    let mut distinct = false;

    match func.args {
        FunctionArguments::None => {}
        FunctionArguments::Subquery(subquery) => {
            let builder = build_query(builder.sub_query_builder(), *subquery)?;
            let subquery = builder.build()?;
            let subquery_expr = ScalarExpr::SubQuery(RelNode::try_from(subquery)?);
            args.push(subquery_expr);
        }
        FunctionArguments::List(list) => {
            match list.duplicate_treatment {
                Some(DuplicateTreatment::Distinct) => distinct = true,
                Some(DuplicateTreatment::All) => not_supported!("ALL argument option"),
                _ => {}
            }

            for arg in list.args {
                match arg {
                    FunctionArg::Named { .. } => not_supported!("FUNCTION: named function arguments"),
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => {
                        let expr = build_scalar_expr(expr, builder.clone())?;
                        args.push(expr);
                    }
                    FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
                        if name.eq_ignore_ascii_case("count") {
                            args.push(scalar(1));
                            count_all = true;
                        } else {
                            not_implemented!("FUNCTION: wildcard expression in argument list")
                        }
                    }
                    FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(expr)) => {
                        let msg = format!("FUNCTION: qualified wildcard {} expression in argument list", expr);
                        not_implemented!(true, msg)
                    }
                }
            }
        }
    }

    // Reject COUNT(*, a1), COUNT(DISTINCT *) and alike.
    if count_all && (args.len() > 1 || distinct) {
        return Err(OptimizerError::argument(format!("FUNCTION: function {} is called with invalid arguments", name)));
    }

    let func = WindowOrAggregateFunction::try_from(name.as_str());
    match func {
        Err(_) if distinct => {
            let message = format!("DISTINCT is not implemented for non-aggregate functions: {}", name);
            return Err(OptimizerError::argument(message));
        }
        _ => {}
    }

    match (func, window_spec) {
        (Ok(func), Some(spec)) => {
            let FunctionExprWindowSpec { partition_by, order_by } = spec;
            if distinct {
                let message = format!("DISTINCT is not implemented for window functions: {}", name);
                return Err(OptimizerError::argument(message));
            }
            Ok(ScalarExpr::WindowAggregate {
                func,
                args,
                partition_by,
                order_by,
            })
        }
        (Ok(WindowOrAggregateFunction::Aggregate(func)), None) => Ok(ScalarExpr::Aggregate {
            func,
            distinct,
            args,
            filter: None,
        }),
        (Ok(WindowOrAggregateFunction::Window(func)), None) => {
            let msg = format!("WINDOW FUNCTION: no window specification for function: {}", func);
            not_supported!(msg)
        }
        _ => {
            if let Ok(func) = ScalarFunction::try_from(name.as_str()) {
                Ok(ScalarExpr::ScalarFunction { func, args })
            } else {
                let msg = format!("FUNCTION: non aggregate function: {}", name);
                not_implemented!(msg)
            }
        }
    }
}

#[derive(Debug)]
struct FunctionExprWindowSpec {
    partition_by: Vec<ScalarExpr>,
    order_by: Vec<ScalarExpr>,
}

fn build_window_spec(builder: OperatorBuilder, window: WindowSpec) -> Result<FunctionExprWindowSpec, OptimizerError> {
    not_implemented!(!window.order_by.is_empty(), "WINDOW SPEC: order by ");
    not_implemented!(window.window_frame.is_some(), "WINDOW SPEC: window frame");

    let partition_by: Vec<Expr> = window.partition_by;
    let _order_by: Vec<OrderByExpr> = window.order_by;
    let partition_by: Result<Vec<ScalarExpr>, OptimizerError> =
        partition_by.into_iter().map(|expr| build_scalar_expr(expr, builder.clone())).collect();

    Ok(FunctionExprWindowSpec {
        partition_by: partition_by?,
        order_by: vec![],
    })
}

fn convert_data_type(input: SqlDataType) -> Result<DataType, OptimizerError> {
    match input {
        SqlDataType::Char(_) => not_implemented!("Data type: Char"),
        SqlDataType::Varchar(_) => not_implemented!("Data type: Varchar"),
        SqlDataType::Nvarchar(_) => not_implemented!("Data type: NVarchar"),
        SqlDataType::Uuid => not_implemented!("Data type: Uuid"),
        SqlDataType::Clob(_) => not_implemented!("Data type: Clob"),
        SqlDataType::Binary(_) => not_implemented!("Data type: Binary"),
        SqlDataType::Varbinary(_) => not_implemented!("Data type: Varbinary"),
        SqlDataType::Blob(_) => not_implemented!("Data type: Blob"),
        SqlDataType::Decimal(_) => not_implemented!("Data type: Decimal"),
        SqlDataType::Float(_) => not_implemented!("Data type: Float"),
        SqlDataType::TinyInt(_) => not_implemented!("Data type: TinyInt"),
        SqlDataType::SmallInt(_) => not_implemented!("Data type: SmallInt"),
        SqlDataType::Int(_) => {
            //FIXME: Use appropriate numeric type.
            Ok(DataType::Int32)
        }
        SqlDataType::BigInt(_) => {
            //FIXME: Use appropriate numeric type.
            Ok(DataType::Int32)
        }
        SqlDataType::Real => not_implemented!("Data type: Real"),
        SqlDataType::Double => not_implemented!("Data type: Double"),
        SqlDataType::Boolean => Ok(DataType::Bool),
        SqlDataType::Date => Ok(DataType::Date),
        SqlDataType::Time(precision, tz) => {
            not_implemented!(precision.is_some(), "Data type: Time with precision");
            not_implemented!(tz != TimezoneInfo::None, "Data type: Time with time zone");

            Ok(DataType::Time)
        }
        SqlDataType::Timestamp(precision, tz) => {
            not_implemented!(precision.is_some(), "Data type: Timestamp with precision");
            not_implemented!(tz != TimezoneInfo::None, "Data type: Timestamp with time zone");

            Ok(DataType::Timestamp(false /*should be ignored by cast*/))
        }
        SqlDataType::Interval => not_implemented!("Data type: Interval"),
        SqlDataType::Regclass => not_implemented!("Data type: Regclass"),
        SqlDataType::Text => not_implemented!("Data type: Text"),
        SqlDataType::String(length) => {
            not_implemented!(length.is_some(), "Data type: String with precision");

            Ok(DataType::String)
        }
        SqlDataType::Bytea => not_implemented!("Data type: Bytea"),
        SqlDataType::Custom(_, _) => not_implemented!("Data type: Custom data type"),
        SqlDataType::Array(_) => not_implemented!("Data type: Array"),
        SqlDataType::Enum(_) => not_implemented!("Data type: Enum"),
        SqlDataType::Set(_) => not_implemented!("Data type: Set"),
        SqlDataType::UnsignedTinyInt(_) => not_implemented!("Data type: Unsigned TinyInt"),
        SqlDataType::UnsignedSmallInt(_) => not_implemented!("Data type: Unsigned SmallInt"),
        SqlDataType::UnsignedInt(_) => not_implemented!("Data type: Unsigned Int"),
        SqlDataType::UnsignedBigInt(_) => not_implemented!("Data type: Unsigned BigiInt"),
        data_type => {
            let msg = format!("Data type: {}", data_type);
            not_implemented!(msg)
        }
    }
}

impl From<ParserError> for OptimizerError {
    fn from(e: ParserError) -> Self {
        match e {
            ParserError::TokenizerError(err) => OptimizerError::internal(format!("Tokenizer error: {}", err)),
            ParserError::ParserError(err) => OptimizerError::internal(format!("Parser error: {}", err)),
            ParserError::RecursionLimitExceeded => {
                OptimizerError::internal("Parser error: recursion limit exceeded".to_string())
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::sql::testing::{run_sql_expression_tests, run_sql_tests};

    fn run_test_cases(str: &str) {
        let catalog_str = r#"
catalog:
  - table: a
    columns: a1:int32, a2:int32, a3:int32, a4:int32
  - table: b
    columns: b1:int32, b2:int32, b3:int32
  - table: c
    columns: c1:int32, c2:int32, c3:int32
  - table: ab
    columns: a1:int32, b2:int32
"#;
        run_sql_tests(str, catalog_str);
    }

    #[test]
    fn test_basic() {
        let text = include_str!("basic_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_exprs_basic() {
        let text = include_str!("expr_basic_tests.yaml");
        run_sql_expression_tests(text, "");
    }

    #[test]
    fn test_exprs_case() {
        let text = include_str!("expr_case_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_exprs_intervals() {
        let text = include_str!("expr_interval_tests.yaml");
        run_sql_expression_tests(text, "");
    }

    // joins

    #[test]
    fn test_joins() {
        let text = include_str!("joins_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_inner_join() {
        let text = include_str!("join_inner_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_right_join() {
        let text = include_str!("join_right_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_left_join() {
        let text = include_str!("join_left_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_full_join() {
        let text = include_str!("join_full_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_cross_join() {
        let text = include_str!("join_cross_tests.yaml");
        run_test_cases(text);
    }

    // aggregates

    #[test]
    fn test_aggregates() {
        let text = include_str!("aggregate_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_window_aggregates() {
        let text = include_str!("window_aggregate_tests.yaml");
        run_test_cases(text);
    }

    // set operators

    #[test]
    fn test_set_operators() {
        let text = include_str!("set_operator_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_cte() {
        let text = include_str!("cte_tests.yaml");
        run_test_cases(text);
    }

    // subqueries

    #[test]
    fn test_subqueries() {
        let text = include_str!("subqueries_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_correlated_subqueries_exists() {
        let text = include_str!("correlated_exists_tests.yaml");
        run_test_cases(text);
    }

    #[test]
    fn test_correlated_subqueries_in_subquery() {
        let text = include_str!("correlated_in_subquery_tests.yaml");
        run_test_cases(text);
    }
}
