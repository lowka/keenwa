use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{CatalogRef, TableBuilder};
use crate::datatypes::DataType;
use crate::meta::{ColumnId, ColumnMetadata, Metadata, MutableMetadata};
use crate::operators::expr::{AggregateFunction, BinaryOp, Expr, ExprRewriter, ExprVisitor};
use crate::operators::join::{JoinCondition, JoinOn};
use crate::operators::logical::{LogicalExpr, LogicalExprVisitor};
use crate::operators::{ExprMemo, GroupRef, Operator, OperatorExpr, Properties, RelExpr, RelNode, ScalarNode};
use crate::properties::logical::LogicalProperties;
use crate::properties::statistics::Statistics;
use std::collections::{BTreeMap, HashMap, HashSet};

/// TestOperatorTreeBuilder allows to build a fully initialized [operator tree] from another [operator tree] that is not.
/// This is a workaround until there is an implementation of a builder that can create an [operator tree] an SQL syntax tree.
///
/// * It initialises logical properties of every operator in the given tree.
/// * It builds all required metadata (registers columns, tables, etc.)
/// * In case when an operator has statistics it uses those statistics instead of computed ones.
/// * It registers tables referenced in a query in a provided [catalogue](crate::catalog::Catalog).
/// * It uses a [memo](crate::memo::Memo) to recursively memoize operators.
///
/// [operator tree]: crate::operators::Operator
pub struct TestOperatorTreeBuilder<'a> {
    memo: &'a mut ExprMemo,
    catalog: CatalogRef,
    tables: Vec<(String, Vec<(String, DataType)>)>,
    table_column_ids: HashMap<(String, String), ColumnId>,
    table_statistics: HashMap<String, usize>,
    metadata: MutableMetadata, // HashMap<ColumnId, ColumnMetadata>,
}

impl<'a> TestOperatorTreeBuilder<'a> {
    pub fn new(
        memo: &'a mut ExprMemo,
        catalog: CatalogRef,
        tables: Vec<(String, Vec<(String, DataType)>)>,
        table_statistics: HashMap<String, usize>,
    ) -> Self {
        TestOperatorTreeBuilder {
            memo,
            catalog,
            tables,
            table_column_ids: HashMap::new(),
            table_statistics,
            metadata: MutableMetadata::new(),
        }
    }

    /// Builds a fully initialized operator tree from the given operator tree.
    /// * It first locates all tables in the given operator tree to register them in [catalogue](crate::catalog::Catalog).
    /// * It the recursively traverses the given [operator tree](crate::operators::Operator)
    /// initializes their properties operator and copies them into a memo with exception of the root operator.
    /// Its properties are initialized but it is not copied into a memo because the later is done by the optimizer.
    pub fn build_initialized(mut self, expr: Operator) -> (Operator, Metadata) {
        self.init_metadata_and_catalog(&expr);

        let Operator { expr, properties } = expr;
        let expr = match expr {
            OperatorExpr::Relational(rel_expr) => {
                let (expr, properties) = self.build_rel_operator(rel_expr, properties);
                Operator::new(expr, properties)
            }
            OperatorExpr::Scalar(_) => panic!("top level scalar expressions are not supported"),
        };

        (expr, self.metadata.to_metadata())
    }

    fn init_metadata_and_catalog(&mut self, expr: &Operator) {
        struct CollectTables {
            tables: HashSet<String>,
        }
        impl LogicalExprVisitor for CollectTables {
            fn post_visit(&mut self, expr: &LogicalExpr) {
                if let LogicalExpr::Get { source, .. } = expr {
                    self.tables.insert(source.into());
                }
            }
        }

        let mut collector = CollectTables { tables: HashSet::new() };
        expr.expr().as_relational().as_logical().accept(&mut collector);
        let referenced_tables = collector.tables;

        let mut tables = HashMap::new();
        let mut next_col_id = 1;
        let mut initial_metadata = BTreeMap::new();

        for (table_name, columns) in self.tables.clone() {
            if !referenced_tables.contains(&table_name) {
                continue;
            }

            let mut column_id_column = HashMap::new();
            let mut table = TableBuilder::new(table_name.as_str());

            for (name, tpe) in columns.iter() {
                table = table.add_column(name, tpe.clone());
                column_id_column.insert(next_col_id, String::from(name));
                self.table_column_ids.insert((table_name.clone(), name.into()), next_col_id);

                next_col_id += 1;
            }

            let row_count = self.table_statistics[&table_name];
            let table = table.add_row_count(row_count).build();

            for (id, column) in column_id_column {
                let column = table.get_column(&column).expect("Column does not exist");
                let column_meta = ColumnMetadata::new_table_column(
                    column.name().clone(),
                    column.data_type().clone(),
                    table_name.clone(),
                );
                initial_metadata.insert(id, column_meta);
            }

            tables.insert(table_name, table);
        }

        for (_id, column) in initial_metadata {
            self.metadata.add_column(column);
        }

        for (_, table) in tables {
            let catalog = self.catalog.as_any().downcast_ref::<MutableCatalog>().expect("Expected a MutableCatalog");
            catalog.add_table(crate::catalog::DEFAULT_SCHEMA, table);
        }
    }

    fn do_build_operator(&mut self, expr: Operator) -> RelNode {
        let Operator { expr, properties } = expr;
        match expr {
            OperatorExpr::Relational(rel_expr) => {
                let (expr, properties) = self.build_rel_operator(rel_expr, properties);
                self.intern_rel_expr(expr, properties)
            }
            OperatorExpr::Scalar(_) => panic!("top level scalar expressions are not supported"),
        }
    }

    fn build_rel_operator(&mut self, rel_expr: RelExpr, properties: Properties) -> (OperatorExpr, Properties) {
        match rel_expr {
            RelExpr::Logical(expr) => match *expr {
                LogicalExpr::Projection { input, exprs, columns } => {
                    self.build_projection(input, exprs, columns, properties)
                }
                LogicalExpr::Select { input, filter } => self.build_select(input, filter, properties),
                LogicalExpr::Aggregate {
                    input,
                    aggr_exprs,
                    group_exprs,
                } => self.build_aggregate(input, aggr_exprs, group_exprs, properties),
                LogicalExpr::Join { left, right, condition } => self.build_join(left, right, condition, properties),
                LogicalExpr::Get { source, columns } => self.build_get(source, columns, properties),
                LogicalExpr::Union { left, right, all } => {
                    self.build_set_operator(left, right, SetOp::Union, all, properties)
                }
                LogicalExpr::Intersect { left, right, all } => {
                    self.build_set_operator(left, right, SetOp::Intersect, all, properties)
                }
                LogicalExpr::Except { left, right, all } => {
                    self.build_set_operator(left, right, SetOp::Expect, all, properties)
                }
            },
            RelExpr::Physical(_) => panic!("Physical exprs are not supported"),
        }
    }

    fn build_get(
        &mut self,
        source: String,
        columns: Vec<ColumnId>,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        for column_id in columns.iter() {
            let column_meta = self.metadata.get_column(column_id);
            let table_name = column_meta.table().expect("Get expression contains unexpected column");
            assert_eq!(table_name, source.as_str(), "column source");
        }

        let expr = LogicalExpr::Get {
            source,
            columns: columns.clone(),
        };

        let statistics = properties.logical.statistics().cloned();
        let required = properties.required;

        let output_columns = columns;
        let logical = LogicalProperties::new(output_columns, statistics);

        let properties = Properties::new(logical, required);
        (OperatorExpr::from(expr), properties)
    }

    fn build_select(
        &mut self,
        input: RelNode,
        filter: Option<ScalarNode>,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let input = self.build_rel_node(input);
        let input_properties = input.props();
        let input_logical = input_properties.logical();

        let filter = match filter {
            Some(filter) => {
                let (expr, properties) = self.build_scalar_node(filter);
                let group = self.intern_expr(expr, properties);
                Some(ScalarNode::Group(group))
            }
            None => None,
        };

        let statistics = properties.logical.statistics().cloned();
        let required = properties.required;

        let output_columns = input_logical.output_columns().to_vec();
        let logical = LogicalProperties::new(output_columns, statistics);
        let properties = Properties::new(logical, required);

        let expr = LogicalExpr::Select { input, filter };

        (OperatorExpr::from(expr), properties)
    }

    fn build_projection(
        &mut self,
        input: RelNode,
        exprs: Vec<Expr>,
        columns: Vec<ColumnId>,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let input = self.build_rel_node(input);
        let input_properties = input.props();
        let input_logical = input_properties.logical();

        let (expr, properties) = if !columns.is_empty() {
            assert!(exprs.is_empty(), "exprs must be empty");

            for column_id in columns.iter() {
                check_column_exists(column_id, input_logical, "Invalid column id in projection column")
            }

            let expr = LogicalExpr::Projection {
                input,
                exprs: vec![],
                columns: columns.clone(),
            };

            let statistics = properties.logical.statistics().cloned();
            let required = properties.required;

            let logical = LogicalProperties::new(columns, statistics);
            let properties = Properties::new(logical, required);

            (expr, properties)
        } else {
            let mut columns = Vec::with_capacity(exprs.len());
            for expr in exprs {
                match expr {
                    Expr::Column(column_id) => {
                        check_column_exists(
                            &column_id,
                            input_properties.logical(),
                            "Invalid column id in projection expr",
                        );
                        columns.push(column_id);
                    }
                    _ => {
                        let expr = self.build_scalar_operator(expr, input_logical);
                        let column_name = "?column?".to_string();
                        let data_type = self.resolve_expr_type(&expr);
                        let column_meta = ColumnMetadata::new_synthetic_column(column_name, data_type, Some(expr));
                        let column_id = self.metadata.add_column(column_meta);
                        columns.push(column_id);
                    }
                }
            }

            let expr = LogicalExpr::Projection {
                input,
                exprs: vec![],
                columns: columns.clone(),
            };

            let statistics = properties.logical.statistics().cloned();
            let required = properties.required;

            let logical = LogicalProperties::new(columns, statistics);
            let properties = Properties::new(logical, required);

            (expr, properties)
        };

        (OperatorExpr::from(expr), properties)
    }

    fn build_join(
        &mut self,
        left: RelNode,
        right: RelNode,
        condition: JoinCondition,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let left = self.build_rel_node(left);
        let right = self.build_rel_node(right);
        let left_properties = left.props();
        let right_properties = right.props();

        let condition = match condition {
            JoinCondition::Using(using) => {
                for (l, r) in using.columns() {
                    check_column_exists(
                        l,
                        left_properties.logical(),
                        "Invalid column one the left side of join condition",
                    );
                    check_column_exists(r, right_properties.logical(), "Invalid column on the right of join condition")
                }
                JoinCondition::Using(using)
            }
            JoinCondition::On(on_expr) => {
                let columns = on_expr.get_columns();
                let JoinOn { expr } = on_expr;

                for col_id in columns {
                    let left_columns = left_properties.logical().output_columns();
                    let right_columns = right_properties.logical().output_columns();
                    assert!(
                        left_columns.contains(&col_id) || right_columns.contains(&col_id),
                        "Invalid column in ON expression. Expr: {}",
                        expr.expr()
                    );
                }

                let (expr, properties) = self.build_scalar_node(expr);
                let group = self.intern_expr(expr, properties);
                JoinCondition::On(JoinOn::new(ScalarNode::Group(group)))
            }
        };

        let mut output_columns = left_properties.logical.output_columns().to_vec();
        output_columns.extend_from_slice(right_properties.logical.output_columns());

        let statistics = properties.logical.statistics().cloned();
        let required = properties.required;

        let logical = LogicalProperties::new(output_columns, statistics);
        let properties = Properties::new(logical, required);

        let expr = LogicalExpr::Join { left, right, condition };

        (OperatorExpr::from(expr), properties)
    }

    fn build_aggregate(
        &mut self,
        input: RelNode,
        aggr_exprs: Vec<ScalarNode>,
        group_exprs: Vec<ScalarNode>,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let input = self.build_rel_node(input);
        let input_properties = input.props();
        let input_logical = input_properties.logical();

        struct CollectGroupByColumns<'a> {
            columns: &'a mut Vec<ColumnId>,
            input_properties: &'a LogicalProperties,
        }
        impl ExprVisitor for CollectGroupByColumns<'_> {
            fn post_visit(&mut self, expr: &Expr) {
                if let Expr::Column(column_id) = expr {
                    self.columns.push(*column_id);
                    check_column_exists(column_id, self.input_properties, "Invalid column in a group by expression")
                }
            }
        }

        struct ValidateAggregateExpr<'a> {
            group_by_columns: &'a [ColumnId],
            input_properties: &'a LogicalProperties,
            in_aggregate: bool,
        }

        impl ExprVisitor for ValidateAggregateExpr<'_> {
            fn pre_visit(&mut self, expr: &Expr) {
                if let Expr::Aggregate { .. } = expr {
                    assert!(!self.in_aggregate, "Nested aggregate expressions are not allowed");
                    self.in_aggregate = true;
                }
            }

            fn post_visit(&mut self, expr: &Expr) {
                match expr {
                    Expr::Column(column_id) if !self.in_aggregate => {
                        assert!(
                            self.group_by_columns.contains(column_id),
                            "Column {} must appear in the GROUP BY clause",
                            column_id
                        )
                    }
                    Expr::Column(column_id) if self.in_aggregate => {
                        check_column_exists(
                            column_id,
                            self.input_properties,
                            "Invalid column in an aggregate expression",
                        );
                    }
                    Expr::Aggregate { .. } => {
                        self.in_aggregate = false;
                    }
                    _ => {}
                }
            }
        }

        let mut output_columns = Vec::with_capacity(aggr_exprs.len());

        let mut new_group_exprs = Vec::with_capacity(group_exprs.len());
        let mut group_by_columns = Vec::new();
        let mut collector = CollectGroupByColumns {
            columns: &mut group_by_columns,
            input_properties: input_logical,
        };

        for group_expr in group_exprs.into_iter().map(extract_expr) {
            let group_expr = self.build_scalar_operator(group_expr, input_logical);
            group_expr.accept(&mut collector);
            let scalar_expr = self.intern_scalar_expr(group_expr);
            new_group_exprs.push(scalar_expr);
        }

        let mut new_aggr_exprs = Vec::with_capacity(aggr_exprs.len());

        for aggr_expr in aggr_exprs.into_iter().map(extract_expr) {
            let aggr_expr = self.build_scalar_operator(aggr_expr, input_logical);

            let mut aggr_validator = ValidateAggregateExpr {
                group_by_columns: &group_by_columns,
                input_properties: input_logical,
                in_aggregate: false,
            };
            aggr_expr.accept(&mut aggr_validator);

            let column_id = self.add_aggregate_column(aggr_expr.clone(), input_logical);
            output_columns.push(column_id);

            let scalar_expr = self.intern_scalar_expr(aggr_expr);
            new_aggr_exprs.push(scalar_expr);
        }

        let statistics = properties.logical.statistics().cloned();
        let required = properties.required;

        let logical = LogicalProperties::new(output_columns, statistics);
        let properties = Properties::new(logical, required);

        let expr = LogicalExpr::Aggregate {
            input,
            aggr_exprs: new_aggr_exprs,
            group_exprs: new_group_exprs,
        };

        (OperatorExpr::from(expr), properties)
    }

    fn build_set_operator(
        &mut self,
        left: RelNode,
        right: RelNode,
        set_op: SetOp,
        all: bool,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let left = self.build_rel_node(left);
        let right = self.build_rel_node(right);
        let left_properties = left.props();
        let right_properties = right.props();

        let left_columns = left_properties.logical().output_columns();
        let right_columns = right_properties.logical().output_columns();

        assert_eq!(left_columns.len(), right_columns.len(), "Number of columns does not match");

        let mut output_columns = Vec::with_capacity(left_columns.len());

        for (i, (l, r)) in left_columns.iter().zip(right_columns.iter()).enumerate() {
            let left_column = self.metadata.get_column(l);
            let right_column = self.metadata.get_column(r);
            let left_type = left_column.data_type().clone();
            let right_type = right_column.data_type().clone();

            assert_eq!(
                left_type, right_type,
                "Data type of {}-th column does not match. Type casting is not implemented",
                i
            );

            let column_meta = ColumnMetadata::new_synthetic_column("".to_string(), left_type, None);
            let column_id = self.metadata.add_column(column_meta);
            output_columns.push(column_id);
        }

        let statistics = properties.logical.statistics().cloned();
        let required = properties.required;

        let logical = LogicalProperties::new(output_columns, statistics);
        let properties = Properties::new(logical, required);

        let expr = match set_op {
            SetOp::Union => LogicalExpr::Union { left, right, all },
            SetOp::Intersect => LogicalExpr::Intersect { left, right, all },
            SetOp::Expect => LogicalExpr::Except { left, right, all },
        };

        (OperatorExpr::from(expr), properties)
    }

    fn add_aggregate_column(&mut self, expr: Expr, _input_properties: &LogicalProperties) -> ColumnId {
        let expr_type = self.resolve_expr_type(&expr);
        let column_name = if let Expr::Aggregate { func, .. } = &expr {
            func.to_string()
        } else {
            "".to_string()
        };
        let column_meta = ColumnMetadata::new_synthetic_column(column_name, expr_type, Some(expr));

        self.metadata.add_column(column_meta)
    }

    fn build_rel_node(&mut self, node: RelNode) -> RelNode {
        match node {
            RelNode::Expr(expr) => self.do_build_operator(*expr),
            RelNode::Group(_) => panic!("Expected an expression"),
        }
    }

    fn build_scalar_node(&mut self, scalar: ScalarNode) -> (OperatorExpr, Properties) {
        match scalar {
            ScalarNode::Expr(expr) => {
                let expr = *expr;
                let properties = expr.properties;
                match expr.expr {
                    OperatorExpr::Relational(expr) => panic!("ScalarNode contains a relational expr: {:?}", expr),
                    OperatorExpr::Scalar(scalar_expr) => {
                        let new_scalar_expr = self.build_scalar_operator(scalar_expr, properties.logical());
                        let properties = self.build_scalar_expr_properties(properties);
                        (OperatorExpr::from(new_scalar_expr), properties)
                    }
                }
            }
            ScalarNode::Group(group) => panic!("Expected an expression but got a group: {:?}", group),
        }
    }

    fn build_scalar_operator(&mut self, expr: Expr, _input_properties: &LogicalProperties) -> Expr {
        struct InternNestedRelExprs<F> {
            build_metadata: F,
        }
        impl<F> ExprRewriter for InternNestedRelExprs<F>
        where
            F: FnMut(RelNode) -> RelNode,
        {
            fn rewrite(&mut self, expr: Expr) -> Expr {
                if let Expr::SubQuery(rel_node) = expr {
                    let rel_node = (self.build_metadata)(rel_node);
                    Expr::SubQuery(rel_node)
                } else {
                    expr
                }
            }
        }
        let f = |rel_node: RelNode| self.build_rel_node(rel_node);
        let mut rewriter = InternNestedRelExprs { build_metadata: f };
        expr.rewrite(&mut rewriter)
    }

    fn intern_rel_expr(&mut self, expr: OperatorExpr, properties: Properties) -> RelNode {
        let group = self.intern_expr(expr, properties);
        RelNode::Group(group)
    }

    fn intern_scalar_expr(&mut self, expr: Expr) -> ScalarNode {
        let expr = Operator::from(OperatorExpr::from(expr));
        let (group, _) = self.memo.insert(expr);
        ScalarNode::Group(group)
    }

    fn intern_expr(&mut self, expr: OperatorExpr, properties: Properties) -> GroupRef {
        let expr = Operator::new(OperatorExpr::from(expr), properties);
        let (group, _) = self.memo.insert(expr);
        group
    }

    fn build_scalar_expr_properties(&self, properties: Properties) -> Properties {
        let Properties { logical, required } = properties;
        let LogicalProperties {
            output_columns,
            statistics,
        } = logical;
        let statistics = statistics.or_else(|| Some(Statistics::default()));

        Properties::new(LogicalProperties::new(output_columns, statistics), required)
    }

    fn resolve_expr_type(&self, expr: &Expr) -> DataType {
        match expr {
            Expr::Column(column_id) => self.metadata.get_column(column_id).data_type().clone(),
            Expr::Scalar(value) => value.data_type(),
            Expr::BinaryExpr { lhs, op, rhs } => {
                let left_tpe = self.resolve_expr_type(lhs);
                let right_tpe = self.resolve_expr_type(rhs);
                self.resolve_binary_expr_type(left_tpe, op, right_tpe)
            }
            Expr::Not(expr) => {
                let tpe = self.resolve_expr_type(expr);
                assert_eq!(tpe, DataType::Bool, "Invalid argument type for NOT operator");
                tpe
            }
            Expr::Aggregate { func, args, .. } => {
                for (i, arg) in args.iter().enumerate() {
                    let arg_tpe = self.resolve_expr_type(arg);
                    let expected_tpe = match func {
                        AggregateFunction::Avg
                        | AggregateFunction::Max
                        | AggregateFunction::Min
                        | AggregateFunction::Sum => DataType::Int32,
                        AggregateFunction::Count => arg_tpe.clone(),
                    };
                    assert_eq!(
                        &arg_tpe, &expected_tpe,
                        "Invalid argument type for aggregate function {}. Argument#{} {}",
                        func, i, arg
                    );
                }
                DataType::Int32
            }
            Expr::SubQuery(_) => DataType::Int32,
        }
    }

    fn resolve_binary_expr_type(&self, lhs: DataType, _op: &BinaryOp, rhs: DataType) -> DataType {
        assert_eq!(lhs, rhs, "Types does not match");
        DataType::Bool
    }
}

#[derive(Debug)]
enum SetOp {
    Union,
    Intersect,
    Expect,
}

fn extract_expr(scalar_node: ScalarNode) -> Expr {
    match scalar_node {
        ScalarNode::Expr(expr) => match expr.expr {
            OperatorExpr::Relational(_) => panic!("ScalarNode contains a relational expression"),
            OperatorExpr::Scalar(expr) => expr,
        },
        ScalarNode::Group(_) => panic!("Expected a scalar expression but got a group"),
    }
}

fn check_column_exists(column_id: &ColumnId, input_properties: &LogicalProperties, message: &str) {
    let output_columns = input_properties.output_columns();
    assert!(
        output_columns.contains(column_id),
        "{}: Unexpected column {}. Expected: {:?}",
        message,
        column_id,
        output_columns
    )
}

#[cfg(test)]
mod test {
    use crate::catalog::mutable::MutableCatalog;
    use crate::datatypes::DataType;
    use crate::memo::{format_memo, Memo};
    use crate::meta::{ColumnId, ColumnMetadata};
    use crate::operators::expr::{AggregateFunction, BinaryOp, Expr};
    use crate::operators::join::JoinCondition;
    use crate::operators::logical::LogicalExpr;
    use crate::operators::operator_tree::TestOperatorTreeBuilder;
    use crate::operators::scalar::ScalarValue;
    use crate::operators::{Operator, RelNode, ScalarNode};
    use crate::rules::testing::format_expr;
    use itertools::Itertools;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_get() {
        let expr = LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .to_operator();

        expect_expr(
            expr,
            r#"
LogicalGet A cols=[1, 2]
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
"#,
        )
    }

    #[test]
    fn test_join() {
        let left = LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .to_operator();

        let right = LogicalExpr::Get {
            source: "B".into(),
            columns: vec![3, 4],
        }
        .to_operator();

        let join = LogicalExpr::Join {
            left: left.into(),
            right: right.into(),
            condition: JoinCondition::using(vec![(1, 3)]),
        };

        expect_expr(
            join.to_operator(),
            r#"
LogicalJoin using=[(1, 3)]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [1, 2, 3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
Memo:
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_select_simple_filter() {
        let filter = Some(
            Expr::BinaryExpr {
                lhs: Box::new(Expr::Column(1)),
                op: BinaryOp::Gt,
                rhs: Box::new(Expr::Scalar(ScalarValue::Int32(37))),
            }
            .into(),
        );
        test_select(
            filter,
            r#"
LogicalSelect
  input: LogicalGet A cols=[1, 2]
  filter: Expr col:1 > 37
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
Memo:
  01 Expr col:1 > 37
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_select_nested_sub_query_filter() {
        let sub_query = new_sub_query();

        let filter = Some(
            Expr::BinaryExpr {
                lhs: Box::new(Expr::SubQuery(RelNode::from(sub_query))),
                op: BinaryOp::Gt,
                rhs: Box::new(Expr::Scalar(ScalarValue::Int32(37))),
            }
            .into(),
        );

        test_select(
            filter,
            r#"
LogicalSelect
  input: LogicalGet A cols=[1, 2]
  filter: Expr SubQuery 02 > 37
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32, expr: count(col:3)
Memo:
  03 Expr SubQuery 02 > 37
  02 LogicalProjection input=01 cols=[5]
  01 LogicalGet B cols=[3]
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    fn test_select(filter: Option<ScalarNode>, expected: &str) {
        let select = LogicalExpr::Select {
            input: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            filter,
        }
        .to_operator();
        expect_expr(select, expected)
    }

    #[test]
    fn test_project_exprs() {
        projection_test(
            vec![],
            vec![Expr::Column(1), Expr::Scalar(ScalarValue::Int32(10))],
            r#"
LogicalProjection cols=[1, 3]
  input: LogicalGet A cols=[1, 2]
  output cols: [1, 3]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 ?column? Int32, expr: 10
Memo:
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_project_with_nested_sub_query() {
        let sub_query = new_sub_query();
        let rel_node = RelNode::Expr(Box::new(sub_query));

        projection_test(
            vec![],
            vec![Expr::Column(1), Expr::SubQuery(rel_node)],
            r#"
LogicalProjection cols=[1, 6]
  input: LogicalGet A cols=[1, 2]
  output cols: [1, 6]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32, expr: count(col:3)
  col:6 ?column? Int32, expr: SubQuery 02
Memo:
  02 LogicalProjection input=01 cols=[5]
  01 LogicalGet B cols=[3]
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    fn projection_test(columns: Vec<ColumnId>, exprs: Vec<Expr>, expected: &str) {
        let expr = LogicalExpr::Projection {
            input: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            exprs,
            columns,
        }
        .to_operator();

        expect_expr(expr, expected)
    }

    #[test]
    fn test_aggregate_no_groups() {
        let expr = LogicalExpr::Aggregate {
            input: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            aggr_exprs: vec![
                Expr::Aggregate {
                    func: AggregateFunction::Count,
                    args: vec![Expr::Column(1)],
                    filter: None,
                }
                .into(),
                Expr::Aggregate {
                    func: AggregateFunction::Max,
                    args: vec![Expr::Column(1)],
                    filter: None,
                }
                .into(),
            ],
            group_exprs: vec![],
        }
        .to_operator();

        expect_expr(
            expr,
            r#"
LogicalAggregate
  input: LogicalGet A cols=[1, 2]
  : Expr count(col:1)
  : Expr max(col:1)
  output cols: [3, 4]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 count Int32, expr: count(col:1)
  col:4 max Int32, expr: max(col:1)
Memo:
  02 Expr max(col:1)
  01 Expr count(col:1)
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_aggregate_with_groups() {
        let expr = LogicalExpr::Aggregate {
            input: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            aggr_exprs: vec![Expr::Aggregate {
                func: AggregateFunction::Count,
                args: vec![Expr::Column(1)],
                filter: None,
            }
            .into()],
            group_exprs: vec![Expr::Column(1).into()],
        }
        .to_operator();

        expect_expr(
            expr,
            r#"
LogicalAggregate
  input: LogicalGet A cols=[1, 2]
  : Expr count(col:1)
  : Expr col:1
  output cols: [3]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 count Int32, expr: count(col:1)
Memo:
  02 Expr count(col:1)
  01 Expr col:1
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_union() {
        let expr = LogicalExpr::Union {
            left: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            right: LogicalExpr::Get {
                source: "B".into(),
                columns: vec![3, 4],
            }
            .into(),
            all: false,
        }
        .to_operator();

        expect_expr(
            expr,
            r#"
LogicalUnion all=false
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [5, 6]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32
  col:6 ?column? Int32
Memo:
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_intersect() {
        let expr = LogicalExpr::Intersect {
            left: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            right: LogicalExpr::Get {
                source: "B".into(),
                columns: vec![3, 4],
            }
            .into(),
            all: false,
        }
        .to_operator();

        expect_expr(
            expr,
            r#"
LogicalIntersect all=false
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [5, 6]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32
  col:6 ?column? Int32
Memo:
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    #[test]
    fn test_except() {
        let expr = LogicalExpr::Except {
            left: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            right: LogicalExpr::Get {
                source: "B".into(),
                columns: vec![3, 4],
            }
            .into(),
            all: false,
        }
        .to_operator();

        expect_expr(
            expr,
            r#"
LogicalExcept all=false
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
  output cols: [5, 6]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32
  col:6 ?column? Int32
Memo:
  01 LogicalGet B cols=[3, 4]
  00 LogicalGet A cols=[1, 2]
"#,
        )
    }

    fn new_sub_query() -> Operator {
        LogicalExpr::Projection {
            input: LogicalExpr::Get {
                source: "B".into(),
                columns: vec![3],
            }
            .into(),
            exprs: vec![Expr::Aggregate {
                func: AggregateFunction::Count,
                args: vec![Expr::Column(3)],
                filter: None,
            }],
            columns: vec![],
        }
        .to_operator()
    }

    fn expect_expr(expr: Operator, expected: &str) {
        let tables = vec![
            ("A".into(), vec![("a1".into(), DataType::Int32), ("a2".into(), DataType::Int32)]),
            ("B".into(), vec![("b1".into(), DataType::Int32), ("b2".into(), DataType::Int32)]),
        ];
        let mut memo = Memo::new();
        let catalog = Arc::new(MutableCatalog::new());
        let builder = TestOperatorTreeBuilder::new(
            &mut memo,
            catalog,
            tables,
            HashMap::from([("A".into(), 10), ("B".into(), 10)]),
        );

        let (expr, metadata) = builder.build_initialized(expr);
        let mut buf = String::new();
        buf.push('\n');

        buf.push_str(format_expr(&expr).as_str());
        buf.push('\n');

        buf.push_str(format!("  output cols: {:?}\n", expr.logical().output_columns()).as_str());
        buf.push_str("Metadata:\n");

        let columns: Vec<(ColumnId, ColumnMetadata)> =
            metadata.columns().map(|(k, v)| (k, v.clone())).sorted_by(|a, b| a.0.cmp(&b.0)).collect();

        for (id, column) in columns {
            let expr = column.expr();
            let table = column.table();
            let column_name = if !column.name().is_empty() {
                column.name().as_str()
            } else {
                "?column?"
            };
            let column_info = match (expr, table) {
                (None, None) => format!("  col:{} {} {:?}", id, column_name, column.data_type()),
                (None, Some(table)) => format!("  col:{} {}.{} {:?}", id, table, column_name, column.data_type()),
                (Some(expr), _) => format!("  col:{} {} {:?}, expr: {}", id, column_name, column.data_type(), expr),
            };
            buf.push_str(column_info.as_str());
            buf.push('\n');
        }

        if !memo.is_empty() {
            buf.push_str("Memo:\n");
            let memo_as_string = format_memo(&memo);
            let lines = memo_as_string
                .split('\n')
                .map(|l| {
                    if !l.is_empty() {
                        let mut s = String::new();
                        s.push_str("  ");
                        s.push_str(l);
                        s
                    } else {
                        l.to_string()
                    }
                })
                .join("\n");

            buf.push_str(lines.as_str());
        }

        assert_eq!(buf.as_str(), expected);
    }
}
