use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{Column, ColumnRef, TableBuilder};
use crate::datatypes::DataType;
use crate::memo::Memo;
use crate::meta::ColumnId;
use crate::operators::expr::{AggregateFunction, Expr, ExprRewriter, ExprVisitor};
use crate::operators::join::JoinCondition;
use crate::operators::logical::LogicalExpr;
use crate::operators::{Operator, OperatorExpr, Properties, RelExpr, RelNode, ScalarNode};
use crate::properties::logical::LogicalProperties;
use std::collections::HashMap;

#[derive(Debug)]
struct ColumnMetadata {
    column: ColumnRef,
    expr: Option<Expr>,
}

struct MetadataBuilder<'a> {
    memo: &'a mut Memo<Operator>,
    catalog: MutableCatalog,
    metadata: HashMap<ColumnId, ColumnMetadata>,
}

impl<'a> MetadataBuilder<'a> {
    pub fn new(memo: &'a mut Memo<Operator>, data: Vec<(String, Vec<(String, DataType)>)>) -> Self {
        let mut metadata = HashMap::new();
        let mut tables = HashMap::new();
        let mut column_id_column = HashMap::new();
        let mut max_id = 1;

        for (table_name, columns) in data {
            let mut table = TableBuilder::new(table_name.as_str());
            for (name, tpe) in columns.iter() {
                table = table.add_column(name, tpe.clone());
                column_id_column.insert(max_id, (String::from(table_name.as_str()), String::from(name)));
                max_id += 1;
            }
            let table = table.build();
            tables.insert(table_name, table);
        }

        for (id, (table, column)) in column_id_column {
            let table = tables.get(&table).expect("Table does not exist");
            let column = table.get_column(&column).expect("Column does not exist");
            let column_meta = ColumnMetadata { column, expr: None };
            metadata.insert(id, column_meta);
        }

        let catalog = MutableCatalog::new();
        for (_, table) in tables {
            catalog.add_table(crate::catalog::DEFAULT_SCHEMA, table);
        }

        MetadataBuilder {
            memo,
            catalog,
            metadata,
        }
    }

    pub fn build_metadata(&mut self, expr: Operator) -> Operator {
        let (expr, properties) = self.do_build_metadata(expr);
        Operator::new(expr, properties)
    }

    fn do_build_metadata(&mut self, expr: Operator) -> (OperatorExpr, Properties) {
        let Operator { expr, properties } = expr;
        match expr {
            OperatorExpr::Relational(rel_expr) => {
                let (expr, properties) = self.build_rel_expr_metadata(rel_expr, properties);
                (expr, properties)
            }
            OperatorExpr::Scalar(_) => todo!(),
        }
    }

    fn build_rel_expr_metadata(&mut self, rel_expr: RelExpr, properties: Properties) -> (OperatorExpr, Properties) {
        match rel_expr {
            RelExpr::Logical(expr) => match *expr {
                LogicalExpr::Projection { input, exprs, columns } => {
                    self.build_projection_metadata(input, exprs, columns, properties)
                }
                LogicalExpr::Select { input, filter } => self.build_select_metadata(input, filter, properties),
                LogicalExpr::Aggregate {
                    input,
                    aggr_exprs,
                    group_exprs,
                } => self.build_aggregate_metadata(input, aggr_exprs, group_exprs, properties),
                LogicalExpr::Join { left, right, condition } => {
                    self.build_join_metadata(left, right, condition, properties)
                }
                LogicalExpr::Get { source, columns } => self.build_get_metadata(source, columns, properties),
                LogicalExpr::Union { left, right, all } => {
                    self.build_set_op_metadata(left, right, SetOp::Union, all, properties)
                }
                LogicalExpr::Intersect { left, right, all } => {
                    self.build_set_op_metadata(left, right, SetOp::Intersect, all, properties)
                }
                LogicalExpr::Except { left, right, all } => {
                    self.build_set_op_metadata(left, right, SetOp::Expect, all, properties)
                }
            },
            RelExpr::Physical(_) => panic!("Physical exprs are not supported"),
        }
    }

    fn build_get_metadata(
        &mut self,
        source: String,
        columns: Vec<ColumnId>,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        for column_id in columns.iter() {
            let column_meta = self.metadata.get(column_id).expect("Unknown column id");
            let table_name = column_meta.column.table().expect("Get expression contains unexpected column");
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

    fn build_select_metadata(
        &mut self,
        input: RelNode,
        filter: Option<ScalarNode>,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let (input, input_properties) = self.build_rel_node_metadata(input);
        let input_logical = input_properties.logical();

        let filter = match filter {
            Some(filter) => {
                let (expr, properties) = self.build_scalar_node_metadata(filter);
                Some(ScalarNode::Expr(Box::new(Operator::new(expr, properties))))
            }
            None => None,
        };

        let statistics = properties.logical.statistics().cloned();
        let required = properties.required;

        let output_columns = input_logical.output_columns().to_vec();
        let logical = LogicalProperties::new(output_columns, statistics);
        let properties = Properties::new(logical, required);

        let expr = LogicalExpr::Select {
            input: RelNode::from(Operator::new(input, input_properties)),
            filter,
        };

        (OperatorExpr::from(expr), properties)
    }

    fn build_projection_metadata(
        &mut self,
        input: RelNode,
        exprs: Vec<Expr>,
        columns: Vec<ColumnId>,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let (input, input_properties) = self.build_rel_node_metadata(input);
        let input_logical = input_properties.logical();

        let (expr, properties) = if !columns.is_empty() {
            assert!(exprs.is_empty(), "exprs must be empty");

            for column_id in columns.iter() {
                check_column_exists(column_id, input_logical, "Invalid column id in projection column")
            }

            let expr = LogicalExpr::Projection {
                input: RelNode::from(Operator::new(input, input_properties)),
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
                        let expr = self.build_scalar_expr_metadata(expr, input_logical);
                        let expr_type = self.resolve_expr_type(&expr);
                        let column = Column::new("?column?".to_string(), None, expr_type);
                        let column_meta = ColumnMetadata {
                            column: ColumnRef::new(column),
                            expr: Some(expr),
                        };

                        let col_id = self.metadata.len() + 1;
                        columns.push(col_id);
                        self.metadata.insert(col_id, column_meta);
                    }
                }
            }

            let expr = LogicalExpr::Projection {
                input: RelNode::from(Operator::new(input, input_properties)),
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

    fn build_join_metadata(
        &mut self,
        left: RelNode,
        right: RelNode,
        condition: JoinCondition,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let (left, left_properties) = self.build_rel_node_metadata(left);
        let (right, right_properties) = self.build_rel_node_metadata(right);

        match &condition {
            JoinCondition::Using(using) => {
                for (l, r) in using.columns() {
                    check_column_exists(
                        l,
                        left_properties.logical(),
                        "Invalid column one the left side of join condition",
                    );
                    check_column_exists(r, right_properties.logical(), "Invalid column on the right of join condition")
                }
            }
        }

        let mut output_columns = left_properties.logical.output_columns().to_vec();
        output_columns.extend_from_slice(right_properties.logical.output_columns());

        let statistics = properties.logical.statistics().cloned();
        let required = properties.required;

        let logical = LogicalProperties::new(output_columns, statistics);
        let properties = Properties::new(logical, required);

        let expr = LogicalExpr::Join {
            left: RelNode::from(Operator::new(left, left_properties)),
            right: RelNode::from(Operator::new(right, right_properties)),
            condition,
        };

        (OperatorExpr::from(expr), properties)
    }

    fn build_aggregate_metadata(
        &mut self,
        input: RelNode,
        aggr_exprs: Vec<ScalarNode>,
        group_exprs: Vec<ScalarNode>,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let (input, input_properties) = self.build_rel_node_metadata(input);
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
            let group_expr = self.build_scalar_expr_metadata(group_expr, input_logical);
            group_expr.accept(&mut collector);
            new_group_exprs.push(ScalarNode::from(group_expr));
        }

        let mut new_aggr_exprs = Vec::with_capacity(aggr_exprs.len());

        for aggr_expr in aggr_exprs.into_iter().map(extract_expr) {
            let aggr_expr = self.build_scalar_expr_metadata(aggr_expr, input_logical);

            let mut aggr_validator = ValidateAggregateExpr {
                group_by_columns: &group_by_columns,
                input_properties: input_logical,
                in_aggregate: false,
            };
            aggr_expr.accept(&mut aggr_validator);

            let column_id = self.add_aggregate_column(aggr_expr.clone(), input_logical);
            output_columns.push(column_id);

            new_aggr_exprs.push(ScalarNode::from(aggr_expr));
        }

        let statistics = properties.logical.statistics().cloned();
        let required = properties.required;

        let logical = LogicalProperties::new(output_columns, statistics);
        let properties = Properties::new(logical, required);

        let expr = LogicalExpr::Aggregate {
            input: RelNode::from(Operator::new(input, input_properties)),
            aggr_exprs: new_aggr_exprs,
            group_exprs: new_group_exprs,
        };

        (OperatorExpr::from(expr), properties)
    }

    fn build_set_op_metadata(
        &mut self,
        left: RelNode,
        right: RelNode,
        set_op: SetOp,
        all: bool,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        let (left_expr, left_properties) = self.build_rel_node_metadata(left);
        let (right_expr, right_properties) = self.build_rel_node_metadata(right);

        let left_columns = left_properties.logical().output_columns();
        let right_columns = right_properties.logical().output_columns();

        assert_eq!(left_columns.len(), right_columns.len(), "Number of columns does not match");

        let mut output_columns = Vec::with_capacity(left_columns.len());

        for (i, (l, r)) in left_columns.iter().zip(right_columns.iter()).enumerate() {
            let left_column = &self.metadata[l];
            let right_column = &self.metadata[r];
            let left_type = left_column.column.data_type().clone();
            let right_type = right_column.column.data_type().clone();

            assert_eq!(left_type, right_type, "Data type of {}-th does not match. Type casting is not implemented", i);

            let column_id = self.add_column("?column?".into(), left_type);
            output_columns.push(column_id);
        }

        let statistics = properties.logical.statistics().cloned();
        let required = properties.required;

        let logical = LogicalProperties::new(output_columns, statistics);
        let properties = Properties::new(logical, required);

        let expr = match set_op {
            SetOp::Union => LogicalExpr::Union {
                left: RelNode::from(Operator::new(left_expr, left_properties)),
                right: RelNode::from(Operator::new(right_expr, right_properties)),
                all,
            },
            SetOp::Intersect => LogicalExpr::Intersect {
                left: RelNode::from(Operator::new(left_expr, left_properties)),
                right: RelNode::from(Operator::new(right_expr, right_properties)),
                all,
            },
            SetOp::Expect => LogicalExpr::Except {
                left: RelNode::from(Operator::new(left_expr, left_properties)),
                right: RelNode::from(Operator::new(right_expr, right_properties)),
                all,
            },
        };

        (OperatorExpr::from(expr), properties)
    }

    fn add_aggregate_column(&mut self, expr: Expr, _input_properties: &LogicalProperties) -> ColumnId {
        let expr_type = self.resolve_expr_type(&expr);
        let column_name = if let Expr::Aggregate { func, .. } = &expr {
            func.to_string()
        } else {
            "?column?".to_string()
        };
        let column = Column::new(column_name, None, expr_type);
        let column_meta = ColumnMetadata {
            column: ColumnRef::new(column),
            expr: Some(expr),
        };

        let col_id = self.metadata.len() + 1;
        self.metadata.insert(col_id, column_meta);
        col_id
    }

    fn add_column(&mut self, column_name: String, data_type: DataType) -> ColumnId {
        let column = Column::new(column_name, None, data_type);
        let column_meta = ColumnMetadata {
            column: ColumnRef::new(column),
            expr: None,
        };

        let col_id = self.metadata.len() + 1;
        self.metadata.insert(col_id, column_meta);
        col_id
    }

    fn build_rel_node_metadata(&mut self, node: RelNode) -> (OperatorExpr, Properties) {
        match node {
            RelNode::Expr(expr) => self.do_build_metadata(*expr),
            RelNode::Group(_) => panic!("Expected an expression"),
        }
    }

    fn build_scalar_node_metadata(&mut self, scalar: ScalarNode) -> (OperatorExpr, Properties) {
        match scalar {
            ScalarNode::Expr(expr) => {
                let expr = *expr;
                let properties = expr.properties;
                match expr.expr {
                    OperatorExpr::Relational(expr) => panic!("ScalarNode contains a relational expr: {:?}", expr),
                    OperatorExpr::Scalar(scalar_expr) => {
                        let new_scalar_expr = self.build_scalar_expr_metadata(scalar_expr, properties.logical());
                        (OperatorExpr::from(new_scalar_expr), properties)
                    }
                }
            }
            ScalarNode::Group(group) => panic!("Expected an expression but got a group: {:?}", group),
        }
    }

    fn build_scalar_expr_metadata(&mut self, expr: Expr, _input_properties: &LogicalProperties) -> Expr {
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
        let f = |rel_node: RelNode| {
            let (expr, properties) = self.build_rel_node_metadata(rel_node);
            let expr = Operator::new(expr, properties);
            let (group, _) = self.memo.insert(expr);
            RelNode::Group(group)
        };
        let mut rewriter = InternNestedRelExprs { build_metadata: f };
        expr.rewrite(&mut rewriter)
    }

    fn resolve_expr_type(&self, expr: &Expr) -> DataType {
        match expr {
            Expr::Column(column_id) => self.metadata[column_id].column.data_type().clone(),
            Expr::Scalar(value) => value.data_type(),
            Expr::BinaryExpr { lhs, op, rhs } => {
                let left_tpe = self.resolve_expr_type(lhs);
                let right_tpe = self.resolve_expr_type(rhs);
                assert_eq!(left_tpe, right_tpe, "Types does not match");
                op.return_type()
            }
            Expr::Not(expr) => {
                let expr_type = self.resolve_expr_type(expr);
                assert_eq!(expr_type, DataType::Bool, "Invalid argument type for NOT operator");
                expr_type
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
            Expr::SubQuery(expr) => {
                let props = expr.props();
                let columns = props.logical().output_columns();
                assert_eq!(columns.len(), 1, "Subquery must return exactly 1 column. But got {:?}", columns);
                let column_metadata = &self.metadata[&columns[0]];
                column_metadata.column.data_type().clone()
            }
        }
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
    use crate::datatypes::DataType;
    use crate::dump::MetadataBuilder;
    use crate::memo::{format_memo, Memo};
    use crate::meta::ColumnId;
    use crate::operators::expr::{AggregateFunction, BinaryOp, Expr};
    use crate::operators::join::JoinCondition;
    use crate::operators::logical::LogicalExpr;
    use crate::operators::scalar::ScalarValue;
    use crate::operators::{Operator, RelNode, ScalarNode};
    use crate::rules::testing::format_expr;
    use itertools::Itertools;
    use std::collections::BTreeMap;
    use std::iter::FromIterator;

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
  col:3 B.b1 Int32
  col:4 B.b2 Int32
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
  col:3 B.b1 Int32
  col:4 B.b2 Int32
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
  filter: Expr SubQuery 01 > 37
  output cols: [1, 2]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32, expr: count(col:3)
Memo:
  01 LogicalProjection input=00 cols=[5]
  00 LogicalGet B cols=[3]
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
LogicalProjection cols=[1, 5]
  input: LogicalGet A cols=[1, 2]
  output cols: [1, 5]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32, expr: 10
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
  col:6 ?column? Int32, expr: SubQuery 01
Memo:
  01 LogicalProjection input=00 cols=[5]
  00 LogicalGet B cols=[3]
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
  output cols: [5, 6]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 count Int32, expr: count(col:1)
  col:6 max Int32, expr: max(col:1)
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
  output cols: [5]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 count Int32, expr: count(col:1)
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
                source: "A".into(),
                columns: vec![1, 2],
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
  right: LogicalGet A cols=[1, 2]
  output cols: [5, 6]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32
  col:6 ?column? Int32
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
                source: "A".into(),
                columns: vec![1, 2],
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
  right: LogicalGet A cols=[1, 2]
  output cols: [5, 6]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32
  col:6 ?column? Int32
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
                source: "A".into(),
                columns: vec![1, 2],
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
  right: LogicalGet A cols=[1, 2]
  output cols: [5, 6]
Metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32
  col:6 ?column? Int32
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
        let mut builder = MetadataBuilder::new(&mut memo, tables);

        let expr = builder.build_metadata(expr);
        let mut buf = String::new();
        buf.push('\n');

        buf.push_str(format_expr(&expr).as_str());
        buf.push('\n');

        buf.push_str(format!("  output cols: {:?}\n", expr.logical().output_columns()).as_str());
        buf.push_str("Metadata:\n");

        for (id, column) in BTreeMap::from_iter(builder.metadata.into_iter()) {
            let column_name = if column.expr.is_none() && column.column.table().is_none() {
                format!("  col:{} {} {:?}", id, column.column.name(), column.column.data_type())
            } else if column.expr.is_none() {
                format!(
                    "  col:{} {}.{} {:?}",
                    id,
                    column.column.table().unwrap(),
                    column.column.name(),
                    column.column.data_type()
                )
            } else {
                format!(
                    "  col:{} {} {:?}, expr: {}",
                    id,
                    column.column.name(),
                    column.column.data_type(),
                    column.expr.as_ref().unwrap(),
                )
            };
            buf.push_str(column_name.as_str());
            buf.push('\n');
        }

        if !builder.memo.is_empty() {
            buf.push_str("Memo:\n");
            let memo_as_string = format_memo(builder.memo);
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
