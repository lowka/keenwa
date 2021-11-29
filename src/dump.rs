use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{Catalog, CatalogRef, Column, ColumnRef, TableBuilder};
use crate::datatypes::DataType;
use crate::memo::Memo;
use crate::meta::ColumnId;
use crate::operators::expr::Expr;
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
    fn new(memo: &'a mut Memo<Operator>, data: Vec<(String, Vec<(String, DataType)>)>) -> Self {
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

    fn build_metadata(&mut self, expr: Operator) -> Operator {
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
                LogicalExpr::Aggregate { .. } => todo!(),
                LogicalExpr::Join { left, right, condition } => {
                    self.build_join_metadata(left, right, condition, properties)
                }
                LogicalExpr::Get { source, columns } => self.build_get_metadata(source, columns, properties),
                LogicalExpr::Union { .. } => todo!(),
                LogicalExpr::Intersect { .. } => todo!(),
                LogicalExpr::Except { .. } => todo!(),
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

        let output_columns = columns.clone();
        let statistics = properties.logical.statistics().cloned();

        let logical = LogicalProperties::new(output_columns, statistics);
        let required = properties.required.clone();

        let expr = LogicalExpr::Get { source, columns };
        let properties = Properties::new(logical, required);
        (OperatorExpr::from(expr), properties)
    }

    fn build_select_metadata(
        &mut self,
        input: RelNode,
        filter: Option<ScalarNode>,
        properties: Properties,
    ) -> (OperatorExpr, Properties) {
        todo!()
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
                let exists = input_logical.output_columns().contains(column_id);

                assert!(
                    exists,
                    "Unexpected output column: {}. Input columns: {:?}",
                    column_id,
                    input_logical.output_columns()
                )
            }

            let expr = LogicalExpr::Projection {
                input: RelNode::from(Operator::new(input, input_properties)),
                exprs: vec![],
                columns: columns.clone(),
            };

            let statistics = properties.logical.statistics().cloned();
            let logical = LogicalProperties::new(columns, statistics);
            let properties = Properties::new(logical, properties.required);

            (expr, properties)
        } else {
            let mut columns = Vec::with_capacity(exprs.len());
            for expr in exprs {
                match expr {
                    Expr::Column(column_id) => {
                        let exists = input_properties.logical.output_columns().contains(&column_id);
                        assert!(
                            exists,
                            "Unexpected output column: {}. Input columns: {:?}",
                            column_id,
                            input_logical.output_columns()
                        );
                        columns.push(column_id);
                    }
                    _ => {
                        let expr = if let Expr::SubQuery(rel_node) = expr {
                            let group = match rel_node {
                                RelNode::Expr(expr) => {
                                    let (group, expr_id) = self.memo.insert(*expr);
                                    group
                                }
                                RelNode::Group(group) => group,
                            };
                            Expr::SubQuery(RelNode::Group(group))
                        } else {
                            expr
                        };

                        let column = Column::new("?column?".to_string(), None, DataType::Int32);
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
            let logical = LogicalProperties::new(columns, statistics);
            let properties = Properties::new(logical, properties.required);

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

        // match condition {
        //     JoinCondition::Using(using) => {}
        // }

        let mut output_columns = left_properties.logical.output_columns().to_vec();
        output_columns.extend_from_slice(right_properties.logical.output_columns());

        let statistics = properties.logical.statistics().cloned();
        let logical = LogicalProperties::new(output_columns, statistics);
        let properties = Properties::new(logical, properties.required);

        let expr = LogicalExpr::Join {
            left: RelNode::from(Operator::new(left, left_properties)),
            right: RelNode::from(Operator::new(right, right_properties)),
            condition,
        };

        (OperatorExpr::from(expr), properties)
    }

    fn build_rel_node_metadata(&mut self, node: RelNode) -> (OperatorExpr, Properties) {
        match node {
            RelNode::Expr(expr) => self.do_build_metadata(*expr),
            RelNode::Group(_) => panic!("Expected an expression"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::datatypes::DataType;
    use crate::dump::MetadataBuilder;
    use crate::memo::{Memo, MemoExpr, StringMemoFormatter};
    use crate::meta::ColumnId;
    use crate::operators::expr::Expr;
    use crate::operators::join::JoinCondition;
    use crate::operators::logical::LogicalExpr;
    use crate::operators::scalar::ScalarValue;
    use crate::operators::{Operator, RelNode};
    use crate::rules::testing::format_expr;
    use std::collections::{BTreeMap, HashMap};
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
metadata:
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
metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
"#,
        )
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
metadata:
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
        let expr = LogicalExpr::Get {
            source: "A".into(),
            columns: vec![1, 2],
        }
        .to_operator();

        let rel_node = RelNode::Expr(Box::new(expr));

        projection_test(
            vec![],
            vec![Expr::Column(1), Expr::SubQuery(rel_node)],
            r#"
LogicalProjection cols=[1, 5]
  input: LogicalGet A cols=[1, 2]
  output cols: [1, 5]
metadata:
  col:1 A.a1 Int32
  col:2 A.a2 Int32
  col:3 B.b1 Int32
  col:4 B.b2 Int32
  col:5 ?column? Int32, expr: SubQuery 00
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

        buf.push_str("metadata:\n");

        for (id, column) in BTreeMap::from_iter(builder.metadata.into_iter()) {
            let s = if column.expr.is_none() {
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
            buf.push_str(s.as_str());
            buf.push('\n');
        }
        assert_eq!(expected, buf.as_str())
    }
}
