use crate::error::OptimizerError;
use crate::operators::relational::logical::{LogicalExcept, LogicalExpr, LogicalIntersect, LogicalUnion};
use crate::operators::relational::physical::{Append, HashedSetOp, PhysicalExpr, Unique};
use crate::rules::{Rule, RuleContext, RuleMatch, RuleResult, RuleType};

#[derive(Debug)]
pub struct UnionRule;

impl Rule for UnionRule {
    fn name(&self) -> String {
        "UnionRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Union { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        if let LogicalExpr::Union(LogicalUnion {
            left,
            right,
            all,
            columns,
        }) = expr
        {
            let expr = if *all {
                PhysicalExpr::Append(Append {
                    left: left.clone(),
                    right: right.clone(),
                    columns: columns.clone(),
                })
            } else {
                let required_ordering = ctx.required_properties().and_then(|r| r.ordering());
                let inputs = &[left.clone(), right.clone()];
                let ordering = Unique::derive_input_orderings(required_ordering, inputs, columns);

                PhysicalExpr::Unique(Unique {
                    inputs: vec![left.clone(), right.clone()],
                    on_expr: None,
                    columns: columns.clone(),
                    ordering,
                })
            };
            Ok(Some(RuleResult::Implementation(expr)))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug)]
pub struct HashSetOpRule;

impl Rule for HashSetOpRule {
    fn name(&self) -> String {
        "HashSetOpRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Intersect { .. } | LogicalExpr::Except { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        let expr = match expr {
            LogicalExpr::Intersect(LogicalIntersect {
                left,
                right,
                all,
                columns,
            }) => PhysicalExpr::HashedSetOp(HashedSetOp {
                left: left.clone(),
                right: right.clone(),
                intersect: true,
                all: *all,
                columns: columns.clone(),
            }),
            LogicalExpr::Except(LogicalExcept {
                left,
                right,
                all,
                columns,
            }) => PhysicalExpr::HashedSetOp(HashedSetOp {
                left: left.clone(),
                right: right.clone(),
                intersect: false,
                all: *all,
                columns: columns.clone(),
            }),
            _ => return Ok(None),
        };
        Ok(Some(RuleResult::Implementation(expr)))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::meta::testing::TestMetadata;
    use crate::rules::testing::{new_src, RuleTester};

    #[test]
    fn test_sorted_inputs_union() {
        let mut tester = RuleTester::new(UnionRule);
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);

        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();

        let col3 = metadata.column("B").build();
        let col4 = metadata.column("B").build();

        let col5 = metadata.synthetic_column().build();
        let col6 = metadata.synthetic_column().build();

        let union_expr = |all: bool| {
            LogicalExpr::Union(LogicalUnion {
                left: new_src("A", vec![col1, col2]),
                right: new_src("B", vec![col3, col4]),
                all,
                columns: vec![col5, col6],
            })
        };

        let union = union_expr(false);
        tester.apply(
            &union,
            r#"
Unique cols=[5, 6] left_ord=[+1, +2] right_ord=[+3, +4]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]    
"#,
        );

        let union_all = union_expr(true);
        tester.apply(
            &union_all,
            r#"
Append cols=[5, 6]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]    
"#,
        );
    }

    #[test]
    fn test_intercept_hashset_op() {
        let mut tester = RuleTester::new(HashSetOpRule);
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);

        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();

        let col3 = metadata.column("B").build();
        let col4 = metadata.column("B").build();

        let col5 = metadata.synthetic_column().build();
        let col6 = metadata.synthetic_column().build();

        let intersect_expr = |all: bool| {
            LogicalExpr::Intersect(LogicalIntersect {
                left: new_src("A", vec![col1, col2]),
                right: new_src("B", vec![col3, col4]),
                all,
                columns: vec![col5, col6],
            })
        };

        let intersect = intersect_expr(false);
        tester.apply(
            &intersect,
            r#"
HashedSetOp intersect=true all=false cols=[5, 6]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
    "#,
        );

        let intersect_all = intersect_expr(true);
        tester.apply(
            &intersect_all,
            r#"
HashedSetOp intersect=true all=true cols=[5, 6]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
    "#,
        );
    }

    #[test]
    fn test_except_hashset_op() {
        let mut tester = RuleTester::new(HashSetOpRule);
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);

        let col1 = metadata.column("A").build();
        let col2 = metadata.column("A").build();

        let col3 = metadata.column("B").build();
        let col4 = metadata.column("B").build();

        let col5 = metadata.synthetic_column().build();
        let col6 = metadata.synthetic_column().build();

        let except_expr = |all: bool| {
            LogicalExpr::Except(LogicalExcept {
                left: new_src("A", vec![col1, col2]),
                right: new_src("B", vec![col3, col4]),
                all,
                columns: vec![col5, col6],
            })
        };

        let expect = except_expr(false);
        tester.apply(
            &expect,
            r#"
HashedSetOp intersect=false all=false cols=[5, 6]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
    "#,
        );

        let expect_all = except_expr(true);
        tester.apply(
            &expect_all,
            r#"
HashedSetOp intersect=false all=true cols=[5, 6]
  left: LogicalGet A cols=[1, 2]
  right: LogicalGet B cols=[3, 4]
    "#,
        );
    }
}
