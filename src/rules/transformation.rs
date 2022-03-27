//! Transformation rules. See [rules module](super).

use crate::error::OptimizerError;
use crate::operators::relational::join::{get_non_empty_join_columns_pair, JoinCondition, JoinType};
use crate::operators::relational::logical::{LogicalExpr, LogicalJoin};
use crate::rules::{Rule, RuleContext, RuleMatch, RuleResult, RuleType};

#[derive(Debug)]
pub struct JoinCommutativityRule;

impl Rule for JoinCommutativityRule {
    fn name(&self) -> String {
        "JoinCommutativityRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Transformation
    }

    fn matches(&self, _ctx: &RuleContext, operator: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(
            operator,
            LogicalExpr::Join(LogicalJoin {
                join_type: JoinType::Inner,
                ..
            })
        ) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join(LogicalJoin {
                join_type,
                left,
                right,
                condition,
            }) => {
                // do not replace ON condition.
                if let Some((left_columns, right_columns)) = get_non_empty_join_columns_pair(left, right, condition) {
                    let expr = LogicalExpr::Join(LogicalJoin {
                        join_type: join_type.clone(),
                        left: right.clone(),
                        right: left.clone(),
                        condition: JoinCondition::using(
                            right_columns.into_iter().zip(left_columns.into_iter()).collect(),
                        ),
                    });

                    Ok(Some(RuleResult::Substitute(expr)))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
pub struct JoinAssociativityRule;

impl JoinAssociativityRule {
    // [AxB]xC -> Ax[BxC]
    fn left_side_condition(
        top_condition: &JoinCondition,
        inner_condition: &JoinCondition,
    ) -> Option<(JoinCondition, JoinCondition)> {
        let (_, top_right_columns) = match top_condition {
            JoinCondition::Using(using) => using.get_columns_pair(),
            _ => return None,
        };

        let (inner_left_columns, inner_right_columns) = match inner_condition {
            JoinCondition::Using(using) => using.get_columns_pair(),
            JoinCondition::On(_) => return None,
        };

        let inner =
            JoinCondition::using(inner_right_columns.clone().into_iter().zip(top_right_columns.into_iter()).collect());

        let top = JoinCondition::using(inner_left_columns.into_iter().zip(inner_right_columns.into_iter()).collect());

        Some((top, inner))
    }

    fn right_side_condition(
        top_condition: &JoinCondition,
        inner_condition: &JoinCondition,
    ) -> Option<(JoinCondition, JoinCondition)> {
        let (top_left_columns, _) = match top_condition {
            JoinCondition::Using(using) => using.get_columns_pair(),
            _ => return None,
        };

        let (inner_left_columns, inner_right_columns) = match inner_condition {
            JoinCondition::Using(using) => using.get_columns_pair(),
            JoinCondition::On(_) => return None,
        };

        let inner =
            JoinCondition::using(top_left_columns.clone().into_iter().zip(inner_left_columns.into_iter()).collect());

        let top = JoinCondition::using(top_left_columns.into_iter().zip(inner_right_columns.into_iter()).collect());

        Some((top, inner))
    }
}

impl Rule for JoinAssociativityRule {
    fn name(&self) -> String {
        "JoinAssociativityRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Transformation
    }

    fn group_rule(&self) -> bool {
        true
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(
            expr,
            LogicalExpr::Join(LogicalJoin {
                join_type: JoinType::Inner,
                condition: JoinCondition::Using(..),
                ..
            })
        ) {
            Some(RuleMatch::Group)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        if let LogicalExpr::Join(LogicalJoin {
            join_type: JoinType::Inner,
            left: top_left,
            right: top_right,
            condition: top_condition,
        }) = expr
        {
            match (top_left.expr().logical(), top_right.expr().logical()) {
                // [AxB]xC -> Ax[BxC]
                (
                    LogicalExpr::Join(LogicalJoin {
                        join_type: JoinType::Inner,
                        left: inner_left,
                        right: inner_right,
                        condition: inner_condition,
                    }),
                    _,
                ) => {
                    if let Some((new_top_condition, new_inner_condition)) =
                        Self::left_side_condition(top_condition, inner_condition)
                    {
                        let expr = LogicalExpr::Join(LogicalJoin {
                            join_type: JoinType::Inner,
                            left: inner_left.clone(),
                            right: LogicalExpr::Join(LogicalJoin {
                                join_type: JoinType::Inner,
                                left: inner_right.clone(),
                                right: top_right.clone(),
                                condition: new_inner_condition,
                            })
                            .into(),
                            condition: new_top_condition,
                        });
                        return Ok(Some(RuleResult::Substitute(expr)));
                    }
                }
                // Ax[BxC] -> [AxB]xC
                (
                    _,
                    LogicalExpr::Join(LogicalJoin {
                        join_type: JoinType::Inner,
                        left: inner_left,
                        right: inner_right,
                        condition: inner_condition,
                    }),
                ) => {
                    if let Some((new_top_condition, new_inner_condition)) =
                        Self::right_side_condition(top_condition, inner_condition)
                    {
                        let expr = LogicalExpr::Join(LogicalJoin {
                            join_type: JoinType::Inner,
                            left: LogicalExpr::Join(LogicalJoin {
                                join_type: JoinType::Inner,
                                left: top_left.clone(),
                                right: inner_left.clone(),
                                condition: new_inner_condition,
                            })
                            .into(),
                            right: inner_right.clone(),
                            condition: new_top_condition,
                        });

                        return Ok(Some(RuleResult::Substitute(expr)));
                    }
                }
                _ => {}
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::operators::relational::logical::LogicalGet;
    use crate::rules::testing::RuleTester;

    use super::*;

    #[test]
    fn test_join_commutativity_rule() {
        let expr = LogicalExpr::Join(LogicalJoin {
            join_type: JoinType::Inner,
            left: LogicalExpr::Get(LogicalGet {
                source: "A".into(),
                columns: vec![1, 2],
            })
            .into(),
            right: LogicalExpr::Get(LogicalGet {
                source: "B".into(),
                columns: vec![3, 4],
            })
            .into(),
            condition: JoinCondition::using(vec![(1, 3)]),
        });

        let mut tester = RuleTester::new(JoinCommutativityRule);
        tester.apply(
            &expr,
            r#"
LogicalJoin type=Inner using=[(3, 1)]
  left: LogicalGet B cols=[3, 4]
  right: LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_join_associativity_rule1() {
        let expr = LogicalExpr::Join(LogicalJoin {
            join_type: JoinType::Inner,
            left: LogicalExpr::Join(LogicalJoin {
                join_type: JoinType::Inner,
                left: LogicalExpr::Get(LogicalGet {
                    source: "A".into(),
                    columns: vec![1, 2],
                })
                .into(),
                right: LogicalExpr::Get(LogicalGet {
                    source: "B".into(),
                    columns: vec![3, 4],
                })
                .into(),
                condition: JoinCondition::using(vec![(1, 4)]),
            })
            .into(),
            right: LogicalExpr::Get(LogicalGet {
                source: "C".into(),
                columns: vec![5, 6],
            })
            .into(),
            condition: JoinCondition::using(vec![(1, 6)]),
        });
        // [AxB(1,4)]xC(1,6) => A(1,6)x[BxC(4,6)]

        let mut tester = RuleTester::new(JoinAssociativityRule);
        tester.apply(
            &expr,
            r#"
LogicalJoin type=Inner using=[(1, 4)]
  left: LogicalGet A cols=[1, 2]
  right: LogicalJoin type=Inner using=[(4, 6)]
    left: LogicalGet B cols=[3, 4]
    right: LogicalGet C cols=[5, 6]
"#,
        );
    }

    #[test]
    fn test_join_associativity_rule2() {
        let expr = LogicalExpr::Join(LogicalJoin {
            join_type: JoinType::Inner,
            left: LogicalExpr::Get(LogicalGet {
                source: "A".into(),
                columns: vec![1, 2],
            })
            .into(),
            right: LogicalExpr::Join(LogicalJoin {
                join_type: JoinType::Inner,
                left: LogicalExpr::Get(LogicalGet {
                    source: "B".into(),
                    columns: vec![3, 4],
                })
                .into(),
                right: LogicalExpr::Get(LogicalGet {
                    source: "C".into(),
                    columns: vec![5, 6],
                })
                .into(),
                condition: JoinCondition::using(vec![(3, 6)]),
            })
            .into(),
            condition: JoinCondition::using(vec![(1, 3)]),
        });
        // A(1,3)x[BxC(3,6)] => [AxB(1,3)]xC(1,6)

        let mut tester = RuleTester::new(JoinAssociativityRule);
        tester.apply(
            &expr,
            r#"
LogicalJoin type=Inner using=[(1, 6)]
  left: LogicalJoin type=Inner using=[(1, 3)]
    left: LogicalGet A cols=[1, 2]
    right: LogicalGet B cols=[3, 4]
  right: LogicalGet C cols=[5, 6]
"#,
        );
    }
}
