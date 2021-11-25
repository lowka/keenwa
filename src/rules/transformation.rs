use crate::error::OptimizerError;
use crate::operators::join::JoinCondition;
use crate::operators::logical::LogicalExpr;
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
        if matches!(operator, LogicalExpr::Join { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join { left, right, condition } => {
                let (left_columns, right_columns) = match condition {
                    JoinCondition::Using(using) => using.get_columns_pair(),
                };

                let expr = LogicalExpr::Join {
                    left: right.clone(),
                    right: left.clone(),
                    condition: JoinCondition::using(right_columns.into_iter().zip(left_columns.into_iter()).collect()),
                };
                Ok(Some(RuleResult::Substitute(expr)))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
pub struct JoinAssociativityRule;

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
        if matches!(expr, LogicalExpr::Join { .. }) {
            Some(RuleMatch::Group)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        match expr {
            LogicalExpr::Join {
                left: top_left,
                right: top_right,
                condition: top_condition,
            } => {
                let (top_left_columns, top_right_columns) = match top_condition {
                    JoinCondition::Using(using) => using.get_columns_pair(),
                };

                match (top_left.expr().as_logical(), top_right.expr().as_logical()) {
                    // [AxB]xC -> Ax[BxC]
                    (
                        LogicalExpr::Join {
                            left: inner_left,
                            right: inner_right,
                            condition: inner_condition,
                        },
                        _,
                    ) => {
                        let (inner_left_columns, inner_right_columns) = match inner_condition {
                            JoinCondition::Using(using) => using.get_columns_pair(),
                        };

                        let expr = LogicalExpr::Join {
                            left: inner_left.clone(),
                            right: LogicalExpr::Join {
                                left: inner_right.clone(),
                                right: top_right.clone(),
                                condition: JoinCondition::using(
                                    inner_right_columns
                                        .clone()
                                        .into_iter()
                                        .zip(top_right_columns.into_iter())
                                        .collect(),
                                ),
                            }
                            .into(),
                            condition: JoinCondition::using(
                                inner_left_columns.into_iter().zip(inner_right_columns.into_iter()).collect(),
                            ),
                        };
                        return Ok(Some(RuleResult::Substitute(expr)));
                    }
                    // Ax[BxC] -> [AxB]xC
                    (
                        _,
                        LogicalExpr::Join {
                            left: inner_left,
                            right: inner_right,
                            condition: inner_condition,
                        },
                    ) => {
                        let (inner_left_columns, inner_right_columns) = match inner_condition {
                            JoinCondition::Using(using) => using.get_columns_pair(),
                        };
                        let expr = LogicalExpr::Join {
                            left: LogicalExpr::Join {
                                left: top_left.clone(),
                                right: inner_left.clone(),
                                condition: JoinCondition::using(
                                    top_left_columns.clone().into_iter().zip(inner_left_columns.into_iter()).collect(),
                                ),
                            }
                            .into(),
                            right: inner_right.clone(),
                            condition: JoinCondition::using(
                                top_left_columns.into_iter().zip(inner_right_columns.into_iter()).collect(),
                            ),
                        };
                        return Ok(Some(RuleResult::Substitute(expr)));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::testing::RuleTester;

    #[test]
    fn test_join_commutativity_rule() {
        let expr = LogicalExpr::Join {
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
            condition: JoinCondition::using(vec![(1, 3)]),
        };

        let mut tester = RuleTester::new(JoinCommutativityRule);
        tester.apply(
            &expr,
            r#"
LogicalJoin using=[(3, 1)]
  left: LogicalGet B cols=[3, 4]
  right: LogicalGet A cols=[1, 2]
"#,
        );
    }

    #[test]
    fn test_join_associativity_rule1() {
        let expr = LogicalExpr::Join {
            left: LogicalExpr::Join {
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
                condition: JoinCondition::using(vec![(1, 4)]),
            }
            .into(),
            right: LogicalExpr::Get {
                source: "C".into(),
                columns: vec![5, 6],
            }
            .into(),
            condition: JoinCondition::using(vec![(1, 6)]),
        };
        // [AxB(1,4)]xC(1,6) => A(1,6)x[BxC(4,6)]

        let mut tester = RuleTester::new(JoinAssociativityRule);
        tester.apply(
            &expr,
            r#"
LogicalJoin using=[(1, 4)]
  left: LogicalGet A cols=[1, 2]
  right: LogicalJoin using=[(4, 6)]
      left: LogicalGet B cols=[3, 4]
      right: LogicalGet C cols=[5, 6]
"#,
        );
    }

    #[test]
    fn test_join_associativity_rule2() {
        let expr = LogicalExpr::Join {
            left: LogicalExpr::Get {
                source: "A".into(),
                columns: vec![1, 2],
            }
            .into(),
            right: LogicalExpr::Join {
                left: LogicalExpr::Get {
                    source: "B".into(),
                    columns: vec![3, 4],
                }
                .into(),
                right: LogicalExpr::Get {
                    source: "C".into(),
                    columns: vec![5, 6],
                }
                .into(),
                condition: JoinCondition::using(vec![(3, 6)]),
            }
            .into(),
            condition: JoinCondition::using(vec![(1, 3)]),
        };
        // A(1,3)x[BxC(3,6)] => [AxB(1,3)]xC(1,6)

        let mut tester = RuleTester::new(JoinAssociativityRule);
        tester.apply(
            &expr,
            r#"
LogicalJoin using=[(1, 6)]
  left: LogicalJoin using=[(1, 3)]
      left: LogicalGet A cols=[1, 2]
      right: LogicalGet B cols=[3, 4]
  right: LogicalGet C cols=[5, 6]
"#,
        );
    }
}
