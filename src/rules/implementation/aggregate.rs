use crate::error::OptimizerError;
use crate::operators::relational::logical::{LogicalAggregate, LogicalExpr};
use crate::operators::relational::physical::{HashAggregate, PhysicalExpr, StreamingAggregate};
use crate::rules::{Rule, RuleContext, RuleMatch, RuleResult, RuleType};

#[derive(Debug)]
pub struct HashAggregateRule;

impl Rule for HashAggregateRule {
    fn name(&self) -> String {
        "HashAggregateRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Aggregate { .. }) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        if let LogicalExpr::Aggregate(LogicalAggregate {
            input,
            aggr_exprs,
            group_exprs,
            having,
            columns,
        }) = expr
        {
            let expr = PhysicalExpr::HashAggregate(HashAggregate {
                input: input.clone(),
                aggr_exprs: aggr_exprs.clone(),
                group_exprs: group_exprs.clone(),
                having: having.clone(),
                columns: columns.clone(),
            });
            Ok(Some(RuleResult::Implementation(expr)))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug)]
pub struct StreamingAggregateRule;

impl Rule for StreamingAggregateRule {
    fn name(&self) -> String {
        "StreamingAggregateRule".into()
    }

    fn rule_type(&self) -> RuleType {
        RuleType::Implementation
    }

    fn matches(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Option<RuleMatch> {
        if matches!(expr, LogicalExpr::Aggregate(_)) {
            Some(RuleMatch::Expr)
        } else {
            None
        }
    }

    fn apply(&self, _ctx: &RuleContext, expr: &LogicalExpr) -> Result<Option<RuleResult>, OptimizerError> {
        if let LogicalExpr::Aggregate(LogicalAggregate {
            input,
            aggr_exprs,
            group_exprs,
            having,
            columns,
        }) = expr
        {
            if let Some(ordering) = StreamingAggregate::derive_input_ordering(group_exprs) {
                let expr = PhysicalExpr::StreamingAggregate(StreamingAggregate {
                    input: input.clone(),
                    aggr_exprs: aggr_exprs.clone(),
                    group_exprs: group_exprs.clone(),
                    having: having.clone(),
                    columns: columns.clone(),
                    ordering,
                });
                Ok(Some(RuleResult::Implementation(expr)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}
