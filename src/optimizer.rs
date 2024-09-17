//! Cost-based optimizer based on the Cascades Optimizer Framework paper.
//!
//! Basic algorithm:
//!
//! ```text
//!  - schedule OptimizeGroup for the top operator of an operator tree.
//!   
//!  - [OptimizeGroup]: for every expression in the group:
//!       - if the expression is a relational expression:
//!          schedule OptimizeExpr
//!       - if the expression is a scalar expression w/o subqueries:
//!          mark group as optimised.
//!       - if the expression is a scalar expression with subquery:
//!          schedule OptimizeGroup for that subquery.
//!
//!       Also schedule [ExploreGroup] in order to search for alternative plans.
//!
//!  - [OptimizeExpr]: select rules available for that expression,
//!       schedule ApplyRule for each (rule, expression).  
//!
//!  - [ApplyRule]:
//!      - if a rule is a transformation rule schedule OptimizeExpr for
//!        the logical expression it produced.
//!      - if a rule is an implementation rule schedule OptimizeInputs for
//!        the physical expression it produced.
//!
//!  - [OptimizeInputs]:
//!       - Schedule OptimizeInputs*,
//!       - Schedule OptimizeGroup for every input expression.
//!       - When all inputs has been optimized - compute the cost of the expression
//!       and add it the candidate list. If the expression has the lowest cost
//!       make it the best expression in the group.
//!
//!  * - We need [OptimizeInputs] when all inputs have been optimized.
//! ```
//!

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display, Formatter};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use itertools::Itertools;

use crate::cost::{Cost, CostEstimationContext, CostEstimator, COST_PER_OPERATOR};
use crate::error::OptimizerError;
use crate::memo::{format_memo, ExprId, GroupId, MemoExpr, NewChildExprs, Props};
use crate::meta::MetadataRef;
use crate::operators::relational::physical::PhysicalExpr;
use crate::operators::relational::{RelExpr, RelNode};
use crate::operators::scalar::{expr_with_new_inputs, ScalarExpr};
use crate::operators::{ExprMemo, ExprRef, Operator, OperatorExpr, OuterScope, Properties};
use crate::properties::physical::RequiredProperties;
use crate::rules::{
    DerivePropertyMode, PhysicalPropertiesProvider, RequiredPropertiesQueue, RuleContext, RuleId, RuleMatch,
    RuleResult, RuleSet, RuleType,
};

/// Cost-based optimizer. See [module docs](self) for an overview of the algorithm.
pub struct Optimizer<R, T, C = DefaultOptimizedExprCallback> {
    rule_set: Arc<R>,
    cost_estimator: Arc<T>,
    result_callback: Arc<C>,
}

impl<R, T> Optimizer<R, T> {
    /// Creates a new instance of `Optimizer`.
    pub fn new(rule_set: Arc<R>, cost_estimator: Arc<T>) -> Self {
        Optimizer {
            rule_set,
            cost_estimator,
            result_callback: Arc::new(DefaultOptimizedExprCallback),
        }
    }
}

impl<R, T, C> Optimizer<R, T, C>
where
    R: RuleSet,
    T: CostEstimator,
    C: OptimizedExprCallback,
{
    /// Creates a new instance of `Optimizer` with the specified [OptimizedExprCallback].
    pub fn with_callback(rule_set: Arc<R>, cost_estimator: Arc<T>, result_callback: Arc<C>) -> Self {
        Optimizer {
            rule_set,
            cost_estimator,
            result_callback,
        }
    }

    /// Optimizes the given operator tree.
    pub fn optimize(&self, expr: Operator, memo: &mut ExprMemo) -> Result<Operator, OptimizerError> {
        let mut runtime_state = RuntimeState::new();

        log::debug!("Optimizing expression: {:?}", expr);

        let required_property = match expr.props() {
            Properties::Scalar(_) => None,
            Properties::Relational(props) => props.physical.required.clone(),
        };
        let scope = OuterScope::from_properties(expr.props());
        let root_expr = memo.insert_group(expr, &scope)?;
        let root_group_id = root_expr.state().memo_group_id();

        let ctx = OptimizationContext {
            group_id: root_group_id,
            required_properties: Rc::new(required_property),
        };

        log::debug!("Initial ctx: {}, initial memo:\n{}", ctx, format_memo(memo));

        runtime_state.tasks.schedule(Task::OptimizeGroup { ctx: ctx.clone() });

        self.do_optimize(&mut runtime_state, memo)?;

        let state = runtime_state.state.get_state(&ctx)?;
        assert!(state.optimized, "Root node has not been optimized: {:?}", state);

        log::debug!("Final memo:\n{}", format_memo(memo));

        self.build_result(runtime_state, memo, &ctx)
    }

    fn do_optimize(&self, runtime_state: &mut RuntimeState, memo: &mut ExprMemo) -> Result<(), OptimizerError> {
        let start_time = Instant::now();

        while let Some(task) = runtime_state.tasks.retrieve() {
            log::debug!("{}", &task);

            runtime_state.stats.number_of_tasks += 1;
            runtime_state.stats.max_stack_depth =
                runtime_state.stats.max_stack_depth.max(runtime_state.tasks.len() + 1);

            match task {
                Task::OptimizeGroup { ctx } => {
                    optimize_groups(runtime_state, memo, ctx, self.rule_set.as_ref())?;
                }
                Task::GroupOptimized { ctx } => {
                    group_optimized(runtime_state, memo, ctx)?;
                }
                Task::OptimizeExpr { ctx, expr, explore } => {
                    optimize_expr(runtime_state, ctx, expr, explore, self.rule_set.as_ref(), memo.metadata().get_ref());
                }
                Task::ApplyRule {
                    ctx,
                    expr,
                    binding,
                    explore,
                } => {
                    apply_rule(runtime_state, memo, ctx, expr, binding, explore, self.rule_set.as_ref())?;
                }
                Task::EnforceProperties { ctx, enforcer_task } => {
                    enforce_properties(runtime_state, memo, ctx, enforcer_task, self.rule_set.as_ref())?;
                }
                Task::OptimizeInputs { ctx, expr, inputs } => {
                    optimize_inputs(runtime_state, ctx, expr, inputs, self.cost_estimator.as_ref())?;
                }
                Task::ExprOptimized { ctx, expr } => {
                    expr_optimized(runtime_state, ctx, expr);
                }
                Task::ExploreGroup { ctx } => {
                    explore_group(runtime_state, memo, ctx, self.rule_set.as_ref())?;
                }
            }
        }

        runtime_state.stats.optimization_time = start_time.elapsed();
        Ok(())
    }

    fn build_result(
        &self,
        runtime_state: RuntimeState,
        memo: &ExprMemo,
        ctx: &OptimizationContext,
    ) -> Result<Operator, OptimizerError> {
        let mut stats = runtime_state.stats;
        let state = runtime_state.state;

        let result_time = Instant::now();
        let expr = copy_out_best_expr(self.result_callback.as_ref(), &state, memo, ctx)?;
        stats.result_time = result_time.elapsed();

        log::debug!("Stats: {:?}", stats);

        Ok(expr)
    }
}

impl<R, T, C> Debug for Optimizer<R, T, C>
where
    R: RuleSet + Debug,
    T: CostEstimator + Debug,
    C: OptimizedExprCallback + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Optimizer")
            .field("rule_set", self.rule_set.as_ref())
            .field("cost_estimator", self.cost_estimator.as_ref())
            .field("result_callback", self.result_callback.as_ref())
            .finish()
    }
}

/// A reference to the best expression in a memo-group.
#[derive(Debug, Clone)]
pub enum BestExprRef<'a> {
    /// A relational expression.
    Relational(&'a PhysicalExpr),
    /// A scalar expression.
    Scalar(&'a ScalarExpr),
}

/// A callback called by the optimizer when an optimized plan is being built.
pub trait OptimizedExprCallback {
    /// Called for each expression in the optimized plan.
    fn on_best_expr<C>(&self, expr: BestExprRef, ctx: &C)
    where
        C: BestExprContext;
}

/// Provides additional information about an expression chosen by the optimizer as the best expression.
pub trait BestExprContext {
    /// Returns the cost of the expression.
    fn cost(&self) -> Cost;

    /// Returns properties of the expression.
    fn props(&self) -> &Properties;

    /// Returns the identifier of a group the expression belongs to.
    fn group_id(&self) -> GroupId;

    /// Returns the number of child expressions.
    fn num_children(&self) -> usize;

    /// Returns the identifier of a group of the i-th child expression.
    fn child_group_id(&self, i: usize) -> GroupId;

    /// Returns physical properties required by the i-th child expression.
    fn child_required(&self, i: usize) -> Option<&RequiredProperties>;
}

// TODO: Add search bounds
#[derive(Debug)]
enum Task {
    OptimizeGroup {
        ctx: OptimizationContext,
    },
    GroupOptimized {
        ctx: OptimizationContext,
    },
    OptimizeExpr {
        ctx: OptimizationContext,
        expr: ExprRef,
        explore: bool,
    },
    ApplyRule {
        ctx: OptimizationContext,
        expr: ExprRef,
        binding: RuleBinding,
        explore: bool,
    },
    EnforceProperties {
        ctx: OptimizationContext,
        enforcer_task: EnforcerTask,
    },
    OptimizeInputs {
        ctx: OptimizationContext,
        expr: ExprRef,
        inputs: OptimizeInputsState,
    },
    ExprOptimized {
        ctx: OptimizationContext,
        expr: ExprRef,
    },
    ExploreGroup {
        ctx: OptimizationContext,
    },
}

impl Display for Task {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Task::OptimizeGroup { ctx } => write!(f, "OptimizeGroup: {}", ctx),
            Task::GroupOptimized { ctx } => write!(f, "GroupOptimized: {}", ctx),
            Task::OptimizeExpr { ctx, expr, explore } => {
                write!(f, "OptimizeExpr: {} expr: {} explore: {}", ctx, expr, explore)
            }
            Task::ApplyRule {
                ctx,
                expr,
                explore,
                binding,
            } => {
                write!(f, "ApplyRule: {} expr: {} rule_id: {} explore: {}", ctx, expr, binding.rule_id, explore)
            }
            Task::OptimizeInputs { ctx, expr, inputs } => {
                write!(f, "OptimizeInputs: {} expr: {}, inputs: {}", ctx, expr, inputs)
            }
            Task::EnforceProperties { ctx, enforcer_task } => {
                write!(f, "EnforceProperties: {} {}", ctx, enforcer_task)
            }
            Task::ExprOptimized { ctx, expr } => {
                write!(f, "ExprOptimized: {} expr: {}", ctx, expr)
            }
            Task::ExploreGroup { ctx } => write!(f, "ExploreGroup: {}", ctx),
        }
    }
}

#[derive(Debug)]
enum EnforcerTask {
    /// Enforcer operator is used to provide physical properties required by an optimization context.
    EnforceProperties(EnforcePropertiesTask),
    /// Explore alternative plan using an enforcer operator as a root of a plan subtree.
    ExploreAlternatives,
}

impl Display for EnforcerTask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EnforcerTask::EnforceProperties(expr) => write!(f, "enforce properties: {}", expr),
            EnforcerTask::ExploreAlternatives => write!(f, "explore alternative plans"),
        }
    }
}

#[derive(Debug)]
enum EnforcePropertiesTask {
    /// Initialises task that will provide required properties.
    Prepare(ExprRef),
    /// Executes that task.
    Execute {
        expr: ExprRef,
        input: ExprRef,
        queue: RequiredPropertiesQueue,
    },
    /// Called when task completes.
    Done(ExprRef),
}

impl Display for EnforcePropertiesTask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EnforcePropertiesTask::Prepare(expr) => write!(f, "Prepare expr: {}", expr),
            EnforcePropertiesTask::Execute { expr, input, queue } => {
                write!(f, "Execute expr: {} input: {} queue: {}", expr, input, queue)
            }
            EnforcePropertiesTask::Done(expr) => write!(f, "Done expr: {}", expr),
        }
    }
}

fn optimize_groups<R>(
    runtime_state: &mut RuntimeState,
    memo: &ExprMemo,
    ctx: OptimizationContext,
    rule_set: &R,
) -> Result<(), OptimizerError>
where
    R: RuleSet,
{
    if let Some(impl_forms) = ctx.get_implementations() {
        let state = runtime_state.state.init_or_get_state(&ctx);
        if state.optimized {
            log::debug!("Group has already been optimized: {}", &ctx);
            runtime_state.stats.tasks.optimize_group_optimized += 1;
            return Ok(());
        }
        runtime_state.tasks.schedule(Task::GroupOptimized { ctx: ctx.clone() });

        // If there are multiple implementation forms of the required properties then
        // we must optimize this group under every implementation form of these required properties.
        // When optimization is complete we chose the best expression among all implementation
        // forms (See group_optimized).
        for impl_ctx in impl_forms {
            do_optimize_group(runtime_state, memo, impl_ctx, rule_set)?;
        }

        // Return early because otherwise we get a compile error:
        // error[E0505]: cannot move out of `ctx` because it is borrowed
        // at the line `if let Some(impl_forms) = ctx.get_implementations()`
        return Ok(());
    }

    do_optimize_group(runtime_state, memo, ctx, rule_set)
}

fn do_optimize_group<R>(
    runtime_state: &mut RuntimeState,
    memo: &ExprMemo,
    ctx: OptimizationContext,
    rule_set: &R,
) -> Result<(), OptimizerError>
where
    R: RuleSet,
{
    runtime_state.stats.tasks.optimize_group += 1;

    let state = runtime_state.state.init_or_get_state(&ctx);
    if state.optimized {
        log::debug!("Group has already been optimized: {}", &ctx);
        runtime_state.stats.tasks.optimize_group_optimized += 1;
        return Ok(());
    }

    runtime_state.tasks.schedule(Task::GroupOptimized { ctx: ctx.clone() });

    // Because the runtime_state and the group_state are borrowed mutably it is not possible
    // to split the code bellow into two different functions.
    // error[E0499]: cannot borrow `*runtime_state` as mutable more than once at a time
    let group = memo.get_group(&ctx.group_id)?;
    if group.expr().is_scalar() {
        let expr = group.mexpr();

        if expr.children().len() == 0 {
            let best_expr = BestExpr::new(expr.clone(), 0.0, vec![]);
            state.best_expr = Some(best_expr);
        } else {
            schedule_optimize_subqueries_task(runtime_state, &ctx, expr)?;
        }
    } else {
        runtime_state.tasks.schedule(Task::ExploreGroup { ctx: ctx.clone() });

        for expr in group.mexprs() {
            match expr.expr().relational() {
                RelExpr::Logical(_) => runtime_state.tasks.schedule(Task::OptimizeExpr {
                    ctx: ctx.clone(),
                    expr: expr.clone(),
                    explore: false,
                }),
                RelExpr::Physical(_) => {
                    schedule_optimize_rel_inputs_task(runtime_state, &ctx, &expr, rule_set)?;
                }
            }
        }
    }
    Ok(())
}

fn group_optimized(
    runtime_state: &mut RuntimeState,
    memo: &ExprMemo,
    ctx: OptimizationContext,
) -> Result<(), OptimizerError> {
    let normalized_state = runtime_state.state.get_state(&ctx)?;
    let mut normalized_state_best_expr = normalized_state.best_expr.clone();

    // If there multiple implementation forms of the required properties then
    // we must choose the best expression among all implementation forms and
    // assign it to the normalized form (See schedule_optimize_rel_inputs_task).
    // Otherwise use the existing best expression.
    if let Some(impl_forms) = ctx.get_implementations() {
        for impl_ctx in impl_forms {
            let impl_state = runtime_state.state.get_state(&impl_ctx)?;
            match (normalized_state_best_expr.as_ref(), impl_state.best_expr.as_ref()) {
                (None, None) => {
                    let message = format!(
                        "No best expr exists in neither normalized ctx nor in impl ctx. Normalized: {}, impl: {}",
                        ctx, impl_ctx
                    );
                    return Err(OptimizerError::internal(message));
                }
                (Some(best), Some(impl_expr)) if best.cost > impl_expr.cost => {
                    normalized_state_best_expr = Some(impl_expr.clone())
                }
                (None, Some(impl_expr)) => normalized_state_best_expr = Some(impl_expr.clone()),
                _ => {}
            };
        }
    }

    let state = runtime_state.state.get_state_mut(&ctx)?;
    if let Some(best_expr) = normalized_state_best_expr {
        state.best_expr = Some(best_expr);
    }

    if state.best_expr.is_none() {
        let group = memo.get_group(&ctx.group_id)?;
        let message = format!("No best expr for ctx: {}. first expression: {:?}", ctx, group.mexpr());
        Err(OptimizerError::internal(message))
    } else {
        state.optimized = true;
        Ok(())
    }
}

fn optimize_expr<R>(
    runtime_state: &mut RuntimeState,
    ctx: OptimizationContext,
    expr: ExprRef,
    explore: bool,
    rule_set: &R,
    metadata: MetadataRef,
) where
    R: RuleSet,
{
    runtime_state.stats.tasks.optimize_expr += 1;

    assert!(
        expr.expr().is_relational(),
        "optimize expr should not be called for non relational expressions: {:?}",
        expr
    );

    let expr_id = expr.id();
    let mut bindings = Vec::new();

    let rule_ctx = RuleContext::new(ctx.required_properties.clone(), metadata);
    for (rule_id, rule) in rule_set.get_rules() {
        let group_id = ctx.group_id;
        if runtime_state.applied_rules.is_applied(rule_id, &expr_id, &group_id) {
            continue;
        }
        // FIXME: Apply only transformation rules during exploration phase
        if !explore && rule.group_rule() {
            continue;
        }
        if let Some(rule_match) = rule.matches(&rule_ctx, expr.expr().relational().logical()) {
            let binding = RuleBinding {
                rule_id: *rule_id,
                expr: expr.clone(),
                group: match rule_match {
                    RuleMatch::Expr => None,
                    RuleMatch::Group => Some(ctx.group_id),
                },
                rule_type: rule.rule_type(),
                #[cfg(test)]
                rule_name: rule.name(),
            };
            // First transformation rules then implementation rules
            // Ordering is reversed because we are using an explicit stack
            let ordering = match rule.rule_type() {
                RuleType::Transformation => 2,
                RuleType::Implementation => 1,
            };
            bindings.push((rule_id, ordering, binding))
        }
    }

    bindings.sort_by(|a, b| a.1.cmp(&b.1));

    if bindings.is_empty() {
        runtime_state.stats.tasks.optimize_expr_no_rules_matched += 1;
    }

    for (_, _, binding) in bindings {
        runtime_state.tasks.schedule(Task::ApplyRule {
            ctx: ctx.clone(),
            expr: expr.clone(),
            binding,
            explore,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_rule<R>(
    runtime_state: &mut RuntimeState,
    memo: &mut ExprMemo,
    ctx: OptimizationContext,
    expr: ExprRef,
    binding: RuleBinding,
    explore: bool,
    rule_set: &R,
) -> Result<(), OptimizerError>
where
    R: RuleSet,
{
    runtime_state.stats.tasks.apply_rule += 1;
    let expr_id = expr.id();
    let rule_id = binding.rule_id;
    let rule_type = &binding.rule_type;
    let group_id = ctx.group_id;

    if rule_type == &RuleType::Transformation && runtime_state.applied_rules.is_applied(&rule_id, &expr_id, &group_id) {
        return Ok(());
    }

    let operator = expr.expr();
    let metadata = memo.metadata().get_ref();

    let rule_ctx = RuleContext::new(ctx.required_properties.clone(), metadata);
    let result = rule_set.apply_rule(&rule_id, &rule_ctx, operator.relational().logical())?;

    if let Some(result) = result {
        match result {
            RuleResult::Substitute(expr) => {
                let new_operator = OperatorExpr::from(expr);
                let new_operator = Operator::from(new_operator);
                let memo_group = memo.get_group(&ctx.group_id)?;
                let scope = OuterScope::from_properties(memo_group.props());
                let group_token = memo_group.to_group_token();
                let new_expr = memo.insert_group_member(group_token, new_operator, &scope)?;
                let new_expr = new_expr.state().memo_expr();

                runtime_state.applied_rules.mark_applied(&rule_id, &binding);

                log::debug!(" + Logical expr: {} ctx: {}", new_expr, ctx);

                runtime_state.tasks.schedule(Task::OptimizeExpr {
                    ctx,
                    expr: new_expr,
                    explore,
                });
            }
            RuleResult::Implementation(expr) => {
                let new_operator = OperatorExpr::from(expr);
                let new_operator = Operator::from(new_operator);
                let memo_group = memo.get_group(&ctx.group_id)?;
                let scope = OuterScope::from_properties(memo_group.props());
                let group_token = memo_group.to_group_token();
                let new_expr = memo.insert_group_member(group_token, new_operator, &scope)?;
                let new_expr = new_expr.state().memo_expr();
                let state = runtime_state.state.get_state(&ctx)?;

                if state.is_expr_optimized(&new_expr.id()) {
                    return Ok(());
                }

                log::debug!(" + Physical expr: {} ctx: {}", new_expr, ctx);

                schedule_optimize_rel_inputs_task(runtime_state, &ctx, &new_expr, rule_set)?;
            }
        }
    }
    Ok(())
}

fn schedule_optimize_rel_inputs_task<R>(
    runtime_state: &mut RuntimeState,
    ctx: &OptimizationContext,
    expr: &ExprRef,
    rule_set: &R,
) -> Result<(), OptimizerError>
where
    R: RuleSet,
{
    if !runtime_state.state.is_expr_optimized(ctx, expr) {
        runtime_state.tasks.schedule(Task::ExprOptimized {
            ctx: ctx.clone(),
            expr: expr.clone(),
        });

        let task = get_optimize_rel_inputs_task(ctx, expr, rule_set)?;
        runtime_state.tasks.schedule(task);
    }

    Ok(())
}

fn create_enforcer<R>(
    required_properties: &RequiredProperties,
    input_group_id: GroupId,
    memo: &mut ExprMemo,
    rule_set: &R,
) -> Result<(ExprRef, Option<RequiredProperties>), OptimizerError>
where
    R: RuleSet,
{
    let group = memo.get_group(&input_group_id)?;
    let scope = OuterScope::from_properties(group.props());
    let input = RelNode::try_from_group(group)?;

    let properties_provider = rule_set.get_physical_properties_provider();
    let (enforcer_expr, remaining_properties) = properties_provider.create_enforcer(required_properties, input)?;
    let enforcer_expr = OperatorExpr::from(enforcer_expr);
    // ENFORCERS: Currently enforcers can not be retrieved from group.exprs()
    // because adding an enforcer to a group won't allow the optimizer to complete the optimization process:
    // 1) When an enforcer is added from explore_group task the next optimize_group task will call
    // optimize_inputs on the enforcer which then calls optimize group on the same group and we have an infinite loop.
    // 2) copy_out_best_expr traverses its child expressions recursively so adding an enforcer to a group which
    // is used as its input results an infinite loop during the traversal.
    let enforcer_expr = memo.insert_group(Operator::from(enforcer_expr), &scope).unwrap();
    let enforcer_expr = enforcer_expr.state().memo_expr();

    Ok((enforcer_expr, remaining_properties))
}

fn enforce_properties<R>(
    runtime_state: &mut RuntimeState,
    memo: &mut ExprMemo,
    ctx: OptimizationContext,
    enforcer_task: EnforcerTask,
    rule_set: &R,
) -> Result<(), OptimizerError>
where
    R: RuleSet,
{
    runtime_state.stats.tasks.enforce_properties += 1;

    match enforcer_task {
        EnforcerTask::EnforceProperties(EnforcePropertiesTask::Prepare(expr)) => {
            let required_properties = ctx.required_properties.as_ref().as_ref().expect("No required properties");
            let properties_provider = rule_set.get_physical_properties_provider();
            let requirements = properties_provider.get_enforcer_order(required_properties)?;

            let next_stage = EnforcePropertiesTask::Execute {
                expr: expr.clone(),
                // start from an input expression
                input: expr,
                queue: requirements,
            };

            runtime_state.tasks.schedule(Task::EnforceProperties {
                ctx,
                enforcer_task: EnforcerTask::EnforceProperties(next_stage),
            });
            Ok(())
        }
        EnforcerTask::EnforceProperties(EnforcePropertiesTask::Execute {
            expr,
            input: input_expr,
            mut queue,
        }) => {
            let (required_properties, remaining_properties, enforced_properties) = queue.pop();
            let (enforcer_expr, _) = create_enforcer(&required_properties, input_expr.group_id(), memo, rule_set)?;

            if !queue.is_empty() {
                // continue with an enforcer operator
                let next_stage = EnforcePropertiesTask::Execute {
                    expr,
                    input: enforcer_expr.clone(),
                    queue,
                };
                runtime_state.tasks.schedule(Task::EnforceProperties {
                    ctx: ctx.clone(),
                    enforcer_task: EnforcerTask::EnforceProperties(next_stage),
                });
            } else {
                let next_stage = EnforcePropertiesTask::Done(expr);
                runtime_state.tasks.schedule(Task::EnforceProperties {
                    ctx: ctx.clone(),
                    enforcer_task: EnforcerTask::EnforceProperties(next_stage),
                });
            };

            // An enforcer is optimized under an optimization context that includes
            // "would be enforced properties" (all properties enforced so far) +
            // properties provided by the enforcer.
            let enforcer_opt_ctx = OptimizationContext {
                group_id: ctx.group_id,
                required_properties: Rc::new(Some(enforced_properties)),
            };
            let remaining_properties = Rc::new(remaining_properties);

            // Inputs of an enforcer operator are optimized w/o properties provided by that enforcer.
            let task = OptimizeInputsTaskBuilder::new(enforcer_opt_ctx, enforcer_expr)
                .require_properties(remaining_properties.clone())
                .build();

            runtime_state.tasks.schedule(task);

            // We optimize an operator w/o properties provided by the enforcer.
            let input_ctx = OptimizationContext {
                group_id: ctx.group_id,
                required_properties: remaining_properties,
            };

            // Check to see whether we have already optimized the operator
            // w/o properties provided by the enforcer.
            if !runtime_state.state.is_expr_optimized(&input_ctx, &input_expr) {
                runtime_state.tasks.schedule(Task::ExprOptimized {
                    ctx: input_ctx.clone(),
                    expr: input_expr.clone(),
                });

                let physical_expr = input_expr.expr().relational().physical();
                let required_input_properties = physical_expr.get_required_input_properties();
                // We must optimize the expression using the properties required by an operator.
                let task = OptimizeInputsTaskBuilder::new(input_ctx, input_expr)
                    .use_required_properties(required_input_properties)
                    .build();

                runtime_state.tasks.schedule(task);
            }

            Ok(())
        }
        EnforcerTask::EnforceProperties(EnforcePropertiesTask::Done(_)) => {
            // TODO: Ensure that expression has been optimized?
            let _ = runtime_state.state.get_state(&ctx)?;
            Ok(())
        }
        EnforcerTask::ExploreAlternatives => {
            let required_properties = ctx.required_properties.as_ref().as_ref().unwrap();
            let (enforcer_expr, remaining_properties) =
                create_enforcer(required_properties, ctx.group_id, memo, rule_set)?;
            // An enforcer operator is used to explore alternative plans.
            // We should optimize its inputs w/o properties provided by that enforcer.
            let optimize_inputs = OptimizeInputsTaskBuilder::new(ctx.clone(), enforcer_expr)
                .require_properties(Rc::new(remaining_properties))
                .build();

            runtime_state.tasks.schedule(optimize_inputs);
            Ok(())
        }
    }
}

fn optimize_inputs<T>(
    runtime_state: &mut RuntimeState,
    ctx: OptimizationContext,
    expr: ExprRef,
    mut inputs: OptimizeInputsState,
    cost_estimator: &T,
) -> Result<(), OptimizerError>
where
    T: CostEstimator,
{
    runtime_state.stats.tasks.optimize_inputs += 1;

    let expr_id = expr.id();

    if let Some(input_ctx) = inputs.next_input() {
        runtime_state.tasks.schedule(Task::OptimizeInputs { ctx, expr, inputs });

        runtime_state.tasks.schedule(Task::OptimizeGroup { ctx: input_ctx })
    } else {
        let cost = match expr.expr() {
            OperatorExpr::Relational(rel_expr) => {
                let logical_properties = expr.props().relational().logical();
                let statistics = logical_properties.statistics();
                let (cost_ctx, inputs_cost) = new_cost_estimation_ctx(&inputs, &runtime_state.state)?;
                let expr_cost = cost_estimator.estimate_cost(rel_expr.physical(), &cost_ctx, statistics);
                // a plan that has more operators must have a higher cost
                expr_cost + inputs_cost + COST_PER_OPERATOR
            }
            OperatorExpr::Scalar(_) => {
                let (_, inputs_cost) = new_cost_estimation_ctx(&inputs, &runtime_state.state)?;
                inputs_cost
            }
        };

        log::debug!("Expr cost: {} ctx: {} expr: {} {}", cost, ctx, expr_id, expr);

        let candidate = BestExpr::new(expr, cost, inputs.inputs);
        let state = runtime_state.state.init_or_get_state(&ctx);

        match state.best_expr.as_mut() {
            None => state.best_expr = Some(candidate),
            Some(best) if best.cost > cost => *best = candidate,
            _ => {}
        }
    }
    Ok(())
}

fn schedule_optimize_subqueries_task(
    runtime_state: &mut RuntimeState,
    ctx: &OptimizationContext,
    expr: &ExprRef,
) -> Result<(), OptimizerError> {
    // Optimize nested subqueries with their required properties.
    let required_props = expr.children().map(|expr| expr.props().relational().physical.required.clone()).collect();

    let task = OptimizeInputsTaskBuilder::new(ctx.clone(), expr.clone())
        .use_required_properties(Some(required_props))
        .build();

    runtime_state.tasks.schedule(task);
    Ok(())
}

fn get_optimize_rel_inputs_task<R>(
    ctx: &OptimizationContext,
    expr: &ExprRef,
    rule_set: &R,
) -> Result<Task, OptimizerError>
where
    R: RuleSet,
{
    let physical_expr = expr.expr().relational().physical();
    let required_input_properties = physical_expr.get_required_input_properties();
    let properties_provider = rule_set.get_physical_properties_provider();

    let result = match ctx.required_properties.as_ref() {
        Some(required_properties) => Some(properties_provider.derive_properties(physical_expr, required_properties)?),
        None => None,
    };

    let task = match result {
        None if required_input_properties.is_none() => {
            // Optimization context has no required properties + expression does not require any property from its inputs.
            // -> optimize each child expression using required properties of child expressions.
            // In this case every child expression is optimized using required properties of its memo group.
            OptimizeInputsTaskBuilder::new(ctx.clone(), expr.clone()).use_child_properties().build()
        }
        Some(DerivePropertyMode::PropertyIsProvided) | None => {
            // Expression provides the required properties or expression requires properties from its children.
            // -> optimize child expressions with properties required by the expression.
            OptimizeInputsTaskBuilder::new(ctx.clone(), expr.clone())
                .use_required_properties(required_input_properties)
                .build()
        }
        Some(DerivePropertyMode::PropertyIsRetained) => {
            // Expression retains the required properties
            // -> optimize child expressions under the same optimization context.
            OptimizeInputsTaskBuilder::new(ctx.clone(), expr.clone())
                .require_properties(ctx.required_properties.clone())
                .build()
        }
        Some(DerivePropertyMode::ApplyEnforcer) => {
            // Expression can not provide the required properties itself
            // -> add an enforcer operator.
            let enforcer_task = EnforcerTask::EnforceProperties(EnforcePropertiesTask::Prepare(expr.clone()));
            Task::EnforceProperties {
                ctx: ctx.clone(),
                enforcer_task,
            }
        }
    };
    Ok(task)
}

fn expr_optimized(runtime_state: &mut RuntimeState, ctx: OptimizationContext, expr: ExprRef) {
    let state = runtime_state.state.init_or_get_state(&ctx);
    let expr_id = expr.id();

    state.mark_expr_optimized(expr_id);
}

fn explore_group<R>(
    runtime_state: &mut RuntimeState,
    memo: &ExprMemo,
    ctx: OptimizationContext,
    rule_set: &R,
) -> Result<(), OptimizerError>
where
    R: RuleSet,
{
    if runtime_state.enable_explore_groups {
        let group = memo.get_group(&ctx.group_id)?;
        if let Some(required_properties) = ctx.required_properties.as_ref() {
            let properties_provider = rule_set.get_physical_properties_provider();
            if properties_provider.can_explore_with_enforcer(group.expr().relational(), required_properties) {
                runtime_state.tasks.schedule(Task::EnforceProperties {
                    ctx: ctx.clone(),
                    enforcer_task: EnforcerTask::ExploreAlternatives,
                })
            }
        }

        for expr in group.mexprs() {
            match expr.expr().relational() {
                RelExpr::Logical(_) => runtime_state.tasks.schedule(Task::OptimizeExpr {
                    ctx: ctx.clone(),
                    expr: expr.clone(),
                    explore: true,
                }),
                RelExpr::Physical(_) => {}
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct OptimizationContext {
    group_id: GroupId,
    required_properties: Rc<Option<RequiredProperties>>,
}

impl OptimizationContext {
    fn get_implementations(&self) -> Option<impl Iterator<Item = OptimizationContext> + '_> {
        if let Some(r) = self.required_properties.as_ref() {
            r.get_implementations().map(move |r| {
                r.map(move |r| {
                    let mut ctx = self.clone();
                    ctx.required_properties = Rc::new(Some(r));
                    ctx
                })
            })
        } else {
            None
        }
    }
}

impl Display for OptimizationContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ group: {}", self.group_id)?;
        if let Some(required_properties) = self.required_properties.as_ref() {
            write!(f, " {}", required_properties)?;
        }
        write!(f, " }}")
    }
}

struct RuntimeState {
    tasks: TaskQueue,
    state: State,
    applied_rules: AppliedRules,
    stats: Stats,
    // TODO:
    enable_explore_groups: bool,
}

impl RuntimeState {
    fn new() -> Self {
        RuntimeState {
            tasks: TaskQueue::default(),
            state: State::default(),
            stats: Stats::default(),
            applied_rules: AppliedRules::default(),
            enable_explore_groups: true,
        }
    }
}

#[derive(Debug, Default)]
struct Stats {
    number_of_tasks: usize,
    max_stack_depth: usize,
    optimization_time: Duration,
    result_time: Duration,
    tasks: TaskStats,
}

#[derive(Debug, Default)]
struct TaskStats {
    optimize_group: usize,
    optimize_expr: usize,
    apply_rule: usize,
    optimize_inputs: usize,
    enforce_properties: usize,
    explore_group: usize,
    explore_expr: usize,

    optimize_group_optimized: usize,
    optimize_expr_no_rules_matched: usize,
    explore_expr_no_rules_matched: usize,
}

#[derive(Debug, Default)]
struct TaskQueue {
    queue: VecDeque<Task>,
}

impl TaskQueue {
    fn schedule(&mut self, task: Task) {
        log::debug!(" + {}", &task);

        self.queue.push_back(task);
    }

    fn retrieve(&mut self) -> Option<Task> {
        self.queue.pop_back()
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

#[derive(Debug, Default)]
struct State {
    groups: HashMap<OptimizationContext, GroupState>,
}

impl State {
    fn init_or_get_state(&mut self, ctx: &OptimizationContext) -> &mut GroupState {
        match self.groups.entry(ctx.clone()) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(GroupState::default()),
        }
    }

    fn get_state(&self, ctx: &OptimizationContext) -> Result<&GroupState, OptimizerError> {
        match self.groups.get(ctx) {
            Some(state) => Ok(state),
            None => Err(OptimizerError::internal(format!("Group state should have been created: {}", ctx))),
        }
    }

    fn get_state_mut(&mut self, ctx: &OptimizationContext) -> Result<&mut GroupState, OptimizerError> {
        match self.groups.get_mut(ctx) {
            Some(state) => Ok(state),
            None => Err(OptimizerError::internal(format!("Group state should have been created: {}", ctx))),
        }
    }

    fn is_expr_optimized(&self, ctx: &OptimizationContext, expr: &ExprRef) -> bool {
        self.groups.get(ctx).map(|s| s.is_expr_optimized(&expr.id())).unwrap_or_default()
    }
}

#[derive(Debug, Default)]
struct GroupState {
    optimized: bool,
    explored: bool,
    optimized_exprs: HashSet<ExprId>,
    best_expr: Option<BestExpr>,
}

impl GroupState {
    fn is_expr_optimized(&self, expr_id: &ExprId) -> bool {
        self.optimized_exprs.contains(expr_id)
    }

    // FIXME: works incorrectly - currently physical exprs instead of logical exprs are marked as optimized
    fn mark_expr_optimized(&mut self, expr_id: ExprId) {
        self.optimized_exprs.insert(expr_id);
    }
}

#[derive(Debug, Clone)]
struct BestExpr {
    expr: ExprRef,
    cost: Cost,
    inputs: Vec<OptimizationContext>,
}

impl BestExpr {
    fn new(expr: ExprRef, cost: Cost, inputs: Vec<OptimizationContext>) -> Self {
        BestExpr { expr, cost, inputs }
    }
}

impl Display for BestExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ expr: {}", self.expr)?;
        write!(f, " cost: {}", self.cost)?;
        write!(f, " input_ctx: [{}]", self.inputs.iter().join(", "))?;
        write!(f, "] }}")
    }
}

#[derive(Debug)]
struct RuleBinding {
    rule_id: RuleId,
    expr: ExprRef,
    group: Option<GroupId>,
    rule_type: RuleType,
    #[cfg(test)]
    rule_name: String,
}

#[derive(Debug, Default)]
struct AppliedRules {
    exprs: HashSet<(RuleId, ExprId)>,
    groups: HashSet<(RuleId, GroupId)>,
}

impl AppliedRules {
    fn is_applied(&self, rule_id: &RuleId, expr_id: &ExprId, group_id: &GroupId) -> bool {
        if self.exprs.contains(&(*rule_id, *expr_id)) {
            true
        } else {
            self.groups.contains(&(*rule_id, *group_id))
        }
    }

    fn mark_applied(&mut self, rule_id: &RuleId, binding: &RuleBinding) {
        let group = binding.group.as_ref();
        let expr_id = binding.expr.id();

        self.exprs.insert((*rule_id, expr_id));

        if let Some(group) = group {
            self.groups.insert((*rule_id, *group));
        }
    }
}

#[derive(Debug)]
struct OptimizeInputsState {
    inputs: Vec<OptimizationContext>,
    current_index: usize,
}

impl OptimizeInputsState {
    fn next_input(&mut self) -> Option<OptimizationContext> {
        if self.current_index < self.inputs.len() {
            let ctx = self.inputs[self.current_index].clone();
            self.current_index += 1;
            Some(ctx)
        } else {
            None
        }
    }
}

/// A builder to create instances of [Task::OptimizeInputs].
struct OptimizeInputsTaskBuilder {
    ctx: OptimizationContext,
    expr: ExprRef,
    input_properties: Vec<Rc<Option<RequiredProperties>>>,
}

impl OptimizeInputsTaskBuilder {
    /// Creates an instance of [OptimizeInputsTaskBuilder] for the given expression
    /// under the specified optimization context.
    fn new(ctx: OptimizationContext, expr: ExprRef) -> Self {
        OptimizeInputsTaskBuilder {
            ctx,
            expr,
            input_properties: Vec::new(),
        }
    }

    /// Specifies the required physical properties for each child expression.
    /// * If properties are present then each `i-th` child expression is optimized using `i-th` properties from the given `Vec`.
    /// * If properties are `None` then each child expression is optimized with no required properties.
    fn use_required_properties(
        mut self,
        properties: Option<Vec<Option<RequiredProperties>>>,
    ) -> OptimizeInputsTaskBuilder {
        match properties {
            Some(properties) => {
                self.input_properties = self
                    .expr
                    .children()
                    .zip(properties.into_iter())
                    .map(|(expr, props)| match (expr.expr(), props) {
                        (OperatorExpr::Relational(_), props) => Rc::new(props),
                        (OperatorExpr::Scalar(_), _) => Rc::new(None),
                    })
                    .collect();
            }
            None => {
                let num_children = self.expr.children().len();
                self.input_properties = std::iter::repeat(Rc::new(None)).take(num_children).collect()
            }
        }
        self
    }

    /// Every child expression should be optimized under the properties required by the properties of it group.
    fn use_child_properties(mut self) -> OptimizeInputsTaskBuilder {
        self.input_properties = self
            .expr
            .children()
            .map(|expr| match expr.expr() {
                OperatorExpr::Relational(_) => Rc::new(expr.props().relational().physical().required.clone()),
                // Nested subqueries are optimized using their properties.
                // See schedule_optimize_subqueries_task.
                OperatorExpr::Scalar(_) => Rc::new(None),
            })
            .collect();
        self
    }

    /// Every child expression of this expression must be optimized under the given required physical properties.
    fn require_properties(mut self, properties: Rc<Option<RequiredProperties>>) -> OptimizeInputsTaskBuilder {
        self.input_properties = std::iter::repeat(properties)
            .take(self.expr.children().len())
            .zip(self.expr.children())
            .map(|(r, expr)| match expr.expr() {
                OperatorExpr::Relational(_) => r,
                // Nested subqueries are optimized using their properties.
                // See schedule_optimize_rel_inputs_task.
                OperatorExpr::Scalar(_) => Rc::new(None),
            })
            .collect();
        self
    }

    /// Builds an OptimizeInputs task.
    pub fn build(self) -> Task {
        let input_properties = self.input_properties;
        let child_iter = self.expr.children();

        assert_eq!(
            input_properties.len(),
            child_iter.len(),
            "Number of input properties is not equal not to the number of child expressions"
        );

        let inputs = input_properties
            .into_iter()
            .zip(child_iter)
            .map(|(required, expr)| OptimizationContext {
                group_id: expr.group_id(),
                required_properties: required,
            })
            .collect();

        let inputs = OptimizeInputsState {
            inputs,
            current_index: 0,
        };

        Task::OptimizeInputs {
            ctx: self.ctx,
            expr: self.expr.clone(),
            inputs,
        }
    }
}

impl Display for OptimizeInputsState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ inputs=[{}]", self.inputs.iter().join(", "))?;
        write!(f, " current_index={}", self.current_index)?;
        write!(f, "]")
    }
}

fn new_cost_estimation_ctx(
    inputs: &OptimizeInputsState,
    state: &State,
) -> Result<(CostEstimationContext, Cost), OptimizerError> {
    let capacity = inputs.inputs.len();
    let mut best_exprs = Vec::with_capacity(capacity);
    let mut input_cost = 0.0;

    for ctx in inputs.inputs.iter() {
        let group_state = state.get_state(ctx)?;
        if let Some(best_expr) = group_state.best_expr.as_ref() {
            best_exprs.push(best_expr.expr.clone());
            input_cost += best_expr.cost;
        } else {
            return Err(OptimizerError::internal(format!("No best expr for input group: {:?}", ctx)));
        }
    }

    Ok((CostEstimationContext { inputs: best_exprs }, input_cost))
}

/// A [OptimizedExprCallback] that does nothing.
#[derive(Debug)]
pub struct DefaultOptimizedExprCallback;

impl OptimizedExprCallback for DefaultOptimizedExprCallback {
    fn on_best_expr<C>(&self, _expr: BestExprRef, _ctx: &C)
    where
        C: BestExprContext,
    {
        //no-op
    }
}

struct OptimizerResultCallbackContext<'o> {
    ctx: &'o OptimizationContext,
    best_expr: &'o BestExpr,
    properties: &'o Properties,
    inputs: &'o [OptimizationContext],
}

impl<'o> BestExprContext for OptimizerResultCallbackContext<'o> {
    fn cost(&self) -> Cost {
        self.best_expr.cost
    }

    fn props(&self) -> &Properties {
        self.properties
    }

    fn group_id(&self) -> GroupId {
        self.ctx.group_id
    }

    fn num_children(&self) -> usize {
        self.inputs.len()
    }

    fn child_group_id(&self, i: usize) -> GroupId {
        self.inputs[i].group_id
    }

    fn child_required(&self, i: usize) -> Option<&RequiredProperties> {
        self.inputs[i].required_properties.as_ref().as_ref()
    }
}

fn copy_out_best_expr<'o, T>(
    result_callback: &T,
    state: &'o State,
    memo: &'o ExprMemo,
    ctx: &'o OptimizationContext,
) -> Result<Operator, OptimizerError>
where
    T: OptimizedExprCallback,
{
    let group_state = state.get_state(ctx)?;
    match &group_state.best_expr {
        Some(best_expr) => {
            let best_expr_ref = match best_expr.expr.expr() {
                OperatorExpr::Relational(rel_expr) => match rel_expr {
                    RelExpr::Logical(expr) => {
                        let message =
                            format!("Invalid best expression. Expected a physical operator but got: {:?}", expr);
                        Err(OptimizerError::internal(message))
                    }
                    RelExpr::Physical(expr) => Ok(BestExprRef::Relational(expr)),
                },
                OperatorExpr::Scalar(expr) => Ok(BestExprRef::Scalar(expr)),
            }?;

            let inputs = &best_expr.inputs[0..best_expr.inputs.len()];
            let mut new_inputs = VecDeque::with_capacity(inputs.len());

            for input_ctx in &best_expr.inputs {
                let out = copy_out_best_expr(result_callback, state, memo, input_ctx)?;
                new_inputs.push_back(out);
            }

            let group = memo.get_group(&ctx.group_id)?;
            let properties = group.props();
            let best_expr_ctx = OptimizerResultCallbackContext {
                ctx,
                best_expr,
                properties,
                inputs,
            };
            result_callback.on_best_expr(best_expr_ref.clone(), &best_expr_ctx);
            //TODO: Copy required properties.
            let mut new_inputs = NewChildExprs::new(new_inputs);
            let new_expr = match best_expr_ref {
                BestExprRef::Relational(expr) => OperatorExpr::from(expr.with_new_inputs(&mut new_inputs)?),
                BestExprRef::Scalar(expr) => OperatorExpr::from(expr_with_new_inputs(expr, &mut new_inputs)?),
            };
            let result = Operator::from(new_expr);
            Ok(result)
        }
        None => {
            let message = format!("No best expression for ctx: {}", ctx);
            Err(OptimizerError::internal(message))
        }
    }
}
