use itertools::Itertools;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display, Formatter};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::cost::{Cost, CostEstimationContext, CostEstimator};
use crate::memo::{format_memo, ExprId, GroupId, MemoExprCallback};
use crate::meta::Metadata;
use crate::operators::{ExprCallback, ExprMemo, ExprRef, GroupRef, InputExpr, Operator, OperatorExpr, Properties};
use crate::properties::logical::LogicalProperties;
use crate::properties::logical::PropertiesProvider;
use crate::properties::physical::PhysicalProperties;
use crate::rules::{RuleContext, RuleId, RuleMatch, RuleResult, RuleSet, RuleType};
use crate::util::{BestExprContext, ResultCallback};

/// Cost-based optimizer.
pub struct Optimizer<R, T, C, P> {
    rule_set: Rc<R>,
    cost_estimator: Rc<T>,
    memo_callback: Rc<ExprCallback>,
    result_callback: Rc<C>,
    properties_provider_type: std::marker::PhantomData<P>,
}

impl<R, T, C, P> Optimizer<R, T, C, P>
where
    R: RuleSet,
    T: CostEstimator,
    C: ResultCallback,
    P: PropertiesProvider + 'static,
{
    /// Creates a new instance of `Optimizer`.
    pub fn new(rule_set: Rc<R>, cost_estimator: Rc<T>, properties_provider: Rc<P>, result_callback: Rc<C>) -> Self {
        #[derive(Debug)]
        struct PropagateProperties<P> {
            properties_provider: Rc<P>,
        }

        impl<P> MemoExprCallback for PropagateProperties<P>
        where
            P: PropertiesProvider,
        {
            type Expr = Operator;
            type Attrs = Properties;

            fn new_expr(&self, expr: &Self::Expr, attrs: Self::Attrs) -> Self::Attrs {
                // Every time a new expression is added to a memo we need to compute logical properties of that expression.
                let Properties { logical, required } = attrs;
                let logical = self
                    .properties_provider
                    .build_properties(expr.expr(), logical.statistics())
                    // If we has not been able to assemble logical properties for the given expression
                    // than something has gone terribly wrong and we have no other option but to unwrap an error.
                    .expect("Failed to build logical properties");

                Properties::new(logical, required)
            }
        }

        Optimizer {
            rule_set,
            cost_estimator,
            memo_callback: Rc::new(PropagateProperties { properties_provider }),
            result_callback,
            properties_provider_type: Default::default(),
        }
    }

    /// Optimizes the given operator tree.
    pub fn optimize(&self, expr: Operator, metadata: Metadata) -> Result<Operator, String> {
        let mut runtime_state = RuntimeState::new();
        let mut memo = ExprMemo::with_callback(self.memo_callback.clone());

        log::debug!("Optimizing expression: {:?}", expr);

        let required_property = expr.required().clone();
        let (root_group, _) = memo.insert(expr);

        let ctx = OptimizationContext {
            group: root_group,
            required_properties: required_property,
        };

        log::debug!("Initial ctx: {}, initial memo:\n{}", ctx, format_memo(&memo));

        runtime_state.tasks.schedule(Task::OptimizeGroup { ctx: ctx.clone() });

        self.do_optimize(&mut runtime_state, &mut memo, &metadata);

        let state = runtime_state.state.get_state(&ctx);
        assert!(state.optimized, "Root node has not been optimized: {:?}", state);

        log::debug!("Final memo:\n{}", format_memo(&memo));

        self.get_result(runtime_state, &memo, &ctx)
    }

    fn do_optimize(&self, runtime_state: &mut RuntimeState, memo: &mut ExprMemo, metadata: &Metadata) {
        let start_time = Instant::now();

        while let Some(task) = runtime_state.tasks.retrieve() {
            log::debug!("{}", &task);

            runtime_state.stats.number_of_tasks += 1;
            runtime_state.stats.max_stack_depth =
                runtime_state.stats.max_stack_depth.max(runtime_state.tasks.len() + 1);

            match task {
                Task::OptimizeGroup { ctx } => {
                    optimize_group(runtime_state, ctx, self.rule_set.as_ref());
                }
                Task::GroupOptimized { ctx } => {
                    group_optimized(runtime_state, ctx);
                }
                Task::OptimizeExpr { ctx, expr, explore } => {
                    optimize_expr(runtime_state, ctx, expr, explore, self.rule_set.as_ref(), metadata);
                }
                Task::ApplyRule {
                    ctx,
                    expr,
                    binding,
                    explore,
                } => {
                    apply_rule(runtime_state, memo, ctx, expr, binding, explore, self.rule_set.as_ref(), metadata);
                }
                Task::EnforceProperties { ctx, expr } => {
                    enforce_properties(runtime_state, memo, ctx, expr, self.rule_set.as_ref());
                }
                Task::OptimizeInputs { ctx, expr, inputs } => {
                    optimize_inputs(runtime_state, ctx, expr, inputs, self.cost_estimator.as_ref());
                }
                Task::ExprOptimized { ctx, expr } => {
                    expr_optimized(runtime_state, ctx, expr);
                }
                Task::ExploreGroup { ctx } => {
                    explore_group(runtime_state, ctx, self.rule_set.as_ref());
                }
            }
        }

        runtime_state.stats.optimization_time = start_time.elapsed();
    }

    fn get_result(
        &self,
        runtime_state: RuntimeState,
        memo: &ExprMemo,
        ctx: &OptimizationContext,
    ) -> Result<Operator, String> {
        let mut stats = runtime_state.stats;
        let state = runtime_state.state;

        let result_time = Instant::now();
        let expr = copy_out_best_expr(self.result_callback.as_ref(), &state, memo, ctx)?;
        stats.result_time = result_time.elapsed();

        log::debug!("Stats: {:?}", stats);

        Ok(expr)
    }
}

impl<R, T, C, P> Debug for Optimizer<R, T, C, P>
where
    R: RuleSet,
    T: CostEstimator,
    C: ResultCallback,
    P: PropertiesProvider + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Optimizer")
            .field("rule_set", self.rule_set.as_ref())
            .field("cost_estimator", self.cost_estimator.as_ref())
            //FIXME: add properties_provider to debug output
            // .field("properties_provider", self.memo_callback.as_ref())
            .field("result_callback", self.result_callback.as_ref())
            .finish()
    }
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
        expr: ExprRefOption,
    },
    OptimizeInputs {
        ctx: OptimizationContext,
        expr: ExprRef,
        inputs: InputContexts,
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
            Task::EnforceProperties { ctx, expr } => {
                write!(f, "EnforceProperties: {} expr: {}", ctx, expr)
            }
            Task::ExprOptimized { ctx, expr } => {
                write!(f, "ExprOptimized: {} expr: {}", ctx, expr)
            }
            Task::ExploreGroup { ctx } => write!(f, "ExploreGroup: {}", ctx),
        }
    }
}

#[derive(Debug)]
struct ExprRefOption {
    expr: Option<ExprRef>,
}

impl ExprRefOption {
    fn some(expr: ExprRef) -> Self {
        ExprRefOption { expr: Some(expr) }
    }
    fn none() -> Self {
        ExprRefOption { expr: None }
    }
}

impl Display for ExprRefOption {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.expr.as_ref() {
            None => write!(f, "none"),
            Some(expr) => write!(f, "{}", expr),
        }
    }
}

fn optimize_group<R>(runtime_state: &mut RuntimeState, ctx: OptimizationContext, rule_set: &R)
where
    R: RuleSet,
{
    runtime_state.stats.tasks.optimize_group += 1;

    let state = runtime_state.state.init_or_get_state(&ctx);
    if state.optimized {
        log::debug!("Group has already been optimized: {}", &ctx);
        runtime_state.stats.tasks.optimize_group_optimized += 1;
        return;
    }

    runtime_state.tasks.schedule(Task::GroupOptimized { ctx: ctx.clone() });

    runtime_state.tasks.schedule(Task::ExploreGroup { ctx: ctx.clone() });

    for expr in ctx.group.mexprs() {
        let expr_id = expr.id();
        match expr.expr() {
            OperatorExpr::Logical(_) => {
                assert!(!state.is_expr_optimized(&expr_id), "already optimized");

                runtime_state.tasks.schedule(Task::OptimizeExpr {
                    ctx: ctx.clone(),
                    expr: expr.clone(),
                    explore: false,
                })
            }
            OperatorExpr::Physical(_) => {
                let task = get_optimize_inputs_task(&ctx, expr, rule_set);
                runtime_state.tasks.schedule(task);
            }
        }
    }
}

fn group_optimized(runtime_state: &mut RuntimeState, ctx: OptimizationContext) {
    let state = runtime_state.state.init_or_get_state(&ctx);

    assert!(state.best_expr.is_some(), "No best expr for ctx: {:?}", ctx);
    state.optimized = true;
}

fn optimize_expr<R>(
    runtime_state: &mut RuntimeState,
    ctx: OptimizationContext,
    expr: ExprRef,
    explore: bool,
    rule_set: &R,
    metadata: &Metadata,
) where
    R: RuleSet,
{
    runtime_state.stats.tasks.optimize_expr += 1;

    let expr_id = expr.id();
    let mut bindings = Vec::new();

    let rule_ctx = RuleContext::new(ctx.required_properties.clone(), metadata);
    for (rule_id, rule) in rule_set.get_rules() {
        let group_id = ctx.group_id();
        if runtime_state.applied_rules.is_applied(rule_id, &expr_id, &group_id) {
            continue;
        }
        // FIXME: Apply only transformation rules during exploration phase
        if !explore && rule.group_rule() {
            continue;
        }
        if let Some(rule_match) = rule.matches(&rule_ctx, expr.expr().as_logical()) {
            let binding = RuleBinding {
                rule_id: *rule_id,
                expr: expr.clone(),
                group: match rule_match {
                    RuleMatch::Expr => None,
                    RuleMatch::Group => Some(ctx.group.clone()),
                },
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

    runtime_state.tasks.schedule(Task::ExprOptimized {
        ctx: ctx.clone(),
        expr: expr.clone(),
    });

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
    metadata: &Metadata,
) where
    R: RuleSet,
{
    runtime_state.stats.tasks.apply_rule += 1;
    let expr_id = expr.id();
    let rule_id = binding.rule_id;
    let group_id = ctx.group_id();

    if runtime_state.applied_rules.is_applied(&rule_id, &expr_id, &group_id) {
        // panic!("Rule#{} has already been applied. Binding: {:?}", rule_id, binding);
        return;
    }

    runtime_state.applied_rules.mark_applied(&rule_id, &binding);

    let result = {
        let operator = expr.expr();
        let rule_ctx = RuleContext::new(ctx.required_properties.clone(), metadata);
        rule_set
            .apply_rule(&rule_id, &rule_ctx, operator.as_logical())
            .expect("Failed to apply a rule")
    };

    if let Some(result) = result {
        match result {
            RuleResult::Substitute(expr) => {
                let new_operator = OperatorExpr::Logical(Box::new(expr));
                let (_, new_expr) = memo.insert_member(&ctx.group, Operator::from(new_operator));

                log::debug!(" + Logical expression: {}", &new_expr);

                runtime_state.tasks.schedule(Task::OptimizeExpr {
                    ctx,
                    expr: new_expr,
                    explore,
                });
            }
            RuleResult::Implementation(expr) => {
                let new_operator = OperatorExpr::Physical(Box::new(expr));
                let new_operator = Operator::from(new_operator);
                let (_, new_expr) = memo.insert_member(&ctx.group, new_operator);

                log::debug!(" + Physical expression: {}", &new_expr);

                let task = get_optimize_inputs_task(&ctx, &new_expr, rule_set);
                runtime_state.tasks.schedule(task);
            }
        }
    }
}

fn enforce_properties<R>(
    runtime_state: &mut RuntimeState,
    memo: &mut ExprMemo,
    ctx: OptimizationContext,
    expr: ExprRefOption,
    rule_set: &R,
) where
    R: RuleSet,
{
    runtime_state.stats.tasks.enforce_properties += 1;

    assert!(
        !ctx.required_properties.is_empty(),
        "Can not apply an enforcer - no required properties. ctx: {}, expr: {}",
        ctx,
        expr
    );

    let (enforcer_expr, remaining_properties) = {
        let input = InputExpr::Group(ctx.group.clone());

        let (enforcer_expr, remaining_properties) = rule_set
            .create_enforcer(&ctx.required_properties, input)
            .expect("Failed to apply an enforcer");
        let enforcer_expr = OperatorExpr::from(enforcer_expr);
        // ENFORCERS: Currently enforcers can not be retrieved from group.exprs()
        // because adding an enforcer to a group won't allow the optimizer to complete the optimization process:
        // 1) When an enforcer is added from explore_group task the next optimize_group task will call
        // optimize_inputs on the enforcer which then calls optimize group on the same group and we have an infinite loop.
        // 2) copy_out_best_expr traverses its input expressions recursively so adding an enforcer to a group which
        // is used as an input expression results an infinite loop during the traversal.
        let (_, enforcer_expr) = memo.insert(Operator::from(enforcer_expr));

        (enforcer_expr, remaining_properties)
    };

    let ExprRefOption { expr } = expr;
    if let Some(expr) = expr {
        runtime_state.tasks.schedule(Task::OptimizeInputs {
            ctx: ctx.clone(),
            expr: enforcer_expr,
            inputs: InputContexts::completed(ctx.group.clone(), remaining_properties.clone()),
        });

        let input_ctx = OptimizationContext {
            group: ctx.group,
            required_properties: remaining_properties,
        };
        // combine with remaining_properties ???
        let required_properties = expr.expr().as_physical().build_required_properties();
        let inputs = InputContexts::with_required_properties(&expr, required_properties);

        runtime_state.tasks.schedule(Task::OptimizeInputs {
            ctx: input_ctx,
            expr,
            inputs,
        })
    } else {
        let inputs = InputContexts::new(&enforcer_expr, remaining_properties);

        runtime_state.tasks.schedule(Task::OptimizeInputs {
            ctx,
            expr: enforcer_expr,
            inputs,
        })
    }
}

fn optimize_inputs<T>(
    runtime_state: &mut RuntimeState,
    ctx: OptimizationContext,
    expr: ExprRef,
    mut inputs: InputContexts,
    cost_estimator: &T,
) where
    T: CostEstimator,
{
    runtime_state.stats.tasks.optimize_inputs += 1;

    let expr_id = expr.id();

    if let Some(input_ctx) = inputs.next_input() {
        runtime_state.tasks.schedule(Task::OptimizeInputs { ctx, expr, inputs });

        runtime_state.tasks.schedule(Task::OptimizeGroup { ctx: input_ctx })
    } else {
        let group = &ctx.group;
        let logical_properties = group.attrs().logical();
        let statistics = logical_properties.statistics();
        let cost_ctx = new_cost_estimation_ctx(&inputs, &runtime_state.state);
        let cost = cost_estimator.estimate_cost(expr.expr().as_physical(), &cost_ctx, statistics);

        log::debug!("Expr cost: {} ctx: {} expr: {} {}", cost, ctx, expr_id, expr);

        let candidate = BestExpr {
            expr,
            cost,
            inputs: inputs.inputs,
        };

        let mut state = runtime_state.state.init_or_get_state(&ctx);
        state.mark_expr_optimized(expr_id);
        match state.best_expr.as_mut() {
            None => state.best_expr = Some(candidate),
            Some(best) if best.cost > cost => *best = candidate,
            _ => {}
        }
    }
}

fn get_optimize_inputs_task<R>(ctx: &OptimizationContext, expr: &ExprRef, rule_set: &R) -> Task
where
    R: RuleSet,
{
    let physical_expr = expr.expr().as_physical();
    let required_properties = physical_expr.build_required_properties();
    let (provides_property, retains_property) = rule_set
        .evaluate_properties(physical_expr, &ctx.required_properties)
        .expect("Invalid expr or required physical properties");

    if ctx.required_properties.is_empty() && required_properties.is_none() {
        // Optimization context has no required properties + expression does not require any property from its inputs.
        // -> optimize each input expression using required properties for group attributes (if any)
        let inputs = InputContexts::from_inputs(expr);
        Task::OptimizeInputs {
            ctx: ctx.clone(),
            expr: expr.clone(),
            inputs,
        }
    } else if retains_property {
        // Expression retains the required properties.
        // -> optimize each input expression with required properties from the current optimization context.
        //??? combine with required properties

        let inputs = InputContexts::new(expr, ctx.required_properties.clone());
        Task::OptimizeInputs {
            ctx: ctx.clone(),
            expr: expr.clone(),
            inputs,
        }
    } else if provides_property {
        // Expression can provide the required properties OR
        // -> optimize input expressions with properties required by the expression.
        // OR
        // Expression requires properties from its inputs.
        // -> same as the above

        // FIXME: MergeSort -> 'provides' conflicts with 'retains'
        let inputs = InputContexts::with_required_properties(expr, required_properties);
        Task::OptimizeInputs {
            ctx: ctx.clone(),
            expr: expr.clone(),
            inputs,
        }
    } else {
        // Expression can not provide the required properties itself
        // -> add an enforcer operator.
        Task::EnforceProperties {
            ctx: ctx.clone(),
            expr: ExprRefOption::some(expr.clone()),
        }
    }
}

fn expr_optimized(runtime_state: &mut RuntimeState, ctx: OptimizationContext, expr: ExprRef) {
    let state = runtime_state.state.init_or_get_state(&ctx);
    let expr_id = expr.id();

    state.mark_expr_optimized(expr_id);
}

fn explore_group<R>(runtime_state: &mut RuntimeState, ctx: OptimizationContext, rule_set: &R)
where
    R: RuleSet,
{
    if !runtime_state.enable_explore_groups {
        return;
    }

    if rule_set.can_explore_with_enforcer(ctx.group.expr().as_logical(), &ctx.required_properties) {
        runtime_state.tasks.schedule(Task::EnforceProperties {
            ctx: ctx.clone(),
            expr: ExprRefOption::none(),
        })
    }

    for expr in ctx.group.mexprs() {
        match expr.expr() {
            OperatorExpr::Logical(_) => runtime_state.tasks.schedule(Task::OptimizeExpr {
                ctx: ctx.clone(),
                expr: expr.clone(),
                explore: true,
            }),
            OperatorExpr::Physical(_) => {}
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct OptimizationContext {
    group: GroupRef,
    required_properties: PhysicalProperties,
}

impl OptimizationContext {
    fn group_id(&self) -> GroupId {
        self.group.id()
    }
}

impl Display for OptimizationContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ group: {}", self.group.id())?;
        if !self.required_properties.is_empty() {
            write!(f, " {}", self.required_properties)?;
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

#[derive(Debug, Clone, Default)]
pub(crate) struct Stats {
    number_of_tasks: usize,
    max_stack_depth: usize,
    optimization_time: Duration,
    result_time: Duration,
    tasks: TaskStats,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TaskStats {
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

    fn get_state(&self, ctx: &OptimizationContext) -> &GroupState {
        self.groups.get(ctx).expect("Group state should have been created")
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

#[derive(Debug)]
struct BestExpr {
    expr: ExprRef,
    cost: Cost,
    inputs: Vec<OptimizationContext>,
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
pub struct RuleBinding {
    rule_id: RuleId,
    expr: ExprRef,
    group: Option<GroupRef>,
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
            self.groups.insert((*rule_id, group.id()));
        }
    }
}

#[derive(Debug)]
struct InputContexts {
    inputs: Vec<OptimizationContext>,
    current_index: usize,
}

impl InputContexts {
    fn with_required_properties(expr: &ExprRef, required_properties: Option<Vec<PhysicalProperties>>) -> Self {
        let required_properties = match required_properties {
            None => (0..expr.inputs().len()).map(|_| PhysicalProperties::none()).collect(),
            Some(props) => {
                let num = expr.inputs().len();
                assert_eq!(props.len(), num,
                           "Number of required properties must be equal to the number of input expressions. Expected: {} but got {}", num, props.len());
                props
            }
        };
        let inputs = expr
            .inputs()
            .iter()
            .zip(required_properties.into_iter())
            .map(|(group, required)| OptimizationContext {
                group: group.clone(),
                required_properties: required,
            })
            .collect();
        InputContexts {
            inputs,
            current_index: 0,
        }
    }

    fn new(expr: &ExprRef, required_properties: PhysicalProperties) -> Self {
        InputContexts {
            inputs: expr
                .inputs()
                .iter()
                .map(|group| OptimizationContext {
                    group: group.clone(),
                    required_properties: required_properties.clone(),
                })
                .collect(),
            current_index: 0,
        }
    }

    fn completed(group: GroupRef, required_properties: PhysicalProperties) -> Self {
        let mut ctx = InputContexts {
            inputs: vec![OptimizationContext {
                group,
                required_properties,
            }],
            current_index: 0,
        };
        ctx.next_input();
        ctx
    }

    fn from_inputs(expr: &ExprRef) -> Self {
        let inputs = expr
            .inputs()
            .iter()
            .map(|group| {
                let required_properties = group.attrs().required();
                OptimizationContext {
                    group: group.clone(),
                    required_properties: required_properties.clone(),
                }
            })
            .collect();
        InputContexts {
            inputs,
            current_index: 0,
        }
    }

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

impl Display for InputContexts {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ inputs=[{}]", self.inputs.iter().join(", "))?;
        write!(f, " current_index={}", self.current_index)?;
        write!(f, "]")
    }
}

fn new_cost_estimation_ctx(inputs: &InputContexts, state: &State) -> CostEstimationContext {
    let capacity = inputs.inputs.len();
    let mut input_groups = Vec::with_capacity(capacity);
    let mut input_costs = Vec::with_capacity(capacity);

    for ctx in inputs.inputs.iter() {
        let group_state = state.get_state(ctx);
        let cost = group_state.best_expr.as_ref().map(|e| e.cost);
        match cost {
            Some(cost) => {
                let group = ctx.group.clone();
                input_groups.push(group);
                input_costs.push(cost);
            }
            //FIXME: Is this really a hard error?
            None => panic!("No best expr for input group: {:?}", ctx),
        }
    }

    CostEstimationContext {
        input_costs,
        input_groups,
    }
}

#[derive(Debug)]
struct OptimizerResultCallbackContext<'o> {
    ctx: &'o OptimizationContext,
    best_expr: &'o BestExpr,
    inputs: &'o [OptimizationContext],
}

impl<'o> BestExprContext for OptimizerResultCallbackContext<'o> {
    fn cost(&self) -> Cost {
        self.best_expr.cost
    }

    fn logical(&self) -> &LogicalProperties {
        self.ctx.group.attrs().logical()
    }

    fn required(&self) -> &PhysicalProperties {
        self.ctx.group.attrs().required()
    }

    fn group_id(&self) -> GroupId {
        self.ctx.group_id()
    }

    fn inputs_num(&self) -> usize {
        self.inputs.len()
    }

    fn input_group_id(&self, i: usize) -> GroupId {
        self.inputs[i].group_id()
    }

    fn input_required(&self, i: usize) -> &PhysicalProperties {
        &self.inputs[i].required_properties
    }
}

fn copy_out_best_expr<'o, T>(
    result_callback: &T,
    state: &'o State,
    memo: &'o ExprMemo,
    ctx: &'o OptimizationContext,
) -> Result<Operator, String>
where
    T: ResultCallback,
{
    let group_state = state.get_state(ctx);
    match &group_state.best_expr {
        Some(best_expr) => {
            let operator = best_expr.expr.expr();
            match operator {
                OperatorExpr::Logical(expr) => {
                    Err(format!("Invalid best expression. Expected a physical operator but got: {:?}", expr))
                }
                OperatorExpr::Physical(expr) => {
                    let inputs = &best_expr.inputs[0..best_expr.inputs.len()];
                    let mut new_inputs = Vec::with_capacity(inputs.len());

                    for input_ctx in &best_expr.inputs {
                        let out = copy_out_best_expr(result_callback, state, memo, input_ctx)?;
                        new_inputs.push(InputExpr::from(out));
                    }

                    let best_expr_ctx = OptimizerResultCallbackContext { ctx, best_expr, inputs };
                    result_callback.on_best_expr(expr, &best_expr_ctx);
                    let result = Operator::from(OperatorExpr::from(expr.with_new_inputs(new_inputs)));
                    Ok(result)
                }
            }
        }
        None => Err(format!("No best expression for ctx: {}", ctx)),
    }
}
