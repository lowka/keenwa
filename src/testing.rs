use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fmt::Display;
use std::rc::Rc;
use std::sync::{Arc, Once};

use crate::catalog::mutable::MutableCatalog;
use crate::catalog::{Catalog, CatalogRef, TableBuilder, DEFAULT_SCHEMA};
use crate::cost::simple::SimpleCostEstimator;
use crate::datatypes::DataType;
use crate::error::OptimizerError;
use crate::memo::{MemoExpr, MemoExprFormatter, MemoFormatterFlags, StringMemoFormatter};
use crate::meta::MutableMetadata;
use crate::operators::builder::{MemoizeOperators, OperatorBuilder};
use crate::operators::properties::LogicalPropertiesBuilder;
use crate::operators::{Operator, OperatorMemoBuilder};
use crate::optimizer::{BestExprContext, BestExprRef, OptimizedExprCallback, Optimizer};
use crate::rules::implementation::{EmptyRule, GetToScanRule, LimitOffsetRule, ProjectionRule, SelectRule, ValuesRule};
use crate::rules::testing::TestRuleSet;
use crate::rules::RuleSet;
use crate::rules::{Rule, StaticRuleSetBuilder};
use crate::statistics::simple::{PrecomputedSelectivityStatistics, SimpleCatalogStatisticsBuilder};

static INIT_LOG: Once = Once::new();

/// Provides a test setup for the [optimizer].
///
/// [optimizer]: crate::optimizer::Optimizer
pub struct OptimizerTester {
    rules: Box<dyn Fn(CatalogRef) -> Vec<Box<dyn Rule>>>,
    rules_filter: Box<dyn Fn(&Box<dyn Rule>) -> bool>,
    shuffle_rules: bool,
    explore_with_enforcer: bool,
    row_count_per_table: HashMap<String, usize>,
    update_selectivity: Box<dyn Fn(&PrecomputedSelectivityStatistics)>,
    operator: Box<dyn Fn(OperatorBuilder) -> Result<Operator, OptimizerError>>,
    update_catalog: Box<dyn Fn(&MutableCatalog) -> Result<(), OptimizerError>>,
}

impl OptimizerTester {
    pub fn new() -> Self {
        INIT_LOG.call_once(pretty_env_logger::init);

        OptimizerTester {
            rules: Box::new(|_| Vec::new()),
            rules_filter: Box::new(|_r| true),
            shuffle_rules: true,
            explore_with_enforcer: true,
            row_count_per_table: HashMap::new(),
            operator: Box::new(|_| panic!("operator has not been specified")),
            update_selectivity: Box::new(|_| {}),
            update_catalog: Box::new(|_| Ok(())),
        }
    }

    /// A toggle to enable/disable rule shuffling.
    pub fn shuffle_rules(&mut self, value: bool) {
        self.shuffle_rules = value;
    }

    /// See [RuleSet#can_explore_with_enforcer] for details.
    ///
    /// [RuleSet#can_explore_with_enforcer]: crate::rules::RuleSet::can_explore_with_enforcer()
    pub fn explore_with_enforcer(&mut self, value: bool) {
        self.explore_with_enforcer = value
    }

    /// Adds additional rules to the optimizer.
    /// The initial configuration contains implementation rules for basic operators (selection, projection, retrieval from source etc.).
    pub fn add_rules<F>(&mut self, f: F)
    where
        F: Fn(CatalogRef) -> Vec<Box<dyn Rule>> + 'static,
    {
        self.rules = Box::new(f);
    }

    /// Rules that satisfy the given predicate won't be used by the optimizer.
    pub fn disable_rules<F>(&mut self, f: F)
    where
        F: Fn(&Box<dyn Rule>) -> bool + 'static,
    {
        self.rules_filter = Box::new(move |r| !(f)(r));
    }

    /// Resets a filter set by [Self::disable_rules()] method.
    pub fn reset_rule_filters(&mut self) {
        self.rules_filter = Box::new(|_r| true);
    }

    /// Sets a function that can modify a database catalog used by the optimizer.
    pub fn update_catalog<F>(&mut self, f: F)
    where
        F: Fn(&MutableCatalog) -> Result<(), OptimizerError> + 'static,
    {
        self.update_catalog = Box::new(f);
    }

    /// Sets row count for the given table.
    pub fn set_table_row_count(&mut self, table: &str, row_count: usize) {
        self.row_count_per_table.insert(table.into(), row_count);
    }

    /// Updates selectivity statistics
    //FIXME: Combine with set_table_access_cost
    pub fn update_statistics<F>(&mut self, f: F)
    where
        F: Fn(&PrecomputedSelectivityStatistics) + 'static,
    {
        self.update_selectivity = Box::new(f)
    }

    /// Set a function that produces an operator tree.
    pub fn set_operator<F>(&mut self, f: F)
    where
        F: Fn(OperatorBuilder) -> Result<Operator, OptimizerError> + 'static,
    {
        self.operator = Box::new(f)
    }

    /// Optimizes the given operator tree and compares the result with expected plan.
    /// `expected_plan` can be written in two forms with description (1) or without description (2):
    ///
    /// 1. With description:
    /// ```text
    ///     query: Some description
    ///     expr 1
    ///     expr 2
    ///```
    /// 2. Without description:
    ///```text
    ///     expr 1
    ///     expr 2
    ///```
    pub fn optimize(&mut self, expected_plan: &str) {
        let catalog = Arc::new(MutableCatalog::new());
        let tables = TestTables::new();
        let _columns = tables.columns.clone();

        tables.register(catalog.as_ref(), &self.row_count_per_table);
        // {
        //     let mutable_catalog = self.catalog.as_any().downcast_ref::<MutableCatalog>().unwrap();
        //     tables.register_statistics(mutable_catalog, self.row_count_per_table.clone());
        // }

        let selectivity_provider = Rc::new(PrecomputedSelectivityStatistics::new());
        let statistics_builder = SimpleCatalogStatisticsBuilder::new(catalog.clone(), selectivity_provider.clone());
        let properties_builder = LogicalPropertiesBuilder::new(statistics_builder);

        let metadata = Rc::new(MutableMetadata::new());
        let memo = OperatorMemoBuilder::new(metadata.clone()).build_with_properties(properties_builder);

        let mut memoization = MemoizeOperators::new(memo);
        let mutable_catalog = catalog.as_any().downcast_ref::<MutableCatalog>().unwrap();

        (self.update_catalog)(mutable_catalog).expect("Failed to update a catalog");
        (self.update_selectivity)(selectivity_provider.as_ref());

        let builder = OperatorBuilder::new(memoization.take_callback(), catalog.clone(), metadata);
        let operator = (self.operator)(builder).expect("Failed to build an operator");

        let mut default_rules: Vec<Box<dyn Rule>> = vec![
            Box::new(GetToScanRule::new(catalog.clone())),
            Box::new(SelectRule),
            Box::new(ProjectionRule),
            Box::new(LimitOffsetRule),
            Box::new(ValuesRule),
            Box::new(EmptyRule),
        ];
        default_rules.extend((self.rules)(catalog.clone()));

        let rules = default_rules.into_iter().filter(|r| (self.rules_filter)(r));
        let rule_set = StaticRuleSetBuilder::new()
            .add_rules(rules)
            .explore_with_enforcer(self.explore_with_enforcer)
            .build();

        assert!(!rule_set.get_rules().count() > 0, "No rules have been specified");

        let rules = TestRuleSet::new(rule_set, self.shuffle_rules);
        let cost_estimator = SimpleCostEstimator::new();
        let test_result = Rc::new(RefCell::new(VecDeque::new()));
        let result_callback = TestResultBuilder::new(test_result.clone());

        let optimizer = Optimizer::with_callback(Arc::new(rules), Arc::new(cost_estimator), Arc::new(result_callback));

        let mut memo = memoization.into_memo();

        let _opt_expr = optimizer.optimize(operator, &mut memo).expect("Failed to optimize an operator tree");
        let expected_lines: Vec<String> = expected_plan.split('\n').map(|l| l.to_string()).collect();
        let label = if expected_plan.starts_with("query:") {
            expected_lines[0].trim()
        } else {
            "plan"
        };

        let test_result = TestResultBuilder::result_as_string(test_result);
        let expected_plan = expected_lines.iter().skip(1).map(|s| s.as_ref()).collect::<Vec<&str>>().join("\n");
        assert_eq!(test_result, expected_plan, "{} does not match", label);
    }
}

pub struct TestTables {
    columns: Vec<(String, Vec<(String, DataType)>)>,
}

impl TestTables {
    pub fn new() -> Self {
        TestTables {
            columns: vec![
                ("A".into(), vec![("a1".into(), DataType::Int32), ("a2".into(), DataType::Int32)]),
                ("B".into(), vec![("b1".into(), DataType::Int32), ("b2".into(), DataType::Int32)]),
                ("C".into(), vec![("c1".into(), DataType::Int32), ("c2".into(), DataType::Int32)]),
            ],
        }
    }

    pub fn register(&self, catalog: &dyn Catalog, statistics: &HashMap<String, usize>) {
        let mutable_catalog = catalog.as_any().downcast_ref::<MutableCatalog>().unwrap();

        for (table_name, columns) in self.columns.iter() {
            let mut table_builder = TableBuilder::new(table_name.as_str());
            for (name, tpe) in columns {
                table_builder = table_builder.add_column(name.as_str(), tpe.clone());
            }
            if let Some(rows) = statistics.get(table_name) {
                table_builder = table_builder.add_row_count(*rows);
            }
            let table = table_builder.build().expect("Test table");
            mutable_catalog.add_table(DEFAULT_SCHEMA, table).expect("Failed to add a table");
        }
    }
}

#[derive(Debug)]
struct TestResultBuilder {
    content: Rc<RefCell<VecDeque<String>>>,
}

impl OptimizedExprCallback for TestResultBuilder {
    fn on_best_expr<C>(&self, expr: BestExprRef, ctx: &C)
    where
        C: BestExprContext,
    {
        let group_id = ctx.group_id();
        let mut buf = String::new();
        buf.push_str(format!("{:02} ", group_id).as_str());
        let fmt = StringMemoFormatter::new(&mut buf, MemoFormatterFlags::All);
        let mut fmt = BestExprFormatter::new(fmt, ctx);
        match expr {
            BestExprRef::Relational(expr) => expr.format_expr(&mut fmt),
            BestExprRef::Scalar(expr) => expr.format_expr(&mut fmt),
        }

        let mut content = self.content.borrow_mut();
        content.push_front(buf);
    }
}

impl TestResultBuilder {
    fn new(content: Rc<RefCell<VecDeque<String>>>) -> Self {
        TestResultBuilder { content }
    }

    fn result_as_string(content: Rc<RefCell<VecDeque<String>>>) -> String {
        let content = content.borrow();
        let mut buf = String::new();
        for e in content.iter() {
            buf.push_str(e);
            buf.push('\n');
        }
        buf
    }
}

struct BestExprFormatter<'a, 'c, C>
where
    C: BestExprContext,
{
    fmt: StringMemoFormatter<'a>,
    ctx: &'c C,
    input_index: usize,
    exprs_written: bool,
}

impl<'a, 'c, C> BestExprFormatter<'a, 'c, C>
where
    C: BestExprContext,
{
    fn new(fmt: StringMemoFormatter<'a>, ctx: &'c C) -> Self {
        BestExprFormatter {
            fmt,
            ctx,
            input_index: 0,
            exprs_written: false,
        }
    }

    fn do_write_expr(&mut self) {
        let i = self.input_index;
        let input_group_id = self.ctx.child_group_id(i);
        let input_required = self.ctx.child_required(i);

        if i == 0 {
            self.fmt.push('[');
        }
        match input_required {
            None => {}
            Some(required) => {
                if let Some(ordering) = required.ordering() {
                    self.fmt.push_str(format!("ord:{}=", ordering).as_str());
                } else {
                    panic!("Unexpected required property!: {:?}", required)
                }
            }
        }
        self.fmt.push_str(format!("{:02}", input_group_id).as_str());
        self.input_index += 1;
    }

    fn exprs_written(&mut self) {
        if !self.exprs_written && self.input_index == self.ctx.num_children() {
            self.fmt.push(']');
            self.exprs_written = true;
        }
    }
}

impl<'a, 'c, C> MemoExprFormatter for BestExprFormatter<'a, 'c, C>
where
    C: BestExprContext,
{
    fn write_name(&mut self, name: &str) {
        self.fmt.write_name(name);
    }

    fn write_source(&mut self, source: &str) {
        self.fmt.write_source(source);
    }

    fn write_expr<T>(&mut self, _name: &str, _input: impl AsRef<T>)
    where
        T: MemoExpr,
    {
        self.fmt.push(' ');
        self.do_write_expr();
        self.exprs_written();
    }

    fn write_exprs<T>(&mut self, _name: &str, input: impl ExactSizeIterator<Item = impl AsRef<T>>) {
        for _ in input {
            self.fmt.push(' ');
            self.do_write_expr();
        }
        self.exprs_written();
    }

    fn write_value<D>(&mut self, name: &str, value: D)
    where
        D: Display,
    {
        self.fmt.write_value(name, value);
    }

    fn write_values<D>(&mut self, name: &str, values: impl ExactSizeIterator<Item = D>)
    where
        D: Display,
    {
        self.fmt.write_values(name, values);
    }

    fn flags(&self) -> &MemoFormatterFlags {
        self.fmt.flags()
    }
}
