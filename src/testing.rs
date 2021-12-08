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
use crate::memo::{ExprNodeRef, MemoExpr, MemoExprFormatter, StringMemoFormatter};
use crate::meta::{Metadata, MutableMetadata};
use crate::operators::builder::{NoMemoization, OperatorBuilder};

use crate::operators::{ExprMemo, Operator};
use crate::optimizer::{Optimizer, SetPropertiesCallback};
use crate::properties::logical::LogicalPropertiesBuilder;
use crate::properties::statistics::CatalogStatisticsBuilder;
use crate::rules::implementation::{GetToScanRule, ProjectionRule, SelectRule};
use crate::rules::testing::TestRuleSet;
use crate::rules::Rule;
use crate::rules::StaticRuleSet;
use crate::util::{BestExprContext, BestExprRef, ResultCallback};

static INIT_LOG: Once = Once::new();

/// Provides a test setup for the [optimizer].
///
/// [optimizer]: crate::optimizer::Optimizer
pub struct OptimizerTester {
    catalog: CatalogRef,
    properties_builder: Rc<LogicalPropertiesBuilder>,
    mutable_metadata: Rc<MutableMetadata>,
    rules: Box<dyn Fn(CatalogRef) -> Vec<Box<dyn Rule>>>,
    rules_filter: Box<dyn Fn(&Box<dyn Rule>) -> bool>,
    table_access_costs: HashMap<String, usize>,
    shuffle_rules: bool,
    explore_with_enforcer: bool,
    update_catalog: Box<dyn Fn(&MutableCatalog)>,
    use_metadata: Option<Metadata>,
}

impl OptimizerTester {
    pub fn new() -> Self {
        INIT_LOG.call_once(pretty_env_logger::init);

        let catalog = Arc::new(MutableCatalog::new());
        let statistics_builder = CatalogStatisticsBuilder::new(catalog.clone());
        let properties_builder = Rc::new(LogicalPropertiesBuilder::new(Box::new(statistics_builder)));
        let mutable_metadata = Rc::new(MutableMetadata::new());

        OptimizerTester {
            catalog,
            properties_builder,
            mutable_metadata,
            rules: Box::new(|_| Vec::new()),
            rules_filter: Box::new(|_r| true),
            table_access_costs: HashMap::new(),
            shuffle_rules: true,
            explore_with_enforcer: true,
            update_catalog: Box::new(|_| {}),
            use_metadata: None,
        }
    }

    /// Creates a new [operator builder](crate::operators::builder::OperatorBuilder).
    pub fn builder(&self) -> OperatorBuilder {
        let tables = TestTables::new();

        tables.register(self.catalog.as_ref());

        // Currently instances of OperatorBuilders live between calls to Tester::optimize so it is not possible
        // to retrieve a mutable reference memo.
        let memoization = Rc::new(NoMemoization);
        OperatorBuilder::new(
            memoization,
            self.catalog.clone(),
            self.mutable_metadata.clone(),
            self.properties_builder.clone(),
        )
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

    pub fn set_table_access_cost(&mut self, table: &str, cost: usize) {
        self.table_access_costs.insert(table.into(), cost);
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
        F: Fn(&MutableCatalog) + 'static,
    {
        self.update_catalog = Box::new(f);
    }

    pub fn build_query(&mut self, builder: OperatorBuilder) -> Operator {
        let (operator, metadata) = builder.build();
        self.use_metadata = Some(metadata);
        operator
    }

    /// Optimizes the given operator tree and compares the result with expected plan.
    /// `expected_plan` can be written in two forms with description (1) or without description (2):
    ///
    /// 1. With description:
    ///
    ///     query: Some description
    ///     expr 1
    ///     expr 2
    ///
    /// 2. Without description:
    ///
    ///     expr 1
    ///     expr 2
    ///
    pub fn optimize(&mut self, operator: Operator, expected_plan: &str) -> Result<(), OptimizerError> {
        let tables = TestTables::new();
        let _columns = tables.columns.clone();

        let (operator, metadata) = if let Some(metadata) = self.use_metadata.take() {
            let mutable_catalog = self.catalog.as_any().downcast_ref::<MutableCatalog>().unwrap();

            tables.register_statistics(mutable_catalog, self.table_access_costs.clone());

            (operator, metadata)
        } else {
            // let builder =
            //     TestOperatorTreeBuilder::new(&mut memo, self.catalog.clone(), columns, self.table_access_costs.clone());
            // builder.build_initialized(operator)
            panic!("Use OperatorBuilder to create an operator")
        };

        let mutable_catalog = self.catalog.as_any().downcast_ref::<MutableCatalog>().unwrap();
        (self.update_catalog)(mutable_catalog);

        let mut default_rules: Vec<Box<dyn Rule>> = vec![
            Box::new(GetToScanRule::new(self.catalog.clone())),
            Box::new(SelectRule),
            Box::new(ProjectionRule),
        ];
        default_rules.extend((self.rules)(self.catalog.clone()));

        let rules: Vec<Box<dyn Rule>> = default_rules.into_iter().filter(|r| (self.rules_filter)(r)).collect();
        assert!(!rules.is_empty(), "No rules have been specified");

        let rules = TestRuleSet::new(StaticRuleSet::new(rules), self.shuffle_rules, self.explore_with_enforcer);
        let cost_estimator = SimpleCostEstimator::new();
        let test_result = Rc::new(RefCell::new(VecDeque::new()));
        let result_callback = TestResultBuilder::new(test_result.clone());

        let optimizer = Optimizer::new(Rc::new(rules), Rc::new(cost_estimator), Rc::new(result_callback));

        let propagate_properties = SetPropertiesCallback::new(self.properties_builder.clone());
        let memo_callback = Rc::new(propagate_properties);
        let mut memo = ExprMemo::with_callback(memo_callback);

        let _opt_expr = optimizer
            .optimize(operator, metadata, &mut memo)
            .expect("Failed to optimize an operator tree");
        let expected_lines: Vec<String> = expected_plan.split('\n').map(|l| l.to_string()).collect();
        let label = if expected_plan.starts_with("query:") {
            expected_lines[0].trim()
        } else {
            "plan"
        };

        let test_result = TestResultBuilder::result_as_string(test_result);
        let expected_plan = expected_lines.iter().skip(1).map(|s| s.as_ref()).collect::<Vec<&str>>().join("\n");
        assert_eq!(test_result, expected_plan, "{} does not match", label);
        Ok(())
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

    pub fn register(&self, catalog: &dyn Catalog) {
        let mutable_catalog = catalog.as_any().downcast_ref::<MutableCatalog>().unwrap();

        for table in self.columns.clone() {
            let table_name = table.0;
            let columns = table.1;
            let mut table_builder = TableBuilder::new(table_name.as_str());
            for (name, tpe) in columns {
                table_builder = table_builder.add_column(name.as_str(), tpe);
            }
            let table = table_builder.build();
            mutable_catalog.add_table(DEFAULT_SCHEMA, table);
        }
    }

    fn register_statistics(&self, catalog: &dyn Catalog, statistics: HashMap<String, usize>) {
        let mutable_catalog = catalog.as_any().downcast_ref::<MutableCatalog>().unwrap();

        for (name, cost) in statistics {
            if name.starts_with("Index:") {
                continue;
            }
            let table = mutable_catalog
                .get_table(name.as_str())
                .unwrap_or_else(|| panic!("Unexpected table {}", name));
            let table_builder = TableBuilder::from_table(table.as_ref());
            let table = table_builder.add_row_count(cost).build();

            mutable_catalog.add_table(DEFAULT_SCHEMA, table);
        }
    }
}

#[derive(Debug)]
struct TestResultBuilder {
    content: Rc<RefCell<VecDeque<String>>>,
}

impl ResultCallback for TestResultBuilder {
    fn on_best_expr<C>(&self, expr: BestExprRef, ctx: &C)
    where
        C: BestExprContext,
    {
        let group_id = ctx.group_id();
        let mut buf = String::new();
        buf.push_str(format!("{:02} ", group_id).as_str());
        let fmt = StringMemoFormatter::new(&mut buf);
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
    // buf: &'a mut String,
    fmt: StringMemoFormatter<'a>,
    ctx: &'c C,
    input_index: usize,
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

    fn write_expr<'e, T>(&mut self, _name: &str, _input: impl Into<ExprNodeRef<'e, T>>)
    where
        T: MemoExpr + 'e,
    {
        self.fmt.push(' ');

        let i = self.input_index;
        let input_group_id = self.ctx.child_group_id(i);
        let input_required = self.ctx.child_required(i);

        if i == 0 {
            self.fmt.push('[');
        }
        match input_required.as_option() {
            None | Some(None) => {}
            Some(Some(ordering)) => {
                self.fmt.push_str(format!("ord:{:?}=", ordering.columns()).as_str());
            }
        }
        self.fmt.push_str(format!("{:02}", input_group_id).as_str());

        self.input_index += 1;
        if self.input_index == self.ctx.num_children() {
            self.fmt.push(']');
        }
    }

    fn write_value<D>(&mut self, name: &str, value: D)
    where
        D: Display,
    {
        self.fmt.write_value(name, value);
    }

    fn write_values<D>(&mut self, name: &str, values: &[D])
    where
        D: Display,
    {
        self.fmt.write_values(name, values);
    }
}
