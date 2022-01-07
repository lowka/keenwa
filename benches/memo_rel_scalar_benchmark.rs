use criterion::{black_box, criterion_group, criterion_main, Criterion};
use keenwa::memo::{
    CopyInExprs, CopyInNestedExprs, Expr, ExprContext, ExprGroupPtr, ExprPtr, Memo, MemoExpr, MemoExprFormatter,
    NewChildExprs, Props,
};
use std::fmt::{Display, Formatter};

type RelNode = keenwa::memo::RelNode<TestOperator>;
type ScalarNode = keenwa::memo::ScalarNode<TestOperator>;

#[derive(Debug, Clone)]
enum TestExpr {
    Relational(TestRelExpr),
    Scalar(TestScalarExpr),
}

#[derive(Debug, Clone)]
enum TestRelExpr {
    Scan { src: &'static str },
    Filter { input: RelNode, filter: ScalarNode },
    Join { left: RelNode, right: RelNode },
}

impl Expr for TestExpr {
    type RelExpr = TestRelExpr;
    type ScalarExpr = TestScalarExpr;

    fn new_rel(expr: Self::RelExpr) -> Self {
        TestExpr::Relational(expr)
    }

    fn new_scalar(expr: Self::ScalarExpr) -> Self {
        TestExpr::Scalar(expr)
    }

    fn relational(&self) -> &Self::RelExpr {
        unreachable!()
    }

    fn scalar(&self) -> &Self::ScalarExpr {
        unreachable!()
    }

    fn is_scalar(&self) -> bool {
        match self {
            TestExpr::Relational(_) => false,
            TestExpr::Scalar(_) => true,
        }
    }
}

#[derive(Debug, Clone)]
enum TestScalarExpr {
    Value(i32),
    Gt {
        lhs: Box<TestScalarExpr>,
        rhs: Box<TestScalarExpr>,
    },
    SubQuery(RelNode),
}

impl Display for TestScalarExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TestScalarExpr::Value(value) => write!(f, "{}", value),
            TestScalarExpr::Gt { lhs, rhs } => {
                write!(f, "{} > {}", lhs, rhs)
            }
            TestScalarExpr::SubQuery(rel_node) => match rel_node.expr_ref() {
                ExprPtr::Owned(expr) => {
                    let ptr: *const TestExpr = &**expr;
                    write!(f, "SubQuery expr_ptr {:?}", ptr)
                }
                ExprPtr::Memo(group) => {
                    write!(f, "SubQuery {}", group.id())
                }
            },
        }
    }
}

impl TestScalarExpr {
    fn with_new_inputs(&self, inputs: &mut NewChildExprs<TestOperator>) -> Self {
        match self {
            TestScalarExpr::Value(_) => self.clone(),
            TestScalarExpr::Gt { lhs, rhs } => {
                let lhs = lhs.with_new_inputs(inputs);
                let rhs = rhs.with_new_inputs(inputs);
                TestScalarExpr::Gt {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }
            }
            TestScalarExpr::SubQuery(_) => TestScalarExpr::SubQuery(inputs.rel_node()),
        }
    }

    fn copy_in_nested<T>(&self, visitor: &mut CopyInNestedExprs<TestOperator, T>) {
        match self {
            TestScalarExpr::Value(_) => {}
            TestScalarExpr::Gt { lhs, rhs } => {
                lhs.copy_in_nested(visitor);
                rhs.copy_in_nested(visitor);
            }
            TestScalarExpr::SubQuery(expr) => {
                visitor.visit_expr(expr);
            }
        }
    }
}

struct TraversalWrapper<'c, 'm, T> {
    ctx: &'c mut CopyInExprs<'m, TestOperator, T>,
}

impl<T> TraversalWrapper<'_, '_, T> {
    fn enter_expr(&mut self, expr: &TestOperator) -> ExprContext<TestOperator> {
        self.ctx.enter_expr(expr)
    }

    fn visit_rel(&mut self, expr_ctx: &mut ExprContext<TestOperator>, rel: &RelNode) {
        self.ctx.visit_expr_node(expr_ctx, rel);
    }

    fn visit_scalar(&mut self, expr_ctx: &mut ExprContext<TestOperator>, scalar: &ScalarNode) {
        self.ctx.visit_expr_node(expr_ctx, scalar);
    }

    fn copy_in_nested(&mut self, expr_ctx: &mut ExprContext<TestOperator>, expr: &TestScalarExpr) {
        let visitor = CopyInNestedExprs::new(self.ctx, expr_ctx);
        visitor.execute(expr, |expr, c: &mut CopyInNestedExprs<TestOperator, T>| {
            expr.copy_in_nested(c);
        })
    }

    fn copy_in(self, expr: &TestOperator, expr_ctx: ExprContext<TestOperator>) {
        self.ctx.copy_in(expr, expr_ctx)
    }
}

#[derive(Debug, Clone)]
struct TestOperator {
    expr: ExprPtr<TestOperator>,
    group: ExprGroupPtr<TestOperator>,
}

#[derive(Debug, Clone)]
enum TestProps {
    Rel(RelProps),
    Scalar(ScalarProps),
}

impl Props for TestProps {
    type RelProps = RelProps;
    type ScalarProps = ScalarProps;

    fn new_rel(props: Self::RelProps) -> Self {
        TestProps::Rel(props)
    }

    fn new_scalar(props: Self::ScalarProps) -> Self {
        TestProps::Scalar(props)
    }

    fn relational(&self) -> &Self::RelProps {
        unreachable!()
    }

    fn scalar(&self) -> &Self::ScalarProps {
        unreachable!()
    }
}

#[derive(Debug, Clone, Default)]
struct RelProps {
    _a: i32,
}

#[derive(Debug, Clone, Default)]
struct ScalarProps {
    sub_queries: Vec<TestOperator>,
}
impl MemoExpr for TestOperator {
    type Expr = TestExpr;
    type Props = TestProps;

    fn from_parts(expr: ExprPtr<Self>, group: ExprGroupPtr<Self>) -> Self {
        TestOperator { expr, group }
    }

    fn expr_ptr(&self) -> &ExprPtr<Self> {
        &self.expr
    }

    fn group_ptr(&self) -> &ExprGroupPtr<Self> {
        &self.group
    }

    fn copy_in<T>(&self, ctx: &mut CopyInExprs<Self, T>) {
        let mut ctx = TraversalWrapper { ctx };
        let mut expr_ctx = ctx.enter_expr(self);
        match self.expr() {
            TestExpr::Relational(expr) => match expr {
                TestRelExpr::Scan { .. } => {}
                TestRelExpr::Filter { input, filter } => {
                    ctx.visit_rel(&mut expr_ctx, input);
                    ctx.visit_scalar(&mut expr_ctx, filter);
                }
                TestRelExpr::Join { left, right, .. } => {
                    ctx.visit_rel(&mut expr_ctx, left);
                    ctx.visit_rel(&mut expr_ctx, right);
                }
            },
            TestExpr::Scalar(expr) => ctx.copy_in_nested(&mut expr_ctx, expr),
        }
        ctx.copy_in(self, expr_ctx)
    }

    fn expr_with_new_children(expr: &Self::Expr, mut inputs: NewChildExprs<Self>) -> Self::Expr {
        match expr {
            TestExpr::Relational(expr) => {
                let expr = match expr {
                    TestRelExpr::Scan { src } => {
                        inputs.expect_len(0, "Scan");
                        TestRelExpr::Scan { src }
                    }
                    TestRelExpr::Filter { .. } => {
                        inputs.expect_len(2, "Filter");
                        TestRelExpr::Filter {
                            input: inputs.rel_node(),
                            filter: inputs.scalar_node(),
                        }
                    }
                    TestRelExpr::Join { .. } => {
                        inputs.expect_len(2, "Join");
                        TestRelExpr::Join {
                            left: inputs.rel_node(),
                            right: inputs.rel_node(),
                        }
                    }
                };
                TestExpr::Relational(expr)
            }
            TestExpr::Scalar(expr) => TestExpr::Scalar(expr.with_new_inputs(&mut inputs)),
        }
    }

    fn new_properties_with_nested_sub_queries(
        _props: Self::Props,
        sub_queries: impl Iterator<Item = Self>,
    ) -> Self::Props {
        TestProps::Scalar(ScalarProps {
            sub_queries: sub_queries.collect(),
        })
    }

    fn num_children(&self) -> usize {
        match self.expr() {
            TestExpr::Relational(expr) => match expr {
                TestRelExpr::Scan { .. } => 0,
                TestRelExpr::Filter { .. } => 2,
                TestRelExpr::Join { .. } => 2,
            },
            TestExpr::Scalar(_) => self.props().scalar().sub_queries.len(),
        }
    }

    fn get_child(&self, i: usize) -> Option<&Self> {
        match self.expr() {
            TestExpr::Relational(expr) => match expr {
                TestRelExpr::Scan { .. } => None,
                TestRelExpr::Filter { input, .. } if i == 0 => Some(input.mexpr()),
                TestRelExpr::Filter { filter, .. } if i == 1 => Some(filter.mexpr()),
                TestRelExpr::Filter { .. } => None,
                TestRelExpr::Join { left, .. } if i == 0 => Some(left.mexpr()),
                TestRelExpr::Join { right, .. } if i == 1 => Some(right.mexpr()),
                TestRelExpr::Join { .. } => None,
            },
            TestExpr::Scalar(_) => None,
        }
    }

    fn format_expr<F>(expr: &Self::Expr, _props: &Self::Props, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match expr {
            TestExpr::Relational(expr) => match expr {
                TestRelExpr::Scan { src } => {
                    f.write_name("Scan");
                    f.write_source(src);
                }
                TestRelExpr::Filter { input, filter } => {
                    f.write_name("Filter");
                    f.write_expr("input", input);
                    f.write_expr("filter", filter);
                }
                TestRelExpr::Join { left, right } => {
                    f.write_name("Join");
                    f.write_expr("left", left);
                    f.write_expr("right", right);
                }
            },
            TestExpr::Scalar(expr) => {
                f.write_name("Expr");
                f.write_value("", expr)
            }
        }
    }
}

impl From<TestExpr> for TestOperator {
    fn from(expr: TestExpr) -> Self {
        let props = match expr {
            TestExpr::Relational(_) => TestProps::Rel(RelProps::default()),
            TestExpr::Scalar(_) => TestProps::Scalar(ScalarProps::default()),
        };
        TestOperator {
            expr: ExprPtr::new(expr),
            group: ExprGroupPtr::new(props),
        }
    }
}

impl From<TestRelExpr> for TestOperator {
    fn from(expr: TestRelExpr) -> Self {
        TestOperator {
            expr: ExprPtr::new(TestExpr::Relational(expr)),
            group: ExprGroupPtr::new(TestProps::Rel(RelProps::default())),
        }
    }
}

impl From<TestScalarExpr> for TestOperator {
    fn from(expr: TestScalarExpr) -> Self {
        TestOperator {
            expr: ExprPtr::new(TestExpr::Scalar(expr)),
            group: ExprGroupPtr::new(TestProps::Scalar(ScalarProps::default())),
        }
    }
}

impl From<TestOperator> for RelNode {
    fn from(expr: TestOperator) -> Self {
        RelNode::from_mexpr(expr)
    }
}

impl From<TestOperator> for ScalarNode {
    fn from(expr: TestOperator) -> Self {
        ScalarNode::from_mexpr(expr)
    }
}

fn memo_bench(c: &mut Criterion) {
    let query = query1();

    c.bench_function("memo_rel_scalar_query1", |b| {
        b.iter(|| {
            let mut memo = Memo::new(());
            let query = TestOperator::from(query.clone());
            let expr = memo.insert_group(query);
            black_box(expr);
        });
    });

    let query = query2();

    c.bench_function("memo_rel_scalar_query2", |b| {
        b.iter(|| {
            let mut memo = Memo::new(());
            let query = TestOperator::from(query.clone());
            let expr = memo.insert_group(query);
            black_box(expr);
        });
    });
}

fn query1() -> TestExpr {
    let scan_a = TestRelExpr::Scan { src: "A" };
    let scan_b = TestRelExpr::Scan { src: "B" };
    let join = TestRelExpr::Join {
        left: TestOperator::from(scan_a).into(),
        right: TestOperator::from(scan_b).into(),
    };

    let query = TestRelExpr::Filter {
        input: TestOperator::from(join).into(),
        filter: TestOperator::from(TestScalarExpr::Value(111)).into(),
    };
    TestExpr::Relational(query)
}

fn query2() -> TestExpr {
    let scan_a = query1();
    let scan_b = query1();
    let join = TestRelExpr::Join {
        left: TestOperator::from(scan_a).into(),
        right: TestOperator::from(scan_b).into(),
    };

    let query = TestRelExpr::Filter {
        input: TestOperator::from(join).into(),
        filter: TestOperator::from(TestScalarExpr::Value(111)).into(),
    };
    TestExpr::Relational(query)
}

criterion_group!(benches, memo_bench,);
criterion_main!(benches);
