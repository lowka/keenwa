use criterion::{black_box, criterion_group, criterion_main, Criterion};
use keenwa::memo::{
    CopyInExprs, CopyInNestedExprs, Expr, ExprContext, ExprNode, ExprNodeRef, Memo, MemoExpr, MemoExprFormatter,
    MemoGroupRef, NewChildExprs, Properties,
};

use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
enum TestExpr {
    Relational(TestRelExpr),
    Scalar(TestScalarExpr),
}

#[derive(Debug, Clone)]
enum TestRelExpr {
    Scan { src: &'static str },
    Filter { input: RelExpr, filter: ScalarExpr },
    Join { left: RelExpr, right: RelExpr },
}

impl Expr for TestExpr {}

impl TestExpr {
    fn as_relational(&self) -> &TestRelExpr {
        match self {
            TestExpr::Relational(expr) => expr,
            TestExpr::Scalar(_) => panic!("Expected a relational expr"),
        }
    }

    fn as_scalar(&self) -> &TestScalarExpr {
        match self {
            TestExpr::Relational(_) => panic!("Expected relational expr"),
            TestExpr::Scalar(expr) => expr,
        }
    }
}

#[derive(Debug, Clone)]
enum RelExpr {
    Expr(Box<TestOperator>),
    Group(MemoGroupRef<TestOperator>),
}

impl RelExpr {
    fn get_ref(&self) -> ExprNodeRef<TestOperator> {
        match self {
            RelExpr::Expr(expr) => ExprNodeRef::Expr(&**expr),
            RelExpr::Group(group) => ExprNodeRef::Group(group),
        }
    }
}

impl From<ExprNode<TestOperator>> for RelExpr {
    fn from(expr: ExprNode<TestOperator>) -> Self {
        match expr {
            ExprNode::Expr(expr) => RelExpr::Expr(expr),
            ExprNode::Group(group) => RelExpr::Group(group),
        }
    }
}

#[derive(Debug, Clone)]
enum ScalarExpr {
    Expr(Box<TestOperator>),
    Group(MemoGroupRef<TestOperator>),
}

impl ScalarExpr {
    fn get_ref(&self) -> ExprNodeRef<TestOperator> {
        match self {
            ScalarExpr::Expr(expr) => ExprNodeRef::Expr(&**expr),
            ScalarExpr::Group(group) => ExprNodeRef::Group(group),
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
    SubQuery(RelExpr),
}

impl Display for TestScalarExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TestScalarExpr::Value(value) => write!(f, "{}", value),
            TestScalarExpr::Gt { lhs, rhs } => {
                write!(f, "{} > {}", lhs, rhs)
            }
            TestScalarExpr::SubQuery(query) => match query {
                RelExpr::Expr(expr) => {
                    let ptr: *const TestExpr = expr.expr();
                    write!(f, "SubQuery expr_ptr {:?}", ptr)
                }
                RelExpr::Group(group) => {
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
            TestScalarExpr::SubQuery(_) => TestScalarExpr::SubQuery(RelExpr::from(inputs.expr())),
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
                visitor.visit_expr(expr.get_ref());
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

    fn visit_rel(&mut self, expr_ctx: &mut ExprContext<TestOperator>, rel: &RelExpr) {
        self.ctx.visit_expr_node(expr_ctx, rel.get_ref());
    }

    fn visit_scalar(&mut self, expr_ctx: &mut ExprContext<TestOperator>, scalar: &ScalarExpr) {
        self.ctx.visit_expr_node(expr_ctx, scalar.get_ref());
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
    expr: TestExpr,
    props: TestProps,
}

#[derive(Debug, Clone, Default)]
struct TestProps {
    a: i32,
}

impl Properties for TestProps {}

impl MemoExpr for TestOperator {
    type Expr = TestExpr;
    type Props = TestProps;

    fn expr(&self) -> &Self::Expr {
        &self.expr
    }

    fn props(&self) -> &Self::Props {
        &self.props
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

    fn with_new_children(&self, mut inputs: NewChildExprs<Self>) -> Self {
        let expr = match self.expr() {
            TestExpr::Relational(expr) => {
                let expr = match expr {
                    TestRelExpr::Scan { src } => {
                        assert!(inputs.is_empty(), "expects no inputs");
                        TestRelExpr::Scan { src }
                    }
                    TestRelExpr::Filter { .. } => {
                        assert_eq!(inputs.len(), 2, "expects 1 input");
                        TestRelExpr::Filter {
                            input: RelExpr::from(inputs.expr()),
                            filter: ScalarExpr::from(inputs.expr()),
                        }
                    }
                    TestRelExpr::Join { .. } => {
                        assert_eq!(inputs.len(), 2, "expects 2 inputs");
                        TestRelExpr::Join {
                            left: RelExpr::from(inputs.expr()),
                            right: RelExpr::from(inputs.expr()),
                        }
                    }
                };
                TestExpr::Relational(expr)
            }
            TestExpr::Scalar(expr) => TestExpr::Scalar(expr.with_new_inputs(&mut inputs)),
        };
        TestOperator {
            expr,
            props: self.props.clone(),
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self.expr() {
            TestExpr::Relational(expr) => match expr {
                TestRelExpr::Scan { src } => {
                    f.write_name("Scan");
                    f.write_source(src);
                }
                TestRelExpr::Filter { input, filter } => {
                    f.write_name("Filter");
                    f.write_expr("input", input.get_ref());
                    f.write_expr("filter", filter.get_ref());
                }
                TestRelExpr::Join { left, right } => {
                    f.write_name("Join");
                    f.write_expr("left", left.get_ref());
                    f.write_expr("right", right.get_ref());
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
        TestOperator {
            expr,
            props: TestProps::default(),
        }
    }
}

impl From<TestRelExpr> for TestOperator {
    fn from(expr: TestRelExpr) -> Self {
        TestOperator {
            expr: TestExpr::Relational(expr),
            props: TestProps::default(),
        }
    }
}

impl From<TestScalarExpr> for TestOperator {
    fn from(expr: TestScalarExpr) -> Self {
        TestOperator {
            expr: TestExpr::Scalar(expr),
            props: TestProps::default(),
        }
    }
}

impl From<TestOperator> for RelExpr {
    fn from(expr: TestOperator) -> Self {
        RelExpr::Expr(Box::new(expr))
    }
}

impl From<TestOperator> for ScalarExpr {
    fn from(expr: TestOperator) -> Self {
        ScalarExpr::Expr(Box::new(expr))
    }
}

impl From<ExprNode<TestOperator>> for ScalarExpr {
    fn from(expr: ExprNode<TestOperator>) -> Self {
        match expr {
            ExprNode::Expr(expr) => ScalarExpr::Expr(expr),
            ExprNode::Group(group) => ScalarExpr::Group(group),
        }
    }
}

fn memo_bench(c: &mut Criterion) {
    let query = query1();

    c.bench_function("memo_rel_scalar_query1", |b| {
        b.iter(|| {
            let mut memo = Memo::new(());
            let query = TestOperator::from(query.clone());
            let (group, _) = memo.insert_group(query);
            black_box(group);
        });
    });

    let query = query2();

    c.bench_function("memo_rel_scalar_query2", |b| {
        b.iter(|| {
            let mut memo = Memo::new(());
            let query = TestOperator::from(query.clone());
            let (group, _) = memo.insert_group(query);
            black_box(group);
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
