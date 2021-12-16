use criterion::{black_box, criterion_group, criterion_main, Criterion};
use keenwa::memo::{
    CopyInExprs, Expr, ExprNode, ExprNodeRef, Memo, MemoExpr, MemoExprFormatter, NewChildExprs, Properties,
};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
enum TestExpr {
    Scan {
        src: &'static str,
    },
    Filter {
        input: ExprNode<TestOperator>,
        filter: TestScalarExpr,
    },
    Join {
        left: ExprNode<TestOperator>,
        right: ExprNode<TestOperator>,
    },
}

#[derive(Debug, Clone)]
enum TestScalarExpr {
    Value(i32),
    Eq {
        lhs: Box<TestScalarExpr>,
        rhs: Box<TestScalarExpr>,
    },
}

impl Display for TestScalarExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TestScalarExpr::Value(v) => write!(f, "{}", v),
            TestScalarExpr::Eq { lhs, rhs } => write!(f, "{} = {}", lhs, rhs),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct TestProps {
    a: i32,
}

impl Properties for TestProps {}

impl Expr for TestExpr {}

#[derive(Debug, Clone)]
struct TestOperator {
    expr: TestExpr,
    props: TestProps,
}

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
        let mut expr_ctx = ctx.enter_expr(self);
        match self.expr() {
            TestExpr::Scan { .. } => {}
            TestExpr::Filter { input, .. } => {
                ctx.visit_expr_node(&mut expr_ctx, input);
            }
            TestExpr::Join { left, right, .. } => {
                ctx.visit_expr_node(&mut expr_ctx, left);
                ctx.visit_expr_node(&mut expr_ctx, right);
            }
        }
        ctx.copy_in(self, expr_ctx);
    }

    fn create(expr: Self::Expr, props: Self::Props) -> Self {
        TestOperator { expr, props }
    }

    fn new_expr(expr: &Self::Expr, mut inputs: NewChildExprs<Self>) -> (Self::Expr, Option<Self::Props>) {
        let expr = match expr {
            TestExpr::Scan { src } => {
                assert!(inputs.is_empty(), "expects no inputs");
                TestExpr::Scan { src: src.clone() }
            }
            TestExpr::Filter { filter, .. } => {
                assert_eq!(inputs.len(), 1, "expects 1 input");
                TestExpr::Filter {
                    input: inputs.expr(),
                    filter: filter.clone(),
                }
            }
            TestExpr::Join { .. } => {
                assert_eq!(inputs.len(), 2, "expects 2 inputs");
                TestExpr::Join {
                    left: inputs.expr(),
                    right: inputs.expr(),
                }
            }
        };
        (expr, None)
    }

    fn num_children(&self) -> usize {
        match self.expr() {
            TestExpr::Scan { .. } => 0,
            TestExpr::Filter { .. } => 1,
            TestExpr::Join { .. } => 2,
        }
    }

    fn get_child<'a>(&'a self, i: usize, _props: &'a Self::Props) -> Option<ExprNodeRef<'a, Self>> {
        match self.expr() {
            TestExpr::Scan { .. } => None,
            TestExpr::Filter { input, .. } if i == 0 => Some(input.into()),
            TestExpr::Filter { .. } => None,
            TestExpr::Join { left, .. } if i == 0 => Some(left.into()),
            TestExpr::Join { right, .. } if i == 1 => Some(right.into()),
            TestExpr::Join { .. } => None,
        }
    }

    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match self.expr() {
            TestExpr::Scan { src } => {
                f.write_name("Scan");
                f.write_source(src);
            }
            TestExpr::Filter { input, filter } => {
                f.write_name("Filter");
                f.write_expr("input", input);
                f.write_value("filter", filter);
            }
            TestExpr::Join { left, right } => {
                f.write_name("Join");
                f.write_expr("left", left);
                f.write_expr("right", right);
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

fn memo_bench(c: &mut Criterion) {
    let query = query1();

    c.bench_function("memo_basic_query1", |b| {
        b.iter(|| {
            let mut memo = Memo::new(());
            let query = TestOperator::from(query.clone());
            let (group, _) = memo.insert_group(query);
            black_box(group);
        });
    });

    let query = query2();

    c.bench_function("memo_basic_query2", |b| {
        b.iter(|| {
            let mut memo = Memo::new(());
            let query = TestOperator::from(query.clone());
            let (group, _) = memo.insert_group(query);
            black_box(group);
        });
    });
}

fn query1() -> TestExpr {
    let scan_a = TestExpr::Scan { src: "A" };
    let scan_b = TestExpr::Scan { src: "B" };
    let join = TestExpr::Join {
        left: TestOperator::from(scan_a).into(),
        right: TestOperator::from(scan_b).into(),
    };

    TestExpr::Filter {
        input: TestOperator::from(join).into(),
        filter: TestScalarExpr::Value(111),
    }
}

fn query2() -> TestExpr {
    let scan_a = query1();
    let scan_b = query1();
    let join = TestExpr::Join {
        left: TestOperator::from(scan_a).into(),
        right: TestOperator::from(scan_b).into(),
    };

    TestExpr::Filter {
        input: TestOperator::from(join).into(),
        filter: TestScalarExpr::Value(111),
    }
}

criterion_group!(benches, memo_bench,);

criterion_main!(benches);
