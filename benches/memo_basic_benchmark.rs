use criterion::{black_box, criterion_group, criterion_main, Criterion};
use keenwa::memo::{Attributes, CopyInExprs, Expr, InputNode, Memo, MemoExpr, MemoExprFormatter, NewInputs};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
enum TestExpr {
    Scan {
        src: &'static str,
    },
    Filter {
        input: InputNode<TestOperator>,
        filter: TestScalarExpr,
    },
    Join {
        left: InputNode<TestOperator>,
        right: InputNode<TestOperator>,
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
struct TestAttrs {
    a: i32,
}

impl Attributes for TestAttrs {}

impl Expr for TestExpr {}

#[derive(Debug, Clone)]
struct TestOperator {
    expr: TestExpr,
    attrs: TestAttrs,
}

impl MemoExpr for TestOperator {
    type Expr = TestExpr;
    type Attrs = TestAttrs;

    fn expr(&self) -> &Self::Expr {
        &self.expr
    }

    fn attrs(&self) -> &Self::Attrs {
        &self.attrs
    }

    fn copy_in(&self, ctx: &mut CopyInExprs<Self>) {
        let mut expr_ctx = ctx.enter_expr(self);
        match self.expr() {
            TestExpr::Scan { .. } => {}
            TestExpr::Filter { input, .. } => {
                ctx.visit_input(&mut expr_ctx, input);
            }
            TestExpr::Join { left, right, .. } => {
                ctx.visit_input(&mut expr_ctx, left);
                ctx.visit_input(&mut expr_ctx, right);
            }
        }
        ctx.copy_in(self, expr_ctx);
    }

    fn with_new_inputs(&self, mut inputs: NewInputs<Self>) -> Self {
        let expr = match self.expr() {
            TestExpr::Scan { src } => {
                assert!(inputs.is_empty(), "expects no inputs");
                TestExpr::Scan { src: src.clone() }
            }
            TestExpr::Filter { filter, .. } => {
                assert_eq!(inputs.len(), 1, "expects 1 input");
                TestExpr::Filter {
                    input: inputs.input(),
                    filter: filter.clone(),
                }
            }
            TestExpr::Join { .. } => {
                assert_eq!(inputs.len(), 2, "expects 2 inputs");
                TestExpr::Join {
                    left: inputs.input(),
                    right: inputs.input(),
                }
            }
        };
        TestOperator {
            expr,
            attrs: self.attrs.clone(),
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
                f.write_input("input", input);
                f.write_value("filter", filter);
            }
            TestExpr::Join { left, right } => {
                f.write_name("Join");
                f.write_input("left", left);
                f.write_input("right", right);
            }
        }
    }
}

impl From<TestExpr> for TestOperator {
    fn from(expr: TestExpr) -> Self {
        TestOperator {
            expr,
            attrs: TestAttrs::default(),
        }
    }
}

fn memo_bench(c: &mut Criterion) {
    let query = query1();

    c.bench_function("memo_basic_query1", |b| {
        b.iter(|| {
            let mut memo = Memo::new();
            let query = TestOperator::from(query.clone());
            let (group, _) = memo.insert(query);
            black_box(group);
        });
    });

    let query = query2();

    c.bench_function("memo_basic_query2", |b| {
        b.iter(|| {
            let mut memo = Memo::new();
            let query = TestOperator::from(query.clone());
            let (group, _) = memo.insert(query);
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
