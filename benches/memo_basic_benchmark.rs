use criterion::{black_box, criterion_group, criterion_main, Criterion};
use keenwa::memo::{CopyInExprs, Expr, Memo, MemoExpr, MemoExprFormatter, MemoExprState, NewChildExprs, Props};
use std::fmt::{Display, Formatter};

type RelNode = keenwa::memo::RelNode<TestOperator>;

#[derive(Debug, Clone)]
enum TestExpr {
    Scan { src: &'static str },
    Filter { input: RelNode, filter: TestScalarExpr },
    Join { left: RelNode, right: RelNode },
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

impl Props for TestProps {
    type RelProps = TestProps;
    type ScalarProps = TestProps;

    fn new_rel(props: Self::RelProps) -> Self {
        props
    }

    fn new_scalar(props: Self::ScalarProps) -> Self {
        props
    }

    fn relational(&self) -> &Self::RelProps {
        self
    }

    fn scalar(&self) -> &Self::ScalarProps {
        self
    }
}

impl Expr for TestExpr {
    type RelExpr = TestExpr;
    type ScalarExpr = TestExpr;

    fn new_rel(expr: Self::RelExpr) -> Self {
        expr
    }

    fn new_scalar(expr: Self::ScalarExpr) -> Self {
        expr
    }

    fn relational(&self) -> &Self::RelExpr {
        unreachable!()
    }

    fn scalar(&self) -> &Self::ScalarExpr {
        unreachable!()
    }

    fn is_scalar(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
struct TestOperator {
    state: MemoExprState<TestOperator>,
}

impl MemoExpr for TestOperator {
    type Expr = TestExpr;
    type Props = TestProps;

    fn from_state(state: MemoExprState<Self>) -> Self {
        TestOperator { state }
    }

    fn state(&self) -> &MemoExprState<Self> {
        &self.state
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

    fn expr_with_new_children(expr: &Self::Expr, mut inputs: NewChildExprs<Self>) -> Self::Expr {
        match expr {
            TestExpr::Scan { src } => {
                inputs.expect_len(0, "Scan");
                TestExpr::Scan { src }
            }
            TestExpr::Filter { filter, .. } => {
                inputs.expect_len(1, "Filter");
                TestExpr::Filter {
                    input: inputs.rel_node(),
                    filter: filter.clone(),
                }
            }
            TestExpr::Join { .. } => {
                inputs.expect_len(2, "Join");
                TestExpr::Join {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                }
            }
        }
    }

    fn new_properties_with_nested_sub_queries(
        _props: Self::Props,
        _sub_queries: impl Iterator<Item = Self>,
    ) -> Self::Props {
        unimplemented!()
    }

    fn num_children(&self) -> usize {
        match self.expr() {
            TestExpr::Scan { .. } => 0,
            TestExpr::Filter { .. } => 1,
            TestExpr::Join { .. } => 2,
        }
    }

    fn get_child(&self, i: usize) -> Option<&Self> {
        match self.expr() {
            TestExpr::Scan { .. } => None,
            TestExpr::Filter { input, .. } if i == 0 => Some(input.mexpr()),
            TestExpr::Filter { .. } => None,
            TestExpr::Join { left, .. } if i == 0 => Some(left.mexpr()),
            TestExpr::Join { right, .. } if i == 1 => Some(right.mexpr()),
            TestExpr::Join { .. } => None,
        }
    }

    fn format_expr<F>(expr: &Self::Expr, _props: &Self::Props, f: &mut F)
    where
        F: MemoExprFormatter,
    {
        match expr {
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
            state: MemoExprState::new(expr, TestProps::default()),
        }
    }
}

impl From<TestOperator> for RelNode {
    fn from(expr: TestOperator) -> Self {
        RelNode::from_mexpr(expr)
    }
}

fn memo_bench(c: &mut Criterion) {
    let query = query1();

    c.bench_function("memo_basic_query1", |b| {
        b.iter(|| {
            let mut memo = Memo::new(());
            let query = TestOperator::from(query.clone());
            let expr = memo.insert_group(query);
            black_box(expr);
        });
    });

    let query = query2();

    c.bench_function("memo_basic_query2", |b| {
        b.iter(|| {
            let mut memo = Memo::new(());
            let query = TestOperator::from(query.clone());
            let expr = memo.insert_group(query);
            black_box(expr);
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
