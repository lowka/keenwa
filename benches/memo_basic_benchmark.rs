use criterion::{black_box, criterion_group, criterion_main, Criterion};
use keenwa::memo::{
    CopyInExprs, Expr, ExprGroupRef, ExprRef, Memo, MemoExpr, MemoExprFormatter, MemoExprNode, MemoExprNodeRef,
    NewChildExprs, Properties, SubQueries,
};
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

impl Properties for TestProps {
    type RelProps = TestProps;
    type ScalarProps = TestProps;

    fn new_rel(props: Self::RelProps) -> Self {
        props
    }

    fn new_scalar(props: Self::ScalarProps) -> Self {
        props
    }

    fn as_relational(&self) -> &Self::RelProps {
        self
    }

    fn as_scalar(&self) -> &Self::ScalarProps {
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

    fn as_relational(&self) -> &Self::RelExpr {
        unreachable!()
    }

    fn as_scalar(&self) -> &Self::ScalarExpr {
        unreachable!()
    }

    fn is_scalar(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
struct TestOperator {
    expr: ExprRef<TestOperator>,
    group: ExprGroupRef<TestOperator>,
}

impl MemoExpr for TestOperator {
    type Expr = TestExpr;
    type Props = TestProps;

    fn create(expr: ExprRef<Self>, group: ExprGroupRef<Self>) -> Self {
        TestOperator { expr, group }
    }

    fn expr_ref(&self) -> &ExprRef<Self> {
        &self.expr
    }

    fn group_ref(&self) -> &ExprGroupRef<Self> {
        &self.group
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

    fn with_new_children(expr: &Self::Expr, mut inputs: NewChildExprs<Self>) -> Self::Expr {
        match expr {
            TestExpr::Scan { src } => {
                assert!(inputs.is_empty(), "expects no inputs");
                TestExpr::Scan { src: src.clone() }
            }
            TestExpr::Filter { filter, .. } => {
                assert_eq!(inputs.len(), 1, "expects 1 input");
                TestExpr::Filter {
                    input: inputs.rel_node(),
                    filter: filter.clone(),
                }
            }
            TestExpr::Join { .. } => {
                assert_eq!(inputs.len(), 2, "expects 2 inputs");
                TestExpr::Join {
                    left: inputs.rel_node(),
                    right: inputs.rel_node(),
                }
            }
        }
    }

    fn new_properties_with_nested_sub_queries<'a>(
        _props: Self::Props,
        _sub_queries: impl Iterator<Item = &'a MemoExprNode<Self>>,
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

    fn get_child(&self, i: usize) -> Option<MemoExprNodeRef<Self>> {
        match self.expr() {
            TestExpr::Scan { .. } => None,
            TestExpr::Filter { input, .. } if i == 0 => Some(input.into()),
            TestExpr::Filter { .. } => None,
            TestExpr::Join { left, .. } if i == 0 => Some(left.into()),
            TestExpr::Join { right, .. } if i == 1 => Some(right.into()),
            TestExpr::Join { .. } => None,
        }
    }

    fn get_sub_queries(&self) -> Option<SubQueries<Self>> {
        unreachable!()
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
            expr: ExprRef::Detached(Box::new(expr)),
            group: ExprGroupRef::Detached(Box::new(TestProps::default())),
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
