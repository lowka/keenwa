use std::fmt::{Display, Formatter};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use keenwa::error::OptimizerError;

use keenwa::memo::{
    CopyInExprs, CopyInNestedExprs, Expr, ExprContext, MemoBuilder, MemoExpr, MemoExprFormatter, MemoExprState,
    NewChildExprs, Props,
};

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

    fn is_relational(&self) -> bool {
        match self {
            TestExpr::Relational(_) => true,
            TestExpr::Scalar(_) => false,
        }
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
            TestScalarExpr::SubQuery(rel_node) => match rel_node.state() {
                MemoExprState::Owned(_) => {
                    let ptr: *const TestExpr = &*rel_node.state().expr();
                    write!(f, "SubQuery expr_ptr {:?}", ptr)
                }
                MemoExprState::Memo(state) => {
                    write!(f, "SubQuery {}", state.group_id())
                }
            },
        }
    }
}

impl TestScalarExpr {
    fn with_new_inputs(&self, inputs: &mut NewChildExprs<TestOperator>) -> Result<Self, OptimizerError> {
        let expr = match self {
            TestScalarExpr::Value(_) => self.clone(),
            TestScalarExpr::Gt { lhs, rhs } => {
                let lhs = lhs.with_new_inputs(inputs)?;
                let rhs = rhs.with_new_inputs(inputs)?;
                TestScalarExpr::Gt {
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }
            }
            TestScalarExpr::SubQuery(_) => TestScalarExpr::SubQuery(inputs.rel_node()?),
        };
        Ok(expr)
    }

    fn copy_in_nested<T>(&self, visitor: &mut CopyInNestedExprs<TestOperator, T>) -> Result<(), OptimizerError> {
        match self {
            TestScalarExpr::Value(_) => Ok(()),
            TestScalarExpr::Gt { lhs, rhs } => {
                lhs.copy_in_nested(visitor)?;
                rhs.copy_in_nested(visitor)
            }
            TestScalarExpr::SubQuery(expr) => visitor.visit_expr(expr),
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

    fn visit_rel(&mut self, expr_ctx: &mut ExprContext<TestOperator>, rel: &RelNode) -> Result<(), OptimizerError> {
        self.ctx.visit_expr_node(expr_ctx, rel)
    }

    fn visit_scalar(
        &mut self,
        expr_ctx: &mut ExprContext<TestOperator>,
        scalar: &ScalarNode,
    ) -> Result<(), OptimizerError> {
        self.ctx.visit_expr_node(expr_ctx, scalar)
    }

    fn copy_in_nested(
        &mut self,
        expr_ctx: &mut ExprContext<TestOperator>,
        expr: &TestScalarExpr,
    ) -> Result<(), OptimizerError> {
        let visitor = CopyInNestedExprs::new(self.ctx, expr_ctx);
        visitor.execute(expr, |expr, c: &mut CopyInNestedExprs<TestOperator, T>| expr.copy_in_nested(c))
    }

    fn copy_in(self, expr: &TestOperator, expr_ctx: ExprContext<TestOperator>) -> Result<(), OptimizerError> {
        self.ctx.copy_in(expr, expr_ctx)
    }
}

#[derive(Debug, Clone)]
struct TestOperator {
    state: MemoExprState<TestOperator>,
}

struct TestScope;

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

    fn to_relational(self) -> Self::RelProps {
        unreachable!()
    }

    fn to_scalar(self) -> Self::ScalarProps {
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
    type Scope = TestScope;

    fn from_state(state: MemoExprState<Self>) -> Self {
        TestOperator { state }
    }

    fn state(&self) -> &MemoExprState<Self> {
        &self.state
    }

    fn copy_in<T>(&self, ctx: &mut CopyInExprs<Self, T>) -> Result<(), OptimizerError> {
        let mut ctx = TraversalWrapper { ctx };
        let mut expr_ctx = ctx.enter_expr(self);
        match self.expr() {
            TestExpr::Relational(expr) => match expr {
                TestRelExpr::Scan { .. } => {}
                TestRelExpr::Filter { input, filter } => {
                    ctx.visit_rel(&mut expr_ctx, input)?;
                    ctx.visit_scalar(&mut expr_ctx, filter)?
                }
                TestRelExpr::Join { left, right, .. } => {
                    ctx.visit_rel(&mut expr_ctx, left)?;
                    ctx.visit_rel(&mut expr_ctx, right)?;
                }
            },
            TestExpr::Scalar(expr) => ctx.copy_in_nested(&mut expr_ctx, expr)?,
        }
        ctx.copy_in(self, expr_ctx)
    }

    fn expr_with_new_children(
        expr: &Self::Expr,
        mut inputs: NewChildExprs<Self>,
    ) -> Result<Self::Expr, OptimizerError> {
        let expr = match expr {
            TestExpr::Relational(expr) => {
                let expr = match expr {
                    TestRelExpr::Scan { src } => {
                        inputs.expect_len(0, "Scan")?;
                        TestRelExpr::Scan { src }
                    }
                    TestRelExpr::Filter { .. } => {
                        inputs.expect_len(2, "Filter")?;
                        TestRelExpr::Filter {
                            input: inputs.rel_node()?,
                            filter: inputs.scalar_node()?,
                        }
                    }
                    TestRelExpr::Join { .. } => {
                        inputs.expect_len(2, "Join")?;
                        TestRelExpr::Join {
                            left: inputs.rel_node()?,
                            right: inputs.rel_node()?,
                        }
                    }
                };
                TestExpr::Relational(expr)
            }
            TestExpr::Scalar(expr) => TestExpr::Scalar(expr.with_new_inputs(&mut inputs)?),
        };
        Ok(expr)
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
            state: MemoExprState::new(expr, props),
        }
    }
}

impl From<TestRelExpr> for TestOperator {
    fn from(expr: TestRelExpr) -> Self {
        TestOperator {
            state: MemoExprState::new(TestExpr::Relational(expr), TestProps::Rel(RelProps::default())),
        }
    }
}

impl From<TestScalarExpr> for TestOperator {
    fn from(expr: TestScalarExpr) -> Self {
        TestOperator {
            state: MemoExprState::new(TestExpr::Scalar(expr), TestProps::Scalar(ScalarProps::default())),
        }
    }
}

fn memo_bench(c: &mut Criterion) {
    let query = query1();

    c.bench_function("memo_rel_scalar_query1", |b| {
        b.iter(|| {
            let mut memo = MemoBuilder::new(()).build();
            let query = TestOperator::from(query.clone());
            let expr = memo.insert_group(query, &TestScope);
            black_box(expr.unwrap());
        });
    });

    let query = query2();

    c.bench_function("memo_rel_scalar_query2", |b| {
        b.iter(|| {
            let mut memo = MemoBuilder::new(()).build();
            let query = TestOperator::from(query.clone());
            let expr = memo.insert_group(query, &TestScope);
            black_box(expr.unwrap());
        });
    });
}

fn query1() -> TestRelExpr {
    let scan_a = TestRelExpr::Scan { src: "A" };
    let scan_b = TestRelExpr::Scan { src: "B" };
    let join = TestRelExpr::Join {
        left: RelNode::new_expr(scan_a),
        right: RelNode::new_expr(scan_b),
    };

    TestRelExpr::Filter {
        input: RelNode::new_expr(join),
        filter: ScalarNode::new_expr(TestScalarExpr::Value(111)),
    }
}

fn query2() -> TestExpr {
    let scan_a = query1();
    let scan_b = query1();

    let join = TestRelExpr::Join {
        left: RelNode::new_expr(scan_a),
        right: RelNode::new_expr(scan_b),
    };

    let query = TestRelExpr::Filter {
        input: RelNode::new_expr(join),
        filter: ScalarNode::new_expr(TestScalarExpr::Value(111)),
    };
    TestExpr::Relational(query)
}

criterion_group!(benches, memo_bench,);
criterion_main!(benches);
