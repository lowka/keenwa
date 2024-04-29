use crate::error::OptimizerError;
use crate::memo::{format_memo, GroupId, Memo, MemoBuilder, MemoExpr, MemoExprRef, MemoGroupRef};

pub trait MemoExprScopeProvider: Sized {
    type Expr: MemoExpr<Scope = Self>;

    fn build_scope(expr: &Self::Expr) -> Self;
}

pub fn new_memo<E>() -> Memo<E, ()>
where
    E: MemoExpr,
{
    MemoBuilder::new(()).build()
}

pub fn insert_group<E, T, S>(memo: &mut Memo<E, T>, expr: E) -> Result<(GroupId, MemoExprRef<E>), OptimizerError>
where
    E: MemoExpr<Scope = S>,
    S: MemoExprScopeProvider<Expr = E>,
{
    let scope = S::build_scope(&expr);
    let expr = memo.insert_group(expr, &scope)?;
    let group_id = expr.state().memo_group_id();
    let expr_ref = expr.state().memo_expr();

    Ok((group_id, expr_ref))
}

pub fn insert_group_member<E, T, S>(
    memo: &mut Memo<E, T>,
    group_id: GroupId,
    expr: E,
) -> Result<(GroupId, MemoExprRef<E>), OptimizerError>
where
    E: MemoExpr<Scope = S>,
    S: MemoExprScopeProvider<Expr = E>,
{
    let group = memo.get_group(&group_id)?;
    let token = group.to_group_token();
    let scope = S::build_scope(&expr);
    let expr = memo.insert_group_member(token, expr, &scope)?;

    let expr_ref = expr.state().memo_expr();

    Ok((group_id, expr_ref))
}

pub fn expect_group_size<E, T>(memo: &Memo<E, T>, group_id: &GroupId, size: usize)
where
    E: MemoExpr,
{
    let group = memo.get_group(group_id).expect("Group does not exists");
    assert_eq!(group.mexprs().count(), size, "group#{}", group_id);
}

pub fn expect_group_exprs<E, T>(group: &MemoGroupRef<E, T>, expected: Vec<MemoExprRef<E>>)
where
    E: MemoExpr,
{
    let actual: Vec<MemoExprRef<E>> = group.mexprs().collect();
    assert_eq!(actual, expected, "group#{} exprs", group.id());
}

pub fn expect_memo<E, T>(memo: &Memo<E, T>, expected: &str)
where
    E: MemoExpr,
{
    let lines: Vec<String> = expected.split('\n').map(String::from).collect();
    let expected = lines.join("\n");

    let buf = format_memo(memo);
    assert_eq!(buf.trim(), expected.trim());
}

pub fn expect_memo_with_props<E, T>(memo: &Memo<E, T>, expected: &str)
where
    E: MemoExpr,
{
    let lines: Vec<String> = expected.split('\n').map(String::from).collect();
    let expected = lines.join("\n");

    let buf = format_memo(memo);
    assert_eq!(buf.trim(), expected.trim());
}

#[cfg(test)]
mod test {
    use crate::error::OptimizerError;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fmt::{Display, Formatter};
    use std::rc::Rc;

    use crate::memo::{
        format_memo, CopyInExprs, CopyInNestedExprs, Expr, ExprContext, MemoBuilder, MemoExpr, MemoExprFormatter,
        MemoExprState, MemoFormatterFlags, MemoGroupCallback, NewChildExprs, Props, StringMemoFormatter,
    };

    use super::*;

    type RelNode = crate::memo::RelNode<TestOperator>;
    type ScalarNode = crate::memo::ScalarNode<TestOperator>;
    type TestResult = Result<(), OptimizerError>;

    #[derive(Debug, Clone)]
    enum TestExpr {
        Relational(TestRelExpr),
        Scalar(TestScalarExpr),
    }

    #[derive(Debug, Clone)]
    enum TestRelExpr {
        Leaf(&'static str),
        Node { input: RelNode },
        Nodes { inputs: Vec<RelNode> },
        Filter { input: RelNode, filter: ScalarNode },
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
            match self {
                TestExpr::Relational(expr) => expr,
                TestExpr::Scalar(_) => panic!(),
            }
        }

        fn scalar(&self) -> &Self::ScalarExpr {
            match self {
                TestExpr::Relational(_) => panic!(),
                TestExpr::Scalar(expr) => expr,
            }
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

    impl From<TestRelExpr> for RelNode {
        fn from(expr: TestRelExpr) -> Self {
            RelNode::new_expr(expr)
        }
    }

    impl From<TestScalarExpr> for ScalarNode {
        fn from(expr: TestScalarExpr) -> Self {
            ScalarNode::new_expr(expr)
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

        fn copy_in_nested<D>(&self, collector: &mut CopyInNestedExprs<TestOperator, D>) -> Result<(), OptimizerError> {
            match self {
                TestScalarExpr::Value(_) => Ok(()),
                TestScalarExpr::Gt { lhs, rhs } => {
                    lhs.copy_in_nested(collector)?;
                    rhs.copy_in_nested(collector)
                }
                TestScalarExpr::SubQuery(expr) => collector.visit_expr(expr),
            }
        }
    }

    impl Display for TestScalarExpr {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match self {
                TestScalarExpr::Value(value) => write!(f, "{}", value),
                TestScalarExpr::Gt { lhs, rhs } => {
                    write!(f, "({} > {})", lhs, rhs)
                }
                TestScalarExpr::SubQuery(query) => match query.state() {
                    MemoExprState::Owned(state) => {
                        let ptr: *const TestExpr = state.expr.as_ptr();
                        write!(f, "SubQuery expr_ptr {:?}", ptr)
                    }
                    MemoExprState::Memo(state) => {
                        write!(f, "SubQuery {}", state.group_id())
                    }
                },
            }
        }
    }

    struct TraversalWrapper<'c, 'm, D> {
        ctx: &'c mut CopyInExprs<'m, TestOperator, D>,
    }

    impl<D> TraversalWrapper<'_, '_, D> {
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
            expr: &TestOperator,
        ) -> Result<(), OptimizerError> {
            let nested_ctx = CopyInNestedExprs::new(self.ctx, expr_ctx);
            let scalar = expr.expr().scalar();
            nested_ctx
                .execute(scalar, |expr, collect: &mut CopyInNestedExprs<TestOperator, D>| expr.copy_in_nested(collect))
        }

        fn copy_in(self, expr: &TestOperator, expr_ctx: ExprContext<TestOperator>) -> Result<(), OptimizerError> {
            self.ctx.copy_in(expr, expr_ctx)
        }
    }

    #[derive(Debug, Clone)]
    struct TestOperator {
        state: MemoExprState<TestOperator>,
    }

    impl From<TestRelExpr> for TestOperator {
        fn from(expr: TestRelExpr) -> Self {
            TestOperator {
                state: MemoExprState::new(TestExpr::Relational(expr), TestProps::Rel(RelProps::default())),
            }
        }
    }

    struct TestScope;

    impl MemoExprScopeProvider for TestScope {
        type Expr = TestOperator;

        fn build_scope(_expr: &Self::Expr) -> Self {
            TestScope
        }
    }

    #[derive(Debug, Clone)]
    enum TestProps {
        Rel(RelProps),
        Scalar(ScalarProps),
    }

    #[derive(Debug, Clone, PartialEq, Default)]
    struct RelProps {
        a: i32,
    }

    #[derive(Debug, Clone, Default)]
    struct ScalarProps {
        sub_queries: Vec<TestOperator>,
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

        fn relational(&self) -> &RelProps {
            match self {
                TestProps::Rel(expr) => expr,
                TestProps::Scalar(_) => panic!("Expected relational properties"),
            }
        }

        fn scalar(&self) -> &ScalarProps {
            match self {
                TestProps::Rel(_) => panic!("Expected scalar properties"),
                TestProps::Scalar(expr) => expr,
            }
        }

        fn to_relational(self) -> Self::RelProps {
            match self {
                TestProps::Rel(expr) => expr,
                TestProps::Scalar(_) => panic!("Expected relational properties"),
            }
        }

        fn to_scalar(self) -> Self::ScalarProps {
            match self {
                TestProps::Rel(_) => panic!("Expected scalar properties"),
                TestProps::Scalar(expr) => expr,
            }
        }
    }

    impl TestOperator {
        fn with_rel_props(self, a: i32) -> Self {
            let state = match self.state {
                MemoExprState::Owned(state) => MemoExprState::Owned(state.with_props(TestProps::Rel(RelProps { a }))),
                MemoExprState::Memo(_) => unimplemented!(),
            };
            TestOperator { state }
        }

        fn into_rel_node(self) -> Result<RelNode, OptimizerError> {
            RelNode::try_from(self)
        }
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
                    TestRelExpr::Leaf(_) => {}
                    TestRelExpr::Node { input } => {
                        ctx.visit_rel(&mut expr_ctx, input)?;
                    }
                    TestRelExpr::Nodes { inputs } => {
                        for input in inputs {
                            ctx.visit_rel(&mut expr_ctx, input)?;
                        }
                    }
                    TestRelExpr::Filter { input, filter } => {
                        ctx.visit_rel(&mut expr_ctx, input)?;
                        ctx.visit_scalar(&mut expr_ctx, filter)?;
                    }
                },
                TestExpr::Scalar(_expr) => {
                    ctx.copy_in_nested(&mut expr_ctx, self)?;
                }
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
                        TestRelExpr::Leaf(s) => TestRelExpr::Leaf(s.clone()),
                        TestRelExpr::Node { .. } => TestRelExpr::Node {
                            input: inputs.rel_node()?,
                        },
                        TestRelExpr::Nodes { inputs: input_exprs } => TestRelExpr::Nodes {
                            inputs: inputs.rel_nodes(input_exprs.len())?,
                        },
                        TestRelExpr::Filter { .. } => TestRelExpr::Filter {
                            input: inputs.rel_node()?,
                            filter: inputs.scalar_node()?,
                        },
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
                    TestRelExpr::Leaf(_) => 0,
                    TestRelExpr::Node { .. } => 1,
                    TestRelExpr::Nodes { inputs } => inputs.len(),
                    TestRelExpr::Filter { .. } => 2,
                },
                TestExpr::Scalar(_) => self.props().scalar().sub_queries.len(),
            }
        }

        fn get_child(&self, i: usize) -> Option<&Self> {
            match self.expr() {
                TestExpr::Relational(expr) => match expr {
                    TestRelExpr::Leaf(_) => None,
                    TestRelExpr::Node { input } if i == 0 => Some(input.mexpr()),
                    TestRelExpr::Node { .. } => None,
                    TestRelExpr::Nodes { inputs } if i < inputs.len() => inputs.get(i).map(|e| e.mexpr()),
                    TestRelExpr::Nodes { .. } => None,
                    TestRelExpr::Filter { input, .. } if i == 0 => Some(input.mexpr()),
                    TestRelExpr::Filter { filter, .. } if i == 1 => Some(filter.mexpr()),
                    TestRelExpr::Filter { .. } => None,
                },
                TestExpr::Scalar(_) => {
                    let props = self.props().scalar();
                    props.sub_queries.get(i)
                }
            }
        }

        fn format_expr<F>(expr: &Self::Expr, props: &Self::Props, f: &mut F)
        where
            F: MemoExprFormatter,
        {
            match expr {
                TestExpr::Relational(expr) => match expr {
                    TestRelExpr::Leaf(source) => {
                        f.write_name("Leaf");
                        f.write_source(source);
                    }
                    TestRelExpr::Node { input } => {
                        f.write_name("Node");
                        f.write_expr("", input);
                    }
                    TestRelExpr::Nodes { inputs } => {
                        f.write_name("Nodes");
                        f.write_exprs("", inputs.iter());
                    }
                    TestRelExpr::Filter { input, filter } => {
                        f.write_name("Filter");
                        f.write_expr("", input);
                        f.write_expr("filter", filter);
                    }
                },
                TestExpr::Scalar(expr) => {
                    f.write_name("Expr");
                    f.write_value("", expr);
                }
            }

            if !expr.is_scalar() && props.relational() != &RelProps::default() {
                let props = props.relational();
                f.write_value("a", props.a)
            }
        }
    }

    #[test]
    fn test_basics() -> TestResult {
        let mut memo = new_memo();
        let expr1 = TestOperator::from(TestRelExpr::Leaf("aaaa")).with_rel_props(100);
        let (group_id, expr) = insert_group(&mut memo, expr1)?;
        let group = memo.get_group(&group_id)?;

        assert_eq!(group.props().relational(), &RelProps { a: 100 }, "group properties");
        assert_eq!(expr.group_id(), group.id(), "expr group");

        assert_eq!(
            format!("{:#?}", group.props()),
            format!("{:#?}", expr.props()),
            "groups properties and expr properties must be equal"
        );

        Ok(())
    }

    #[test]
    fn test_rel_node_eq() {
        let expr = TestRelExpr::Leaf("a");
        let props = RelProps::default();
        let rel_node = RelNode::new(expr, props.clone());
        assert_eq!(rel_node, rel_node, "self eq");

        assert_eq!(rel_node, rel_node.clone(), "clone eq");

        let expr1 = TestRelExpr::Leaf("a");
        let duplicate = RelNode::new(expr1, props);
        assert_ne!(rel_node, duplicate, "rel nodes in the owned state when they store to the same expr");
    }

    #[test]
    fn test_rel_node_hash() {
        let mut hash_map = HashMap::new();

        let expr = TestRelExpr::Leaf("a");
        let props = RelProps::default();
        let rel_node = RelNode::new(expr, props.clone());

        hash_map.insert(rel_node.clone(), 1);
        assert_eq!(hash_map.get(&rel_node), Some(&1));

        let existing = hash_map.insert(rel_node, 2);
        assert_eq!(existing, Some(1));

        let expr1 = TestRelExpr::Leaf("a");
        let duplicate = RelNode::new(expr1, props);
        assert_eq!(hash_map.get(&duplicate), None);
    }

    #[test]
    fn test_scalar_node_eq() {
        let expr = TestScalarExpr::Value(1);
        let props = ScalarProps::default();
        let scalar_node = ScalarNode::new(expr, props.clone());
        assert_eq!(scalar_node, scalar_node, "self eq");

        assert_eq!(scalar_node, scalar_node.clone(), "clone eq");

        let expr1 = TestScalarExpr::Value(1);
        let duplicate = ScalarNode::new(expr1, props);
        assert_ne!(scalar_node, duplicate, "scalar nodes in the owned state when they store to the same expr");
    }

    #[test]
    fn test_scalar_node_hash() {
        let mut hash_map = HashMap::new();

        let expr = TestScalarExpr::Value(1);
        let props = ScalarProps::default();
        let scalar_node = ScalarNode::new(expr, props.clone());

        hash_map.insert(scalar_node.clone(), 1);
        assert_eq!(hash_map.get(&scalar_node), Some(&1));

        let existing = hash_map.insert(scalar_node, 2);
        assert_eq!(existing, Some(1));

        let expr1 = TestScalarExpr::Value(1);
        let duplicate = ScalarNode::new(expr1, props);
        assert_eq!(hash_map.get(&duplicate), None);
    }

    #[test]
    fn test_properties() {
        let mut memo = new_memo();

        let leaf = RelNode::new(TestRelExpr::Leaf("a"), RelProps { a: 10 });
        let inner = RelNode::new(TestRelExpr::Node { input: leaf }, RelProps { a: 15 });
        let expr = RelNode::from(TestRelExpr::Node { input: inner });
        let outer = TestOperator::from(TestRelExpr::Node { input: expr }).with_rel_props(20);

        let _ = insert_group(&mut memo, outer);

        expect_memo_with_props(
            &memo,
            r#"
03 Node 02 a=20
02 Node 01
01 Node 00 a=15
00 Leaf a a=10
"#,
        )
    }

    #[test]
    fn test_child_expr() -> TestResult {
        let mut memo = new_memo();

        let leaf = RelNode::new(TestRelExpr::Leaf("a"), RelProps { a: 10 });
        let expr = TestOperator::from(TestRelExpr::Node { input: leaf.clone() });
        let (_, expr) = insert_group(&mut memo, expr)?;

        match expr.expr().relational() {
            TestRelExpr::Node { input } => {
                let children: Vec<_> = expr.children().collect();
                assert_eq!(
                    format!("{:?}", input.expr()),
                    format!("{:?}", children[0].expr().relational()),
                    "child expr"
                );
                assert_eq!(input.props(), leaf.props(), "input props");
            }
            _ => panic!("Unexpected expression: {:?}", expr),
        }
        Ok(())
    }

    #[test]
    fn test_memo_trivial_expr() -> TestResult {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let (group1, expr1) = insert_group(&mut memo, expr.clone())?;
        let (group2, expr2) = insert_group(&mut memo, expr)?;

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1, 1);

        let group = memo.get_group(&group1)?;
        expect_group_exprs(&group, vec![expr1]);

        expect_memo(
            &memo,
            r#"
00 Leaf a
"#,
        );

        Ok(())
    }

    #[test]
    fn test_memo_node_leaf_expr() -> TestResult {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Node {
            input: RelNode::from(TestRelExpr::Leaf("a")),
        });

        let (group1, expr1) = insert_group(&mut memo, expr.clone())?;
        let (group2, expr2) = insert_group(&mut memo, expr)?;

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );

        Ok(())
    }

    #[test]
    fn test_memo_node_multiple_leaves() -> TestResult {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Nodes {
            inputs: vec![RelNode::from(TestRelExpr::Leaf("a")), RelNode::from(TestRelExpr::Leaf("b"))],
        });

        let (group1, expr1) = insert_group(&mut memo, expr.clone())?;
        let (group2, expr2) = insert_group(&mut memo, expr)?;

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1, 1);

        expect_memo(
            &memo,
            r#"
02 Nodes [00, 01]
01 Leaf b
00 Leaf a
"#,
        );

        Ok(())
    }

    #[test]
    fn test_memo_node_nested_duplicates() -> TestResult {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Nodes {
            inputs: vec![
                RelNode::from(TestRelExpr::Node {
                    input: RelNode::from(TestRelExpr::Leaf("a")),
                }),
                RelNode::from(TestRelExpr::Leaf("b")),
                RelNode::from(TestRelExpr::Node {
                    input: RelNode::from(TestRelExpr::Leaf("a")),
                }),
            ],
        });

        let (group1, expr1) = insert_group(&mut memo, expr.clone())?;
        let (group2, expr2) = insert_group(&mut memo, expr)?;

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1, 1);

        expect_memo(
            &memo,
            r#"
03 Nodes [01, 02, 01]
02 Leaf b
01 Node 00
00 Leaf a
"#,
        );

        Ok(())
    }

    #[test]
    fn test_memo_node_duplicate_leaves() -> TestResult {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Nodes {
            inputs: vec![
                RelNode::from(TestRelExpr::Leaf("a")),
                RelNode::from(TestRelExpr::Leaf("b")),
                RelNode::from(TestRelExpr::Leaf("a")),
            ],
        });

        let (group1, expr1) = insert_group(&mut memo, expr.clone())?;
        let (group2, expr2) = insert_group(&mut memo, expr)?;

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1, 1);

        expect_memo(
            &memo,
            r#"
02 Nodes [00, 01, 00]
01 Leaf b
00 Leaf a
"#,
        );

        Ok(())
    }

    #[test]
    fn test_insert_member() -> TestResult {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));

        let (group_id, _) = insert_group(&mut memo, expr)?;
        expect_memo(
            &memo,
            r#"
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestRelExpr::Leaf("a0"));
        let _ = insert_group_member(&mut memo, group_id, expr)?;

        expect_memo(
            &memo,
            r#"
00 Leaf a
   Leaf a0
"#,
        );

        Ok(())
    }

    #[test]
    fn test_insert_member_to_a_leaf_group() -> TestResult {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Node {
            input: RelNode::from(TestRelExpr::Leaf("a")),
        });

        let (group_id, _) = insert_group(&mut memo, expr)?;
        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );

        let group = memo.get_group(&group_id)?;
        let child_expr = group.mexpr().children().next().unwrap();
        let child_group_id = child_expr.group_id();
        let expr = TestOperator::from(TestRelExpr::Leaf("a0"));
        let _ = insert_group_member(&mut memo, child_group_id, expr)?;

        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
   Leaf a0
"#,
        );

        Ok(())
    }

    #[test]
    fn test_insert_member_to_the_top_group() -> TestResult {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Node {
            input: RelNode::from(TestRelExpr::Leaf("a")),
        });

        let (group_id, _) = insert_group(&mut memo, expr)?;
        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestRelExpr::Node {
            input: RelNode::from(TestRelExpr::Leaf("a0")),
        });
        let _ = insert_group_member(&mut memo, group_id, expr)?;

        expect_memo(
            &memo,
            r#"
02 Leaf a0
01 Node 00
   Node 02
00 Leaf a
"#,
        );

        Ok(())
    }

    #[test]
    fn test_memo_debug_formatting() -> TestResult {
        let mut memo = MemoBuilder::new(123).build();
        assert_eq!(format!("{:?}", memo), "Memo { num_groups: 0, num_exprs: 0, expr_to_group: {}, metadata: 123 }");

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let (group_id, expr) = insert_group(&mut memo, expr)?;

        assert_eq!(
            format!("{:?}", memo),
            "Memo { num_groups: 1, num_exprs: 1, expr_to_group: {ExprId(0): GroupId(0)}, metadata: 123 }"
        );
        assert_eq!(format!("{:?}", expr), "MemoExprRef { id: ExprId(0) }");

        let group = memo.get_group(&group_id)?;
        assert_eq!(format!("{:?}", group.mexprs()), "[MemoExprRef { id: ExprId(0) }]");

        let expr = TestOperator::from(TestRelExpr::Node {
            input: RelNode::try_from(group.to_memo_expr())?,
        });
        assert_eq!(
            format!("{:?}", expr.children()),
            "[TestOperator { state: Memo(MemoizedExpr { expr_id: ExprId(0), group_id: GroupId(0) }) }]"
        );
        Ok(())
    }

    #[test]
    fn test_trivial_nestable_expr() -> TestResult {
        let sub_expr = RelNode::from(TestRelExpr::Leaf("a"));
        let expr = TestScalarExpr::Gt {
            lhs: Box::new(TestScalarExpr::Value(100)),
            rhs: Box::new(TestScalarExpr::SubQuery(sub_expr)),
        };
        let expr = TestOperator {
            state: MemoExprState::new(TestExpr::Scalar(expr), TestProps::Scalar(ScalarProps::default())),
        };

        let mut memo = new_memo();
        let _ = insert_group(&mut memo, expr)?;

        expect_memo(
            &memo,
            r#"
01 Expr (100 > SubQuery 00)
00 Leaf a
"#,
        );

        Ok(())
    }

    #[test]
    fn test_scalar_expr_complex_nested_exprs() -> TestResult {
        let sub_expr = RelNode::from(TestRelExpr::Filter {
            input: RelNode::from(TestRelExpr::Leaf("a")),
            filter: TestScalarExpr::Value(111).into(),
        });

        let inner_filter = TestScalarExpr::Gt {
            lhs: Box::new(TestScalarExpr::Value(100)),
            rhs: Box::new(TestScalarExpr::SubQuery(sub_expr)),
        };

        let sub_expr2 = TestScalarExpr::SubQuery(RelNode::from(TestRelExpr::Leaf("b")));
        let filter_expr = TestScalarExpr::Gt {
            lhs: Box::new(sub_expr2.clone()),
            rhs: Box::new(inner_filter.clone()),
        };

        let expr = TestOperator::from(TestRelExpr::Filter {
            input: RelNode::from(TestRelExpr::Leaf("a")),
            filter: filter_expr.into(),
        });

        let mut memo = new_memo();
        let (group_id, _) = insert_group(&mut memo, expr)?;

        expect_memo(
            &memo,
            r#"
05 Filter 00 filter=04
04 Expr (SubQuery 01 > (100 > SubQuery 03))
03 Filter 00 filter=02
02 Expr 111
01 Leaf b
00 Leaf a
"#,
        );

        let group = memo.get_group(&group_id)?;
        let children: Vec<_> = group.mexpr().children().collect();
        assert_eq!(
            children.len(),
            2,
            "Filter expression has 2 child expressions: an input and a filter: {:?}",
            children
        );

        let expr = &children[1];
        assert_eq!(expr.children().len(), 2, "Filter expression has 2 nested expressions: {:?}", expr.children());

        let filter_expr = TestScalarExpr::Gt {
            lhs: Box::new(inner_filter),
            rhs: Box::new(sub_expr2),
        };

        let expr = TestOperator::from(TestRelExpr::Filter {
            input: RelNode::from(TestRelExpr::Leaf("a")),
            filter: filter_expr.into(),
        });

        let mut memo = new_memo();
        let _ = insert_group(&mut memo, expr)?;

        expect_memo(
            &memo,
            r#"
05 Filter 00 filter=04
04 Expr ((100 > SubQuery 02) > SubQuery 03)
03 Leaf b
02 Filter 00 filter=01
01 Expr 111
00 Leaf a
"#,
        );

        Ok(())
    }

    #[test]
    fn test_callback() -> TestResult {
        #[derive(Debug)]
        struct Callback {
            added: Rc<RefCell<Vec<String>>>,
        }
        impl MemoGroupCallback for Callback {
            type Expr = TestExpr;
            type Props = TestProps;
            type Scope = TestScope;
            type Metadata = ();

            fn new_group(
                &self,
                expr: &Self::Expr,
                _scope: &Self::Scope,
                provided_props: Self::Props,
                _metadata: &Self::Metadata,
            ) -> Result<Self::Props, OptimizerError> {
                let mut added = self.added.borrow_mut();
                let mut buf = String::new();
                let mut fmt = StringMemoFormatter::new(&mut buf, MemoFormatterFlags::All);
                TestOperator::format_expr(expr, &provided_props, &mut fmt);
                added.push(buf);
                Ok(provided_props)
            }
        }

        let added = Rc::new(RefCell::new(vec![]));
        let callback = Callback { added: added.clone() };
        let mut memo = MemoBuilder::new(()).set_callback(Rc::new(callback)).build();

        insert_group(&mut memo, TestOperator::from(TestRelExpr::Leaf("A")))?;

        {
            let added = added.borrow();
            assert_eq!(added[0], "Leaf A");
            assert_eq!(added.len(), 1);
        }

        let leaf = RelNode::from(TestRelExpr::Leaf("B"));
        let expr = TestOperator::from(TestRelExpr::Node { input: leaf });
        insert_group(&mut memo, expr.clone())?;

        {
            let added = added.borrow();
            assert_eq!(added[1], "Leaf B");
            assert_eq!(added[2], "Node 01");
            assert_eq!(added.len(), 3);
        }

        insert_group(&mut memo, expr)?;
        {
            let added = added.borrow();
            assert_eq!(added.len(), 3, "added duplicates. memo:\n{}", format_memo(&memo));
        }
        Ok(())
    }

    #[test]
    fn test_formatter() -> TestResult {
        let mut memo = new_memo();

        let node_a = TestOperator::from(TestRelExpr::Leaf("a"));
        let node_a = memo.insert_group(node_a, &TestScope)?;

        let node = TestOperator::from(TestRelExpr::Node {
            input: RelNode::from(TestRelExpr::Leaf("b")),
        });
        let node = memo.insert_group(node, &TestScope)?;

        let nodes = TestOperator::from(TestRelExpr::Nodes {
            inputs: vec![node_a.clone().into_rel_node()?, node.clone().into_rel_node()?],
        });
        let nodes = memo.insert_group(nodes, &TestScope)?;

        fn format_expr(expr: &TestOperator, expected: &str) {
            let mut buf = String::new();
            let mut fmt = StringMemoFormatter::new(&mut buf, MemoFormatterFlags::All);
            TestOperator::format_expr(expr.expr(), expr.props(), &mut fmt);
            assert_eq!(buf, expected, "expr format")
        }

        format_expr(&node_a, "Leaf a");
        format_expr(&node, "Node 01");
        format_expr(&nodes, "Nodes [00, 02]");

        Ok(())
    }

    #[test]
    fn test_formatter_flags() {
        assert!(!MemoFormatterFlags::None.has_flag(&MemoFormatterFlags::None));
        assert!(!MemoFormatterFlags::None.has_flag(&MemoFormatterFlags::IncludeProps));
        assert!(!MemoFormatterFlags::None.has_flag(&MemoFormatterFlags::Digest));
        assert!(!MemoFormatterFlags::None.has_flag(&MemoFormatterFlags::All));

        assert!(!MemoFormatterFlags::Digest.has_flag(&MemoFormatterFlags::None));
        assert!(MemoFormatterFlags::Digest.has_flag(&MemoFormatterFlags::Digest));
        assert!(!MemoFormatterFlags::Digest.has_flag(&MemoFormatterFlags::IncludeProps));
        assert!(!MemoFormatterFlags::Digest.has_flag(&MemoFormatterFlags::All));

        assert!(!MemoFormatterFlags::IncludeProps.has_flag(&MemoFormatterFlags::None));
        assert!(!MemoFormatterFlags::IncludeProps.has_flag(&MemoFormatterFlags::Digest));
        assert!(MemoFormatterFlags::IncludeProps.has_flag(&MemoFormatterFlags::IncludeProps));
        assert!(!MemoFormatterFlags::IncludeProps.has_flag(&MemoFormatterFlags::All));

        assert!(!MemoFormatterFlags::All.has_flag(&MemoFormatterFlags::None));
        assert!(!MemoFormatterFlags::All.has_flag(&MemoFormatterFlags::Digest));
        assert!(MemoFormatterFlags::All.has_flag(&MemoFormatterFlags::IncludeProps));
        assert!(MemoFormatterFlags::All.has_flag(&MemoFormatterFlags::All));
    }
}
