use crate::memo::{GroupId, Memo, MemoBuilder, MemoExpr};

#[cfg(not(feature = "unsafe_memo"))]
use super::default_impl::*;
#[cfg(feature = "unsafe_memo")]
use super::unsafe_impl::*;

pub fn new_memo<E>() -> Memo<E, ()>
where
    E: MemoExpr,
{
    MemoBuilder::new(()).build()
}

pub fn insert_group<E, T>(memo: &mut Memo<E, T>, expr: E) -> (GroupId, MemoExprRef<E>)
where
    E: MemoExpr,
{
    let expr = memo.insert_group(expr);
    let group_id = expr.state().memo_group_id();
    let expr_ref = expr.state().memo_expr();

    (group_id, expr_ref.clone())
}

pub fn insert_group_member<E, T>(memo: &mut Memo<E, T>, group_id: GroupId, expr: E) -> (GroupId, MemoExprRef<E>)
where
    E: MemoExpr,
{
    let group = memo.get_group(&group_id);
    let token = group.to_group_token();
    let expr = memo.insert_group_member(token, expr);

    let expr_ref = expr.state().memo_expr();

    (group_id, expr_ref.clone())
}

pub fn expect_group_size<E, T>(memo: &Memo<E, T>, group_id: &GroupId, size: usize)
where
    E: MemoExpr,
{
    let group = memo.get_group(group_id);
    assert_eq!(group.mexprs().count(), size, "group#{}", group_id);
}

pub fn expect_group_exprs<E, T>(group: &MemoGroupRef<E, T>, expected: Vec<MemoExprRef<E>>)
where
    E: MemoExpr,
{
    let actual: Vec<MemoExprRef<E>> = group.mexprs().map(|e| e.clone()).collect();
    assert_eq!(actual, expected, "group#{} exprs", group.id());
}

pub fn expect_memo<E, T>(memo: &Memo<E, T>, expected: &str)
where
    E: MemoExpr,
{
    let lines: Vec<String> = expected.split('\n').map(String::from).collect();
    let expected = lines.join("\n");

    let buf = format_memo_impl(&memo.memo_impl);
    assert_eq!(buf.trim(), expected.trim());
}

pub fn expect_memo_with_props<E, T>(memo: &Memo<E, T>, expected: &str)
where
    E: MemoExpr,
{
    let lines: Vec<String> = expected.split('\n').map(String::from).collect();
    let expected = lines.join("\n");

    let buf = format_memo_impl(&memo.memo_impl);
    assert_eq!(buf.trim(), expected.trim());
}
