//! Default implementation of a memo data structure.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use triomphe::Arc;

use crate::memo::{
    create_group_properties, make_digest, CopyInExprs, Expr, ExprContext, MemoExpr, MemoGroupCallbackRef,
    NewChildExprs, OwnedExpr, Props, StringMemoFormatter,
};

/// Implementation of a memo data structure in which [MemoizedExpr](self::MemoizedExpr) are backed by `Arc`s.
/// Internally this implementation uses `Vec`s as a storage for both groups and expressions.
pub struct MemoImpl<E, T>
where
    E: MemoExpr,
{
    groups: Vec<(Vec<ExprId>, Arc<MemoGroupData<E>>)>,
    exprs: Vec<E>,
    expr_cache: HashMap<String, ExprId>,
    expr_to_group: HashMap<ExprId, GroupId>,
    metadata: T,
    callback: Option<MemoGroupCallbackRef<E::Expr, E::Props, T>>,
}

impl<E, T> MemoImpl<E, T>
where
    E: MemoExpr,
{
    pub(crate) fn new(metadata: T) -> Self {
        MemoImpl {
            exprs: Vec::new(),
            groups: Vec::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            metadata,
            callback: None,
        }
    }

    pub(crate) fn with_callback(metadata: T, callback: MemoGroupCallbackRef<E::Expr, E::Props, T>) -> Self {
        MemoImpl {
            exprs: Vec::new(),
            groups: Vec::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            metadata,
            callback: Some(callback),
        }
    }

    pub(crate) fn get_group(&self, group_id: &GroupId) -> MemoGroupRef<E, T> {
        let (group_exprs, group_data) = &self.groups[group_id.index()];
        let first_expr_id = group_exprs[0];
        let expr = self.exprs[first_expr_id.index()].clone();

        let first_expr = MemoExprRef {
            id: first_expr_id,
            expr,
        };
        MemoGroupRef {
            group_id: *group_id,
            data: group_data,
            memo: self,
            exprs: group_exprs,
            expr: first_expr,
        }
    }

    pub(crate) fn metadata(&self) -> &T {
        &self.metadata
    }

    pub(crate) fn num_groups(&self) -> usize {
        self.groups.len()
    }

    pub(crate) fn num_exprs(&self) -> usize {
        self.exprs.len()
    }

    pub(crate) fn expr_to_group(&self) -> &HashMap<ExprId, GroupId> {
        &self.expr_to_group
    }

    fn add_group(&mut self, expr_id: ExprId, props: E::Props) -> Arc<MemoGroupData<E>> {
        let group_id = GroupId(self.groups.len());
        let group_data = MemoGroupData { group_id, props };
        let group_data = Arc::new(group_data);
        self.groups.push((vec![expr_id], group_data.clone()));
        group_data
    }

    fn add_expr(&mut self, expr_id: ExprId, expr: E::Expr, group_data: Arc<MemoGroupData<E>>) -> E {
        let group_id = group_data.group_id;
        let expr_data = MemoExprData { expr_id, expr };
        let expr_data = Arc::new(expr_data);
        let memo_state = MemoizedExpr {
            expr: expr_data,
            group: group_data,
        };
        let expr = E::from_state(MemoExprState::Memo(memo_state));
        self.expr_to_group.insert(expr_id, group_id);
        self.exprs.push(expr.clone());
        expr
    }

    pub(crate) fn get_first_memo_expr(&self, group_id: GroupId) -> E {
        let (group_exprs, _) = &self.groups[group_id.index()];
        let first_expr_id = group_exprs[0];

        self.exprs[first_expr_id.index()].clone()
    }
}

/// Uniquely identifies a memo group in a memo.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct GroupId(usize);

impl GroupId {
    fn index(&self) -> usize {
        self.0
    }
}

impl Display for GroupId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:02}", self.0)
    }
}

impl Debug for GroupId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GroupId").field(&self.0).finish()
    }
}

/// Uniquely identifies a memo expression in a memo.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct ExprId(usize);

impl ExprId {
    fn index(&self) -> usize {
        self.0
    }
}

impl Display for ExprId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:02}", self.0)
    }
}

impl Debug for ExprId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ExprId").field(&self.0).finish()
    }
}

/// A borrowed memo group. A memo group is a group of logically equivalent expressions.
/// Expressions are logically equivalent when they produce the same result.
pub struct MemoGroupRef<'a, E, T>
where
    E: MemoExpr,
{
    group_id: GroupId,
    memo: &'a MemoImpl<E, T>,
    data: &'a Arc<MemoGroupData<E>>,
    exprs: &'a [ExprId],
    // the first expression in this group
    expr: MemoExprRef<E>,
}

impl<'a, E, T> MemoGroupRef<'a, E, T>
where
    E: MemoExpr,
{
    /// Returns an opaque identifier of this memo group.
    pub fn id(&self) -> GroupId {
        self.group_id
    }

    /// Returns properties shared by all expressions in this memo group.
    pub fn props(&self) -> &E::Props {
        let (_, group_data) = &self.memo.groups[self.group_id.index()];
        &group_data.props
    }

    /// Returns a reference to an expression of the first memo expression in this memo group.
    pub fn expr(&self) -> &E::Expr {
        self.expr.expr()
    }

    /// Returns a reference to the first memo expression of this memo group.
    pub fn mexpr(&self) -> &MemoExprRef<E> {
        &self.expr
    }

    /// Returns an iterator over the memo expressions that belong to this memo group.
    pub fn mexprs(&self) -> MemoGroupIter<E, T> {
        MemoGroupIter {
            group_id: self.group_id,
            data: self.data,
            memo: self.memo,
            exprs: self.exprs,
            position: 0,
        }
    }

    /// Consumes this borrowed group and returns a [group token]('MemoGroupToken') that can be used
    /// to add an expression to this memo group.
    pub fn to_group_token(self) -> MemoGroupToken<E> {
        MemoGroupToken {
            group_id: self.group_id,
            marker: Default::default(),
        }
    }

    /// Convert this borrowed memo group to the first memo expression in this group.
    pub fn to_memo_expr(self) -> E {
        self.expr.expr
    }
}

/// A token that allows the owner to add an expression into a memo group this token was obtained from.
///
/// A memo group token can be obtained from a memo group by calling [to_group_token](MemoGroupRef::to_group_token)
/// of that memo group.
#[derive(Clone)]
pub struct MemoGroupToken<E> {
    group_id: GroupId,
    marker: std::marker::PhantomData<E>,
}

impl<E> Debug for MemoGroupToken<E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoGroupToken").field("id", &self.group_id).finish()
    }
}

/// An iterator over expressions in a memo group.
pub struct MemoGroupIter<'a, E, T>
where
    E: MemoExpr,
{
    group_id: GroupId,
    memo: &'a MemoImpl<E, T>,
    data: &'a Arc<MemoGroupData<E>>,
    exprs: &'a [ExprId],
    position: usize,
}

impl<'a, E, T> Iterator for MemoGroupIter<'a, E, T>
where
    E: MemoExpr,
{
    type Item = MemoExprRef<E>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(expr_id) = self.exprs.get(self.position) {
            let expr = self.memo.exprs[expr_id.index()].clone();
            self.position += 1;

            Some(MemoExprRef { id: *expr_id, expr })
        } else {
            None
        }
    }
}

impl<'m, E, T> Debug for MemoGroupIter<'m, E, T>
where
    E: MemoExpr + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let iter = MemoGroupIter {
            group_id: self.group_id,
            memo: self.memo,
            data: self.data,
            exprs: self.exprs,
            position: self.position,
        };
        f.debug_list().entries(iter).finish()
    }
}

struct MemoExprData<E>
where
    E: MemoExpr,
{
    expr_id: ExprId,
    expr: E::Expr,
}

impl<E> MemoExprData<E>
where
    E: MemoExpr,
{
    fn expr_id(&self) -> ExprId {
        self.expr_id
    }

    fn expr(&self) -> &E::Expr {
        &self.expr
    }
}

pub(crate) struct MemoGroupData<E>
where
    E: MemoExpr,
{
    group_id: GroupId,
    props: E::Props,
}

impl<E> MemoGroupData<E>
where
    E: MemoExpr,
{
    fn props(&self) -> &E::Props {
        &self.props
    }
}

/// A reference to a [memo expression](self::MemoExpr) that can be cheaply cloned (internally its just [an identifier](self::ExprId) and a pointer).
#[derive(Clone)]
pub struct MemoExprRef<E>
where
    E: MemoExpr,
{
    id: ExprId,
    expr: E,
}

impl<E> MemoExprRef<E>
where
    E: MemoExpr,
{
    /// Returns an opaque identifier of this memo expression.
    pub fn id(&self) -> ExprId {
        self.id
    }

    /// Returns the expression this memo expression represents.
    pub fn expr(&self) -> &E::Expr {
        self.expr.expr()
    }

    /// Returns a reference to the properties of the memo group this expression belongs to.
    pub fn props(&self) -> &E::Props {
        self.expr.props()
    }

    /// Returns an identifier of a memo group this memo expression belongs to.
    pub fn group_id(&self) -> GroupId {
        self.expr.state().memo_group_id()
    }

    /// Returns an iterator over child expressions of this memo expression.{
    pub fn children(&self) -> MemoExprInputsIter<E> {
        MemoExprInputsIter {
            expr: &self.expr,
            position: 0,
            num_children: self.expr.num_children(),
        }
    }
}

impl<E> PartialEq for MemoExprRef<E>
where
    E: MemoExpr,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.expr.state().equal(other.expr.state())
    }
}

impl<E> Eq for MemoExprRef<E> where E: MemoExpr {}

/// An iterator over child expressions of a memo expression.
pub struct MemoExprInputsIter<'a, E> {
    expr: &'a E,
    position: usize,
    num_children: usize,
}

impl<'a, E> Iterator for MemoExprInputsIter<'a, E>
where
    E: MemoExpr,
{
    type Item = MemoExprRef<E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(expr) = self.expr.get_child(self.position) {
            self.position += 1;
            match expr.state() {
                // MemoExprRef.expr stores a memo expression whose child expressions are replaced with references.
                MemoExprState::Owned(_) => unreachable!("MemoExprInputsIter contains non memoized child expressions"),
                MemoExprState::Memo(state) => {
                    let expr = MemoExprRef {
                        id: state.expr.expr_id,
                        expr: E::from_state(expr.state().clone()),
                    };
                    Some(expr)
                }
            }
        } else {
            None
        }
    }
}

impl<'a, E> Debug for MemoExprInputsIter<'a, E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let iter = MemoExprInputsIter {
            expr: self.expr,
            position: self.position,
            num_children: self.num_children,
        };
        f.debug_list().entries(iter).finish()
    }
}

impl<'a, E> ExactSizeIterator for MemoExprInputsIter<'a, E>
where
    E: MemoExpr,
{
    fn len(&self) -> usize {
        self.num_children
    }
}

/// Represents a state of a [memo expression](super::MemoExpr).
///
/// - In the `Owned` state a memo expression stores expression and properties.
/// - In the `Memo` state a memo expression stores a reference to expression stored in a memo.
#[derive(Clone)]
pub enum MemoExprState<E>
where
    E: MemoExpr,
{
    Owned(OwnedExpr<E>),
    Memo(MemoizedExpr<E>),
}

impl<E> MemoExprState<E>
where
    E: MemoExpr,
{
    /// Creates an owned state with the given expression and properties.
    pub fn new(expr: E::Expr, props: E::Props) -> Self {
        MemoExprState::Owned(OwnedExpr {
            expr: Arc::new(expr),
            props: Arc::new(props),
        })
    }

    /// Returns a reference to an expression.
    /// * In the owned state this method returns a reference to the underlying expression.
    /// * In the memo state this method returns a reference to the first expression in the [memo group][self::MemoGroupRef].
    pub fn expr(&self) -> &E::Expr {
        match self {
            MemoExprState::Owned(state) => state.expr.as_ref(),
            MemoExprState::Memo(state) => state.expr.expr(),
        }
    }

    /// Returns a reference to properties.
    ///
    /// * In the owned state this method returns a reference to the underlying properties.
    /// * In the memo state this method returns a reference to properties of a memo group the expression belongs to.
    pub fn props(&self) -> &E::Props {
        match self {
            MemoExprState::Owned(state) => state.props.as_ref(),
            MemoExprState::Memo(state) => state.group.props(),
        }
    }

    /// Returns `true` if this `MemoExprState` is in the `Memo` state.
    pub fn is_memo(&self) -> bool {
        match self {
            MemoExprState::Owned(_) => false,
            MemoExprState::Memo(_) => true,
        }
    }

    /// Returns a reference to a memo expression or panics if this `MemoExprState` is not in the memo state.
    /// This method should only be called on memoized expressions.
    ///
    /// # Panics
    ///
    /// This method panics if this `MemoExprState` is in the owned state.
    pub fn memo_expr(&self) -> MemoExprRef<E> {
        match self {
            MemoExprState::Owned(_) => panic!("This should only be called on memoized expressions"),
            MemoExprState::Memo(state) => MemoExprRef {
                id: state.expr.expr_id,
                expr: E::from_state(self.clone()),
            },
        }
    }

    /// Returns an identifier of a memo group or panics if this `MemoExprState` is not in the `Memo` state.
    /// This method should only be called on memoized expressions.
    ///
    /// # Panics
    ///
    /// This method panics if this `MemoExprState` is in the `Owned` state.
    pub fn memo_group_id(&self) -> GroupId {
        match self {
            MemoExprState::Owned(_) => panic!("This should only be called on memoized expressions"),
            MemoExprState::Memo(state) => state.group_id(),
        }
    }

    pub(crate) fn equal(&self, other: &Self) -> bool {
        let this = self.get_eq_state();
        let that = other.get_eq_state();
        // Expressions are equal if their expr pointers point to the same expression.
        match (this, that) {
            // Raw pointer equality for owned expressions is used here because we should not check non-memoized
            // expressions for equality.
            ((Some(x), _), (Some(y), _)) => std::ptr::eq(x, y),
            ((_, Some(x)), (_, Some(y))) => x == y,
            // Unreachable
            _ => false,
        }
    }

    pub(crate) fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            MemoExprState::Owned(s) => {
                let ptr: *const E::Expr = s.expr.as_ptr();
                ptr.hash(state)
            }
            MemoExprState::Memo(s) => s.expr_id().hash(state),
        }
    }

    fn get_eq_state(&self) -> (Option<*const E::Expr>, Option<ExprId>)
    where
        E: MemoExpr,
    {
        match self {
            MemoExprState::Owned(state) => {
                let ptr: *const E::Expr = state.expr.as_ptr();
                (Some(ptr), None)
            }
            MemoExprState::Memo(state) => (None, Some(state.expr_id())),
        }
    }
}

impl<E, T, P> Debug for MemoExprState<E>
where
    E: MemoExpr<Expr = T, Props = P> + Debug,
    T: Expr + Debug,
    P: Props + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoExprState::Owned(state) => f.debug_tuple("Owned").field(state).finish(),
            MemoExprState::Memo(state) => f.debug_tuple("Memo").field(state).finish(),
        }
    }
}

/// Represents an expression stored in a [memo](crate::memo::Memo).
/// This is the `memo` state of a [MemoExprState](self::MemoExprState).
#[derive(Clone)]
// get_expr_ref fails to compiler under 1.57.
// note: rustc 1.57.0 (f1edd0429 2021-11-29) running on x86_64-apple-darwin
// thread 'rustc' panicked at 'called `Option::unwrap()` on a `None` value'
// /compiler/rustc_hir/src/definitions.rs:452:14
pub struct MemoizedExpr<E>
where
    E: MemoExpr,
{
    expr: Arc<MemoExprData<E>>,
    group: Arc<MemoGroupData<E>>,
}

impl<E> MemoizedExpr<E>
where
    E: MemoExpr,
{
    /// Returns an identifier of a group this expression belongs to.
    pub fn group_id(&self) -> GroupId {
        self.group.group_id
    }

    fn expr_id(&self) -> ExprId {
        self.expr.expr_id()
    }
    //
    // pub(super) fn expr(&self) -> &E::Expr {
    //     self.expr.expr()
    // }
    //
    // pub(super) fn props(&self) -> &E::Props {
    //     self.group.props()
    // }
    //
    // pub fn get_expr_ref(&self, state: &MemoExprState<E>) -> MemoExprRef<E> {
    //     MemoExprRef {
    //         id: self.expr.expr_id,
    //         expr: E::from_state(state.clone()),
    //     }
    // }
}

impl<E, T, P> Debug for MemoizedExpr<E>
where
    E: MemoExpr<Expr = T, Props = P> + Debug,
    T: Expr + Debug,
    P: Props + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoizedExpr")
            .field("expr_id", &self.expr.expr_id())
            .field("group_id", &self.group_id())
            .finish()
    }
}

pub(crate) fn copy_in_expr_impl<E, T>(copy_in_exprs: &mut CopyInExprs<E, T>, expr: &E, expr_ctx: ExprContext<E>)
where
    E: MemoExpr,
{
    let expr_id = ExprId(copy_in_exprs.memo.exprs.len());

    let ExprContext { children, parent } = expr_ctx;
    let props = create_group_properties(expr, &children);

    let expr = E::expr_with_new_children(expr.expr(), NewChildExprs::new(children));
    let digest = make_digest::<E>(&expr, &props);

    let (expr_id, added) = match copy_in_exprs.memo.expr_cache.entry(digest) {
        Entry::Occupied(o) => (*o.get(), false),
        Entry::Vacant(v) => {
            v.insert(expr_id);
            (expr_id, true)
        }
    };

    let expr = if added {
        if let Some(token) = parent {
            let parent_id = token.group_id;
            let (group_exprs, expr_group_data) =
                copy_in_exprs.memo.groups.get_mut(parent_id.index()).expect("Unexpected group");
            let expr_group_data = expr_group_data.clone();
            group_exprs.push(expr_id);

            copy_in_exprs.memo.add_expr(expr_id, expr, expr_group_data)
        } else {
            let props = if let Some(callback) = &copy_in_exprs.memo.callback {
                callback.new_group(&expr, &props, &copy_in_exprs.memo.metadata)
            } else {
                props
            };

            let group_data = copy_in_exprs.memo.add_group(expr_id, props);
            copy_in_exprs.memo.add_expr(expr_id, expr, group_data)
        }
    } else {
        let expr = copy_in_exprs.memo.exprs.get(expr_id.index()).expect("Unexpected expression id");
        expr.clone()
    };
    copy_in_exprs.result = Some(expr);
}

/// Builds a textual representation of the given memo.
pub(crate) fn format_memo_impl<E, T>(memo: &MemoImpl<E, T>) -> String
where
    E: MemoExpr,
{
    let mut buf = String::new();
    let mut f = StringMemoFormatter::new(&mut buf);

    for (exprs, group) in memo.groups.iter().rev() {
        f.push_str(format!("{} ", group.group_id).as_str());
        for (i, expr_id) in exprs.iter().enumerate() {
            if i > 0 {
                // newline + 3 spaces
                f.push_str("\n   ");
            }
            let expr = &memo.exprs[expr_id.index()];
            E::format_expr(expr.expr(), expr.props(), &mut f);
        }
        f.push('\n');
    }

    buf
}
