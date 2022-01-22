mod store;

use crate::memo::unsafe_impl::store::{
    AppendOnlyStore, ImmutableRef, Store, StoreElementId, StoreElementIter, StoreElementRef,
};
use crate::memo::{
    create_group_properties, make_digest, CopyIn, CopyInExprs, DisplayMemoExprFormatter, Expr, ExprContext, MemoExpr,
    MemoGroupCallback, NewChildExprs, OwnedExpr, Props, StringMemoFormatter,
};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;
use triomphe::Arc;

/// The number of elements per page.
#[doc(hidden)]
const PAGE_SIZE: usize = 8;

/// Implementation of a memo that uses raw pointers.
///
/// # Safety
///
/// Groups (a set of expressions) and properties (properties of a memo group) are stored
/// separately in order not to break aliasing rules. Since there is no API to obtain mutable
/// references to memo expressions and properties it should be safe for user code work
/// with them (taking into account the fact that memo expressions should not outlive
/// the memo they are stored in) this is not the case for groups (see source code for more details).
///
/// * It's a caller's responsibility to ensure that [memo expressions](super::MemoExpr),
/// [references to memo expressions](self::MemoExprRef) will not outlive the memo.
///
pub(crate) struct MemoImpl<E, T>
where
    E: MemoExpr,
{
    groups: GroupDataStore<E>,
    exprs: AppendOnlyStore<E>,
    props: AppendOnlyStore<E::Props>,
    expr_cache: HashMap<String, ExprId>,
    expr_to_group: HashMap<ExprId, GroupId>,
    metadata: T,
    callback: Option<Rc<dyn MemoGroupCallback<Expr = E::Expr, Props = E::Props, Metadata = T>>>,
}

impl<E, T> MemoImpl<E, T>
where
    E: MemoExpr,
{
    pub fn new(metadata: T) -> Self {
        MemoImpl {
            groups: GroupDataStore::new(),
            exprs: AppendOnlyStore::new(),
            props: AppendOnlyStore::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            metadata,
            callback: None,
        }
    }

    pub fn with_callback(
        metadata: T,
        callback: Rc<dyn MemoGroupCallback<Expr = E::Expr, Props = E::Props, Metadata = T>>,
    ) -> Self {
        MemoImpl {
            groups: GroupDataStore::new(),
            exprs: AppendOnlyStore::new(),
            props: AppendOnlyStore::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            metadata,
            callback: Some(callback),
        }
    }

    pub fn insert_group(&mut self, expr: E) -> E {
        let copy_in = CopyIn {
            memo: self,
            parent: None,
        };
        copy_in.execute(&expr)
    }

    pub fn insert_group_member(&mut self, token: MemoGroupToken<E>, expr: E) -> E {
        let copy_in = CopyIn {
            memo: self,
            parent: Some(token),
        };
        copy_in.execute(&expr)
    }

    pub fn get_group(&self, group_id: &GroupId) -> MemoGroupRef<E, T> {
        let group_ref = self.get_group_ref(group_id);
        MemoGroupRef {
            id: *group_id,
            data_ref: group_ref.inner,
            marker: Default::default(),
        }
    }

    pub fn metadata(&self) -> &T {
        &self.metadata
    }

    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    pub fn num_exprs(&self) -> usize {
        self.exprs.len()
    }

    pub fn expr_to_group(&self) -> &HashMap<ExprId, GroupId> {
        &self.expr_to_group
    }

    fn get_expr_ref(&self, expr_id: &ExprId) -> MemoExprRef<E> {
        let expr_ref = self
            .exprs
            .get(expr_id.index())
            .unwrap_or_else(|| panic!("expr id is invalid: {}", expr_id));

        MemoExprRef::new(*expr_id, expr_ref)
    }

    fn add_group(&mut self, group: MemoGroupData<E>) -> MemoGroupToken<E> {
        let (id, elem_ref) = self.groups.insert(group);
        let group_ref = InternalGroupRef::new(GroupId(id), elem_ref);

        MemoGroupToken(group_ref)
    }

    fn add_expr(
        &mut self,
        expr: E::Expr,
        membership_token: MemoGroupToken<E>,
    ) -> (InternalGroupRef<E>, MemoExprRef<E>) {
        // SAFETY:
        // An expression can only be added to a group if a caller owns a membership token.
        // That token can be obtained in two ways:
        //
        // 1) when a new group is added to a memo (internal API) - add_group returns a membership token
        // which is then passed to this method - no references to MemoGroupData exists
        // and we can safely obtain a mutable reference to via groups.get_mut
        //
        // 2) by consuming MemoGroupRef (public API) - in this case borrow checker will reject a program
        // in which both a reference to the underlying MemoGroupData and the MemoGroupToken to that MemoGroupData
        // exists at the same time.
        let group_ref = membership_token.into_inner();
        let group_id = group_ref.id;

        // Make a reference to group data in order to retrieve a group properties reference stored in the memo.
        let props_ref = unsafe {
            let group_data = group_ref.inner.get();
            group_data.props_ref.clone()
        };
        // At this point no references to the group data exists.
        let state = MemoExprState::Memo(MemoizedExpr {
            // We can store ExprPtr in the owned state and ExprGroupPtr in the memo state
            // because there is no direct way to retrieve this expression.
            expr: ExprPtr::Owned(Arc::new(expr)),
            group: ExprGroupPtr::Memo((group_id, props_ref)),
        });
        let expr = E::from_state(state);

        let (id, elem_ref) = self.exprs.insert(expr);
        let expr_id = ExprId(id);
        let expr_ref = MemoExprRef::new(expr_id, elem_ref);

        // SAFETY: The following code is safe because MemoGroupToken guarantees the absence of
        // other references to MemoGroupData.
        unsafe {
            let memo_group = self.groups.get_mut(group_id.index()).unwrap();
            memo_group.exprs.push(expr_ref.clone());
        }
        // From now on we can use group_ref to access the group's data because
        // the mutable reference to that data is no longer exists.

        self.expr_to_group.insert(expr_id, group_id);

        (group_ref, expr_ref)
    }

    pub(crate) fn get_first_memo_expr(&self, group_id: GroupId) -> E {
        let group_ref = self.get_group(&group_id);
        group_ref.to_memo_expr()
    }

    fn get_group_ref(&self, group_id: &GroupId) -> InternalGroupRef<E> {
        let group_ref = self
            .groups
            .get(group_id.index())
            .unwrap_or_else(|| panic!("group id is invalid: {}", group_id));

        InternalGroupRef::new(*group_id, group_ref)
    }
}

/// Uniquely identifies a memo group in a memo.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct GroupId(StoreElementId);

impl GroupId {
    fn index(&self) -> StoreElementId {
        self.0
    }
}

impl Display for GroupId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:02}", self.0.index())
    }
}

impl Debug for GroupId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GroupId").field(&self.0).finish()
    }
}

/// Uniquely identifies a memo expression in a memo.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct ExprId(StoreElementId);

impl ExprId {
    fn index(&self) -> StoreElementId {
        self.0
    }
}

impl Display for ExprId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:02}", self.0.index())
    }
}

impl Debug for ExprId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ExprId").field(&self.0).finish()
    }
}

/// A borrowed memo group. A memo group is a group of logically equivalent expressions.
/// Expressions are logically equivalent when they produce the same result.
// Although T is never used it added to have the same generic parameters as safe_memo::MemoGroupRef
pub struct MemoGroupRef<'a, E, T>
where
    E: MemoExpr,
{
    id: GroupId,
    data_ref: StoreElementRef<MemoGroupData<E>>,
    marker: std::marker::PhantomData<(&'a E, T)>,
}

impl<'a, E, T> MemoGroupRef<'a, E, T>
where
    E: MemoExpr,
{
    /// Returns an opaque identifier of this memo group.
    pub fn id(&self) -> GroupId {
        self.id
    }

    /// Returns a reference to an expression of the first memo expression in this memo group.
    pub fn expr(&self) -> &E::Expr {
        let group = self.get_memo_group();
        group.exprs[0].expr()
    }

    /// Returns a reference to the first memo expression of this memo group.
    pub fn mexpr(&self) -> &MemoExprRef<E> {
        let group = self.get_memo_group();
        &group.exprs[0]
    }

    /// Returns an iterator over the memo expressions that belong to this memo group.
    pub fn mexprs(&self) -> MemoGroupIter<E> {
        let group = self.get_memo_group();
        MemoGroupIter { group, position: 0 }
    }

    /// Returns properties shared by all expressions in this memo group.
    pub fn props(&self) -> &E::Props {
        let group = self.get_memo_group();
        group.props_ref.get()
    }

    /// Consumes this borrowed group and returns a [membership token]('MemoGroupToken') that can be used
    /// to add an expression to this memo group.
    pub fn to_membership_token(self) -> MemoGroupToken<E> {
        MemoGroupToken(InternalGroupRef::new(self.id, self.data_ref))
    }

    /// Convert this borrowed memo group to the first memo expression in this group.
    pub fn to_memo_expr(self) -> E {
        let data = self.get_memo_group();
        let group_id = self.id;

        let expr_ref = data.exprs[0].clone();
        let props_ref = data.props_ref.clone();

        let state = MemoExprState::Memo(MemoizedExpr {
            expr: ExprPtr::Memo(expr_ref),
            group: ExprGroupPtr::Memo((group_id, props_ref)),
        });
        E::from_state(state)
    }

    fn get_memo_group(&self) -> &MemoGroupData<E> {
        // SAFETY: Making a reference to the group data should be safe because MemoGroupRef
        // was immutably borrowed from a memo.
        unsafe { self.data_ref.get() }
    }
}

impl<'a, E, T> Debug for MemoGroupRef<'a, E, T>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoGroupRef").field("id", &self.id).finish()
    }
}

pub(crate) fn debug_fmt_impl<M, T, E, P>(memo: &MemoImpl<M, T>, f: &mut Formatter<'_>) -> std::fmt::Result
where
    M: MemoExpr<Expr = E, Props = P> + Debug,
    T: Debug,
    E: Debug,
    P: Debug,
{
    f.debug_struct("Memo")
        .field("groups", &memo.groups.len())
        .field("exprs", &memo.exprs.len())
        .field("expr_to_group", &memo.expr_to_group)
        .field("metadata", &memo.metadata)
        .finish()
}

/// A group membership token that allows the owner to add an expression into a memo group this token was obtained from.
///
/// A Membership token can be obtained from a memo group by calling [to_membership_token](MemoGroupRef::to_membership_token)
/// of that memo group.
pub struct MemoGroupToken<E>(InternalGroupRef<E>)
where
    E: MemoExpr;

impl<E> MemoGroupToken<E>
where
    E: MemoExpr,
{
    fn into_inner(self) -> InternalGroupRef<E> {
        self.0
    }
}

impl<E> Debug for MemoGroupToken<E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoGroupToken").field("id", &self.0.id).finish()
    }
}

/// Represents a state of a [memo expression](super::MemoExpr).
///
/// In the `Owned` state a memo expression stores expression and properties.
/// In the `Memo` state a memo expression stores a reference to expression stored in a memo.
#[derive(Clone)]
// We can not move to MemoExprState to memo due to compiler bug
// see comments in safe_memo::MemoizedExpr::get_expr_ref
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
            MemoExprState::Memo(state) => &state.expr.expr(),
        }
    }

    /// Returns a reference to properties.
    ///
    /// * In the owned state this method returns a reference to the underlying properties.
    /// * In the memo state this method returns a reference to properties of a memo group the expression belongs to.
    pub fn props(&self) -> &E::Props {
        match self {
            MemoExprState::Owned(state) => state.props.as_ref(),
            MemoExprState::Memo(state) => &state.group.props(),
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
            // Clone MemoExprRef to make API compatible with safe memo
            MemoExprState::Memo(state) => state.expr.memo_expr().clone(),
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
            MemoExprState::Memo(state) => state.group.memo_group_id(),
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

/// Represents an expression stored in a memo.
#[derive(Clone)]
pub struct MemoizedExpr<E>
where
    E: MemoExpr,
{
    expr: ExprPtr<E>,
    group: ExprGroupPtr<E>,
}

impl<E> MemoizedExpr<E>
where
    E: MemoExpr,
{
    /// Returns an identifier of a group this expression belongs to.
    pub fn group_id(&self) -> GroupId {
        self.group.memo_group_id()
    }

    fn expr_id(&self) -> ExprId {
        self.expr.expr_id()
    }

    // pub(super) fn expr(&self) -> &E::Expr {
    //     self.expr.expr()
    // }
    //
    // pub(super) fn props(&self) -> &E::Props {
    //     self.group.props()
    // }
    //
    // pub fn get_expr_ref(&self, _state: &MemoExprState<E>) -> MemoExprRef<E> {
    //     self.expr.memo_expr().clone()
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
            .field("expr_id", &self.expr.memo_expr().id())
            .field("group_id", &self.group_id())
            .finish()
    }
}

/// Iterator over expressions of a memo group.
pub struct MemoGroupIter<'m, E>
where
    E: MemoExpr,
{
    group: &'m MemoGroupData<E>,
    position: usize,
}

impl<'m, E> Iterator for MemoGroupIter<'m, E>
where
    E: MemoExpr,
{
    type Item = &'m MemoExprRef<E>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.group.exprs.len() {
            let expr = &self.group.exprs[self.position];
            self.position += 1;
            Some(expr)
        } else {
            None
        }
    }
}

impl<'m, E> Debug for MemoGroupIter<'m, E>
where
    E: MemoExpr + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.group.exprs.iter().skip(self.position)).finish()
    }
}

/// A reference to a [memo expression](self::MemoExpr). Since [memo expression](self::MemoExpr) is immutable
/// a reference to it can be cheaply cloned (internally its just [an identifier](self::ExprId) and a pointer).
///
/// # Safety
///
/// A reference to a memo expression is valid until a [`memo`](crate::memo::Memo) it belongs to is dropped.
pub struct MemoExprRef<E>
where
    E: MemoExpr,
{
    id: ExprId,
    expr_ref: ImmutableRef<E>,
}

impl<E> MemoExprRef<E>
where
    E: MemoExpr,
{
    fn new(id: ExprId, expr_ref: ImmutableRef<E>) -> Self {
        MemoExprRef { id, expr_ref }
    }

    /// Returns an opaque identifier of this memo expression.
    pub fn id(&self) -> ExprId {
        self.id
    }

    /// Returns the expression this memo expression represents.
    pub fn expr(&self) -> &E::Expr {
        let expr = self.get_memo_expr();
        expr.expr()
    }

    /// Returns an identifier of a memo group this memo expression belongs to.
    pub fn group_id(&self) -> GroupId {
        let expr = self.get_memo_expr();
        expr.state().memo_group_id()
    }

    /// Returns a reference to the properties of the memo group this expression belongs to.
    pub fn props(&self) -> &E::Props {
        let expr = self.get_memo_expr();
        expr.state().props()
    }

    /// Returns an iterator over child expressions of this memo expression.
    pub fn children(&self) -> MemoExprInputsIter<E> {
        let expr = self.get_memo_expr();
        MemoExprInputsIter {
            expr,
            position: 0,
            num_children: expr.num_children(),
        }
    }

    fn get_memo_expr(&self) -> &E {
        self.expr_ref.get()
    }
}

impl<E> Clone for MemoExprRef<E>
where
    E: MemoExpr,
{
    fn clone(&self) -> Self {
        MemoExprRef::new(self.id, self.expr_ref.clone())
    }
}

impl<E> PartialEq for MemoExprRef<E>
where
    E: MemoExpr,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.expr_ref == other.expr_ref
    }
}

impl<E> Eq for MemoExprRef<E> where E: MemoExpr {}

impl<E> Debug for MemoExprRef<E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoExprRef").field("id", &self.id).finish()
    }
}

/// Iterator over child expressions of a memo expression.
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
                MemoExprState::Memo(state) => Some(state.expr.memo_expr().clone()),
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

#[derive(Clone)]
enum ExprPtr<E>
where
    E: MemoExpr,
{
    // Used for new expressions and for expressions is stored in a memo (see Memo::add_expr).
    // In later case such ExprPtr should not be accessible to via public API.
    Owned(Arc<E::Expr>),
    Memo(MemoExprRef<E>),
}

impl<E> ExprPtr<E>
where
    E: MemoExpr,
{
    fn expr(&self) -> &E::Expr {
        match self {
            ExprPtr::Owned(expr) => expr.as_ref(),
            ExprPtr::Memo(expr) => expr.expr(),
        }
    }

    fn is_memo(&self) -> bool {
        match self {
            ExprPtr::Owned(_) => false,
            ExprPtr::Memo(_) => true,
        }
    }

    fn memo_expr(&self) -> &MemoExprRef<E> {
        match self {
            ExprPtr::Owned(_) => unreachable!("This should only be called on memoized expressions"),
            ExprPtr::Memo(expr) => expr,
        }
    }

    fn expr_id(&self) -> ExprId {
        match self {
            ExprPtr::Owned(_) => unreachable!("This should only be called on memoized expressions"),
            ExprPtr::Memo(expr) => expr.id,
        }
    }
}

#[derive(Clone)]
enum ExprGroupPtr<E>
where
    E: MemoExpr,
{
    Owned(Arc<E::Props>),
    Memo((GroupId, ImmutableRef<E::Props>)),
}

impl<E> ExprGroupPtr<E>
where
    E: MemoExpr,
{
    pub fn props(&self) -> &E::Props {
        match self {
            ExprGroupPtr::Owned(props) => props.as_ref(),
            ExprGroupPtr::Memo((_, props)) => props.get(),
        }
    }

    fn memo_group_id(&self) -> GroupId {
        match self {
            ExprGroupPtr::Owned(_) => panic!("This should only be called on memoized expressions"),
            ExprGroupPtr::Memo((group, _)) => *group,
        }
    }

    fn from_memo_group(group_id: GroupId, props: ImmutableRef<E::Props>) -> Self {
        ExprGroupPtr::Memo((group_id, props))
    }
}

struct MemoGroupData<E>
where
    E: MemoExpr,
{
    group_id: GroupId,
    exprs: Vec<MemoExprRef<E>>,
    props_ref: ImmutableRef<E::Props>,
}

impl<E> MemoGroupData<E>
where
    E: MemoExpr,
{
    fn new(group_id: GroupId, props_ref: ImmutableRef<E::Props>) -> Self {
        MemoGroupData {
            group_id,
            exprs: vec![],
            props_ref,
        }
    }
}

impl<E, P> Debug for MemoGroupData<E>
where
    E: MemoExpr<Props = P> + Debug,
    P: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoGroupData")
            .field("group_id", &self.group_id)
            .field("props", self.props_ref.get())
            .field("exprs", &self.exprs)
            .finish()
    }
}

pub(crate) fn copy_in_expr_impl<E, T>(copy_in_exprs: &mut CopyInExprs<E, T>, expr: &E, expr_ctx: ExprContext<E>)
where
    E: MemoExpr,
{
    let expr_id = ExprId(copy_in_exprs.memo.exprs.next_id());

    let ExprContext { children, parent } = expr_ctx;
    let props = create_group_properties(expr, &children);

    let expr = E::expr_with_new_children(expr.expr(), NewChildExprs::new(children));
    let digest = make_digest::<E>(&expr, &props);

    let (expr_id, added) = match copy_in_exprs.memo.expr_cache.entry(digest) {
        Entry::Occupied(o) => {
            let expr_id = o.get();
            (*expr_id, false)
        }
        Entry::Vacant(v) => {
            v.insert(expr_id);
            (expr_id, true)
        }
    };

    let (group_ref, expr_ref) = if added {
        if let Some(parent) = parent {
            let (group_ref, expr_ref) = copy_in_exprs.memo.add_expr(expr, parent);

            (group_ref, expr_ref)
        } else {
            let group_id = GroupId(copy_in_exprs.memo.groups.next_id());
            let props = if let Some(callback) = copy_in_exprs.memo.callback.as_ref() {
                callback.new_group(&expr, &props, &copy_in_exprs.memo.metadata)
            } else {
                props
            };
            let (_, props_ref) = copy_in_exprs.memo.props.insert(props);
            let memo_group = MemoGroupData::new(group_id, props_ref);
            let token = copy_in_exprs.memo.add_group(memo_group);
            // Token should be passed to add_expr
            let (group_ref, expr_ref) = copy_in_exprs.memo.add_expr(expr, token);

            (group_ref, expr_ref)
        }
    } else {
        let group_id = copy_in_exprs.memo.expr_to_group.get(&expr_id).expect("Unexpected expression");
        let group_ref = copy_in_exprs.memo.get_group_ref(group_id);
        let expr_ref = copy_in_exprs.memo.get_expr_ref(&expr_id);

        (group_ref, expr_ref)
    };
    // SAFETY: At this point no reference to the data behind group_ref exists and we can safely
    // dereference this group_ref.
    let expr = unsafe {
        let group_id = group_ref.id;
        let props_ref = group_ref.inner.get().props_ref.clone();
        let state = MemoExprState::Memo(MemoizedExpr {
            // Both ExprPtr and ExprGroupPtr should be in the memo state
            // Because memo.exprs stores ExprPtr in the Owned state.
            expr: ExprPtr::Memo(expr_ref),
            group: ExprGroupPtr::Memo((group_id, props_ref)),
        });
        E::from_state(state)
    };
    // Reference to the group data no longer exists.
    copy_in_exprs.result = Some(expr);
}

struct InternalGroupRef<E>
where
    E: MemoExpr,
{
    id: GroupId,
    inner: StoreElementRef<MemoGroupData<E>>,
}

impl<E> InternalGroupRef<E>
where
    E: MemoExpr,
{
    fn new(id: GroupId, inner: StoreElementRef<MemoGroupData<E>>) -> Self {
        InternalGroupRef { id, inner }
    }
}

impl<E> Display for MemoExprRef<E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} ", self.id)?;
        let mut fmt = DisplayMemoExprFormatter { fmt: f };
        E::format_expr(self.expr(), self.props(), &mut fmt);
        write!(f, "]")?;
        Ok(())
    }
}

/// A wrapper around [store::Store] that stores [MemoGroupData] and provides safe iterators over its elements.
struct GroupDataStore<E>
where
    E: MemoExpr,
{
    store: Store<MemoGroupData<E>>,
}

impl<E> GroupDataStore<E>
where
    E: MemoExpr,
{
    fn new() -> Self {
        GroupDataStore {
            store: Store::new(PAGE_SIZE),
        }
    }

    fn insert(&mut self, data: MemoGroupData<E>) -> (StoreElementId, StoreElementRef<MemoGroupData<E>>) {
        self.store.insert(data)
    }

    fn get(&self, elem_id: StoreElementId) -> Option<StoreElementRef<MemoGroupData<E>>> {
        self.store.get(elem_id)
    }

    unsafe fn get_mut(&mut self, elem_id: StoreElementId) -> Option<&mut MemoGroupData<E>> {
        self.store.get_mut(elem_id)
    }

    fn next_id(&self) -> StoreElementId {
        self.store.next_id()
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn iter(&self) -> GroupDataIter<E> {
        GroupDataIter {
            iter: self.store.iter(),
        }
    }
}

struct GroupDataIter<'a, E>
where
    E: MemoExpr,
{
    iter: StoreElementIter<'a, MemoGroupData<E>>,
}

impl<'a, E> Iterator for GroupDataIter<'a, E>
where
    E: MemoExpr,
{
    type Item = GroupDataRef<'a, MemoGroupData<E>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| GroupDataRef {
            // SAFETY: This is safe because there is no mutable reference to the underlying group data
            // because this iterator is used when a caller has an owned reference to a memo.
            inner: unsafe { &*e.get_ptr() },
        })
    }
}

impl<'a, E> DoubleEndedIterator for GroupDataIter<'a, E>
where
    E: MemoExpr,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|e| GroupDataRef {
            // SAFETY: This is safe because there is no mutable reference to the underlying group data.
            inner: unsafe { &*e.get_ptr() },
        })
    }
}

struct GroupDataRef<'a, E> {
    inner: &'a E,
}

impl<E> Deref for GroupDataRef<'_, E> {
    type Target = E;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

/// Builds a textual representation of the given memo.
pub(crate) fn format_memo_impl<E, T>(memo: &MemoImpl<E, T>) -> String
where
    E: MemoExpr,
{
    let mut buf = String::new();
    let mut f = StringMemoFormatter::new(&mut buf);

    for group in memo.groups.iter().rev() {
        f.push_str(format!("{} ", group.group_id).as_str());
        // SAFETY: No mutable references to the underlying group data exists.
        for (i, expr) in group.exprs.iter().enumerate() {
            if i > 0 {
                // newline + 3 spaces
                f.push_str("\n   ");
            }
            E::format_expr(expr.expr(), expr.props(), &mut f);
        }
        f.push('\n');
    }

    buf
}