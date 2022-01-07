mod store;

use crate::memo::store::{AppendOnlyStore, ImmutableRef, StoreElementIter};
use crate::memo::store::{Store, StoreElementId, StoreElementRef};
use itertools::Itertools;
use std::collections::VecDeque;
use std::fmt::{Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
};

/// `Memo` is the primary data structure used by the cost-based optimizer:
///  * It stores each expression as a group of logically equivalent expressions.
///  * It provides memoization of identical subexpressions within an expression tree.
///
/// # Safety
///
/// The optimizer guarantees that references to both memo groups and memo expressions never outlive the memo they are referencing.
// TODO: Examples
// TODO: add generic implementation of a copy-out method.
// TODO: Document ExprPtr, ExprGroupPtr - memo expression parts.
//  MemoExprRef, MemoGroupRef, MemoExprGroupRef
pub struct Memo<E, T>
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

/// The number of elements per store page.
#[doc(hidden)]
const PAGE_SIZE: usize = 8;

impl<E, T> Memo<E, T>
where
    E: MemoExpr,
{
    /// Creates a new memo and the given metadata.
    pub fn new(metadata: T) -> Self {
        Memo {
            groups: GroupDataStore::new(),
            exprs: AppendOnlyStore::new(),
            props: AppendOnlyStore::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            metadata,
            callback: None,
        }
    }

    /// Creates a new memo with the given metadata and the callback.
    pub fn with_callback(
        metadata: T,
        callback: Rc<dyn MemoGroupCallback<Expr = E::Expr, Props = E::Props, Metadata = T>>,
    ) -> Self {
        Memo {
            groups: GroupDataStore::new(),
            exprs: AppendOnlyStore::new(),
            props: AppendOnlyStore::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            metadata,
            callback: Some(callback),
        }
    }

    /// Copies the given expression `expr` into this memo. if this memo does not contain the given expression
    /// a new memo group is created and this method returns a reference to it. Otherwise returns a reference
    /// to the already existing expression.
    pub fn insert_group(&mut self, expr: E) -> E {
        let copy_in = CopyIn {
            memo: self,
            parent: None,
            depth: 0,
        };
        copy_in.execute(&expr)
    }

    /// Copies the expression `expr` into this memo and adds it to the memo group that granted the given [membership token](MemoGroupToken).
    /// If an identical expression already exist this method simply returns a reference to that expression.
    ///
    /// # Panics
    ///
    /// This method panics if group with the given id does not exist.
    pub fn insert_group_member(&mut self, token: MemoGroupToken<E>, expr: E) -> E {
        let copy_in = CopyIn {
            memo: self,
            parent: Some(token),
            depth: 0,
        };
        copy_in.execute(&expr)
    }

    /// Return a reference to a memo group with the given id or panics if it does not exists.
    ///
    /// # Panics
    ///
    /// This method panics if group with the given id does not exist.
    pub fn get_group(&self, group_id: &GroupId) -> MemoExprGroupRef<E> {
        let group_ref = self.get_group_ref(group_id);
        MemoExprGroupRef {
            id: *group_id,
            data_ref: group_ref.inner,
            marker: Default::default(),
        }
    }

    /// Returns a reference to the metadata.
    pub fn metadata(&self) -> &T {
        &self.metadata
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
        let group_ref = MemoGroupRef::new(GroupId(id), elem_ref);

        MemoGroupToken(group_ref)
    }

    fn add_expr(&mut self, expr: E::Expr, membership_token: MemoGroupToken<E>) -> (MemoGroupRef<E>, MemoExprRef<E>) {
        // SAFETY:
        // An expression can only be added to a group if a caller owns a membership token.
        // That token can be obtained in two ways:
        //
        // 1) when a new group is added to a memo (internal API) - add_group returns a membership token
        // which is then passed to this method - no references to MemoGroupData exists
        // and we can safely obtain a mutable reference to via groups.get_mut
        //
        // 2) by consuming MemoExprGroupRef (public API) - in this case borrow checker will reject a program
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
        let expr = E::from_parts(ExprPtr::Owned(Box::new(expr)), ExprGroupPtr::from_memo_group(group_id, props_ref));

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

    // WARNING: &mut is required here see the "SAFETY" comment bellow.
    fn get_first_memo_expr(&mut self, group_id: &GroupId) -> E {
        let group_ref = self.get_group_ref(group_id);
        // SAFETY:
        // At this point no mutable reference to the group data exists
        // because this method is called by an owner of the mutable reference to a memo.
        let expr_ref = unsafe {
            let group_data = group_ref.inner.get();
            group_data.exprs[0].clone()
        };

        E::from_memo_refs(group_ref, expr_ref, internal::Private)
        // No other reference to the group data exists.
    }

    fn get_group_ref(&self, group_id: &GroupId) -> MemoGroupRef<E> {
        let group_ref = self
            .groups
            .get(group_id.index())
            .unwrap_or_else(|| panic!("group id is invalid: {}", group_id));

        MemoGroupRef::new(*group_id, group_ref)
    }
}

/// A borrowed memo group. A memo group is a group of logically equivalent expressions.
/// Expressions are logically equivalent when they produce the same result.
///
/// In contrast to [MemoGroupRef] this reference provides methods for iteration over memo expressions.
pub struct MemoExprGroupRef<'a, E>
where
    E: MemoExpr,
{
    id: GroupId,
    data_ref: StoreElementRef<MemoGroupData<E>>,
    marker: std::marker::PhantomData<&'a E>,
}

impl<'a, E> MemoExprGroupRef<'a, E>
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
        MemoGroupToken(MemoGroupRef::new(self.id, self.data_ref))
    }

    /// Convert this borrowed memo group to the first memo expression in that group.
    pub fn to_memo_expr(self) -> E {
        let data = self.get_memo_group();
        let group_id = self.id;

        let expr_ref = data.exprs[0].clone();
        let props_ref = data.props_ref.clone();

        E::from_parts(ExprPtr::Memo(expr_ref), ExprGroupPtr::from_memo_group(group_id, props_ref))
    }

    fn get_memo_group(&self) -> &MemoGroupData<E> {
        // SAFETY: Making a reference to the group data should be safe because MemoExprGroupRef
        // was immutably borrowed from a memo.
        unsafe { self.data_ref.get() }
    }
}

impl<'a, E> Debug for MemoExprGroupRef<'a, E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoExprGroupRef").field("id", &self.id).finish()
    }
}

impl<M, T, E, P> Debug for Memo<M, T>
where
    M: MemoExpr<Expr = E, Props = P> + Debug,
    T: Debug,
    E: Debug,
    P: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memo")
            .field("groups", &self.groups.len())
            .field("exprs", &self.exprs.len())
            .field("expr_to_group", &self.expr_to_group)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// A group membership token that allows the owner to add an expression into a memo group this token was obtained from.
///
/// A Membership token can be obtained from a memo group by calling [to_membership_token](MemoExprGroupRef::to_membership_token)
/// of that memo group.
pub struct MemoGroupToken<E>(MemoGroupRef<E>)
where
    E: MemoExpr;

impl<E> MemoGroupToken<E>
where
    E: MemoExpr,
{
    fn into_inner(self) -> MemoGroupRef<E> {
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

/// A trait that must be implemented by an expression that can be copied into a [`memo`](self::Memo).
// TODO: Docs.
// FIXME: Return empy_props see comment in ExprGroupRef.
// FIXME: Require debug from MemoExpr, Expr and Props ?
pub trait MemoExpr: Clone {
    /// The type of the expression.
    type Expr: Expr;
    /// The type of the properties.
    type Props: Props;

    /// Returns the expression this memo expression stores.
    /// This method is a shorthand for `memo_expr.expr_ptr().expr()`.
    fn expr(&self) -> &Self::Expr {
        self.expr_ptr().expr()
    }

    /// Returns properties associated with this expression.
    /// This method is a shorthand for `memo_expr.group_ptr().props()`.
    fn props(&self) -> &Self::Props {
        self.group_ptr().props()
    }

    /// Creates a new memo expression from the given expression and group.
    fn from_parts(expr: ExprPtr<Self>, group: ExprGroupPtr<Self>) -> Self;

    /// Returns a pointer to the underlying expression.
    fn expr_ptr(&self) -> &ExprPtr<Self>;

    /// Returns a pointer to a group this memo expression belongs to.
    fn group_ptr(&self) -> &ExprGroupPtr<Self>;

    /// Recursively traverses this expression and copies it into a memo.
    fn copy_in<T>(&self, visitor: &mut CopyInExprs<Self, T>);

    /// Creates a new expression from the given expression by replacing its child expressions
    /// provided by the given [NewChildExprs](self::NewChildExprs).
    fn expr_with_new_children(expr: &Self::Expr, inputs: NewChildExprs<Self>) -> Self::Expr;

    /// Called when a scalar expression with the given nested sub-queries should be added to a memo.
    /// Returns properties that contain the given child expressions.
    fn new_properties_with_nested_sub_queries(
        props: Self::Props,
        sub_queries: impl Iterator<Item = Self>,
    ) -> Self::Props;

    /// Returns the number of child expressions of this memo expression.
    fn num_children(&self) -> usize;

    /// Returns the i-th child expression of this memo expression.
    fn get_child(&self, i: usize) -> Option<&Self>;

    /// Builds a textual representation of the given expression.
    fn format_expr<F>(expr: &Self::Expr, props: &Self::Props, f: &mut F)
    where
        F: MemoExprFormatter;

    /// Internal API.
    ///
    /// Creates a memo expression from the given group and expression.
    #[doc(hidden)]
    fn from_memo_refs(group: MemoGroupRef<Self>, expr: MemoExprRef<Self>, _private: internal::Private) -> Self {
        let group_id = group.id;
        let group_ref = group.inner;
        // SAFETY no other reference to the underlying group data exists so we can safely access the group data.
        let props_ref = unsafe { group_ref.get().props_ref.clone() };

        Self::from_parts(ExprPtr::Memo(expr), ExprGroupPtr::from_memo_group(group_id, props_ref))
    }
}

/// A trait for the expression type.
pub trait Expr: Clone {
    /// The type of relational expression.
    type RelExpr;
    /// The type of scalar expression.
    type ScalarExpr;

    /// Creates a relational expression from the given expression.
    fn new_rel(expr: Self::RelExpr) -> Self;

    /// Creates a scalar expression from the given expression.
    fn new_scalar(expr: Self::ScalarExpr) -> Self;

    /// Returns a reference to the underlying relational expression.
    ///
    /// # Panics
    ///
    /// This method panics if the underlying expression is not a relational expression.
    fn relational(&self) -> &Self::RelExpr;

    /// Returns a reference to the underlying scalar expression.
    ///
    /// # Panics
    ///
    /// This method panics if the underlying expression is not a scalar expression.
    fn scalar(&self) -> &Self::ScalarExpr;

    /// Returns `true` if this is a scalar expression.
    fn is_scalar(&self) -> bool;
}

/// A trait for properties of the expression.
// FIXME: Rename to Props.
pub trait Props: Clone {
    /// The type of relational properties.
    type RelProps;
    /// The type of scalar properties.
    type ScalarProps;

    /// Creates relational properties.
    fn new_rel(props: Self::RelProps) -> Self;

    /// Creates scalar properties.
    fn new_scalar(props: Self::ScalarProps) -> Self;

    /// Returns a reference to the underlying relational properties.
    ///
    /// # Panics
    ///
    /// This method panics if the underlying properties are not relational.
    fn relational(&self) -> &Self::RelProps;

    /// Return a reference to the scalar properties.
    ///
    /// # Panics
    ///
    /// This method panics if the underlying properties are not scalar.
    fn scalar(&self) -> &Self::ScalarProps;
}

/// `ExprPtr` is a pointer to an expression.
///
/// `ExprPtr` reference has two states `owned` and `memo`:
/// * In the `Owned` state this reference owns an expression and that expression is stored in the heap.
/// * In the `Memo` is a reference to a [memo expression](self::MemoExprRef).
#[derive(Clone)]
pub enum ExprPtr<E>
where
    E: MemoExpr,
{
    /// `ExprPtr` owns the expression.
    Owned(Box<E::Expr>),
    /// `ExprPtr` holds reference to a [memo expression](self::MemoExprRef).
    Memo(MemoExprRef<E>),
}

impl<E> ExprPtr<E>
where
    E: MemoExpr,
{
    /// Creates an instance of `ExprPtr` that owns the given expression.
    pub fn new(expr: E::Expr) -> Self {
        ExprPtr::Owned(Box::new(expr))
    }

    /// Returns a reference to an expression.
    /// * In the owned state this method returns a reference to the underlying expression.
    /// * In the memo state this method returns a reference to the first expression in the [memo group](self::MemoGroupRef).
    pub fn expr(&self) -> &E::Expr {
        match self {
            ExprPtr::Owned(expr) => expr.as_ref(),
            ExprPtr::Memo(expr) => expr.expr(),
        }
    }

    /// Returns `true` if this `ExprPtr` is in the `Memo` state.
    pub fn is_memo(&self) -> bool {
        match self {
            ExprPtr::Owned(_) => false,
            ExprPtr::Memo(_) => true,
        }
    }

    /// Returns a reference to a memo expression or panics if this `ExpGroupRef` is not in the memo state.
    /// This method should only be called on memoized expressions.
    ///
    /// # Panics
    ///
    /// This method panics if this `ExpGroupRef` owns the properties instead of holding a [MemoExprRef](self::MemoExprRef).
    pub fn memo_expr(&self) -> &MemoExprRef<E> {
        match self {
            ExprPtr::Owned(_) => panic!("This should only be called on memoized expressions"),
            ExprPtr::Memo(expr) => expr,
        }
    }

    fn equal(&self, other: &ExprPtr<E>) -> bool {
        let this = self.get_eq_state();
        let that = other.get_eq_state();
        // Expressions are equal if their expr pointers point to the same expression.
        match (this, that) {
            // Raw pointer equality for owned expressions is used here because we should not check non-memoized
            // expressions for equality.
            ((Some(x), _), (Some(y), _)) => std::ptr::eq(x, y),
            ((_, Some(x)), (_, Some(y))) => x == y,
            _ => false,
        }
    }

    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            ExprPtr::Owned(expr) => {
                let ptr: *const E::Expr = &**expr;
                ptr.hash(state)
            }
            ExprPtr::Memo(expr) => expr.id().hash(state),
        }
    }

    fn get_eq_state(&self) -> (Option<*const E::Expr>, Option<ExprId>)
    where
        E: MemoExpr,
    {
        match self {
            ExprPtr::Owned(expr) => {
                let ptr: *const E::Expr = &**expr;
                (Some(ptr), None)
            }
            ExprPtr::Memo(expr) => (None, Some(expr.id())),
        }
    }
}

impl<E, T> Debug for ExprPtr<E>
where
    E: MemoExpr<Expr = T>,
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprPtr::Owned(expr) => f.debug_tuple("Owned").field(&*expr).finish(),
            ExprPtr::Memo(expr) => f.debug_tuple("Memo").field(expr).finish(),
        }
    }
}

/// `ExprGroupPtr` is a pointer to a memo group an expression belongs to. It has two states `Owned` and `Memo`:
///
/// * In the `Owned` state an expression has not been copied into a memo and an instance of `ExprGroupPtr`
/// owns properties and stores a pointer to a heap allocation.
/// * In the `Memo` state it stores a reference to a [memo group](self::MemoGroupRef).
pub struct ExprGroupPtr<E>
where
    E: MemoExpr,
{
    inner: ExprGroupPtrState<E>,
}

impl<E> ExprGroupPtr<E>
where
    E: MemoExpr,
{
    /// Creates an instance of `ExprGroupPtr` that owns the given properties.
    pub fn new(props: E::Props) -> Self {
        ExprGroupPtr {
            inner: ExprGroupPtrState::Owned(Box::new(props)),
        }
    }

    /// Returns a reference to properties of a memo group behind this pointer.
    ///
    /// * In the owned state this method returns a reference to properties owned by this pointer.
    /// * In the memo state this method returns a reference of a memo group the expression belongs to.
    pub fn props(&self) -> &E::Props {
        match self.inner() {
            ExprGroupPtrState::Owned(props) => props.as_ref(),
            ExprGroupPtrState::Memo((_, props)) => props.get(),
        }
    }

    /// Returns `true` if this `ExprGroupPtr` is in the `Memo` state.
    pub fn is_memo(&self) -> bool {
        match self.inner() {
            ExprGroupPtrState::Owned(_) => false,
            ExprGroupPtrState::Memo(_) => true,
        }
    }

    /// Returns an identifier of a memo group or panics if this `ExprGroupPtr` is not in the `Memo` state.
    /// This method should only be called on memoized expressions.
    ///
    /// # Panics
    ///
    /// This method panics if this `ExprGroupPtr` owns the properties instead of holding a [MemoGroupRef](self::MemoGroupRef).
    pub fn memo_group_id(&self) -> GroupId {
        match self.inner() {
            ExprGroupPtrState::Owned(_) => panic!("This should only be called on memoized expressions"),
            ExprGroupPtrState::Memo((group, _)) => *group,
        }
    }

    fn from_memo_group(group_id: GroupId, props: ImmutableRef<E::Props>) -> Self {
        ExprGroupPtr {
            inner: ExprGroupPtrState::Memo((group_id, props)),
        }
    }

    fn inner(&self) -> &ExprGroupPtrState<E> {
        &self.inner
    }
}

impl<E, P> Debug for ExprGroupPtr<E>
where
    E: MemoExpr<Props = P>,
    P: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            ExprGroupPtrState::Owned(props) => f.debug_tuple("Owned").field(&*props).finish(),
            ExprGroupPtrState::Memo((group_id, _)) => f.debug_tuple("Memo").field(group_id).finish(),
        }
    }
}

impl<E> Clone for ExprGroupPtr<E>
where
    E: MemoExpr,
{
    fn clone(&self) -> Self {
        let inner = match &self.inner {
            ExprGroupPtrState::Owned(props) => ExprGroupPtrState::Owned(props.clone()),
            ExprGroupPtrState::Memo(group) => ExprGroupPtrState::Memo(group.clone()),
        };
        ExprGroupPtr { inner }
    }
}

enum ExprGroupPtrState<E>
where
    E: MemoExpr,
{
    /// Owned state.
    //TODO: Since transformation rules do not use properties => they should not pay the cost of a heap allocation of an
    // empty properties object.
    // Copying an alternative expression into a memo also requires MemoExpr which consists of an expression and properties
    // -> this means that it also requires a heap allocation.
    // Add MemoExpr::empty_props(?) and Owned state (Box<PropsRef> | Empty(const T)).
    Owned(Box<E::Props>),
    /// `ExprGroupPtr` holds a reference to a [memo group](self::MemoGroupRef) an expression belongs to.
    //TODO: Store instead (GroupId, StoreRef<Props>) because MemoGroupRef could alias
    Memo((GroupId, ImmutableRef<E::Props>)),
}

/// Callback that is called when a new memo group is added to a memo.
//FIXME: rename to MemoGroupCallback
pub trait MemoGroupCallback {
    /// The type of expression.
    type Expr: Expr;
    /// The type of properties of the expression.
    type Props: Props;
    /// The type of metadata stored by the memo.
    type Metadata;

    /// Called when a new memo group is added to memo and returns properties to be shared among all expressions in this group.
    /// Where `expr` is the expression that created the memo group and `props` are properties associated with that expression.
    // If no properties has been provided contains empty properties (see [`MemoExpr::empty_props`](self::MemoExpr::empty_props) ).
    //FIXME: should accept a context to pass extra stuff.
    fn new_group(&self, expr: &Self::Expr, props: &Self::Props, metadata: &Self::Metadata) -> Self::Props;
}

/// A relational node of an expression tree.
pub struct RelNode<E>(E);

impl<E, T, RelExpr, P, RelProps> RelNode<E>
where
    E: MemoExpr<Expr = T, Props = P>,
    T: Expr<RelExpr = RelExpr>,
    P: Props<RelProps = RelProps>,
{
    /// Creates a relational node of an expression tree from the given relational expression and properties.
    pub fn new(expr: RelExpr, props: RelProps) -> Self {
        let expr = ExprPtr::new(T::new_rel(expr));
        let group = ExprGroupPtr::new(P::new_rel(props));
        let expr = E::from_parts(expr, group);
        RelNode(expr)
    }

    /// Creates a relational node of an expression tree from the given memo expression.
    ///
    /// # Note
    ///
    /// Caller must guarantee that the given expression is a relational expression.
    pub fn from_mexpr(expr: E) -> Self {
        RelNode(expr)
    }

    /// Creates a relational node of an expression tree from the given memo group.
    ///
    /// # Note
    ///
    /// Caller must guarantee that the given group is a group of relational expressions.
    pub fn from_group(group: MemoExprGroupRef<'_, E>) -> Self {
        let expr = group.to_memo_expr();
        RelNode(expr)
    }

    /// Returns a reference to the underlying expression.
    ///
    /// # Panics
    ///
    /// This method panics if this node does not hold a relational expression.
    pub fn expr<'e>(&'e self) -> &'e T::RelExpr
    where
        T: 'e,
    {
        let expr = self.0.expr_ptr();
        let expr = expr.expr();
        expr.relational()
    }

    /// Returns a reference to properties associated with this node:
    /// * if this node is an expression returns a reference to the properties of the underlying expression.
    /// * If this node is a memo group returns a reference to the properties of the first expression of this memo group.
    ///
    /// # Panics
    ///
    /// This method panics if this node does not hold a relational expression.
    pub fn props<'e>(&'e self) -> &'e RelProps
    where
        T: 'e,
        P: 'e,
    {
        self.0.props().relational()
    }

    /// Returns an [self::ExprPtr] of the underlying memo expression.
    pub fn expr_ref(&self) -> &ExprPtr<E> {
        self.0.expr_ptr()
    }

    /// Returns a reference to the underlying memo expression.
    pub fn mexpr(&self) -> &E {
        &self.0
    }

    /// Consumes this relational expression and returns the underlying memo expression.
    pub fn into_inner(self) -> E {
        self.0
    }
}

impl<E> AsRef<E> for RelNode<E>
where
    E: MemoExpr,
{
    fn as_ref(&self) -> &E {
        &self.0
    }
}

impl<E> Clone for RelNode<E>
where
    E: Clone,
{
    fn clone(&self) -> Self {
        RelNode(self.0.clone())
    }
}

impl<E> Debug for RelNode<E>
where
    E: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RelNode").field(&self.0).finish()
    }
}

impl<E> PartialEq for RelNode<E>
where
    E: MemoExpr,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.expr_ptr().equal(other.0.expr_ptr())
    }
}

impl<E> Eq for RelNode<E> where E: MemoExpr {}

impl<E> Hash for RelNode<E>
where
    E: MemoExpr,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.expr_ptr().hash(state)
    }
}

/// A scalar node of an expression tree.
pub struct ScalarNode<E>(E);

impl<E, T, ScalarExpr, P, ScalarProps> ScalarNode<E>
where
    E: MemoExpr<Expr = T, Props = P>,
    T: Expr<ScalarExpr = ScalarExpr>,
    P: Props<ScalarProps = ScalarProps>,
{
    /// Creates a scalar node of an expression tree from the given scalar expression and properties.
    pub fn new(expr: ScalarExpr, props: ScalarProps) -> Self {
        let expr = ExprPtr::new(T::new_scalar(expr));
        let group = ExprGroupPtr::new(P::new_scalar(props));
        let expr = E::from_parts(expr, group);
        ScalarNode(expr)
    }

    /// Creates a scalar node of an expression tree from the given memo expression.
    ///
    /// # Note
    ///
    /// Caller must guarantee that the given expression is a scalar expression.
    pub fn from_mexpr(expr: E) -> Self {
        ScalarNode(expr)
    }

    /// Creates a relational node of an expression tree from the given memo group.
    ///
    /// # Note
    ///
    /// Caller must guarantee that the given group is a group of relational expressions.
    pub fn from_group(group: MemoExprGroupRef<'_, E>) -> Self {
        let expr = group.to_memo_expr();
        ScalarNode(expr)
    }

    /// Returns a reference to the underlying scalar expression.
    ///
    /// # Panics
    ///
    /// This method panics if this not does not hold a scalar expression.
    pub fn expr<'e>(&'e self) -> &'e T::ScalarExpr
    where
        T: 'e,
    {
        let expr = self.0.expr_ptr();
        let expr = expr.expr();
        expr.scalar()
    }

    /// Returns a reference to properties associated with this node:
    /// * if this node is an expression returns a reference to the properties of the underlying expression.
    /// * If this node is a memo group returns a reference to the properties of the first expression of this memo group.
    pub fn props<'e>(&'e self) -> &'e ScalarProps
    where
        T: 'e,
        P: 'e,
    {
        self.0.props().scalar()
    }

    /// Returns an [self::ExprPtr] of the underlying memo expression.
    pub fn expr_ref(&self) -> &ExprPtr<E> {
        self.0.expr_ptr()
    }

    /// Returns a reference to the underlying memo expression.
    pub fn mexpr(&self) -> &E {
        &self.0
    }

    /// Consumes this scalar expression and returns the underlying memo expression.
    pub fn into_inner(self) -> E {
        self.0
    }
}

impl<E> AsRef<E> for ScalarNode<E>
where
    E: MemoExpr,
{
    fn as_ref(&self) -> &E {
        &self.0
    }
}

impl<E> Clone for ScalarNode<E>
where
    E: Clone,
{
    fn clone(&self) -> Self {
        ScalarNode(self.0.clone())
    }
}

impl<E> Debug for ScalarNode<E>
where
    E: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ScalarNode").field(&self.0).finish()
    }
}

impl<E> PartialEq for ScalarNode<E>
where
    E: MemoExpr,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.expr_ptr().equal(other.0.expr_ptr())
    }
}

impl<E> Eq for ScalarNode<E> where E: MemoExpr {}

impl<E> Hash for ScalarNode<E>
where
    E: MemoExpr,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.expr_ptr().hash(state)
    }
}

/// `MemoGroupRef` is a reference to a memo group. In contrast to [MemoExprGroupRef],
/// users of public API can only use this reference as a holder of an identifier of some group.
pub struct MemoGroupRef<E>
where
    E: MemoExpr,
{
    id: GroupId,
    inner: StoreElementRef<MemoGroupData<E>>,
}

impl<E> MemoGroupRef<E>
where
    E: MemoExpr,
{
    fn new(id: GroupId, inner: StoreElementRef<MemoGroupData<E>>) -> Self {
        MemoGroupRef { id, inner }
    }

    /// Returns an opaque identifier of this memo group.
    pub fn id(&self) -> GroupId {
        self.id
    }
}

impl<E> PartialEq for MemoGroupRef<E>
where
    E: MemoExpr,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.inner == other.inner
    }
}

impl<E> Eq for MemoGroupRef<E> where E: MemoExpr {}

impl<E> Hash for MemoGroupRef<E>
where
    E: MemoExpr,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id.0 .0);
    }
}

impl<E> Debug for MemoGroupRef<E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoGroupRef").field("id", &self.id).finish()
    }
}

/// An iterator over expressions of a memo group.
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

/// A reference to a [memo expression](self::MemoExpr).
///
/// # Safety
///
/// A reference to a memo expression is valid until a [`memo`](self::Memo) it belongs to is dropped.
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
        expr.group_ptr().memo_group_id()
    }

    /// Returns a reference to the properties of the memo group this expression belongs to.
    pub fn props(&self) -> &E::Props {
        let expr = self.get_memo_expr();
        expr.group_ptr().props()
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
        if self.position < self.num_children {
            let position = self.position;
            self.position += 1;
            self.expr.get_child(position).map(|e| match e.expr_ptr() {
                // MemoExprRef.expr store a memo expression whose child expressions are replaced with references.
                ExprPtr::Owned(_) => unreachable!(),
                ExprPtr::Memo(expr_ref) => expr_ref.clone(),
            })
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
        f.debug_tuple("GroupId").field(&self.0.index()).finish()
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
        f.debug_tuple("ExprId").field(&self.0.index()).finish()
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

/// Provides methods to traverse an expression tree and copy it into a memo.
pub struct CopyInExprs<'a, E, T>
where
    E: MemoExpr,
{
    memo: &'a mut Memo<E, T>,
    parent: Option<MemoGroupToken<E>>,
    depth: usize,
    result: Option<E>,
}

impl<'a, E, T> CopyInExprs<'a, E, T>
where
    E: MemoExpr,
{
    fn new(memo: &'a mut Memo<E, T>, parent: Option<MemoGroupToken<E>>, depth: usize) -> Self {
        CopyInExprs {
            memo,
            parent,
            depth,
            result: None,
        }
    }

    /// Initialises data structures required to traverse child expressions of the given expression `expr`.
    pub fn enter_expr(&mut self, expr: &E) -> ExprContext<E> {
        ExprContext {
            children: VecDeque::new(),
            parent: self.parent.take(),
            // TODO: Add method that splits MemoExpr into Self::Expr and Self::Props
            //  this will allow to remove the clone() call below .
            props: expr.props().clone(),
        }
    }

    /// Visits the given child expression and recursively copies that expression into the memo:
    /// * If the given expression is an expression this methods recursively copies it into the memo.
    /// * If the child expression is a group this method returns a reference to that group.
    pub fn visit_expr_node(&mut self, expr_ctx: &mut ExprContext<E>, expr_node: impl AsRef<E>) {
        let input = expr_node.as_ref();
        let child_expr = match input.group_ptr().inner() {
            ExprGroupPtrState::Owned(_) => {
                let copy_in = CopyIn {
                    memo: self.memo,
                    parent: None,
                    depth: self.depth + 1,
                };
                copy_in.execute(input)
            }
            ExprGroupPtrState::Memo((group_id, _)) => self.memo.get_first_memo_expr(group_id),
        };
        expr_ctx.children.push_back(child_expr);
    }

    /// Visits the given optional child expression if it is present and recursively copies it into a memo.
    /// This method is equivalent to:
    /// ```text
    /// if let Some(expr_node) = expr_node {
    ///   visitor.visit_expr_node(expr_ctx, expr_node);
    /// }
    /// ```
    pub fn visit_opt_expr_node<'e>(&mut self, expr_ctx: &mut ExprContext<E>, expr_node: Option<impl AsRef<E>>) {
        if let Some(expr_node) = expr_node {
            self.visit_expr_node(expr_ctx, expr_node);
        };
    }

    /// Copies the expression into the memo.
    pub fn copy_in(&mut self, expr: &E, expr_ctx: ExprContext<E>) {
        let expr_id = ExprId(self.memo.exprs.next_id());

        let ExprContext {
            children,
            props,
            parent,
        } = expr_ctx;

        let props = if expr.expr().is_scalar() && !children.is_empty() {
            // Collect nested relational expressions from a scalar expression and store them in group properties.
            E::new_properties_with_nested_sub_queries(
                props,
                children.iter().map(|e| match e.expr_ptr() {
                    ExprPtr::Owned(_) => {
                        unreachable!("ExprContext.children must contain only references to memo groups")
                    }
                    ExprPtr::Memo(_) => e.clone(),
                }),
            )
        } else {
            props
        };

        let expr = E::expr_with_new_children(expr.expr(), NewChildExprs::new(children));
        let digest = make_digest::<E>(&expr, &props);

        let (expr_id, added) = match self.memo.expr_cache.entry(digest) {
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
                let (group_ref, expr_ref) = self.memo.add_expr(expr, parent);

                (group_ref, expr_ref)
            } else {
                let group_id = GroupId(self.memo.groups.next_id());
                let props = if let Some(callback) = self.memo.callback.as_ref() {
                    callback.new_group(&expr, &props, &self.memo.metadata)
                } else {
                    props
                };
                let (_, props_ref) = self.memo.props.insert(props);
                let memo_group = MemoGroupData::new(group_id, props_ref);
                let token = self.memo.add_group(memo_group);
                // Token should be passed to add_expr
                let (group_ref, expr_ref) = self.memo.add_expr(expr, token);

                (group_ref, expr_ref)
            }
        } else {
            let group_id = self.memo.expr_to_group.get(&expr_id).expect("Unexpected expression");
            let group_ref = self.memo.get_group_ref(group_id);
            let expr_ref = self.memo.get_expr_ref(&expr_id);

            (group_ref, expr_ref)
        };
        // At this point no reference to the data behind group_ref exists and we can safely
        // dereference this group_ref.
        let expr = E::from_memo_refs(group_ref, expr_ref, internal::Private);
        // Reference to the group data no longer exists.
        self.result = Some(expr);
    }
}

/// Stores information that is used to build a new memo expression.
pub struct ExprContext<E>
where
    E: MemoExpr,
{
    children: VecDeque<E>,
    parent: Option<MemoGroupToken<E>>,
    props: E::Props,
}

/// A stack-like data structure used by [MemoExpr::expr_with_new_children].
/// It stores new child expressions and provides convenient methods of their retrieval.
pub struct NewChildExprs<E>
where
    E: MemoExpr,
{
    children: VecDeque<E>,
    capacity: usize,
    index: usize,
}

impl<E> NewChildExprs<E>
where
    E: MemoExpr,
{
    /// Creates an instance of `NewChildExprs`.
    pub fn new(children: VecDeque<E>) -> Self {
        NewChildExprs {
            capacity: children.len(),
            index: 0,
            children,
        }
    }

    /// Ensures that this container holds exactly `n` expressions.
    ///
    /// # Panics
    ///
    /// This method panics if the number of expressions is not equal to the number of elements in the underling stack.
    pub fn expect_len(&self, n: usize, operator: &str) {
        assert_eq!(self.capacity, n, "{}: Unexpected number of child expressions", operator);
    }

    /// Retrieves the next relational expression.
    ///
    /// # Panics
    ///
    /// This method panics if there is no relational expressions left or the retrieved expression is
    /// not a relational expression.
    pub fn rel_node(&mut self) -> RelNode<E> {
        let expr = self.expr();
        Self::expect_relational(&expr);
        RelNode::from_mexpr(expr)
    }

    /// Retrieves the next `n` relational expressions.
    ///
    /// # Panics
    ///
    /// This method panics if there is not enough expressions left or some of the retrieved expressions are
    /// not relational expressions.
    pub fn rel_nodes(&mut self, n: usize) -> Vec<RelNode<E>> {
        self.exprs(n)
            .into_iter()
            .map(|i| {
                Self::expect_relational(&i);
                RelNode::from_mexpr(i)
            })
            .collect()
    }

    /// Retrieves the next scalar expression.
    ///
    /// # Panics
    ///
    /// This method panics if there is no expressions left or the retrieved expression is
    /// not a scalar expression.
    pub fn scalar_node(&mut self) -> ScalarNode<E> {
        let expr = self.expr();
        Self::expect_scalar(&expr);
        ScalarNode::from_mexpr(expr)
    }

    /// Retrieves the next `n` scalar expressions.
    ///
    /// # Panics
    ///
    /// This method panics if there is not enough expressions left or some of the retrieved expressions are
    /// not scalar expressions.
    pub fn scalar_nodes(&mut self, n: usize) -> Vec<ScalarNode<E>> {
        self.exprs(n)
            .into_iter()
            .map(|i| {
                Self::expect_scalar(&i);
                ScalarNode::from_mexpr(i)
            })
            .collect()
    }

    fn expr(&mut self) -> E {
        self.ensure_available(1);
        self.next_index()
    }

    fn exprs(&mut self, n: usize) -> Vec<E> {
        self.ensure_available(n);
        let mut children = Vec::with_capacity(n);
        for _ in 0..n {
            children.push(self.next_index());
        }
        children
    }

    fn next_index(&mut self) -> E {
        // the assertion in ensure_available guarantees that `children` has enough elements.
        self.children.pop_front().unwrap()
    }

    fn ensure_available(&mut self, n: usize) {
        let next_index = self.index + n;
        assert!(
            next_index <= self.capacity,
            "Unable to retrieve {} expression(s). Next index: {}, current: {}, total exprs: {}",
            n,
            next_index,
            self.index,
            self.children.len()
        );
        self.index += n;
    }

    fn expect_relational(expr: &E) {
        assert!(!expr.expr().is_scalar(), "expected a relational expression");
    }

    fn expect_scalar(expr: &E) {
        assert!(expr.expr().is_scalar(), "expected a scalar expression");
    }
}

impl<E> Debug for NewChildExprs<E>
where
    E: MemoExpr + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewChildExprs")
            .field("capacity", &self.capacity)
            .field("index", &self.index)
            .field("children", &self.children)
            .finish()
    }
}

/// Provides methods to collect nested expressions from an expression and copy them into a memo.
/// Can be used to support nested relational expressions inside a scalar expression.
//TODO: Examples
pub struct CopyInNestedExprs<'a, 'c, E, T>
where
    E: MemoExpr,
{
    ctx: &'c mut CopyInExprs<'a, E, T>,
    expr_ctx: &'c mut ExprContext<E>,
    nested_exprs: Vec<E>,
}

impl<'a, 'c, E, T> CopyInNestedExprs<'a, 'c, E, T>
where
    E: MemoExpr,
{
    /// Creates a new nested expression collector.
    pub fn new(ctx: &'c mut CopyInExprs<'a, E, T>, expr_ctx: &'c mut ExprContext<E>) -> Self {
        CopyInNestedExprs {
            ctx,
            expr_ctx,
            nested_exprs: Vec::new(),
        }
    }

    /// Traverses the given expression `expr` of some arbitrary type.
    /// When traversal completes all collected nested expressions are copies into a memo.
    pub fn execute<F, S>(mut self, expr: &S, f: F)
    where
        F: Fn(&S, &mut Self),
    {
        (f)(expr, &mut self);
        // Visit collected nested expressions so that they will be added to the given expression as child expressions.
        for expr_node in self.nested_exprs {
            let rel_node = RelNode::from_mexpr(expr_node);
            self.ctx.visit_expr_node(self.expr_ctx, rel_node);
        }
    }

    /// Copies the given nested expression into a memo.
    pub fn visit_expr(&mut self, expr: impl AsRef<E>) {
        let expr = expr.as_ref();
        match expr.group_ptr().inner() {
            ExprGroupPtrState::Owned(_) => {
                expr.copy_in(self.ctx);
                let child_expr = std::mem::take(&mut self.ctx.result).expect("Failed to copy in a nested expressions");
                self.add_child_expr(child_expr);
            }
            ExprGroupPtrState::Memo((group_id, _)) => {
                let child_expr = self.ctx.memo.get_first_memo_expr(group_id);
                self.add_child_expr(child_expr)
            }
        }
    }

    fn add_child_expr(&mut self, expr: E) {
        self.nested_exprs.push(expr)
    }
}

#[doc(hidden)]
struct CopyIn<'a, E, T>
where
    E: MemoExpr,
{
    memo: &'a mut Memo<E, T>,
    parent: Option<MemoGroupToken<E>>,
    depth: usize,
}

impl<'a, E, T> CopyIn<'a, E, T>
where
    E: MemoExpr,
{
    fn execute(self, expr: &E) -> E {
        let mut ctx = CopyInExprs::new(self.memo, self.parent, self.depth);
        expr.copy_in(&mut ctx);
        ctx.result.unwrap()
    }
}

/// Builds a textual representation of the given memo.
pub(crate) fn format_memo<E, T>(memo: &Memo<E, T>) -> String
where
    E: MemoExpr,
{
    let mut buf = String::new();
    let mut f = StringMemoFormatter::new(&mut buf);

    for group in memo.groups.iter().rev() {
        // SAFETY: No mutable references to the underlying group data exists.
        // let group = unsafe { group.get() };
        f.push_str(format!("{} ", group.group_id).as_str());
        for (i, expr) in group.exprs.iter().enumerate() {
            if i > 0 {
                // newline + 3 spaces
                f.push_str("\n   ");
            }
            // expr.mexpr().format_expr(&mut f);
            E::format_expr(expr.expr(), expr.props(), &mut f);
        }
        f.push('\n');
    }

    buf
}

/// Provides methods to build a textual representation of an expression.
pub trait MemoExprFormatter {
    /// Writes a name of an expression.
    fn write_name(&mut self, name: &str);

    /// Writes a value of `source` attribute of an expression.
    fn write_source(&mut self, source: &str);

    /// Writes a child expression.
    fn write_expr<T>(&mut self, name: &str, input: impl AsRef<T>)
    where
        T: MemoExpr;

    /// Writes a child expression it is present. This method is equivalent to:
    /// ```text
    /// if let Some(expr) = expr {
    ///   fmt.write_expr(name, expr);
    /// }
    /// ```
    fn write_expr_if_present<T>(&mut self, name: &str, expr: Option<impl AsRef<T>>)
    where
        T: MemoExpr,
    {
        if let Some(expr) = expr {
            self.write_expr(name, expr);
        }
    }

    /// Writes a value of some attribute of an expression.
    fn write_value<D>(&mut self, name: &str, value: D)
    where
        D: Display;

    /// Writes values of some attribute of an expression.
    fn write_values<D>(&mut self, name: &str, values: &[D])
    where
        D: Display;
}

pub(crate) struct StringMemoFormatter<'b> {
    buf: &'b mut String,
}

impl<'b> StringMemoFormatter<'b> {
    pub fn new(buf: &'b mut String) -> Self {
        StringMemoFormatter { buf }
    }

    pub fn push(&mut self, c: char) {
        self.buf.push(c);
    }

    pub fn push_str(&mut self, s: &str) {
        self.buf.push_str(s);
    }
}

impl MemoExprFormatter for StringMemoFormatter<'_> {
    fn write_name(&mut self, name: &str) {
        self.buf.push_str(name);
    }

    fn write_source(&mut self, source: &str) {
        self.buf.push(' ');
        self.buf.push_str(source);
    }

    fn write_expr<T>(&mut self, name: &str, input: impl AsRef<T>)
    where
        T: MemoExpr,
    {
        self.buf.push(' ');
        if !name.is_empty() {
            self.buf.push_str(name);
            self.buf.push('=');
        }
        let s = format_node_ref(input.as_ref());
        self.buf.push_str(s.as_str());
    }

    fn write_value<D>(&mut self, name: &str, value: D)
    where
        D: Display,
    {
        self.buf.push(' ');
        if !name.is_empty() {
            self.buf.push_str(name);
            self.buf.push('=');
        }
        self.buf.push_str(value.to_string().as_str());
    }

    fn write_values<D>(&mut self, name: &str, values: &[D])
    where
        D: Display,
    {
        if values.is_empty() {
            return;
        }
        self.buf.push(' ');
        self.buf.push_str(name);
        self.buf.push_str("=[");
        self.push_str(values.iter().join(", ").as_str());
        self.buf.push(']');
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

struct DisplayMemoExprFormatter<'f, 'a> {
    fmt: &'f mut Formatter<'a>,
}

impl<'f, 'a> MemoExprFormatter for DisplayMemoExprFormatter<'f, 'a> {
    fn write_name(&mut self, name: &str) {
        self.fmt.write_str(name).unwrap();
    }

    fn write_source(&mut self, source: &str) {
        self.fmt.write_char(' ').unwrap();
        self.fmt.write_str(source).unwrap();
    }

    fn write_expr<T>(&mut self, _name: &str, input: impl AsRef<T>)
    where
        T: MemoExpr,
    {
        let s = format_node_ref(input.as_ref());
        self.fmt.write_char(' ').unwrap();
        self.fmt.write_str(s.as_str()).unwrap();
    }

    fn write_value<D>(&mut self, _name: &str, _value: D)
    where
        D: Display,
    {
        // Do not print value for Display trait
    }

    fn write_values<D>(&mut self, _name: &str, _values: &[D])
    where
        D: Display,
    {
        // Do not print values for Display trait
    }
}

fn format_node_ref<T>(input: &T) -> String
where
    T: MemoExpr,
{
    match input.expr_ptr() {
        ExprPtr::Owned(expr) => {
            // This only happens when expression has not been added to a memo.
            let ptr: *const T::Expr = &**expr;
            format!("{:?}", ptr)
        }
        ExprPtr::Memo(group) => format!("{}", group.id()),
    }
}

fn make_digest<E>(expr: &E::Expr, props: &E::Props) -> String
where
    E: MemoExpr,
{
    let mut buf = String::new();
    let mut fmt = StringMemoFormatter::new(&mut buf);
    E::format_expr(expr, props, &mut fmt);
    buf
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
            // SAFETY: This is safe because there is no mutable reference to the underlying group data.
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

mod internal {
    #[doc(hidden)]
    pub struct Private;
}

#[cfg(test)]
mod test {
    use super::*;
    use std::cell::RefCell;

    type RelNode = super::RelNode<TestOperator>;
    type ScalarNode = super::ScalarNode<TestOperator>;

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

        fn is_scalar(&self) -> bool {
            match self {
                TestExpr::Relational(_) => false,
                TestExpr::Scalar(_) => true,
            }
        }
    }

    impl From<TestOperator> for RelNode {
        fn from(expr: TestOperator) -> Self {
            RelNode::from_mexpr(expr)
        }
    }

    impl From<TestScalarExpr> for ScalarNode {
        fn from(expr: TestScalarExpr) -> Self {
            ScalarNode::new(expr, ScalarProps::default())
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
                TestScalarExpr::SubQuery(_) => TestScalarExpr::SubQuery(inputs.rel_node()),
            }
        }

        fn copy_in_nested<D>(&self, collector: &mut CopyInNestedExprs<TestOperator, D>) {
            match self {
                TestScalarExpr::Value(_) => {}
                TestScalarExpr::Gt { lhs, rhs } => {
                    lhs.copy_in_nested(collector);
                    rhs.copy_in_nested(collector);
                }
                TestScalarExpr::SubQuery(expr) => {
                    collector.visit_expr(expr);
                }
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
                TestScalarExpr::SubQuery(query) => match query.expr_ref() {
                    ExprPtr::Owned(expr) => {
                        let ptr: *const TestExpr = expr.as_ref();
                        write!(f, "SubQuery expr_ptr {:?}", ptr)
                    }
                    ExprPtr::Memo(expr) => {
                        write!(f, "SubQuery {}", expr.group_id())
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

        fn visit_rel(&mut self, expr_ctx: &mut ExprContext<TestOperator>, rel: &RelNode) {
            self.ctx.visit_expr_node(expr_ctx, rel);
        }

        fn visit_scalar(&mut self, expr_ctx: &mut ExprContext<TestOperator>, scalar: &ScalarNode) {
            self.ctx.visit_expr_node(expr_ctx, scalar);
        }

        fn copy_in_nested(&mut self, expr_ctx: &mut ExprContext<TestOperator>, expr: &TestOperator) {
            let nested_ctx = CopyInNestedExprs::new(self.ctx, expr_ctx);
            let scalar = expr.expr().scalar();
            nested_ctx.execute(scalar, |expr, collect: &mut CopyInNestedExprs<TestOperator, D>| {
                expr.copy_in_nested(collect);
            })
        }

        fn copy_in(self, expr: &TestOperator, expr_ctx: ExprContext<TestOperator>) {
            self.ctx.copy_in(expr, expr_ctx)
        }
    }

    #[derive(Debug, Clone)]
    struct TestOperator {
        expr: ExprPtr<TestOperator>,
        group: ExprGroupPtr<TestOperator>,
    }

    impl From<TestRelExpr> for TestOperator {
        fn from(expr: TestRelExpr) -> Self {
            TestOperator {
                expr: ExprPtr::new(TestExpr::Relational(expr)),
                group: ExprGroupPtr::new(TestProps::Rel(RelProps::default())),
            }
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
    }

    impl TestOperator {
        fn with_rel_props(self, a: i32) -> Self {
            TestOperator {
                expr: self.expr,
                group: ExprGroupPtr::new(TestProps::Rel(RelProps { a })),
            }
        }
    }

    impl MemoExpr for TestOperator {
        type Expr = TestExpr;
        type Props = TestProps;

        fn from_parts(expr: ExprPtr<Self>, group: ExprGroupPtr<Self>) -> Self {
            TestOperator { expr, group }
        }

        fn expr_ptr(&self) -> &ExprPtr<Self> {
            &self.expr
        }

        fn group_ptr(&self) -> &ExprGroupPtr<Self> {
            &self.group
        }

        fn copy_in<T>(&self, ctx: &mut CopyInExprs<Self, T>) {
            let mut ctx = TraversalWrapper { ctx };
            let mut expr_ctx = ctx.enter_expr(self);
            match self.expr() {
                TestExpr::Relational(expr) => match expr {
                    TestRelExpr::Leaf(_) => {}
                    TestRelExpr::Node { input } => {
                        ctx.visit_rel(&mut expr_ctx, input);
                    }
                    TestRelExpr::Nodes { inputs } => {
                        for input in inputs {
                            ctx.visit_rel(&mut expr_ctx, input);
                        }
                    }
                    TestRelExpr::Filter { input, filter } => {
                        ctx.visit_rel(&mut expr_ctx, input);
                        ctx.visit_scalar(&mut expr_ctx, filter);
                    }
                },
                TestExpr::Scalar(_expr) => {
                    ctx.copy_in_nested(&mut expr_ctx, self);
                }
            }
            ctx.copy_in(self, expr_ctx)
        }

        fn expr_with_new_children(expr: &Self::Expr, mut inputs: NewChildExprs<Self>) -> Self::Expr {
            match expr {
                TestExpr::Relational(expr) => {
                    let expr = match expr {
                        TestRelExpr::Leaf(s) => TestRelExpr::Leaf(s.clone()),
                        TestRelExpr::Node { .. } => TestRelExpr::Node {
                            input: inputs.rel_node(),
                        },
                        TestRelExpr::Nodes { inputs: input_exprs } => TestRelExpr::Nodes {
                            inputs: inputs.rel_nodes(input_exprs.len()),
                        },
                        TestRelExpr::Filter { .. } => TestRelExpr::Filter {
                            input: inputs.rel_node(),
                            filter: inputs.scalar_node(),
                        },
                    };
                    TestExpr::Relational(expr)
                }
                TestExpr::Scalar(expr) => TestExpr::Scalar(expr.with_new_inputs(&mut inputs)),
            }
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
                TestExpr::Scalar(_) => self.group.props().scalar().sub_queries.len(),
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
                    let props = self.group.props().scalar();
                    props.sub_queries.get(i)
                }
            }
        }

        fn format_expr<F>(expr: &Self::Expr, _props: &Self::Props, f: &mut F)
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
                        for input in inputs {
                            f.write_expr("", input);
                        }
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
        }
    }

    #[test]
    fn test_basics() {
        let mut memo = new_memo();
        let expr1 = TestOperator::from(TestRelExpr::Leaf("aaaa")).with_rel_props(100);
        let (group_id, expr) = insert_group(&mut memo, expr1);
        let group = memo.get_group(&group_id);

        assert_eq!(group.props().relational(), &RelProps { a: 100 }, "group properties");
        assert_eq!(expr.group_id(), group.id(), "expr group");

        let expr_by_id = memo.get_expr_ref(&expr.id());
        assert_eq!(expr, expr_by_id);

        assert_eq!(memo.groups.len(), 1, "group num");
        assert_eq!(memo.exprs.len(), 1, "expr num");
        assert_eq!(memo.expr_cache.len(), 1, "expr cache size");

        assert_eq!(
            format!("{:#?}", group.props()),
            format!("{:#?}", expr.props()),
            "groups properties and expr properties must be equal"
        );
    }

    #[test]
    fn test_properties() {
        let mut memo = new_memo();

        let leaf = TestOperator::from(TestRelExpr::Leaf("a")).with_rel_props(10);
        let inner = TestOperator::from(TestRelExpr::Node { input: leaf.into() }).with_rel_props(15);
        let expr = TestOperator::from(TestRelExpr::Node { input: inner.into() });
        let outer = TestOperator::from(TestRelExpr::Node { input: expr.into() }).with_rel_props(20);

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
    fn test_child_expr() {
        let mut memo = new_memo();

        let leaf = TestOperator::from(TestRelExpr::Leaf("a")).with_rel_props(10);
        let expr = TestOperator::from(TestRelExpr::Node {
            input: leaf.clone().into(),
        });
        let (_, expr) = insert_group(&mut memo, expr);

        match expr.expr().relational() {
            TestRelExpr::Node { input } => {
                let children: Vec<_> = expr.children().collect();
                assert_eq!(
                    format!("{:?}", input.expr()),
                    format!("{:?}", children[0].expr().relational()),
                    "child expr"
                );
                assert_eq!(input.props(), leaf.props().relational(), "input props");
            }
            _ => panic!("Unexpected expression: {:?}", expr),
        }
    }

    #[test]
    fn test_memo_trivial_expr() {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let (group1, expr1) = insert_group(&mut memo, expr.clone());
        let (group2, expr2) = insert_group(&mut memo, expr);

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1, 1);
        let group = memo.get_group(&group1);
        expect_group_exprs(&group, vec![expr1]);

        expect_memo(
            &memo,
            r#"
00 Leaf a
"#,
        )
    }

    #[test]
    fn test_memo_node_leaf_expr() {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Node {
            input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
        });

        let (group1, expr1) = insert_group(&mut memo, expr.clone());
        let (group2, expr2) = insert_group(&mut memo, expr);

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );
    }

    #[test]
    fn test_memo_node_multiple_leaves() {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Nodes {
            inputs: vec![
                TestOperator::from(TestRelExpr::Leaf("a")).into(),
                TestOperator::from(TestRelExpr::Leaf("b")).into(),
            ],
        });

        let (group1, expr1) = insert_group(&mut memo, expr.clone());
        let (group2, expr2) = insert_group(&mut memo, expr);

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1, 1);

        expect_memo(
            &memo,
            r#"
02 Nodes 00 01
01 Leaf b
00 Leaf a
"#,
        );
    }

    #[test]
    fn test_memo_node_nested_duplicates() {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Nodes {
            inputs: vec![
                TestOperator::from(TestRelExpr::Node {
                    input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
                })
                .into(),
                TestOperator::from(TestRelExpr::Leaf("b")).into(),
                TestOperator::from(TestRelExpr::Node {
                    input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
                })
                .into(),
            ],
        });

        let (group1, expr1) = insert_group(&mut memo, expr.clone());
        let (group2, expr2) = insert_group(&mut memo, expr);

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1, 1);

        expect_memo(
            &memo,
            r#"
03 Nodes 01 02 01
02 Leaf b
01 Node 00
00 Leaf a
"#,
        );
    }

    #[test]
    fn test_memo_node_duplicate_leaves() {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Nodes {
            inputs: vec![
                TestOperator::from(TestRelExpr::Leaf("a")).into(),
                TestOperator::from(TestRelExpr::Leaf("b")).into(),
                TestOperator::from(TestRelExpr::Leaf("a")).into(),
            ],
        });

        let (group1, expr1) = insert_group(&mut memo, expr.clone());
        let (group2, expr2) = insert_group(&mut memo, expr);

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1, 1);

        expect_memo(
            &memo,
            r#"
02 Nodes 00 01 00
01 Leaf b
00 Leaf a
"#,
        );
    }

    #[test]
    fn test_insert_member() {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));

        let (group_id, _) = insert_group(&mut memo, expr);
        expect_memo(
            &memo,
            r#"
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestRelExpr::Leaf("a0"));
        let _ = insert_group_member(&mut memo, group_id, expr);

        expect_memo(
            &memo,
            r#"
00 Leaf a
   Leaf a0
"#,
        );
    }

    #[test]
    fn test_insert_member_to_a_leaf_group() {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Node {
            input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
        });

        let (group_id, _) = insert_group(&mut memo, expr);
        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );

        let group = memo.get_group(&group_id);
        let child_expr = group.mexpr().children().next().unwrap();
        let child_group_id = child_expr.group_id();
        let expr = TestOperator::from(TestRelExpr::Leaf("a0"));
        let _ = insert_group_member(&mut memo, child_group_id, expr);

        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
   Leaf a0
"#,
        );
    }

    #[test]
    fn test_insert_member_to_the_top_group() {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Node {
            input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
        });

        let (group_id, _) = insert_group(&mut memo, expr);
        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestRelExpr::Node {
            input: TestOperator::from(TestRelExpr::Leaf("a0")).into(),
        });
        let _ = insert_group_member(&mut memo, group_id, expr);

        expect_memo(
            &memo,
            r#"
02 Leaf a0
01 Node 00
   Node 02
00 Leaf a
"#,
        );
    }

    #[test]
    fn test_memo_debug_formatting() {
        let mut memo = new_memo();
        assert_eq!(format!("{:?}", memo), "Memo { groups: 0, exprs: 0, expr_to_group: {}, metadata: () }");

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let (group_id, expr) = insert_group(&mut memo, expr);

        assert_eq!(
            format!("{:?}", memo),
            "Memo { groups: 1, exprs: 1, expr_to_group: {ExprId(0): GroupId(0)}, metadata: () }"
        );
        assert_eq!(format!("{:?}", expr), "MemoExprRef { id: ExprId(0) }");

        let group = memo.get_group(&group_id);
        let mut mexpr_iter = group.mexprs();

        assert_eq!(format!("{:?}", mexpr_iter), "[MemoExprRef { id: ExprId(0) }]");
        mexpr_iter.next();
        assert_eq!(format!("{:?}", mexpr_iter), "[]");
    }

    #[test]
    fn test_trivial_nestable_expr() {
        let sub_expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let expr = TestScalarExpr::Gt {
            lhs: Box::new(TestScalarExpr::Value(100)),
            rhs: Box::new(TestScalarExpr::SubQuery(sub_expr.into())),
        };
        let expr = TestOperator {
            expr: ExprPtr::new(TestExpr::Scalar(expr)),
            group: ExprGroupPtr::new(TestProps::Scalar(ScalarProps::default())),
        };

        let mut memo = new_memo();
        let _ = insert_group(&mut memo, expr);

        expect_memo(
            &memo,
            r#"
01 Expr (100 > SubQuery 00)
00 Leaf a
"#,
        );
    }

    #[test]
    fn test_scalar_expr_complex_nested_exprs() {
        let sub_expr = TestOperator::from(TestRelExpr::Filter {
            input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
            filter: TestScalarExpr::Value(111).into(),
        });

        let inner_filter = TestScalarExpr::Gt {
            lhs: Box::new(TestScalarExpr::Value(100)),
            rhs: Box::new(TestScalarExpr::SubQuery(sub_expr.into())),
        };

        let sub_expr2 = TestScalarExpr::SubQuery(TestOperator::from(TestRelExpr::Leaf("b")).into());
        let filter_expr = TestScalarExpr::Gt {
            lhs: Box::new(sub_expr2.clone()),
            rhs: Box::new(inner_filter.clone()),
        };

        let expr = TestOperator::from(TestRelExpr::Filter {
            input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
            filter: filter_expr.into(),
        });

        let mut memo = new_memo();
        let (group_id, _) = insert_group(&mut memo, expr);

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

        let group = memo.get_group(&group_id);
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
            input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
            filter: filter_expr.into(),
        });

        let mut memo = new_memo();
        let _ = insert_group(&mut memo, expr);

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
    }

    #[test]
    fn test_callback() {
        #[derive(Debug)]
        struct Callback {
            added: Rc<RefCell<Vec<String>>>,
        }
        impl MemoGroupCallback for Callback {
            type Expr = TestExpr;
            type Props = TestProps;
            type Metadata = ();

            fn new_group(&self, expr: &Self::Expr, props: &Self::Props, _metadata: &Self::Metadata) -> Self::Props {
                let mut added = self.added.borrow_mut();
                let mut buf = String::new();
                let mut fmt = StringMemoFormatter::new(&mut buf);
                TestOperator::format_expr(expr, props, &mut fmt);
                added.push(buf);
                props.clone()
            }
        }

        let added = Rc::new(RefCell::new(vec![]));
        let callback = Callback { added: added.clone() };
        let mut memo = Memo::with_callback((), Rc::new(callback));

        insert_group(&mut memo, TestOperator::from(TestRelExpr::Leaf("A")));

        {
            let added = added.borrow();
            assert_eq!(added[0], "Leaf A");
            assert_eq!(added.len(), 1);
        }

        let leaf = TestOperator::from(TestRelExpr::Leaf("B"));
        let expr = TestOperator::from(TestRelExpr::Node { input: leaf.into() });
        insert_group(&mut memo, expr.clone());

        {
            let added = added.borrow();
            assert_eq!(added[1], "Leaf B");
            assert_eq!(added[2], "Node 01");
            assert_eq!(added.len(), 3);
        }

        insert_group(&mut memo, expr);
        {
            let added = added.borrow();
            assert_eq!(added.len(), 3, "added duplicates. memo:\n{}", format_test_memo(&memo, false));
        }
    }

    fn new_memo() -> Memo<TestOperator, ()> {
        Memo::new(())
    }

    fn insert_group(memo: &mut Memo<TestOperator, ()>, expr: TestOperator) -> (GroupId, MemoExprRef<TestOperator>) {
        let expr = memo.insert_group(expr);
        let group_id = expr.group_ptr().memo_group_id();
        let expr_ref = expr.expr_ptr().memo_expr();

        (group_id, expr_ref.clone())
    }

    fn insert_group_member(
        memo: &mut Memo<TestOperator, ()>,
        group_id: GroupId,
        expr: TestOperator,
    ) -> (GroupId, MemoExprRef<TestOperator>) {
        let group = memo.get_group(&group_id);
        let token = group.to_membership_token();
        let expr = memo.insert_group_member(token, expr);

        let expr_ref = expr.expr_ptr().memo_expr();

        (group_id, expr_ref.clone())
    }

    fn expect_group_size<T>(memo: &Memo<TestOperator, T>, group_id: &GroupId, size: usize) {
        let group = memo.get_group(group_id);
        assert_eq!(group.mexprs().count(), size, "group#{}", group_id);
    }

    fn expect_group_exprs(group: &MemoExprGroupRef<TestOperator>, expected: Vec<MemoExprRef<TestOperator>>) {
        let actual: Vec<MemoExprRef<TestOperator>> = group.mexprs().cloned().collect();
        assert_eq!(actual, expected, "group#{} exprs", group.id());
    }

    fn expect_memo<T>(memo: &Memo<TestOperator, T>, expected: &str) {
        let lines: Vec<String> = expected.split('\n').map(String::from).collect();
        let expected = lines.join("\n");

        let buf = format_test_memo(memo, false);
        assert_eq!(buf.trim(), expected.trim());
    }

    fn expect_memo_with_props<T>(memo: &Memo<TestOperator, T>, expected: &str) {
        let lines: Vec<String> = expected.split('\n').map(String::from).collect();
        let expected = lines.join("\n");

        let buf = format_test_memo(memo, true);
        assert_eq!(buf.trim(), expected.trim());
    }

    fn format_test_memo<T>(memo: &Memo<TestOperator, T>, include_props: bool) -> String {
        let mut buf = String::new();
        let mut f = StringMemoFormatter::new(&mut buf);
        for group in memo.groups.iter().rev() {
            // SAFETY: No mutable references to the underlying group data exists.
            f.push_str(format!("{} ", group.group_id).as_str());
            for (i, expr) in group.exprs.iter().enumerate() {
                if i > 0 {
                    // new line + 3 spaces
                    f.push_str("\n   ");
                    TestOperator::format_expr(expr.expr(), expr.props(), &mut f);
                } else {
                    TestOperator::format_expr(expr.expr(), expr.props(), &mut f);

                    if include_props && group.props_ref.get().relational() != &RelProps::default() {
                        let props = group.props_ref.get().relational();
                        f.write_value("a", props.a)
                    }
                }
            }
            f.push('\n');
        }
        buf
    }
}
