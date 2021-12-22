mod store;

use crate::memo::store::{Store, StoreElementId, StoreElementRef};
use itertools::Itertools;
use std::collections::VecDeque;
use std::fmt::{Display, Formatter, Write};
use std::hash::{Hash, Hasher};
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
pub struct Memo<E, T>
where
    E: MemoExpr,
{
    groups: Store<MemoGroupData<E>>,
    exprs: Store<E>,
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
            groups: Store::new(PAGE_SIZE),
            exprs: Store::new(PAGE_SIZE),
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
            groups: Store::new(PAGE_SIZE),
            exprs: Store::new(PAGE_SIZE),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            metadata,
            callback: Some(callback),
        }
    }

    /// Copies the given expression `expr` into this memo. if this memo does not contain the given expression
    /// a new memo group is created and this method returns a reference to it. Otherwise returns a reference
    /// to the already existing expression.
    pub fn insert_group(&mut self, expr: E) -> (MemoGroupRef<E>, MemoExprRef<E>) {
        let copy_in = CopyIn {
            memo: self,
            parent: None,
            depth: 0,
        };
        copy_in.execute(&expr)
    }

    /// Copies the expression `expr` into this memo and adds it to the given group.
    /// If an identical expression already exist this method simply returns a reference to that expression.
    pub fn insert_group_member(&mut self, group: &MemoGroupRef<E>, expr: E) -> (MemoGroupRef<E>, MemoExprRef<E>) {
        let copy_in = CopyIn {
            memo: self,
            parent: Some(group.clone()),
            depth: 0,
        };
        copy_in.execute(&expr)
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

    fn get_group_ref(&self, group_id: &GroupId) -> MemoGroupRef<E> {
        let group_ref = self
            .groups
            .get(group_id.index())
            .unwrap_or_else(|| panic!("group id is invalid: {}", group_id));

        MemoGroupRef::new(*group_id, group_ref)
    }

    fn add_group(&mut self, group: MemoGroupData<E>) -> MemoGroupRef<E> {
        let (id, elem_ref) = self.groups.insert(group);
        MemoGroupRef::new(GroupId(id), elem_ref)
    }

    fn add_expr(&mut self, expr: E::Expr, parent: MemoGroupRef<E>) -> MemoExprRef<E> {
        let expr = E::create(ExprRef::Detached(Box::new(expr)), ExprGroupRef::Memo(parent.clone()));
        let (id, elem_ref) = self.exprs.insert(expr);
        let expr_ref = MemoExprRef::new(ExprId(id), elem_ref);

        self.add_expr_to_group(&parent, expr_ref.clone());
        expr_ref
    }

    fn add_expr_to_group(&mut self, group: &MemoGroupRef<E>, expr: MemoExprRef<E>) {
        let group_id = group.id();
        let expr_id = expr.id();
        let memo_group = self.groups.get_mut(group_id.index()).unwrap();

        memo_group.exprs.push(expr);
        self.expr_to_group.insert(expr_id, group_id);
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
            .field("groups", &self.groups)
            .field("exprs", &self.exprs)
            .field("expr_to_group", &self.expr_to_group)
            .field("metadata", &self.metadata)
            .finish()
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
    /// This method is a shorthand for `memo_expr.expr_ref().expr()`.
    fn expr(&self) -> &Self::Expr {
        self.expr_ref().expr()
    }

    /// Returns properties associated with this expression.
    /// This method is a shorthand for `memo_expr.group_ref().props()`.
    fn props(&self) -> &Self::Props {
        self.group_ref().props()
    }

    /// Creates a new memo expression from the given expression and group.
    fn create(expr: ExprRef<Self>, group: ExprGroupRef<Self>) -> Self;

    /// Creates a new memo expression from the first expression of the given memo group.
    fn from_group(group: MemoGroupRef<Self>) -> Self {
        Self::create(ExprRef::Memo(group.mexpr().clone()), ExprGroupRef::Memo(group))
    }

    /// Returns a reference to underlying expression.
    fn expr_ref(&self) -> &ExprRef<Self>;

    /// Returns a reference to a group this memo expression belong to.
    fn group_ref(&self) -> &ExprGroupRef<Self>;

    /// Recursively traverses this expression and copies it into a memo.
    fn copy_in<T>(&self, visitor: &mut CopyInExprs<Self, T>);

    /// Creates a new expression from the given expression by replacing its child expressions
    /// provided by the given [NewChildExprs](self::NewChildExprs).
    fn expr_with_new_children(expr: &Self::Expr, inputs: NewChildExprs<Self>) -> Self::Expr;

    /// Called when a scalar expression with nested sub-queries should be added to a memo.
    /// Return properties that contain the given memo groups.
    fn new_properties_with_nested_sub_queries(
        props: Self::Props,
        sub_queries: impl Iterator<Item = MemoGroupRef<Self>>,
    ) -> Self::Props;

    /// Returns the number of child expressions of this memo expression.
    fn num_children(&self) -> usize;

    /// Returns the i-th child expression of this memo expression.
    fn get_child(&self, i: usize) -> Option<ChildNodeRef<Self>>;

    /// Returns nested sub-queries of this memo expression.
    fn get_sub_queries(&self) -> Option<SubQueries<Self>>;

    /// Builds a textual representation of the given expression.
    fn format_expr<F>(expr: &Self::Expr, props: &Self::Props, f: &mut F)
    where
        F: MemoExprFormatter;
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
    fn as_relational(&self) -> &Self::RelExpr;

    /// Returns a reference to the underlying scalar expression.
    ///
    /// # Panics
    ///
    /// This method panics if the underlying expression is not a scalar expression.
    fn as_scalar(&self) -> &Self::ScalarExpr;

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
    fn as_relational(&self) -> &Self::RelProps;

    /// Return a reference to the scalar properties.
    ///
    /// # Panics
    ///
    /// This method panics if the underlying properties are not scalar.
    fn as_scalar(&self) -> &Self::ScalarProps;
}

/// Provides access to nested sub-queries of a scalar memo-expression.
pub struct SubQueries<'a, E>
where
    E: MemoExpr,
{
    sub_queries: &'a [MemoGroupRef<E>],
}

impl<'a, E> SubQueries<'a, E>
where
    E: MemoExpr,
{
    pub fn new(sub_queries: &'a [MemoGroupRef<E>]) -> Self {
        SubQueries { sub_queries }
    }

    /// Returns the number of nested sub-queries of a memo expression.
    pub fn num(&self) -> usize {
        self.sub_queries.len()
    }

    /// Returns the i-th nested sub-query of a memo expression.
    pub fn get_sub_query(&self, i: usize) -> Option<MemoGroupRef<E>> {
        self.sub_queries.get(i).cloned()
    }
}

/// `ExprRef` is a reference to an expression.
///
/// `ExprRef` reference has two states `detached` and `memo`:
/// * In the detached state this reference stores an expression.
/// * In the memo is a reference to a [memo expression](self::MemoExprRef).
#[derive(Debug, Clone)]
pub enum ExprRef<E>
where
    E: MemoExpr,
{
    /// The underlying expression.
    Detached(Box<E::Expr>),
    /// A reference to a [memo expression](self::MemoExprRef).
    Memo(MemoExprRef<E>),
}

impl<E> ExprRef<E>
where
    E: MemoExpr,
{
    /// Returns a reference to an expression.
    /// * In the detached state this method returns a reference to the underlying expression.
    /// * In the memo state this method returns a reference to the first expression in the [memo group](self::MemoGroupRef).
    pub fn expr(&self) -> &E::Expr {
        match self {
            ExprRef::Detached(expr) => expr.as_ref(),
            ExprRef::Memo(expr) => expr.expr(),
        }
    }
}

/// `ExprGroupRef` is a reference to a group an expression belongs to.
///
/// `ExprGroupRef` has two states `detached` and `memo`:
/// * In the detached state this reference stores properties associated with the expression.
/// * In the memo state it stores a reference to a [memo group](self::MemoGroupRef).
#[derive(Clone)]
pub enum ExprGroupRef<E>
where
    E: MemoExpr,
{
    /// Detached state.
    //TODO: Since transformation rules do not use properties => they should not pay the cost of a heap allocation of an
    // empty properties object.
    // Copying an alternative expression into a memo also requires MemoExpr which consists of an expression and properties
    // -> this means that it also requires a heap allocation.
    // Add MemoExpr::empty_props(?) and Detached state (Box<PropsRef> | Empty(const T)).
    Detached(Box<E::Props>),
    /// A reference to a [memo group](self::MemoGroupRef) an expression belongs to.
    Memo(MemoGroupRef<E>),
}

impl<E> ExprGroupRef<E>
where
    E: MemoExpr,
{
    /// Returns properties of this group.
    /// * In the detached state this methods returns a reference to properties associated with the expression.
    /// * In the memo state this method returns a reference of a memo group the expression belongs to.
    pub fn props(&self) -> &E::Props {
        match self {
            ExprGroupRef::Detached(props) => props.as_ref(),
            ExprGroupRef::Memo(group) => group.props(),
        }
    }

    /// Returns an iterator over expressions in this group.
    /// If this group is in the detached state than that iterator produces no elements.
    pub fn mexprs(&self) -> MemoGroupIter<E> {
        match self {
            ExprGroupRef::Detached(_) => MemoGroupIter {
                group: None,
                position: 0,
            },
            ExprGroupRef::Memo(group) => group.mexprs(),
        }
    }

    fn memo_group(&self) -> &MemoGroupRef<E> {
        match self {
            // This method is only called from MemoExprRef which stores ExprGroupRef in the memo state.
            ExprGroupRef::Detached(_) => unreachable!(),
            ExprGroupRef::Memo(group) => group,
        }
    }
}

impl<E, P> Debug for ExprGroupRef<E>
where
    E: MemoExpr<Props = P>,
    P: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExpGroup(")?;
        match self {
            ExprGroupRef::Detached(group) => write!(f, "{:?}", group)?,
            ExprGroupRef::Memo(group) => write!(f, "{:?}", group)?,
        }
        write!(f, ")")
    }
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
    T: Expr<RelExpr = RelExpr> + 'static,
    P: Props<RelProps = RelProps> + 'static,
{
    /// Creates a relational node of an expression tree from the given relational expression and properties.
    pub fn new(expr: RelExpr, props: RelProps) -> Self {
        let expr = ExprRef::Detached(Box::new(T::new_rel(expr)));
        let group = ExprGroupRef::Detached(Box::new(P::new_rel(props)));
        let expr = E::create(expr, group);
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
    pub fn from_group(group: MemoGroupRef<E>) -> Self {
        let expr = E::from_group(group);
        RelNode(expr)
    }

    /// Returns a reference to the underlying expression.
    ///
    /// # Panics
    ///
    /// This method panics if this node does not hold a relational expression.
    pub fn expr(&self) -> &T::RelExpr {
        let expr = self.0.expr_ref();
        let expr = expr.expr();
        expr.as_relational()
    }

    /// Returns a reference to properties associated with this node:
    /// * if this node is an expression returns a reference to the properties of the underlying expression.
    /// * If this node is a memo group returns a reference to the properties of the first expression of this memo group.
    pub fn props(&self) -> &RelProps {
        self.0.props().as_relational()
    }

    /// Returns an [self::ExprRef] of the underlying memo expression.
    pub fn expr_ref(&self) -> &ExprRef<E> {
        self.0.expr_ref()
    }
}

impl<T> From<ChildNode<T>> for RelNode<T>
where
    T: MemoExpr,
{
    fn from(expr: ChildNode<T>) -> Self {
        let expr = match expr {
            ChildNode::Expr(expr) => expr,
            ChildNode::Group(group) => T::from_group(group),
        };
        RelNode(expr)
    }
}

impl<'a, T> From<&'a RelNode<T>> for ChildNodeRef<'a, T>
where
    T: MemoExpr + 'static,
{
    fn from(node: &'a RelNode<T>) -> Self {
        match node.0.expr_ref() {
            ExprRef::Detached(_) => ChildNodeRef::Expr(&node.0),
            ExprRef::Memo(expr) => ChildNodeRef::Group(expr.mgroup()),
        }
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
        f.debug_struct("RelNode").field("0", &self.0).finish()
    }
}

/// A scalar node of an expression tree.
pub struct ScalarNode<E>(E);

impl<E, T, ScalarExpr, P, ScalarProps> ScalarNode<E>
where
    E: MemoExpr<Expr = T, Props = P>,
    T: Expr<ScalarExpr = ScalarExpr> + 'static,
    P: Props<ScalarProps = ScalarProps> + 'static,
{
    /// Creates a scalar node of an expression tree from the given scalar expression and properties.
    pub fn new(expr: ScalarExpr, props: ScalarProps) -> Self {
        let expr = ExprRef::Detached(Box::new(T::new_scalar(expr)));
        let group = ExprGroupRef::Detached(Box::new(P::new_scalar(props)));
        let expr = E::create(expr, group);
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

    /// Creates a scalar node of an expression tree from the given memo group.
    ///
    /// # Note
    ///
    /// Caller must guarantee that the given group is a group of scalar expressions.
    pub fn from_group(group: MemoGroupRef<E>) -> Self {
        let expr = E::from_group(group);
        ScalarNode(expr)
    }

    /// Returns a reference to the underlying scalar expression.
    ///
    /// # Panics
    ///
    /// This method panics if this not does not hold a scalar expression.
    pub fn expr(&self) -> &T::ScalarExpr {
        let expr = self.0.expr_ref();
        let expr = expr.expr();
        expr.as_scalar()
    }

    /// Returns a reference to properties associated with this node:
    /// * if this node is an expression returns a reference to the properties of the underlying expression.
    /// * If this node is a memo group returns a reference to the properties of the first expression of this memo group.
    pub fn props(&self) -> &ScalarProps {
        self.0.props().as_scalar()
    }

    /// Returns an [self::ExprRef] of the underlying memo expression.
    pub fn expr_ref(&self) -> &ExprRef<E> {
        self.0.expr_ref()
    }
}

impl<E> From<ChildNode<E>> for ScalarNode<E>
where
    E: MemoExpr,
{
    fn from(expr: ChildNode<E>) -> Self {
        let expr = match expr {
            ChildNode::Expr(expr) => expr,
            ChildNode::Group(group) => E::from_group(group),
        };
        ScalarNode(expr)
    }
}

impl<'a, T> From<&'a ScalarNode<T>> for ChildNodeRef<'a, T>
where
    T: MemoExpr + 'static,
{
    fn from(node: &'a ScalarNode<T>) -> Self {
        match node.0.expr_ref() {
            ExprRef::Detached(_) => ChildNodeRef::Expr(&node.0),
            ExprRef::Memo(expr) => ChildNodeRef::Group(expr.mgroup()),
        }
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
        f.debug_struct("ScalarNode").field("0", &self.0).finish()
    }
}

/// `ChildNode` represents an expression in within an expression tree.
/// An instance of `ChildNode` can be an expression or a memo group. Initially every expression in an expression tree
/// has a direct link its child expressions (represented by [ChildNode::Expr]).
/// When an expression is copied into a [`memo`](self::Memo) its child expressions are replaced with
/// references to memo groups (a reference to a memo group is represented by [ChildNode::Group]).
///
/// This enum is used by [NewChildExprs] to create an expression with the specified child expressions.
#[derive(Clone)]
pub enum ChildNode<T>
where
    T: MemoExpr,
{
    /// A node is an expression. This variant is used when an expression is copied from a memo.
    Expr(T),
    /// A node is a memo group. This variant is used when an expression is copied into a memo.
    Group(MemoGroupRef<T>),
}

impl<T> ChildNode<T>
where
    T: MemoExpr,
{
    fn expr(&self) -> &T::Expr {
        match self {
            ChildNode::Expr(expr) => expr.expr(),
            ChildNode::Group(group) => group.expr(),
        }
    }
}

impl<E> From<E> for ChildNode<E>
where
    E: MemoExpr,
{
    fn from(expr: E) -> Self {
        ChildNode::Expr(expr)
    }
}

impl<E> Debug for ChildNode<E>
where
    E: MemoExpr + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ChildNode::Expr(expr) => write!(f, "Expr({:?})", expr),
            ChildNode::Group(group) => write!(f, "Group({:?})", group.id),
        }
    }
}

/// A reference to a child expression of a [memo expression](self::MemoExpr).
/// * If the memo expression has not been added to a memo stores a reference to that expression.
/// * If the memo expression has been added to a memo returns a reference to a memo group of that child expression.
#[derive(Debug)]
pub enum ChildNodeRef<'a, E>
where
    E: MemoExpr,
{
    /// Reference to a child expression.
    Expr(&'a E),
    /// Reference to a memo-group of that child expression.
    Group(&'a MemoGroupRef<E>),
}

impl<'a, E> From<&'a ChildNode<E>> for ChildNodeRef<'a, E>
where
    E: MemoExpr,
{
    fn from(expr: &'a ChildNode<E>) -> Self {
        match expr {
            ChildNode::Expr(expr) => ChildNodeRef::Expr(expr),
            ChildNode::Group(group) => ChildNodeRef::Group(group),
        }
    }
}

/// `MemoGroupRef` is a reference to a memo group. A memo group is a group of logically equivalent expressions.
/// Expressions are logically equivalent when they produce the same result.
///
/// #Safety
/// A reference to a memo group is valid until a [`memo`](self::Memo) it belongs to is dropped.
pub struct MemoGroupRef<E>
where
    E: MemoExpr,
{
    id: GroupId,
    group_ref: StoreElementRef<MemoGroupData<E>>,
}

impl<E> MemoGroupRef<E>
where
    E: MemoExpr,
{
    fn new(id: GroupId, group_ref: StoreElementRef<MemoGroupData<E>>) -> Self {
        MemoGroupRef { id, group_ref }
    }

    /// Returns an opaque identifier of this memo group.
    pub fn id(&self) -> GroupId {
        let group = self.get_memo_group();
        group.group_id
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
        MemoGroupIter {
            group: Some(group),
            position: 0,
        }
    }

    /// Returns properties shared by all expressions in this memo group.
    pub fn props(&self) -> &E::Props {
        let group = self.get_memo_group();
        &group.props
    }

    fn get_memo_group(&self) -> &MemoGroupData<E> {
        self.group_ref.as_ref()
    }
}

impl<E> Clone for MemoGroupRef<E>
where
    E: MemoExpr,
{
    fn clone(&self) -> Self {
        MemoGroupRef::new(self.id, self.group_ref.clone())
    }
}

impl<E> PartialEq for MemoGroupRef<E>
where
    E: MemoExpr,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.group_ref == other.group_ref
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
        write!(f, "MemoGroupRef {{ id: {:?} }}", self.id)
    }
}

/// An iterator over expressions of a memo group.
pub struct MemoGroupIter<'m, E>
where
    E: MemoExpr,
{
    group: Option<&'m MemoGroupData<E>>,
    position: usize,
}

impl<'m, E> Iterator for MemoGroupIter<'m, E>
where
    E: MemoExpr,
{
    type Item = &'m MemoExprRef<E>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(group) = self.group {
            if self.position < group.exprs.len() {
                let expr = &group.exprs[self.position];
                self.position += 1;
                Some(expr)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl<'m, E> Debug for MemoGroupIter<'m, E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoGroupIter([")?;
        if let Some(group) = self.group {
            for i in self.position..group.exprs.len() {
                write!(f, "{:?}", group.exprs[i])?;
            }
        }
        write!(f, "])")
    }
}

/// A reference to a memo expression.
///
/// #Safety
/// A reference to a memo expression is valid until a [`memo`](self::Memo) it belongs to is dropped.
pub struct MemoExprRef<E>
where
    E: MemoExpr,
{
    id: ExprId,
    expr_ref: StoreElementRef<E>,
}

impl<E> MemoExprRef<E>
where
    E: MemoExpr,
{
    fn new(id: ExprId, expr_ref: StoreElementRef<E>) -> Self {
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

    /// Returns a reference to the underlying memo expression stored inside a memo.
    ///
    /// # Warning
    ///
    /// Cloning the expression behind this reference can be expensive.
    pub fn mexpr(&self) -> &E {
        self.get_memo_expr()
    }

    /// Returns a reference to the memo group this memo expression belongs to.
    pub fn mgroup(&self) -> &MemoGroupRef<E> {
        let expr = self.get_memo_expr();
        expr.group_ref().memo_group()
    }

    /// Returns a reference to the properties of the memo group this expression belongs to.
    pub fn props(&self) -> &E::Props {
        let expr = self.get_memo_expr();
        expr.group_ref().props()
    }

    /// Returns an iterator over child expressions of this memo expression.
    pub fn children(&self) -> MemoExprInputsIter<E> {
        let expr = self.get_memo_expr();
        if let Some(sub_queries) = self.mexpr().get_sub_queries() {
            MemoExprInputsIter {
                expr,
                position: 0,
                num_children: sub_queries.num(),
            }
        } else {
            MemoExprInputsIter {
                expr,
                position: 0,
                num_children: expr.num_children(),
            }
        }
    }

    fn get_memo_expr(&self) -> &E {
        self.expr_ref.as_ref()
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
        write!(f, "MemoExprRef {{ id: {:?} }}", self.id)
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
    type Item = MemoGroupRef<E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.num_children {
            let position = self.position;

            self.position += 1;

            if let Some(sub_queries) = self.expr.get_sub_queries() {
                sub_queries.get_sub_query(position)
            } else {
                self.expr
                    .get_child(position)
                    .map(|e| match e {
                        ChildNodeRef::Expr(_) => unreachable!(),
                        ChildNodeRef::Group(group) => Some(group.clone()),
                    })
                    .flatten()
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
        let mut i = 0;
        write!(f, "[")?;
        while self.position + i < self.num_children {
            let p = self.position + i;
            let expr = self.expr.get_child(p).unwrap();
            if i > 0 {
                write!(f, ", ")?;
            }
            match expr {
                ChildNodeRef::Expr(_) => unreachable!(),
                ChildNodeRef::Group(group) => write!(f, "{:?}", group)?,
            }
            i += 1;
        }
        write!(f, "]")
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
        write!(f, "GroupId({:})", self.0.index())
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
        write!(f, "ExprId({:})", self.0.index())
    }
}

#[derive(Debug)]
struct MemoGroupData<E>
where
    E: MemoExpr,
{
    group_id: GroupId,
    exprs: Vec<MemoExprRef<E>>,
    props: E::Props,
}

impl<E> MemoGroupData<E>
where
    E: MemoExpr,
{
    fn new(group_id: GroupId, props: E::Props) -> Self {
        MemoGroupData {
            group_id,
            exprs: vec![],
            props,
        }
    }
}

/// Provides methods to traverse an expression tree and copy it into a memo.
pub struct CopyInExprs<'a, E, T>
where
    E: MemoExpr,
{
    memo: &'a mut Memo<E, T>,
    parent: Option<MemoGroupRef<E>>,
    depth: usize,
    result: Option<(MemoGroupRef<E>, MemoExprRef<E>)>,
}

impl<'a, E, T> CopyInExprs<'a, E, T>
where
    E: MemoExpr,
{
    fn new(memo: &'a mut Memo<E, T>, parent: Option<MemoGroupRef<E>>, depth: usize) -> Self {
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
            parent: self.parent.clone(),
            // TODO: Add a method that splits MemoExpr into Self::Expr and Self::Props
            //  this will allow to remove the clone() call below .
            props: expr.props().clone(),
        }
    }

    /// Visits the given child expression and recursively copies that expression into the memo:
    /// * If the given expression is an expression this methods recursively copies it into the memo.
    /// * If the child expression is a group this method returns a reference to that group.
    pub fn visit_expr_node<'e>(&mut self, expr_ctx: &mut ExprContext<E>, expr_node: impl Into<ChildNodeRef<'e, E>>)
    where
        E: 'e,
    {
        let input: ChildNodeRef<E> = expr_node.into();
        match input {
            ChildNodeRef::Expr(expr) => {
                let copy_in = CopyIn {
                    memo: self.memo,
                    parent: None,
                    depth: self.depth + 1,
                };
                let (group, _expr) = copy_in.execute(expr);
                expr_ctx.children.push_back(ChildNode::Group(group));
            }
            ChildNodeRef::Group(group) => {
                expr_ctx.children.push_back(ChildNode::Group(group.clone()));
            }
        }
    }

    /// Visits the given optional child expression if it is present and recursively copies it into a memo.
    /// This method is equivalent to:
    /// ```text
    /// if let Some(expr_node) = expr_node {
    ///   visitor.visit_expr_node(expr_ctx, expr_node);
    /// }
    /// ```
    pub fn visit_opt_expr_node<'e>(
        &mut self,
        expr_ctx: &mut ExprContext<E>,
        expr_node: Option<impl Into<ChildNodeRef<'e, E>>>,
    ) where
        E: MemoExpr + 'e,
    {
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
                children.iter().map(|e| match e {
                    ChildNode::Expr(_) => {
                        unreachable!("ExprContext.children must contain only references to memo groups")
                    }
                    ChildNode::Group(group) => group.clone(),
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

        if added {
            let (group_ref, expr_ref) = if let Some(parent) = parent {
                let expr_ref = self.memo.add_expr(expr, parent.clone());

                (parent, expr_ref)
            } else {
                let group_id = GroupId(self.memo.groups.next_id());
                let props = if let Some(callback) = self.memo.callback.as_ref() {
                    callback.new_group(&expr, &props, &self.memo.metadata)
                } else {
                    props
                };
                let memo_group = MemoGroupData::new(group_id, props);
                let group_ref = self.memo.add_group(memo_group);
                let expr_ref = self.memo.add_expr(expr, group_ref.clone());

                (group_ref, expr_ref)
            };

            self.result = Some((group_ref, expr_ref));
        } else {
            let group_id = self.memo.expr_to_group.get(&expr_id).expect("Unexpected expression");
            let group_ref = self.memo.get_group_ref(group_id);
            let expr_ref = self.memo.get_expr_ref(&expr_id);

            self.result = Some((group_ref, expr_ref));
        }
    }
}

/// Stores information that is used to build a new memo expression.
pub struct ExprContext<E>
where
    E: MemoExpr,
{
    children: VecDeque<ChildNode<E>>,
    parent: Option<MemoGroupRef<E>>,
    props: E::Props,
}

/// A stack-like data structure used by [MemoExpr::expr_with_new_children].
/// It stores new child expressions and provides convenient methods of their retrieval.
#[derive(Debug)]
pub struct NewChildExprs<E>
where
    E: MemoExpr,
{
    children: VecDeque<ChildNode<E>>,
    capacity: usize,
    index: usize,
}

impl<E> NewChildExprs<E>
where
    E: MemoExpr,
{
    /// Creates an instance of `NewChildExprs`.
    pub fn new(children: VecDeque<ChildNode<E>>) -> Self {
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
        RelNode::from(expr)
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
                RelNode::from(i)
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
        ScalarNode::from(expr)
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
                ScalarNode::from(i)
            })
            .collect()
    }

    fn expr(&mut self) -> ChildNode<E> {
        self.ensure_available(1);
        self.next_index()
    }

    fn exprs(&mut self, n: usize) -> Vec<ChildNode<E>> {
        self.ensure_available(n);
        let mut children = Vec::with_capacity(n);
        for _ in 0..n {
            children.push(self.next_index());
        }
        children
    }

    fn next_index(&mut self) -> ChildNode<E> {
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

    fn expect_relational(expr: &ChildNode<E>) {
        assert!(!expr.expr().is_scalar(), "expected a relational expression");
    }

    fn expect_scalar(expr: &ChildNode<E>) {
        assert!(expr.expr().is_scalar(), "expected a scalar expression");
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
    nested_exprs: Vec<MemoGroupRef<E>>,
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
        for input in self.nested_exprs {
            self.ctx.visit_expr_node(self.expr_ctx, &ChildNode::Group(input));
        }
    }

    /// Copies the given nested expression into a memo.
    pub fn visit_expr(&mut self, expr: ChildNodeRef<E>) {
        match expr {
            ChildNodeRef::Expr(expr) => {
                expr.copy_in(self.ctx);
                let (group, _) = std::mem::take(&mut self.ctx.result).expect("Failed to copy in a nested expressions");
                self.add_group(group);
            }
            ChildNodeRef::Group(group) => self.add_group(group.clone()),
        }
    }

    fn add_group(&mut self, group: MemoGroupRef<E>) {
        self.nested_exprs.push(group)
    }
}

#[doc(hidden)]
struct CopyIn<'a, E, T>
where
    E: MemoExpr,
{
    memo: &'a mut Memo<E, T>,
    parent: Option<MemoGroupRef<E>>,
    depth: usize,
}

impl<'a, E, T> CopyIn<'a, E, T>
where
    E: MemoExpr,
{
    fn execute(self, expr: &E) -> (MemoGroupRef<E>, MemoExprRef<E>) {
        let mut ctx = CopyInExprs::new(self.memo, self.parent.clone(), self.depth);
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
        let group = group.as_ref();
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
    fn write_expr<'e, T>(&mut self, name: &str, input: impl Into<ChildNodeRef<'e, T>>)
    where
        T: MemoExpr + 'e;

    /// Writes a child expression it is present. This method is equivalent to:
    /// ```text
    /// if let Some(expr) = expr {
    ///   fmt.write_expr(name, expr);
    /// }
    /// ```
    fn write_expr_if_present<'e, T>(&mut self, name: &str, expr: Option<impl Into<ChildNodeRef<'e, T>>>)
    where
        T: MemoExpr + 'e,
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

    fn write_expr<'e, T>(&mut self, name: &str, input: impl Into<ChildNodeRef<'e, T>>)
    where
        T: MemoExpr + 'e,
    {
        self.buf.push(' ');
        if !name.is_empty() {
            self.buf.push_str(name);
            self.buf.push('=');
        }
        let s = format_node_ref(input.into());
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

    fn write_expr<'e, T>(&mut self, _name: &str, input: impl Into<ChildNodeRef<'e, T>>)
    where
        T: MemoExpr + 'e,
    {
        let s = format_node_ref(input.into());
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

fn format_node_ref<T>(input: ChildNodeRef<'_, T>) -> String
where
    T: MemoExpr,
{
    match input {
        ChildNodeRef::Expr(expr) => {
            // This only happens when expression has not been added to a memo.
            let ptr: *const T::Expr = &*expr.expr();
            format!("{:?}", ptr)
        }
        ChildNodeRef::Group(group) => format!("{}", group.id()),
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

        fn as_relational(&self) -> &Self::RelExpr {
            match self {
                TestExpr::Relational(expr) => expr,
                TestExpr::Scalar(_) => panic!(),
            }
        }

        fn as_scalar(&self) -> &Self::ScalarExpr {
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

    impl TestExpr {
        fn as_relational(&self) -> &TestRelExpr {
            match self {
                TestExpr::Relational(expr) => expr,
                TestExpr::Scalar(_) => panic!("Expected a relational expr"),
            }
        }

        fn as_scalar(&self) -> &TestScalarExpr {
            match self {
                TestExpr::Relational(_) => panic!("Expected relational expr"),
                TestExpr::Scalar(expr) => expr,
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
                    collector.visit_expr(expr.into());
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
                    ExprRef::Detached(expr) => {
                        let ptr: *const TestExpr = expr.as_ref();
                        write!(f, "SubQuery expr_ptr {:?}", ptr)
                    }
                    ExprRef::Memo(expr) => {
                        write!(f, "SubQuery {}", expr.mgroup().id())
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
            let scalar = expr.expr().as_scalar();
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
        expr: ExprRef<TestOperator>,
        group: ExprGroupRef<TestOperator>,
    }

    impl From<TestRelExpr> for TestOperator {
        fn from(expr: TestRelExpr) -> Self {
            TestOperator {
                expr: ExprRef::Detached(Box::new(TestExpr::Relational(expr))),
                group: ExprGroupRef::Detached(Box::new(TestProps::Rel(RelProps::default()))),
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
        sub_queries: Vec<MemoGroupRef<TestOperator>>,
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

        fn as_relational(&self) -> &RelProps {
            match self {
                TestProps::Rel(expr) => expr,
                TestProps::Scalar(_) => panic!("Expected relational properties"),
            }
        }

        fn as_scalar(&self) -> &ScalarProps {
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
                group: ExprGroupRef::Detached(Box::new(TestProps::Rel(RelProps { a }))),
            }
        }
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
            sub_queries: impl Iterator<Item = MemoGroupRef<Self>>,
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
                TestExpr::Scalar(_) => self.group.props().as_scalar().sub_queries.len(),
            }
        }

        fn get_child(&self, i: usize) -> Option<ChildNodeRef<Self>> {
            match self.expr() {
                TestExpr::Relational(expr) => match expr {
                    TestRelExpr::Leaf(_) => None,
                    TestRelExpr::Node { input } if i == 0 => Some(input.into()),
                    TestRelExpr::Node { .. } => None,
                    TestRelExpr::Nodes { inputs } if i < inputs.len() => inputs.get(i).map(|e| e.into()),
                    TestRelExpr::Nodes { .. } => None,
                    TestRelExpr::Filter { input, .. } if i == 0 => Some(input.into()),
                    TestRelExpr::Filter { filter, .. } if i == 1 => Some(filter.into()),
                    TestRelExpr::Filter { .. } => None,
                },
                TestExpr::Scalar(_) => {
                    let props = self.group.props().as_scalar();
                    props.sub_queries.get(i).map(ChildNodeRef::Group)
                }
            }
        }

        fn get_sub_queries(&self) -> Option<SubQueries<Self>> {
            match self.props() {
                TestProps::Rel(_) => None,
                TestProps::Scalar(props) => Some(SubQueries::new(&props.sub_queries)),
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
        let (group, expr) = memo.insert_group(expr1);

        assert_eq!(group.props().as_relational(), &RelProps { a: 100 }, "group properties");
        assert_eq!(expr.mgroup(), &group, "expr group");

        let group_by_id = memo.get_group_ref(&group.id());
        assert_eq!(group, group_by_id);

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

        let _ = memo.insert_group(outer);

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
        let (_, expr) = memo.insert_group(expr);

        match expr.expr().as_relational() {
            TestRelExpr::Node { input } => {
                let children: Vec<_> = expr.children().collect();
                assert_eq!(
                    format!("{:?}", input.expr()),
                    format!("{:?}", children[0].expr().as_relational()),
                    "child expr"
                );
                assert_eq!(input.props(), leaf.props().as_relational(), "input props");
            }
            _ => panic!("Unexpected expression: {:?}", expr),
        }
    }

    #[test]
    fn test_memo_trivial_expr() {
        let mut memo = new_memo();

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let (group1, expr1) = memo.insert_group(expr.clone());
        let (group2, expr2) = memo.insert_group(expr);

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1.id(), 1);
        expect_group_exprs(&group1, vec![expr1]);

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

        let (group1, expr1) = memo.insert_group(expr.clone());
        let (group2, expr2) = memo.insert_group(expr);

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

        let (group1, expr1) = memo.insert_group(expr.clone());
        let (group2, expr2) = memo.insert_group(expr);

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1.id(), 1);

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

        let (group1, expr1) = memo.insert_group(expr.clone());
        let (group2, expr2) = memo.insert_group(expr);

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1.id(), 1);

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

        let (group1, expr1) = memo.insert_group(expr.clone());
        let (group2, expr2) = memo.insert_group(expr);

        assert_eq!(group1, group2);
        assert_eq!(expr1, expr2);

        expect_group_size(&memo, &group1.id(), 1);

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

        let (group, _) = memo.insert_group(expr);
        expect_memo(
            &memo,
            r#"
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestRelExpr::Leaf("a0"));
        let _ = memo.insert_group_member(&group, expr);

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

        let (group, _) = memo.insert_group(expr);
        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestRelExpr::Leaf("a0"));
        let children: Vec<_> = group.mexpr().children().collect();
        let _ = memo.insert_group_member(&children[0], expr);

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

        let (group, _) = memo.insert_group(expr);
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
        let _ = memo.insert_group_member(&group, expr);

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
        assert_eq!(format!("{:?}", memo), "Memo { groups: [], exprs: [], expr_to_group: {}, metadata: () }");

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let (group, expr) = memo.insert_group(expr);

        assert_eq!(format!("{:?}", group), "MemoGroupRef { id: GroupId(0) }");
        assert_eq!(format!("{:?}", expr), "MemoExprRef { id: ExprId(0) }");

        let mut mexpr_iter = group.mexprs();

        assert_eq!(format!("{:?}", mexpr_iter), "MemoGroupIter([MemoExprRef { id: ExprId(0) }])");
        mexpr_iter.next();
        assert_eq!(format!("{:?}", mexpr_iter), "MemoGroupIter([])");
    }

    #[test]
    fn test_trivial_nestable_expr() {
        let sub_expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let expr = TestScalarExpr::Gt {
            lhs: Box::new(TestScalarExpr::Value(100)),
            rhs: Box::new(TestScalarExpr::SubQuery(sub_expr.into())),
        };
        let expr = TestOperator {
            expr: ExprRef::Detached(Box::new(TestExpr::Scalar(expr))),
            group: ExprGroupRef::Detached(Box::new(TestProps::Scalar(ScalarProps::default()))),
        };

        let mut memo = new_memo();
        let _ = memo.insert_group(expr);

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
        let (group, _) = memo.insert_group(expr);

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

        let children: Vec<_> = group.mexpr().children().collect();
        assert_eq!(
            children.len(),
            2,
            "Filter expression has 2 child expressions: an input and a filter: {:?}",
            children
        );

        let expr = children[1].mexpr();
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
        let (_, _) = memo.insert_group(expr);

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

        memo.insert_group(TestOperator::from(TestRelExpr::Leaf("A")));

        {
            let added = added.borrow();
            assert_eq!(added[0], "Leaf A");
            assert_eq!(added.len(), 1);
        }

        let leaf = TestOperator::from(TestRelExpr::Leaf("B"));
        let expr = TestOperator::from(TestRelExpr::Node { input: leaf.into() });
        memo.insert_group(expr.clone());

        {
            let added = added.borrow();
            assert_eq!(added[1], "Leaf B");
            assert_eq!(added[2], "Node 01");
            assert_eq!(added.len(), 3);
        }

        memo.insert_group(expr);
        {
            let added = added.borrow();
            assert_eq!(added.len(), 3, "added duplicates. memo:\n{}", format_test_memo(&memo, false));
        }
    }

    fn new_memo() -> Memo<TestOperator, ()> {
        Memo::new(())
    }

    fn expect_group_size<T>(memo: &Memo<TestOperator, T>, group_id: &GroupId, size: usize) {
        let group = memo.groups.get(group_id.index()).unwrap();
        let group = group.as_ref();
        assert_eq!(group.exprs.len(), size, "group#{}", group_id);
    }

    fn expect_group_exprs(group: &MemoGroupRef<TestOperator>, expected: Vec<MemoExprRef<TestOperator>>) {
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
            let group = group.as_ref();
            f.push_str(format!("{} ", group.group_id).as_str());
            for (i, expr) in group.exprs.iter().enumerate() {
                if i > 0 {
                    // new line + 3 spaces
                    f.push_str("\n   ");
                    TestOperator::format_expr(expr.expr(), expr.props(), &mut f);
                } else {
                    TestOperator::format_expr(expr.expr(), expr.props(), &mut f);

                    if include_props && group.props.as_relational() != &RelProps::default() {
                        let props = group.props.as_relational();
                        f.write_value("a", props.a)
                    }
                }
            }
            f.push('\n');
        }
        buf
    }
}
