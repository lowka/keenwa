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
pub struct Memo<T>
where
    T: MemoExpr,
{
    groups: Vec<MemoGroupData<T>>,
    exprs: Vec<MemoExprData<T>>,
    expr_cache: HashMap<String, ExprId>,
    expr_to_group: HashMap<ExprId, GroupId>,
    // TODO: NoOpCallback
    callback: Option<Rc<dyn MemoExprCallback<Expr = T, Props = T::Props>>>,
}

impl<T> Memo<T>
where
    T: MemoExpr,
{
    /// Creates a new memo.
    pub fn new() -> Self {
        Memo {
            groups: Vec::new(),
            exprs: Vec::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            callback: None,
        }
    }

    /// Creates a new memo with the given callback.
    pub(crate) fn with_callback(callback: Rc<dyn MemoExprCallback<Expr = T, Props = T::Props>>) -> Self {
        //TODO: Unit test for a Memo with a callback.
        Memo {
            groups: Vec::new(),
            exprs: Vec::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            callback: Some(callback),
        }
    }

    /// Copies the expression `expr` into this memo.
    pub fn insert(&mut self, expr: T) -> (MemoGroupRef<T>, MemoExprRef<T>) {
        let copy_in = CopyIn {
            memo: self,
            parent: None,
            depth: 0,
        };
        copy_in.execute(&expr)
    }

    /// Copies the expression `expr` into this memo and adds it to the given group.
    pub fn insert_member(&mut self, group: &MemoGroupRef<T>, expr: T) -> (MemoGroupRef<T>, MemoExprRef<T>) {
        let copy_in = CopyIn {
            memo: self,
            parent: Some(group.clone()),
            depth: 0,
        };
        copy_in.execute(&expr)
    }

    fn get_expr_ref(&self, expr_id: &ExprId) -> MemoExprRef<T> {
        assert!(self.exprs.get(expr_id.index()).is_some(), "expr id is invalid: {}", expr_id);
        MemoExprRef::new(*expr_id, self as *const Memo<T>)
    }

    fn get_group_ref(&self, group_id: &GroupId) -> MemoGroupRef<T> {
        assert!(self.groups.get(group_id.index()).is_some(), "group id is invalid: {}", group_id);
        MemoGroupRef::new(*group_id, self as *const Memo<T>)
    }

    fn add_group(&mut self, group: MemoGroupData<T>) -> MemoGroupRef<T> {
        let group_id = group.group_id;
        self.groups.push(group);
        self.get_group_ref(&group_id)
    }

    fn add_expr(&mut self, expr: MemoExprData<T>) -> MemoExprRef<T> {
        let expr_id = expr.expr_id;
        self.exprs.push(expr);
        self.get_expr_ref(&expr_id)
    }

    fn add_expr_to_group(&mut self, group: &MemoGroupRef<T>, expr: MemoExprRef<T>) {
        let group_id = group.id();
        let expr_id = expr.id();
        let memo_group = &mut self.groups[group_id.index()];

        memo_group.exprs.push(expr);
        self.expr_to_group.insert(expr_id, group_id);
    }
}

impl<T> Debug for Memo<T>
where
    T: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memo")
            .field("groups", &self.groups)
            .field("exprs", &self.exprs)
            .field("expr_to_group", &self.expr_to_group)
            .finish()
    }
}

/// A trait that must be implemented by an expression that can be copied into a [`memo`](crate::memo::Memo).
pub trait MemoExpr: Debug + Clone {
    /// The type of the expression.
    type Expr: Expr;
    /// The type of the properties.
    type Props: Properties;

    /// Returns the expression this memo expression stores.
    fn expr(&self) -> &Self::Expr;

    /// Returns properties associated with this expression.
    fn props(&self) -> &Self::Props;

    /// Recursively traverses this expression and copies it into a memo.
    fn copy_in(&self, visitor: &mut CopyInExprs<Self>);

    /// Creates a new expression from this expression by replacing its child expressions with the new ones.
    fn with_new_children(&self, children: NewChildExprs<Self>) -> Self;

    /// Builds a textual representation of this expression.
    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter;
}

/// A trait for the expression type.
pub trait Expr: Debug + Clone {}

/// A trait for properties of the expression.
pub trait Properties: Debug + Clone {}

/// A callback that is called when a new expression is added to a memo.
pub trait MemoExprCallback: Debug {
    /// The type of expression.
    type Expr: MemoExpr;
    /// The type of properties of the expression.
    type Props: Properties;

    /// Called when the given expression `expr` with properties `props` is added to a memo.
    fn new_expr(&self, expr: &Self::Expr, props: Self::Props) -> Self::Props;
}

/// `ExprNode` represents an expression in within an expression tree. A node can be an expression or a memo group.
/// Initially every expression in an expression tree has a direct link its subexpressions (represented by [`ExprNode::Expr`](crate::memo::ExprNode::Expr)).
/// When an expression is copied into a [`memo`](crate::memo::Memo) its subexpressions are replaced with
/// references to memo groups (a reference to a memo group is represented by [`ExprNode::Group`](crate::memo::ExprNode::Group) variant).
#[derive(Clone)]
pub enum ExprNode<T>
where
    T: MemoExpr,
{
    /// A node is another expression.
    Expr(Box<T>),
    /// A node is a memo group.
    Group(MemoGroupRef<T>),
}

impl<T> ExprNode<T>
where
    T: MemoExpr,
{
    /// Returns an expression this `ExprNode` points to.
    /// If this node is an expression this method returns a reference to the actual expression.
    /// If this node is a memo group this method returns a reference to the first memo expression in this memo group.
    pub fn expr(&self) -> &T::Expr {
        match self {
            ExprNode::Expr(expr) => expr.expr(),
            ExprNode::Group(group) => group.expr(),
        }
    }

    /// Returns a reference to properties of an expression this `ExprNode` points to.
    /// If this node is an expression this method returns a reference to the properties of the memo group the expression belongs to.
    /// If this node is a memo group this method returns a reference to the properties of this memo group.
    pub fn props(&self) -> &T::Props {
        match self {
            ExprNode::Expr(expr) => expr.props(),
            ExprNode::Group(group) => group.props(),
        }
    }
}

impl<T> From<T> for ExprNode<T>
where
    T: MemoExpr,
{
    fn from(o: T) -> Self {
        ExprNode::Expr(Box::new(o))
    }
}

impl<T> Debug for ExprNode<T>
where
    T: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprNode::Expr(expr) => write!(f, "Expr({:?})", expr),
            ExprNode::Group(group) => write!(f, "Group({:?})", group.id),
        }
    }
}

/// An [`ExprNode`](crate::memo::ExprNode) that holds a reference to an expression instead of owning it.
#[derive(Debug)]
pub enum ExprNodeRef<'a, T>
where
    T: MemoExpr,
{
    /// A reference is a reference to an expression.
    Expr(&'a T),
    /// A reference is a reference to a memo-group.
    Group(&'a MemoGroupRef<T>),
}

impl<'a, T> From<&'a ExprNode<T>> for ExprNodeRef<'a, T>
where
    T: MemoExpr,
{
    fn from(expr: &'a ExprNode<T>) -> Self {
        match expr {
            ExprNode::Expr(expr) => ExprNodeRef::Expr(&**expr),
            ExprNode::Group(group) => ExprNodeRef::Group(group),
        }
    }
}

/// `MemoGroupRef` is a reference to a memo group. A memo group is a group of logically equivalent expressions.
/// Expressions are logically equivalent when they produce the same result.
///
/// #Safety
/// A reference to a memo group is valid until a [`memo`](crate::memo::Memo) it belongs to is dropped.
pub struct MemoGroupRef<T>
where
    T: MemoExpr,
{
    id: GroupId,
    memo: *const Memo<T>,
}

impl<T> MemoGroupRef<T>
where
    T: MemoExpr,
{
    fn new(id: GroupId, ptr: *const Memo<T>) -> Self {
        MemoGroupRef { id, memo: ptr }
    }

    /// Returns an opaque identifier of this memo group.
    pub fn id(&self) -> GroupId {
        let group = self.get_memo_group();
        group.group_id
    }

    /// Returns a reference to an expression of the first memo expression in this memo group.
    pub fn expr(&self) -> &T::Expr {
        let group = self.get_memo_group();
        group.exprs[0].expr()
    }

    /// Returns a reference to the first memo expression of this memo group.
    pub fn mexpr(&self) -> &MemoExprRef<T> {
        let group = self.get_memo_group();
        &group.exprs[0]
    }

    /// Returns an iterator over the memo expressions that belong to this memo group.
    pub fn mexprs(&self) -> MemoGroupIter<T> {
        let group = self.get_memo_group();
        MemoGroupIter { group, position: 0 }
    }

    /// Returns properties shared by all expressions in this memo group.
    pub fn props(&self) -> &T::Props {
        let group = self.get_memo_group();
        &group.props
    }

    /// Returns the number of expressions in this memo group.
    pub fn len(&self) -> usize {
        let group = self.get_memo_group();
        group.exprs.len()
    }

    fn get_memo_group(&self) -> &MemoGroupData<T> {
        let memo = unsafe {
            // Safety: This pointer is always valid. Because:
            // 1) Lifetime of a Memo is tightly controlled by an instance of the optimizer that created it.
            // 2) Optimized expression never contains InputNode::Group(..).
            &*(self.memo)
        };
        &memo.groups[self.id.index()]
    }
}

impl<T> Clone for MemoGroupRef<T>
where
    T: MemoExpr,
{
    fn clone(&self) -> Self {
        MemoGroupRef::new(self.id, self.memo)
    }
}

impl<T> PartialEq for MemoGroupRef<T>
where
    T: MemoExpr,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.memo == other.memo
    }
}

impl<T> Eq for MemoGroupRef<T> where T: MemoExpr {}

impl<T> Hash for MemoGroupRef<T>
where
    T: MemoExpr,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id.0);
        state.write_usize(self.memo as usize);
    }
}

impl<T> Debug for MemoGroupRef<T>
where
    T: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoGroupRef {{ id: {:?} }}", self.id)
    }
}

/// An iterator over expressions of a memo group.
pub struct MemoGroupIter<'m, T>
where
    T: MemoExpr,
{
    group: &'m MemoGroupData<T>,
    position: usize,
}

impl<'m, T> Iterator for MemoGroupIter<'m, T>
where
    T: MemoExpr,
{
    type Item = &'m MemoExprRef<T>;

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

impl<'m, T> Debug for MemoGroupIter<'m, T>
where
    T: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoGroupIter([")?;
        for i in self.position..self.group.exprs.len() {
            write!(f, "{:?}", self.group.exprs[i])?;
        }
        write!(f, "])")
    }
}

/// A reference to a memo expression.
///
/// #Safety
/// A reference to a memo expression is valid until a [`memo`](crate::memo::Memo) it belongs to is dropped.
pub struct MemoExprRef<T>
where
    T: MemoExpr,
{
    id: ExprId,
    memo: *const Memo<T>,
}

impl<T> MemoExprRef<T>
where
    T: MemoExpr,
{
    fn new(id: ExprId, memo: *const Memo<T>) -> Self {
        MemoExprRef { id, memo }
    }

    /// Returns an opaque identifier of this memo expression.
    pub fn id(&self) -> ExprId {
        let expr = self.get_memo_expr();
        expr.expr_id
    }

    /// Returns the expression this memo expression represents.
    pub fn expr(&self) -> &T::Expr {
        let expr = self.get_memo_expr();
        expr.expr.expr()
    }

    /// Returns the memo expression this reference points to.
    pub fn mexpr(&self) -> &T {
        let expr = self.get_memo_expr();
        &expr.expr
    }

    /// Returns a reference to the memo group this memo expression belongs to.
    pub fn mgroup(&self) -> &MemoGroupRef<T> {
        let expr = self.get_memo_expr();
        &expr.group
    }

    /// Returns references to child expressions of this memo expression.
    pub fn children(&self) -> &[MemoGroupRef<T>] {
        let expr = self.get_memo_expr();
        &expr.children
    }

    fn get_memo_expr(&self) -> &MemoExprData<T> {
        let memo = unsafe {
            // Safety: This pointer is always valid. Because:
            // 1) Lifetime of a Memo is tightly controlled by an instance of the optimizer that created it.
            // 2) Optimized expression never contains InputNode::Group(..).
            &*(self.memo)
        };
        &memo.exprs[self.id.index()]
    }
}

impl<T> Clone for MemoExprRef<T>
where
    T: MemoExpr,
{
    fn clone(&self) -> Self {
        MemoExprRef::new(self.id, self.memo)
    }
}

impl<T> PartialEq for MemoExprRef<T>
where
    T: MemoExpr,
{
    fn eq(&self, other: &Self) -> bool {
        self.memo == other.memo
    }
}

impl<T> Eq for MemoExprRef<T> where T: MemoExpr {}

impl<T> Debug for MemoExprRef<T>
where
    T: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoExprRef {{ id: {:?} }}", self.id)
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
        write!(f, "GroupId({:})", self.0)
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
        write!(f, "{:01}", self.0)
    }
}

impl Debug for ExprId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExprId({:})", self.0)
    }
}

#[derive(Debug)]
struct MemoGroupData<T>
where
    T: MemoExpr,
{
    group_id: GroupId,
    exprs: Vec<MemoExprRef<T>>,
    props: T::Props,
}

impl<T> MemoGroupData<T>
where
    T: MemoExpr,
{
    fn new(group_id: GroupId, props: T::Props) -> Self {
        MemoGroupData {
            group_id,
            exprs: vec![],
            props,
        }
    }
}

#[derive(Debug)]
struct MemoExprData<T>
where
    T: MemoExpr,
{
    expr_id: ExprId,
    expr: T,
    children: Vec<MemoGroupRef<T>>,
    group: MemoGroupRef<T>,
}

impl<T> MemoExprData<T>
where
    T: MemoExpr,
{
    fn new(expr_id: ExprId, expr: T, children: Vec<MemoGroupRef<T>>, group: MemoGroupRef<T>) -> Self {
        MemoExprData {
            expr_id,
            expr,
            children,
            group,
        }
    }
}

/// Provides methods to traverse an expression tree and copy it into a memo.
pub struct CopyInExprs<'a, T>
where
    T: MemoExpr,
{
    memo: &'a mut Memo<T>,
    parent: Option<MemoGroupRef<T>>,
    depth: usize,
    result: Option<(MemoGroupRef<T>, MemoExprRef<T>)>,
}

impl<'a, T> CopyInExprs<'a, T>
where
    T: MemoExpr,
{
    fn new(memo: &'a mut Memo<T>, parent: Option<MemoGroupRef<T>>, depth: usize) -> Self {
        CopyInExprs {
            memo,
            parent,
            depth,
            result: None,
        }
    }

    /// Initialises data structures required to traverse child expressions of the given expression `expr`.
    pub fn enter_expr(&mut self, expr: &T) -> ExprContext<T> {
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
    pub fn visit_expr_node<'e>(&mut self, expr_ctx: &mut ExprContext<T>, expr_node: impl Into<ExprNodeRef<'e, T>>)
    where
        T: 'e,
    {
        let input: ExprNodeRef<T> = expr_node.into();
        match input {
            ExprNodeRef::Expr(expr) => {
                let copy_in = CopyIn {
                    memo: self.memo,
                    parent: None,
                    depth: self.depth + 1,
                };
                let (group, _expr) = copy_in.execute(expr);
                expr_ctx.children.push_back(ExprNode::Group(group));
            }
            ExprNodeRef::Group(group) => {
                expr_ctx.children.push_back(ExprNode::Group(group.clone()));
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
        expr_ctx: &mut ExprContext<T>,
        expr_node: Option<impl Into<ExprNodeRef<'e, T>>>,
    ) where
        T: MemoExpr + 'e,
    {
        if let Some(expr_node) = expr_node {
            self.visit_expr_node(expr_ctx, expr_node);
        };
    }

    /// Copies the expression into the memo.
    pub fn copy_in(&mut self, expr: &T, expr_ctx: ExprContext<T>) {
        let expr_id = ExprId(self.memo.exprs.len());
        let input_groups = expr_ctx
            .children
            .iter()
            .map(|i| match i {
                ExprNode::Group(g) => g.clone(),
                ExprNode::Expr(e) => panic!("Expected a group but got an expression: {:?}", e),
            })
            .collect();

        let ExprContext {
            children,
            props,
            parent,
        } = expr_ctx;

        let expr = expr.with_new_children(NewChildExprs::new(children));
        let digest = make_digest(&expr);

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
                let memo_expr = MemoExprData::new(expr_id, expr, input_groups, parent.clone());
                let expr_ref = self.memo.add_expr(memo_expr);

                self.memo.add_expr_to_group(&parent, expr_ref.clone());

                (parent, expr_ref)
            } else {
                let group_id = GroupId(self.memo.groups.len());
                let props = if let Some(callback) = self.memo.callback.as_ref() {
                    callback.new_expr(&expr, props)
                } else {
                    props
                };
                let memo_group = MemoGroupData::new(group_id, props);
                let group_ref = self.memo.add_group(memo_group);

                let memo_expr = MemoExprData::new(expr_id, expr, input_groups, group_ref.clone());
                let expr_ref = self.memo.add_expr(memo_expr);

                self.memo.add_expr_to_group(&group_ref, expr_ref.clone());

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

/// Stores information used to create a new memo expression.
pub struct ExprContext<T>
where
    T: MemoExpr,
{
    children: VecDeque<ExprNode<T>>,
    parent: Option<MemoGroupRef<T>>,
    props: T::Props,
}

/// A stack-like data structure used by [MemoExpr::with_new_children].
/// It stores new child expressions and provides convenient methods of their retrieval.
#[derive(Debug)]
pub struct NewChildExprs<T>
where
    T: MemoExpr,
{
    children: VecDeque<ExprNode<T>>,
    capacity: usize,
    index: usize,
}

impl<T> NewChildExprs<T>
where
    T: MemoExpr,
{
    /// Creates an instance of `NewChildExprs`.
    pub fn new(children: VecDeque<ExprNode<T>>) -> Self {
        NewChildExprs {
            capacity: children.len(),
            index: 0,
            children,
        }
    }

    /// If the total number of expressions in the underlying stack.
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Returns `true` if the underlying stack is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieves the next child expression.
    ///
    /// # Panics
    ///
    /// This method panics if there is no expression left.
    pub fn expr(&mut self) -> ExprNode<T> {
        self.ensure_available(1);
        self.next_index()
    }

    /// Retrieves the next `n` child expressions.
    ///
    /// # Panics
    ///
    /// This method panics if there is not enough expressions left.
    pub fn exprs(&mut self, n: usize) -> Vec<ExprNode<T>> {
        self.ensure_available(n);
        let mut children = Vec::with_capacity(n);
        for _ in 0..n {
            children.push(self.next_index());
        }
        children
    }

    /// Retrieves the remaining child expressions. If there is no remaining expressions returns an empty `Vec`.
    pub fn remaining(mut self) -> Vec<ExprNode<T>> {
        self.index = self.children.len();
        self.children.into_iter().collect()
    }

    fn next_index(&mut self) -> ExprNode<T> {
        self.children.pop_front().expect("")
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
}

/// Provides methods to collect nested expressions from an expression and copy them into a memo.
/// Can be used to support nested relational expressions inside scalar expressions.
//TODO: Examples
pub struct CopyInNestedExprs<'a, 'c, T>
where
    T: MemoExpr,
{
    ctx: &'c mut CopyInExprs<'a, T>,
    expr_ctx: &'c mut ExprContext<T>,
    nested_exprs: Vec<MemoGroupRef<T>>,
}

impl<'a, 'c, T> CopyInNestedExprs<'a, 'c, T>
where
    T: MemoExpr,
{
    /// Creates a new nested expression collector.
    pub fn new(ctx: &'c mut CopyInExprs<'a, T>, expr_ctx: &'c mut ExprContext<T>) -> Self {
        CopyInNestedExprs {
            ctx,
            expr_ctx,
            nested_exprs: Vec::new(),
        }
    }

    /// Traverses the given expression `expr` of some arbitrary type.
    /// When traversal completes all collected nested expressions are copies into a memo.
    pub fn execute<F, E>(mut self, expr: &E, f: F)
    where
        F: Fn(&E, &mut Self),
    {
        (f)(expr, &mut self);
        // Visit collected nested expressions so that they will be added to the given expression as child expressions.
        for input in self.nested_exprs {
            self.ctx.visit_expr_node(self.expr_ctx, &ExprNode::Group(input));
        }
    }

    /// Copies the given nested expression into a memo.
    pub fn visit_expr(&mut self, expr: ExprNodeRef<T>) {
        match expr {
            ExprNodeRef::Expr(expr) => {
                expr.copy_in(self.ctx);
                let (group, _) = std::mem::take(&mut self.ctx.result).expect("Failed to copy in a nested expressions");
                self.add_group(group);
            }
            ExprNodeRef::Group(group) => self.add_group(group.clone()),
        }
    }

    fn add_group(&mut self, group: MemoGroupRef<T>) {
        self.nested_exprs.push(group)
    }
}

struct CopyIn<'a, T>
where
    T: MemoExpr,
{
    memo: &'a mut Memo<T>,
    parent: Option<MemoGroupRef<T>>,
    depth: usize,
}

impl<'a, T> CopyIn<'a, T>
where
    T: MemoExpr,
{
    fn execute(self, expr: &T) -> (MemoGroupRef<T>, MemoExprRef<T>) {
        let mut ctx = CopyInExprs::new(self.memo, self.parent.clone(), self.depth);
        expr.copy_in(&mut ctx);
        ctx.result.unwrap()
    }
}

/// Builds a textual representation of the given memo.
pub(crate) fn format_memo<T>(memo: &Memo<T>) -> String
where
    T: MemoExpr,
{
    let mut buf = String::new();
    let mut f = StringMemoFormatter::new(&mut buf);

    for group in memo.groups.iter().rev() {
        f.push_str(format!("{} ", group.group_id).as_str());
        for (i, expr) in group.exprs.iter().enumerate() {
            if i > 0 {
                // newline + 3 spaces
                f.push_str("\n   ");
            }
            expr.mexpr().format_expr(&mut f);
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
    fn write_expr<'e, T>(&mut self, name: &str, input: impl Into<ExprNodeRef<'e, T>>)
    where
        T: MemoExpr + 'e;

    /// Writes a child expression it is present. This method is equivalent to:
    /// ```text
    /// if let Some(expr) = expr {
    ///   fmt.write_expr(name, expr);
    /// }
    /// ```
    fn write_expr_if_present<'e, T>(&mut self, name: &str, expr: Option<impl Into<ExprNodeRef<'e, T>>>)
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

    fn write_expr<'e, T>(&mut self, name: &str, input: impl Into<ExprNodeRef<'e, T>>)
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

impl<T> Display for MemoExprRef<T>
where
    T: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let expr = self.get_memo_expr();
        write!(f, "[{} ", expr.expr_id)?;
        let mut fmt = DisplayMemoExprFormatter { fmt: f };
        expr.expr.format_expr(&mut fmt);
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

    fn write_expr<'e, T>(&mut self, _name: &str, input: impl Into<ExprNodeRef<'e, T>>)
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

fn format_node_ref<T>(input: ExprNodeRef<'_, T>) -> String
where
    T: MemoExpr,
{
    match input {
        ExprNodeRef::Expr(expr) => {
            // This only happens when expression has not been added to a memo.
            let ptr: *const T::Expr = &*expr.expr();
            format!("{:?}", ptr)
        }
        ExprNodeRef::Group(group) => format!("{}", group.id()),
    }
}

fn make_digest<T>(expr: &T) -> String
where
    T: MemoExpr,
{
    let mut buf = String::new();
    let mut fmt = StringMemoFormatter::new(&mut buf);
    expr.format_expr(&mut fmt);
    buf
}

#[cfg(test)]
mod test {
    use super::*;
    use std::cell::RefCell;

    #[derive(Debug, Clone)]
    enum TestExpr {
        Relational(TestRelExpr),
        Scalar(TestScalarExpr),
    }

    #[derive(Debug, Clone)]
    enum TestRelExpr {
        Leaf(&'static str),
        Node { input: RelExpr },
        Nodes { inputs: Vec<RelExpr> },
        Filter { input: RelExpr, filter: ScalarExpr },
    }

    impl Expr for TestExpr {}

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

    #[derive(Debug, Clone)]
    enum RelExpr {
        Expr(Box<TestOperator>),
        Group(MemoGroupRef<TestOperator>),
    }

    impl RelExpr {
        fn expr(&self) -> &TestRelExpr {
            let expr = match self {
                RelExpr::Expr(expr) => expr.expr(),
                RelExpr::Group(group) => group.expr(),
            };
            expr.as_relational()
        }

        fn props(&self) -> &TestProps {
            match self {
                RelExpr::Expr(expr) => expr.props(),
                RelExpr::Group(group) => group.props(),
            }
        }

        // replace with From<Self> -> InputNodeRef ?
        fn get_ref(&self) -> ExprNodeRef<TestOperator> {
            match self {
                RelExpr::Expr(expr) => ExprNodeRef::Expr(&**expr),
                RelExpr::Group(group) => ExprNodeRef::Group(group),
            }
        }
    }

    #[derive(Debug, Clone)]
    enum ScalarExpr {
        Expr(Box<TestOperator>),
        Group(MemoGroupRef<TestOperator>),
    }

    impl ScalarExpr {
        // replace with From<Self> -> InputNodeRef ?
        fn get_ref(&self) -> ExprNodeRef<TestOperator> {
            match self {
                ScalarExpr::Expr(expr) => ExprNodeRef::Expr(&**expr),
                ScalarExpr::Group(group) => ExprNodeRef::Group(group),
            }
        }
    }

    impl From<TestOperator> for RelExpr {
        fn from(expr: TestOperator) -> Self {
            RelExpr::Expr(Box::new(expr))
        }
    }

    impl From<ExprNode<TestOperator>> for RelExpr {
        fn from(expr: ExprNode<TestOperator>) -> Self {
            match expr {
                ExprNode::Expr(expr) => RelExpr::Expr(expr),
                ExprNode::Group(group) => RelExpr::Group(group),
            }
        }
    }

    impl From<TestScalarExpr> for ScalarExpr {
        fn from(expr: TestScalarExpr) -> Self {
            ScalarExpr::Expr(Box::new(TestOperator {
                expr: TestExpr::Scalar(expr),
                props: Default::default(),
            }))
        }
    }

    impl From<ExprNode<TestOperator>> for ScalarExpr {
        fn from(expr: ExprNode<TestOperator>) -> Self {
            match expr {
                ExprNode::Expr(expr) => ScalarExpr::Expr(expr),
                ExprNode::Group(group) => ScalarExpr::Group(group),
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
        SubQuery(RelExpr),
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
                TestScalarExpr::SubQuery(_) => TestScalarExpr::SubQuery(RelExpr::from(inputs.expr())),
            }
        }

        fn copy_in_nested(&self, collector: &mut CopyInNestedExprs<TestOperator>) {
            match self {
                TestScalarExpr::Value(_) => {}
                TestScalarExpr::Gt { lhs, rhs } => {
                    lhs.copy_in_nested(collector);
                    rhs.copy_in_nested(collector);
                }
                TestScalarExpr::SubQuery(expr) => {
                    collector.visit_expr(expr.get_ref());
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
                TestScalarExpr::SubQuery(query) => match query {
                    RelExpr::Expr(expr) => {
                        let ptr: *const TestExpr = expr.expr();
                        write!(f, "SubQuery expr_ptr {:?}", ptr)
                    }
                    RelExpr::Group(group) => {
                        write!(f, "SubQuery {}", group.id())
                    }
                },
            }
        }
    }

    struct TraversalWrapper<'c, 'm> {
        ctx: &'c mut CopyInExprs<'m, TestOperator>,
    }

    impl TraversalWrapper<'_, '_> {
        fn enter_expr(&mut self, expr: &TestOperator) -> ExprContext<TestOperator> {
            self.ctx.enter_expr(expr)
        }

        fn visit_rel(&mut self, expr_ctx: &mut ExprContext<TestOperator>, rel: &RelExpr) {
            self.ctx.visit_expr_node(expr_ctx, rel.get_ref());
        }

        fn visit_scalar(&mut self, expr_ctx: &mut ExprContext<TestOperator>, scalar: &ScalarExpr) {
            self.ctx.visit_expr_node(expr_ctx, scalar.get_ref());
        }

        fn copy_in_nested(&mut self, expr_ctx: &mut ExprContext<TestOperator>, expr: &TestOperator) {
            let nested_ctx = CopyInNestedExprs::new(self.ctx, expr_ctx);
            let scalar = expr.expr().as_scalar();
            nested_ctx.execute(scalar, |expr, collect: &mut CopyInNestedExprs<TestOperator>| {
                expr.copy_in_nested(collect);
            })
        }

        fn copy_in(self, expr: &TestOperator, expr_ctx: ExprContext<TestOperator>) {
            self.ctx.copy_in(expr, expr_ctx)
        }
    }

    #[derive(Debug, Clone)]
    struct TestOperator {
        expr: TestExpr,
        props: TestProps,
    }

    impl From<TestRelExpr> for TestOperator {
        fn from(expr: TestRelExpr) -> Self {
            TestOperator {
                expr: TestExpr::Relational(expr),
                props: TestProps::default(),
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Default)]
    struct TestProps {
        a: i32,
    }

    impl Properties for TestProps {}

    impl TestOperator {
        fn with_props(self, a: i32) -> Self {
            TestOperator {
                expr: self.expr,
                props: TestProps { a },
            }
        }
    }

    impl MemoExpr for TestOperator {
        type Expr = TestExpr;
        type Props = TestProps;

        fn expr(&self) -> &Self::Expr {
            &self.expr
        }

        fn props(&self) -> &Self::Props {
            &self.props
        }

        fn copy_in(&self, ctx: &mut CopyInExprs<Self>) {
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

        fn with_new_children(&self, mut inputs: NewChildExprs<Self>) -> Self {
            let expr = match self.expr() {
                TestExpr::Relational(expr) => {
                    let expr = match expr {
                        TestRelExpr::Leaf(s) => TestRelExpr::Leaf(s.clone()),
                        TestRelExpr::Node { .. } => TestRelExpr::Node {
                            input: RelExpr::from(inputs.expr()),
                        },
                        TestRelExpr::Nodes { .. } => TestRelExpr::Nodes {
                            inputs: inputs.remaining().into_iter().map(RelExpr::from).collect(),
                        },
                        TestRelExpr::Filter { .. } => TestRelExpr::Filter {
                            input: RelExpr::from(inputs.expr()),
                            filter: ScalarExpr::from(inputs.expr()),
                        },
                    };
                    TestExpr::Relational(expr)
                }
                TestExpr::Scalar(expr) => TestExpr::Scalar(expr.with_new_inputs(&mut inputs)),
            };
            TestOperator {
                expr,
                props: self.props.clone(),
            }
        }

        fn format_expr<F>(&self, f: &mut F)
        where
            F: MemoExprFormatter,
        {
            match self.expr() {
                TestExpr::Relational(expr) => match expr {
                    TestRelExpr::Leaf(source) => {
                        f.write_name("Leaf");
                        f.write_source(source);
                    }
                    TestRelExpr::Node { input } => {
                        f.write_name("Node");
                        f.write_expr("", input.get_ref());
                    }
                    TestRelExpr::Nodes { inputs } => {
                        f.write_name("Nodes");
                        for input in inputs {
                            f.write_expr("", input.get_ref());
                        }
                    }
                    TestRelExpr::Filter { input, filter } => {
                        f.write_name("Filter");
                        f.write_expr("", input.get_ref());
                        f.write_expr("filter", filter.get_ref());
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
        let mut memo = Memo::new();
        let expr = TestOperator::from(TestRelExpr::Leaf("aaaa")).with_props(100);
        let (group, expr) = memo.insert(expr);

        assert_eq!(group.props(), &TestProps { a: 100 }, "group properties");
        assert_eq!(expr.mgroup(), &group, "expr group");

        let group_by_id = memo.get_group_ref(&group.id());
        assert_eq!(group, group_by_id);

        let expr_by_id = memo.get_expr_ref(&expr.id());
        assert_eq!(expr, expr_by_id);

        assert_eq!(memo.groups.len(), 1, "group num");
        assert_eq!(memo.exprs.len(), 1, "expr num");
        assert_eq!(memo.expr_cache.len(), 1, "expr cache size");
    }

    #[test]
    fn test_properties() {
        let mut memo = Memo::new();

        let leaf = TestOperator::from(TestRelExpr::Leaf("a")).with_props(10);
        let inner = TestOperator::from(TestRelExpr::Node { input: leaf.into() }).with_props(15);
        let expr = TestOperator::from(TestRelExpr::Node { input: inner.into() });
        let outer = TestOperator::from(TestRelExpr::Node { input: expr.into() }).with_props(20);

        let _ = memo.insert(outer);

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
        let mut memo = Memo::new();

        let leaf = TestOperator::from(TestRelExpr::Leaf("a")).with_props(10);
        let expr = TestOperator::from(TestRelExpr::Node {
            input: leaf.clone().into(),
        });
        let (_, expr) = memo.insert(expr);

        match expr.expr().as_relational() {
            TestRelExpr::Node { input } => {
                assert_eq!(
                    format!("{:?}", input.expr()),
                    format!("{:?}", expr.children()[0].expr().as_relational()),
                    "child expr"
                );
                assert_eq!(input.props(), leaf.props(), "input props");
            }
            _ => panic!("Unexpected expression: {:?}", expr),
        }
    }

    #[test]
    fn test_memo_trivial_expr() {
        let mut memo = Memo::new();

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let (group1, expr1) = memo.insert(expr.clone());
        let (group2, expr2) = memo.insert(expr);

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
        let mut memo = Memo::new();

        let expr = TestOperator::from(TestRelExpr::Node {
            input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
        });

        let (group1, expr1) = memo.insert(expr.clone());
        let (group2, expr2) = memo.insert(expr);

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
        let mut memo = Memo::new();

        let expr = TestOperator::from(TestRelExpr::Nodes {
            inputs: vec![
                TestOperator::from(TestRelExpr::Leaf("a")).into(),
                TestOperator::from(TestRelExpr::Leaf("b")).into(),
            ],
        });

        let (group1, expr1) = memo.insert(expr.clone());
        let (group2, expr2) = memo.insert(expr);

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
        let mut memo = Memo::new();

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

        let (group1, expr1) = memo.insert(expr.clone());
        let (group2, expr2) = memo.insert(expr);

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
        let mut memo = Memo::new();

        let expr = TestOperator::from(TestRelExpr::Nodes {
            inputs: vec![
                TestOperator::from(TestRelExpr::Leaf("a")).into(),
                TestOperator::from(TestRelExpr::Leaf("b")).into(),
                TestOperator::from(TestRelExpr::Leaf("a")).into(),
            ],
        });

        let (group1, expr1) = memo.insert(expr.clone());
        let (group2, expr2) = memo.insert(expr);

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
        let mut memo = Memo::new();

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));

        let (group, _) = memo.insert(expr);
        expect_memo(
            &memo,
            r#"
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestRelExpr::Leaf("a0"));
        let _ = memo.insert_member(&group, expr);

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
        let mut memo = Memo::new();

        let expr = TestOperator::from(TestRelExpr::Node {
            input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
        });

        let (group, _) = memo.insert(expr);
        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestRelExpr::Leaf("a0"));
        let _ = memo.insert_member(&group.mexpr().children()[0], expr);

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
        let mut memo = Memo::new();

        let expr = TestOperator::from(TestRelExpr::Node {
            input: TestOperator::from(TestRelExpr::Leaf("a")).into(),
        });

        let (group, _) = memo.insert(expr);
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
        let _ = memo.insert_member(&group, expr);

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
        let mut memo = Memo::new();
        assert_eq!(format!("{:?}", memo), "Memo { groups: [], exprs: [], expr_to_group: {} }");

        let expr = TestOperator::from(TestRelExpr::Leaf("a"));
        let (group, expr) = memo.insert(expr);

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
            expr: TestExpr::Scalar(expr),
            props: Default::default(),
        };

        let mut memo = Memo::new();
        let _ = memo.insert(expr);

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

        let mut memo = Memo::new();
        let (group, _) = memo.insert(expr);

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

        let children = group.mexpr().children();
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

        let mut memo = Memo::new();
        let (_, _) = memo.insert(expr);

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
        impl MemoExprCallback for Callback {
            type Expr = TestOperator;
            type Props = TestProps;

            fn new_expr(&self, expr: &Self::Expr, props: Self::Props) -> Self::Props {
                let mut added = self.added.borrow_mut();
                let mut buf = String::new();
                let mut fmt = StringMemoFormatter::new(&mut buf);
                expr.format_expr(&mut fmt);
                added.push(buf);
                props
            }
        }

        let added = Rc::new(RefCell::new(vec![]));
        let callback = Callback { added: added.clone() };
        let mut memo = Memo::with_callback(Rc::new(callback));

        memo.insert(TestOperator::from(TestRelExpr::Leaf("A")));

        {
            let added = added.borrow();
            assert_eq!(added[0], "Leaf A");
            assert_eq!(added.len(), 1);
        }

        let leaf = TestOperator::from(TestRelExpr::Leaf("B"));
        let expr = TestOperator::from(TestRelExpr::Node { input: leaf.into() });
        memo.insert(expr.clone());

        {
            let added = added.borrow();
            assert_eq!(added[1], "Leaf B");
            assert_eq!(added[2], "Node 01");
            assert_eq!(added.len(), 3);
        }

        memo.insert(expr);
        {
            let added = added.borrow();
            assert_eq!(added.len(), 3, "added duplicates. memo:\n{}", format_test_memo(&memo, false));
        }
    }

    fn expect_group_size(memo: &Memo<TestOperator>, group_id: &GroupId, size: usize) {
        let group = &memo.groups[group_id.index()];
        assert_eq!(group.exprs.len(), size, "group#{}", group_id);
    }

    fn expect_group_exprs(group: &MemoGroupRef<TestOperator>, expected: Vec<MemoExprRef<TestOperator>>) {
        let actual: Vec<MemoExprRef<TestOperator>> = group.mexprs().cloned().collect();
        assert_eq!(actual, expected, "group#{} exprs", group.id());
    }

    fn expect_memo(memo: &Memo<TestOperator>, expected: &str) {
        let lines: Vec<String> = expected.split('\n').map(String::from).collect();
        let expected = lines.join("\n");

        let buf = format_test_memo(memo, false);
        assert_eq!(buf.trim(), expected.trim());
    }

    fn expect_memo_with_props(memo: &Memo<TestOperator>, expected: &str) {
        let lines: Vec<String> = expected.split('\n').map(String::from).collect();
        let expected = lines.join("\n");

        let buf = format_test_memo(memo, true);
        assert_eq!(buf.trim(), expected.trim());
    }

    fn format_test_memo(memo: &Memo<TestOperator>, include_props: bool) -> String {
        let mut buf = String::new();
        let mut f = StringMemoFormatter::new(&mut buf);
        for group in memo.groups.iter().rev() {
            f.push_str(format!("{} ", group.group_id).as_str());
            for (i, expr) in group.exprs.iter().enumerate() {
                if i > 0 {
                    // new line + 3 spaces
                    f.push_str("\n   ");
                    expr.mexpr().format_expr(&mut f);
                } else {
                    expr.mexpr().format_expr(&mut f);
                    if include_props && group.props != TestProps::default() {
                        f.write_value("a", &group.props.a)
                    }
                }
            }
            f.push('\n');
        }
        buf
    }
}
