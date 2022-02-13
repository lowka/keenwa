use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use itertools::Itertools;
use triomphe::Arc;

pub use memo_impl::*;

// DEFAULT MEMO

#[cfg(feature = "default_memo")]
#[cfg(not(feature = "unsafe_memo"))]
#[cfg(not(feature = "docs"))]
pub mod default_impl;

#[cfg(feature = "default_memo")]
#[cfg(not(feature = "unsafe_memo"))]
mod memo_impl {
    // crate imports
    #[doc(hidden)]
    pub(crate) use default_impl::{copy_in_expr_impl, format_memo_impl};
    // pub imports
    #[doc(hidden)]
    pub use default_impl::{
        ExprId, GroupId, MemoExprRef, MemoExprState, MemoGroupRef, MemoGroupToken, MemoImpl, MemoizedExpr,
    };

    // hide re-exports
    #[doc(hidden)]
    pub use super::default_impl;
}

// UNSAFE MEMO

#[cfg(feature = "unsafe_memo")]
#[cfg(not(feature = "docs"))]
pub mod unsafe_impl;

#[cfg(feature = "unsafe_memo")]
#[cfg(not(feature = "docs"))]
mod memo_impl {
    // crate imports
    #[doc(hidden)]
    pub(crate) use unsafe_impl::{copy_in_expr_impl, format_memo_impl};
    // pub imports
    #[doc(hidden)]
    pub use unsafe_impl::{
        ExprId, GroupId, MemoExprRef, MemoExprState, MemoGroupRef, MemoGroupToken, MemoImpl, MemoizedExpr,
    };

    // hide re-exports
    #[doc(hidden)]
    pub use super::unsafe_impl;
}

// DOCS ONLY

#[cfg(any(feature = "docs"))]
pub mod default_impl;
#[cfg(any(feature = "docs"))]
pub mod unsafe_impl;
// We need to duplicate memo_impl because docs are built with --all-features flag
// and *_memo features are mutually exclusive.
// There must be a way to do this without duplicating the entire mod memo_impl though.
#[cfg(feature = "docs")]
mod memo_impl {
    // crate imports
    #[doc(hidden)]
    pub(crate) use default_impl::{copy_in_expr_impl, format_memo_impl};
    // pub imports
    #[doc(hidden)]
    pub use default_impl::{
        ExprId, GroupId, MemoExprRef, MemoExprState, MemoGroupRef, MemoGroupToken, MemoImpl, MemoizedExpr,
    };

    // hide re-exports
    #[doc(hidden)]
    pub use super::default_impl;
}
// DOCS ONLY ENDS

// testing
#[cfg(test)]
pub mod testing;
#[cfg(test)]
mod tests;

/// A builder to construct an instance of a [memo](self::Memo).
pub struct MemoBuilder<E, T>
where
    E: MemoExpr,
{
    callback: Option<MemoGroupCallbackRef<E::Expr, E::Props, T>>,
    metadata: T,
}

impl<E, T> MemoBuilder<E, T>
where
    E: MemoExpr,
{
    /// Creates a new builder with the given metadata.
    pub fn new(metadata: T) -> Self {
        MemoBuilder {
            metadata,
            callback: None,
        }
    }

    /// Sets a callback.
    pub fn set_callback(mut self, callback: MemoGroupCallbackRef<E::Expr, E::Props, T>) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Build a new instance of a [memo](self::Memo).
    pub fn build(self) -> Memo<E, T> {
        if let Some(callback) = self.callback {
            let memo_impl = MemoImpl::with_callback(self.metadata, callback);
            Memo { memo_impl }
        } else {
            let memo_impl = MemoImpl::new(self.metadata);
            Memo { memo_impl }
        }
    }
}

/// The type of [memo group callback][self::MemoGroupCallback] used by [Memo].
pub type MemoGroupCallbackRef<E, P, T> = Rc<dyn MemoGroupCallback<Expr = E, Props = P, Metadata = T>>;

/// `Memo` is the primary data structure used by the cost-based optimizer:
///  * It stores each expression as a group of logically equivalent expressions.
///  * It provides memoization of identical subexpressions within an expression tree.
// TODO: Examples
// TODO: add generic implementation of a copy-out method.
//  MemoExprRef, MemoGroupRef
pub struct Memo<E, T>
where
    E: MemoExpr,
{
    #[cfg(feature = "default_memo")]
    #[cfg(not(feature = "unsafe_memo"))]
    memo_impl: MemoImpl<E, T>,
    #[cfg(feature = "unsafe_memo")]
    memo_impl: MemoImpl<E, T>,
}

impl<E, T> Memo<E, T>
where
    E: MemoExpr,
{
    /// Copies the given expression `expr` into this memo. if this memo does not contain the given expression
    /// a new memo group is created and this method returns a reference to it. Otherwise returns a reference
    /// to the already existing expression.
    pub fn insert_group(&mut self, expr: E) -> E {
        self.memo_impl.insert_group(expr)
    }

    /// Copies the expression `expr` into this memo and adds it to the memo group that granted the given [membership token](MemoGroupToken).
    /// If an identical expression already exist this method simply returns a reference to that expression.
    ///
    /// # Panics
    ///
    /// This method panics if group with the given id does not exist.
    pub fn insert_group_member(&mut self, token: MemoGroupToken<E>, expr: E) -> E {
        self.memo_impl.insert_group_member(token, expr)
    }

    /// Return a reference to a memo group with the given id or panics if it does not exists.
    ///
    /// # Panics
    ///
    /// This method panics if group with the given id does not exist.
    pub fn get_group(&self, group_id: &GroupId) -> MemoGroupRef<E, T> {
        self.memo_impl.get_group(group_id)
    }

    /// Returns a reference to the metadata.
    pub fn metadata(&self) -> &T {
        self.memo_impl.metadata()
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
            .field("num_groups", &self.memo_impl.num_groups())
            .field("num_exprs", &self.memo_impl.num_exprs())
            .field("expr_to_group", self.memo_impl.expr_to_group())
            .field("metadata", self.memo_impl.metadata())
            .finish()
    }
}

/// A trait that must be implemented by an expression that can be copied into a [`memo`](self::Memo).
// TODO: Docs.
pub trait MemoExpr: Clone {
    /// The type of the expression.
    type Expr: Expr;
    /// The type of the properties.
    type Props: Props;

    /// Returns a reference to the underlying expression (see [MemoExprState::expr]).
    /// This method is a shorthand for `memo_expr.state().expr()`.
    fn expr(&self) -> &Self::Expr {
        self.state().expr()
    }

    /// Returns properties associated with this memo expression (see [MemoExprState::props]).
    /// This method is a shorthand for `memo_expr.state().props()`.
    fn props(&self) -> &Self::Props {
        self.state().props()
    }

    /// Creates a new memo expression from the given state.
    fn from_state(state: MemoExprState<Self>) -> Self;

    /// Returns a references to the state of the this memo expression.
    fn state(&self) -> &MemoExprState<Self>;

    /// Recursively traverses this memo expression and copies it into a memo.
    fn copy_in<T>(&self, visitor: &mut CopyInExprs<Self, T>);

    /// Creates a new expression from the given expression `expr` by replacing its child expressions
    /// with expressions provided by the given [NewChildExprs](self::NewChildExprs).
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

    /// Returns an iterator over child expressions of this memo expression.
    fn children(&self) -> MemoExprChildIter<Self> {
        MemoExprChildIter {
            expr: self,
            position: 0,
        }
    }

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

/// A trait for properties of a memo group.
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

    /// Consumes these properties and returns the underlying relational properties or panics if these properties are not relational.
    ///
    /// # Panics
    ///
    /// This method panics if the underlying properties are not relational.
    fn to_relational(self) -> Self::RelProps;

    /// Consumes these properties and returns the underlying scalar properties or panics if these properties are not scalar.
    ///
    /// # Panics
    ///
    /// This method panics if the underlying properties are not scalar.
    fn to_scalar(self) -> Self::ScalarProps;
}

/// Callback that is called when a new memo group is added to a [memo](self::Memo).
pub trait MemoGroupCallback {
    /// The type of expression.
    type Expr: Expr;
    /// The type of properties of the expression.
    type Props: Props;
    /// The type of metadata stored by a memo.
    type Metadata;

    /// Called when a new memo group is added to a memo and returns properties to be shared among all expressions in this group.
    /// Where `expr` is the expression that created the memo group and `props` are properties associated with that expression.
    fn new_group(&self, expr: &Self::Expr, props: &Self::Props, metadata: &Self::Metadata) -> Self::Props;
}

// It should be possible to use the same MemoExprState for both safe_memo and unsafe_memo in the future
// but rustc 1.57.0 (f1edd0429 2021-11-29) panics when compiles safe_memo::MemoizedExpr::get_expr_ref
//
// /// Represents a state of a memo expression.
// ///
// /// In the `Owned` state a memo expression stores expression and properties.
// /// In the `Memo` state a memo expression stores a reference to expression stored in a memo.
// #[derive(Clone)]
// pub enum MemoExprState<E>
// where
//     E: MemoExpr,
// {
//     Owned(OwnedExpr<E>),
//     Memo(MemoizedExpr<E>),
// }
//
// impl<E> MemoExprState<E>
// where
//     E: MemoExpr,
// {
//     /// Creates an owned state with the given expression and properties.
//     pub fn new(expr: E::Expr, props: E::Props) -> Self {
//         MemoExprState::Owned(OwnedExpr {
//             expr: Arc::new(expr),
//             props: Arc::new(props),
//         })
//     }
//
//     /// Returns a reference to an expression.
//     /// * In the owned state this method returns a reference to the underlying expression.
//     /// * In the memo state this method returns a reference to the first expression in the [memo group][self::MemoGroupRef].
//     pub fn expr(&self) -> &E::Expr {
//         match self {
//             MemoExprState::Owned(state) => state.expr.as_ref(),
//             MemoExprState::Memo(state) => &state.expr(),
//         }
//     }
//
//     /// Returns a reference to properties.
//     ///
//     /// * In the owned state this method returns a reference to the underlying properties.
//     /// * In the memo state this method returns a reference to properties of a memo group the expression belongs to.
//     pub fn props(&self) -> &E::Props {
//         match self {
//             MemoExprState::Owned(state) => state.props.as_ref(),
//             MemoExprState::Memo(state) => &state.props(),
//         }
//     }
//
//     /// Returns `true` if this `MemoExprState` is in the `Memo` state.
//     pub fn is_memo(&self) -> bool {
//         match self {
//             MemoExprState::Owned(_) => false,
//             MemoExprState::Memo(_) => true,
//         }
//     }
//
//     /// Returns a reference to a memo expression or panics if this `MemoExprState` is not in the memo state.
//     /// This method should only be called on memoized expressions.
//     ///
//     /// # Panics
//     ///
//     /// This method panics if this `MemoExprState` is in the owned state.
//     pub fn memo_expr(&self) -> MemoExprRef<E> {
//         match self {
//             MemoExprState::Owned(_) => panic!("This should only be called on memoized expressions"),
//             MemoExprState::Memo(state) => get_expr_ref(state, self),
//         }
//     }
//
//     /// Returns an identifier of a memo group or panics if this `MemoExprState` is not in the `Memo` state.
//     /// This method should only be called on memoized expressions.
//     ///
//     /// # Panics
//     ///
//     /// This method panics if this `MemoExprState` is in the `Owned` state.
//     pub fn memo_group_id(&self) -> GroupId {
//         match self {
//             MemoExprState::Owned(_) => panic!("This should only be called on memoized expressions"),
//             MemoExprState::Memo(state) => state.group_id(),
//         }
//     }
//
//     pub(crate) fn equal(&self, other: &Self) -> bool {
//         let this = self.get_eq_state();
//         let that = other.get_eq_state();
//         // Expressions are equal if their expr pointers point to the same expression.
//         match (this, that) {
//             // Raw pointer equality for owned expressions is used here because we should not check non-memoized
//             // expressions for equality.
//             ((Some(x), _), (Some(y), _)) => std::ptr::eq(x, y),
//             ((_, Some(x)), (_, Some(y))) => x == y,
//             _ => false,
//         }
//     }
//
//     pub(crate) fn hash<H: Hasher>(&self, state: &mut H) {
//         match self {
//             MemoExprState::Owned(s) => {
//                 let ptr: *const E::Expr = s.expr.as_ptr();
//                 ptr.hash(state)
//             }
//             MemoExprState::Memo(s) => s.expr_id().hash(state),
//         }
//     }
//
//     fn get_eq_state(&self) -> (Option<*const E::Expr>, Option<ExprId>)
//     where
//         E: MemoExpr,
//     {
//         match self {
//             MemoExprState::Owned(state) => {
//                 let ptr: *const E::Expr = state.expr.as_ptr();
//                 (Some(ptr), None)
//             }
//             MemoExprState::Memo(state) => (None, Some(state.expr_id())),
//         }
//     }
// }

// impl<E, T, P> Debug for MemoExprState<E>
// where
//     E: MemoExpr<Expr = T, Props = P> + Debug,
//     T: Expr + Debug,
//     P: Props + Debug,
// {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         match self {
//             MemoExprState::Owned(state) => f.debug_tuple("Owned").field(state).finish(),
//             MemoExprState::Memo(state) => f.debug_tuple("Memo").field(state).finish(),
//         }
//     }
// }

/// Represents an expression that has not been copied into a [memo](self::Memo).
/// This is the `owned` state of a [MemoExprState](self::MemoExprState).
#[derive(Clone)]
pub struct OwnedExpr<E>
where
    E: MemoExpr,
{
    pub(crate) expr: Arc<E::Expr>,
    pub(crate) props: Arc<E::Props>,
}

impl<E> OwnedExpr<E>
where
    E: MemoExpr,
{
    /// Creates a copy of this expression that uses the given properties.
    pub fn with_props(&self, props: E::Props) -> Self {
        OwnedExpr {
            expr: self.expr.clone(),
            props: Arc::new(props),
        }
    }
}

impl<E, T, P> Debug for OwnedExpr<E>
where
    E: MemoExpr<Expr = T, Props = P>,
    T: Expr + Debug,
    P: Props + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedExpr")
            .field("expr", self.expr.as_ref())
            .field("props", &self.props.as_ref())
            .finish()
    }
}

/// An iterator over child expressions of a [memo expression](self::MemoExpr).
pub struct MemoExprChildIter<'a, E> {
    expr: &'a E,
    position: usize,
}

impl<'a, E> Iterator for MemoExprChildIter<'a, E>
where
    E: MemoExpr,
{
    type Item = &'a E;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(expr) = self.expr.get_child(self.position) {
            self.position += 1;
            Some(expr)
        } else {
            None
        }
    }
}

impl<'a, E> ExactSizeIterator for MemoExprChildIter<'a, E>
where
    E: MemoExpr,
{
    fn len(&self) -> usize {
        self.expr.num_children()
    }
}

impl<'a, E> Debug for MemoExprChildIter<'a, E>
where
    E: MemoExpr + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let iter = MemoExprChildIter {
            expr: self.expr,
            position: self.position,
        };
        f.debug_list().entries(iter).finish()
    }
}

/// Provides methods to build a textual representation of a memo expression.
pub trait MemoExprFormatter {
    /// Writes a name of a memo expression.
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

    /// Writes child expressions.
    fn write_exprs<T>(&mut self, name: &str, input: impl ExactSizeIterator<Item = impl AsRef<T>>)
    where
        T: MemoExpr;

    /// Writes a value of some attribute of a memo expression.
    fn write_value<D>(&mut self, name: &str, value: D)
    where
        D: Display;

    /// Writes values of some attribute of a memo expression.
    fn write_values<D>(&mut self, name: &str, values: &[D])
    where
        D: Display;
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
        let state = MemoExprState::new(T::new_rel(expr), P::new_rel(props));
        let expr = E::from_state(state);
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
    pub fn from_group<M>(group: MemoGroupRef<'_, E, M>) -> Self {
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
        self.0.expr().relational()
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

    /// Returns the [state](self::MemoExprState) of the underlying memo expression.
    pub fn state(&self) -> &MemoExprState<E> {
        self.0.state()
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
        self.0.state().equal(other.0.state())
    }
}

impl<E> Eq for RelNode<E> where E: MemoExpr {}

impl<E> Hash for RelNode<E>
where
    E: MemoExpr,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.state().hash(state)
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
        let state = MemoExprState::new(T::new_scalar(expr), P::new_scalar(props));
        let expr = E::from_state(state);
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
    pub fn from_group<M>(group: MemoGroupRef<'_, E, M>) -> Self {
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
        self.0.expr().scalar()
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

    /// Returns the [state](self::MemoExprState) of the underlying memo expression.
    pub fn state(&self) -> &MemoExprState<E> {
        self.0.state()
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
        self.0.state().equal(other.0.state())
    }
}

impl<E> Eq for ScalarNode<E> where E: MemoExpr {}

impl<E> Hash for ScalarNode<E>
where
    E: MemoExpr,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.state().hash(state)
    }
}

/// Provides methods to traverse an expression tree and copy it into a [memo](self::Memo).
pub struct CopyInExprs<'a, E, T>
where
    E: MemoExpr,
{
    pub(crate) memo: &'a mut MemoImpl<E, T>,
    parent: Option<MemoGroupToken<E>>,
    pub(crate) result: Option<E>,
}

impl<'a, E, T> CopyInExprs<'a, E, T>
where
    E: MemoExpr,
{
    fn new(memo: &'a mut MemoImpl<E, T>, parent: Option<MemoGroupToken<E>>) -> Self {
        CopyInExprs {
            memo,
            parent,
            result: None,
        }
    }

    /// Initialises data structures required to traverse child expressions of the given expression `expr`.
    pub fn enter_expr(&mut self, expr: &E) -> ExprContext<E> {
        ExprContext::new(expr, self.parent.take())
    }

    /// Visits the given child expression and recursively copies that expression into the memo:
    /// * If the given expression is in the [owned state](self::OwnedExpr) this methods recursively copies it into the memo.
    /// * If the child expression is in the [memo state](self::MemoizedExpr) this method returns a reference to that group.
    pub fn visit_expr_node(&mut self, expr_ctx: &mut ExprContext<E>, expr_node: impl AsRef<E>) {
        let input = expr_node.as_ref();
        let child_expr = match input.state() {
            MemoExprState::Owned(_) => {
                let copy_in = CopyIn {
                    memo: self.memo,
                    parent: None,
                };
                copy_in.execute(input)
            }
            MemoExprState::Memo(state) => self.memo.get_first_memo_expr(state.group_id()),
        };
        expr_ctx.children.push_back(child_expr);
    }

    /// Visits the given optional child expression if it is present and recursively copies it into a [memo](self::Memo).
    /// This method is equivalent to:
    /// ```text
    /// if let Some(expr_node) = expr_node {
    ///   visitor.visit_expr_node(expr_ctx, expr_node);
    /// }
    /// ```
    pub fn visit_opt_expr_node(&mut self, expr_ctx: &mut ExprContext<E>, expr_node: Option<impl AsRef<E>>) {
        if let Some(expr_node) = expr_node {
            self.visit_expr_node(expr_ctx, expr_node);
        };
    }

    /// Copies the expression into the memo.
    pub fn copy_in(&mut self, expr: &E, expr_ctx: ExprContext<E>) {
        copy_in_expr_impl(self, expr, expr_ctx)
    }
}

/// Stores information that is used to build a new [memo expression](self::MemoExpr).
pub struct ExprContext<E>
where
    E: MemoExpr,
{
    pub(crate) children: VecDeque<E>,
    parent: Option<MemoGroupToken<E>>,
}

impl<E> ExprContext<E>
where
    E: MemoExpr,
{
    pub fn new(_expr: &E, parent: Option<MemoGroupToken<E>>) -> Self {
        ExprContext {
            children: VecDeque::new(),
            parent,
        }
    }
}

pub(crate) fn create_group_properties<E>(expr: &E, children: &VecDeque<E>) -> E::Props
where
    E: MemoExpr,
{
    if expr.expr().is_scalar() && !children.is_empty() {
        let props = E::new_properties_with_nested_sub_queries(
            expr.props().clone(),
            children.iter().map(|i| match i.state() {
                MemoExprState::Owned(_) => {
                    unreachable!("ExprContext.children must contain only references to memo groups")
                }
                MemoExprState::Memo(_) => i.clone(),
            }),
        );
        props
    } else {
        expr.props().clone()
    }
}

struct CopyIn<'a, E, T>
where
    E: MemoExpr,
{
    memo: &'a mut MemoImpl<E, T>,
    parent: Option<MemoGroupToken<E>>,
}

impl<'a, E, T> CopyIn<'a, E, T>
where
    E: MemoExpr,
{
    fn execute(mut self, expr: &E) -> E {
        let mut visitor = CopyInExprs::new(self.memo, self.parent.take());
        expr.copy_in(&mut visitor);
        visitor.result.unwrap()
    }
}

/// Provides methods to collect nested expressions from an expression and copy them into a [memo](self::Memo).
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
        match expr.state() {
            MemoExprState::Owned(_) => {
                expr.copy_in(self.ctx);
                let child_expr = std::mem::take(&mut self.ctx.result).expect("Failed to copy in a nested expressions");
                self.add_child_expr(child_expr);
            }
            MemoExprState::Memo(state) => {
                let child_expr = self.ctx.memo.get_first_memo_expr(state.group_id());
                self.add_child_expr(child_expr)
            }
        }
    }

    fn add_child_expr(&mut self, expr: E) {
        self.nested_exprs.push(expr)
    }
}

pub(crate) struct StringMemoFormatter<'b> {
    buf: &'b mut String,
    write_expr_name: bool,
    pos: usize,
}

impl<'b> StringMemoFormatter<'b> {
    pub fn new(buf: &'b mut String) -> Self {
        let len = buf.len();
        StringMemoFormatter {
            buf,
            write_expr_name: true,
            pos: len,
        }
    }

    pub fn write_expr_name(&mut self, value: bool) {
        self.write_expr_name = value;
    }

    pub fn push(&mut self, c: char) {
        self.buf.push(c);
    }

    pub fn push_str(&mut self, s: &str) {
        self.buf.push_str(s);
    }

    fn pad_value(&mut self) {
        if self.pos != self.buf.len() {
            self.buf.push(' ');
        }
    }
}

impl MemoExprFormatter for StringMemoFormatter<'_> {
    fn write_name(&mut self, name: &str) {
        self.pos = self.buf.len();
        if self.write_expr_name {
            self.buf.push_str(name);
        }
    }

    fn write_source(&mut self, source: &str) {
        self.pad_value();
        self.buf.push_str(source);
    }

    fn write_expr<T>(&mut self, name: &str, input: impl AsRef<T>)
    where
        T: MemoExpr,
    {
        self.pad_value();
        if !name.is_empty() {
            self.buf.push_str(name);
            self.buf.push('=');
        }
        let s = format_node_ref(input.as_ref());
        self.buf.push_str(s.as_str());
    }

    fn write_exprs<T>(&mut self, name: &str, input: impl ExactSizeIterator<Item = impl AsRef<T>>)
    where
        T: MemoExpr,
    {
        if input.len() == 0 {
            return;
        }
        self.pad_value();
        if !name.is_empty() {
            self.buf.push_str(name);
            self.buf.push('=');
        }
        self.buf.push('[');
        for (i, expr) in input.enumerate() {
            if i > 0 {
                self.buf.push_str(", ");
            }
            let s = format_node_ref(expr.as_ref());
            self.buf.push_str(s.as_str());
        }
        self.buf.push(']');
    }

    fn write_value<D>(&mut self, name: &str, value: D)
    where
        D: Display,
    {
        self.pad_value();
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
        self.pad_value();
        self.buf.push_str(name);
        self.buf.push_str("=[");
        self.push_str(values.iter().join(", ").as_str());
        self.buf.push(']');
    }
}

/// Formats [MemoExprRef](self::MemoExprRef).
pub(crate) struct DisplayMemoExprFormatter<'f, 'a> {
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

    fn write_exprs<T>(&mut self, _name: &str, input: impl ExactSizeIterator<Item = impl AsRef<T>>)
    where
        T: MemoExpr,
    {
        for i in input {
            let s = format_node_ref(i.as_ref());
            self.fmt.write_char(' ').unwrap();
            self.fmt.write_str(s.as_str()).unwrap();
        }
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
    match input.state() {
        MemoExprState::Owned(state) => {
            // This only happens when expression has not been added to a memo.
            let ptr: *const T::Expr = state.expr.as_ptr();
            format!("{:?}", ptr)
        }
        MemoExprState::Memo(_) => format!("{}", input.state().memo_group_id()),
    }
}

/// Crates a short representation of an expression that can be used to uniquely identify it.
pub(crate) fn make_digest<E>(expr: &E::Expr, props: &E::Props) -> String
where
    E: MemoExpr,
{
    let mut buf = String::new();
    let mut fmt = StringMemoFormatter::new(&mut buf);
    E::format_expr(expr, props, &mut fmt);
    buf
}

/// Builds a textual representation of the given memo.
pub(crate) fn format_memo<E, T>(memo: &Memo<E, T>) -> String
where
    E: MemoExpr,
{
    format_memo_impl(&memo.memo_impl)
}

impl<E> Debug for MemoExprRef<E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoExprRef").field("id", &self.id()).finish()
    }
}

impl<E> Display for MemoExprRef<E>
where
    E: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} ", self.id())?;
        let mut fmt = DisplayMemoExprFormatter { fmt: f };
        E::format_expr(self.expr(), self.props(), &mut fmt);
        write!(f, "]")?;
        Ok(())
    }
}
