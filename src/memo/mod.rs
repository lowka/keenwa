//! Memo data structure.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fmt::{Debug, Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use itertools::Itertools;

use crate::error::OptimizerError;
use crate::memo::arc::MemoArc;

pub mod arc;
#[cfg(test)]
mod tests;

/// A builder to construct an instance of a [memo](self::Memo).
pub struct MemoBuilder<E, T>
where
    E: MemoExpr,
{
    callback: Option<MemoGroupCallbackRef<E, T>>,
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
    pub fn set_callback(mut self, callback: MemoGroupCallbackRef<E, T>) -> Self {
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
// Use trait bounds here to simplify the type signature.
// Turn off the type alias bounds are not enforced by default warning for this type.
#[allow(type_alias_bounds)]
pub type MemoGroupCallbackRef<E, T>
where
    E: MemoExpr,
= Rc<
    dyn MemoGroupCallback<
        Expr = <E as MemoExpr>::Expr,
        Props = <E as MemoExpr>::Props,
        Scope = <E as MemoExpr>::Scope,
        Metadata = T,
    >,
>;

/// `Memo` is the primary data structure used by the cost-based optimizer:
///  * It stores each expression as a group of logically equivalent expressions.
///  * It provides memoization of identical subexpressions within an expression tree.
// TODO: Examples
// TODO: add generic implementation of a copy-out method.
pub struct Memo<E, T>
where
    E: MemoExpr,
{
    memo_impl: MemoImpl<E, T>,
}

impl<E, T> Memo<E, T>
where
    E: MemoExpr,
{
    /// Copies the given expression `expr` into this memo. if this memo does not contain the given expression
    /// a new memo group is created and this method returns a reference to it. Otherwise returns a reference
    /// to the already existing expression.
    /// If the given token belong to another memo this method returns an error.
    pub fn insert_group(&mut self, expr: E, scope: &E::Scope) -> Result<E, OptimizerError> {
        let copy_in = CopyIn {
            scope,
            memo: &mut self.memo_impl,
            parent: None,
        };
        copy_in.execute(&expr)
    }

    /// Copies the expression `expr` into this memo and adds it to the memo group that granted the given [membership token](MemoGroupToken).
    /// If an identical expression already exist this method simply returns a reference to that expression.
    /// If the given token belong to another memo this method returns an error.
    pub fn insert_group_member(
        &mut self,
        token: MemoGroupToken<E>,
        expr: E,
        scope: &E::Scope,
    ) -> Result<E, OptimizerError> {
        let copy_in = CopyIn {
            scope,
            memo: &mut self.memo_impl,
            parent: Some(token),
        };
        copy_in.execute(&expr)
    }

    /// Return a reference to a memo group with the given id or an error if it does not exists.
    pub fn get_group(&self, group_id: &GroupId) -> Result<MemoGroupRef<E, T>, OptimizerError> {
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

/// A trait that must be implemented by an expression that can be copied into a [`memo`](self::Memo).
// TODO: Docs.
pub trait MemoExpr: Clone {
    /// The type of the expression.
    type Expr: Expr;
    /// The type of the properties.
    type Props: Props;
    /// The type of the scope. The scope of a memo expression contains
    /// external information that can be used to build properties of a memo expression group.
    type Scope;

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
    fn copy_in<T>(&self, visitor: &mut CopyInExprs<Self, T>) -> Result<(), OptimizerError>;

    /// Creates a new expression from the given expression `expr` by replacing its child expressions
    /// with expressions provided by the given [NewChildExprs].
    fn expr_with_new_children(expr: &Self::Expr, inputs: NewChildExprs<Self>) -> Result<Self::Expr, OptimizerError>;

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

    /// Returns `true` if this is a relational expression.
    fn is_relational(&self) -> bool;

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
    /// The type of scope.
    type Scope;
    /// The type of metadata stored by a memo.
    type Metadata;

    /// Called when a new memo group is added to a memo and returns properties to be shared among all expressions in this group.
    /// Where `expr` is the expression that created the memo group and `props` are properties associated with that expression.
    ///
    /// #Note:
    ///
    /// If a call returns an error than a caller (a memo) unwraps it because the optimization process
    /// can not continue when a memo group has no logical properties.
    fn new_group(
        &self,
        expr: &Self::Expr,
        scope: &Self::Scope,
        provided_props: Self::Props,
        metadata: &Self::Metadata,
    ) -> Result<Self::Props, OptimizerError>;
}

/// Represents an expression that has not been copied into a [memo](Memo).
/// This is the `owned` state of a [MemoExprState].
#[derive(Clone)]
pub struct OwnedExpr<E>
where
    E: MemoExpr,
{
    pub(crate) expr: MemoArc<E::Expr>,
    pub(crate) props: MemoArc<E::Props>,
}

impl<E> OwnedExpr<E>
where
    E: MemoExpr,
{
    /// Creates a copy of this expression that uses the given properties.
    pub fn with_props(&self, props: E::Props) -> Self {
        OwnedExpr {
            expr: self.expr.clone(),
            props: MemoArc::new(props),
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
    fn write_values<D>(&mut self, name: &str, values: impl ExactSizeIterator<Item = D>)
    where
        D: Display;

    /// Returns [flags](MemoFormatterFlags) specified for this formatter.
    fn flags(&self) -> &MemoFormatterFlags;
}

/// Specifies flags to be used by the [formatter](MemoExprFormatter).
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum MemoFormatterFlags {
    /// No flags are set.
    None,
    /// Instruct a formatter to build a representation of a memo expression
    /// that uniquely identifies that expression.
    Digest,
    /// Instruct a formatter to include properties
    /// of a memo expression in its output.
    IncludeProps,
    /// All flags except for [Digest](Self::Digest) are set.
    All,
    //???: Add custom flags
}

impl MemoFormatterFlags {
    /// Checks whether the given flag is set.u
    pub fn has_flag(&self, flag: &MemoFormatterFlags) -> bool {
        let this_mask = self.get_mask();
        let flag_mask = flag.get_mask();
        // None flag is never set.
        flag_mask != 0 && (this_mask & flag_mask) == flag_mask
    }

    fn get_mask(&self) -> u32 {
        match self {
            MemoFormatterFlags::None => 0,
            MemoFormatterFlags::Digest => 1,
            MemoFormatterFlags::IncludeProps => 2,
            MemoFormatterFlags::All => u32::MAX & (!1),
        }
    }
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
    pub fn expect_len(&self, n: usize, operator: &str) -> Result<(), OptimizerError> {
        if self.capacity != n {
            let message = format!(
                "{}: Unexpected number of child expressions. Expected {} but got {}",
                operator, n, self.capacity
            );
            Err(OptimizerError::argument(message))
        } else {
            Ok(())
        }
    }

    /// Retrieves the next relational expression.
    ///
    /// This method return an error if there are no relational expressions left or the retrieved expression is
    /// not a relational expression.
    pub fn rel_node(&mut self) -> Result<RelNode<E>, OptimizerError> {
        self.expr(&RelNode::try_from)
    }

    /// Retrieves the next `n` relational expressions.
    ///
    /// This method returns an error if there are not enough expressions left or some of the retrieved expressions are
    /// not relational expressions.
    pub fn rel_nodes(&mut self, n: usize) -> Result<Vec<RelNode<E>>, OptimizerError> {
        self.exprs(n, &RelNode::try_from)
    }

    /// Retrieves the next optional relational expression.
    /// * If `expr` is [Some] returns the next relational expression.
    /// * If `expr` is [None] returns [None].
    ///
    /// This method returns an error if there are no expressions left or the retrieved expression is
    /// not a relational expression.
    pub fn rel_opt_node(&mut self, expr: Option<&RelNode<E>>) -> Result<Option<RelNode<E>>, OptimizerError> {
        self.expr_opt(expr, &RelNode::try_from)
    }

    /// Retrieves the next scalar expression.
    /// This method returns an error if there are no expressions left or the retrieved expression is
    /// not a scalar expression.
    pub fn scalar_node(&mut self) -> Result<ScalarNode<E>, OptimizerError> {
        self.expr(&ScalarNode::try_from)
    }

    /// Retrieves the next `n` scalar expressions.
    ///
    /// This method returns an error if there are not enough expressions left or some of the retrieved expressions are
    /// not scalar expressions.
    pub fn scalar_nodes(&mut self, n: usize) -> Result<Vec<ScalarNode<E>>, OptimizerError> {
        self.exprs(n, &ScalarNode::try_from)
    }

    /// Retrieves the next optional scalar expression.
    /// * If `expr` is [Some] returns the next scalar expression.
    /// * If `expr` is [None] returns [None].
    ///
    /// This method returns an error if there are no expressions left or the retrieved expression is
    /// not a scalar expression.
    pub fn scalar_opt_node(&mut self, expr: Option<&ScalarNode<E>>) -> Result<Option<ScalarNode<E>>, OptimizerError> {
        self.expr_opt(expr, &ScalarNode::try_from)
    }

    fn expr<F, T>(&mut self, f: &F) -> Result<T, OptimizerError>
    where
        F: Fn(E) -> Result<T, OptimizerError>,
    {
        match self.children.pop_front() {
            Some(expr) => (f)(expr),
            None => Err(OptimizerError::internal("Unable to retrieve an expression")),
        }
    }

    fn exprs<F, T>(&mut self, n: usize, f: &F) -> Result<Vec<T>, OptimizerError>
    where
        F: Fn(E) -> Result<T, OptimizerError>,
    {
        if self.children.len() < n {
            let message = format!("Unable to retrieve {} expression(s). Total: {}", n, self.children.len());
            Err(OptimizerError::internal(message))
        } else {
            let mut children = Vec::with_capacity(n);
            for _ in 0..n {
                // Unwrapping won't panic because we have at least n elements.
                let expr = self.children.pop_front().unwrap();
                let expr = (f)(expr)?;
                children.push(expr);
            }
            Ok(children)
        }
    }

    fn expr_opt<F, T>(&mut self, expr: Option<&T>, f: &F) -> Result<Option<T>, OptimizerError>
    where
        F: Fn(E) -> Result<T, OptimizerError>,
    {
        match expr {
            Some(_) => Ok(Some(self.expr(f)?)),
            None => Ok(None),
        }
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

    /// Creates a relational node of an expression tree from the given relational expression.
    /// This method is a shorthand for `RelNode::new(expr, RelProps::default())`
    /// if `RelProps` implements [Default].
    pub fn new_expr(expr: RelExpr) -> Self
    where
        RelProps: Default,
    {
        RelNode::new(expr, RelProps::default())
    }

    /// Creates a relational node of an expression tree from the given memo expression.
    /// If the given expression is not relational expression this method returns an error.
    pub fn try_from(expr: E) -> Result<Self, OptimizerError> {
        if !expr.expr().is_relational() {
            Err(OptimizerError::argument("Expected a relational expression"))
        } else {
            Ok(RelNode(expr))
        }
    }

    /// Creates a relational node of an expression tree from the given memo group.
    /// If the given group does not contain relational expressions this method returns an error.
    pub fn try_from_group<M>(group: MemoGroupRef<'_, E, M>) -> Result<Self, OptimizerError> {
        let expr = group.to_memo_expr();
        if !expr.expr().is_relational() {
            Err(OptimizerError::argument("Expected a relational expression"))
        } else {
            Ok(RelNode(expr))
        }
    }

    /// Returns a reference to the underlying expression.
    pub fn expr<'e>(&'e self) -> &'e T::RelExpr
    where
        T: 'e,
    {
        self.0.expr().relational()
    }

    /// Returns a reference to properties associated with this node:
    /// * if this node is an expression returns a reference to the properties of the underlying expression.
    /// * If this node is a memo group returns a reference to the properties of the first expression of this memo group.
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

    /// Creates a scalar node of an expression tree from the given scalar expression and default properties.
    /// This method is a shorthand for `ScalarNode::new(expr, ScalarProps::default())`
    /// if `ScalarProps` implements [Default].
    pub fn new_expr(expr: ScalarExpr) -> Self
    where
        ScalarProps: Default,
    {
        ScalarNode::new(expr, ScalarProps::default())
    }

    /// Creates a scalar node of an expression tree from the given memo expression.
    /// if the given expression is not a scalar expression this method returns an error.
    pub fn try_from(expr: E) -> Result<Self, OptimizerError> {
        if expr.expr().is_scalar() {
            Ok(ScalarNode(expr))
        } else {
            Err(OptimizerError::internal("Expected a scalar expression"))
        }
    }

    /// Creates a relational node of an expression tree from the given memo group.
    /// If the given group does not contain scalar expressions this method returns an error.
    pub fn try_from_group<M>(group: MemoGroupRef<'_, E, M>) -> Result<Self, OptimizerError> {
        let expr = group.to_memo_expr();
        if !expr.expr().is_scalar() {
            Err(OptimizerError::argument("Expected a scalar expression"))
        } else {
            Ok(ScalarNode(expr))
        }
    }

    /// Returns a reference to the underlying scalar expression.
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
    scope: &'a E::Scope,
    pub(crate) memo: &'a mut MemoImpl<E, T>,
    parent: Option<MemoGroupToken<E>>,
    pub(crate) result: Option<E>,
}

impl<'a, E, T> CopyInExprs<'a, E, T>
where
    E: MemoExpr,
{
    fn new(scope: &'a E::Scope, memo: &'a mut MemoImpl<E, T>, parent: Option<MemoGroupToken<E>>) -> Self {
        CopyInExprs {
            scope,
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
    pub fn visit_expr_node(
        &mut self,
        expr_ctx: &mut ExprContext<E>,
        expr_node: impl AsRef<E>,
    ) -> Result<(), OptimizerError> {
        let input = expr_node.as_ref();
        let child_expr = match input.state() {
            MemoExprState::Owned(_) => {
                let copy_in = CopyIn {
                    scope: self.scope,
                    memo: self.memo,
                    parent: None,
                };
                copy_in.execute(input)?
            }
            MemoExprState::Memo(state) => self.memo.get_first_memo_expr(state.group_id())?,
        };
        expr_ctx.children.push_back(child_expr);
        Ok(())
    }

    /// Visits the given optional child expression if it is present and recursively copies it into a [memo](self::Memo).
    /// This method is equivalent to:
    /// ```text
    /// if let Some(expr_node) = expr_node {
    ///   visitor.visit_expr_node(expr_ctx, expr_node);
    /// }
    /// ```
    pub fn visit_opt_expr_node(
        &mut self,
        expr_ctx: &mut ExprContext<E>,
        expr_node: Option<impl AsRef<E>>,
    ) -> Result<(), OptimizerError> {
        if let Some(expr_node) = expr_node {
            self.visit_expr_node(expr_ctx, expr_node)
        } else {
            Ok(())
        }
    }

    /// Copies the expression into the memo.
    pub fn copy_in(&mut self, expr: &E, expr_ctx: ExprContext<E>) -> Result<(), OptimizerError> {
        match self.copy_in_internal(expr, expr_ctx) {
            // Do not expose an added expression in public API.
            Ok(expr) => {
                self.result = Some(expr);
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn copy_in_internal(&mut self, expr: &E, expr_ctx: ExprContext<E>) -> Result<E, OptimizerError> {
        let expr_id = ExprId(self.memo.exprs.len());

        let ExprContext { children, parent } = expr_ctx;
        let props = create_group_properties(expr, &children);

        let expr = E::expr_with_new_children(expr.expr(), NewChildExprs::new(children))?;
        let digest = make_digest::<E>(&expr, &props);

        let (expr_id, added) = match self.memo.expr_cache.entry(digest) {
            Entry::Occupied(o) => (*o.get(), false),
            Entry::Vacant(v) => {
                v.insert(expr_id);
                (expr_id, true)
            }
        };

        if added {
            if let Some(token) = parent {
                self.memo.add_expr_to_group(expr_id, expr, token.group_id)
            } else {
                let props = if let Some(callback) = &self.memo.callback {
                    callback.new_group(&expr, self.scope, props, &self.memo.metadata)?
                } else {
                    props
                };
                self.memo.add_new_expr(expr_id, expr, props)
            }
        } else {
            self.memo.get_expr(expr_id)
        }
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
    scope: &'a E::Scope,
    memo: &'a mut MemoImpl<E, T>,
    parent: Option<MemoGroupToken<E>>,
}

impl<'a, E, T> CopyIn<'a, E, T>
where
    E: MemoExpr,
{
    fn execute(mut self, expr: &E) -> Result<E, OptimizerError> {
        let mut visitor = CopyInExprs::new(self.scope, self.memo, self.parent.take());
        expr.copy_in(&mut visitor)?;
        match visitor.result {
            Some(expr) => Ok(expr),
            None => unreachable!("Expressions: result expression has not been set"),
        }
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
    pub fn execute<F, S>(mut self, expr: &S, f: F) -> Result<(), OptimizerError>
    where
        F: Fn(&S, &mut Self) -> Result<(), OptimizerError>,
    {
        (f)(expr, &mut self)?;
        // Visit collected nested expressions so that they will be added to the given expression as child expressions.
        for expr_node in self.nested_exprs {
            let rel_node = RelNode::try_from(expr_node)?;
            self.ctx.visit_expr_node(self.expr_ctx, rel_node)?;
        }
        Ok(())
    }

    /// Copies the given nested expression into a memo.
    pub fn visit_expr(&mut self, expr: impl AsRef<E>) -> Result<(), OptimizerError> {
        let expr = expr.as_ref();
        match expr.state() {
            MemoExprState::Owned(_) => {
                expr.copy_in(self.ctx)?;
                match std::mem::take(&mut self.ctx.result) {
                    Some(child_expr) => self.add_child_expr(child_expr),
                    None => unreachable!("Nested expressions: result expression has not been set"),
                }
            }
            MemoExprState::Memo(state) => {
                let child_expr = self.ctx.memo.get_first_memo_expr(state.group_id())?;
                self.add_child_expr(child_expr)
            }
        }
        Ok(())
    }

    fn add_child_expr(&mut self, expr: E) {
        self.nested_exprs.push(expr)
    }
}

pub(crate) struct StringMemoFormatter<'b> {
    buf: &'b mut String,
    flags: MemoFormatterFlags,
    write_expr_name: bool,
    pos: usize,
}

impl<'b> StringMemoFormatter<'b> {
    pub fn new(buf: &'b mut String, flags: MemoFormatterFlags) -> Self {
        let len = buf.len();
        StringMemoFormatter {
            buf,
            flags,
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

    fn write_values<D>(&mut self, name: &str, mut values: impl ExactSizeIterator<Item = D>)
    where
        D: Display,
    {
        if values.len() == 0 {
            return;
        }
        self.pad_value();
        self.buf.push_str(name);
        self.buf.push_str("=[");
        self.push_str(values.join(", ").as_str());
        self.buf.push(']');
    }

    fn flags(&self) -> &MemoFormatterFlags {
        &self.flags
    }
}

/// Formats [MemoExprRef](self::MemoExprRef).
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

    fn write_values<D>(&mut self, _name: &str, _values: impl ExactSizeIterator<Item = D>)
    where
        D: Display,
    {
        // Do not print values for Display trait
    }

    fn flags(&self) -> &MemoFormatterFlags {
        &MemoFormatterFlags::None
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
    let mut fmt = StringMemoFormatter::new(&mut buf, MemoFormatterFlags::Digest);
    E::format_expr(expr, props, &mut fmt);
    buf
}

/// Builds a textual representation of the given memo.
pub(crate) fn format_memo<E, T>(memo: &Memo<E, T>) -> String
where
    E: MemoExpr,
{
    let mut buf = String::new();
    let mut f = StringMemoFormatter::new(&mut buf, MemoFormatterFlags::None);

    for (exprs, group) in memo.memo_impl.groups.iter().rev() {
        f.push_str(format!("{} ", group.group_id).as_str());
        for (i, expr_id) in exprs.iter().enumerate() {
            if i > 0 {
                // newline + 3 spaces
                f.push_str("\n   ");
            }
            let expr = &memo.memo_impl.exprs[expr_id.index()];
            E::format_expr(expr.expr(), expr.props(), &mut f);
        }
        f.push('\n');
    }

    buf
}

pub struct MemoImpl<E, T>
where
    E: MemoExpr,
{
    groups: Vec<(Vec<ExprId>, MemoArc<MemoGroupData<E>>)>,
    exprs: Vec<E>,
    expr_cache: HashMap<String, ExprId>,
    expr_to_group: HashMap<ExprId, GroupId>,
    metadata: T,
    callback: Option<MemoGroupCallbackRef<E, T>>,
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

    pub(crate) fn with_callback(metadata: T, callback: MemoGroupCallbackRef<E, T>) -> Self {
        MemoImpl {
            exprs: Vec::new(),
            groups: Vec::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            metadata,
            callback: Some(callback),
        }
    }

    pub(crate) fn get_group(&self, group_id: &GroupId) -> Result<MemoGroupRef<E, T>, OptimizerError> {
        match self.groups.get(group_id.index()).map(|(group_exprs, group_data)| {
            // A group always contains at least one expression and that expression always exists.
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
        }) {
            Some(group) => Ok(group),
            None => Err(OptimizerError::internal("Unexpected group id")),
        }
    }

    fn metadata(&self) -> &T {
        &self.metadata
    }

    fn num_groups(&self) -> usize {
        self.groups.len()
    }

    fn num_exprs(&self) -> usize {
        self.exprs.len()
    }

    fn expr_to_group(&self) -> &HashMap<ExprId, GroupId> {
        &self.expr_to_group
    }

    fn add_expr_to_group(&mut self, expr_id: ExprId, expr: E::Expr, group_id: GroupId) -> Result<E, OptimizerError> {
        match self.groups.get_mut(group_id.index()) {
            Some((group_exprs, group_data)) => {
                group_exprs.push(expr_id);
                let group_data = group_data.clone();

                self.add_expr_internal(expr_id, expr, group_data)
            }
            None => Err(OptimizerError::internal("Unexpected group id")),
        }
    }

    fn add_new_expr(&mut self, expr_id: ExprId, expr: E::Expr, props: E::Props) -> Result<E, OptimizerError> {
        let group_id = GroupId(self.groups.len());
        let group_data = MemoGroupData { group_id, props };
        let group_data = MemoArc::new(group_data);
        self.groups.push((vec![expr_id], group_data.clone()));

        self.add_expr_internal(expr_id, expr, group_data)
    }

    fn add_expr_internal(
        &mut self,
        expr_id: ExprId,
        expr: E::Expr,
        group_data: MemoArc<MemoGroupData<E>>,
    ) -> Result<E, OptimizerError> {
        let group_id = group_data.group_id;
        let expr_data = MemoExprData { expr_id, expr };
        let expr_data = MemoArc::new(expr_data);
        let memo_state = MemoizedExpr {
            expr: expr_data,
            group: group_data,
        };
        let expr = E::from_state(MemoExprState::Memo(memo_state));
        self.expr_to_group.insert(expr_id, group_id);
        self.exprs.push(expr.clone());
        Ok(expr)
    }

    fn get_expr(&self, expr_id: ExprId) -> Result<E, OptimizerError> {
        if let Some(expr) = self.exprs.get(expr_id.index()) {
            Ok(expr.clone())
        } else {
            Err(OptimizerError::internal("Unexpected expression id"))
        }
    }

    fn get_first_memo_expr(&self, group_id: GroupId) -> Result<E, OptimizerError> {
        match self.groups.get(group_id.index()) {
            Some((group_exprs, _)) => {
                // A group always contains at least one expression and that expression always exists.
                let first_expr_id = group_exprs[0];
                let expr = self.exprs[first_expr_id.index()].clone();
                Ok(expr)
            }
            None => Err(OptimizerError::internal("Unexpected group id")),
        }
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
    data: &'a MemoArc<MemoGroupData<E>>,
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
    data: &'a MemoArc<MemoGroupData<E>>,
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

/// Represents a state of a [memo expression](MemoExpr).
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
            expr: MemoArc::new(expr),
            props: MemoArc::new(props),
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

/// Represents an expression stored in a [memo](Memo).
/// This is the `memo` state of a [MemoExprState].
#[derive(Clone)]
// get_expr_ref fails to compiler under 1.57.
// note: rustc 1.57.0 (f1edd0429 2021-11-29) running on x86_64-apple-darwin
// thread 'rustc' panicked at 'called `Option::unwrap()` on a `None` value'
// /compiler/rustc_hir/src/definitions.rs:452:14
pub struct MemoizedExpr<E>
where
    E: MemoExpr,
{
    expr: MemoArc<MemoExprData<E>>,
    group: MemoArc<MemoGroupData<E>>,
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
