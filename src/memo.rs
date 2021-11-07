use std::fmt::{Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
};

/// Memo is the primary data structure used by the cost-based optimizer:
///  * It stores each expression as a group of logically equivalent expressions.
///  * It provides memoization of identical sub-expressions within an expression tree.
///
/// #Safety
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
    callback: Option<Rc<dyn MemoExprCallback<Expr = T, Attrs = T::Attrs>>>,
}

impl<T> Memo<T>
where
    T: MemoExpr,
{
    /// Creates a new memo.
    pub(crate) fn new() -> Self {
        Memo {
            groups: Vec::new(),
            exprs: Vec::new(),
            expr_cache: HashMap::new(),
            expr_to_group: HashMap::new(),
            callback: None,
        }
    }

    /// Creates a new memo with the given callback.
    pub(crate) fn with_callback(callback: Rc<dyn MemoExprCallback<Expr = T, Attrs = T::Attrs>>) -> Self {
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

/// A trait that must be implemented by an expression that can be copied into a [memo].
///
/// [memo]: crate::memo::Memo
pub trait MemoExpr: Debug + Clone {
    /// The type of the expression.
    type Expr: Debug + Clone;
    /// The type of the attributes.
    type Attrs: Attributes;

    /// Returns the expression this memo expression stores.
    fn expr(&self) -> &Self::Expr;

    /// Returns attributes associated with this expression.
    fn attrs(&self) -> &Self::Attrs;

    /// Recursively traverses this expression and copies it into a memo.
    //TODO: this method should consume the expression.
    fn copy_in(&self, ctx: &mut TraversalContext<Self>);

    /// Creates a new expression from this expression by replacing its input expressions with the new ones.
    fn with_new_inputs(&self, inputs: Vec<InputNode<Self>>) -> Self;

    /// Builds a textual representation of this expression.
    fn format_expr<F>(&self, f: &mut F)
    where
        F: MemoExprFormatter;
}

/// A trait for attributes of the expression.
pub trait Attributes: Debug + Clone {}

/// A callback that is called when a new expression is added to a memo.
pub trait MemoExprCallback: Debug {
    /// The type of expression.
    type Expr: MemoExpr;
    /// The type of attributes of the expression.
    type Attrs: Attributes;

    /// Called when the given expression `expr` with attributes `attrs` is added to a memo.
    fn new_expr(&self, expr: &Self::Expr, attrs: Self::Attrs) -> Self::Attrs;
}

/// An input node of an expression. It can be either an expression or a memo group.
#[derive(Clone)]
pub enum InputNode<T>
where
    T: MemoExpr,
{
    /// Input node is another expression.
    Expr(Box<T>),
    /// Input node is a memo group.
    Group(MemoGroupRef<T>),
}

impl<T> InputNode<T>
where
    T: MemoExpr,
{
    /// Returns an expression this `InputNode` points to.
    /// If this node is an expression this method returns a reference to the actual expression.
    /// If this node is a memo group this method returns a reference to the first memo expression in this memo group.
    pub fn expr(&self) -> &T::Expr {
        match self {
            InputNode::Expr(expr) => expr.expr(),
            InputNode::Group(group) => group.expr(),
        }
    }

    /// Returns attributes of an expression this `InputNode` points to.
    /// If this node is an expression this method returns a reference to the attributes of the memo group the expression belongs to.
    /// If this node is a memo group this method returns a reference to the attributes of this memo group.
    pub fn attrs(&self) -> &T::Attrs {
        match self {
            InputNode::Expr(expr) => expr.attrs(),
            InputNode::Group(group) => group.attrs(),
        }
    }
}

impl<T> From<T> for InputNode<T>
where
    T: MemoExpr,
{
    fn from(o: T) -> Self {
        InputNode::Expr(Box::new(o))
    }
}

impl<T> Debug for InputNode<T>
where
    T: MemoExpr,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InputNode::Expr(expr) => write!(f, "Expr({:?})", expr),
            InputNode::Group(group) => write!(f, "Group({:?})", group.id),
        }
    }
}

/// A `MemoGroupRef` is a reference to a memo group. A memo group is a group of logically equivalent expressions.
/// Expressions are logically equivalent when they produce the same result.
///
/// #Safety
/// A reference to a memo group is valid until a [memo] it belongs to is dropped.
///
/// [memo]: crate::memo::Memo
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

    /// Returns attributes shared by all expressions in this memo group.
    pub fn attrs(&self) -> &T::Attrs {
        let group = self.get_memo_group();
        &group.attrs
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
/// A reference to a memo expression is valid until a [memo] it belongs to is dropped.
///
/// [memo]: crate::memo::Memo
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

    /// Returns references to the inputs of this memo expression.
    pub fn inputs(&self) -> &[MemoGroupRef<T>] {
        let expr = self.get_memo_expr();
        &expr.inputs
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
    attrs: T::Attrs,
}

impl<T> MemoGroupData<T>
where
    T: MemoExpr,
{
    fn new(group_id: GroupId, attrs: T::Attrs) -> Self {
        MemoGroupData {
            group_id,
            exprs: vec![],
            attrs,
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
    inputs: Vec<MemoGroupRef<T>>,
    group: MemoGroupRef<T>,
}

impl<T> MemoExprData<T>
where
    T: MemoExpr,
{
    fn new(expr_id: ExprId, expr: T, inputs: Vec<MemoGroupRef<T>>, group: MemoGroupRef<T>) -> Self {
        MemoExprData {
            expr_id,
            expr,
            inputs,
            group,
        }
    }
}

/// Provides methods to traverse an expression tree and copy it into a memo.
pub struct TraversalContext<'a, T>
where
    T: MemoExpr,
{
    memo: &'a mut Memo<T>,
    parent: Option<MemoGroupRef<T>>,
    depth: usize,
    result: Option<(MemoGroupRef<T>, MemoExprRef<T>)>,
}

impl<'a, T> TraversalContext<'a, T>
where
    T: MemoExpr,
{
    fn new(memo: &'a mut Memo<T>, parent: Option<MemoGroupRef<T>>, depth: usize) -> Self {
        TraversalContext {
            memo,
            parent,
            depth,
            result: None,
        }
    }

    /// Initialises data structures required to traverse the given expression `expr`.
    pub fn enter_expr(&mut self, expr: &T) -> ExprContext<T> {
        ExprContext {
            inputs: vec![],
            parent: self.parent.clone(),
            // TODO: Add a method that splits MemoExpr into Self::Expr and Self::Attrs
            //  this will allow to remove the clone() call below .
            attrs: expr.attrs().clone(),
        }
    }

    /// Visits the given input expression and recursively copies that expression into the memo:
    /// * If the input expression is an expression this methods recursively copies it into the memo.
    /// * If the input expression is a group this method returns a reference to that group.
    pub fn visit_input(
        &mut self,
        expr_ctx: &mut ExprContext<T>,
        input: &InputNode<T>,
    ) -> (MemoGroupRef<T>, MemoExprRef<T>) {
        match input {
            InputNode::Expr(e) => {
                let copy_in = CopyIn {
                    memo: self.memo,
                    parent: None,
                    depth: self.depth + 1,
                };
                let (group, expr) = copy_in.execute(e);
                expr_ctx.inputs.push(InputNode::Group(group.clone()));
                (group, expr)
            }
            InputNode::Group(group) => {
                expr_ctx.inputs.push(InputNode::Group(group.clone()));
                let expr = group.mexpr();
                (group.clone(), expr.clone())
            }
        }
    }

    /// Copies the expression into the memo.
    /// At this point all the inputs expressions should have been copied into the memo as well.
    pub fn copy_in(&mut self, expr: &T, expr_ctx: ExprContext<T>) {
        let expr_id = ExprId(self.memo.exprs.len());
        let input_groups = expr_ctx
            .inputs
            .iter()
            .map(|i| match i {
                InputNode::Group(g) => g.clone(),
                InputNode::Expr(e) => panic!("Expected a group but got an expression: {:?}", e),
            })
            .collect();

        let ExprContext {
            inputs: input_nodes,
            attrs,
            parent,
        } = expr_ctx;

        let expr = expr.with_new_inputs(input_nodes);
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
                let attrs = if let Some(callback) = self.memo.callback.as_ref() {
                    callback.new_expr(&expr, attrs)
                } else {
                    attrs
                };
                let memo_group = MemoGroupData::new(group_id, attrs);
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
    inputs: Vec<InputNode<T>>,
    parent: Option<MemoGroupRef<T>>,
    attrs: T::Attrs,
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
        let mut ctx = TraversalContext::new(self.memo, self.parent.clone(), self.depth);
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

    /// Writes an input expression.
    fn write_input<T>(&mut self, name: &str, input: &InputNode<T>)
    where
        T: MemoExpr;

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

    fn write_input<T>(&mut self, name: &str, input: &InputNode<T>)
    where
        T: MemoExpr,
    {
        self.buf.push(' ');
        if !name.is_empty() {
            self.buf.push_str(name);
            self.buf.push('=');
        }
        match input {
            InputNode::Expr(expr) => panic!("Input expressions are not supported: {:?}", expr),
            InputNode::Group(group) => {
                self.buf.push_str(format!("{}", group.id()).as_str());
            }
        }
    }

    fn write_value<D>(&mut self, name: &str, value: D)
    where
        D: Display,
    {
        self.buf.push(' ');
        self.buf.push_str(name);
        self.buf.push('=');
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
        self.buf.push('=');
        self.buf.push('[');
        for (i, value) in values.iter().enumerate() {
            if i > 0 {
                self.buf.push_str(", ");
            }
            self.buf.push_str(&value.to_string());
        }
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

    fn write_input<T>(&mut self, _name: &str, input: &InputNode<T>)
    where
        T: MemoExpr,
    {
        match input {
            InputNode::Expr(_) => panic!("Expr inputs inside a memo are not allowed"),
            InputNode::Group(group) => {
                self.fmt.write_fmt(format_args!(" {}", group.id())).unwrap();
            }
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

    #[derive(Debug, Clone)]
    enum TestExpr {
        Leaf(&'static str),
        Node { input: InputNode<TestOperator> },
        Nodes { inputs: Vec<InputNode<TestOperator>> },
    }

    #[derive(Debug, Clone)]
    struct TestOperator {
        expr: TestExpr,
        attrs: Option<TestAttrs>,
    }

    impl From<TestExpr> for TestOperator {
        fn from(expr: TestExpr) -> Self {
            TestOperator { expr, attrs: None }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct TestAttrs {
        a: i32,
    }

    impl Attributes for Option<TestAttrs> {}

    impl TestOperator {
        fn with_attrs(self, a: i32) -> Self {
            TestOperator {
                expr: self.expr,
                attrs: Some(TestAttrs { a }),
            }
        }
    }

    impl MemoExpr for TestOperator {
        type Expr = TestExpr;
        type Attrs = Option<TestAttrs>;

        fn expr(&self) -> &Self::Expr {
            &self.expr
        }

        fn attrs(&self) -> &Self::Attrs {
            &self.attrs
        }

        fn copy_in(&self, ctx: &mut TraversalContext<Self>) {
            let mut expr_ctx = ctx.enter_expr(self);
            match &self.expr {
                TestExpr::Leaf(_) => {}
                TestExpr::Node { input } => {
                    ctx.visit_input(&mut expr_ctx, input);
                }
                TestExpr::Nodes { inputs } => {
                    for input in inputs {
                        ctx.visit_input(&mut expr_ctx, input);
                    }
                }
            }
            ctx.copy_in(self, expr_ctx)
        }

        fn with_new_inputs(&self, mut inputs: Vec<InputNode<Self>>) -> Self {
            let expr = match &self.expr {
                TestExpr::Leaf(s) => TestExpr::Leaf(s.clone()),
                TestExpr::Node { .. } => TestExpr::Node {
                    input: inputs.swap_remove(0),
                },
                TestExpr::Nodes { .. } => TestExpr::Nodes { inputs },
            };
            TestOperator {
                expr,
                attrs: self.attrs.clone(),
            }
        }

        fn format_expr<F>(&self, f: &mut F)
        where
            F: MemoExprFormatter,
        {
            match &self.expr {
                TestExpr::Leaf(source) => {
                    f.write_name("Leaf");
                    f.write_source(source);
                }
                TestExpr::Node { input } => {
                    f.write_name("Node");
                    f.write_input("", input);
                }
                TestExpr::Nodes { inputs } => {
                    f.write_name("Nodes");
                    for input in inputs {
                        f.write_input("", input);
                    }
                }
            }
        }
    }

    #[test]
    fn test_basics() {
        let mut memo = Memo::new();
        let expr = TestOperator::from(TestExpr::Leaf("aaaa")).with_attrs(100);
        let (group, expr) = memo.insert(expr);

        assert_eq!(group.attrs(), &Some(TestAttrs { a: 100 }), "group attributes");
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
    fn test_attributes() {
        let mut memo = Memo::new();

        let leaf = TestOperator::from(TestExpr::Leaf("a")).with_attrs(10);
        let inner = TestOperator::from(TestExpr::Node { input: leaf.into() }).with_attrs(15);
        let expr = TestOperator::from(TestExpr::Node { input: inner.into() });
        let outer = TestOperator::from(TestExpr::Node { input: expr.into() }).with_attrs(20);

        let _ = memo.insert(outer);

        expect_memo_with_attrs(
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
    fn test_input_node() {
        let mut memo = Memo::new();

        let leaf = TestOperator::from(TestExpr::Leaf("a")).with_attrs(10);
        let expr = TestOperator::from(TestExpr::Node {
            input: leaf.clone().into(),
        });
        let (_, expr) = memo.insert(expr);

        match expr.expr() {
            TestExpr::Node { input } => {
                assert_eq!(format!("{:?}", input.expr()), format!("{:?}", expr.inputs()[0].expr()), "input expr");
                assert_eq!(input.attrs(), leaf.attrs(), "input attrs");
            }
            _ => panic!("Unexpected expression: {:?}", expr),
        }
    }

    #[test]
    fn test_memo_trivial_expr() {
        let mut memo = Memo::new();

        let expr = TestOperator::from(TestExpr::Leaf("a"));
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

        let expr = TestOperator::from(TestExpr::Node {
            input: TestOperator::from(TestExpr::Leaf("a")).into(),
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

        let expr = TestOperator::from(TestExpr::Nodes {
            inputs: vec![
                TestOperator::from(TestExpr::Leaf("a")).into(),
                TestOperator::from(TestExpr::Leaf("b")).into(),
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

        let expr = TestOperator::from(TestExpr::Nodes {
            inputs: vec![
                TestOperator::from(TestExpr::Node {
                    input: TestOperator::from(TestExpr::Leaf("a")).into(),
                })
                .into(),
                TestOperator::from(TestExpr::Leaf("b")).into(),
                TestOperator::from(TestExpr::Node {
                    input: TestOperator::from(TestExpr::Leaf("a")).into(),
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

        let expr = TestOperator::from(TestExpr::Nodes {
            inputs: vec![
                TestOperator::from(TestExpr::Leaf("a")).into(),
                TestOperator::from(TestExpr::Leaf("b")).into(),
                TestOperator::from(TestExpr::Leaf("a")).into(),
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

        let expr = TestOperator::from(TestExpr::Leaf("a"));

        let (group, _) = memo.insert(expr);
        expect_memo(
            &memo,
            r#"
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestExpr::Leaf("a0"));
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

        let expr = TestOperator::from(TestExpr::Node {
            input: TestOperator::from(TestExpr::Leaf("a")).into(),
        });

        let (group, _) = memo.insert(expr);
        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestExpr::Leaf("a0"));
        let _ = memo.insert_member(&group.mexpr().inputs()[0], expr);

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

        let expr = TestOperator::from(TestExpr::Node {
            input: TestOperator::from(TestExpr::Leaf("a")).into(),
        });

        let (group, _) = memo.insert(expr);
        expect_memo(
            &memo,
            r#"
01 Node 00
00 Leaf a
"#,
        );

        let expr = TestOperator::from(TestExpr::Node {
            input: TestOperator::from(TestExpr::Leaf("a0")).into(),
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

        let expr = TestOperator::from(TestExpr::Leaf("a"));
        let (group, expr) = memo.insert(expr);

        assert_eq!(format!("{:?}", group), "MemoGroupRef { id: GroupId(0) }");
        assert_eq!(format!("{:?}", expr), "MemoExprRef { id: ExprId(0) }");

        let mut mexpr_iter = group.mexprs();

        assert_eq!(format!("{:?}", mexpr_iter), "MemoGroupIter([MemoExprRef { id: ExprId(0) }])");
        mexpr_iter.next();
        assert_eq!(format!("{:?}", mexpr_iter), "MemoGroupIter([])");
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

    fn expect_memo_with_attrs(memo: &Memo<TestOperator>, expected: &str) {
        let lines: Vec<String> = expected.split('\n').map(String::from).collect();
        let expected = lines.join("\n");

        let buf = format_test_memo(memo, true);
        assert_eq!(buf.trim(), expected.trim());
    }

    fn format_test_memo(memo: &Memo<TestOperator>, include_attrs: bool) -> String {
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
                    if include_attrs {
                        if let Some(attrs) = group.attrs.as_ref() {
                            f.write_value("a", &attrs.a)
                        }
                    }
                }
            }
            f.push('\n');
        }
        buf
    }
}
