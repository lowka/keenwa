use crate::error::OptimizerError;
use crate::meta::{ColumnId, MetadataRef, MutableMetadata, RelationId};
use crate::operators::builder::TableAlias;
use crate::operators::relational::RelNode;
use crate::operators::scalar::exprs;
use std::collections::HashMap;
use std::rc::Rc;

/// Stores columns and relations visible to/accessible from the current node of an operator tree.
#[derive(Debug, Clone)]
pub struct OperatorScope {
    //FIXME: It is not necessary to store output relation in both `relation` and `relations` fields.
    relation: RelationInScope,
    relations: RelationsInScope,
    // FIXME: Rewrite OperatorScope so it uses the parent scope.
    //  (currently parent is never set to anything but None)
    parent: Option<Rc<OperatorScope>>,
    // FIXED: store relations that are not accessible from this scope.
    //  (eg. SELECT .. FROM a JOIN (SELECT * FROM b1 WHERE b1 = a1)
    //    in the above query column a1 should not be accessible
    // columns from the outer scope.
    outer_columns: Vec<ColumnId>,
    // Common table expressions accessible from this scope.
    with: HashMap<String, CteInScope>,
}

/// Common table expression.
#[derive(Debug, Clone)]
pub struct CteInScope {
    /// The relational operator that represents this common table expression.
    pub expr: RelNode,
    /// The output columns of that operator. (<name>, id).
    pub relation: Vec<(String, ColumnId)>,
}

impl OperatorScope {
    pub fn new_source(relation: RelationInScope) -> Self {
        let relations = RelationsInScope::from_relation(relation.clone());

        OperatorScope {
            relation,
            parent: None,
            relations,
            outer_columns: Vec::new(),
            with: HashMap::new(),
        }
    }

    pub fn from_columns(columns: Vec<(String, ColumnId)>, outer_columns: Vec<ColumnId>) -> Self {
        let relation = RelationInScope::from_columns(columns);
        let relations = RelationsInScope::from_relation(relation.clone());

        OperatorScope {
            relation,
            parent: None,
            relations,
            outer_columns,
            with: HashMap::new(),
        }
    }

    pub fn new_child_scope(&self, metadata: MetadataRef) -> Self {
        let outer_columns = self.get_outer_columns(metadata);

        let mut relations = RelationsInScope::from_relation(self.relation.clone());
        relations.add_all(self.relations.clone());

        OperatorScope {
            relation: RelationInScope::from_columns(vec![]),
            parent: Some(Rc::new(self.clone())),
            relations,
            outer_columns,
            with: self.with.clone(),
        }
    }

    pub fn new_scope(&self, metadata: MetadataRef) -> Self {
        let outer_columns = self.get_outer_columns(metadata);
        let relations = if self.relation.columns.is_empty() {
            // The current expression has not been added to the operator tree
            assert!(!self.relations.relations.is_empty(), "Unexpected number of relations");
            RelationsInScope::from_relation(self.relations.relations[0].clone())
        } else {
            RelationsInScope::new()
        };

        OperatorScope {
            relation: RelationInScope::from_columns(vec![]),
            parent: None,
            relations,
            outer_columns,
            with: self.with.clone(),
        }
    }

    pub fn with_new_columns(&self, columns: Vec<(String, ColumnId)>) -> Self {
        let relation = RelationInScope::from_columns(columns);
        let relations = RelationsInScope::from_relation(relation.clone());

        OperatorScope {
            relation,
            parent: None,
            relations,
            outer_columns: self.outer_columns.clone(),
            with: self.with.clone(),
        }
    }

    pub fn join(self, right: OperatorScope) -> Self {
        let mut columns = self.relation.columns;
        columns.extend_from_slice(&right.relation.columns);

        let mut relations = self.relations;
        relations.add(right.relation);
        relations.add_all(right.relations);

        OperatorScope {
            relation: RelationInScope::from_columns(columns),
            relations,
            parent: self.parent,
            // Use outer column from the left side of the join
            // because the right side contain the subset of those columns.
            outer_columns: self.outer_columns,
            with: self.with,
        }
    }

    pub fn set_relation(&mut self, relation: RelationInScope) {
        // ??? Update relations fields
        self.relation = relation
    }

    pub fn set_outer_columns(&mut self, outer_columns: Vec<ColumnId>) {
        self.outer_columns = outer_columns;
    }

    pub fn add_relations(&mut self, other: OperatorScope) {
        self.relations.add_all(other.relations);
    }

    pub fn set_alias(&mut self, alias: TableAlias, metadata: &MutableMetadata) -> Result<(), OptimizerError> {
        fn set_alias(
            relation: &mut RelationInScope,
            alias: String,
            renamed_columns: Vec<String>,
            metadata: &MutableMetadata,
        ) -> Result<(), OptimizerError> {
            let renamed_columns_len = renamed_columns.len();
            // rename first len(columns) from the relation.
            let mut columns: Vec<(String, ColumnId)> =
                renamed_columns.into_iter().enumerate().map(|(i, c)| (c, relation.columns[i].1)).collect();
            // use remaining columns from the relation.
            columns.extend(relation.columns.drain(renamed_columns_len..));

            if relation.name.is_empty() {
                if let Some(relation_id) = relation.relation_id {
                    metadata.rename_relation(&relation_id, alias.clone());
                }
                relation.name = alias.clone();
                relation.alias = Some(alias);
                relation.columns = columns;
                Ok(())
            } else if relation.relation_id.is_some() {
                relation.alias = Some(alias);
                relation.columns = columns;
                Ok(())
            } else {
                let message = format!("BUG: a relation has no relation_id but has a name. Relation: {:?}", relation);
                Err(OptimizerError::internal(message))
            }
        }

        if let Some(relation) = self.relations.relations.first_mut() {
            let output_relation = &mut self.relation;
            let TableAlias { name: alias, columns } = alias;

            set_alias(relation, alias.clone(), columns.clone(), metadata)?;
            set_alias(output_relation, alias, columns, metadata)
        } else {
            Err(OptimizerError::internal("OperatorScope is empty!"))
        }
    }

    pub fn resolve_columns(
        &self,
        qualifier: Option<&str>,
        metadata: &MutableMetadata,
    ) -> Result<Vec<(String, ColumnId)>, OptimizerError> {
        if let Some(qualifier) = qualifier {
            let relation = self.relations.find_relation(qualifier);

            match relation {
                Some(relation) => {
                    let columns = relation
                        .columns
                        .iter()
                        .map(|(_, id)| {
                            // Should never panic because relation contains only valid ids.
                            let meta = metadata.get_column(id);
                            let column_name = meta.name().to_string();
                            (column_name, *id)
                        })
                        .collect();
                    Ok(columns)
                }
                None if self.relations.find_relation_by_name(qualifier).is_some() => {
                    Err(OptimizerError::argument(format!("Invalid reference to relation {}", qualifier)))
                }
                None => Err(OptimizerError::argument(format!("Unknown relation {}", qualifier))),
            }
        } else {
            Ok(self.relation.columns.to_vec())
        }
    }

    pub fn columns(&self) -> &[(String, ColumnId)] {
        &self.relation.columns
    }

    pub fn find_column_by_name(&self, name: &str) -> Option<ColumnId> {
        let f = |r: &RelationInScope, rs: &RelationsInScope| {
            r.find_column_by_name(name).or_else(|| rs.find_column_by_name(name))
        };
        self.find_in_scope(&f)
    }

    pub fn find_column_by_id(&self, id: ColumnId) -> Option<ColumnId> {
        let f =
            |r: &RelationInScope, rs: &RelationsInScope| r.find_column_by_id(&id).or_else(|| rs.find_column_by_id(&id));
        self.find_in_scope(&f)
    }

    pub fn parent(&self) -> Option<Rc<OperatorScope>> {
        self.parent.clone()
    }

    pub fn into_columns(self) -> Vec<(String, ColumnId)> {
        self.relation.columns
    }

    pub fn output_relation(&self) -> &RelationInScope {
        &self.relation
    }

    pub fn outer_columns(&self) -> &[ColumnId] {
        &self.outer_columns
    }

    pub fn add_cte(&mut self, name: String, cte: CteInScope) -> bool {
        if self.with.insert(name, cte).is_some() {
            false
        } else {
            // Remove output columns otherwise they are going be to be added
            // to the outer scope of an operator (see get_outer_columns in new_scope/new_child_scope)
            self.relation.columns = Vec::new();
            for r in self.relations.relations.iter_mut() {
                if r.relation_id == self.relation.relation_id {
                    r.columns = Vec::new();
                }
            }
            true
        }
    }

    pub fn find_cte(&self, name: &str) -> Option<&CteInScope> {
        self.with.get(name)
    }

    fn find_in_scope<F>(&self, f: &F) -> Option<ColumnId>
    where
        F: Fn(&RelationInScope, &RelationsInScope) -> Option<ColumnId>,
    {
        let mut scope = self;
        loop {
            let relations = &scope.relations;
            if let Some(id) = (f)(&self.relation, relations) {
                return Some(id);
            } else if let Some(parent) = &scope.parent {
                scope = parent.as_ref();
            } else {
                return None;
            }
        }
    }

    fn get_outer_columns(&self, metadata: MetadataRef) -> Vec<ColumnId> {
        // Outer columns of a scope include:
        // + the outer columns from the scope.
        // + the columns returned by the output relation and the columns they reference
        // (in case when a column is an expression/a synthetic column) .
        let mut outer_columns = self.outer_columns.to_vec();

        outer_columns.extend(self.columns().iter().flat_map(|(_, id)| {
            let column = metadata.get_column(id);
            let mut columns = vec![*id];
            if let Some(expr) = column.expr() {
                columns.extend(exprs::collect_columns(expr));
            }
            columns
        }));

        outer_columns
    }
}

#[derive(Debug, Clone)]
struct RelationsInScope {
    relations: Vec<RelationInScope>,
}

impl RelationsInScope {
    fn new() -> Self {
        RelationsInScope { relations: Vec::new() }
    }

    fn from_relation(relation: RelationInScope) -> Self {
        if !relation.columns.is_empty() {
            RelationsInScope {
                relations: vec![relation],
            }
        } else {
            RelationsInScope::new()
        }
    }

    fn add(&mut self, relation: RelationInScope) {
        for r in self.relations.iter() {
            if r.relation_id == relation.relation_id {
                return;
            }
        }
        self.relations.push(relation);
    }

    fn add_all(&mut self, relations: RelationsInScope) {
        let existing: Vec<RelationId> = self.relations.iter().filter_map(|r| r.relation_id).collect();
        let new_relations = relations
            .relations
            .into_iter()
            .filter(|r| !matches!(&r.relation_id, Some(id) if existing.contains(id)));

        self.relations.extend(new_relations);
    }

    fn find_relation(&self, name_or_alias: &str) -> Option<&RelationInScope> {
        self.relations.iter().find(|r| r.has_name_or_alias(name_or_alias))
    }

    fn find_relation_by_name(&self, name: &str) -> Option<&RelationInScope> {
        self.relations.iter().find(|r| r.has_name(name))
    }

    fn find_column_by_id(&self, id: &ColumnId) -> Option<ColumnId> {
        self.relations
            .iter()
            .find_map(|r| r.columns.iter().find(|(_, col_id)| col_id == id).map(|(_, id)| *id))
    }

    fn find_column_by_name(&self, name: &str) -> Option<ColumnId> {
        // FIXME: use object names instead of str.
        if let Some(pos) = name.find('.') {
            let (relation_name, col_name) = name.split_at(pos);
            let relation = self.relations.iter().find(|r| r.has_name_or_alias(relation_name));
            if let Some(relation) = relation {
                relation
                    .columns
                    .iter()
                    .find(|(name, _id)| name.eq_ignore_ascii_case(&col_name[1..]))
                    .map(|(_, id)| *id)
            } else {
                None
            }
        } else {
            self.relations.iter().find_map(|r| {
                r.columns
                    .iter()
                    .find(|(col_name, _)| col_name.eq_ignore_ascii_case(name))
                    .map(|(_, id)| *id)
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct RelationInScope {
    name: String,
    alias: Option<String>,
    relation_id: Option<RelationId>,
    // (name/alias, column_id)
    columns: Vec<(String, ColumnId)>,
}

impl RelationInScope {
    pub fn new(relation_id: RelationId, name: String, columns: Vec<(String, ColumnId)>) -> Self {
        RelationInScope {
            name,
            alias: None,
            relation_id: Some(relation_id),
            columns,
        }
    }

    pub fn from_columns(columns: Vec<(String, ColumnId)>) -> Self {
        RelationInScope {
            name: String::from(""),
            alias: None,
            relation_id: None,
            columns,
        }
    }

    pub fn columns(&self) -> &[(String, ColumnId)] {
        &self.columns
    }

    fn has_name_or_alias(&self, name_or_alias: &str) -> bool {
        if let Some(alias) = &self.alias {
            alias.eq_ignore_ascii_case(name_or_alias)
        } else {
            self.has_name(name_or_alias)
        }
    }

    fn has_name(&self, name: &str) -> bool {
        self.name.eq_ignore_ascii_case(name)
    }

    fn find_column_by_name(&self, name: &str) -> Option<ColumnId> {
        if let Some((table, name)) = name.split_once('.') {
            match table {
                _ if self.name != table => None,
                _ if self.alias.as_ref().map(|a| a == table) == Some(false) => None,
                _ => {
                    let f = |(col_name, _): &(String, ColumnId)| col_name == name;
                    self.find_column(&f)
                }
            }
        } else {
            let f = |(col_name, _): &(String, ColumnId)| col_name == name;
            self.find_column(&f)
        }
    }

    fn find_column_by_id(&self, id: &ColumnId) -> Option<ColumnId> {
        let f = |(_, col_id): &(String, ColumnId)| col_id == id;
        self.find_column(&f)
    }

    fn find_column<F>(&self, predicate: &F) -> Option<ColumnId>
    where
        F: Fn(&(String, ColumnId)) -> bool,
    {
        self.columns
            .iter()
            .find_map(|column| if (predicate)(column) { Some(column.1) } else { None })
    }
}
