use crate::error::OptimizerError;
use crate::meta::{ColumnId, ColumnMetadata};
use crate::operators::builder::scope::OperatorScope;
use crate::operators::builder::{OperatorBuilder, RewriteExprs, ValidateProjectionExpr};
use crate::operators::scalar::types::resolve_expr_type;
use crate::operators::scalar::{ScalarExpr, ScalarNode};

static NOT_NAMED_COLUMN: &str = "?column?";

pub(super) struct ProjectionListBuilder<'a> {
    builder: &'a mut OperatorBuilder,
    scope: &'a OperatorScope,
    projection: ProjectionList,
    allow_aggregates: bool,
}

impl<'a> ProjectionListBuilder<'a> {
    pub fn new(builder: &'a mut OperatorBuilder, scope: &'a OperatorScope, allow_aggregates: bool) -> Self {
        ProjectionListBuilder {
            builder,
            scope,
            projection: ProjectionList::default(),
            allow_aggregates,
        }
    }

    pub fn add_expr(&mut self, expr: ScalarExpr) -> Result<(ColumnId, &ScalarExpr), OptimizerError> {
        let i = self.projection.projection.len();
        match expr {
            ScalarExpr::Wildcard(qualifier) => {
                let columns = self.scope.resolve_columns(qualifier.as_deref(), &self.builder.metadata)?;

                for (name, id) in columns {
                    self.add_column(id, name)?;
                }
            }
            _ => {
                let metadata = self.builder.metadata.clone();
                let mut rewriter = RewriteExprs::new(self.scope, metadata, ValidateProjectionExpr::projection_expr());
                let expr = expr.rewrite(&mut rewriter)?;
                match expr {
                    ScalarExpr::Column(id) => {
                        let name = {
                            // Should never panic because RewriteExprs set an error when encounters an unknown column id.
                            let meta = self.builder.metadata.get_column(&id);
                            meta.name().to_string()
                        };
                        self.add_column(id, name)?
                    }
                    ScalarExpr::Alias(ref inner_expr, ref name) => {
                        //Stores inner expression in meta column metadata.
                        self.add_synthetic_column(expr.clone(), name.clone(), *inner_expr.clone())?
                    }
                    ScalarExpr::Aggregate { ref func, .. } => {
                        self.add_synthetic_column(expr.clone(), format!("{}", func), expr)?
                    }
                    ScalarExpr::WindowAggregate { ref func, .. } => {
                        self.add_synthetic_column(expr.clone(), format!("{}", func), expr)?
                    }
                    ScalarExpr::SubQuery(ref rel) => {
                        // FIXME: presentation is absent for this subquery. Return an error?
                        let col_name = rel
                            .props()
                            .physical
                            .presentation
                            .as_ref()
                            .map(|p| p.columns[0].0.clone())
                            .unwrap_or_else(|| String::from(NOT_NAMED_COLUMN));
                        self.add_synthetic_column(expr.clone(), col_name, expr)?
                    }
                    _ => self.add_synthetic_column(expr.clone(), NOT_NAMED_COLUMN.into(), expr)?,
                };
            }
        };

        let column_id = self.projection.column_ids[i];
        let expr = self.projection.projection[i].expr();
        Ok((column_id, expr))
    }

    fn add_column(&mut self, id: ColumnId, name: String) -> Result<(), OptimizerError> {
        let expr = self.builder.add_scalar_node(ScalarExpr::Column(id), self.scope)?;

        self.projection.add_expr(id, name, expr);

        Ok(())
    }

    fn add_synthetic_column(
        &mut self,
        expr: ScalarExpr,
        name: String,
        column_expr: ScalarExpr,
    ) -> Result<(), OptimizerError> {
        if matches!(&expr, ScalarExpr::Aggregate { .. }) && !self.allow_aggregates {
            return Err(OptimizerError::argument(
                "Aggregate expressions are not allowed in projection operator. Use AggregateBuilder to construct aggregate operators",
            ));
        }

        let data_type = resolve_expr_type(&expr, &self.builder.metadata)?;
        let column_meta = ColumnMetadata::new_synthetic_column(name.clone(), data_type, Some(column_expr));
        let id = self.builder.metadata.add_column(column_meta);
        let expr = self.builder.add_scalar_node(expr, self.scope)?;

        self.projection.add_expr(id, name, expr);

        Ok(())
    }

    pub fn build(self) -> ProjectionList {
        self.projection
    }
}

#[derive(Default)]
pub struct ProjectionList {
    pub column_ids: Vec<ColumnId>,
    pub output_columns: Vec<(String, ColumnId)>,
    pub projection: Vec<ScalarNode>,
}

impl ProjectionList {
    fn add_expr(&mut self, id: ColumnId, name: String, expr: ScalarNode) {
        self.projection.push(expr);
        self.column_ids.push(id);
        self.output_columns.push((name, id));
    }
}
