//! Physical properties. See [PhysicalProperties].

use crate::meta::ColumnId;
use itertools::Itertools;
use std::fmt::{Display, Formatter};

use crate::properties::OrderingChoice;

/// Physical properties of an operator.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct PhysicalProperties {
    /// Physical properties required by an operator.
    pub required: Option<RequiredProperties>,
    /// Presentation defines names and an order of columns returned by an operator.
    pub presentation: Option<Presentation>,
}

impl PhysicalProperties {
    /// Returns physical properties object that has no requirements.
    pub const fn none() -> Self {
        PhysicalProperties {
            required: None,
            presentation: None,
        }
    }

    /// Creates a new physical properties with the given requirements.
    pub fn with_required(required: RequiredProperties) -> Self {
        PhysicalProperties {
            required: Some(required),
            presentation: None,
        }
    }
}

impl Default for PhysicalProperties {
    fn default() -> Self {
        PhysicalProperties::none()
    }
}

impl Display for PhysicalProperties {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ ")?;
        if let Some(required) = self.required.as_ref() {
            write!(f, "required: {}", required)?;
        }
        write!(f, " }}")
    }
}

/// Physical properties required by an operator.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RequiredProperties {
    ordering: Option<OrderingChoice>,
}

impl RequiredProperties {
    /// Creates properties that require the given ordering.
    pub fn new_with_ordering(ordering: OrderingChoice) -> Self {
        RequiredProperties {
            ordering: Some(ordering),
        }
    }

    /// Returns `true` if there are any required properties.
    pub fn is_some(&self) -> bool {
        self.ordering.is_some()
    }

    /// Returns the required [ordering](OrderingChoice).
    pub fn ordering(&self) -> Option<&OrderingChoice> {
        self.ordering.as_ref()
    }

    /// Returns new required properties that do not require [ordering](OrderingChoice).
    pub fn without_ordering(self) -> RequiredProperties {
        RequiredProperties { ordering: None }
    }
}

impl Display for RequiredProperties {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ ")?;
        if let Some(ordering) = self.ordering.as_ref() {
            write!(f, "ordering: {}", ordering)?;
        }
        write!(f, " }}")
    }
}

/// Presentation defines columns in the order expected by the operator.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct Presentation {
    /// Columns in the order expected by the operator.
    pub columns: Vec<(String, ColumnId)>,
}

impl Display for Presentation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ ")?;
        let output_columns = self.columns.iter().map(|c| format!("{}=col:{}", c.0, c.1)).join(", ");
        write!(f, "columns: {}", output_columns)?;
        write!(f, " }}")
    }
}
