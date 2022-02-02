use std::fmt::{Display, Formatter};

use crate::properties::OrderingChoice;

/// Physical properties required by an operator.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct PhysicalProperties {
    inner: Option<Values>,
}

impl PhysicalProperties {
    /// Returns physical properties object that has no requirements.
    pub const fn none() -> Self {
        PhysicalProperties { inner: None }
    }

    /// Creates a new physical properties object that require the the given ordering.
    pub fn new(ordering: OrderingChoice) -> Self {
        let values = Values {
            ordering: Some(ordering),
        };
        PhysicalProperties { inner: Some(values) }
    }

    /// Returns `true` if this physical properties object has no requirements.
    pub fn is_empty(&self) -> bool {
        self.inner.is_none()
    }

    /// Returns the required ordering.
    pub fn ordering(&self) -> Option<&OrderingChoice> {
        match &self.inner {
            None => None,
            Some(p) => {
                assert!(p.ordering.is_some(), "Unexpected physical property. Only ordering is supported: {:?}", self);
                p.ordering.as_ref()
            }
        }
    }

    /// Returns this physical properties object in the form of an [`Option`](std::option::Option).
    pub fn as_option(&self) -> Option<Option<&OrderingChoice>> {
        self.inner.as_ref().map(|_| self.ordering())
    }

    /// Returns a copy of this physical properties without ordering requirements.
    pub fn without_ordering(self) -> PhysicalProperties {
        PhysicalProperties::none()
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
        if let Some(Some(ordering)) = self.as_option() {
            write!(f, "ordering: {}", ordering)?;
        }
        write!(f, " }}")
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Values {
    ordering: Option<OrderingChoice>,
}
