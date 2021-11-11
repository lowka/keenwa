use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::properties::OrderingChoice;

/// Physical properties required by an operator.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct PhysicalProperties {
    inner: Option<Rc<Values>>,
}

impl PhysicalProperties {
    const EMPTY: PhysicalProperties = PhysicalProperties { inner: None };

    /// Returns physical properties object that has no requirements.
    pub const fn none() -> Self {
        PhysicalProperties::EMPTY
    }

    /// Creates a new physical properties object that require the the given ordering.
    pub fn new(ordering: OrderingChoice) -> Self {
        let values = Values {
            ordering: Some(ordering),
        };
        PhysicalProperties {
            inner: Some(Rc::new(values)),
        }
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

    /// Returns this physical properties object in the form of an [`Option`](std::core::Option).
    pub fn as_option(&self) -> Option<Option<&OrderingChoice>> {
        self.inner.as_ref().map(|_| self.ordering())
    }
}

impl Default for PhysicalProperties {
    fn default() -> Self {
        PhysicalProperties::none()
    }
}

impl Clone for PhysicalProperties {
    fn clone(&self) -> Self {
        if self.is_empty() {
            PhysicalProperties::EMPTY
        } else {
            PhysicalProperties {
                inner: self.inner.clone(),
            }
        }
    }
}

impl Display for PhysicalProperties {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ ")?;
        match self.as_option() {
            Some(Some(ordering)) => {
                write!(f, "ordering: {}", ordering)?;
            }
            _ => {}
        }
        write!(f, " }}")
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Values {
    ordering: Option<OrderingChoice>,
}
