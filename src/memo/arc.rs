use std::ops::Deref;
use triomphe::Arc;

/// A smart pointer used in [Memo](super::Memo).
/// It is a simple wrapper around [triomphe::Arc] to hide it from the public API.
pub struct MemoArc<T: ?Sized> {
    inner: Arc<T>,
}

impl<T> MemoArc<T> {
    /// Creates a new Arc pointer.
    pub(crate) fn new(value: T) -> Self {
        MemoArc { inner: Arc::new(value) }
    }

    /// Returns the raw pointer.
    pub(crate) fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }
}

impl<T> Deref for MemoArc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<T> AsRef<T> for MemoArc<T> {
    fn as_ref(&self) -> &T {
        self.inner.as_ref()
    }
}

impl<T> Clone for MemoArc<T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        MemoArc {
            inner: self.inner.clone(),
        }
    }
}
