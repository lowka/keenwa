use itertools::Itertools;
use std::alloc::Layout;
use std::fmt::{Debug, Formatter, Write};
use std::ptr::NonNull;

/// Used by [memo](crate::memo::Memo) to store expressions. This data structure stores its data
/// at pages of equal size and allows access to them via [references](self::StoreElementRef)
/// and [opaque identifiers](self::StoreElementId).
///
/// # Safety
///
/// It is callers responsibility to ensure that references to elements of this store does not outlive the store that created them.
pub struct Store<T> {
    page_size: usize,
    len: usize,
    pages: Vec<Page<T>>,
}

impl<T> Store<T> {
    /// Creates a new store with the given number of elements per page.
    ///
    /// # Panics
    ///
    /// Panics if `T` is a zero sized type.
    ///
    pub fn new(page_size: usize) -> Self {
        assert_ne!(std::mem::size_of::<T>(), 0, "Zero sided types are not supported");
        Store {
            page_size,
            len: 0,
            pages: Vec::with_capacity(16),
        }
    }

    /// Inserts the given element to this storage and return both its [identifier](self::StoreElementId)
    /// and a [reference](self::StoreElementRef) to that element.
    pub fn insert(&mut self, elem: T) -> (StoreElementId, StoreElementRef<T>) {
        if self.should_allocate_new_page() {
            self.allocate_new_page()
        };

        // This is a newly allocated page or a page that is not full.
        let page = self.pages.last_mut().unwrap();
        let elem_ref = page.insert(elem);
        let elem_id = self.next_id();

        self.len += 1;

        (elem_id, elem_ref)
    }

    /// Returns a reference to an element with the given id.
    pub fn get(&self, elem_id: StoreElementId) -> Option<StoreElementRef<T>> {
        let (page_idx, elem_idx) = self.to_page_index(elem_id);

        self.pages.get(page_idx).map(|p| p.get(elem_idx)).flatten()
    }

    /// Returns a mutable reference to an element with the given id.
    pub fn get_mut(&mut self, elem_id: StoreElementId) -> Option<&mut T> {
        let (page_idx, elem_idx) = self.to_page_index(elem_id);

        self.pages.get_mut(page_idx).map(|p| p.get_mut(elem_idx)).flatten()
    }

    /// Returns the number of elements in the store.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this store has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of allocated pages.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Returns the identifier that will be assigned to the next element.
    pub fn next_id(&self) -> StoreElementId {
        StoreElementId(self.len)
    }

    /// Returns the number of used by allocated page.
    pub fn allocated_bytes(&self) -> usize {
        self.pages.len() * std::mem::size_of::<Page<T>>()
    }

    /// Returns an iterator over elements of this store.
    pub fn iter(&self) -> StoreElementIter<T> {
        StoreElementIter {
            store: self,
            position: 0,
        }
    }

    fn should_allocate_new_page(&self) -> bool {
        if self.pages.is_empty() {
            true
        } else {
            let last_page = &self.pages[self.pages.len() - 1];
            last_page.is_full()
        }
    }

    fn allocate_new_page(&mut self) {
        let layout = Layout::array::<T>(self.page_size).unwrap();
        let total_size = self.pages.len() * layout.size();

        // Ensure that the new allocation doesn't exceed `isize::MAX` bytes.
        assert!(total_size <= isize::MAX as usize, "Allocation too large");

        let new_ptr = unsafe { std::alloc::alloc(layout) };

        // If allocation fails, `new_ptr` will be null, in which case we abort.
        let page = match NonNull::new(new_ptr as *mut T) {
            Some(ptr) => Page {
                ptr,
                cap: self.page_size,
                len: 0,
            },
            None => std::alloc::handle_alloc_error(layout),
        };
        self.pages.push(page);
    }

    fn to_page_index(&self, elem_id: StoreElementId) -> (usize, usize) {
        let page_idx = elem_id.0 / self.page_size;
        let elem_idx = elem_id.0 % self.page_size;

        (page_idx, elem_idx)
    }
}

impl<T> Debug for Store<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let itr = self.iter();
        f.write_str("[")?;
        f.write_str(itr.map(|r| format!("{:?}", r.as_ref())).join(", ").as_str())?;
        f.write_str("]")
    }
}

/// A reference to an element of a [storage](self::Store).
/// Internally it stores a raw pointer to the location of memory that stores that element.
pub struct StoreElementRef<T> {
    ptr: NonNull<T>,
}

impl<T> StoreElementRef<T> {
    fn new(ptr: *mut T) -> Self {
        StoreElementRef {
            // ptr is always non null
            ptr: NonNull::new(ptr).unwrap(),
        }
    }

    /// Returns a reference to the underlying element.
    pub fn as_ref(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T> Clone for StoreElementRef<T> {
    fn clone(&self) -> Self {
        StoreElementRef { ptr: self.ptr }
    }
}

impl<T> PartialEq for StoreElementRef<T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }
}

impl<T> Eq for StoreElementRef<T> {}

unsafe impl<T: Send> Send for StoreElementRef<T> {}
unsafe impl<T: Sync> Sync for StoreElementRef<T> {}

/// An identifier of an element in a store.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct StoreElementId(pub usize);

impl StoreElementId {
    pub fn index(&self) -> usize {
        self.0
    }
}

struct Page<T> {
    ptr: NonNull<T>,
    cap: usize,
    len: usize,
}

impl<T> Page<T> {
    fn insert(&mut self, elem: T) -> StoreElementRef<T> {
        // Caller guarantees that this page is not full
        let elem_ref = unsafe {
            let elem_ptr = self.ptr.as_ptr().add(self.len);
            std::ptr::write(elem_ptr, elem);
            StoreElementRef::new(elem_ptr)
        };
        self.len += 1;

        elem_ref
    }

    fn get(&self, idx: usize) -> Option<StoreElementRef<T>> {
        if idx < self.len {
            let elem_ref = unsafe {
                let elem_ptr = self.ptr.as_ptr().add(idx);
                StoreElementRef::new(elem_ptr)
            };
            Some(elem_ref)
        } else {
            None
        }
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        if idx < self.len {
            let elem_ref = unsafe {
                let elem_ptr = self.ptr.as_ptr().add(idx);
                &mut *elem_ptr
            };
            Some(elem_ref)
        } else {
            None
        }
    }

    fn is_full(&self) -> bool {
        self.len == self.cap
    }
}

impl<T> Drop for Page<T> {
    fn drop(&mut self) {
        let layout = Layout::array::<T>(self.cap).unwrap();
        unsafe {
            std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

// because NonNull is neither Sync nor Send.
unsafe impl<T: Send> Send for Page<T> {}
unsafe impl<T: Sync> Sync for Page<T> {}

/// An iterator over elements held by a store.
pub struct StoreElementIter<'a, T> {
    store: &'a Store<T>,
    position: usize,
}

impl<T> Iterator for StoreElementIter<'_, T> {
    type Item = StoreElementRef<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.store.len() {
            let elem_ref = self.store.get(StoreElementId(self.position));
            self.position += 1;
            elem_ref
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.store.len, Some(self.store.len))
    }
}

impl<T> DoubleEndedIterator for StoreElementIter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.position < self.store.len() {
            let p = self.store.len() - self.position - 1;
            let elem_ref = self.store.get(StoreElementId(p));
            self.position += 1;
            elem_ref
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use crate::memo::store::{Store, StoreElementId};
    use std::fmt::Debug;

    #[derive(Debug, Eq, PartialEq, Clone)]
    struct Value(usize);

    #[test]
    fn store_is_empty_initially() {
        let store = Store::<Value>::new(2);
        expect_num_pages(&store, 0);
        expect_allocated_bytes(&store, 0);

        assert!(store.get(StoreElementId(0)).is_none(), "no elements");
    }

    #[test]
    fn add_the_first_element() {
        let mut store = Store::new(2);

        expect_added(&mut store, Value(1));
        expect_num_pages(&store, 1);
    }

    #[test]
    fn new_page_is_added_when_a_page_is_full() {
        let mut store = Store::new(3);

        expect_added(&mut store, Value(1));
        expect_num_pages(&store, 1);

        expect_added(&mut store, Value(2));
        expect_num_pages(&store, 1);

        expect_added(&mut store, Value(3));
        expect_num_pages(&store, 1);

        expect_added(&mut store, Value(4));
        expect_num_pages(&store, 2);
    }

    #[test]
    fn store_is_send() {
        let store = Store::<String>::new(3);

        let _r = std::thread::spawn(|| {
            let v = vec![store];
            v
        })
        .join()
        .unwrap();
    }

    #[test]
    fn move_does_not_invalidates_element_refs() {
        let mut store = Store::new(4);
        let (_, ref1) = store.insert(String::from("a"));
        let (_, ref2) = store.insert(String::from("b"));

        let _moved_store = std::thread::spawn(|| store).join().unwrap();

        assert_eq!("a", ref1.as_ref());
        assert_eq!("b", ref2.as_ref());
    }

    #[test]
    fn slot_is_send() {
        let mut store = Store::new(4);
        let (_, elem_ref) = store.insert(String::from("a"));
        let elem_ref = std::thread::spawn(move || elem_ref).join().unwrap();

        assert_eq!("a", elem_ref.as_ref());
    }

    fn expect_added<T>(store: &mut Store<T>, value: T)
    where
        T: Clone + PartialEq + Debug,
    {
        let (id, elem_ref) = store.insert(value.clone());
        assert_eq!(&value, elem_ref.as_ref(), "inserted with id {:?}", id);

        let elem_ref = store.get(id).unwrap();
        assert_eq!(&value, elem_ref.as_ref(), "retrieved by id {:?}", id);
    }

    fn expect_num_pages<T>(store: &Store<T>, expected: usize) {
        assert_eq!(expected, store.num_pages(), "num_pages")
    }

    fn expect_allocated_bytes<T>(store: &Store<T>, expected: usize) {
        assert_eq!(expected, store.allocated_bytes(), "allocated_bytes")
    }
}
