use std::alloc::Layout;
use std::fmt::{Debug, Formatter};
use std::ptr::NonNull;

/// An implementation of arena allocator that provides references to allocated objects without using lifetimes
/// (see [ElementRef](self::ElementRef)). Each arena stores elements of the same type.
///
/// # Safety
///
/// * It is caller's responsibility to ensure that references to objects allocated by this arena does not outlive it.
/// * Caller must also guarantee they do not have both mutable and shared reference to the same object at the same time.
pub struct Arena<T> {
    // blocks of allocated memory.
    blocks: Vec<Block<T>>,
    // the number of objects per block.
    block_size: usize,
    // the total number of allocated objects.
    len: usize,
}

impl<T> Arena<T> {
    /// Creates a new arena with the given number of elements per block.
    ///
    /// # Panics
    ///
    /// Panics if `T` is a zero sized type.
    ///
    pub fn new(block_size: usize) -> Self {
        assert_ne!(std::mem::size_of::<T>(), 0, "Zero sided types are not supported");
        Arena {
            blocks: Vec::new(),
            block_size,
            len: 0,
        }
    }

    /// Allocates the given element into this arena and return both its [index](self::ElementIndex)
    /// and a [reference](self::ElementRef) to that element.
    pub fn allocate(&mut self, elem: T) -> (ElementIndex, ElementRef<T>) {
        if self.should_allocate_new_block() {
            self.allocate_new_block()
        }

        // If the current block is full or arena is empty then allocate_new_block creates a new block.
        let block = self.blocks.last_mut().expect("No block to store allocated objects");
        let elem_ref = block.insert(elem);
        let elem_idx = self.next_idx();

        self.len += 1;

        (elem_idx, elem_ref)
    }

    /// Returns a reference to an element with the given index.
    pub fn get(&self, elem_idx: ElementIndex) -> Option<ElementRef<T>> {
        let (block_idx, elem_idx) = self.to_block_index(elem_idx);

        self.blocks.get(block_idx).map(|p| p.get(elem_idx)).flatten()
    }

    /// Returns a mutable reference to an element with the given index.
    ///
    /// # Safety
    ///
    /// It is the responsibility of a caller to ensure that
    /// no aliasing reference to the element stored by the given index exists.
    pub unsafe fn get_mut(&mut self, elem_idx: ElementIndex) -> Option<&mut T> {
        let (block_idx, elem_idx) = self.to_block_index(elem_idx);

        self.blocks.get_mut(block_idx).map(|p| p.get_mut(elem_idx)).flatten()
    }

    /// Returns the number of elements by this arena.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this arena has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of allocated blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the index that will be assigned to the next element.
    pub fn next_idx(&self) -> ElementIndex {
        ElementIndex(self.len)
    }

    /// Returns the number of used by allocated block.
    pub fn allocated_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Block<T>>()
    }

    /// Returns an iterator over elements allocated by this arena.
    /// This method is marked unsafe for documentation purposes.
    ///
    /// # Safety
    ///
    /// A caller must guarantee that there no mutable references to elements allocated by this arena
    /// during the iteration.
    pub fn iter(&self) -> ElementsIter<T> {
        ElementsIter {
            arena: self,
            position: 0,
        }
    }

    fn should_allocate_new_block(&self) -> bool {
        if self.blocks.is_empty() {
            true
        } else {
            let current_block = &self.blocks[self.blocks.len() - 1];
            current_block.is_full()
        }
    }

    fn allocate_new_block(&mut self) {
        let layout = Layout::array::<T>(self.block_size).unwrap();
        let total_size = self.blocks.len() * layout.size();

        // FIXME: Do we really need this check ?
        // Ensure that the new allocation doesn't exceed `isize::MAX` bytes.
        assert!(total_size <= isize::MAX as usize, "Allocation too large");

        let new_ptr = unsafe { std::alloc::alloc(layout) };

        // If allocation fails, `new_ptr` will be null, in which case we abort.
        let block = match NonNull::new(new_ptr as *mut T) {
            Some(ptr) => Block {
                ptr,
                cap: self.block_size,
                len: 0,
            },
            None => std::alloc::handle_alloc_error(layout),
        };
        self.blocks.push(block);
    }

    fn to_block_index(&self, elem_id: ElementIndex) -> (usize, usize) {
        let block_idx = elem_id.0 / self.block_size;
        let elem_idx = elem_id.0 % self.block_size;

        (block_idx, elem_idx)
    }
}

/// A reference to an element held by [arena](self::Arena).
/// Internally it stores a raw pointer to the location of memory that stores that element.
pub struct ElementRef<T> {
    ptr: NonNull<T>,
}

impl<T> ElementRef<T> {
    fn new(ptr: *mut T) -> Self {
        ElementRef {
            // ptr is always non null
            ptr: NonNull::new(ptr).unwrap(),
        }
    }

    /// Returns a reference to the underlying element.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that there no mutable references to the underlying data.
    /// See [Arena::get_mut].
    pub unsafe fn get(&self) -> &T {
        self.ptr.as_ref()
    }

    /// Returns a raw pointer to the underlying element.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that there no mutable references to the underlying data.
    /// See [Arena::get_mut].
    pub unsafe fn get_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

impl<T> Clone for ElementRef<T> {
    fn clone(&self) -> Self {
        ElementRef { ptr: self.ptr }
    }
}

impl<T> PartialEq for ElementRef<T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }
}

impl<T> Eq for ElementRef<T> {}

unsafe impl<T: Send> Send for ElementRef<T> {}
unsafe impl<T: Sync> Sync for ElementRef<T> {}

/// An index of an element in an arena.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct ElementIndex(pub usize);

impl ElementIndex {
    pub fn index(&self) -> usize {
        self.0
    }
}

impl Debug for ElementIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Represents a block of memory that stores allocated objects.
struct Block<T> {
    ptr: NonNull<T>,
    cap: usize,
    len: usize,
}

impl<T> Block<T> {
    fn insert(&mut self, elem: T) -> ElementRef<T> {
        // SAFETY: Arena::allocate guarantees that this block is not full
        let elem_ref = unsafe {
            let elem_ptr = self.ptr.as_ptr().add(self.len);
            std::ptr::write(elem_ptr, elem);
            ElementRef::new(elem_ptr)
        };
        self.len += 1;

        elem_ref
    }

    fn get(&self, idx: usize) -> Option<ElementRef<T>> {
        if idx < self.len {
            // SAFETY: `idx` < `len` so it is safe.
            let elem_ref = unsafe {
                let elem_ptr = self.ptr.as_ptr().add(idx);
                ElementRef::new(elem_ptr)
            };
            Some(elem_ref)
        } else {
            None
        }
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        if idx < self.len {
            // SAFETY: `idx` < `len` so it is safe.
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

impl<T> Drop for Block<T> {
    fn drop(&mut self) {
        let layout = Layout::array::<T>(self.cap).unwrap();
        // SAFETY: This is safe because block stores exactly `len` elements.
        unsafe {
            for i in 0..self.len {
                let elem_ptr = self.ptr.as_ptr().add(i);
                let elem = std::ptr::read(elem_ptr);
                drop(elem)
            }
            std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

// because NonNull is neither Sync nor Send.
unsafe impl<T: Send> Send for Block<T> {}
unsafe impl<T: Sync> Sync for Block<T> {}

/// An iterator over elements allocated by an arena.
pub struct ElementsIter<'a, T> {
    arena: &'a Arena<T>,
    position: usize,
}

impl<T> Iterator for ElementsIter<'_, T> {
    type Item = ElementRef<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.arena.len() {
            let elem_ref = self.arena.get(ElementIndex(self.position));
            self.position += 1;
            elem_ref
        } else {
            None
        }
    }
}

impl<T> DoubleEndedIterator for ElementsIter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.position < self.arena.len() {
            let p = self.arena.len() - self.position - 1;
            let elem_ref = self.arena.get(ElementIndex(p));
            self.position += 1;
            elem_ref
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use crate::memo::unsafe_impl::arena::{Arena, ElementIndex, ElementRef};
    use std::fmt::Debug;

    #[derive(Debug, Eq, PartialEq, Clone)]
    struct Value(usize);

    #[test]
    fn test_arena_is_initially_empty() {
        let arena = Arena::<Value>::new(2);
        expect_num_blocks(&arena, 0);
        expect_allocated_bytes(&arena, 0);

        let elem = arena.get(ElementIndex(0));
        assert!(elem.is_none(), "no elements");
    }

    #[test]
    fn test_add_the_first_element() {
        let mut arena = Arena::new(2);

        expect_added(&mut arena, Value(1));
        expect_num_blocks(&arena, 1);
    }

    #[test]
    fn test_new_block_is_added_when_a_block_is_full() {
        let mut arena = Arena::new(3);

        expect_added(&mut arena, Value(1));
        expect_num_blocks(&arena, 1);

        expect_added(&mut arena, Value(2));
        expect_num_blocks(&arena, 1);

        expect_added(&mut arena, Value(3));
        expect_num_blocks(&arena, 1);

        expect_added(&mut arena, Value(4));
        expect_num_blocks(&arena, 2);
    }

    #[test]
    fn test_arena_is_send() {
        let arena = Arena::<String>::new(3);

        let _r = std::thread::spawn(|| {
            let v = vec![arena];
            v
        })
        .join()
        .unwrap();
    }

    #[test]
    fn test_move_does_not_invalidates_element_refs() {
        let mut arena = Arena::new(4);
        let (_, ref1) = arena.allocate(String::from("a"));
        let (_, ref2) = arena.allocate(String::from("b"));

        let _moved_store = std::thread::spawn(|| arena).join().unwrap();

        assert_eq!("a", read_safely(&ref1));
        assert_eq!("b", read_safely(&ref2));
    }

    #[test]
    fn test_element_ref_is_send() {
        let mut arena = Arena::new(4);
        let (_, elem_ref) = arena.allocate(String::from("a"));
        let elem_ref = std::thread::spawn(move || elem_ref).join().unwrap();

        assert_eq!("a", read_safely(&elem_ref));
    }

    fn expect_added<T>(arena: &mut Arena<T>, value: T)
    where
        T: Clone + PartialEq + Debug,
    {
        let (id, elem_ref) = arena.allocate(value.clone());
        assert_eq!(&value, read_safely(&elem_ref), "inserted with id {:?}", id);

        let elem_ref = arena.get(id).unwrap();
        assert_eq!(&value, read_safely(&elem_ref), "retrieved by id {:?}", id);
    }

    fn expect_num_blocks<T>(arena: &Arena<T>, expected: usize) {
        assert_eq!(expected, arena.num_blocks(), "num_blocks")
    }

    fn expect_allocated_bytes<T>(arena: &Arena<T>, expected: usize) {
        assert_eq!(expected, arena.allocated_bytes(), "allocated_bytes")
    }

    fn read_safely<T>(elem_ref: &ElementRef<T>) -> &T {
        unsafe { elem_ref.get() }
    }
}
