//! A test-only global allocator that remembers the largest single request
//! each thread makes, so a test can prove an allocation stayed bounded even
//! where the unbounded request would have succeeded (macOS hands out large
//! untouched allocations lazily) or aborted the whole test binary.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

fn note(size: usize) {
    // `try_with` because the allocator also runs while thread locals are
    // being torn down.
    let _ = LARGEST.try_with(|c| c.set(c.get().max(size)));
}

struct Tracking;

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged; the bookkeeping neither allocates nor touches the memory.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Tracking = Tracking;

/// Runs `f` and returns its result with the largest single allocation this
/// thread requested while it ran.
pub(crate) fn largest_allocation<R>(f: impl FnOnce() -> R) -> (R, usize) {
    LARGEST.with(|c| c.set(0));
    let r = f();
    (r, LARGEST.with(Cell::get))
}
