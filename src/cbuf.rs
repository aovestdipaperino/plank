//! `malloc`-allocated buffers handed to C code that takes ownership.
//!
//! Most of plank's FFI hands the C engine a borrowed pointer for the duration
//! of a call. A few entry points instead *take ownership* of the buffer and
//! expect the caller to have allocated it the way C `free` can release it:
//! `ds4_prompt_append_vision` moves an embedding's `data` pointer into the
//! output span (`span->embedding = *embedding`, no copy), and the span is later
//! released with `ds4_vision_embedding_free`, which is a plain `free`.
//!
//! Passing a `Vec<f32>`'s pointer into such an entry point is a double free:
//! the C side frees it, and so does the `Vec`. On macOS Rust's global allocator
//! *is* the system malloc, so the first free succeeds and the second aborts the
//! process with `malloc: pointer being freed was not allocated` — a crash that
//! surfaces far from the aliasing that caused it. This module is the one place
//! that mints buffers for those entry points.

/// Copies `src` into a fresh `malloc` allocation and returns the pointer,
/// transferring ownership to the caller (and, in practice, onward to C code
/// that will `free` it).
///
/// Returns null for an empty slice — there is nothing to own — and also if the
/// allocation fails, which the C validates for rather than dereferencing.
#[must_use]
pub fn malloc_copy_f32(src: &[f32]) -> *mut f32 {
    if src.is_empty() {
        return std::ptr::null_mut();
    }
    let bytes = std::mem::size_of_val(src);
    // SAFETY: `bytes` is non-zero, so this is a well-formed allocation request.
    let raw = unsafe { libc::malloc(bytes) }.cast::<f32>();
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `raw` owns `bytes` freshly allocated, uninitialized bytes, which
    // is exactly `src`'s length in floats and cannot overlap it.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), raw, src.len()) };
    raw
}

#[cfg(test)]
mod tests {
    use super::malloc_copy_f32;

    /// The copy is independent of the source and readable through the raw
    /// pointer, and `free` accepts it — the property the FFI ownership
    /// transfer depends on.
    #[test]
    fn a_malloc_copy_round_trips_and_is_freeable() {
        let src = vec![1.0_f32, -2.5, 3.25, 0.0];
        let raw = malloc_copy_f32(&src);
        assert!(!raw.is_null());
        // SAFETY: `raw` owns `src.len()` initialized floats.
        let copy = unsafe { std::slice::from_raw_parts(raw, src.len()) };
        assert_eq!(copy, src.as_slice());
        drop(src);
        // SAFETY: allocated by `malloc` above and not yet freed. A double free
        // here would abort the test process, which is the point of the test.
        unsafe { libc::free(raw.cast()) };
    }

    /// An empty embedding owns nothing, so it gets a null pointer rather than a
    /// zero-sized allocation the C would try to dereference.
    #[test]
    fn an_empty_slice_yields_null() {
        assert!(malloc_copy_f32(&[]).is_null());
    }
}
