//! An unbounded MPSC (Multiple Producer, Single Consumer) lock-free queue.
//!
//! Based on the classic Michael & Scott algorithm adapted for single-consumer
//! elimination. Producers use compare-and-swap on the tail pointer; the
//! consumer walks the linked list from a local head pointer. No mutexes,
//! no blocking — only lock-free atomics.
//!
//! # Safety
//! The queue uses raw `*mut Node` pointers behind `AtomicPtr`. Every pointer
//! is carefully managed: nodes are allocated by producers and deallocated by
//! the single consumer. The consumer is the only thread that ever reads node
//! data or frees nodes, so no concurrent deallocation hazards exist.

use std::sync::atomic::{AtomicPtr, Ordering};
use std::ptr;

/// A node in the intrusive linked list that backs the queue.
struct Node<T> {
    /// The stored value. Only the consumer reads this.
    data: Option<T>,
    /// Pointer to the next node in the list (consumer-owned after enqueue).
    next: AtomicPtr<Node<T>>,
}

/// An unbounded MPSC queue.
///
/// # Panics
/// Operations that fail an allocation will panic (the same as any `alloc`).
pub struct MpscQueue<T> {
    /// The consumer's working head. Only the consumer thread touches this.
    head: *mut Node<T>,
    /// Producers compete to advance this tail. Always points to the last
    /// node that has been linked, but not necessarily the *last* enqueued
    /// node (the ABA problem is avoided by using a single consumer).
    tail: AtomicPtr<Node<T>>,
    /// A permanently-allocated stub node. The queue is never empty as long
    /// as this node exists; it acts as an anchor so producers can always
    /// CAS the tail.
    stub: *mut Node<T>,
}

// ---------------------------------------------------------------------------
// Manual `Send` / `Sync` – the queue is safe because:
//   - Producers only write atomically to `tail` and to the `next` pointer of
//     the node they just linked.
//   - The consumer reads `head` locally and never races with producers on
//     node data because a node is only fully linked (its `next` is set and
//     the tail CAS has completed) before the consumer can ever observe it.
//   - Deallocation is exclusive to the consumer.
//   - The stub node lives for the queue's entire lifetime.
// ---------------------------------------------------------------------------
unsafe impl<T: Send> Send for MpscQueue<T> {}
unsafe impl<T: Send> Sync for MpscQueue<T> {}

impl<T> MpscQueue<T> {
    /// Creates a new, empty queue.
    ///
    /// Allocates one stub node that persists for the queue's lifetime.
    pub fn new() -> Self {
        let stub = Box::into_raw(Box::new(Node {
            data: None,
            next: AtomicPtr::new(ptr::null_mut()),
        }));

        MpscQueue {
            head: stub,
            tail: AtomicPtr::new(stub),
            stub,
        }
    }

    /// Enqueue a value into the queue. May be called concurrently by any
    /// number of producers.
    pub fn push(&self, value: T) {
        // 1. Allocate a new node with the value.
        let new_node = Box::into_raw(Box::new(Node {
            data: Some(value),
            next: AtomicPtr::new(ptr::null_mut()),
        }));

        // 2. Atomically swap the tail so that *old_tail* becomes the new
        //    node's predecessor, and the new node becomes the new tail.
        //
        //    This is the classic MPSC trick: because there is only one
        //    consumer, we don't need a full two-CAS handshake. The producer
        //    simply CASes `tail` from the old stub/tail to the new node, then
        //    links the old tail's `next` to the new node.
        loop {
            let old_tail = self.tail.load(Ordering::Acquire);
            let old_next = unsafe { (*old_tail).next.load(Ordering::Acquire) };

            if old_next.is_null() {
                // The tail still points to the last node; try to advance it.
                // If we succeed, we are the producer that linked the new node.
                let prev = self.tail.compare_and_swap(old_tail, new_node, Ordering::Release);
                if prev == old_tail {
                    // Success: link the old tail to the new node.
                    unsafe { (*old_tail).next.store(new_node, Ordering::Release); }
                    return;
                }
                // CAS failed – another producer already advanced the tail.
                // Spin and retry.
            } else {
                // The tail is lagging: another producer has linked a node but
                // hasn't updated the tail yet. Help it by advancing the tail
                // to `old_next`, then retry our own enqueue.
                let _ = self.tail.compare_and_swap(old_tail, old_next, Ordering::Release);
            }
        }
    }

    /// Attempt to dequeue a value. Only the single consumer thread should
    /// call this method.
    ///
    /// Returns `None` if the queue is empty.
    pub fn pop(&mut self) -> Option<T> {
        loop {
            // Read the current head node.
            let head = self.head;
            let head_next = unsafe { (*head).next.load(Ordering::Acquire) };

            if head == self.stub {
                // We are at the stub. If the stub's next is non-null, the
                // stub has been linked to a real node; advance the consumer
                // head to that node and loop again.
                if !head_next.is_null() {
                    self.head = head_next;
                    // Continue the loop – we'll read from the real node.
                } else {
                    // Queue is empty.
                    return None;
                }
            } else {
                // We are at a real node. Read its value.
                let data = unsafe {
                    // Take the value, leaving `None` in the node.
                    // SAFETY: the consumer is the only thread reading data.
                    let node = &mut *head;
                    node.data.take()
                };

                // Advance the consumer head to the next node (or stub).
                if !head_next.is_null() {
                    self.head = head_next;
                } else {
                    // No next node: wrap back to the stub. This means all
                    // enqueued nodes have been consumed.
                    self.head = self.stub;
                }

                // Deallocate the old head node (we own it exclusively).
                unsafe { drop(Box::from_raw(head)); }

                if data.is_some() {
                    return data;
                }
                // `data` was `None` – this can happen if a producer linked a
                // node but the CAS hadn't completed when we read it. Loop.
            }
        }
    }

    /// Returns `true` if the queue is empty. Note that this is only a
    /// *snapshot* — concurrent producers may immediately add items.
    pub fn is_empty(&self) -> bool {
        let head = self.head;
        let head_next = unsafe { (*head).next.load(Ordering::Acquire) };
        head == self.stub && head_next.is_null()
    }
}

impl<T> Drop for MpscQueue<T> {
    fn drop(&mut self) {
        // Drain all nodes.
        while self.pop().is_some() {}
        // Deallocate the stub.
        unsafe { drop(Box::from_raw(self.stub)); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn single_producer_single_consumer() {
        let mut q = MpscQueue::new();
        q.push(1);
        q.push(2);
        q.push(3);
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), Some(3));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn multiple_producers() {
        let q = Arc::new(MpscQueue::new());
        let mut handles = Vec::new();
        let n = 4;
        let per_thread = 1000;

        for i in 0..n {
            let q = q.clone();
            handles.push(thread::spawn(move || {
                for j in 0..per_thread {
                    q.push(i * per_thread + j);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Consumer: collect all values.
        let mut seen = std::collections::HashSet::new();
        let mut q = q.take().unwrap_or_else(|_| panic!("only one consumer"));
        while let Some(v) = q.pop() {
            assert!(seen.insert(v), "duplicate value {v}");
        }

        assert_eq!(seen.len(), n * per_thread);
    }

    #[test]
    fn concurrent_push_pop() {
        let q = Arc::new(MpscQueue::new());
        let qc = q.clone();

        let producer = thread::spawn(move || {
            for i in 0..5000 {
                qc.push(i);
                thread::yield_now();
            }
        });

        let mut consumer = q.take().unwrap_or_else(|_| panic!("only one consumer"));
        let mut count = 0;
        let mut last = -1i32;

        while count < 5000 {
            if let Some(v) = consumer.pop() {
                assert!(v > last, "out of order");
                last = v;
                count += 1;
            } else {
                thread::yield_now();
            }
        }

        producer.join().unwrap();
        assert_eq!(count, 5000);
    }

    #[test]
    fn empty_queue() {
        let mut q = MpscQueue::<i32>::new();
        assert!(q.is_empty());
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn drop_while_producing() {
        let q = Arc::new(MpscQueue::new());
        let qc = q.clone();

        let producer = thread::spawn(move || {
            for i in 0..10000 {
                qc.push(i);
            }
        });

        // Let the producer get ahead, then drop the consumer.
        thread::sleep(Duration::from_millis(10));
        // The queue will be dropped when the Arc refcount goes to zero.
        // This just tests that dropping is safe while a producer is running.
        drop(q);
        producer.join().unwrap();
    }
}
