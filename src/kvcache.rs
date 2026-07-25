// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The single KV-cache value type and on-disk format.
//!
//! Every persisted KV in plank — the system-prompt checkpoint, per-project
//! tier checkpoints, and per-session payloads — is a [`KVCache`] written by
//! [`KVCache::persist`] and read by [`KVCache::from_file`]. There is exactly
//! one writer and one reader, which is the point: before this type there were
//! three near-identical `<fingerprint>\n<bytes>` implementations and two
//! different payload shapes.
//!
//! A read is fallible *by value*: a missing file, a signature mismatch, a
//! truncated body, and an unrecognized version are all `None`, and `None`
//! always means "prefill instead". No other code makes a trust decision about
//! cached bytes.

use std::io;
use std::path::Path;

use crate::ds4tokens::{self, TokenTranscript};

/// On-disk format version. Bumping it invalidates every cached file, which is
/// safe: all of them are rebuildable caches, so the cost is one re-prefill.
pub const FORMAT_VERSION: u8 = 2;

/// A captured engine KV state plus the token transcript it was captured with.
///
/// The transcript may be empty (tier checkpoints have no conversation yet).
/// Carrying it in the same type is what lets session payloads and tier
/// checkpoints share one format: without it, a resumed session rebuilds its
/// token buffer from text and pays a full re-prefill from the first reply.
#[derive(Debug, Clone, Default)]
pub struct KVCache {
    kv: Vec<u8>,
    transcript: TokenTranscript,
}

impl KVCache {
    /// Wraps captured engine bytes and the transcript they belong to.
    #[must_use]
    pub fn new(kv: Vec<u8>, transcript: TokenTranscript) -> Self {
        Self { kv, transcript }
    }

    /// The opaque engine snapshot bytes.
    #[must_use]
    pub fn kv(&self) -> &[u8] {
        &self.kv
    }

    /// The token transcript captured with this KV (possibly empty).
    #[must_use]
    pub fn transcript(&self) -> &TokenTranscript {
        &self.transcript
    }

    /// Serializes to the on-disk layout:
    /// `<signature>\n<version:u8><encoded transcript><raw kv bytes>`.
    #[must_use]
    pub fn encode(&self, signature: &str) -> Vec<u8> {
        let transcript = ds4tokens::encode(&self.transcript);
        let mut out = Vec::with_capacity(signature.len() + 2 + transcript.len() + self.kv.len());
        out.extend_from_slice(signature.as_bytes());
        out.push(b'\n');
        out.push(FORMAT_VERSION);
        out.extend_from_slice(&transcript);
        out.extend_from_slice(&self.kv);
        out
    }

    /// Parses bytes written by [`encode`](Self::encode), requiring the stored
    /// signature to equal `expect` exactly.
    ///
    /// Returns `None` for anything that is not a well-formed current-version
    /// file with a matching signature. Callers treat `None` as a cache miss.
    #[must_use]
    pub fn decode(bytes: &[u8], expect: &str) -> Option<Self> {
        let nl = bytes.iter().position(|&b| b == b'\n')?;
        if bytes.get(..nl)? != expect.as_bytes() {
            return None;
        }
        let body = bytes.get(nl + 1..)?;
        if *body.first()? != FORMAT_VERSION {
            return None;
        }
        let (transcript, off) = ds4tokens::decode(&body[1..])?;
        let kv = body.get(1 + off..)?.to_vec();
        Some(Self { kv, transcript })
    }

    /// Reads a cache file, requiring the stored signature to equal `expect`.
    /// Any IO error is a miss.
    #[must_use]
    pub fn from_file(path: &Path, expect: &str) -> Option<Self> {
        Self::decode(&std::fs::read(path).ok()?, expect)
    }

    /// Writes this cache to `path` under `signature`, creating parent
    /// directories as needed.
    ///
    /// Writes a sibling temp file and renames it, so a crash mid-write can
    /// never leave a half-written file that still decodes.
    ///
    /// # Errors
    /// Returns the underlying [`io::Error`] when the write or rename fails.
    pub fn persist(&self, path: &Path, signature: &str) -> io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, self.encode(signature))?;
        std::fs::rename(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use crate::ds4tokens::TokenTranscript;

    use super::*;

    #[test]
    fn roundtrips_through_encode_decode() {
        let c = KVCache::new(vec![1, 2, 3, 4], TokenTranscript::new());
        let bytes = c.encode("sig-a");
        let back = KVCache::decode(&bytes, "sig-a").expect("decodes with the right signature");
        assert_eq!(back.kv(), &[1, 2, 3, 4]);
    }

    #[test]
    fn signature_mismatch_is_a_miss() {
        let c = KVCache::new(vec![9], TokenTranscript::new());
        let bytes = c.encode("sig-a");
        assert!(KVCache::decode(&bytes, "sig-b").is_none());
    }

    #[test]
    fn truncated_file_is_a_miss() {
        let c = KVCache::new(vec![1, 2, 3], TokenTranscript::new());
        let bytes = c.encode("sig-a");
        // Drop the trailing half: header survives, body does not.
        assert!(KVCache::decode(&bytes[..bytes.len() / 2], "sig-a").is_none());
        // No newline at all is also a miss, not a panic.
        assert!(KVCache::decode(b"no-newline-here", "sig-a").is_none());
        assert!(KVCache::decode(b"", "sig-a").is_none());
    }

    #[test]
    fn old_version_byte_is_a_miss() {
        let c = KVCache::new(vec![1, 2, 3], TokenTranscript::new());
        let mut bytes = c.encode("sig-a");
        let nl = bytes.iter().position(|&b| b == b'\n').unwrap();
        bytes[nl + 1] = 1; // pretend it is the v1 format
        assert!(KVCache::decode(&bytes, "sig-a").is_none());
    }

    #[test]
    fn persist_then_from_file_roundtrips() {
        let dir = std::env::temp_dir().join(format!("plank-kvcache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.kv");
        let c = KVCache::new(vec![7, 7, 7], TokenTranscript::new());
        c.persist(&path, "sig-a").unwrap();
        let back = KVCache::from_file(&path, "sig-a").expect("reads back");
        assert_eq!(back.kv(), &[7, 7, 7]);
        // Wrong signature reads as a miss, and a missing file too.
        assert!(KVCache::from_file(&path, "sig-b").is_none());
        assert!(KVCache::from_file(&dir.join("absent.kv"), "sig-a").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn persist_leaves_no_partial_file_behind() {
        // The temp file must not survive a successful persist: a stray
        // `t.kv.tmp` would eventually be read as a checkpoint by a future
        // globbing GC pass.
        let dir = std::env::temp_dir().join(format!("plank-kvcache-tmp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.kv");
        KVCache::new(vec![1], TokenTranscript::new())
            .persist(&path, "sig")
            .unwrap();
        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["t.kv".to_owned()]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
