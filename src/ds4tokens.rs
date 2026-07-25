// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Token-buffer transcript for the ds4 backend (issue #58).
//!
//! This is the ds4-local *token-primary* transcript: an append-only buffer of
//! token ids that mirrors the C reference's `w->transcript`, with text always
//! *derived* from tokens rather than the reverse. It replaces the earlier
//! text-primary "splice cache" (`SampledReply`/`plan_splices`/`retain_matched`)
//! whose exactness depended on byte-for-byte text equality between a re-rendered
//! assistant section and a recorded reply — fragile under any trimming or UI
//! drift (see the "KV-cache identity is textual, not structural" entry in
//! `FINDINGS.md`).
//!
//! The transcript is a flat `Vec<i32>` of ids plus a parallel list of [`Span`]s
//! that segment it into chat messages, each carrying the *source key* (role +
//! text) that produced it and how many tokens it occupies. Reconciliation
//! against the UI's rendered transcript is a structural common-prefix match on
//! those keys: a matching prefix keeps its tokens verbatim (the KV prefix stays
//! valid), and at the first divergence everything after is dropped and
//! retokenized — robust to any text drift instead of exact-match fragile,
//! because the live KV is invalidated at that same divergence point anyway.
//!
//! This module is deliberately **FFI-free** so its logic is unit-tested in CI
//! without the native engine. The gated [`crate::ds4engine`] owns the FFI that
//! produces the token ids and derives text; it drives this type.

/// Role of a transcript span. `[tool]` sections map to [`SpanRole::User`], the
/// same collapse the ds4 chat template applies (tool output is fed back as a
/// user-role message).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanRole {
    System,
    User,
    Assistant,
}

impl SpanRole {
    /// Maps a UI section tag (`system`/`user`/`tool`/`assistant`) to a role.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "system" => Some(Self::System),
            "user" | "tool" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            _ => None,
        }
    }

    /// Serialized discriminant for persistence.
    #[must_use]
    pub fn as_u8(self) -> u8 {
        match self {
            Self::System => 0,
            Self::User => 1,
            Self::Assistant => 2,
        }
    }

    /// Inverse of [`SpanRole::as_u8`].
    #[must_use]
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::System),
            1 => Some(Self::User),
            2 => Some(Self::Assistant),
            _ => None,
        }
    }
}

/// One chat message inside the token buffer: the source key (role + text) that
/// produced its tokens, the reasoning mode (meaningful only for assistant
/// replies), and how many contiguous buffer tokens it occupies.
///
/// `text` is the section content for system/user/tool messages and the derived
/// reply text for assistant messages — i.e. the same shape the UI's rendered
/// transcript carries, so reconciliation can compare like with like.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub role: SpanRole,
    /// Reasoning-mode discriminant (mirrors `ffi::Ds4ThinkMode as u8`); 0 for
    /// non-assistant spans.
    pub think: u8,
    pub text: String,
    pub ntokens: usize,
}

/// The source key a rendered section reconciles against: role + text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionKey {
    pub role: SpanRole,
    pub text: String,
}

/// Append-only token transcript: the flat id buffer plus its span index.
///
/// Invariant: `tokens.len() == spans.iter().map(|s| s.ntokens).sum()`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenTranscript {
    tokens: Vec<i32>,
    spans: Vec<Span>,
}

impl TokenTranscript {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// All token ids in order — the live prompt prefix handed to the engine.
    #[must_use]
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// The span index (one entry per chat message).
    #[must_use]
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Appends one message's tokens and records its span. `text` is the source
    /// key text; `think` the reasoning discriminant (0 for non-assistant).
    pub fn push_span(&mut self, role: SpanRole, think: u8, text: String, tokens: &[i32]) {
        self.tokens.extend_from_slice(tokens);
        self.spans.push(Span {
            role,
            think,
            text,
            ntokens: tokens.len(),
        });
    }

    /// Length of the structural common prefix between the buffer's spans and
    /// the incoming `sections`: the count of leading spans whose (role, text)
    /// key matches the corresponding section. This is the reconciliation point —
    /// spans before it are reused verbatim, everything from it on is stale.
    #[must_use]
    pub fn common_prefix(&self, sections: &[SectionKey]) -> usize {
        self.spans
            .iter()
            .zip(sections)
            .take_while(|(span, sec)| span.role == sec.role && span.text == sec.text)
            .count()
    }

    /// Drops all spans (and their tokens) from index `keep` onward, so the
    /// buffer holds exactly the reconciled common prefix. The caller then
    /// re-appends the divergent sections via [`TokenTranscript::push_span`].
    pub fn truncate_spans(&mut self, keep: usize) {
        if keep >= self.spans.len() {
            return;
        }
        let kept_tokens: usize = self.spans[..keep].iter().map(|s| s.ntokens).sum();
        self.tokens.truncate(kept_tokens);
        self.spans.truncate(keep);
    }

    /// Rebuilds the whole buffer from a freshly retokenized set of spans (used
    /// by rewrite operations — compaction, tool-result clearing, rollback —
    /// which detokenize, edit at the text level, then retokenize back in).
    pub fn reset(&mut self, tokens: Vec<i32>, spans: Vec<Span>) {
        debug_assert_eq!(
            tokens.len(),
            spans.iter().map(|s| s.ntokens).sum::<usize>(),
            "token buffer length must equal the sum of span token counts"
        );
        self.tokens = tokens;
        self.spans = spans;
    }
}

/// Magic prefix for the token-transcript persistence payload. Sits ahead of the
/// raw engine KV snapshot bytes so a restore reconstructs the exact token buffer
/// the KV was captured with.
pub const TOKENS_MAGIC: &[u8] = b"plank-tokens-v1\n";

/// Legacy magic from the text-primary splice cache. A payload carrying it has no
/// token buffer this design can use; the engine rebuilds from text (one full
/// re-prefill), matching the C's "rebuilt from text" fallback for stripped
/// sessions. Kept for one release so old sessions still restore rather than
/// erroring.
pub const LEGACY_REPLIES_MAGIC: &[u8] = b"plank-replies-v1\n";

/// Serializes the token transcript ahead of the raw KV bytes. Layout (all
/// little-endian, length-prefixed): magic, `u32` span count, then per span
/// `role:u8 think:u8 ntokens:u32 text_len:u32 text`, then `u32` total token
/// count and that many `i32` ids.
#[must_use]
pub fn encode(transcript: &TokenTranscript) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(TOKENS_MAGIC);
    out.extend_from_slice(
        &u32::try_from(transcript.spans.len())
            .unwrap_or(0)
            .to_le_bytes(),
    );
    for s in &transcript.spans {
        out.push(s.role.as_u8());
        out.push(s.think);
        out.extend_from_slice(&u32::try_from(s.ntokens).unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&u32::try_from(s.text.len()).unwrap_or(0).to_le_bytes());
        out.extend_from_slice(s.text.as_bytes());
    }
    out.extend_from_slice(
        &u32::try_from(transcript.tokens.len())
            .unwrap_or(0)
            .to_le_bytes(),
    );
    for &t in &transcript.tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

/// Parses an [`encode`] header, returning the transcript and the byte offset
/// where the raw KV snapshot begins. Returns `None` for a legacy or malformed
/// payload (the caller then rebuilds from text). The reconstructed transcript's
/// token/span invariant is validated.
#[must_use]
pub fn decode(bytes: &[u8]) -> Option<(TokenTranscript, usize)> {
    let rest = bytes.strip_prefix(TOKENS_MAGIC)?;
    let mut pos = 0usize;
    let take = |pos: &mut usize, n: usize| -> Option<&[u8]> {
        let s = rest.get(*pos..pos.checked_add(n)?)?;
        *pos += n;
        Some(s)
    };
    let read_u32 = |pos: &mut usize| -> Option<usize> {
        let b = take(pos, 4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    let nspans = read_u32(&mut pos)?;
    let mut spans = Vec::with_capacity(nspans.min(1 << 16));
    let mut span_tokens = 0usize;
    for _ in 0..nspans {
        let role = SpanRole::from_u8(take(&mut pos, 1)?[0])?;
        let think = take(&mut pos, 1)?[0];
        let ntokens = read_u32(&mut pos)?;
        let text_len = read_u32(&mut pos)?;
        let text = String::from_utf8(take(&mut pos, text_len)?.to_vec()).ok()?;
        span_tokens = span_tokens.checked_add(ntokens)?;
        spans.push(Span {
            role,
            think,
            text,
            ntokens,
        });
    }
    let ntok = read_u32(&mut pos)?;
    if ntok != span_tokens {
        return None;
    }
    let mut tokens = Vec::with_capacity(ntok.min(1 << 20));
    for _ in 0..ntok {
        let b = take(&mut pos, 4)?;
        tokens.push(i32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    Some((TokenTranscript { tokens, spans }, TOKENS_MAGIC.len() + pos))
}

/// True when `bytes` is a legacy text-primary splice-cache payload (no usable
/// token buffer). The engine treats these as raw KV with no reconciliation
/// state and rebuilds the buffer from text on the next turn.
#[must_use]
pub fn is_legacy_payload(bytes: &[u8]) -> bool {
    bytes.starts_with(LEGACY_REPLIES_MAGIC)
}

#[cfg(test)]
mod tests {
    use super::{SectionKey, Span, SpanRole, TokenTranscript, decode, encode, is_legacy_payload};

    fn key(role: SpanRole, text: &str) -> SectionKey {
        SectionKey {
            role,
            text: text.to_string(),
        }
    }

    fn built() -> TokenTranscript {
        let mut t = TokenTranscript::new();
        t.push_span(SpanRole::System, 0, "sys".into(), &[1, 2]);
        t.push_span(SpanRole::User, 0, "q1".into(), &[3]);
        t.push_span(SpanRole::Assistant, 1, "a1".into(), &[4, 5, 6]);
        t.push_span(SpanRole::User, 0, "q2".into(), &[7]);
        t
    }

    #[test]
    fn invariant_tokens_equal_span_sum() {
        let t = built();
        assert_eq!(t.tokens().len(), 2 + 1 + 3 + 1);
        assert_eq!(t.spans().len(), 4);
    }

    #[test]
    fn common_prefix_matches_leading_sections() {
        let t = built();
        let sections = vec![
            key(SpanRole::System, "sys"),
            key(SpanRole::User, "q1"),
            key(SpanRole::Assistant, "a1"),
            key(SpanRole::User, "q2"),
        ];
        assert_eq!(t.common_prefix(&sections), 4);
    }

    #[test]
    fn common_prefix_stops_at_first_divergence() {
        let t = built();
        // q1 edited: everything from that span on is stale, even though a1/q2
        // would match by text — the KV is invalid past the edit anyway.
        let sections = vec![
            key(SpanRole::System, "sys"),
            key(SpanRole::User, "q1-edited"),
            key(SpanRole::Assistant, "a1"),
            key(SpanRole::User, "q2"),
        ];
        assert_eq!(t.common_prefix(&sections), 1);
    }

    #[test]
    fn common_prefix_stops_on_role_change() {
        let t = built();
        let sections = vec![key(SpanRole::System, "sys"), key(SpanRole::Assistant, "q1")];
        assert_eq!(t.common_prefix(&sections), 1);
    }

    #[test]
    fn common_prefix_handles_shorter_sections() {
        let t = built();
        let sections = vec![key(SpanRole::System, "sys")];
        assert_eq!(t.common_prefix(&sections), 1);
    }

    #[test]
    fn truncate_then_reappend_reconciles() {
        let mut t = built();
        // New turn: q1 unchanged, a1 unchanged, q2 replaced by q2b.
        let sections = vec![
            key(SpanRole::System, "sys"),
            key(SpanRole::User, "q1"),
            key(SpanRole::Assistant, "a1"),
            key(SpanRole::User, "q2b"),
        ];
        let keep = t.common_prefix(&sections);
        assert_eq!(keep, 3);
        t.truncate_spans(keep);
        assert_eq!(t.tokens(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(t.spans().len(), 3);
        t.push_span(SpanRole::User, 0, "q2b".into(), &[9]);
        assert_eq!(t.tokens(), &[1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn truncate_noop_when_keep_covers_all() {
        let mut t = built();
        let before = t.clone();
        t.truncate_spans(4);
        assert_eq!(t, before);
        t.truncate_spans(99);
        assert_eq!(t, before);
    }

    #[test]
    fn reset_replaces_buffer() {
        let mut t = built();
        t.reset(
            vec![10, 11],
            vec![Span {
                role: SpanRole::System,
                think: 0,
                text: "summary".into(),
                ntokens: 2,
            }],
        );
        assert_eq!(t.tokens(), &[10, 11]);
        assert_eq!(t.spans().len(), 1);
    }

    #[test]
    fn payload_round_trips_ahead_of_kv() {
        let t = {
            let mut t = TokenTranscript::new();
            t.push_span(SpanRole::System, 0, "you are helpful".into(), &[1, 2, 3]);
            t.push_span(
                SpanRole::Assistant,
                1,
                "first ünïcode".into(),
                &[-1, 1 << 20],
            );
            t
        };
        let kv = b"raw-engine-kv-bytes";
        let mut wrapped = encode(&t);
        wrapped.extend_from_slice(kv);
        let (decoded, off) = decode(&wrapped).expect("wrap decodes");
        assert_eq!(decoded, t);
        assert_eq!(&wrapped[off..], kv);
    }

    #[test]
    fn empty_transcript_round_trips() {
        let t = TokenTranscript::new();
        let wrapped = encode(&t);
        let (decoded, off) = decode(&wrapped).expect("empty decodes");
        assert_eq!(decoded, t);
        assert_eq!(off, wrapped.len());
    }

    #[test]
    fn legacy_and_malformed_payloads_rejected() {
        assert!(decode(b"raw-kv-no-magic").is_none());
        assert!(decode(super::LEGACY_REPLIES_MAGIC).is_none());
        assert!(is_legacy_payload(super::LEGACY_REPLIES_MAGIC));
        assert!(!is_legacy_payload(super::TOKENS_MAGIC));
        // Truncated header (magic only) is rejected, not misparsed.
        assert!(decode(super::TOKENS_MAGIC).is_none());
        // Token count disagreeing with span sum is rejected.
        let mut t = TokenTranscript::new();
        t.push_span(SpanRole::User, 0, "x".into(), &[1, 2]);
        let mut bad = encode(&t);
        // The trailing layout is [count:u32][id:i32][id:i32]; corrupt the count.
        let count_off = bad.len() - 8 - 4;
        bad[count_off] = 9;
        assert!(decode(&bad).is_none());
    }

    #[test]
    fn role_tag_mapping() {
        assert_eq!(SpanRole::from_tag("tool"), Some(SpanRole::User));
        assert_eq!(SpanRole::from_tag("user"), Some(SpanRole::User));
        assert_eq!(SpanRole::from_tag("system"), Some(SpanRole::System));
        assert_eq!(SpanRole::from_tag("assistant"), Some(SpanRole::Assistant));
        assert_eq!(SpanRole::from_tag("nope"), None);
    }
}
