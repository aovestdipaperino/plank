// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Minisign signature verification for plugin artifacts
//! (`docs/WASM-PLUGINS.md`, "Trust and signing").
//!
//! Signing is about **provenance, not containment** — the sandbox is what stops
//! a plugin doing damage. What a signature buys is that an update from a
//! publisher you already trusted installs without re-prompting, so the trust
//! dialog stays meaningful instead of becoming a thing users click through.
//!
//! Two rules from the design shape everything here:
//!
//! - **A missing signature is not an error; a bad one is.** An unsigned plugin
//!   is the normal case and falls back to first-use hash approval. A plugin
//!   that presents a signature which does not verify is refused outright: it is
//!   the one case where something is actively wrong rather than merely unknown.
//! - **A valid signature never widens capabilities.** Provenance says who
//!   built it, not what it may do. That check lives in [`crate::wasmreg`]; this
//!   module only answers "who signed these bytes".
//!
//! ## The format
//!
//! Verified against minisign 0.12, not just read off the spec. A public key is
//! base64 over `alg(2) || key_id(8) || public_key(32)`; a `.minisig` carries
//! two base64 lines, `alg(2) || key_id(8) || signature(64)` and a global
//! signature over `signature || trusted_comment`.
//!
//! Two algorithms exist and both appear in the wild:
//!
//! - `ED` — Ed25519 over `Blake2b-512(file)`. **minisign's default since
//!   0.10**, which is why [`blake2`] is a dependency: rejecting the default
//!   would mean rejecting essentially every signature the reference tool makes.
//! - `Ed` — Ed25519 over the file itself. What `minisign -l` produces.
//!
//! The global signature is verified too, not skipped as decoration. It covers
//! the trusted comment, which is the only part of a `.minisig` a reader is
//! invited to believe; leaving it unchecked would let anyone edit the filename
//! and timestamp a user is shown while the artifact signature still passed.

use blake2::Digest as _;

/// Why a signature did not verify.
///
/// Distinguished rather than collapsed into a bool because the actions differ:
/// a malformed file is a packaging mistake, an unknown key is a trust question,
/// and a failed verification is the only one that means someone changed the
/// bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SigError {
    /// The `.minisig` or public key was not the shape minisign writes.
    Malformed(&'static str),
    /// An algorithm this build does not implement.
    UnknownAlgorithm([u8; 2]),
    /// The signature is for a different key than the one offered.
    KeyMismatch,
    /// Ed25519 rejected it: these bytes are not what was signed.
    BadSignature,
    /// The artifact signature verified but the trusted comment did not, so the
    /// human-readable part of the file has been altered.
    BadGlobalSignature,
}

impl std::fmt::Display for SigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(what) => write!(f, "malformed signature: {what}"),
            Self::UnknownAlgorithm(alg) => {
                write!(
                    f,
                    "unsupported signature algorithm {:?}",
                    String::from_utf8_lossy(alg)
                )
            }
            Self::KeyMismatch => write!(f, "signature was made by a different key"),
            Self::BadSignature => write!(f, "signature does not match the artifact"),
            Self::BadGlobalSignature => write!(f, "the trusted comment has been altered"),
        }
    }
}

/// A minisign public key: the 32 raw bytes plus the key id it advertises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicKey {
    /// Key id, as minisign prints it. Used to tell "not signed by this
    /// publisher" from "signed by this publisher, badly".
    pub key_id: [u8; 8],
    key: [u8; 32],
}

impl PublicKey {
    /// Parses a minisign public key: either the two-line file or the bare
    /// base64 line that `minisign -G` prints for pasting into a config.
    ///
    /// # Errors
    /// [`SigError::Malformed`] when it is not 42 bytes of base64, and
    /// [`SigError::UnknownAlgorithm`] for a key that is not `Ed`.
    pub fn parse(text: &str) -> Result<Self, SigError> {
        // Last non-empty, non-comment line: a key file leads with an untrusted
        // comment, and a pasted key has no comment at all.
        let line = text
            .lines()
            .map(str::trim)
            .rfind(|l| !l.is_empty() && !l.starts_with("untrusted comment:"))
            .ok_or(SigError::Malformed("no key line"))?;
        let raw = base64_decode(line).ok_or(SigError::Malformed("key is not base64"))?;
        if raw.len() != 42 {
            return Err(SigError::Malformed("key is not 42 bytes"));
        }
        // A public key is always `Ed`; the prehash choice belongs to the
        // signature, not the key.
        if &raw[..2] != b"Ed" {
            return Err(SigError::UnknownAlgorithm([raw[0], raw[1]]));
        }
        let mut key_id = [0u8; 8];
        key_id.copy_from_slice(&raw[2..10]);
        let mut key = [0u8; 32];
        key.copy_from_slice(&raw[10..42]);
        Ok(Self { key_id, key })
    }

    /// The key id **exactly as minisign prints it**: byte-reversed and
    /// upper-case hex.
    ///
    /// Not the raw byte order. minisign stores the id little-endian and
    /// displays it big-endian, so raw bytes `ae03…e1` are shown as
    /// `E171…AE`. A user comparing plank's output against `minisign -G` would
    /// read the raw spelling as a different key entirely, which in a feature
    /// whose whole job is a human matching two identifiers is not a cosmetic
    /// difference.
    #[must_use]
    pub fn key_id_hex(&self) -> String {
        key_id_display(self.key_id)
    }
}

/// Spells a raw key id the way minisign does: byte-reversed, upper-case.
///
/// Free-standing because a *signature* carries a key id for which no public key
/// may be held — that is exactly the "signed by someone unknown" case, and it
/// still has to be named in a message.
#[must_use]
pub fn key_id_display(key_id: [u8; 8]) -> String {
    use std::fmt::Write as _;
    key_id.iter().rev().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02X}");
        out
    })
}

/// A parsed `.minisig`.
#[derive(Debug, Clone)]
pub struct Signature {
    prehashed: bool,
    key_id: [u8; 8],
    sig: [u8; 64],
    global_sig: [u8; 64],
    trusted_comment: String,
}

impl Signature {
    /// Parses a `.minisig` file.
    ///
    /// # Errors
    /// [`SigError::Malformed`] for anything that is not the four-line shape
    /// minisign writes, [`SigError::UnknownAlgorithm`] for an algorithm other
    /// than `ED`/`Ed`.
    pub fn parse(text: &str) -> Result<Self, SigError> {
        let mut sig_b64 = None;
        let mut trusted_comment = None;
        let mut global_b64 = None;
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if let Some(rest) = line.strip_prefix("trusted comment:") {
                // Stored without the prefix and with the single leading space
                // removed, because that is exactly the byte range the global
                // signature covers.
                trusted_comment = Some(rest.strip_prefix(' ').unwrap_or(rest).to_string());
            } else if line.starts_with("untrusted comment:") {
            } else if sig_b64.is_none() {
                sig_b64 = Some(line);
            } else if global_b64.is_none() {
                global_b64 = Some(line);
            }
        }
        let raw = sig_b64
            .and_then(base64_decode)
            .ok_or(SigError::Malformed("no signature line"))?;
        if raw.len() != 74 {
            return Err(SigError::Malformed("signature is not 74 bytes"));
        }
        let prehashed = match &raw[..2] {
            b"ED" => true,
            b"Ed" => false,
            other => return Err(SigError::UnknownAlgorithm([other[0], other[1]])),
        };
        let global = global_b64
            .and_then(base64_decode)
            .ok_or(SigError::Malformed("no global signature line"))?;
        if global.len() != 64 {
            return Err(SigError::Malformed("global signature is not 64 bytes"));
        }
        let mut key_id = [0u8; 8];
        key_id.copy_from_slice(&raw[2..10]);
        let mut sig = [0u8; 64];
        sig.copy_from_slice(&raw[10..74]);
        let mut global_sig = [0u8; 64];
        global_sig.copy_from_slice(&global);
        Ok(Self {
            prehashed,
            key_id,
            sig,
            global_sig,
            trusted_comment: trusted_comment.unwrap_or_default(),
        })
    }

    /// The trusted comment, which is only worth reading *after* [`verify`]
    /// succeeds — that is what makes it trusted.
    #[must_use]
    pub fn trusted_comment(&self) -> &str {
        &self.trusted_comment
    }

    /// The key id this signature claims to come from.
    #[must_use]
    pub fn key_id(&self) -> [u8; 8] {
        self.key_id
    }
}

/// Verifies `signature` over `artifact` under `key`.
///
/// # Errors
/// [`SigError::KeyMismatch`] when the signature names a different key,
/// [`SigError::BadSignature`] when the artifact does not match, and
/// [`SigError::BadGlobalSignature`] when the artifact verified but the trusted
/// comment did not.
pub fn verify(artifact: &[u8], signature: &Signature, key: &PublicKey) -> Result<(), SigError> {
    // Checked first so "you have the wrong publisher's key" cannot present as
    // "these bytes were tampered with", which would send a user hunting a
    // compromise that did not happen.
    if signature.key_id != key.key_id {
        return Err(SigError::KeyMismatch);
    }
    let verifier =
        ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, key.key.as_slice());

    let signed: Vec<u8> = if signature.prehashed {
        let mut hasher = blake2::Blake2b512::new();
        hasher.update(artifact);
        hasher.finalize().to_vec()
    } else {
        artifact.to_vec()
    };
    verifier
        .verify(&signed, &signature.sig)
        .map_err(|_| SigError::BadSignature)?;

    // The global signature covers the artifact signature followed by the
    // trusted comment, with nothing between them.
    let mut global_input = Vec::with_capacity(64 + signature.trusted_comment.len());
    global_input.extend_from_slice(&signature.sig);
    global_input.extend_from_slice(signature.trusted_comment.as_bytes());
    verifier
        .verify(&global_input, &signature.global_sig)
        .map_err(|_| SigError::BadGlobalSignature)
}

/// Counts `=` padding bytes in a base64 quantum. A four-byte loop, spelled as
/// its own function only because the obvious `filter().count()` trips a lint
/// aimed at much larger inputs.
fn bytecount_eq(chunk: &[u8]) -> usize {
    let mut n = 0;
    for &b in chunk {
        if b == b'=' {
            n += 1;
        }
    }
    n
}

/// Decodes standard base64 with padding.
///
/// Hand-rolled rather than pulling a crate for it: this is the only base64 in
/// plank's own code, the alphabet is fixed, and a decoder that rejects
/// everything unexpected is a smaller thing to be sure of than a dependency.
fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let value = |b: u8| -> Option<u32> {
        Some(match b {
            b'A'..=b'Z' => u32::from(b - b'A'),
            b'a'..=b'z' => u32::from(b - b'a') + 26,
            b'0'..=b'9' => u32::from(b - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    };
    let bytes: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        // Padding only ever ends the input, and only one or two bytes of it.
        let pad = bytecount_eq(chunk);
        if pad > 2 {
            return None;
        }
        let mut acc = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            let v = if b == b'=' {
                if i < 2 {
                    return None;
                }
                0
            } else {
                value(b)?
            };
            acc = (acc << 6) | v;
        }
        let full = acc.to_be_bytes();
        out.push(full[1]);
        if pad < 2 {
            out.push(full[2]);
        }
        if pad < 1 {
            out.push(full[3]);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixtures produced by minisign 0.12 itself, not by this code: a verifier
    /// tested only against its own signer can share a misreading of the format
    /// with it and pass. `tests/fixtures/minisign/` records how they were made.
    const PUB: &str = include_str!("../tests/fixtures/minisign/pub.key");
    const ARTIFACT: &[u8] = include_bytes!("../tests/fixtures/minisign/artifact.wasm");
    const SIG_PREHASHED: &str = include_str!("../tests/fixtures/minisign/artifact.wasm.minisig");
    const SIG_LEGACY: &str = include_str!("../tests/fixtures/minisign/artifact.legacy.minisig");
    const OTHER_PUB: &str = include_str!("../tests/fixtures/minisign/other.pub");
    const SIG_OTHER: &str = include_str!("../tests/fixtures/minisign/artifact.other.minisig");

    #[test]
    fn a_real_prehashed_signature_verifies() {
        let key = PublicKey::parse(PUB).expect("key parses");
        let sig = Signature::parse(SIG_PREHASHED).expect("signature parses");
        assert!(sig.prehashed, "minisign 0.12 prehashes by default");
        assert_eq!(verify(ARTIFACT, &sig, &key), Ok(()));
        // The trusted comment is only meaningful once verification passed.
        assert!(
            sig.trusted_comment().contains("file:artifact.wasm"),
            "{:?}",
            sig.trusted_comment()
        );
    }

    /// `minisign -l`. Still in the wild, and the only variant `ring` could
    /// verify without a Blake2b dependency — worth keeping tested so nobody
    /// removes the legacy branch believing it is dead.
    #[test]
    fn a_real_legacy_signature_verifies() {
        let key = PublicKey::parse(PUB).expect("key parses");
        let sig = Signature::parse(SIG_LEGACY).expect("signature parses");
        assert!(!sig.prehashed);
        assert_eq!(verify(ARTIFACT, &sig, &key), Ok(()));
    }

    /// One flipped byte is refused. The whole feature is this assertion.
    #[test]
    fn a_tampered_artifact_is_refused() {
        let key = PublicKey::parse(PUB).expect("key parses");
        let sig = Signature::parse(SIG_PREHASHED).expect("signature parses");
        let mut tampered = ARTIFACT.to_vec();
        tampered.push(b'!');
        assert_eq!(
            verify(&tampered, &sig, &key),
            Err(SigError::BadSignature),
            "an appended byte must not verify"
        );

        let mut flipped = ARTIFACT.to_vec();
        flipped[0] ^= 0x01;
        assert_eq!(verify(&flipped, &sig, &key), Err(SigError::BadSignature));
    }

    /// A signature from a key the user does not trust reads as the wrong key,
    /// not as tampering: the distinction is the difference between "install
    /// this publisher's key" and "these bytes were changed in transit".
    #[test]
    fn another_publishers_signature_is_a_key_mismatch() {
        let key = PublicKey::parse(PUB).expect("key parses");
        let sig = Signature::parse(SIG_OTHER).expect("signature parses");
        assert_eq!(verify(ARTIFACT, &sig, &key), Err(SigError::KeyMismatch));

        // ...and it verifies cleanly under its own key, which proves the
        // fixture is a good signature rather than a broken one.
        let other = PublicKey::parse(OTHER_PUB).expect("other key parses");
        assert_eq!(verify(ARTIFACT, &sig, &other), Ok(()));
    }

    /// Editing the trusted comment is caught by the global signature. Without
    /// that second check a `.minisig` could claim any filename and timestamp
    /// while still passing.
    #[test]
    fn an_edited_trusted_comment_is_caught() {
        let key = PublicKey::parse(PUB).expect("key parses");
        let edited = SIG_PREHASHED.replace("file:artifact.wasm", "file:something-else");
        assert_ne!(edited, SIG_PREHASHED, "the fixture must contain the text");
        let sig = Signature::parse(&edited).expect("still parses");
        assert_eq!(
            verify(ARTIFACT, &sig, &key),
            Err(SigError::BadGlobalSignature),
            "the artifact still matches, so this must fail on the comment"
        );
    }

    #[test]
    fn key_ids_round_trip_and_match_the_signature() {
        let key = PublicKey::parse(PUB).expect("key parses");
        let sig = Signature::parse(SIG_PREHASHED).expect("signature parses");
        assert_eq!(sig.key_id(), key.key_id);
        // The spelling must be the one minisign shows, because a user checks a
        // publisher by eye against `minisign -G` output. The fixture's own
        // untrusted comment is the reference.
        assert_eq!(key.key_id_hex(), "E171F5A2933C03AE");
        assert!(
            PUB.contains(&key.key_id_hex()),
            "plank's spelling must match the comment minisign wrote: {} not in {PUB:?}",
            key.key_id_hex()
        );
    }

    #[test]
    fn malformed_inputs_are_named_rather_than_guessed_at() {
        assert!(matches!(
            PublicKey::parse("untrusted comment: only a comment"),
            Err(SigError::Malformed(_))
        ));
        assert!(matches!(
            PublicKey::parse("not base64 at all!!"),
            Err(SigError::Malformed(_))
        ));
        // Right length, wrong algorithm: 42 bytes beginning "Xx".
        let mut raw = vec![b'X', b'x'];
        raw.extend(std::iter::repeat_n(0u8, 40));
        let bogus = base64_encode(&raw);
        assert!(matches!(
            PublicKey::parse(&bogus),
            Err(SigError::UnknownAlgorithm(_))
        ));
        assert!(matches!(
            Signature::parse("untrusted comment: x\nnot base64\n"),
            Err(SigError::Malformed(_))
        ));
    }

    /// The decoder is plank's own, so it gets its own round trip against a
    /// known-answer vector rather than only being exercised through fixtures.
    #[test]
    fn base64_decodes_padding_and_rejects_junk() {
        assert_eq!(base64_decode("aGVsbG8="), Some(b"hello".to_vec()));
        assert_eq!(base64_decode("aGVsbG8h"), Some(b"hello!".to_vec()));
        assert_eq!(base64_decode("aGk="), Some(b"hi".to_vec()));
        // Whitespace between lines is skipped, as in a key file.
        assert_eq!(base64_decode("aGVs\nbG8="), Some(b"hello".to_vec()));
        assert_eq!(base64_decode(""), None, "empty is not zero bytes");
        assert_eq!(base64_decode("aGVsbG8"), None, "unpadded length is refused");
        assert_eq!(base64_decode("aGVs*G8="), None, "junk byte");
        assert_eq!(base64_decode("=aGVsbG8"), None, "leading padding");
    }

    /// Test-only encoder, for building the malformed inputs above.
    fn base64_encode(bytes: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b = [
                chunk[0],
                chunk.get(1).copied().unwrap_or(0),
                chunk.get(2).copied().unwrap_or(0),
            ];
            let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            for i in 0..4 {
                #[allow(clippy::cast_possible_truncation)]
                if i <= chunk.len() {
                    out.push(ALPHABET[((n >> (18 - 6 * i)) & 0x3f) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }
}
