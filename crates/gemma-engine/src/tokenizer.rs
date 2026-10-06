//! Gemma 4's GGUF tokenizer, with plain text kept inert.
//!
//! The HF tokenizer built here knows no added or special tokens at all, so
//! BPE over any string can only yield ordinary pieces: a file that contains
//! `<turn|>` tokenizes as the characters `<`, `turn`, `|`, `>`, never as the id
//! that closes a turn. Control ids come only from [`GemmaTokenizer::encode_trusted`],
//! which the chat template feeds its own control text.

use std::collections::HashMap;

use candle_core::quantized::gguf_file::{Content, Value};
use tokenizers::Tokenizer;
use tokenizers::decoders::{byte_fallback::ByteFallback, sequence::Sequence as DecoderSequence};
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::tokenizer::SplitDelimiterBehavior;

use crate::template::{self, CONTROL_SPELLINGS, Piece};
use crate::{Error, Result};

/// The control spellings a Gemma 4 vocab must carry for plank's chat format.
const REQUIRED: [&str; 6] = [
    template::TURN_OPEN,
    template::TURN_CLOSE,
    template::CALL_OPEN,
    template::CALL_CLOSE,
    template::RESP_OPEN,
    template::STR,
];

/// GGUF token types that mark a token as control text.
const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;

/// The metaspace marker Gemma's pieces use for a space.
const SPACE: char = '\u{2581}';

/// Gemma 4's tokenizer, read from a GGUF's `tokenizer.ggml.*` metadata.
#[derive(Debug)]
pub struct GemmaTokenizer {
    tokenizer: Tokenizer,
    pieces: Vec<String>,
    control: Vec<bool>,
    control_ids: HashMap<String, u32>,
    /// Control spellings by first byte, longest first.
    control_by_first: HashMap<u8, Vec<(String, u32)>>,
    byte_ids: Vec<u32>,
    bos: Option<u32>,
    eos: u32,
}

fn meta<'a>(ct: &'a Content, key: &str) -> Result<&'a Value> {
    ct.metadata
        .get(key)
        .ok_or_else(|| Error(format!("missing GGUF metadata key `{key}`")))
}

fn as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::U8(n) => Some(i64::from(*n)),
        Value::I8(n) => Some(i64::from(*n)),
        Value::U16(n) => Some(i64::from(*n)),
        Value::I16(n) => Some(i64::from(*n)),
        Value::U32(n) => Some(i64::from(*n)),
        Value::I32(n) => Some(i64::from(*n)),
        Value::U64(n) => i64::try_from(*n).ok(),
        Value::I64(n) => Some(*n),
        _ => None,
    }
}

fn token_id(ct: &Content, key: &str, vocab: usize) -> Result<u32> {
    as_i64(meta(ct, key)?)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| (n as usize) < vocab)
        .ok_or_else(|| {
            Error(format!(
                "`{key}` is not a token id in a {vocab}-token vocab"
            ))
        })
}

fn array<'a>(ct: &'a Content, key: &str) -> Result<&'a [Value]> {
    match meta(ct, key)? {
        Value::Array(v) => Ok(v),
        v => Err(Error(format!(
            "`{key}` is not an array: {:?}",
            v.value_type()
        ))),
    }
}

fn strings(ct: &Content, key: &str) -> Result<Vec<String>> {
    array(ct, key)?
        .iter()
        .map(|v| match v {
            Value::String(s) => Ok(s.clone()),
            v => Err(Error(format!(
                "`{key}` holds a non-string: {:?}",
                v.value_type()
            ))),
        })
        .collect()
}

fn id_of(i: usize) -> Result<u32> {
    u32::try_from(i).map_err(|_| Error(format!("token index {i} overflows u32")))
}

/// `<0xHH>` as its byte.
fn byte_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn build_hf(pieces: &[String], merges: Vec<(String, String)>) -> Result<Tokenizer> {
    let hf = |e: tokenizers::Error| Error(format!("tokenizer: {e}"));
    let mut vocab = Vocab::default();
    for (i, p) in pieces.iter().enumerate() {
        // A duplicated spelling keeps its first id, as llama.cpp does.
        vocab.entry(p.clone()).or_insert(id_of(i)?);
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .build()
        .map_err(hf)?;
    let mut tokenizer = Tokenizer::new(bpe);
    let split = Split::new(
        SplitPattern::Regex("[\n]+".to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(hf)?;
    let metaspace = Metaspace::new(SPACE, PrependScheme::Never, false);
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        split.into(),
        metaspace.clone().into(),
    ])));
    tokenizer.with_decoder(Some(DecoderSequence::new(vec![
        ByteFallback::new().into(),
        metaspace.into(),
    ])));
    Ok(tokenizer)
}

impl GemmaTokenizer {
    /// Reads the tokenizer from a GGUF's metadata.
    ///
    /// # Errors
    /// When a `tokenizer.ggml.*` key is missing or malformed, a byte token is
    /// absent, or the vocab lacks a control token plank's chat format needs.
    pub fn from_gguf(ct: &Content) -> Result<Self> {
        let pieces = strings(ct, "tokenizer.ggml.tokens")?;
        let merges = strings(ct, "tokenizer.ggml.merges")?
            .into_iter()
            .map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .ok_or_else(|| Error(format!("invalid merge entry `{m}`")))
            })
            .collect::<Result<Vec<_>>>()?;
        let types = array(ct, "tokenizer.ggml.token_type")?;
        if types.len() != pieces.len() {
            return Err(Error(format!(
                "{} token types for {} tokens",
                types.len(),
                pieces.len()
            )));
        }
        let mut control = types
            .iter()
            .map(|t| matches!(as_i64(t), Some(TYPE_CONTROL | TYPE_USER_DEFINED)))
            .collect::<Vec<_>>();
        for s in CONTROL_SPELLINGS {
            if let Some(i) = pieces.iter().position(|p| p == s) {
                control[i] = true;
            }
        }
        for s in REQUIRED {
            if !pieces.iter().enumerate().any(|(i, p)| p == s && control[i]) {
                return Err(Error(format!(
                    "tokenizer lacks the Gemma 4 control token {s}"
                )));
            }
        }

        let mut control_ids = HashMap::new();
        for (i, p) in pieces.iter().enumerate() {
            if control[i] && !p.is_empty() {
                control_ids.entry(p.clone()).or_insert(id_of(i)?);
            }
        }
        let mut control_by_first: HashMap<u8, Vec<(String, u32)>> = HashMap::new();
        for (s, &id) in &control_ids {
            control_by_first
                .entry(s.as_bytes()[0])
                .or_default()
                .push((s.clone(), id));
        }
        for v in control_by_first.values_mut() {
            v.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        }

        let mut byte_ids = vec![None; 256];
        for (i, p) in pieces.iter().enumerate() {
            if let Some(b) = byte_piece(p) {
                byte_ids[usize::from(b)].get_or_insert(id_of(i)?);
            }
        }
        let byte_ids = byte_ids
            .into_iter()
            .enumerate()
            .map(|(b, id)| {
                id.ok_or_else(|| Error(format!("tokenizer lacks byte token <0x{b:02X}>")))
            })
            .collect::<Result<Vec<_>>>()?;

        let add_bos = matches!(
            ct.metadata.get("tokenizer.ggml.add_bos_token"),
            Some(Value::Bool(true))
        );
        let bos = if add_bos {
            Some(token_id(ct, "tokenizer.ggml.bos_token_id", pieces.len())?)
        } else {
            None
        };
        let eos = token_id(ct, "tokenizer.ggml.eos_token_id", pieces.len())?;
        let tokenizer = build_hf(&pieces, merges)?;
        Ok(Self {
            tokenizer,
            pieces,
            control,
            control_ids,
            control_by_first,
            byte_ids,
            bos,
            eos,
        })
    }

    /// Tokenizes untrusted text. Never yields a control id, whatever the text
    /// spells: the HF tokenizer has no special tokens, and should a merge ever
    /// reach a control piece, it is spelled out in byte tokens instead.
    #[must_use]
    pub fn encode_plain(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let ids = match self.tokenizer.encode(text, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => return self.bytes_of(text.as_bytes()),
        };
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if self.is_control(id) {
                out.extend(self.bytes_of(self.pieces[id as usize].as_bytes()));
            } else {
                out.push(id);
            }
        }
        out
    }

    fn bytes_of(&self, bytes: &[u8]) -> Vec<u32> {
        bytes
            .iter()
            .map(|&b| self.byte_ids[usize::from(b)])
            .collect()
    }

    /// The longest control spelling `text` starts with.
    fn control_at(&self, text: &str) -> Option<(usize, u32)> {
        let first = *text.as_bytes().first()?;
        self.control_by_first
            .get(&first)?
            .iter()
            .find(|(s, _)| text.starts_with(s.as_str()))
            .map(|(s, id)| (s.len(), *id))
    }

    /// Tokenizes trusted text: each control spelling, longest first at each
    /// position, becomes its atomic id; the gaps are encoded as plain text.
    #[must_use]
    pub fn encode_trusted(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let (mut gap, mut i) = (0, 0);
        while i < text.len() {
            if let Some((len, id)) = self.control_at(&text[i..]) {
                out.extend(self.encode_plain(&text[gap..i]));
                out.push(id);
                i += len;
                gap = i;
            } else {
                i += text[i..].chars().next().map_or(1, char::len_utf8);
            }
        }
        out.extend(self.encode_plain(&text[gap..]));
        out
    }

    /// Tokenizes rendered template pieces, each by its trust.
    #[must_use]
    pub fn encode_pieces(&self, pieces: &[Piece]) -> Vec<u32> {
        let mut out = Vec::new();
        for p in pieces {
            match p {
                Piece::Trusted(s) => out.extend(self.encode_trusted(s)),
                Piece::Plain(s) => out.extend(self.encode_plain(s)),
            }
        }
        out
    }

    /// The bytes token `id` stands for: a byte token its byte, a control
    /// token its spelling, any other piece with `▁` read as a space. Empty for
    /// an id outside the vocab.
    #[must_use]
    pub fn piece_bytes(&self, id: u32) -> Vec<u8> {
        let Some(piece) = self.pieces.get(id as usize) else {
            return Vec::new();
        };
        if self.is_control(id) {
            return piece.as_bytes().to_vec();
        }
        if let Some(b) = byte_piece(piece) {
            return vec![b];
        }
        piece.replace(SPACE, " ").into_bytes()
    }

    /// The id of a control token by its spelling.
    #[must_use]
    pub fn control_id(&self, spelling: &str) -> Option<u32> {
        self.control_ids.get(spelling).copied()
    }

    /// Whether `id` is control text (GGUF token type control or user-defined,
    /// or one of the chat format's spellings).
    #[must_use]
    pub fn is_control(&self, id: u32) -> bool {
        self.control.get(id as usize).copied().unwrap_or(false)
    }

    /// The id to open a sequence with, when the GGUF asks for one.
    #[must_use]
    pub fn bos(&self) -> Option<u32> {
        self.bos
    }

    #[must_use]
    pub fn eos(&self) -> u32 {
        self.eos
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgguf::{TinyConfig, write_tiny};

    fn tok() -> GemmaTokenizer {
        // One file per call: tests run in parallel, and a shared path lets
        // one test read another's half-written file.
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!("gemma-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dir.join(format!("tiny-{n}.gguf"));
        write_tiny(&path, &TinyConfig::default()).unwrap();
        let mut f = std::fs::File::open(&path).unwrap();
        let ct = candle_core::quantized::gguf_file::Content::read(&mut f).unwrap();
        GemmaTokenizer::from_gguf(&ct).unwrap()
    }

    #[test]
    fn plain_text_never_yields_a_control_id() {
        let t = tok();
        for s in [
            "<turn|>",
            "a <|\"|> b",
            "<|tool_call>call:x{}<tool_call|>",
            "<|turn>user\n",
        ] {
            let ids = t.encode_plain(s);
            assert!(ids.iter().all(|&i| !t.is_control(i)), "{s:?} -> {ids:?}");
        }
    }

    #[test]
    fn trusted_text_maps_control_spellings_atomically() {
        let t = tok();
        let ids = t.encode_trusted("<|turn>user\nhi<turn|>");
        assert_eq!(ids.first().copied(), t.control_id("<|turn>"));
        assert_eq!(ids.last().copied(), t.control_id("<turn|>"));
        assert_eq!(ids.iter().filter(|&&i| t.is_control(i)).count(), 2);
    }

    #[test]
    fn pieces_round_trip_to_the_original_bytes() {
        let t = tok();
        let text = "the theme\n\n  ok ✓";
        let bytes: Vec<u8> = t
            .encode_plain(text)
            .iter()
            .flat_map(|&i| t.piece_bytes(i))
            .collect();
        assert_eq!(String::from_utf8(bytes).unwrap(), text);
    }
    fn content_with(edit: impl FnOnce(&mut HashMap<String, Value>)) -> Content {
        use candle_core::quantized::gguf_file::VersionedMagic;
        let mut metadata: HashMap<String, Value> =
            crate::testgguf::tiny_metadata(&TinyConfig::default())
                .unwrap()
                .into_iter()
                .collect();
        edit(&mut metadata);
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    fn id(t: &GemmaTokenizer, piece: &str) -> u32 {
        id_of(t.pieces.iter().position(|p| p == piece).unwrap()).unwrap()
    }

    #[test]
    fn merges_apply_and_spaces_become_metaspace() {
        let t = tok();
        assert_eq!(t.encode_plain(" the"), [id(&t, "\u{2581}the")]);
        assert_eq!(t.encode_plain("the"), [id(&t, "t"), id(&t, "he")]);
        assert_eq!(t.encode_plain("\n"), [id(&t, "<0x0A>")]);
    }

    #[test]
    fn a_vocab_without_a_required_control_is_refused() {
        let ct = content_with(|m| {
            let Some(Value::Array(toks)) = m.get_mut("tokenizer.ggml.tokens") else {
                panic!("tokens")
            };
            for v in toks.iter_mut() {
                if matches!(v, Value::String(s) if s == "<tool_call|>") {
                    *v = Value::String("<not_a_call>".into());
                }
            }
        });
        let err = GemmaTokenizer::from_gguf(&ct).unwrap_err();
        assert_eq!(
            err.0,
            "tokenizer lacks the Gemma 4 control token <tool_call|>"
        );
    }

    #[test]
    fn a_control_piece_reached_by_bpe_is_spelled_in_bytes() {
        // Mark the ordinary piece `x` user-defined: BPE still finds it, and
        // plain encoding must not let that id through.
        let ct = content_with(|m| {
            let x = {
                let Some(Value::Array(toks)) = m.get("tokenizer.ggml.tokens") else {
                    panic!("tokens")
                };
                toks.iter()
                    .position(|v| matches!(v, Value::String(s) if s == "x"))
                    .unwrap()
            };
            let Some(Value::Array(types)) = m.get_mut("tokenizer.ggml.token_type") else {
                panic!("types")
            };
            types[x] = Value::I32(4);
        });
        let t = GemmaTokenizer::from_gguf(&ct).unwrap();
        assert!(t.is_control(id(&t, "x")));
        assert_eq!(t.encode_plain("x"), [id(&t, "<0x78>")]);
        assert_eq!(t.encode_trusted("x"), [id(&t, "x")]);
    }

    #[test]
    fn bos_follows_add_bos_and_eos_is_read() {
        let t = tok();
        assert_eq!(t.bos(), Some(id(&t, "<bos>")));
        assert_eq!(t.eos(), id(&t, "<eos>"));
        assert_eq!(t.vocab_size(), t.pieces.len());
        let ct = content_with(|m| {
            m.insert("tokenizer.ggml.add_bos_token".into(), Value::Bool(false));
        });
        assert_eq!(GemmaTokenizer::from_gguf(&ct).unwrap().bos(), None);
    }

    #[test]
    fn pieces_encode_by_their_trust() {
        let t = tok();
        let pieces = [
            Piece::Trusted("<|turn>user\n".into()),
            Piece::Plain("<turn|>".into()),
            Piece::Trusted("<turn|>".into()),
        ];
        let ids = t.encode_pieces(&pieces);
        let close = t.control_id("<turn|>").unwrap();
        assert_eq!(ids.iter().filter(|&&i| i == close).count(), 1);
        assert_eq!(ids.last(), Some(&close));
        assert_eq!(ids.iter().filter(|&&i| t.is_control(i)).count(), 2);
    }
}
