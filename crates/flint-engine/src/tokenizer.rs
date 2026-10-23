//! Flint's GGUF tokenizer: GPT-2 byte-level BPE with llama.cpp's `flint`
//! pre-tokenizer, with plain text kept inert.
//!
//! The HF tokenizer built here knows no special tokens, so BPE over any
//! string yields ordinary pieces, and should a merge ever reach a control
//! piece it is spelled out in byte tokens instead. Control ids (GGUF token
//! types 3 and 4: the turn markers, `<think>`, `<tool_call>`, ...) come only
//! from [`FlintTokenizer::encode_trusted`], which the chat template feeds
//! its own markers.

use std::collections::HashMap;

use tokenizers::Tokenizer;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::tokenizer::SplitDelimiterBehavior;

use crate::gguf::{Header, Value};
use crate::template::Piece;
use crate::{Result, err};

const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;

/// llama.cpp's `LLAMA_VOCAB_PRE_TYPE_FLINT` expressions, applied in order,
/// each splitting the pieces the one before produced.
const PRE: [&str; 3] = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
];

/// GPT-2's byte-to-character table: printable bytes map to themselves, the
/// rest to code points from 256 up, in byte order.
pub(crate) fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (u32::from(b'!')..=u32::from(b'~'))
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(&cs) {
        out[*b as usize] = char::from_u32(*c).unwrap_or('\u{FFFD}');
    }
    out
}

/// Flint's tokenizer, read from a GGUF's `tokenizer.ggml.*` metadata.
#[derive(Debug)]
pub struct FlintTokenizer {
    tokenizer: Tokenizer,
    pieces: Vec<String>,
    control: Vec<bool>,
    control_ids: HashMap<String, u32>,
    /// Control spellings by first byte, longest first.
    control_by_first: HashMap<u8, Vec<(String, u32)>>,
    byte_ids: Vec<u32>,
    byte_of: HashMap<char, u8>,
    eos: u32,
    eog: Vec<u32>,
}

fn get<'a>(h: &'a Header, k: &str) -> Result<&'a Value> {
    h.get(k)
        .ok_or_else(|| err(format!("missing GGUF metadata key `{k}`")))
}

fn strings(h: &Header, k: &str) -> Result<Vec<String>> {
    get(h, k)?
        .as_array()
        .ok_or_else(|| err(format!("`{k}` is not an array")))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| err(format!("`{k}` holds a non-string")))
        })
        .collect()
}

fn int_of(v: &Value) -> Option<i64> {
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

fn id_of(i: usize) -> Result<u32> {
    u32::try_from(i).map_err(|_| err("vocab larger than u32"))
}

fn token_id(v: &Value, vocab: usize, what: &str) -> Result<u32> {
    int_of(v)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| (n as usize) < vocab)
        .ok_or_else(|| err(format!("{what} is not a token id in a {vocab}-token vocab")))
}

/// The id of each byte's token, the first one on a duplicated spelling.
fn byte_token_ids(table: &[char; 256], pieces: &[String]) -> Result<Vec<u32>> {
    let index: HashMap<&str, usize> = pieces
        .iter()
        .enumerate()
        .rev()
        .map(|(i, p)| (p.as_str(), i))
        .collect();
    table
        .iter()
        .enumerate()
        .map(|(b, c)| {
            index.get(c.to_string().as_str()).map_or_else(
                || Err(err(format!("tokenizer lacks the byte token for {b:#04x}"))),
                |&i| id_of(i),
            )
        })
        .collect()
}

fn build_hf(pieces: &[String], merges: Vec<(String, String)>) -> Result<Tokenizer> {
    let hf = |e: tokenizers::Error| err(format!("tokenizer: {e}"));
    let mut vocab = Vocab::default();
    for (i, p) in pieces.iter().enumerate() {
        // A duplicated spelling keeps its first id, as llama.cpp does.
        vocab.entry(p.clone()).or_insert(id_of(i)?);
    }
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(hf)?;
    let mut tokenizer = Tokenizer::new(bpe);
    let mut steps = Vec::new();
    for re in PRE {
        steps.push(
            Split::new(
                SplitPattern::Regex(re.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(hf)?
            .into(),
        );
    }
    steps.push(ByteLevel::new(false, false, false).into());
    tokenizer.with_pre_tokenizer(Some(Sequence::new(steps)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    Ok(tokenizer)
}

impl FlintTokenizer {
    /// Reads the tokenizer from a GGUF header.
    ///
    /// # Errors
    /// A model other than `gpt2` with pre `flint`, a missing or malformed
    /// `tokenizer.ggml.*` key, a byte without its token, or an id outside
    /// the vocab.
    pub fn from_header(h: &Header) -> Result<Self> {
        for (k, want) in [
            ("tokenizer.ggml.model", "gpt2"),
            ("tokenizer.ggml.pre", "flint"),
        ] {
            let got = get(h, k)?
                .as_str()
                .ok_or_else(|| err(format!("`{k}` is not a string")))?;
            if got != want {
                return Err(err(format!(
                    "`{k}` is `{got}`, flint-engine reads `{want}`"
                )));
            }
        }
        let pieces = strings(h, "tokenizer.ggml.tokens")?;
        let merges = strings(h, "tokenizer.ggml.merges")?
            .into_iter()
            .map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .ok_or_else(|| err(format!("invalid merge entry `{m}`")))
            })
            .collect::<Result<Vec<_>>>()?;
        let types = get(h, "tokenizer.ggml.token_type")?
            .as_array()
            .ok_or_else(|| err("`tokenizer.ggml.token_type` is not an array"))?;
        if types.len() != pieces.len() {
            return Err(err(format!(
                "{} token types for {} tokens",
                types.len(),
                pieces.len()
            )));
        }
        let control: Vec<bool> = types
            .iter()
            .map(|t| matches!(int_of(t), Some(TYPE_CONTROL | TYPE_USER_DEFINED)))
            .collect();
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
        let table = bytes_to_unicode();
        let byte_of: HashMap<char, u8> = table
            .iter()
            .enumerate()
            .map(|(b, c)| (*c, u8::try_from(b).unwrap_or(0)))
            .collect();
        let byte_ids = byte_token_ids(&table, &pieces)?;
        let eos = token_id(
            get(h, "tokenizer.ggml.eos_token_id")?,
            pieces.len(),
            "eos_token_id",
        )?;
        let eog = match h
            .get("tokenizer.ggml.eog_token_ids")
            .and_then(Value::as_array)
        {
            Some(a) => a
                .iter()
                .map(|v| token_id(v, pieces.len(), "an eog_token_ids entry"))
                .collect::<Result<Vec<_>>>()?,
            None => vec![eos],
        };
        let tokenizer = build_hf(&pieces, merges)?;
        Ok(Self {
            tokenizer,
            pieces,
            control,
            control_ids,
            control_by_first,
            byte_ids,
            byte_of,
            eos,
            eog,
        })
    }

    fn bytes_of(&self, bytes: &[u8]) -> Vec<u32> {
        bytes
            .iter()
            .map(|&b| self.byte_ids[usize::from(b)])
            .collect()
    }

    /// Tokenizes untrusted text. Never yields a control id.
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

    fn control_at(&self, text: &str) -> Option<(usize, u32)> {
        let first = *text.as_bytes().first()?;
        self.control_by_first
            .get(&first)?
            .iter()
            .find(|(s, _)| text.starts_with(s.as_str()))
            .map(|(s, id)| (s.len(), *id))
    }

    /// Tokenizes trusted text: each control spelling, longest first at each
    /// position, becomes its id; the gaps are encoded as plain text.
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

    /// Tokenizes template pieces, each by its trust.
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

    /// The bytes token `id` stands for: a control token its spelling, any
    /// other piece read through the byte-level table. Empty outside the vocab.
    #[must_use]
    pub fn piece_bytes(&self, id: u32) -> Vec<u8> {
        let Some(piece) = self.pieces.get(id as usize) else {
            return Vec::new();
        };
        if self.is_control(id) {
            return piece.as_bytes().to_vec();
        }
        let mut out = Vec::with_capacity(piece.len());
        for c in piece.chars() {
            match self.byte_of.get(&c) {
                Some(b) => out.push(*b),
                None => out.extend_from_slice(c.to_string().as_bytes()),
            }
        }
        out
    }

    /// The id of a control token by its spelling.
    #[must_use]
    pub fn control_id(&self, spelling: &str) -> Option<u32> {
        self.control_ids.get(spelling).copied()
    }

    /// Whether `id` is control text (GGUF token type 3 or 4).
    #[must_use]
    pub fn is_control(&self, id: u32) -> bool {
        self.control.get(id as usize).copied().unwrap_or(false)
    }

    /// Whether generation stops at `id`.
    #[must_use]
    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }

    /// The end-of-text id (`tokenizer.ggml.eos_token_id`).
    #[must_use]
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// The end-of-generation ids (`tokenizer.ggml.eog_token_ids`).
    #[must_use]
    pub fn eog(&self) -> &[u32] {
        &self.eog
    }

    /// The number of tokens in the vocab.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgguf::Builder;

    /// Every byte as its own token, two merges, and four control tokens.
    fn tiny() -> FlintTokenizer {
        let table = bytes_to_unicode();
        let mut tokens: Vec<String> = table.iter().map(char::to_string).collect();
        let mut types = vec![1i32; 256];
        for (t, ty) in [
            ("ab", 1),
            ("Ġw", 1),
            ("<|end_of_text|>", 3),
            ("<|user|>", 3),
            ("<think>", 4),
            ("<tool_call>", 4),
        ] {
            tokens.push(t.to_string());
            types.push(ty);
        }
        let eos =
            u32::try_from(tokens.iter().position(|t| t == "<|end_of_text|>").unwrap()).unwrap();
        let user = u32::try_from(tokens.iter().position(|t| t == "<|user|>").unwrap()).unwrap();
        let header = Header::parse(
            &Builder::new()
                .kv("tokenizer.ggml.model", &Value::Str("gpt2".into()))
                .kv("tokenizer.ggml.pre", &Value::Str("flint".into()))
                .kv(
                    "tokenizer.ggml.tokens",
                    &Value::Array(tokens.into_iter().map(Value::Str).collect()),
                )
                .kv(
                    "tokenizer.ggml.token_type",
                    &Value::Array(types.into_iter().map(Value::I32).collect()),
                )
                .kv(
                    "tokenizer.ggml.merges",
                    &Value::Array(vec![Value::Str("a b".into()), Value::Str("Ġ w".into())]),
                )
                .kv("tokenizer.ggml.eos_token_id", &Value::U32(eos))
                .kv(
                    "tokenizer.ggml.eog_token_ids",
                    &Value::Array(vec![Value::U32(eos), Value::U32(user)]),
                )
                .bytes(),
        )
        .unwrap();
        FlintTokenizer::from_header(&header).unwrap()
    }

    fn id(t: &FlintTokenizer, piece: &str) -> u32 {
        u32::try_from(t.pieces.iter().position(|p| p == piece).unwrap()).unwrap()
    }

    #[test]
    fn merges_apply_and_spaces_are_byte_level() {
        let t = tiny();
        assert_eq!(t.encode_plain("ab"), vec![id(&t, "ab")]);
        assert_eq!(t.encode_plain(" w"), vec![id(&t, "Ġw")]);
        let ids = t.encode_plain("ab w");
        let back: Vec<u8> = ids.iter().flat_map(|&i| t.piece_bytes(i)).collect();
        assert_eq!(back, b"ab w");
    }

    #[test]
    fn plain_text_never_yields_a_control_id() {
        let t = tiny();
        for text in ["<|user|>", "<think>", "x<tool_call>y"] {
            let ids = t.encode_plain(text);
            assert!(ids.iter().all(|&i| !t.is_control(i)), "{text}: {ids:?}");
            let back: Vec<u8> = ids.iter().flat_map(|&i| t.piece_bytes(i)).collect();
            assert_eq!(back, text.as_bytes());
        }
    }

    #[test]
    fn trusted_text_maps_control_spellings() {
        let t = tiny();
        let ids = t.encode_trusted("<|user|>ab<think>");
        assert_eq!(
            ids,
            vec![id(&t, "<|user|>"), id(&t, "ab"), id(&t, "<think>")]
        );
        assert_eq!(t.piece_bytes(id(&t, "<think>")), b"<think>");
    }

    #[test]
    fn eog_comes_from_the_gguf() {
        let t = tiny();
        assert!(t.is_eog(id(&t, "<|user|>")));
        assert!(t.is_eog(t.eos()));
        assert!(!t.is_eog(id(&t, "ab")));
    }

    #[test]
    fn utf8_survives_the_byte_level_round_trip() {
        let t = tiny();
        let s = "héllo 日本 🙂";
        let back: Vec<u8> = t
            .encode_plain(s)
            .iter()
            .flat_map(|&i| t.piece_bytes(i))
            .collect();
        assert_eq!(back, s.as_bytes());
    }

    #[test]
    fn refuses_another_pre_tokenizer() {
        let h = Header::parse(
            &Builder::new()
                .kv("tokenizer.ggml.model", &Value::Str("gpt2".into()))
                .kv("tokenizer.ggml.pre", &Value::Str("llama3".into()))
                .bytes(),
        )
        .unwrap();
        assert!(
            FlintTokenizer::from_header(&h)
                .unwrap_err()
                .0
                .contains("llama3")
        );
    }
}
