//! Named directional-steering vectors in `~/.plank/models/vectors.json`.
//!
//! The file is a JSON array with one entry per model, each listing its
//! vectors as base64 of the raw little-endian `f32` matrix the ds4 engine
//! loads:
//!
//! ```json
//! [
//!   { "model": "ds4vision",
//!     "vectors": [ { "name": "heretic", "value": "Pq3vPLnF..." } ] }
//! ]
//! ```
//!
//! An entry's `model` is the plank engine name (`ds4vision`); a GGUF that no
//! engine uses is listed under its file name. [`model_keys`] gives the keys a
//! run should try, in order. Fields this module does not know are kept as
//! they are on every write, and writes go through a temporary sibling so a
//! crash never leaves a truncated store.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};

use crate::Error;

/// The vectors file under the plank directory `root`.
#[must_use]
pub fn store_path_in(root: &Path) -> PathBuf {
    root.join("models").join("vectors.json")
}

/// The keys a model's entry may be listed under, most specific first: the
/// engine name when there is one, then the main file's name.
#[must_use]
pub fn model_keys(engine: Option<&str>, model: &Path) -> Vec<String> {
    let file = model.file_name().map_or_else(
        || model.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let mut keys: Vec<String> = engine.map(str::to_owned).into_iter().collect();
    if !keys.contains(&file) {
        keys.push(file);
    }
    keys
}

/// A `vectors.json` file; it need not exist yet.
#[derive(Debug, Clone)]
pub struct VectorStore {
    path: PathBuf,
}

impl VectorStore {
    /// The store at `path`.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The store under the plank home.
    #[must_use]
    pub fn at_home() -> Self {
        Self::new(store_path_in(&crate::home::plank_dir()))
    }

    /// The file this store reads and writes.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The direction names stored under any of `keys`, in key order then
    /// file order, each name once. A missing store holds none.
    ///
    /// # Errors
    /// Fails when the file exists but is not a valid store, so a typo in a
    /// hand-edited file is reported rather than read as "nothing available".
    pub fn directions(&self, keys: &[String]) -> Result<Vec<String>, Error> {
        let entries = self.read()?;
        let mut names: Vec<String> = Vec::new();
        for name in
            vectors_for(&entries, keys).filter_map(|v| v.get("name").and_then(Value::as_str))
        {
            if !names.iter().any(|n| n == name) {
                names.push(name.to_owned());
            }
        }
        Ok(names)
    }

    /// Whether `model` already has a vector called `name`.
    ///
    /// # Errors
    /// Fails when the file exists but is not a valid store.
    pub fn contains(&self, model: &str, name: &str) -> Result<bool, Error> {
        Ok(find(&self.read()?, model, name).is_some())
    }

    /// The decoded bytes of direction `name`, stored under the first of
    /// `keys` that has it, or `None` when none has it.
    ///
    /// # Errors
    /// Fails when the store is invalid or the value is not base64 of whole
    /// `f32`s.
    pub fn bytes(&self, keys: &[String], name: &str) -> Result<Option<Vec<u8>>, Error> {
        let entries = self.read()?;
        let Some(value) = vectors_for(&entries, keys)
            .find(|v| v.get("name").and_then(Value::as_str) == Some(name))
            .and_then(|v| v.get("value"))
            .and_then(Value::as_str)
        else {
            return Ok(None);
        };
        let key = keys.first().map_or("?", String::as_str);
        decode_vector(value)
            .map(Some)
            .map_err(|why| Error::msg(format!("steering: direction `{name}` of `{key}` {why}")))
    }

    /// Decodes direction `name` into a file under `cache_dir`, named by the
    /// SHA-256 of its bytes, and returns its path. The ds4 engine loads
    /// vectors only from files; naming them by content lets two names for
    /// one vector share a file, and an existing file is reused.
    ///
    /// # Errors
    /// Fails when the store is invalid, no key has that direction (the
    /// message lists the ones there are), the value is malformed, or the file
    /// cannot be written.
    pub fn materialize(
        &self,
        keys: &[String],
        name: &str,
        cache_dir: &Path,
    ) -> Result<PathBuf, Error> {
        let key = keys.first().map_or("?", String::as_str);
        let Some(bytes) = self.bytes(keys, name)? else {
            let known = self.directions(keys)?;
            return Err(Error::msg(if known.is_empty() {
                format!(
                    "steering: {} has no directions for `{key}`; build one with \
                     `pt vectorize <model> --to … --from … -n {name}`",
                    self.path.display()
                )
            } else {
                format!(
                    "steering: no direction `{name}` for `{key}` in {}; it has: {}",
                    self.path.display(),
                    known.join(", ")
                )
            }));
        };
        let hex = Sha256::digest(&bytes)
            .iter()
            .fold(String::new(), |mut hex, b| {
                let _ = write!(hex, "{b:02x}");
                hex
            });
        let file = cache_dir.join(format!("{hex}.f32"));
        if std::fs::metadata(&file).is_ok_and(|m| m.len() == bytes.len() as u64) {
            return Ok(file);
        }
        write_atomic(&file, &bytes)?;
        Ok(file)
    }

    /// Stores `bytes` (a raw little-endian `f32` matrix) as `name` under
    /// `model`, creating the file and its directory when missing. Returns
    /// whether an existing vector was replaced.
    ///
    /// # Errors
    /// Fails when the pair exists and `force` is false, when the file is not
    /// a valid store, or when it cannot be written.
    pub fn put(&self, model: &str, name: &str, bytes: &[u8], force: bool) -> Result<bool, Error> {
        let mut entries = self.read()?;
        let value = encode_base64(bytes);
        let replaced = match find_mut(&mut entries, model, name) {
            Some(_) if !force => {
                return Err(Error::msg(format!(
                    "{}: `{model}` already has a vector named `{name}`; \
                     pass --force to replace it",
                    self.path.display()
                )));
            }
            Some(existing) => {
                set_value(existing, name, &value);
                true
            }
            None => {
                vectors_of(&mut entries, model, &self.path)?
                    .push(json!({ "name": name, "value": value }));
                false
            }
        };
        self.write(&entries)?;
        Ok(replaced)
    }

    /// Classifies each incoming vector against the store, without writing.
    ///
    /// # Errors
    /// Fails when the store is invalid.
    pub fn plan(&self, incoming: Vec<Incoming>) -> Result<Vec<Planned>, Error> {
        let entries = self.read()?;
        Ok(incoming
            .into_iter()
            .map(|incoming| {
                let status = match find(&entries, &incoming.model, &incoming.name) {
                    None => Status::New,
                    Some(v) if v.get("value").and_then(Value::as_str) == Some(&incoming.value) => {
                        Status::Unchanged
                    }
                    Some(_) => Status::Conflict,
                };
                Planned { incoming, status }
            })
            .collect())
    }

    /// Applies a [`plan`](Self::plan): new vectors are added, conflicts are
    /// replaced where `overwrite` says so and skipped otherwise. The store
    /// is read again and written once, only when something changed.
    ///
    /// # Errors
    /// Fails when the store is invalid or cannot be written.
    pub fn merge(
        &self,
        plan: &[Planned],
        mut overwrite: impl FnMut(&Planned) -> bool,
    ) -> Result<MergeReport, Error> {
        let mut entries = self.read()?;
        let mut report = MergeReport::default();
        for p in plan {
            let pair = (p.incoming.model.clone(), p.incoming.name.clone());
            match p.status {
                Status::Unchanged => report.unchanged.push(pair),
                Status::Conflict if !overwrite(p) => report.skipped.push(pair),
                Status::Conflict | Status::New => {
                    if let Some(existing) =
                        find_mut(&mut entries, &p.incoming.model, &p.incoming.name)
                    {
                        *existing = p.incoming.vector.clone();
                        report.replaced.push(pair);
                    } else {
                        vectors_of(&mut entries, &p.incoming.model, &self.path)?
                            .push(p.incoming.vector.clone());
                        report.added.push(pair);
                    }
                }
            }
        }
        if !report.added.is_empty() || !report.replaced.is_empty() {
            self.write(&entries)?;
        }
        Ok(report)
    }

    /// The entries in the file; a missing or blank file is an empty store.
    fn read(&self) -> Result<Vec<Value>, Error> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::io(&self.path, e)),
        };
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        match serde_json::from_str(&text) {
            Ok(Value::Array(entries)) if entries.iter().all(Value::is_object) => Ok(entries),
            Ok(Value::Array(_)) => Err(invalid(&self.path, "every entry must be an object")),
            Ok(_) => Err(invalid(&self.path, "the top level must be a JSON array")),
            Err(e) => Err(invalid(&self.path, &e.to_string())),
        }
    }

    fn write(&self, entries: &[Value]) -> Result<(), Error> {
        let text = serde_json::to_string_pretty(entries).map_err(|e| Error::msg(e.to_string()))?;
        write_atomic(&self.path, format!("{text}\n").as_bytes())
    }
}

/// A vector read from a file to merge into a store.
#[derive(Debug, Clone, PartialEq)]
pub struct Incoming {
    /// The entry's `model`.
    pub model: String,
    /// The vector's `name`.
    pub name: String,
    /// The base64 `value`.
    pub value: String,
    /// The vector object as written, extra fields included.
    vector: Value,
}

impl Incoming {
    /// Decoded size of the vector in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        decode_base64(&self.value).map_or(0, |b| b.len())
    }
}

/// How an incoming vector relates to the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The model has no vector of this name.
    New,
    /// The store already holds exactly this value.
    Unchanged,
    /// The store holds a different value under this name.
    Conflict,
}

/// An incoming vector and what merging it would do.
#[derive(Debug, Clone, PartialEq)]
pub struct Planned {
    /// The vector.
    pub incoming: Incoming,
    /// Its relation to the store.
    pub status: Status,
}

/// What [`VectorStore::merge`] did, as `(model, name)` pairs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergeReport {
    /// Vectors the store did not have.
    pub added: Vec<(String, String)>,
    /// Conflicts replaced by the incoming value.
    pub replaced: Vec<(String, String)>,
    /// Conflicts kept as they were.
    pub skipped: Vec<(String, String)>,
    /// Vectors already stored with the same value.
    pub unchanged: Vec<(String, String)>,
}

/// Reads a vectors file to merge: the same shape as the store, every value
/// base64 of a whole number of `f32`s, and no model/name pair twice.
///
/// # Errors
/// Fails with a message naming `origin` and the first problem.
pub fn parse_file(text: &str, origin: &str) -> Result<Vec<Incoming>, Error> {
    let bad = |why: String| Error::msg(format!("{origin}: not a vectors file ({why})"));
    let Value::Array(entries) = serde_json::from_str(text).map_err(|e| bad(e.to_string()))? else {
        return Err(bad("the top level must be a JSON array".into()));
    };
    let mut out: Vec<Incoming> = Vec::new();
    for entry in &entries {
        let model = entry
            .get("model")
            .and_then(Value::as_str)
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| bad("an entry has no `model`".into()))?;
        let vectors = entry
            .get("vectors")
            .and_then(Value::as_array)
            .ok_or_else(|| bad(format!("`{model}` has no `vectors` array")))?;
        for v in vectors {
            let name = v
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.trim().is_empty())
                .ok_or_else(|| bad(format!("a vector of `{model}` has no `name`")))?;
            let value = v
                .get("value")
                .and_then(Value::as_str)
                .ok_or_else(|| bad(format!("`{model}`/`{name}` has no `value`")))?;
            decode_vector(value).map_err(|why| bad(format!("`{model}`/`{name}` {why}")))?;
            if out.iter().any(|i| i.model == model && i.name == name) {
                return Err(bad(format!("`{model}`/`{name}` appears twice")));
            }
            out.push(Incoming {
                model: model.to_owned(),
                name: name.to_owned(),
                value: value.to_owned(),
                vector: v.clone(),
            });
        }
    }
    Ok(out)
}

/// Whether `text` looks like a vectors file: a JSON array whose entries
/// carry `model` and `vectors`. Used to tell it from a profile manifest.
#[must_use]
pub fn looks_like_file(text: &str) -> bool {
    matches!(
        serde_json::from_str::<Value>(text),
        Ok(Value::Array(entries))
            if entries.iter().all(|e| e.get("model").is_some() && e.get("vectors").is_some())
    )
}

/// Standard base64 with padding.
#[must_use]
pub fn encode_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if i <= chunk.len() {
                out.push(char::from(ALPHABET[(n >> shift & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Decodes standard base64, padded or not; whitespace is ignored. `None` on
/// any other character or an impossible length.
#[must_use]
pub fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    let mut padding = 0;
    for c in text.bytes().filter(|c| !c.is_ascii_whitespace()) {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => {
                padding += 1;
                continue;
            }
            _ => return None,
        };
        if padding > 0 {
            return None;
        }
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            // `acc` holds fewer than 14 bits here, so the shift leaves one byte.
            out.push(u8::try_from(acc >> bits).unwrap_or(u8::MAX));
            acc &= (1 << bits) - 1;
        }
    }
    // Leftover bits are padding and must be zero; six of them is a lone
    // character, which no byte length encodes to.
    (bits < 6 && acc == 0 && padding <= 2).then_some(out)
}

/// A vector's bytes, or why they are not one: valid base64 of a non-empty,
/// whole number of `f32`s.
fn decode_vector(value: &str) -> Result<Vec<u8>, String> {
    let bytes = decode_base64(value).ok_or("is not valid base64")?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(format!(
            "holds {} bytes, not a whole number of f32s",
            bytes.len()
        ));
    }
    Ok(bytes)
}

/// The vectors stored under any of `keys`, in key order then file order.
fn vectors_for<'a>(entries: &'a [Value], keys: &'a [String]) -> impl Iterator<Item = &'a Value> {
    keys.iter().flat_map(move |key| {
        entries
            .iter()
            .filter(move |e| e.get("model").and_then(Value::as_str) == Some(key.as_str()))
            .filter_map(|e| e.get("vectors").and_then(Value::as_array))
            .flatten()
    })
}

fn find<'a>(entries: &'a [Value], model: &str, name: &str) -> Option<&'a Value> {
    entries
        .iter()
        .filter(|e| e.get("model").and_then(Value::as_str) == Some(model))
        .filter_map(|e| e.get("vectors").and_then(Value::as_array))
        .flatten()
        .find(|v| v.get("name").and_then(Value::as_str) == Some(name))
}

fn find_mut<'a>(entries: &'a mut [Value], model: &str, name: &str) -> Option<&'a mut Value> {
    entries
        .iter_mut()
        .filter(|e| e.get("model").and_then(Value::as_str) == Some(model))
        .filter_map(|e| e.get_mut("vectors").and_then(Value::as_array_mut))
        .flatten()
        .find(|v| v.get("name").and_then(Value::as_str) == Some(name))
}

/// Replaces a stored vector's value, keeping its other fields.
fn set_value(vector: &mut Value, name: &str, value: &str) {
    match vector.as_object_mut() {
        Some(fields) => {
            fields.insert("value".into(), Value::String(value.to_owned()));
        }
        None => *vector = json!({ "name": name, "value": value }),
    }
}

/// The `vectors` array of `model`'s entry, creating the entry if needed.
fn vectors_of<'a>(
    entries: &'a mut Vec<Value>,
    model: &str,
    path: &Path,
) -> Result<&'a mut Vec<Value>, Error> {
    let index = if let Some(i) = entries
        .iter()
        .position(|e| e.get("model").and_then(Value::as_str) == Some(model))
    {
        i
    } else {
        entries.push(json!({ "model": model, "vectors": [] }));
        entries.len() - 1
    };
    entries[index]
        .as_object_mut()
        .ok_or_else(|| invalid(path, "an entry is not an object"))?
        .entry("vectors")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| invalid(path, &format!("`vectors` of `{model}` is not an array")))
}

fn invalid(path: &Path, why: &str) -> Error {
    Error::msg(format!(
        "{}: not a vector store ({why}); fix or move it aside",
        path.display()
    ))
}

/// Replaces `file` with `bytes` through a temporary sibling.
pub(crate) fn write_atomic(file: &Path, bytes: &[u8]) -> Result<(), Error> {
    if let Some(dir) = file.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    }
    let mut tmp = file.as_os_str().to_owned();
    tmp.push(format!(".tmp-{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes).map_err(|e| Error::io(&tmp, e))?;
    std::fs::rename(&tmp, file).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        Error::io(file, e)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "plank-lib-vectors-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `[1.0, 0.0]` and `[0.0, 1.0]` as little-endian f32, base64.
    const ONE_ZERO: &str = "AACAPwAAAAA=";
    const ZERO_ONE: &str = "AAAAAAAAgD8=";

    fn keys(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    fn store(dir: &Path) -> VectorStore {
        let path = dir.join("vectors.json");
        std::fs::write(
            &path,
            format!(
                r#"[{{"model":"eng","note":"keep","vectors":[{{"name":"heretic","value":"{ONE_ZERO}","scale":2}},{{"name":"terse","value":"{ONE_ZERO}"}}]}},
                   {{"model":"m.gguf","vectors":[{{"name":"bare","value":"{ONE_ZERO}"}},{{"name":"terse","value":"AAAAAA=="}}]}}]"#
            ),
        )
        .unwrap();
        VectorStore::new(path)
    }

    fn json(store: &VectorStore) -> Value {
        serde_json::from_str(&std::fs::read_to_string(store.path()).unwrap()).unwrap()
    }

    #[test]
    fn base64_round_trips_the_rfc_vectors() {
        for (raw, enc) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encode_base64(raw.as_bytes()), enc);
            assert_eq!(decode_base64(enc).unwrap(), raw.as_bytes(), "{enc}");
        }
        assert_eq!(decode_base64("Zm9v\nYmFy").unwrap(), b"foobar");
        assert!(decode_base64("Zm9v!").is_none());
        assert!(decode_base64("Z").is_none());
        assert!(decode_base64("Zg==Zg").is_none());
    }

    #[test]
    fn the_engine_name_comes_first_then_the_file_name() {
        assert_eq!(
            model_keys(Some("ds4vision"), Path::new("/p/ds4vision.gguf")),
            ["ds4vision", "ds4vision.gguf"]
        );
        assert_eq!(model_keys(None, Path::new("/p/m.gguf")), ["m.gguf"]);
    }

    #[test]
    fn directions_are_listed_by_engine_then_file_name() {
        let dir = scratch("list");
        let s = store(&dir);
        assert_eq!(
            s.directions(&keys(&["eng", "m.gguf"])).unwrap(),
            ["heretic", "terse", "bare"]
        );
        assert!(s.directions(&keys(&["none"])).unwrap().is_empty());
        let missing = VectorStore::new(dir.join("missing.json"));
        assert!(missing.directions(&keys(&["eng"])).unwrap().is_empty());
    }

    #[test]
    fn a_direction_is_decoded_once_into_a_content_named_file() {
        let dir = scratch("materialize");
        let s = store(&dir);
        let cache = dir.join("cache");
        let both = keys(&["eng", "m.gguf"]);
        let a = s.materialize(&both, "heretic", &cache).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), [0, 0, 0x80, 0x3f, 0, 0, 0, 0]);
        assert_eq!(s.materialize(&both, "terse", &cache).unwrap(), a);
        assert_eq!(s.materialize(&both, "bare", &cache).unwrap(), a);
        let err = s.materialize(&keys(&["eng"]), "nope", &cache).unwrap_err();
        assert!(err.to_string().contains("heretic, terse"), "{err}");
        let err = s.materialize(&keys(&["x"]), "nope", &cache).unwrap_err();
        assert!(err.to_string().contains("pt vectorize"), "{err}");
    }

    #[test]
    fn put_creates_refuses_without_force_and_keeps_other_fields() {
        let dir = scratch("put");
        let fresh = VectorStore::new(dir.join("models").join("vectors.json"));
        assert!(!fresh.put("a", "v", &[0, 0, 0x80, 0x3f], false).unwrap());
        assert!(fresh.contains("a", "v").unwrap());

        let s = store(&dir);
        let err = s.put("eng", "heretic", &[0; 4], false).unwrap_err();
        assert!(err.to_string().contains("--force"), "{err}");
        assert!(s.put("eng", "heretic", &[0; 4], true).unwrap());
        let j = json(&s);
        assert_eq!(j[0]["note"], "keep");
        assert_eq!(j[0]["vectors"][0]["scale"], 2);
        assert_eq!(j[0]["vectors"][0]["value"], "AAAAAA==");
    }

    #[test]
    fn a_merge_adds_skips_or_replaces_and_reports_each() {
        let dir = scratch("merge");
        let s = store(&dir);
        let incoming = parse_file(
            &format!(
                r#"[{{"model":"eng","vectors":[
                    {{"name":"heretic","value":"{ONE_ZERO}"}},
                    {{"name":"terse","value":"{ZERO_ONE}"}},
                    {{"name":"fresh","value":"{ZERO_ONE}","by":"someone"}}]}},
                   {{"model":"new","vectors":[{{"name":"x","value":"{ONE_ZERO}"}}]}}]"#
            ),
            "remote",
        )
        .unwrap();
        let plan = s.plan(incoming).unwrap();
        let statuses: Vec<Status> = plan.iter().map(|p| p.status).collect();
        assert_eq!(
            statuses,
            [
                Status::Unchanged,
                Status::Conflict,
                Status::New,
                Status::New
            ]
        );

        // Skip the conflict first: only new vectors land.
        let report = s.merge(&plan, |_| false).unwrap();
        assert_eq!(report.skipped, [("eng".into(), "terse".into())]);
        assert_eq!(report.added.len(), 2);
        assert_eq!(report.unchanged.len(), 1);
        let j = json(&s);
        assert_eq!(j[0]["vectors"][1]["value"], ONE_ZERO, "the conflict kept");
        assert_eq!(j[0]["vectors"][2]["by"], "someone", "extra fields arrive");
        assert_eq!(j[2]["model"], "new");

        // Then overwrite it.
        let plan = s
            .plan(plan.into_iter().map(|p| p.incoming).collect())
            .unwrap();
        let report = s.merge(&plan, |p| p.incoming.name == "terse").unwrap();
        assert_eq!(report.replaced, [("eng".into(), "terse".into())]);
        assert_eq!(json(&s)[0]["vectors"][1]["value"], ZERO_ONE);
    }

    #[test]
    fn a_vectors_file_is_validated_before_anything_merges() {
        for (bad, why) in [
            (r#"{"x":1}"#, "JSON array"),
            (r#"[{"vectors":[]}]"#, "no `model`"),
            (
                r#"[{"model":"m","vectors":[{"name":"v","value":"AAAA"}]}]"#,
                "whole number",
            ),
            (
                r#"[{"model":"m","vectors":[{"name":"v","value":"@@@@"}]}]"#,
                "base64",
            ),
            (
                r#"[{"model":"m","vectors":[{"name":"v","value":"AAAAAA=="},{"name":"v","value":"AAAAAA=="}]}]"#,
                "twice",
            ),
        ] {
            let err = parse_file(bad, "f").unwrap_err().to_string();
            assert!(err.contains(why), "{bad}: {err}");
        }
        assert!(looks_like_file(r#"[{"model":"m","vectors":[]}]"#));
        assert!(!looks_like_file(r#"{"name":"p","profile":{}}"#));
    }

    #[test]
    fn a_store_of_another_shape_is_refused_not_clobbered() {
        let dir = scratch("shape");
        let path = dir.join("vectors.json");
        std::fs::write(&path, r#"{"models":[]}"#).unwrap();
        let s = VectorStore::new(&path);
        assert!(s.put("m", "v", &[0; 4], true).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), r#"{"models":[]}"#);
    }
}
