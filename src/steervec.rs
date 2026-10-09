//! Named directional-steering vectors from `~/.plank/models/vectors.json`.
//!
//! The file is written by `pt vectorize -n NAME` (`../plank-replay`): a JSON
//! array with one entry per model file, each listing its vectors as base64 of
//! the raw little-endian `f32` matrix ds4 loads.
//!
//! ```json
//! [
//!   { "model": "ds4vision.gguf",
//!     "vectors": [ { "name": "heretic", "value": "Pq3vPLnF..." } ] }
//! ]
//! ```
//!
//! An entry's `model` is the plank engine name (`ds4vision`), which is how
//! `pt vectorize` resolves its model argument, through the same catalog. A run
//! on a bare GGUF path that is no engine is matched by the file's name
//! instead; [`model_keys`] gives the keys to try, in order.
//! The C engine only takes a vector as a file, so a chosen one is decoded
//! into `~/.plank/cache/steering/<sha256>.f32`, named by its content: two
//! names for the same vector share one file, and the KV key (which digests
//! the file's bytes, `ds4engine::steering_key`) is the same either way.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest as _, Sha256};

/// The vectors file under the plank directory `root`.
#[must_use]
pub fn store_path_in(root: &Path) -> PathBuf {
    root.join("models").join("vectors.json")
}

/// [`store_path_in`] rooted at `~/.plank`.
#[must_use]
pub fn store_path() -> PathBuf {
    store_path_in(&crate::manifest::plank_dir())
}

/// The keys a model's entry may be listed under, most specific first: the
/// engine name when the run selected one, then the main file's name.
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

/// The vectors stored under any of `keys`, in key order then file order.
fn vectors_for<'a>(
    entries: &'a [serde_json::Value],
    keys: &'a [String],
) -> impl Iterator<Item = &'a serde_json::Value> {
    keys.iter().flat_map(move |key| {
        entries
            .iter()
            .filter(move |e| {
                e.get("model").and_then(serde_json::Value::as_str) == Some(key.as_str())
            })
            .filter_map(|e| e.get("vectors").and_then(serde_json::Value::as_array))
            .flatten()
    })
}

/// The direction names stored for the model under `keys`, in key order then
/// file order, each name once.
///
/// A missing store is no directions. A store that cannot be parsed is an
/// error, so a typo in a hand-edited file is reported rather than read as
/// "nothing available".
///
/// # Errors
/// Returns a message naming the file when it exists but is not a valid store.
pub fn directions_in(store: &Path, keys: &[String]) -> Result<Vec<String>, String> {
    let entries = read(store)?;
    let mut names: Vec<String> = Vec::new();
    for name in vectors_for(&entries, keys)
        .filter_map(|v| v.get("name").and_then(serde_json::Value::as_str))
    {
        if !names.iter().any(|n| n == name) {
            names.push(name.to_owned());
        }
    }
    Ok(names)
}

/// [`directions_in`] on the default store.
///
/// # Errors
/// As [`directions_in`].
pub fn directions(keys: &[String]) -> Result<Vec<String>, String> {
    directions_in(&store_path(), keys)
}

/// Decodes direction `name`, stored under the first of `keys` that has it,
/// into a vector file under `cache_dir`
/// and returns its path. An existing file of the same content is reused.
///
/// # Errors
/// Returns a message when the store is invalid, the model has no direction of
/// that name (the message lists the ones it has), the value is not base64 of
/// whole `f32`s, or the file cannot be written.
pub fn materialize_in(
    store: &Path,
    cache_dir: &Path,
    keys: &[String],
    name: &str,
) -> Result<PathBuf, String> {
    let key = keys.first().map_or("?", String::as_str);
    let entries = read(store)?;
    let value = vectors_for(&entries, keys)
        .find(|v| v.get("name").and_then(serde_json::Value::as_str) == Some(name))
        .and_then(|v| v.get("value"))
        .and_then(serde_json::Value::as_str);
    let Some(value) = value else {
        let known = directions_in(store, keys)?;
        return Err(if known.is_empty() {
            format!(
                "steering: {} has no directions for `{key}`; build one with \
                 `pt vectorize <model> --to … --from … -n {name}`",
                store.display()
            )
        } else {
            format!(
                "steering: no direction `{name}` for `{key}` in {}; it has: {}",
                store.display(),
                known.join(", ")
            )
        });
    };
    let bytes = decode_base64(value)
        .ok_or_else(|| format!("steering: direction `{name}` of `{key}` is not valid base64"))?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(format!(
            "steering: direction `{name}` of `{key}` holds {} bytes, not a whole number of f32s",
            bytes.len()
        ));
    }
    let digest = Sha256::digest(&bytes);
    let hex = digest.iter().fold(String::new(), |mut hex, b| {
        let _ = write!(hex, "{b:02x}");
        hex
    });
    let file = cache_dir.join(format!("{hex}.f32"));
    if std::fs::metadata(&file).is_ok_and(|m| m.len() == bytes.len() as u64) {
        return Ok(file);
    }
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| format!("steering: {}: {e}", cache_dir.display()))?;
    let tmp = cache_dir.join(format!("{hex}.f32.tmp-{}", std::process::id()));
    std::fs::write(&tmp, &bytes).map_err(|e| format!("steering: {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &file).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("steering: {}: {e}", file.display())
    })?;
    Ok(file)
}

/// [`materialize_in`] on the default store and cache.
///
/// # Errors
/// As [`materialize_in`].
pub fn materialize(keys: &[String], name: &str) -> Result<PathBuf, String> {
    let root = crate::manifest::plank_dir();
    materialize_in(
        &store_path_in(&root),
        &root.join("cache").join("steering"),
        keys,
        name,
    )
}

/// The store's entries; a missing or blank file is an empty store.
fn read(store: &Path) -> Result<Vec<serde_json::Value>, String> {
    let text = match std::fs::read_to_string(store) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("steering: {}: {e}", store.display())),
    };
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    match serde_json::from_str(&text) {
        Ok(serde_json::Value::Array(entries)) => Ok(entries),
        Ok(_) => Err(format!(
            "steering: {}: the top level must be a JSON array",
            store.display()
        )),
        Err(e) => Err(format!("steering: {}: {e}", store.display())),
    }
}

/// Decodes standard base64, padded or not; whitespace is ignored. `None` on
/// any other character or an impossible length.
fn decode_base64(text: &str) -> Option<Vec<u8>> {
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

/// A steering direction and the scales to run it at: what a profile's
/// `steering` block names (`crate::profile::ProfileSpec::steering`).
#[derive(Debug, Clone, PartialEq)]
pub struct Steering {
    /// The direction's name (`--dir-steering`), looked up for the model in
    /// `~/.plank/models/vectors.json`.
    pub direction: String,
    /// FFN scale (`--dir-steering-ffn`); 1.0 when the block gives none, like
    /// the C.
    pub ffn: f32,
    /// Attention scale (`--dir-steering-attn`); 0.0 when the block gives none.
    pub attn: f32,
    /// `"from": "user"` (the default) or `"all"`: whether the FFN edit starts
    /// at the user's first message or at the first prompt token. `user`
    /// requires `attn` to be 0, since only the FFN scale can be switched on a
    /// live session, so a block with an attention scale defaults to `all`.
    pub from_user: bool,
}

/// Reads a `steering` block: `{"direction": NAME, "ffn": F, "attn": F,
/// "from": "user"|"all"}`, only `direction` required. The scales are
/// range-checked like the command-line flags; the old `file` key is refused
/// with a pointer to its replacement rather than silently ignored.
///
/// # Errors
/// Describes the first problem found, prefixed `steering:`.
pub fn parse_steering(v: &serde_json::Value) -> Result<Steering, String> {
    if !v.is_object() {
        return Err("steering: must be an object".to_string());
    }
    if v.get("file").is_some() {
        return Err(
            "steering: `file` was replaced by `direction`, a name stored for the model in \
             ~/.plank/models/vectors.json (`pt vectorize … -n NAME`)"
                .to_string(),
        );
    }
    let direction = v
        .get("direction")
        .and_then(serde_json::Value::as_str)
        .filter(|d| !d.trim().is_empty())
        .ok_or("steering: needs a `direction` name")?;
    let scale = |key: &str, default: f32| -> Result<f32, String> {
        let Some(x) = v.get(key) else {
            return Ok(default);
        };
        let n = x
            .as_f64()
            .ok_or_else(|| format!("steering: `{key}` must be a number"))?;
        if !(-100.0..=100.0).contains(&n) {
            return Err(format!("steering: `{key}` must be within -100..100"));
        }
        #[allow(clippy::cast_possible_truncation)]
        Ok(n as f32)
    };
    let (ffn, attn) = (scale("ffn", 1.0)?, scale("attn", 0.0)?);
    let from_user = match v.get("from").and_then(serde_json::Value::as_str) {
        // The default defers the FFN edit, which an attention edit rules out.
        None => attn == 0.0,
        Some("all") => false,
        Some("user") if attn != 0.0 => {
            return Err(
                "steering: `from: user` defers only the FFN edit, so `attn` must be 0".to_string(),
            );
        }
        Some("user") => true,
        Some(other) => {
            return Err(format!(
                "steering: `from` must be \"all\" or \"user\", not `{other}`"
            ));
        }
    };
    Ok(Steering {
        direction: direction.to_string(),
        ffn,
        attn,
        from_user,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "plank-steervec-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `[1.0, 0.0]` as little-endian f32, base64.
    const ONE_ZERO: &str = "AACAPwAAAAA=";

    fn store(dir: &Path) -> PathBuf {
        let path = dir.join("vectors.json");
        std::fs::write(
            &path,
            format!(
                r#"[{{"model":"eng","vectors":[{{"name":"heretic","value":"{ONE_ZERO}"}},{{"name":"terse","value":"{ONE_ZERO}"}}]}},
                   {{"model":"m.gguf","vectors":[{{"name":"bare","value":"{ONE_ZERO}"}},{{"name":"terse","value":"AAAAAA=="}}]}},
                   {{"model":"other","vectors":[{{"name":"x","value":"AAAAAA=="}}]}}]"#
            ),
        )
        .unwrap();
        path
    }

    fn keys(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn base64_round_trips_the_rfc_vectors() {
        for (raw, enc) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foobar", "Zm9vYmFy"),
        ] {
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
        // Engine entries first, then the file's own, each name once.
        assert_eq!(
            directions_in(&s, &keys(&["eng", "m.gguf"])).unwrap(),
            ["heretic", "terse", "bare"]
        );
        assert_eq!(
            directions_in(&s, &keys(&["m.gguf"])).unwrap(),
            ["bare", "terse"]
        );
        assert!(directions_in(&s, &keys(&["none"])).unwrap().is_empty());
        assert!(
            directions_in(&dir.join("missing.json"), &keys(&["eng"]))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_direction_is_decoded_once_into_a_content_named_file() {
        let dir = scratch("materialize");
        let s = store(&dir);
        let cache = dir.join("cache");
        let both = keys(&["eng", "m.gguf"]);
        let a = materialize_in(&s, &cache, &both, "heretic").unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), [0, 0, 0x80, 0x3f, 0, 0, 0, 0]);
        // Same bytes under another name: the same file. The engine's `terse`
        // shadows the one stored under the file name.
        let b = materialize_in(&s, &cache, &both, "terse").unwrap();
        assert_eq!(a, b);
        // A file-name entry is found when the engine has no such name.
        let c = materialize_in(&s, &cache, &both, "bare").unwrap();
        assert_eq!(a, c);
    }

    #[test]
    fn an_unknown_name_lists_what_the_model_has() {
        let dir = scratch("unknown");
        let s = store(&dir);
        let err = materialize_in(&s, &dir, &keys(&["eng"]), "nope").unwrap_err();
        assert!(err.contains("heretic, terse"), "{err}");
        let err = materialize_in(&s, &dir, &keys(&["unknown"]), "nope").unwrap_err();
        assert!(err.contains("pt vectorize"), "{err}");
    }

    #[test]
    fn a_value_of_partial_floats_is_refused() {
        let dir = scratch("partial");
        let s = dir.join("vectors.json");
        std::fs::write(
            &s,
            r#"[{"model":"eng","vectors":[{"name":"bad","value":"AAAA"}]}]"#,
        )
        .unwrap();
        let err = materialize_in(&s, &dir, &keys(&["eng"]), "bad").unwrap_err();
        assert!(err.contains("whole number"), "{err}");
    }

    #[test]
    fn a_malformed_store_is_an_error_not_an_empty_list() {
        let dir = scratch("malformed");
        let s = dir.join("vectors.json");
        std::fs::write(&s, r#"{"models":[]}"#).unwrap();
        assert!(directions_in(&s, &keys(&["eng"])).is_err());
    }

    #[test]
    fn a_steering_block_defaults_ffn_to_one_and_attn_to_zero() {
        let st = parse_steering(&serde_json::json!({"direction": "heretic"})).unwrap();
        assert_eq!(st.direction, "heretic");
        assert!((st.ffn - 1.0).abs() < f32::EPSILON);
        assert!(st.attn.abs() < f32::EPSILON);
        assert!(st.from_user, "deferred start is the default");
    }

    #[test]
    fn an_attention_edit_steers_every_token_by_default() {
        let st =
            parse_steering(&serde_json::json!({"direction": "d", "ffn": 0, "attn": 1})).unwrap();
        assert!((st.attn - 1.0).abs() < f32::EPSILON && st.ffn.abs() < f32::EPSILON);
        assert!(!st.from_user);
        let st = parse_steering(&serde_json::json!({"direction": "d", "ffn": 5, "from": "user"}))
            .unwrap();
        assert!(st.from_user);
    }

    #[test]
    fn a_bad_steering_block_is_refused() {
        for bad in [
            serde_json::json!({"ffn": 3}),
            serde_json::json!({"direction": " ", "ffn": 3}),
            serde_json::json!({"direction": "d", "ffn": 500}),
            serde_json::json!({"direction": "d", "attn": "x"}),
            serde_json::json!({"direction": "d", "from": "later"}),
            serde_json::json!({"direction": "d", "from": "user", "attn": 1}),
            serde_json::json!({"file": "/v/h.f32", "ffn": 3}),
            serde_json::json!("heretic"),
        ] {
            assert!(parse_steering(&bad).is_err(), "{bad}");
        }
        let err = parse_steering(&serde_json::json!({"file": "/v"})).unwrap_err();
        assert!(err.contains("replaced by `direction`"), "{err}");
    }
}
