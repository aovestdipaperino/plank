//! Just enough GGUF reading to learn which model family a file is, and
//! whether it is the checkpoint the engine will pair with a vision encoder.
//!
//! plank has to know this *before* `ds4_engine_open`, and the engine cannot
//! tell it: the C detects the family while opening, but the companion-GGUF
//! path has to be in the options struct by then, and it goes in a different
//! field per family (`mtp_path` for a `DeepSeek` draft model, `ple_path` for a
//! Qwen sidecar). Putting it in the wrong one is a hard error — the C refuses
//! a `ple_path` for a non-Qwen model — so guessing is not an option and
//! opening twice is not either, since the first open pays the whole residency
//! cost.
//!
//! The discriminator is the file's own `general.architecture` string, which is
//! what the C switches on (`config_validate_model`). Only the header and the
//! metadata keys are read, and only up to the key in question.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// The model families plank treats differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelFamily {
    /// `DeepSeek` V4 and anything else the C falls through to.
    #[default]
    Ds4,
    /// Qwen3.8-Flash-Next (`qwen4exp`).
    Qwen,
}

impl From<trace_stream::syntax::ToolSyntax> for ModelFamily {
    /// The two enums answer the same question from different sides — the
    /// dialect is read from the engine's reported shape name after opening,
    /// the family from the file's metadata before it — and they live in
    /// different crates, so they cannot be one type. This is the single place
    /// they are reconciled, rather than a second string matcher.
    fn from(syntax: trace_stream::syntax::ToolSyntax) -> Self {
        match syntax {
            trace_stream::syntax::ToolSyntax::Qwen => Self::Qwen,
            trace_stream::syntax::ToolSyntax::Dsml => Self::Ds4,
        }
    }
}

/// The `general.architecture` value the C matches for Qwen3.8-Flash-Next.
const QWEN_ARCH: &str = "qwen4exp";

/// Refuses to allocate for a declared length beyond this. The file may be
/// truncated or not a GGUF at all, and a bogus 64-bit length would otherwise
/// become an allocation.
const MAX_STRING_BYTES: u64 = 64 * 1024;

/// Metadata keys are walked in order; a file claiming more than this is not
/// one the engine would load either.
const MAX_KV_PAIRS: u64 = 1 << 20;

/// The family of the model at `path`, defaulting to [`ModelFamily::Ds4`].
///
/// Unreadable, truncated, or non-GGUF files read as `Ds4`, which is the same
/// fallthrough the C takes for an unrecognized architecture. Getting a real
/// answer is what matters here; a wrong file fails at open with the engine's
/// own diagnostic, which is better than one invented here.
#[must_use]
pub fn family_of(path: &Path) -> ModelFamily {
    match architecture(path).as_deref() {
        Some(QWEN_ARCH) => ModelFamily::Qwen,
        _ => ModelFamily::Ds4,
    }
}

/// Reads `general.architecture` out of a GGUF file's metadata.
///
/// `None` when the file is not GGUF, is truncated, or has no such key.
#[must_use]
pub fn architecture(path: &Path) -> Option<String> {
    string_value(path, "general.architecture")
}

/// The `deepseek4.checkpoint_variant` value the C requires before it will
/// bind the `DeepSeek` vision encoder (`g_ds4_flash_vision_exp`).
const VISION_EXP_VARIANT: &str = "vision-exp";

/// Whether the engine will accept a vision encoder alongside the model at
/// `path`.
///
/// The C refuses `ds4_engine_open` outright — "--vision requires ... the
/// pinned `DeepSeek` V4 Flash Vision-Exp model" — when a `vision_path` is set
/// and the main GGUF is not that checkpoint, so plank has to know before the
/// open whether to pass one at all. Any other `DeepSeek` V4 checkpoint (a
/// language-only quant, an abliterated re-quant) is text-only; the
/// `view_image` tool then refuses at call time exactly as it does when the
/// encoder file is missing. Qwen is answered elsewhere: the C would accept a
/// Qwen encoder, but plank does not ship one, so this stays a `DeepSeek`
/// question.
#[must_use]
pub fn supports_vision(path: &Path) -> bool {
    string_value(path, "deepseek4.checkpoint_variant").as_deref() == Some(VISION_EXP_VARIANT)
}

/// Reads one string-typed metadata value out of a GGUF file.
///
/// `None` when the file is not GGUF, is truncated, has no such key, or the
/// key holds a non-string value.
#[must_use]
pub fn string_value(path: &Path, wanted: &str) -> Option<String> {
    let mut r = BufReader::new(File::open(path).ok()?);
    if &read_exact::<4>(&mut r)? != b"GGUF" {
        return None;
    }
    let _version = u32::from_le_bytes(read_exact::<4>(&mut r)?);
    let _tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    if kv_count > MAX_KV_PAIRS {
        return None;
    }
    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let ty = u32::from_le_bytes(read_exact::<4>(&mut r)?);
        if key == wanted {
            // Type 8 is STRING. A different type here means the file is not
            // shaped the way the engine expects, so say nothing rather than
            // coerce it.
            return (ty == 8).then(|| read_string(&mut r)).flatten();
        }
        skip_value(&mut r, ty)?;
    }
    None
}

fn read_exact<const N: usize>(r: &mut impl Read) -> Option<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf).ok()?;
    Some(buf)
}

fn read_u64(r: &mut impl Read) -> Option<u64> {
    Some(u64::from_le_bytes(read_exact::<8>(r)?))
}

fn read_string(r: &mut impl Read) -> Option<String> {
    let len = read_u64(r)?;
    if len > MAX_STRING_BYTES {
        return None;
    }
    let mut buf = vec![0u8; usize::try_from(len).ok()?];
    r.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

/// Width of a fixed-size GGUF scalar, or `None` for the variable-size types.
fn scalar_width(ty: u32) -> Option<i64> {
    match ty {
        0 | 1 | 7 => Some(1), // uint8, int8, bool
        2..=3 => Some(2),     // uint16, int16
        4..=6 => Some(4),     // uint32, int32, float32
        10..=12 => Some(8),   // uint64, int64, float64
        _ => None,
    }
}

/// Advances past one metadata value of type `ty`.
fn skip_value<R: Read + Seek>(r: &mut BufReader<R>, ty: u32) -> Option<()> {
    if let Some(width) = scalar_width(ty) {
        r.seek(SeekFrom::Current(width)).ok()?;
        return Some(());
    }
    match ty {
        8 => {
            let len = read_u64(r)?;
            r.seek(SeekFrom::Current(i64::try_from(len).ok()?)).ok()?;
            Some(())
        }
        9 => {
            let elem = u32::from_le_bytes(read_exact::<4>(r)?);
            let count = read_u64(r)?;
            if let Some(width) = scalar_width(elem) {
                let bytes = i64::try_from(count).ok()?.checked_mul(width)?;
                r.seek(SeekFrom::Current(bytes)).ok()?;
                return Some(());
            }
            // A string array is the only variable-width element the format
            // uses in practice; nested arrays are refused rather than walked.
            if elem != 8 {
                return None;
            }
            for _ in 0..count {
                let len = read_u64(r)?;
                r.seek(SeekFrom::Current(i64::try_from(len).ok()?)).ok()?;
            }
            Some(())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a GGUF header with the given metadata keys, so the tests do not
    /// depend on an 80 GB file being present.
    #[derive(Default)]
    struct Gguf {
        kv: Vec<u8>,
        count: u64,
    }

    impl Gguf {
        fn str_val(mut self, key: &str, val: &str) -> Self {
            self.push_str(key);
            self.kv.extend_from_slice(&8u32.to_le_bytes());
            self.push_str(val);
            self.count += 1;
            self
        }

        fn u32_val(mut self, key: &str, val: u32) -> Self {
            self.push_str(key);
            self.kv.extend_from_slice(&4u32.to_le_bytes());
            self.kv.extend_from_slice(&val.to_le_bytes());
            self.count += 1;
            self
        }

        /// A string array, the one variable-width array shape the format uses
        /// — a tokenizer vocab, in practice, and the thing a naive skip walks
        /// straight off the end of.
        fn str_array(mut self, key: &str, vals: &[&str]) -> Self {
            self.push_str(key);
            self.kv.extend_from_slice(&9u32.to_le_bytes());
            self.kv.extend_from_slice(&8u32.to_le_bytes());
            self.kv
                .extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                self.push_str(v);
            }
            self.count += 1;
            self
        }

        fn u32_array(mut self, key: &str, vals: &[u32]) -> Self {
            self.push_str(key);
            self.kv.extend_from_slice(&9u32.to_le_bytes());
            self.kv.extend_from_slice(&4u32.to_le_bytes());
            self.kv
                .extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                self.kv.extend_from_slice(&v.to_le_bytes());
            }
            self.count += 1;
            self
        }

        fn push_str(&mut self, s: &str) {
            self.kv.extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.kv.extend_from_slice(s.as_bytes());
        }

        fn write(self, name: &str) -> std::path::PathBuf {
            let path =
                std::env::temp_dir().join(format!("plank-gguf-{}-{name}", std::process::id()));
            let mut f = File::create(&path).expect("create");
            f.write_all(b"GGUF").unwrap();
            f.write_all(&3u32.to_le_bytes()).unwrap();
            f.write_all(&0u64.to_le_bytes()).unwrap(); // tensor count
            f.write_all(&self.count.to_le_bytes()).unwrap();
            f.write_all(&self.kv).unwrap();
            path
        }
    }

    /// The probe and the dialect selector must agree, or the footer would
    /// name one family while the parser used the other's syntax.
    #[test]
    fn the_dialect_and_the_family_agree() {
        use trace_stream::syntax::ToolSyntax;
        assert_eq!(ModelFamily::from(ToolSyntax::Qwen), ModelFamily::Qwen);
        assert_eq!(ModelFamily::from(ToolSyntax::Dsml), ModelFamily::Ds4);
        assert_eq!(
            ModelFamily::from(ToolSyntax::for_model_name("Qwen3.8 Flash Next")),
            ModelFamily::Qwen
        );
        assert_eq!(
            ModelFamily::from(ToolSyntax::for_model_name("DeepSeek V4 Flash")),
            ModelFamily::Ds4
        );
    }

    #[test]
    fn reads_the_architecture_string() {
        let p = Gguf::default()
            .str_val("general.architecture", "qwen4exp")
            .write("arch");
        assert_eq!(architecture(&p).as_deref(), Some("qwen4exp"));
        assert_eq!(family_of(&p), ModelFamily::Qwen);
        let _ = std::fs::remove_file(p);
    }

    /// The key is not first in a real file, so every other value type has to
    /// be skipped correctly to reach it. A wrong skip width lands mid-value
    /// and reads garbage.
    #[test]
    fn skips_preceding_values_of_every_shape() {
        let p = Gguf::default()
            .u32_val("general.file_type", 7)
            .str_val("general.name", "DeepSeek V4 Flash")
            .u32_array("some.dims", &[1, 2, 3, 4])
            .str_array("tokenizer.ggml.tokens", &["a", "bb", "ccc"])
            .str_val("general.architecture", "deepseek4")
            .write("skip");
        assert_eq!(architecture(&p).as_deref(), Some("deepseek4"));
        assert_eq!(family_of(&p), ModelFamily::Ds4);
        let _ = std::fs::remove_file(p);
    }

    /// Anything the C would not recognize as Qwen must read as the `DeepSeek`
    /// fallthrough, which is the same thing `config_validate_model` does.
    #[test]
    fn unknown_architectures_fall_through_to_ds4() {
        for arch in ["deepseek4", "glm-dsa", "glm5-next", "llama", ""] {
            let p = Gguf::default()
                .str_val("general.architecture", arch)
                .write("other");
            assert_eq!(family_of(&p), ModelFamily::Ds4, "{arch}");
            let _ = std::fs::remove_file(p);
        }
    }

    /// Only the pinned Vision-Exp checkpoint may be opened with a vision
    /// encoder; the C refuses the open for any other `DeepSeek` GGUF. A
    /// language-only or re-quantized checkpoint has no
    /// `deepseek4.checkpoint_variant` key at all.
    #[test]
    fn only_the_vision_exp_checkpoint_supports_vision() {
        let vision = Gguf::default()
            .str_val("general.architecture", "deepseek4")
            .str_val("deepseek4.checkpoint_variant", "vision-exp")
            .write("vision-exp");
        assert!(supports_vision(&vision));
        let _ = std::fs::remove_file(vision);

        let plain = Gguf::default()
            .str_val("general.architecture", "deepseek4")
            .u32_val("deepseek4.block_count", 47)
            .write("plain-ds4");
        assert!(!supports_vision(&plain));
        let _ = std::fs::remove_file(plain);

        let other = Gguf::default()
            .str_val("deepseek4.checkpoint_variant", "something-else")
            .write("other-variant");
        assert!(!supports_vision(&other));
        let _ = std::fs::remove_file(other);

        assert!(!supports_vision(Path::new("/nonexistent/x.gguf")));
    }

    /// A probe is run on whatever path the user passed, so it has to survive
    /// files that are not models at all rather than panic or hang.
    #[test]
    fn junk_and_missing_files_are_not_qwen() {
        assert_eq!(architecture(Path::new("/nonexistent/x.gguf")), None);
        assert_eq!(
            family_of(Path::new("/nonexistent/x.gguf")),
            ModelFamily::Ds4
        );

        let junk = std::env::temp_dir().join(format!("plank-gguf-{}-junk", std::process::id()));
        std::fs::write(&junk, b"not a gguf file at all").unwrap();
        assert_eq!(architecture(&junk), None);
        let _ = std::fs::remove_file(junk);

        // Right magic, truncated before the metadata.
        let cut = std::env::temp_dir().join(format!("plank-gguf-{}-cut", std::process::id()));
        std::fs::write(&cut, b"GGUF\x03\x00\x00\x00").unwrap();
        assert_eq!(architecture(&cut), None);
        let _ = std::fs::remove_file(cut);
    }

    /// A hostile or corrupt file must not turn a 64-bit length into an
    /// allocation.
    #[test]
    fn an_absurd_string_length_is_refused() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // key length
        let p = std::env::temp_dir().join(format!("plank-gguf-{}-huge", std::process::id()));
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(architecture(&p), None);
        let _ = std::fs::remove_file(p);
    }
}
