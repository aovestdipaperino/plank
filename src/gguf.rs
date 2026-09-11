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

/// One detail line about a file the engine open depended on: whether it is
/// there, what a symlink points at, how big it is, and whether this process
/// can actually read it.
///
/// `None` when `path` is `None`, so a companion plank deliberately never
/// passed is not reported as a file that is missing — the same distinction
/// `report_text_only` makes for the vision encoder.
#[must_use]
pub fn file_detail(label: &str, path: Option<&Path>) -> Option<String> {
    use std::fmt::Write as _;
    let path = path?;
    let mut line = format!("- {label}: {}", path.display());
    // `symlink_metadata` first: plank's own default model paths are symlinks
    // by convention (`~/.plank/qwen.gguf` and its sidecar are expected to
    // point at whichever build you keep), so a dangling one is the single most
    // likely cause of an open that fails with the path looking perfectly fine.
    match std::fs::symlink_metadata(path) {
        Err(e) => {
            let _ = write!(line, " — not found ({e})");
            return Some(line);
        }
        Ok(md) if md.file_type().is_symlink() => {
            match std::fs::read_link(path) {
                Ok(target) => {
                    let _ = write!(line, " → {}", target.display());
                }
                Err(e) => {
                    let _ = write!(line, " — symlink unreadable ({e})");
                    return Some(line);
                }
            }
            if !path.exists() {
                line.push_str(" — DANGLING: the symlink target does not exist");
                return Some(line);
            }
        }
        Ok(_) => {}
    }
    match std::fs::metadata(path) {
        Ok(md) if md.is_dir() => line.push_str(" — is a directory, not a file"),
        Ok(md) => {
            let _ = write!(line, ", {}", crate::kvpane::human_bytes(md.len()));
            // Size says nothing about permissions, and an artifact copied in
            // as root is a real way to get here.
            if let Err(e) = File::open(path) {
                let _ = write!(line, " — cannot read it ({e})");
            }
        }
        Err(e) => {
            let _ = write!(line, " — cannot stat ({e})");
        }
    }
    Some(line)
}

/// The line comparing an artifact against the length the installed manifest
/// records for it, when `path` is one of the managed artifacts and the two
/// disagree.
///
/// The decisive check for the failure mode a plain "failed to open" hides
/// worst: a truncated weights file. An interrupted install, a full disk, or a
/// half-copied file opens as a perfectly ordinary path of the right name, and
/// the engine's own complaint about it is a parse error deep in a tensor
/// table. plank already knows the expected byte count, so it can say so.
#[must_use]
pub fn artifact_size_mismatch(path: &Path, family: ModelFamily) -> Option<String> {
    let on_disk = std::fs::metadata(path).ok()?.len();
    let set = crate::manifest::ModelSet::for_family(family);
    let manifest = crate::manifest::read_at(&crate::manifest::installed_path(set))?;
    let same = |a: &Path, b: &Path| {
        a == b
            || match (a.canonicalize(), b.canonicalize()) {
                (Ok(a), Ok(b)) => a == b,
                _ => false,
            }
    };
    let (_, entry) = manifest.files.iter().find(|(kind, _)| {
        crate::manifest::local_path_for(set, kind).is_some_and(|p| same(&p, path))
    })?;
    (entry.bytes != on_disk).then(|| {
        format!(
            "- SIZE MISMATCH: the installed {} manifest records {} for this artifact, but the file is {} — an interrupted or truncated install",
            set.as_str(),
            crate::kvpane::human_bytes(entry.bytes),
            crate::kvpane::human_bytes(on_disk)
        )
    })
}

/// What plank knew when it asked the engine to open a model, for
/// [`open_failure_detail`]. A struct because the answer draws on eight
/// unrelated facts and a positional call of that width is a bug waiting for a
/// refactor to swap two of them.
#[derive(Debug)]
pub struct OpenAttempt<'a> {
    /// The model file handed to the engine.
    pub path: &'a Path,
    /// What `ds4_engine_open` returned.
    pub rc: i32,
    /// Whether it left the out-pointer null.
    pub engine_null: bool,
    /// Family plank detected from the file's own metadata before opening.
    pub family: ModelFamily,
    /// Backend label, already formatted by the caller so this module needs no
    /// FFI type.
    pub backend: &'a str,
    /// Context window requested, in tokens.
    pub ctx_size: i32,
    /// Companion files actually passed, label first. `None` for one plank
    /// deliberately did not pass.
    pub companions: &'a [(&'a str, Option<&'a Path>)],
    /// Whether the Metal kernel sources were absent where the engine looks.
    pub metal_kernels_missing: bool,
}

/// The multi-line body of a "failed to open model" error: every fact plank can
/// establish about why, on its own, without the engine.
///
/// A bare "failed to open model &lt;path&gt;" is the least useful form of a
/// failure that has a handful of cheap and decisive causes: a dangling symlink,
/// a truncated artifact from an interrupted install, a companion sidecar that
/// is absent (a Qwen run cannot open without its PLE file, and `--mtp` on a
/// `DeepSeek` run cannot without the draft checkpoint), a context size the
/// machine cannot hold, or the Metal kernel sources not being where the engine
/// looks. Each gets its own line, and only when it is true, so the message
/// never pads itself out with reassurance that everything is fine.
///
/// Lives here rather than beside the open it describes because everything in
/// it is FFI-free and so stays CI-tested, the same split `ds4tokens` makes:
/// the gated engine wrapper passes its backend as a label and its return code
/// as a number.
#[must_use]
pub fn open_failure_detail(attempt: &OpenAttempt) -> String {
    use std::fmt::Write as _;
    let OpenAttempt {
        path,
        rc,
        engine_null,
        family,
        backend,
        ctx_size,
        companions,
        metal_kernels_missing,
    } = *attempt;
    let mut msg = format!("failed to open model {}", path.display());
    // The two ways the C reports a refusal are worth telling apart: a code is
    // something it decided, a null engine with a zero code is a bug in the
    // glue or an allocation that failed without saying so.
    if rc != 0 {
        let _ = write!(
            msg,
            "
- ds4_engine_open returned {rc}"
        );
    } else if engine_null {
        msg.push_str(
            "
- ds4_engine_open reported success but returned no engine",
        );
    }
    let _ = write!(
        msg,
        "
- opened as: {} family, {backend} backend, context {ctx_size} tokens",
        crate::manifest::ModelSet::for_family(family).as_str()
    );
    if let Some(line) = file_detail("model file", Some(path)) {
        let _ = write!(
            msg,
            "
{line}"
        );
    }
    if let Some(line) = artifact_size_mismatch(path, family) {
        let _ = write!(
            msg,
            "
{line}"
        );
    }
    for (label, companion) in companions {
        if let Some(line) = file_detail(label, *companion) {
            let _ = write!(
                msg,
                "
{line}"
            );
        }
    }
    if metal_kernels_missing {
        msg.push_str(
            "
- Metal kernel sources not found; set DS4_METAL_DIR to a directory \
             containing the .metal files",
        );
    }
    // The C logs its own diagnosis on the way out, and plank renders that log
    // in place on a single row, so the line that actually names the cause is
    // usually the last thing above this message — and easy to read past.
    msg.push_str(
        "
- the engine's own log is on the lines above this one",
    );
    msg
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

    /// A path for one test's fixtures, in the shape the tests above already
    /// use: process-scoped so parallel runs cannot collide.
    fn detail_tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("plank-detail-{}-{name}", std::process::id()))
    }

    /// The file line reports what is wrong: absent, a dangling symlink, a
    /// directory, or present with its size.
    #[test]
    fn the_file_detail_line_names_what_is_wrong() {
        // Never passed is not the same as missing, and says nothing.
        assert_eq!(file_detail("vision encoder", None), None);

        let missing = detail_tmp("missing.gguf");
        let _ = std::fs::remove_file(&missing);
        let line = file_detail("model file", Some(&missing)).expect("a path was given");
        assert!(line.starts_with("- model file: "), "{line}");
        assert!(line.contains("not found"), "{line}");

        // A real file reports its size.
        let real = detail_tmp("real.gguf");
        std::fs::write(&real, vec![7u8; 2048]).expect("write");
        let line = file_detail("model file", Some(&real)).expect("a path was given");
        assert!(line.contains("2.0 KB"), "{line}");
        assert!(!line.contains("not found"), "{line}");

        // The case plank's own symlinked default paths make likely.
        let dangling = detail_tmp("dangling.gguf");
        let _ = std::fs::remove_file(&dangling);
        std::os::unix::fs::symlink(detail_tmp("nowhere.gguf"), &dangling).expect("symlink");
        let line = file_detail("model file", Some(&dangling)).expect("a path was given");
        assert!(line.contains("DANGLING"), "{line}");
        assert!(line.contains("nowhere.gguf"), "the target is named: {line}");

        // A symlink that resolves names its target and still sizes it.
        let good_link = detail_tmp("good.gguf");
        let _ = std::fs::remove_file(&good_link);
        std::os::unix::fs::symlink(&real, &good_link).expect("symlink");
        let line = file_detail("model file", Some(&good_link)).expect("a path was given");
        assert!(line.contains("real.gguf"), "{line}");
        assert!(line.contains("2.0 KB"), "{line}");
        assert!(!line.contains("DANGLING"), "{line}");

        let dir = detail_tmp("dir.gguf");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let line = file_detail("model file", Some(&dir)).expect("a path was given");
        assert!(line.contains("is a directory"), "{line}");

        let _ = std::fs::remove_file(&real);
        let _ = std::fs::remove_file(&dangling);
        let _ = std::fs::remove_file(&good_link);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The assembled message: the return code, what was opened, the file
    /// itself, only the companions that were passed, and the hints — and no
    /// line for anything that is fine.
    #[test]
    fn the_open_failure_message_names_every_fact_it_has() {
        let model = detail_tmp("attempt.gguf");
        std::fs::write(&model, vec![0u8; 4096]).expect("write");
        let ple = detail_tmp("attempt.ple.gguf");
        let _ = std::fs::remove_file(&ple);
        let msg = open_failure_detail(&OpenAttempt {
            path: &model,
            rc: -3,
            engine_null: true,
            family: ModelFamily::Qwen,
            backend: "Metal",
            ctx_size: 1_048_576,
            companions: &[("ple sidecar", Some(&ple)), ("vision encoder", None)],
            metal_kernels_missing: true,
        });
        assert!(msg.starts_with("failed to open model "), "{msg}");
        assert!(msg.contains("returned -3"), "{msg}");
        assert!(
            msg.contains("qwen family, Metal backend, context 1048576 tokens"),
            "{msg}"
        );
        assert!(msg.contains("- model file: "), "{msg}");
        assert!(msg.contains("4.0 KB"), "{msg}");
        // The companion that was passed and is absent gets a line; the one
        // plank never passed gets none.
        assert!(msg.contains("- ple sidecar: "), "{msg}");
        assert!(!msg.contains("vision encoder"), "{msg}");
        assert!(msg.contains("DS4_METAL_DIR"), "{msg}");
        assert!(msg.contains("engine's own log"), "{msg}");

        // A non-zero code and a null engine are different news, and only one
        // of them is reported per failure.
        let msg = open_failure_detail(&OpenAttempt {
            path: &model,
            rc: 0,
            engine_null: true,
            family: ModelFamily::Ds4,
            backend: "Cpu",
            ctx_size: 8192,
            companions: &[],
            metal_kernels_missing: false,
        });
        assert!(msg.contains("success but returned no engine"), "{msg}");
        assert!(!msg.contains("returned 0"), "{msg}");
        assert!(!msg.contains("DS4_METAL_DIR"), "{msg}");
        let _ = std::fs::remove_file(&model);
    }
}
