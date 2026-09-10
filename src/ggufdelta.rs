//! Loading a `.ggd` weight delta as a model.
//!
//! The format lives in the `gguf-delta` crate (`crates/gguf-delta`). This
//! module is plank's side of it: finding the base the delta was cut from,
//! turning base plus delta into a file the engine can map, and the two
//! `plank --gguf-delta-*` subcommands.
//!
//! The engine cannot apply a delta in memory: `model_open` in `refs/ds4/ds4.c`
//! maps the model read-only and, on Metal, `MAP_SHARED`, then wraps slices of
//! that mapping as no-copy `MTLBuffer`s. So [`resolve`] APFS-clones the base
//! into `~/.plank/models/patched/` with `clonefile(2)`, writes the changed
//! spans into the clone, and hands back the clone's path; `main.rs` swaps it
//! in for the configured model path before anything reads a model header, so
//! the rest of startup sees an ordinary GGUF. The clone shares every
//! untouched block with the base and costs only the diverged blocks on disk.
//! Neither the base nor the delta is ever opened for writing.
//!
//! The base is found by following the link recorded in the delta: its
//! `base_path` as given at creation (a relative one, the default when the two
//! files were created side by side, resolves against the `.ggd`'s own
//! directory), then a same-named file beside the delta, then
//! `~/.plank/models/<name>`, then the default model path. A candidate counts
//! only if its size and header hash match.
//!
//! KV caches are deliberately shared between base and derived weights: the
//! engine reports the same shape name for both and nothing here changes it.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use gguf_delta::{CreateOptions, Header, check_base, read_chunks, read_header, write_delta};

pub use gguf_delta::is_delta_path;

/// A delta turned into loadable weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// The patched clone to load.
    pub path: PathBuf,
    /// The base that was cloned.
    pub base: PathBuf,
    pub label: String,
    /// First 12 hex digits of `sha256` over the `.ggd` file.
    pub id: String,
}

impl Resolved {
    /// The parenthetical the startup line appends after the model name.
    #[must_use]
    pub fn describe(&self) -> String {
        let base = self.base.file_name().map_or_else(
            || self.base.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        format!("delta {}-{} on {base}", self.label, self.id)
    }
}

/// Where patched clones live.
#[must_use]
pub fn patched_dir() -> PathBuf {
    crate::manifest::plank_dir().join("models").join("patched")
}

/// Finds the base model `h` describes.
///
/// Candidates in order: `base_path` as recorded (a relative one against the
/// delta's directory), a file named `base_name` beside the delta, then each
/// of `extra`. The first whose size and header hash match wins.
///
/// # Errors
/// Lists every candidate tried and why it was rejected.
pub fn find_base(h: &Header, delta: &Path, extra: &[PathBuf]) -> Result<PathBuf, String> {
    let dir = delta
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let mut candidates: Vec<PathBuf> = Vec::new();
    if !h.base_path.as_os_str().is_empty() {
        if h.base_path.is_absolute() {
            candidates.push(h.base_path.clone());
        } else {
            candidates.push(dir.join(&h.base_path));
        }
    }
    if !h.base_name.is_empty() {
        candidates.push(dir.join(&h.base_name));
    }
    candidates.extend(extra.iter().cloned());
    candidates.dedup();
    let mut reasons = Vec::new();
    for c in &candidates {
        match check_base(h, c) {
            Ok(()) => return Ok(c.clone()),
            Err(why) => reasons.push(format!("  {}: {why}", c.display())),
        }
    }
    Err(format!(
        "cannot find the base model for {} ({}):\n{}",
        delta.display(),
        h.base_name,
        reasons.join("\n")
    ))
}

/// The default places a base may have moved to, beyond the delta's own dir.
fn default_base_candidates(h: &Header) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if !h.base_name.is_empty() {
        v.push(
            crate::manifest::plank_dir()
                .join("models")
                .join(&h.base_name),
        );
    }
    v.push(crate::download::default_model_path());
    v
}

/// `clonefile(2)` `src` to `dst`; where that is impossible (another volume,
/// a non-APFS filesystem, a non-macOS host) fall back to a full copy after
/// saying so, because copying an 87 GB model is not something to do silently.
fn clone_or_copy(src: &Path, dst: &Path) -> io::Result<()> {
    match clonefile(src, dst) {
        Ok(()) => Ok(()),
        Err(e)
            if matches!(
                e.raw_os_error(),
                Some(libc::EXDEV | libc::ENOTSUP | libc::EINVAL)
            ) =>
        {
            copy_with_warning(src, dst)
        }
        Err(e) => Err(e),
    }
}

#[allow(clippy::cast_precision_loss)]
fn copy_with_warning(src: &Path, dst: &Path) -> io::Result<()> {
    let need = fs::metadata(src)?.len();
    if let Some(dir) = dst.parent()
        && let Some(free) = free_bytes(dir)
        && free < need
    {
        return Err(io::Error::other(format!(
            "{} cannot be cloned onto this volume and a full copy needs {need} bytes but only {free} are free",
            src.display()
        )));
    }
    eprintln!(
        "plank: {} is not on the same APFS volume as {}; copying the whole file ({:.1} GiB)",
        src.display(),
        dst.display(),
        need as f64 / (1u64 << 30) as f64
    );
    fs::copy(src, dst).map(|_| ())
}

#[cfg(target_os = "macos")]
fn clonefile(src: &Path, dst: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let s =
        CString::new(src.as_os_str().as_bytes()).map_err(|_| io::Error::other("NUL in path"))?;
    let d =
        CString::new(dst.as_os_str().as_bytes()).map_err(|_| io::Error::other("NUL in path"))?;
    // SAFETY: both pointers are valid NUL-terminated C strings for the call.
    let rc = unsafe { libc::clonefile(s.as_ptr(), d.as_ptr(), 0) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "macos"))]
fn clonefile(_src: &Path, _dst: &Path) -> io::Result<()> {
    Err(io::Error::from_raw_os_error(libc::ENOTSUP))
}

fn free_bytes(dir: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c = CString::new(dir.as_os_str().as_bytes()).ok()?;
    // SAFETY: `st` is a plain struct the call fills in; `c` outlives the call.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &raw mut st) };
    (rc == 0).then(|| u64::from(st.f_bavail) * st.f_frsize)
}

fn sidecar_path(clone: &Path) -> PathBuf {
    let mut s = clone.as_os_str().to_owned();
    s.push(".json");
    PathBuf::from(s)
}

/// Whether the clone at `path` was built from the delta with `delta_sha`.
fn cached_clone_is_current(path: &Path, h: &Header, delta_sha: &str) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    if meta.len() != h.base_size {
        return false;
    }
    let Ok(text) = fs::read_to_string(sidecar_path(path)) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    v.get("delta_sha256").and_then(|s| s.as_str()) == Some(delta_sha)
}

/// Turns the `.ggd` at `delta` into loadable weights under [`patched_dir`].
///
/// # Errors
/// A malformed delta, a base that cannot be found or does not match, or a
/// clone that fails to materialize. Nothing partial is left behind.
pub fn resolve(delta: &Path) -> Result<Resolved, String> {
    let (h, _) = read_header(delta).map_err(|e| e.to_string())?;
    let extra = default_base_candidates(&h);
    resolve_with(delta, &patched_dir(), &extra, &clone_or_copy)
}

/// [`resolve`] with the cache directory, the fallback base locations and the
/// clone primitive injected.
///
/// # Errors
/// As [`resolve`].
pub fn resolve_with(
    delta: &Path,
    patched: &Path,
    extra: &[PathBuf],
    cloner: &dyn Fn(&Path, &Path) -> io::Result<()>,
) -> Result<Resolved, String> {
    let (h, first) = read_header(delta).map_err(|e| e.to_string())?;
    let base = find_base(&h, delta, extra)?;
    let delta_sha = gguf_delta::file_sha256_hex(delta).map_err(|e| e.to_string())?;
    let id = delta_sha[..12].to_owned();
    let base_stem = base
        .file_stem()
        .map_or_else(|| "model".to_owned(), |s| s.to_string_lossy().into_owned());
    let clone = patched.join(format!("{base_stem}+{}-{id}.gguf", h.label));
    let resolved = Resolved {
        path: clone.clone(),
        base: base.clone(),
        label: h.label.clone(),
        id,
    };
    if cached_clone_is_current(&clone, &h, &delta_sha) {
        return Ok(resolved);
    }
    fs::create_dir_all(patched).map_err(|e| format!("cannot create {}: {e}", patched.display()))?;
    let part = clone.with_extension("gguf.part");
    let _ = fs::remove_file(&part);
    let built = (|| -> Result<(), String> {
        cloner(&base, &part).map_err(|e| format!("cannot clone {}: {e}", base.display()))?;
        gguf_delta::apply(delta, &h, first, &part).map_err(|e| e.to_string())?;
        let sidecar = serde_json::json!({
            "base": base.display().to_string(),
            "base_size": h.base_size,
            "delta_path": delta.display().to_string(),
            "delta_sha256": delta_sha,
            "target_sha256": gguf_delta::hex(&h.target_sha256),
            "label": h.label,
            "created": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        });
        fs::write(sidecar_path(&clone), sidecar.to_string())
            .map_err(|e| format!("cannot write sidecar for {}: {e}", clone.display()))?;
        fs::rename(&part, &clone).map_err(|e| format!("cannot rename {}: {e}", part.display()))?;
        Ok(())
    })();
    if let Err(e) = built {
        let _ = fs::remove_file(&part);
        let _ = fs::remove_file(sidecar_path(&clone));
        return Err(e);
    }
    Ok(resolved)
}

/// `plank --gguf-delta-create <base> <target> <out.ggd> [--label NAME] [--hash-base]`.
///
/// Returns the process exit code.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn run_create(args: &[String]) -> i32 {
    const USAGE: &str = "usage: plank --gguf-delta-create <base.gguf> <target.gguf> <out.ggd> [--label NAME] [--hash-base]";
    let mut positional: Vec<&String> = Vec::new();
    let mut opts = CreateOptions::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--label" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("plank: --label needs a value");
                    return 2;
                };
                opts.label = Some(v.clone());
            }
            "--hash-base" => opts.hash_base = true,
            a if a.starts_with("--") => {
                eprintln!("plank: unknown option {a}\n{USAGE}");
                return 2;
            }
            _ => positional.push(&args[i]),
        }
        i += 1;
    }
    let [base, target, out] = positional[..] else {
        eprintln!("{USAGE}");
        return 2;
    };
    let started = std::time::Instant::now();
    match write_delta(Path::new(base), Path::new(target), Path::new(out), &opts) {
        Ok(r) => {
            let ratio = if r.bytes_spanned == 0 {
                0.0
            } else {
                r.payload_bytes as f64 / r.bytes_spanned as f64 * 100.0
            };
            println!(
                "wrote {out}: label {}, {} tensors changed in {} chunks, {} of {} spanned bytes differ, {} bytes compressed ({ratio:.1}% of span), {:.1}s",
                r.label,
                r.tensors_changed,
                r.chunks,
                r.bytes_changed,
                r.bytes_spanned,
                r.payload_bytes,
                started.elapsed().as_secs_f64()
            );
            0
        }
        Err(e) => {
            eprintln!("plank: {e}");
            1
        }
    }
}

/// `plank --gguf-delta-info <file.ggd>`.
///
/// Returns the process exit code.
#[must_use]
pub fn run_info(args: &[String]) -> i32 {
    let Some(path) = args.get(1) else {
        eprintln!("usage: plank --gguf-delta-info <file.ggd>");
        return 2;
    };
    let delta = Path::new(path);
    let (h, first) = match read_header(delta) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("plank: {e}");
            return 1;
        }
    };
    let chunks = match read_chunks(delta, &h, first) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("plank: {e}");
            return 1;
        }
    };
    let base = find_base(&h, delta, &default_base_candidates(&h));
    println!("GGUF weight delta {}", delta.display());
    let layout = base
        .as_ref()
        .ok()
        .and_then(|b| gguf_delta::gguf::layout(b).ok());
    print!("{}", gguf_delta::describe(&h, &chunks, layout.as_ref()));
    match base {
        Ok(b) => println!("  base resolves:  {}", b.display()),
        Err(why) => println!("  base resolves:  no\n{why}"),
    }
    0
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use gguf_delta::testing::Gguf;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("plank-ggd-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn base_builder() -> Gguf {
        Gguf::default()
            .str_val("general.architecture", "deepseek4")
            .str_val("general.name", "Tiny Test Model")
            .tensor("blk.0.a", &[64], 0, &[7u8; 64])
            .tensor("blk.1.b", &[4096], 0, &[3u8; 4096])
            .tensor("blk.2.c", &[100], 0, &[9u8; 100])
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, bytes).unwrap();
        p
    }

    fn cloner(src: &Path, dst: &Path) -> io::Result<()> {
        fs::copy(src, dst).map(|_| ())
    }

    /// Base and a target differing in the last tensor, plus the delta.
    fn fixture(d: &Path) -> (PathBuf, Vec<u8>, PathBuf) {
        let base_bytes = base_builder().bytes();
        let base = write(d, "tiny.gguf", &base_bytes);
        let layout = gguf_delta::gguf::layout(&base).unwrap();
        let c = &layout.tensors[2];
        let mut target = base_bytes;
        target[c.offset as usize..(c.offset + c.len) as usize].fill(42);
        let t = write(d, "tiny-Abliterated.gguf", &target);
        let out = d.join("tiny.ggd");
        write_delta(&base, &t, &out, &CreateOptions::default()).unwrap();
        (base, target, out)
    }

    #[test]
    fn resolve_clones_patches_and_caches() {
        let d = dir("resolve");
        let (base, target, out) = fixture(&d);
        let base_bytes = fs::read(&base).unwrap();
        let patched = d.join("patched");
        let calls = std::cell::Cell::new(0);
        let counting = |s: &Path, dst: &Path| {
            calls.set(calls.get() + 1);
            cloner(s, dst)
        };
        let r = resolve_with(&out, &patched, &[], &counting).unwrap();
        assert_eq!(fs::read(&r.path).unwrap(), target);
        assert_eq!(fs::read(&base).unwrap(), base_bytes, "base untouched");
        assert_eq!(r.base, base);
        assert_eq!(r.label, "abliterated");
        assert_eq!(r.id.len(), 12);
        assert!(r.path.starts_with(&patched));
        assert!(r.describe().starts_with("delta abliterated-"));
        assert!(r.describe().ends_with("on tiny.gguf"));
        assert!(
            fs::read_to_string(sidecar_path(&r.path))
                .unwrap()
                .contains("delta_sha256")
        );
        assert_eq!(calls.get(), 1);

        let again = resolve_with(&out, &patched, &[], &counting).unwrap();
        assert_eq!(again, r);
        assert_eq!(calls.get(), 1, "second resolve reused the clone");

        // A sidecar naming another delta forces a rebuild.
        fs::write(sidecar_path(&r.path), r#"{"delta_sha256":"0000"}"#).unwrap();
        let rebuilt = resolve_with(&out, &patched, &[], &counting).unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(fs::read(rebuilt.path).unwrap(), target);
        assert!(
            fs::read_dir(&patched).unwrap().all(|e| !e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".part"))
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_wrong_base_fails_and_leaves_nothing_behind() {
        let d = dir("wrongbase");
        let (base, _target, out) = fixture(&d);
        // Same size and header, different tensor bytes inside the chunk.
        let mut bad = fs::read(&base).unwrap();
        let layout = gguf_delta::gguf::layout(&base).unwrap();
        bad[(layout.tensors[2].offset + 3) as usize] ^= 0x80;
        fs::write(&base, &bad).unwrap();
        let patched = d.join("patched");
        let e = resolve_with(&out, &patched, &[], &cloner).unwrap_err();
        assert!(e.contains("does not hold the bytes"), "{e}");
        assert!(
            fs::read_dir(&patched).unwrap().next().is_none(),
            "nothing left behind"
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn find_base_follows_the_link_then_siblings_then_extras() {
        let d = dir("find");
        let elsewhere = dir("find-elsewhere");
        let base_bytes = base_builder().bytes();
        let base = write(&d, "b.gguf", &base_bytes);
        let target = write(&d, "t.gguf", &base_bytes);
        let out = elsewhere.join("d.ggd");
        write_delta(&base, &target, &out, &CreateOptions::default()).unwrap();
        let (h, _) = read_header(&out).unwrap();
        assert_eq!(h.base_path, base, "different dir: the path as given");
        // 1. The recorded link.
        assert_eq!(find_base(&h, &out, &[]).unwrap(), base);
        // 2. Link broken, sibling by name.
        fs::rename(&base, elsewhere.join("b.gguf")).unwrap();
        assert_eq!(find_base(&h, &out, &[]).unwrap(), elsewhere.join("b.gguf"));
        // 3. Neither; an extra candidate. A wrong-size sibling is skipped
        //    with a reason, and the missing link is named too.
        fs::rename(elsewhere.join("b.gguf"), d.join("moved.gguf")).unwrap();
        write(&elsewhere, "b.gguf", b"short");
        let e = find_base(&h, &out, &[]).unwrap_err();
        assert!(e.contains("wrong size"), "{e}");
        assert!(e.contains("missing"), "{e}");
        assert_eq!(
            find_base(&h, &out, &[d.join("moved.gguf")]).unwrap(),
            d.join("moved.gguf")
        );
        // A relative link resolves against the delta's directory.
        let mut rel = h.clone();
        rel.base_path = Path::new("..")
            .join(d.file_name().unwrap())
            .join("moved.gguf");
        assert_eq!(
            find_base(&rel, &out, &[]).unwrap(),
            elsewhere.join(&rel.base_path)
        );
        let _ = fs::remove_dir_all(&d);
        let _ = fs::remove_dir_all(&elsewhere);
    }

    #[test]
    fn a_delta_beside_its_base_records_only_the_filename() {
        let d = dir("beside");
        let (base, _target, out) = fixture(&d);
        let (h, _) = read_header(&out).unwrap();
        assert_eq!(h.base_path, PathBuf::from("tiny.gguf"));
        // Move the pair together: the link still resolves.
        let moved = dir("beside-moved");
        fs::rename(&base, moved.join("tiny.gguf")).unwrap();
        fs::rename(&out, moved.join("tiny.ggd")).unwrap();
        assert_eq!(
            find_base(&h, &moved.join("tiny.ggd"), &[]).unwrap(),
            moved.join("tiny.gguf")
        );
        let _ = fs::remove_dir_all(&d);
        let _ = fs::remove_dir_all(&moved);
    }

    #[test]
    fn clone_falls_back_to_copy_on_exdev() {
        let d = dir("exdev");
        let src = write(&d, "s.bin", b"hello world");
        let dst = d.join("dst.bin");
        let exdev = |_s: &Path, _d: &Path| -> io::Result<()> {
            Err(io::Error::from_raw_os_error(libc::EXDEV))
        };
        // The production fallback path, driven with the failing primitive.
        let r = match exdev(&src, &dst) {
            Err(e) if e.raw_os_error() == Some(libc::EXDEV) => copy_with_warning(&src, &dst),
            other => other,
        };
        r.unwrap();
        assert_eq!(fs::read(dst).unwrap(), b"hello world");
        // And the real primitive on this filesystem, whichever branch it takes.
        clone_or_copy(&src, &d.join("c.bin")).unwrap();
        assert_eq!(fs::read(d.join("c.bin")).unwrap(), b"hello world");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn the_subcommands_report_usage_and_run() {
        let d = dir("cli");
        let (base, _target, out) = fixture(&d);
        let s = |x: &str| x.to_owned();
        assert_eq!(run_create(&[s("--gguf-delta-create")]), 2);
        assert_eq!(
            run_create(&[
                s("--gguf-delta-create"),
                s("a"),
                s("b"),
                s("c"),
                s("--label")
            ]),
            2
        );
        assert_eq!(
            run_create(&[
                s("--gguf-delta-create"),
                s("a"),
                s("b"),
                s("c"),
                s("--bogus")
            ]),
            2
        );
        assert_eq!(
            run_create(&[
                s("--gguf-delta-create"),
                s("/nope"),
                s("/nope2"),
                s("/tmp/x.ggd")
            ]),
            1
        );
        let out2 = d.join("cli.ggd");
        assert_eq!(
            run_create(&[
                s("--gguf-delta-create"),
                base.display().to_string(),
                d.join("tiny-Abliterated.gguf").display().to_string(),
                out2.display().to_string(),
                s("--label"),
                s("cli"),
            ]),
            0
        );
        assert_eq!(read_header(&out2).unwrap().0.label, "cli");
        assert_eq!(run_info(&[s("--gguf-delta-info")]), 2);
        assert_eq!(run_info(&[s("--gguf-delta-info"), s("/nope.ggd")]), 1);
        assert_eq!(
            run_info(&[s("--gguf-delta-info"), out.display().to_string()]),
            0
        );
        let _ = fs::remove_dir_all(&d);
    }
}
