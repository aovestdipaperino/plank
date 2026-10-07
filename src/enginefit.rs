//! Which catalog engines this machine can run, and the reason when one cannot.
//!
//! Pure: RAM, free disk, the build's engines and file existence are all passed
//! in, so every rule is tested without a machine. The menu (`enginepick`) and
//! the launch gate (`require_min_ram` in `main.rs`) read the same constants, so
//! the menu never offers a model the load would then refuse.

use std::path::{Path, PathBuf};

use crate::engines::{Catalog, Choice, EngineEntry};

/// One binary gigabyte.
pub const GIB: u64 = 1024 * 1024 * 1024;

/// Minimum physical RAM for an engine served by the C engine (`ds4`, `ds41`,
/// `qwen`). Above it SSD streaming covers any model size, so this floor is the
/// whole of "runs at all" for those families.
pub const MIN_RAM_BYTES: u64 = 96 * GIB;

/// RAM a Gemma engine needs beyond its weights when its catalog entry does
/// not say: about 114 KB of f32 KV per token (E4B) at the engine's default
/// context, rounded up. An entry's `kvBytesPerToken` replaces it.
pub const GEMMA_KV_RESERVE_BYTES: u64 = 4 * GIB;

/// The context a Gemma engine opens with when no `-c` is given:
/// `min(32768, context_length)` (`docs/GEMMA.md`). Both catalog models allow
/// more, so the reserve is sized at the cap.
pub const GEMMA_DEFAULT_CTX: u64 = 32_768;

/// Free space a download must leave behind, so the disk is not filled to the
/// last byte by a model.
pub const DISK_MARGIN_BYTES: u64 = GIB;

/// The engines this binary was built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildCaps {
    /// The C engine (`cfg(ds4_engine)`): `DeepSeek` and Qwen.
    pub ds4: bool,
    /// The native Gemma engine (`feature = "gemma"`).
    pub gemma: bool,
}

impl BuildCaps {
    /// What this binary can run.
    #[must_use]
    pub fn current() -> Self {
        Self {
            ds4: cfg!(ds4_engine),
            gemma: cfg!(feature = "gemma"),
        }
    }
}

/// The facts about this machine the rules read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Machine {
    /// Installed RAM, when known.
    pub ram: Option<u64>,
    /// Free bytes on the volume downloads land on, when known.
    pub free_disk: Option<u64>,
    /// The engines this build can run.
    pub caps: BuildCaps,
}

/// This machine, with downloads landing under `root`.
#[must_use]
pub fn machine(root: &Path) -> Machine {
    Machine {
        ram: crate::download::total_ram_bytes(),
        free_disk: free_bytes(root),
        caps: BuildCaps::current(),
    }
}

/// Whether and how an engine can be used here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fit {
    /// Every file it declares is on disk.
    Installed,
    /// It fits; `bytes` is the size of the missing files, `None` when unknown.
    Download { bytes: Option<u64> },
    /// The background downloader is fetching it now, `percent` of the way.
    /// Selectable: picking it waits for the download already under way.
    Downloading { percent: u8 },
    /// Its whole set was downloaded in the background and verified, and is
    /// waiting in staging: picking it installs it with no further download,
    /// so the disk rule does not apply.
    Staged,
    /// It cannot run here, for `reason`.
    Disabled { reason: String },
}

/// One menu row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineRow {
    /// The engine name, what `engine.model` will be set to.
    pub name: String,
    /// The catalog's `notes`, or empty.
    pub notes: String,
    /// Its state on this machine.
    pub fit: Fit,
}

impl EngineRow {
    /// Whether the menu may pick it.
    #[must_use]
    pub fn selectable(&self) -> bool {
        !matches!(self.fit, Fit::Disabled { .. })
    }
}

/// The KV a Gemma engine holds in RAM at its default context: the entry's
/// `kvBytesPerToken` times [`GEMMA_DEFAULT_CTX`], else [`GEMMA_KV_RESERVE_BYTES`].
///
/// Declared per engine because it varies five-fold between the two models
/// (E4B shares KV across 18 layers; 12B keeps 8 heads on 40 sliding layers)
/// and cannot be read from a file that has not been downloaded yet.
fn gemma_kv_reserve(entry: &EngineEntry) -> u64 {
    serde_json::from_str::<serde_json::Value>(&entry.raw)
        .ok()
        .and_then(|v| v["kvBytesPerToken"].as_u64())
        .map_or(GEMMA_KV_RESERVE_BYTES, |b| {
            b.saturating_mul(GEMMA_DEFAULT_CTX)
        })
}

/// A top-level string field of the engine's verbatim JSON.
fn raw_str(entry: &EngineEntry, key: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(&entry.raw)
        .ok()
        .and_then(|v| v[key].as_str().map(str::to_owned))
}

/// One decimal gigabyte: file sizes and disk figures count in powers of 1000.
const GB: u64 = 1_000_000_000;

/// Whole binary gigabytes of a RAM requirement, rounded up.
fn need_gb(bytes: u64) -> u64 {
    bytes.div_ceil(GIB)
}

/// Whole binary gigabytes of the machine's RAM, rounded down, so a machine
/// never reads as large as a requirement it falls short of.
fn have_gb(bytes: u64) -> u64 {
    bytes / GIB
}

/// A download size for display: one decimal in decimal units, or
/// `size unknown`.
#[must_use]
pub fn size_label(bytes: Option<u64>) -> String {
    bytes.map_or_else(|| "size unknown".to_owned(), |b| human_bytes(b, 1000))
}

/// `bytes` with one decimal in the largest unit it reaches, counting in
/// powers of `base` (1024 for binary sizes, 1000 for decimal ones): KB below
/// `base`², MB below `base`³, else GB.
#[must_use]
pub fn human_bytes(bytes: u64, base: u64) -> String {
    let (div, unit) = if bytes < base * base {
        (base, "KB")
    } else if bytes < base * base * base {
        (base * base, "MB")
    } else {
        (base * base * base, "GB")
    };
    #[allow(clippy::cast_precision_loss)]
    let v = bytes as f64 / div as f64;
    format!("{v:.1} {unit}")
}

/// Evaluates every engine in `catalog`, in name order.
///
/// `downloading` names the engine the background downloader is fetching and
/// how far along it is, when one is. `staged` says whether an engine's whole
/// set is verified and waiting in staging.
#[must_use]
pub fn evaluate(
    root: &Path,
    catalog: &Catalog,
    m: &Machine,
    exists: &dyn Fn(&Path) -> bool,
    downloading: Option<(&str, u8)>,
    staged: &dyn Fn(&str) -> bool,
) -> Vec<EngineRow> {
    catalog
        .engines
        .iter()
        .filter_map(|(name, entry)| {
            let sel = crate::engines::resolve_in(root, catalog, Choice::Named(name)).ok()?;
            Some(EngineRow {
                name: name.clone(),
                notes: raw_str(entry, "notes").unwrap_or_default(),
                fit: fit_of(
                    entry,
                    &sel,
                    m,
                    exists,
                    downloading.and_then(|(d, pct)| (d == name.as_str()).then_some(pct)),
                    staged(name),
                ),
            })
        })
        .collect()
}

fn fit_of(
    entry: &EngineEntry,
    sel: &crate::engines::Selection,
    m: &Machine,
    exists: &dyn Fn(&Path) -> bool,
    downloading: Option<u8>,
    staged: bool,
) -> Fit {
    let roles: [(&str, Option<&PathBuf>); 3] = [
        ("main", Some(&sel.main)),
        ("mtp", sel.mtp.as_ref()),
        ("vision", sel.vision.as_ref()),
    ];
    let mut missing = 0u64;
    let mut size_known = true;
    // The main file is what makes an engine installed: a missing companion is
    // fetched at load by `ensure_side_artifacts`, as on the launch path, and
    // never by the helper, whose full manifest would re-fetch the main.
    let installed = exists(&sel.main);
    for (role, path) in roles {
        if installed {
            break;
        }
        let Some(path) = path else { continue };
        if exists(path) {
            continue;
        }
        if let Some(f) = entry.files.get(role) {
            missing += f.bytes;
        } else if sel.urls.contains_key(role) {
            size_known = false;
        } else {
            return Fit::Disabled {
                reason: format!("file missing at {}", path.display()),
            };
        }
    }
    let gemma = raw_str(entry, "family").as_deref() == Some("gemma");
    if !(if gemma { m.caps.gemma } else { m.caps.ds4 }) {
        return Fit::Disabled {
            reason: "not supported by this build".to_owned(),
        };
    }
    if let Some(ram) = m.ram {
        let need = if gemma {
            entry.files.get("main").map(|f| {
                f.bytes.saturating_add(gemma_kv_reserve(entry))
                    / crate::download::SSD_STREAMING_RAM_PERCENT
                    * 100
            })
        } else {
            Some(MIN_RAM_BYTES)
        };
        if let Some(need) = need.filter(|&n| ram < n) {
            return Fit::Disabled {
                reason: format!(
                    "needs {} GB RAM (this machine: {} GB)",
                    need_gb(need),
                    have_gb(ram)
                ),
            };
        }
    }
    if installed {
        return Fit::Installed;
    }
    // Already committed to, with part of its bytes on disk: the disk rule,
    // which counts every missing byte as still to come, no longer applies.
    if let Some(percent) = downloading {
        return Fit::Downloading { percent };
    }
    // Every byte is already on disk, in staging: installing it is a rename.
    if staged {
        return Fit::Staged;
    }
    if let Some(free) = m.free_disk
        && size_known
        && missing + DISK_MARGIN_BYTES > free
    {
        return Fit::Disabled {
            reason: format!(
                "needs {} GB free disk (have {} GB)",
                (missing + DISK_MARGIN_BYTES).div_ceil(GB),
                free / GB
            ),
        };
    }
    Fit::Download {
        bytes: size_known.then_some(missing),
    }
}

/// Free bytes available to this user on the volume holding `path`, or on its
/// nearest existing ancestor (a fresh machine has no `~/.plank` yet).
#[must_use]
pub fn free_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt as _;
    let dir = path.ancestors().find(|p| p.exists())?;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c` is a NUL-terminated path and `st` a valid out-parameter of
    // the right type; it is read only when the call reports success.
    let rc = unsafe { libc::statvfs(c.as_ptr(), st.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: statvfs returned 0, so it filled `st`.
    let st = unsafe { st.assume_init() };
    #[allow(clippy::useless_conversion)]
    let (avail, frsize) = (u64::from(st.f_bavail), u64::from(st.f_frsize));
    Some(avail.saturating_mul(frsize))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::{Layer, parse};
    use std::path::{Path, PathBuf};

    const CATALOG: &str = r#"{
      "version": 2, "default": "ds4vision",
      "engines": {
        "ds4vision": {"version": 1, "notes": "DeepSeek V4 Flash Vision",
          "main": {"name": "m.gguf", "url": "https://h/m", "bytes": 86000000000, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
          "mtp": {"name": "t.gguf", "url": "https://h/t", "bytes": 6000000000, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},
        "gemma4-e4b": {"version": 1, "notes": "Gemma 4 E4B", "family": "gemma",
          "main": {"name": "g.gguf", "url": "https://h/g", "bytes": 4977171584, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}
      }}"#;

    fn catalog() -> Catalog {
        let mut w = Vec::new();
        parse(CATALOG, Layer::Published, &mut w).expect("parses")
    }

    fn machine(ram_gib: Option<u64>, free_gib: Option<u64>) -> Machine {
        Machine {
            ram: ram_gib.map(|g| g * GIB),
            free_disk: free_gib.map(|g| g * GIB),
            caps: BuildCaps {
                ds4: true,
                gemma: true,
            },
        }
    }

    fn row<'a>(rows: &'a [EngineRow], name: &str) -> &'a EngineRow {
        rows.iter().find(|r| r.name == name).expect("row")
    }

    fn none(_: &Path) -> bool {
        false
    }

    fn no_staged(_: &str) -> bool {
        false
    }

    #[test]
    fn a_large_machine_offers_every_engine_as_a_download() {
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(500)),
            &none,
            None,
            &no_staged,
        );
        assert_eq!(
            row(&rows, "ds4vision").fit,
            Fit::Download {
                bytes: Some(92_000_000_000)
            }
        );
        assert_eq!(
            row(&rows, "gemma4-e4b").fit,
            Fit::Download {
                bytes: Some(4_977_171_584)
            }
        );
        assert_eq!(row(&rows, "gemma4-e4b").notes, "Gemma 4 E4B");
    }

    #[test]
    fn deepseek_below_the_ram_floor_is_disabled_and_says_why() {
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(64), Some(500)),
            &none,
            None,
            &no_staged,
        );
        assert_eq!(
            row(&rows, "ds4vision").fit,
            Fit::Disabled {
                reason: "needs 96 GB RAM (this machine: 64 GB)".into()
            }
        );
        assert!(row(&rows, "gemma4-e4b").selectable());
    }

    #[test]
    fn gemma_needs_its_file_and_kv_within_eighty_percent_of_ram() {
        // 4.98 GB + 4 GiB = ~8.6 GiB, needing ~10.8 GiB of RAM at 80%.
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(8), Some(500)),
            &none,
            None,
            &no_staged,
        );
        assert_eq!(
            row(&rows, "gemma4-e4b").fit,
            Fit::Disabled {
                reason: "needs 11 GB RAM (this machine: 8 GB)".into()
            }
        );
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(16), Some(500)),
            &none,
            None,
            &no_staged,
        );
        assert!(row(&rows, "gemma4-e4b").selectable());
    }

    /// Gemma 4 12B keeps ~688 KB of f32 KV per token (40 sliding layers of 8
    /// heads at 512 values, 8 global layers of 1 head at 1024), so at the 32K
    /// default it needs ~22.5 GB of KV on top of its 7 GB of weights: ~37 GB
    /// of RAM at 80%, not the ~14 GB the flat 4 GiB reserve claimed.
    #[test]
    fn a_catalog_kv_cost_sizes_the_gemma_reserve() {
        let mut w = Vec::new();
        let cat = parse(
            &format!(
                r#"{{"version": 2, "engines": {{"gemma4-12b": {{"version": 1, "family": "gemma",
                  "kvBytesPerToken": 688128,
                  "main": {{"name": "g.gguf", "url": "https://h/g", "bytes": 6975879296, "sha256": "{}"}}}}}}}}"#,
                "a".repeat(64)
            ),
            Layer::Published,
            &mut w,
        )
        .expect("parses");
        let at = |ram| {
            evaluate(
                Path::new("/r"),
                &cat,
                &machine(Some(ram), Some(500)),
                &none,
                None,
                &no_staged,
            )
        };
        assert_eq!(
            row(&at(32), "gemma4-12b").fit,
            Fit::Disabled {
                reason: "needs 35 GB RAM (this machine: 32 GB)".into()
            }
        );
        assert!(row(&at(64), "gemma4-12b").selectable());
    }

    #[test]
    fn a_download_that_does_not_fit_the_disk_is_disabled() {
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(50)),
            &none,
            None,
            &no_staged,
        );
        assert_eq!(
            row(&rows, "ds4vision").fit,
            Fit::Disabled {
                reason: "needs 94 GB free disk (have 53 GB)".into()
            }
        );
        assert!(row(&rows, "gemma4-e4b").selectable());
    }

    #[test]
    fn an_installed_engine_ignores_the_disk_rule() {
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(1)),
            &|_| true,
            None,
            &no_staged,
        );
        assert_eq!(row(&rows, "ds4vision").fit, Fit::Installed);
    }

    #[test]
    fn an_engine_whose_main_exists_is_installed_even_without_a_companion() {
        // A missing companion is fetched at load by `ensure_side_artifacts`,
        // exactly as on the launch path; reading it as a download would make
        // the helper re-fetch the whole main into staging.
        let main = PathBuf::from("/r/ds4vision.gguf");
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(500)),
            &|p| p == main,
            None,
            &no_staged,
        );
        assert_eq!(row(&rows, "ds4vision").fit, Fit::Installed);
    }

    #[test]
    fn only_missing_roles_count_towards_the_download() {
        let mtp = PathBuf::from("/r/ds4vision.mtp.gguf");
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(500)),
            &|p| p == mtp,
            None,
            &no_staged,
        );
        assert_eq!(
            row(&rows, "ds4vision").fit,
            Fit::Download {
                bytes: Some(86_000_000_000)
            }
        );
    }

    #[test]
    fn the_machine_ram_rounds_down_while_the_need_rounds_up() {
        // 10.5 GiB against gemma4-e4b's ~10.8 GiB need.
        let mut m = machine(None, Some(500));
        m.ram = Some(10 * GIB + GIB / 2);
        let rows = evaluate(Path::new("/r"), &catalog(), &m, &none, None, &no_staged);
        assert_eq!(
            row(&rows, "gemma4-e4b").fit,
            Fit::Disabled {
                reason: "needs 11 GB RAM (this machine: 10 GB)".into()
            }
        );
    }

    #[test]
    fn unknown_ram_and_disk_never_disable() {
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(None, None),
            &none,
            None,
            &no_staged,
        );
        assert!(rows.iter().all(EngineRow::selectable));
    }

    #[test]
    fn an_engine_this_build_cannot_run_is_disabled_first() {
        let mut m = machine(Some(16), Some(500));
        m.caps = BuildCaps {
            ds4: false,
            gemma: true,
        };
        let rows = evaluate(Path::new("/r"), &catalog(), &m, &none, None, &no_staged);
        assert_eq!(
            row(&rows, "ds4vision").fit,
            Fit::Disabled {
                reason: "not supported by this build".into()
            }
        );
    }

    #[test]
    fn a_local_engine_without_its_file_is_a_download_only_with_a_url() {
        let mut w = Vec::new();
        let local = parse(
            r#"{"engines": {
                 "mine": {"main": {"path": "/x/mine.gguf"}},
                 "fetchable": {"main": {"path": "/x/f.gguf", "url": "https://h/f.gguf"}}}}"#,
            Layer::Local,
            &mut w,
        )
        .expect("parses");
        let cat = crate::engines::layer(catalog(), local);
        let rows = evaluate(
            Path::new("/r"),
            &cat,
            &machine(Some(128), Some(500)),
            &none,
            None,
            &no_staged,
        );
        assert_eq!(
            row(&rows, "mine").fit,
            Fit::Disabled {
                reason: "file missing at /x/mine.gguf".into()
            }
        );
        assert_eq!(row(&rows, "fetchable").fit, Fit::Download { bytes: None });
    }

    #[test]
    fn the_engine_being_downloaded_reads_downloading_and_stays_selectable() {
        // 50 GB free would fail the disk rule, but the download already holds
        // part of its bytes on disk, so the rule no longer applies to it.
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(50)),
            &none,
            Some(("ds4vision", 43)),
            &no_staged,
        );
        assert_eq!(
            row(&rows, "ds4vision").fit,
            Fit::Downloading { percent: 43 }
        );
        assert!(row(&rows, "ds4vision").selectable());
        // Every other engine is evaluated as before.
        assert_eq!(
            row(&rows, "gemma4-e4b").fit,
            Fit::Download {
                bytes: Some(4_977_171_584)
            }
        );
    }

    #[test]
    fn a_download_does_not_override_the_build_or_ram_rules() {
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(64), Some(500)),
            &none,
            Some(("ds4vision", 43)),
            &no_staged,
        );
        assert_eq!(
            row(&rows, "ds4vision").fit,
            Fit::Disabled {
                reason: "needs 96 GB RAM (this machine: 64 GB)".into()
            }
        );
    }

    #[test]
    fn a_staged_engine_reads_ready_to_install_and_skips_the_disk_rule() {
        // 50 GB free fails the disk rule for ds4vision, but its whole set is
        // already staged on disk: installing it needs no more space.
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(50)),
            &none,
            None,
            &|n| n == "ds4vision",
        );
        assert_eq!(row(&rows, "ds4vision").fit, Fit::Staged);
        assert!(row(&rows, "ds4vision").selectable());
        assert_eq!(
            row(&rows, "gemma4-e4b").fit,
            Fit::Download {
                bytes: Some(4_977_171_584)
            }
        );
    }

    #[test]
    fn a_staged_engine_still_obeys_the_ram_rule() {
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(64), Some(500)),
            &none,
            None,
            &|n| n == "ds4vision",
        );
        assert!(!row(&rows, "ds4vision").selectable());
    }

    #[test]
    fn size_labels_read_in_decimal_gigabytes() {
        assert_eq!(size_label(Some(4_977_171_584)), "5.0 GB");
        assert_eq!(size_label(Some(93_600_000_000)), "93.6 GB");
        assert_eq!(size_label(None), "size unknown");
    }

    #[test]
    fn sizes_below_a_gigabyte_read_in_megabytes_or_kilobytes() {
        assert_eq!(size_label(Some(512_000_000)), "512.0 MB");
        assert_eq!(size_label(Some(1_000_000_000)), "1.0 GB");
        assert_eq!(size_label(Some(28_467)), "28.5 KB");
        assert_eq!(size_label(Some(1_000_000)), "1.0 MB");
        assert_eq!(size_label(Some(0)), "0.0 KB");
    }

    #[test]
    fn free_bytes_walks_up_to_an_existing_directory() {
        let missing = std::env::temp_dir().join("plank-enginefit-no-such-dir/a/b");
        assert!(free_bytes(&missing).is_some_and(|b| b > 0));
    }
}
