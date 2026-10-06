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

/// RAM a Gemma engine needs beyond its weights: about 114 KB of f32 KV per
/// token at the engine's 32768-token default (`docs/GEMMA.md`), rounded up.
pub const GEMMA_KV_RESERVE_BYTES: u64 = 4 * GIB;

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

/// A top-level string field of the engine's verbatim JSON.
fn raw_str(entry: &EngineEntry, key: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(&entry.raw)
        .ok()
        .and_then(|v| v[key].as_str().map(str::to_owned))
}

/// Whole gigabytes, rounded up, for a RAM or disk figure.
fn whole_gb(bytes: u64) -> u64 {
    bytes.div_ceil(GIB)
}

/// A download size for display: one decimal, or `size unknown`.
#[must_use]
pub fn size_label(bytes: Option<u64>) -> String {
    #[allow(clippy::cast_precision_loss)]
    bytes.map_or_else(
        || "size unknown".to_owned(),
        |b| format!("{:.1} GB", b as f64 / GIB as f64),
    )
}

/// Evaluates every engine in `catalog`, in name order.
#[must_use]
pub fn evaluate(
    root: &Path,
    catalog: &Catalog,
    m: &Machine,
    exists: &dyn Fn(&Path) -> bool,
) -> Vec<EngineRow> {
    catalog
        .engines
        .iter()
        .filter_map(|(name, entry)| {
            let sel = crate::engines::resolve_in(root, catalog, Choice::Named(name)).ok()?;
            Some(EngineRow {
                name: name.clone(),
                notes: raw_str(entry, "notes").unwrap_or_default(),
                fit: fit_of(entry, &sel, m, exists),
            })
        })
        .collect()
}

fn fit_of(
    entry: &EngineEntry,
    sel: &crate::engines::Selection,
    m: &Machine,
    exists: &dyn Fn(&Path) -> bool,
) -> Fit {
    let roles: [(&str, Option<&PathBuf>); 3] = [
        ("main", Some(&sel.main)),
        ("mtp", sel.mtp.as_ref()),
        ("vision", sel.vision.as_ref()),
    ];
    let mut missing = 0u64;
    let mut size_known = true;
    let mut installed = true;
    for (role, path) in roles {
        let Some(path) = path else { continue };
        if exists(path) {
            continue;
        }
        installed = false;
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
                (f.bytes + GEMMA_KV_RESERVE_BYTES) / crate::download::SSD_STREAMING_RAM_PERCENT
                    * 100
            })
        } else {
            Some(MIN_RAM_BYTES)
        };
        if let Some(need) = need.filter(|&n| ram < n) {
            return Fit::Disabled {
                reason: format!(
                    "needs {} GB RAM (this machine: {} GB)",
                    whole_gb(need),
                    whole_gb(ram)
                ),
            };
        }
    }
    if installed {
        return Fit::Installed;
    }
    if let Some(free) = m.free_disk
        && size_known
        && missing + DISK_MARGIN_BYTES > free
    {
        return Fit::Disabled {
            reason: format!(
                "needs {} GB free disk (have {} GB)",
                whole_gb(missing + DISK_MARGIN_BYTES),
                free / GIB
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

    #[test]
    fn a_large_machine_offers_every_engine_as_a_download() {
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(500)),
            &none,
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
        );
        assert!(row(&rows, "gemma4-e4b").selectable());
    }

    #[test]
    fn a_download_that_does_not_fit_the_disk_is_disabled() {
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(50)),
            &none,
        );
        assert_eq!(
            row(&rows, "ds4vision").fit,
            Fit::Disabled {
                reason: "needs 87 GB free disk (have 50 GB)".into()
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
        );
        assert_eq!(row(&rows, "ds4vision").fit, Fit::Installed);
    }

    #[test]
    fn only_missing_roles_count_towards_the_download() {
        let main = PathBuf::from("/r/ds4vision.gguf");
        let rows = evaluate(
            Path::new("/r"),
            &catalog(),
            &machine(Some(128), Some(500)),
            &|p| p == main,
        );
        assert_eq!(
            row(&rows, "ds4vision").fit,
            Fit::Download {
                bytes: Some(6_000_000_000)
            }
        );
    }

    #[test]
    fn unknown_ram_and_disk_never_disable() {
        let rows = evaluate(Path::new("/r"), &catalog(), &machine(None, None), &none);
        assert!(rows.iter().all(EngineRow::selectable));
    }

    #[test]
    fn an_engine_this_build_cannot_run_is_disabled_first() {
        let mut m = machine(Some(16), Some(500));
        m.caps = BuildCaps {
            ds4: false,
            gemma: true,
        };
        let rows = evaluate(Path::new("/r"), &catalog(), &m, &none);
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
        let rows = evaluate(Path::new("/r"), &cat, &machine(Some(128), Some(500)), &none);
        assert_eq!(
            row(&rows, "mine").fit,
            Fit::Disabled {
                reason: "file missing at /x/mine.gguf".into()
            }
        );
        assert_eq!(row(&rows, "fetchable").fit, Fit::Download { bytes: None });
    }

    #[test]
    fn size_labels_read_in_gigabytes() {
        assert_eq!(size_label(Some(4_977_171_584)), "4.6 GB");
        assert_eq!(size_label(None), "size unknown");
    }

    #[test]
    fn free_bytes_walks_up_to_an_existing_directory() {
        let missing = std::env::temp_dir().join("plank-enginefit-no-such-dir/a/b");
        assert!(free_bytes(&missing).is_some_and(|b| b > 0));
    }
}
