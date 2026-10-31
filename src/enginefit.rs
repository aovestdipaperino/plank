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

/// RAM a native-engine family (Gemma, Flint) needs beyond its weights when
/// its catalog entry does not say: generous on purpose (4 GiB holds 8K tokens
/// of KV at up to ~500 KB a token). An entry's `kvBytesPerToken` replaces it.
pub const NATIVE_KV_RESERVE_BYTES: u64 = 4 * GIB;

/// Share of physical RAM a native-engine family's (Gemma, Flint) default
/// context may spend on its KV cache, as a divisor: a third
/// (`nativeengine::default_ctx`).
pub const NATIVE_KV_RAM_DIVISOR: usize = 3;

/// The smallest context a native-engine family (Gemma, Flint) opens by
/// default, however little RAM there is: below this an agent session cannot
/// hold its own system prompt and a few tool results. Also what the fit rule
/// reserves KV for, since a machine that can hold the weights and this much
/// KV can run the engine.
pub const NATIVE_MIN_CTX: usize = 8_192;

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
    /// The native Flint engine (`feature = "flint"`).
    pub flint: bool,
}

impl BuildCaps {
    /// What this binary can run.
    #[must_use]
    pub fn current() -> Self {
        Self {
            ds4: cfg!(ds4_engine),
            gemma: cfg!(feature = "gemma"),
            flint: cfg!(feature = "flint"),
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
    /// The largest buffer the Metal device makes, when known (Flint builds on
    /// a Mac with a device only). A model file larger than this cannot run.
    pub metal_max_buffer: Option<u64>,
}

/// This machine, with downloads landing under `root`.
#[must_use]
pub fn machine(root: &Path) -> Machine {
    Machine {
        ram: crate::download::total_ram_bytes(),
        free_disk: free_bytes(root),
        caps: BuildCaps::current(),
        metal_max_buffer: metal_max_buffer(),
    }
}

/// `MTLDevice::maxBufferLength`, probed at most once per process: opening a
/// Metal device is not free, and the menu asks on every evaluation.
#[cfg(feature = "flint")]
fn metal_max_buffer() -> Option<u64> {
    static CACHE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(flint_engine::metal_max_buffer_len)
}

#[cfg(not(feature = "flint"))]
fn metal_max_buffer() -> Option<u64> {
    None
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

/// The KV a native-engine family (Gemma, Flint) needs at the smallest window
/// it opens by default: the entry's `kvBytesPerToken` times
/// [`NATIVE_MIN_CTX`], else [`NATIVE_KV_RESERVE_BYTES`]. A larger machine gets
/// a larger window, so this is the floor that decides whether the engine runs
/// at all.
///
/// Declared per engine because it varies six-fold across the catalog's native
/// engines (about 115 KB a token on Gemma E4B, which shares KV across 18
/// layers; 688 KB on Gemma 12B, which keeps 8 heads on 40 sliding layers;
/// 100 KB on Flint) and cannot be read from a file that has not been
/// downloaded yet.
fn native_kv_reserve(entry: &EngineEntry) -> u64 {
    serde_json::from_str::<serde_json::Value>(&entry.raw)
        .ok()
        .and_then(|v| v["kvBytesPerToken"].as_u64())
        .map_or(NATIVE_KV_RESERVE_BYTES, |b| {
            b.saturating_mul(NATIVE_MIN_CTX as u64)
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

/// The page a Metal no-copy buffer is bound in on Apple silicon. The open-time
/// check (`flintengine`) compares the mapping rounded up to whole pages, so
/// the menu rounds the same way and never offers a file the open refuses.
const METAL_PAGE_BYTES: u64 = 16 * 1024;

/// Why this build cannot run an engine of catalog `family`, or `None` when it
/// can. No hint means the C engine. A hint this build does not know is
/// refused rather than handed to the C engine: a newer catalog can name a
/// family a released binary has never heard of, and treating it as `DeepSeek`
/// would download the file only for the open to fail.
fn unsupported_family(family: Option<&str>, caps: BuildCaps) -> Option<String> {
    let (supported, feature) = match family {
        None => (caps.ds4, None),
        Some("gemma") => (caps.gemma, Some("gemma")),
        Some("flint") => (caps.flint, Some("flint")),
        Some(other) => {
            return Some(format!(
                "needs a newer plank (unknown engine family \"{other}\")"
            ));
        }
    };
    if supported {
        return None;
    }
    Some(feature.map_or_else(
        || "not supported by this build".to_owned(),
        |f| format!("built without {f}"),
    ))
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
    let family = raw_str(entry, "family");
    let native = matches!(family.as_deref(), Some("gemma" | "flint"));
    if let Some(reason) = unsupported_family(family.as_deref(), m.caps) {
        return Fit::Disabled { reason };
    }
    if let Some(ram) = m.ram {
        let need = if native {
            entry.files.get("main").map(|f| {
                f.bytes.saturating_add(native_kv_reserve(entry))
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
    if family.as_deref() == Some("flint")
        && let (Some(lim), Some(main)) = (m.metal_max_buffer, entry.files.get("main"))
        && main.bytes.div_ceil(METAL_PAGE_BYTES) * METAL_PAGE_BYTES > lim
    {
        return Fit::Disabled {
            reason: format!(
                "needs a {} GB GPU buffer (this Mac allows {} GB)",
                need_gb(main.bytes),
                have_gb(lim)
            ),
        };
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
    let dir = path.ancestors().find(|p| p.exists())?;
    crate::platform::free_bytes(dir)
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
                flint: true,
            },
            metal_max_buffer: None,
        }
    }

    const FLINT: &str = r#"{"version": 3, "engines": {"flint": {"version": 1,
      "family": "flint", "kvBytesPerToken": 100352,
      "main": {"name": "flint-nvfp4.gguf", "url": "https://h/f", "bytes": 22863084160,
        "sha256": "97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde"}}}}"#;

    fn flint_row(m: &Machine) -> EngineRow {
        let mut w = Vec::new();
        let cat = parse(FLINT, Layer::Published, &mut w).expect("parses");
        let rows = evaluate(Path::new("/r"), &cat, m, &none, None, &no_staged);
        row(&rows, "flint").clone()
    }

    #[test]
    fn flint_is_selectable_when_ram_and_gpu_buffer_allow() {
        let mut m = machine(Some(64), Some(500));
        m.metal_max_buffer = Some(48 << 30);
        assert!(flint_row(&m).selectable());
    }

    #[test]
    fn flint_larger_than_the_gpu_buffer_is_disabled() {
        let mut m = machine(Some(64), Some(500));
        m.metal_max_buffer = Some(16 << 30);
        assert_eq!(
            flint_row(&m).fit,
            Fit::Disabled {
                reason: "needs a 22 GB GPU buffer (this Mac allows 16 GB)".into()
            }
        );
    }

    #[test]
    fn flint_needs_its_file_and_kv_within_eighty_percent_of_ram() {
        let mut m = machine(Some(24), Some(500));
        m.metal_max_buffer = Some(48 << 30);
        let Fit::Disabled { reason } = flint_row(&m).fit else {
            panic!("selectable at 24 GB");
        };
        assert!(
            reason.starts_with("needs ") && reason.contains("RAM"),
            "{reason}"
        );
    }

    #[test]
    fn flint_is_unsupported_without_the_build_cap() {
        let mut m = machine(Some(64), Some(500));
        m.caps.flint = false;
        assert_eq!(
            flint_row(&m).fit,
            Fit::Disabled {
                reason: "built without flint".into()
            }
        );
    }

    /// The open-time check rounds the file up to whole 16 KiB pages; a file
    /// whose raw size fits but whose rounded size does not is refused here
    /// too, not offered and then refused at open.
    #[test]
    fn flint_within_a_page_of_the_gpu_buffer_is_disabled() {
        let mut m = machine(Some(64), Some(500));
        // The FLINT main is 22863084160 bytes, not a multiple of 16 KiB.
        m.metal_max_buffer = Some(22_863_084_160);
        assert!(!flint_row(&m).selectable());
        m.metal_max_buffer = Some(22_863_084_160_u64.div_ceil(16 * 1024) * 16 * 1024);
        assert!(flint_row(&m).selectable());
    }

    fn family_row(family: &str, caps: BuildCaps) -> EngineRow {
        let json = format!(
            r#"{{"version": 9, "engines": {{"future": {{"version": 1, "family": "{family}",
              "main": {{"name": "f.gguf", "url": "https://h/f", "bytes": 1000,
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}}}}}}}"#
        );
        let mut w = Vec::new();
        let cat = parse(&json, Layer::Published, &mut w).expect("parses");
        let mut m = machine(Some(512), Some(500));
        m.caps = caps;
        let rows = evaluate(Path::new("/r"), &cat, &m, &none, None, &no_staged);
        row(&rows, "future").clone()
    }

    /// A family hint this build does not know is refused, never routed to the
    /// C engine, even on a machine that has it and the RAM for it.
    #[test]
    fn an_unknown_family_needs_a_newer_plank() {
        let all = BuildCaps {
            ds4: true,
            gemma: true,
            flint: true,
        };
        assert_eq!(
            family_row("nova", all).fit,
            Fit::Disabled {
                reason: "needs a newer plank (unknown engine family \"nova\")".into()
            }
        );
    }

    #[test]
    fn a_compiled_out_native_family_names_its_feature() {
        let no_gemma = BuildCaps {
            ds4: true,
            gemma: false,
            flint: true,
        };
        assert_eq!(
            family_row("gemma", no_gemma).fit,
            Fit::Disabled {
                reason: "built without gemma".into()
            }
        );
        let only_ds4 = BuildCaps {
            ds4: true,
            gemma: false,
            flint: false,
        };
        assert_eq!(
            family_row("flint", only_ds4).fit,
            Fit::Disabled {
                reason: "built without flint".into()
            }
        );
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
    /// heads at 512 values, 8 global layers of 1 head at 1024). The engine
    /// sizes its window from the RAM down to an 8K floor, so the machine has
    /// to hold the weights plus 8K of KV (~5.6 GB): ~15 GB at 80%.
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
            row(&at(8), "gemma4-12b").fit,
            Fit::Disabled {
                reason: "needs 15 GB RAM (this machine: 8 GB)".into()
            }
        );
        assert!(row(&at(16), "gemma4-12b").selectable());
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
            flint: true,
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
