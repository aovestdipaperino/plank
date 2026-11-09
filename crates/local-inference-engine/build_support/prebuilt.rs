// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The pre-built ds4 core package: its manifest, its release asset names and
//! the checks a build runs before linking one (#89).
//!
//! A package is a directory, or a `.tar.gz` of one, holding `libds4core.a`,
//! the `ds4core.json` manifest, the `ds4.h` it was compiled against and, for a
//! Metal core, `metal/*.metal`. `PLANK_DS4_EXPORT` writes one from a source
//! build; `PLANK_DS4_PREBUILT` links one instead of compiling the C.
//!
//! Shared by `build.rs` (through `#[path]`) and `tests/prebuilt.rs`, so it
//! uses nothing but `std`, `serde_json` and `sha2`, which are both build- and
//! dev-dependencies of the crate.

#![allow(dead_code)] // each includer uses a different half

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::Path;

use sha2::{Digest, Sha256};

/// The manifest layout this build understands. Bumped when a field changes
/// meaning; an unknown format is refused rather than half-read.
pub const FORMAT: u64 = 1;

/// The manifest's file name inside a package.
pub const MANIFEST: &str = "ds4core.json";

/// The static library's file name inside a package.
pub const LIBRARY: &str = "libds4core.a";

/// The GitHub repository whose releases carry the packages, unless
/// `PLANK_DS4_PREBUILT_REPO` names another.
pub const DEFAULT_REPO: &str = "aovestdipaperino/plank";

/// The CUDA architectures the `ds4core` workflow publishes, oldest first.
/// [`cuda_variant_for`] picks among them for the GPU in the machine.
pub const CUDA_VARIANTS: &[&str] = &["sm_80", "sm_86", "sm_89", "sm_90", "sm_120", "sm_121"];

/// What a package declares about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The ds4 commit the core was compiled from (full 40-hex hash).
    pub ds4_commit: String,
    /// Whether that ds4 tree had uncommitted changes when it was compiled.
    pub ds4_dirty: bool,
    /// The Rust target triple the objects were compiled for.
    pub target: String,
    /// `metal`, `cpu`, `cuda` or `rocm`.
    pub backend: String,
    /// The GPU architecture, `""` for a CPU or Metal core: a CUDA `sm_*`
    /// (`CUDA_ARCH`) or a `ROCm` `gfx*` (`ROCM_ARCH`).
    pub variant: String,
    /// The `NATIVE_CPU_FLAG` the host objects were compiled with. `native`
    /// means the build machine's own CPU, which is not portable.
    pub cpu_flag: String,
    /// The CUDA or `ROCm` toolkit version, `""` when not applicable or unknown.
    pub toolkit: String,
    /// Lowercase hex SHA-256 of `libds4core.a`.
    pub lib_sha256: String,
}

impl Manifest {
    /// Parses `ds4core.json`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("{MANIFEST} is not valid JSON: {e}"))?;
        let format = v.get("format").and_then(serde_json::Value::as_u64);
        if format != Some(FORMAT) {
            return Err(format!(
                "{MANIFEST} has format {}, this build reads format {FORMAT}",
                format.map_or_else(|| "none".to_string(), |f| f.to_string())
            ));
        }
        let field = |name: &str| -> Result<String, String> {
            v.get(name)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("{MANIFEST} has no string field `{name}`"))
        };
        let optional = |name: &str| {
            v.get(name)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let m = Self {
            ds4_commit: field("ds4_commit")?,
            ds4_dirty: v
                .get("ds4_dirty")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            target: field("target")?,
            backend: field("backend")?,
            variant: optional("variant"),
            cpu_flag: optional("cpu_flag"),
            toolkit: optional("toolkit"),
            lib_sha256: field("lib_sha256")?.to_ascii_lowercase(),
        };
        if !is_hex(&m.ds4_commit, 40) {
            return Err(format!(
                "{MANIFEST}: ds4_commit `{}` is not a commit hash",
                m.ds4_commit
            ));
        }
        if !is_hex(&m.lib_sha256, 64) {
            return Err(format!(
                "{MANIFEST}: lib_sha256 `{}` is not a SHA-256",
                m.lib_sha256
            ));
        }
        if !matches!(m.backend.as_str(), "metal" | "cpu" | "cuda" | "rocm") {
            return Err(format!("{MANIFEST}: unknown backend `{}`", m.backend));
        }
        Ok(m)
    }

    /// The manifest as pretty JSON, one field per line (the release workflow
    /// reads `asset` back with `sed`).
    #[must_use]
    pub fn to_json(&self) -> String {
        let v = serde_json::json!({
            "format": FORMAT,
            "asset": asset_name(&self.target, &self.backend, &self.variant),
            "ds4_commit": self.ds4_commit,
            "ds4_dirty": self.ds4_dirty,
            "target": self.target,
            "backend": self.backend,
            "variant": self.variant,
            "cpu_flag": self.cpu_flag,
            "toolkit": self.toolkit,
            "lib_sha256": self.lib_sha256,
        });
        let mut s = serde_json::to_string_pretty(&v).expect("a JSON object serializes");
        s.push('\n');
        s
    }

    /// Checks the manifest against the build that wants to link it. Returns
    /// the warnings worth printing; an incompatibility is an `Err`.
    pub fn check(&self, want: &Expect<'_>) -> Result<Vec<String>, String> {
        let mut warnings = Vec::new();
        if self.target != want.target {
            return Err(format!(
                "the pre-built ds4 core is for {}, this build targets {}",
                self.target, want.target
            ));
        }
        if let Some(backend) = want.backend
            && backend != self.backend
        {
            return Err(format!(
                "the pre-built ds4 core is a {} core, PLANK_DS4_BACKEND asks for {backend}",
                self.backend
            ));
        }
        match want.ds4_commit {
            Some(pin) if !pin.eq_ignore_ascii_case(&self.ds4_commit) => {
                let msg = format!(
                    "the pre-built ds4 core was compiled from ds4 {}, but refs/ds4 pins {}; \
                     plank's FFI struct layouts follow the pinned ds4.h, so a different \
                     commit can corrupt memory",
                    short(&self.ds4_commit),
                    short(pin)
                );
                if !want.allow_commit_mismatch {
                    return Err(format!(
                        "{msg} (set PLANK_DS4_PREBUILT_ALLOW_MISMATCH=1 only if you know the two headers agree)"
                    ));
                }
                warnings.push(msg);
            }
            Some(_) => {}
            None => warnings.push(format!(
                "cannot tell which ds4 commit this checkout pins (no git, set DS4_COMMIT); \
                 trusting the pre-built core's ds4 {}",
                short(&self.ds4_commit)
            )),
        }
        if self.ds4_dirty {
            warnings.push(format!(
                "the pre-built ds4 core was compiled from a modified ds4 {} tree",
                short(&self.ds4_commit)
            ));
        }
        if self.cpu_flag == "native" || self.cpu_flag.is_empty() {
            warnings.push(
                "the pre-built ds4 core was compiled for its build machine's CPU (-march=native); \
                 it may crash with an illegal instruction elsewhere"
                    .to_string(),
            );
        }
        Ok(warnings)
    }
}

/// What the build linking a package requires of it.
#[derive(Debug, Clone, Copy)]
pub struct Expect<'a> {
    /// The Rust target triple being built (`TARGET`).
    pub target: &'a str,
    /// The backend asked for with `PLANK_DS4_BACKEND`, if any.
    pub backend: Option<&'a str>,
    /// The ds4 commit `refs/ds4` pins, when it can be read.
    pub ds4_commit: Option<&'a str>,
    /// `PLANK_DS4_PREBUILT_ALLOW_MISMATCH`: a different commit only warns.
    pub allow_commit_mismatch: bool,
}

/// The release asset holding the package for `target`, `backend` and
/// `variant`: `ds4core-<target>-<backend>[-<variant>].tar.gz`.
#[must_use]
pub fn asset_name(target: &str, backend: &str, variant: &str) -> String {
    if variant.is_empty() {
        format!("ds4core-{target}-{backend}.tar.gz")
    } else {
        format!("ds4core-{target}-{backend}-{variant}.tar.gz")
    }
}

/// The release that carries every package built from ds4 `commit`:
/// `ds4core-<first 12 hex>`. Keyed by the ds4 commit, not plank's version,
/// because that commit is what decides whether a core fits plank's FFI.
#[must_use]
pub fn release_tag(commit: &str) -> String {
    format!("ds4core-{}", short(commit))
}

/// The checksum file published beside the packages of `tag`.
#[must_use]
pub fn sums_name(tag: &str) -> String {
    format!("{tag}-SHA256SUMS")
}

/// The download URL of `file` in release `tag` of `repo`.
#[must_use]
pub fn release_url(repo: &str, tag: &str, file: &str) -> String {
    format!("https://github.com/{repo}/releases/download/{tag}/{file}")
}

/// Looks `name` up in `sha256sum` output (`<hex>  <name>`, or `<hex> *<name>`
/// in binary mode). Returns the lowercase hash.
#[must_use]
pub fn sum_for(sums: &str, name: &str) -> Option<String> {
    sums.lines().find_map(|line| {
        let (hash, file) = line.trim_end().split_once(char::is_whitespace)?;
        let file = file.trim_start().trim_start_matches('*');
        (file == name && is_hex(hash, 64)).then(|| hash.to_ascii_lowercase())
    })
}

/// The CUDA variant to fetch for a GPU of compute capability `cap`
/// (`nvidia-smi --query-gpu=compute_cap`, e.g. `8.9`).
///
/// A cubin runs on the same major at an equal or higher minor, so the highest
/// published variant of the GPU's own major that does not exceed it wins.
/// Failing that (a newer major than any variant), the highest older variant
/// is used: nvcc's `-arch=sm_XY` also embeds `compute_XY` PTX, which the
/// driver JIT-compiles for the newer GPU. `None` when the GPU is older than
/// every variant.
#[must_use]
pub fn cuda_variant_for(cap: &str, variants: &[&str]) -> Option<String> {
    let (major, minor) = parse_cap(cap)?;
    let parsed = variants
        .iter()
        .filter_map(|v| Some((parse_sm(v)?, *v)))
        .collect::<Vec<_>>();
    let same_major = parsed
        .iter()
        .filter(|((ma, mi), _)| *ma == major && *mi <= minor)
        .max_by_key(|(sm, _)| *sm);
    let older = parsed
        .iter()
        .filter(|((ma, _), _)| *ma < major)
        .max_by_key(|(sm, _)| *sm);
    same_major.or(older).map(|(_, v)| (*v).to_string())
}

/// `8.9` → `(8, 9)`.
fn parse_cap(cap: &str) -> Option<(u32, u32)> {
    let (ma, mi) = cap.trim().split_once('.')?;
    Some((ma.parse().ok()?, mi.parse().ok()?))
}

/// `sm_89` → `(8, 9)`, `sm_120` → `(12, 0)`: the last digit is the minor.
fn parse_sm(sm: &str) -> Option<(u32, u32)> {
    let digits = sm.strip_prefix("sm_")?.trim_end_matches('a');
    if digits.len() < 2 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (ma, mi) = digits.split_at(digits.len() - 1);
    Some((ma.parse().ok()?, mi.parse().ok()?))
}

/// The CUDA variant name for a `CUDA_ARCH` value, as an export records it.
#[must_use]
pub fn variant_of_cuda_arch(arch: &str) -> String {
    arch.trim().trim_end_matches('a').to_string()
}

/// The commit a `git ls-files --stage refs/ds4` line names: the index entry
/// of the submodule, so a staged but uncommitted bump already counts.
#[must_use]
pub fn gitlink_commit(ls_files: &str) -> Option<String> {
    let line = ls_files.lines().next()?;
    let mut parts = line.split_whitespace();
    (parts.next()? == "160000")
        .then(|| parts.next())
        .flatten()
        .filter(|c| is_hex(c, 40))
        .map(str::to_ascii_lowercase)
}

/// The `major.minor.patch` CUDA version in `$CUDA_HOME/version.json`.
#[must_use]
pub fn cuda_toolkit_version(version_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(version_json).ok()?;
    v.get("cuda")?.get("version")?.as_str().map(str::to_string)
}

/// The leading number of a dotted version: `12.9.1` → `12`.
#[must_use]
pub fn major(version: &str) -> &str {
    version.split('.').next().unwrap_or_default()
}

/// Lowercase hex SHA-256 of the file at `path`.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The first 12 characters of a commit hash.
#[must_use]
pub fn short(commit: &str) -> &str {
    commit.get(..12).unwrap_or(commit)
}
