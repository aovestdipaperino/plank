// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The pre-built ds4 core's manifest, asset names and checks
//! (`build_support/prebuilt.rs`, shared with `build.rs`). Pure logic: runs on
//! every OS, with or without the C engine.

#[path = "../build_support/prebuilt.rs"]
mod prebuilt;

use prebuilt::{Expect, Manifest};

const COMMIT: &str = "92abf5fc04ac001c25a2a4ba338a1e1130456a36";
const OTHER: &str = "0123456789abcdef0123456789abcdef01234567";
const SHA: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

fn manifest() -> Manifest {
    Manifest {
        ds4_commit: COMMIT.to_string(),
        ds4_dirty: false,
        target: "x86_64-unknown-linux-gnu".to_string(),
        backend: "cuda".to_string(),
        variant: "sm_89".to_string(),
        cpu_flag: "-march=x86-64-v3".to_string(),
        toolkit: "12.9.1".to_string(),
        lib_sha256: SHA.to_string(),
    }
}

fn expect() -> Expect<'static> {
    Expect {
        target: "x86_64-unknown-linux-gnu",
        backend: Some("cuda"),
        ds4_commit: Some(COMMIT),
        allow_commit_mismatch: false,
    }
}

#[test]
fn manifest_round_trips_and_names_its_asset() {
    let m = manifest();
    let json = m.to_json();
    assert_eq!(Manifest::parse(&json).unwrap(), m);
    // The release workflow reads the asset back with sed: one field per line.
    assert!(
        json.lines().any(
            |l| l.trim() == r#""asset": "ds4core-x86_64-unknown-linux-gnu-cuda-sm_89.tar.gz","#
        ),
        "{json}"
    );
}

#[test]
fn parse_refuses_unknown_formats_and_bad_fields() {
    let good: serde_json::Value = serde_json::from_str(&manifest().to_json()).unwrap();
    let with = |key: &str, value: serde_json::Value| {
        let mut v = good.clone();
        v[key] = value;
        Manifest::parse(&v.to_string())
    };
    assert!(with("format", 2.into()).unwrap_err().contains("format 2"));
    assert!(
        with("backend", "vulkan".into())
            .unwrap_err()
            .contains("vulkan")
    );
    assert!(with("ds4_commit", "main".into()).is_err());
    assert!(with("lib_sha256", "abc".into()).is_err());
    let mut v = good.clone();
    v.as_object_mut().unwrap().remove("target");
    assert!(
        Manifest::parse(&v.to_string())
            .unwrap_err()
            .contains("target")
    );
    assert!(Manifest::parse("not json").is_err());
}

#[test]
fn a_matching_package_passes_without_warnings() {
    assert_eq!(manifest().check(&expect()).unwrap(), Vec::<String>::new());
}

#[test]
fn wrong_target_or_backend_is_refused() {
    let mut m = manifest();
    m.target = "aarch64-unknown-linux-gnu".to_string();
    assert!(
        m.check(&expect())
            .unwrap_err()
            .contains("aarch64-unknown-linux-gnu")
    );
    let mut m = manifest();
    m.backend = "rocm".to_string();
    assert!(m.check(&expect()).unwrap_err().contains("rocm"));
    // With no backend asked for, the package's own backend is accepted.
    let any = Expect {
        backend: None,
        ..expect()
    };
    assert!(m.check(&any).is_ok());
}

#[test]
fn another_ds4_commit_is_refused_unless_allowed() {
    let mut m = manifest();
    m.ds4_commit = OTHER.to_string();
    let err = m.check(&expect()).unwrap_err();
    assert!(
        err.contains(&OTHER[..12]) && err.contains(&COMMIT[..12]),
        "{err}"
    );
    assert!(err.contains("PLANK_DS4_PREBUILT_ALLOW_MISMATCH"), "{err}");
    let allow = Expect {
        allow_commit_mismatch: true,
        ..expect()
    };
    let warnings = m.check(&allow).unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    // The pin compares case-insensitively.
    let upper = COMMIT.to_ascii_uppercase();
    let pinned_upper = Expect {
        ds4_commit: Some(&upper),
        ..expect()
    };
    assert!(manifest().check(&pinned_upper).unwrap().is_empty());
}

#[test]
fn unknown_pin_dirty_tree_and_native_cpu_only_warn() {
    let mut m = manifest();
    m.ds4_dirty = true;
    m.cpu_flag = "native".to_string();
    let unpinned = Expect {
        ds4_commit: None,
        ..expect()
    };
    let warnings = m.check(&unpinned).unwrap();
    assert_eq!(warnings.len(), 3, "{warnings:?}");
}

#[test]
fn asset_and_release_names() {
    assert_eq!(
        prebuilt::asset_name("aarch64-apple-darwin", "metal", ""),
        "ds4core-aarch64-apple-darwin-metal.tar.gz"
    );
    assert_eq!(prebuilt::release_tag(COMMIT), "ds4core-92abf5fc04ac");
    assert_eq!(
        prebuilt::sums_name("ds4core-92abf5fc04ac"),
        "ds4core-92abf5fc04ac-SHA256SUMS"
    );
    assert_eq!(
        prebuilt::release_url("o/r", "t", "f.tar.gz"),
        "https://github.com/o/r/releases/download/t/f.tar.gz"
    );
}

#[test]
fn sums_lookup_matches_whole_names_only() {
    let sums = format!(
        "{SHA}  ds4core-x-cuda-sm_89.tar.gz\n{} *ds4core-x-cpu.tar.gz\n",
        "A".repeat(64)
    );
    assert_eq!(
        prebuilt::sum_for(&sums, "ds4core-x-cuda-sm_89.tar.gz").as_deref(),
        Some(SHA)
    );
    assert_eq!(
        prebuilt::sum_for(&sums, "ds4core-x-cpu.tar.gz"),
        Some("a".repeat(64))
    );
    assert_eq!(prebuilt::sum_for(&sums, "ds4core-x-cuda.tar.gz"), None);
}

#[test]
fn cuda_variant_follows_binary_compatibility() {
    let v = prebuilt::CUDA_VARIANTS;
    let pick = |cap| prebuilt::cuda_variant_for(cap, v);
    assert_eq!(pick("8.0").as_deref(), Some("sm_80"));
    assert_eq!(pick("8.6").as_deref(), Some("sm_86"));
    // Orin (8.7) runs the sm_86 cubin; Ada (8.9) has its own.
    assert_eq!(pick("8.7").as_deref(), Some("sm_86"));
    assert_eq!(pick("8.9").as_deref(), Some("sm_89"));
    assert_eq!(pick("9.0").as_deref(), Some("sm_90"));
    // Blackwell datacenter (10.0) has no variant of its major: sm_90's PTX.
    assert_eq!(pick("10.0").as_deref(), Some("sm_90"));
    assert_eq!(pick("12.0").as_deref(), Some("sm_120"));
    assert_eq!(pick("12.1").as_deref(), Some("sm_121"));
    // Turing is older than every variant.
    assert_eq!(pick("7.5"), None);
    assert_eq!(pick("garbage"), None);
}

#[test]
fn cuda_arch_variant_names() {
    assert_eq!(prebuilt::variant_of_cuda_arch("sm_120a"), "sm_120");
    assert_eq!(prebuilt::variant_of_cuda_arch(" sm_89 "), "sm_89");
}

#[test]
fn gitlink_and_toolkit_parsing() {
    let line = format!("160000 {COMMIT} 0\trefs/ds4\n");
    assert_eq!(prebuilt::gitlink_commit(&line).as_deref(), Some(COMMIT));
    assert_eq!(
        prebuilt::gitlink_commit(&format!("100644 {COMMIT} 0\tfile")),
        None
    );
    assert_eq!(prebuilt::gitlink_commit(""), None);
    let json = r#"{"cuda": {"name": "CUDA SDK", "version": "12.9.1"}, "nvcc": {}}"#;
    assert_eq!(
        prebuilt::cuda_toolkit_version(json).as_deref(),
        Some("12.9.1")
    );
    assert_eq!(prebuilt::major("12.9.1"), "12");
}

#[test]
fn sha256_of_a_file() {
    let dir = std::env::temp_dir().join(format!("plank-prebuilt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let empty = dir.join("empty");
    std::fs::write(&empty, b"").unwrap();
    assert_eq!(prebuilt::sha256_file(&empty).unwrap(), SHA);
    let _ = std::fs::remove_dir_all(&dir);
}
