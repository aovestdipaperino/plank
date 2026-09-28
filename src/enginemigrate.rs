// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! One-shot migration from the `ModelSet` layout (`ds4flash.*`, `*.manifest`,
//! `staging-*/`, `downloads/job*.json`) to the engine layout. Idempotent: each
//! step moves a file only when its old name exists and its new one does not,
//! so a second run finds nothing to do. Top level of `~/.plank` only.

use std::path::{Path, PathBuf};

use crate::manifest::EngineId;

/// Old installed manifest → engine. Order matters only for the ds41 default
/// rule (decided before any move, from the raw filesystem state).
const MANIFESTS: [(&str, EngineId); 3] = [
    ("ds4.manifest", EngineId::DS4VISION),
    ("ds41.manifest", EngineId::DS41),
    ("qwen.manifest", EngineId::QWEN),
];

/// Old staging dir leaf, engine, old manifest leaf inside it.
const STAGING: [(&str, EngineId, &str); 3] = [
    ("staging", EngineId::DS4VISION, "ds4.manifest"),
    ("staging-ds41", EngineId::DS41, "ds41.manifest"),
    ("staging-qwen", EngineId::QWEN, "qwen.manifest"),
];

/// Old job file (under `downloads/`) → engine.
const JOBS: [(&str, EngineId); 3] = [
    ("job.json", EngineId::DS4VISION),
    ("job-ds41.json", EngineId::DS41),
    ("job-qwen.json", EngineId::QWEN),
];

/// Runs the migration under `root`, returning one warning per skipped move.
#[must_use]
pub fn migrate_in(root: &Path) -> Vec<String> {
    let mut warn = Vec::new();
    // Decided before anything moves: only `ds41.manifest` recorded, nothing
    // that would mean a ds4 install exists too.
    let ds41_only = root.join("ds41.manifest").exists()
        && !root.join("ds4.manifest").exists()
        && !root.join("ds4flash.gguf").exists()
        && !root.join("qwen.manifest").exists();

    for (old, id, role) in crate::engines::LEGACY_ARTIFACTS {
        if let Some(new) = crate::manifest::local_path_for_in(root, id, role) {
            move_one(&root.join(old), &new, &mut warn);
        }
    }
    for (old, id) in MANIFESTS {
        convert_manifest(
            &root.join(old),
            &crate::manifest::installed_path_in(root, id),
            &mut warn,
        );
    }
    // A helper spawned by an older plank writes only under the old staging
    // dirs and reads its job file, so those two steps wait while one holds
    // the lock: moving them underneath it would corrupt its download. The
    // top-level renames above are safe, since no helper touches them, and a
    // later launch finishes the rest.
    if !crate::downloader::running_in(root) {
        migrate_staging(root, &mut warn);
        for (old, id) in JOBS {
            let from = root.join("downloads").join(old);
            let to = crate::downloader::job_path_in(root, id);
            if from != to {
                convert_manifest(&from, &to, &mut warn);
            }
        }
    }
    let local = root.join("engines.local.json");
    if ds41_only && !local.exists() {
        let _ = std::fs::write(&local, "{\"default\":\"ds41\"}\n");
    }
    warn
}

/// [`migrate_in`] rooted at `~/.plank`.
#[must_use]
pub fn migrate() -> Vec<String> {
    migrate_in(&crate::manifest::plank_dir())
}

/// Renames `from` to `to` (moving a symlink itself, never its target).
fn move_one(from: &Path, to: &Path, warn: &mut Vec<String>) {
    if std::fs::symlink_metadata(from).is_err() {
        return;
    }
    if std::fs::symlink_metadata(to).is_ok() {
        warn.push(format!(
            "plank: not migrating {} — {} already exists",
            from.display(),
            to.display()
        ));
        return;
    }
    if let Some(parent) = to.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::rename(from, to) {
        warn.push(format!("plank: could not migrate {}: {e}", from.display()));
    }
}

/// Rewrites an old manifest (`files.dspark` → `files.mtp`) at `to`, then
/// removes `from`. An unparsable old manifest is dropped silently: it read as
/// absent before migration too.
fn convert_manifest(from: &Path, to: &Path, warn: &mut Vec<String>) {
    let Ok(text) = std::fs::read_to_string(from) else {
        return;
    };
    if to.exists() {
        warn.push(format!(
            "plank: not migrating {} — {} already exists",
            from.display(),
            to.display()
        ));
        return;
    }
    let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&text) else {
        let _ = std::fs::remove_file(from);
        return;
    };
    if let Some(files) = v
        .get_mut("files")
        .and_then(serde_json::Value::as_object_mut)
        && let Some(d) = files.remove("dspark")
    {
        files.insert("mtp".to_string(), d);
    }
    if let Some(parent) = to.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let out = serde_json::to_string_pretty(&v).unwrap_or(text);
    let tmp = to.with_extension(match to.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{ext}.tmp"),
        None => "tmp".to_string(),
    });
    let result = std::fs::write(&tmp, out).and_then(|()| std::fs::rename(&tmp, to));
    match result {
        Ok(()) => {
            let _ = std::fs::remove_file(from);
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            warn.push(format!("plank: could not migrate {}: {e}", from.display()));
        }
    }
}

/// Moves old flat staging dirs into `staging/<engine>/`.
///
/// `staging/` itself is both the old DS4 dir and the new parent, so its loose
/// files are moved first, into `staging/ds4vision/`.
fn migrate_staging(root: &Path, warn: &mut Vec<String>) {
    for (old_leaf, id, old_manifest) in STAGING {
        let old = root.join(old_leaf);
        let new = crate::manifest::staging_dir_in(root, id);
        let Ok(entries) = std::fs::read_dir(&old) else {
            continue;
        };
        let files: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        let mut manifest_file = None;
        let warn_before = warn.len();
        for f in files {
            let Some(name) = f.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name == old_manifest {
                manifest_file = Some(f);
                continue;
            }
            let renamed = name.replacen("dspark", "mtp", 1);
            move_one(&f, &new.join(renamed), warn);
        }
        // The staged manifest lands last, and only if every other move in this
        // dir succeeded — otherwise the old manifest stays in place so a later
        // run retries the whole directory together.
        if let Some(f) = manifest_file
            && warn.len() == warn_before
        {
            convert_manifest(
                &f,
                &crate::manifest::staged_manifest_path_in(root, id),
                warn,
            );
        }
        if old_leaf != "staging" {
            let _ = std::fs::remove_dir(&old);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::EngineId;

    fn scratch(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("plank-migrate-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn old_manifest(kinds: &[&str]) -> String {
        let files: Vec<String> = kinds
            .iter()
            .map(|k| {
                format!(
                    r#""{k}":{{"name":"{k}","url":"https://h/{k}","bytes":5,"sha256":"{}"}}"#,
                    "b".repeat(64)
                )
            })
            .collect();
        format!(
            r#"{{"version":4,"released":"x","notes":"n","files":{{{}}}}}"#,
            files.join(",")
        )
    }

    #[test]
    fn ds4_files_and_manifest_move_to_ds4vision() {
        let r = scratch("ds4");
        for f in [
            "ds4flash.gguf",
            "ds4flash.vision.gguf",
            "ds4flash.dspark.gguf",
        ] {
            std::fs::write(r.join(f), f).unwrap();
        }
        std::fs::write(
            r.join("ds4.manifest"),
            old_manifest(&["main", "vision", "dspark"]),
        )
        .unwrap();
        let w = migrate_in(&r);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(
            std::fs::read_to_string(r.join("ds4vision.gguf")).unwrap(),
            "ds4flash.gguf"
        );
        assert!(r.join("ds4vision.vision.gguf").exists());
        assert!(r.join("ds4vision.mtp.gguf").exists());
        assert!(!r.join("ds4flash.gguf").exists());
        let m =
            crate::manifest::read_at(&crate::manifest::installed_path_in(&r, EngineId::DS4VISION))
                .unwrap();
        assert!(m.files.contains_key("mtp") && !m.files.contains_key("dspark"));
        assert_eq!(m.version, 4);
        assert!(!r.join("ds4.manifest").exists());
    }

    #[test]
    fn ds41_files_move_and_a_ds41_only_root_defaults_to_ds41() {
        let r = scratch("ds41");
        std::fs::write(r.join("ds41flash.gguf"), "m").unwrap();
        std::fs::write(r.join("ds41flash.vision.gguf"), "v").unwrap();
        std::fs::write(r.join("ds41.manifest"), old_manifest(&["main", "vision"])).unwrap();
        let _ = migrate_in(&r);
        assert!(r.join("ds41.gguf").exists() && r.join("ds41.vision.gguf").exists());
        assert!(crate::manifest::installed_path_in(&r, EngineId::DS41).exists());
        let local = std::fs::read_to_string(r.join("engines.local.json")).unwrap();
        assert!(local.contains(r#""default":"ds41""#) || local.contains(r#""default": "ds41""#));
    }

    #[test]
    fn an_existing_local_file_is_never_overwritten() {
        let r = scratch("keep-local");
        std::fs::write(r.join("ds41.manifest"), old_manifest(&["main"])).unwrap();
        std::fs::write(r.join("engines.local.json"), "{\"default\":\"qwen\"}").unwrap();
        let _ = migrate_in(&r);
        assert_eq!(
            std::fs::read_to_string(r.join("engines.local.json")).unwrap(),
            "{\"default\":\"qwen\"}"
        );
    }

    #[test]
    fn qwen_files_stay_and_its_manifest_is_converted() {
        let r = scratch("qwen");
        std::fs::write(r.join("qwen.gguf"), "m").unwrap();
        std::fs::write(r.join("qwen.manifest"), old_manifest(&["main", "vision"])).unwrap();
        let _ = migrate_in(&r);
        assert!(r.join("qwen.gguf").exists());
        assert!(crate::manifest::installed_path_in(&r, EngineId::QWEN).exists());
        assert!(!r.join("engines.local.json").exists());
    }

    #[test]
    fn a_conflicting_destination_is_skipped_with_one_warning() {
        let r = scratch("conflict");
        std::fs::write(r.join("ds4flash.gguf"), "old").unwrap();
        std::fs::write(r.join("ds4vision.gguf"), "new").unwrap();
        let w = migrate_in(&r);
        assert_eq!(w.len(), 1, "{w:?}");
        assert_eq!(
            std::fs::read_to_string(r.join("ds4vision.gguf")).unwrap(),
            "new"
        );
        assert!(r.join("ds4flash.gguf").exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_is_moved_not_followed() {
        let r = scratch("symlink");
        let target = r.join("real-weights.gguf");
        std::fs::write(&target, "w").unwrap();
        std::os::unix::fs::symlink(&target, r.join("ds4flash.gguf")).unwrap();
        let _ = migrate_in(&r);
        let moved = r.join("ds4vision.gguf");
        assert!(
            std::fs::symlink_metadata(&moved)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(std::fs::read_link(&moved).unwrap(), target);
    }

    #[test]
    fn old_staging_and_jobs_move_under_the_engine() {
        let r = scratch("staging");
        std::fs::create_dir_all(r.join("staging")).unwrap();
        std::fs::write(r.join("staging/dspark.part"), "p").unwrap();
        std::fs::write(r.join("staging/main.gguf"), "s").unwrap();
        std::fs::write(
            r.join("staging/ds4.manifest"),
            old_manifest(&["main", "dspark"]),
        )
        .unwrap();
        std::fs::create_dir_all(r.join("staging-qwen")).unwrap();
        std::fs::write(r.join("staging-qwen/main.part"), "q").unwrap();
        std::fs::create_dir_all(r.join("downloads")).unwrap();
        std::fs::write(r.join("downloads/job.json"), old_manifest(&["main"])).unwrap();
        std::fs::write(r.join("downloads/job-qwen.json"), old_manifest(&["main"])).unwrap();
        let _ = migrate_in(&r);
        let ds = crate::manifest::staging_dir_in(&r, EngineId::DS4VISION);
        assert!(ds.join("mtp.part").exists());
        assert!(ds.join("main.gguf").exists());
        let staged = crate::manifest::read_at(&crate::manifest::staged_manifest_path_in(
            &r,
            EngineId::DS4VISION,
        ))
        .unwrap();
        assert!(staged.files.contains_key("mtp"));
        assert!(
            crate::manifest::staging_dir_in(&r, EngineId::QWEN)
                .join("main.part")
                .exists()
        );
        assert!(!r.join("staging-qwen").exists());
        assert!(crate::downloader::job_path_in(&r, EngineId::DS4VISION).exists());
        assert!(crate::downloader::job_path_in(&r, EngineId::QWEN).exists());
    }

    #[test]
    fn a_second_run_is_a_no_op() {
        let r = scratch("idem");
        std::fs::write(r.join("ds4flash.gguf"), "m").unwrap();
        std::fs::write(r.join("ds4.manifest"), old_manifest(&["main"])).unwrap();
        assert!(migrate_in(&r).is_empty());
        assert!(migrate_in(&r).is_empty());
        assert!(r.join("ds4vision.gguf").exists());
    }

    #[test]
    fn ds41_and_qwen_both_present_does_not_default_to_ds41() {
        let r = scratch("ds41-qwen");
        std::fs::write(r.join("ds41.manifest"), old_manifest(&["main"])).unwrap();
        std::fs::write(r.join("qwen.manifest"), old_manifest(&["main"])).unwrap();
        let _ = migrate_in(&r);
        assert!(!r.join("engines.local.json").exists());
    }

    #[test]
    fn ds41_staging_files_move_into_the_ds41_staging_dir() {
        let r = scratch("staging-ds41");
        std::fs::create_dir_all(r.join("staging-ds41")).unwrap();
        std::fs::write(r.join("staging-ds41/main.part"), "m").unwrap();
        let _ = migrate_in(&r);
        assert!(
            crate::manifest::staging_dir_in(&r, EngineId::DS41)
                .join("main.part")
                .exists()
        );
        assert!(!r.join("staging-ds41").exists());
    }

    #[test]
    fn staged_dspark_sha256_becomes_mtp_sha256() {
        let r = scratch("staged-sha");
        std::fs::create_dir_all(r.join("staging")).unwrap();
        std::fs::write(r.join("staging/dspark.gguf.sha256"), "hash").unwrap();
        let _ = migrate_in(&r);
        let new = crate::manifest::staging_dir_in(&r, EngineId::DS4VISION).join("mtp.gguf.sha256");
        assert!(new.exists());
        assert_eq!(std::fs::read_to_string(new).unwrap(), "hash");
    }

    #[test]
    fn convert_manifest_keeps_from_when_to_already_exists() {
        let r = scratch("convert-keep");
        std::fs::write(r.join("ds4.manifest"), old_manifest(&["main"])).unwrap();
        let to = crate::manifest::installed_path_in(&r, EngineId::DS4VISION);
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::write(&to, "preexisting").unwrap();
        let w = migrate_in(&r);
        assert!(r.join("ds4.manifest").exists());
        assert_eq!(std::fs::read_to_string(&to).unwrap(), "preexisting");
        assert_eq!(
            w.iter().filter(|m| m.contains("ds4.manifest")).count(),
            1,
            "{w:?}"
        );
    }

    #[test]
    fn a_skipped_staging_move_keeps_the_old_manifest_in_place() {
        let r = scratch("staging-order");
        std::fs::create_dir_all(r.join("staging")).unwrap();
        std::fs::write(r.join("staging/main.gguf"), "old").unwrap();
        std::fs::write(r.join("staging/ds4.manifest"), old_manifest(&["main"])).unwrap();
        let conflict_dir = crate::manifest::staging_dir_in(&r, EngineId::DS4VISION);
        std::fs::create_dir_all(&conflict_dir).unwrap();
        std::fs::write(conflict_dir.join("main.gguf"), "new").unwrap();
        let _ = migrate_in(&r);
        assert!(
            !crate::manifest::staged_manifest_path_in(&r, EngineId::DS4VISION).exists(),
            "manifest must not land while a sibling move was skipped"
        );
        assert!(r.join("staging/ds4.manifest").exists());
    }

    #[test]
    fn a_live_downloader_defers_only_staging_and_jobs() {
        let r = scratch("locked");
        std::fs::write(r.join("ds4flash.gguf"), "m").unwrap();
        std::fs::write(r.join("ds4.manifest"), old_manifest(&["main"])).unwrap();
        std::fs::create_dir_all(r.join("staging-qwen")).unwrap();
        std::fs::write(r.join("staging-qwen/main.part"), "q").unwrap();
        std::fs::create_dir_all(r.join("downloads")).unwrap();
        std::fs::write(r.join("downloads/job.json"), old_manifest(&["main"])).unwrap();
        let held = crate::downloader::try_lock_in(&r).expect("lock");
        let w = migrate_in(&r);
        assert!(w.is_empty(), "deferral is silent: {w:?}");
        assert!(
            r.join("ds4vision.gguf").exists() && !r.join("ds4flash.gguf").exists(),
            "a top-level artifact no helper touches moves even while locked"
        );
        assert!(crate::manifest::installed_path_in(&r, EngineId::DS4VISION).exists());
        assert!(
            r.join("staging-qwen/main.part").exists(),
            "the old staging dir a live helper writes into stays put"
        );
        assert!(r.join("downloads/job.json").exists());
        assert!(!crate::downloader::job_path_in(&r, EngineId::DS4VISION).exists());
        assert!(!crate::manifest::staging_dir_in(&r, EngineId::QWEN).exists());

        // Once the helper is gone, the next launch finishes the job.
        drop(held);
        assert!(migrate_in(&r).is_empty());
        assert!(
            crate::manifest::staging_dir_in(&r, EngineId::QWEN)
                .join("main.part")
                .exists()
        );
        assert!(crate::downloader::job_path_in(&r, EngineId::DS4VISION).exists());
    }
}
