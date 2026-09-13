//! The transcript family tag, verified in its own process.
//!
//! `session::set_family` writes a process-global, so the effect of *setting*
//! it cannot be observed from the unit tests: they all share one process, and
//! whichever test set the family first would decide the extension every later
//! test resolved. An integration test gets a process to itself.

use plank::gguf::ModelFamily;
use plank::session::{SessionStore, family, family_ext, set_family};

#[test]
fn setting_the_family_changes_the_transcript_extension() {
    // Unset means ds4: every transcript written before the families split was
    // a DeepSeek one, which is also what the migration assumes.
    assert_eq!(family(), ModelFamily::Ds4);

    let dir = std::env::temp_dir().join(format!("plank-famext-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = SessionStore::open(&dir).expect("open");

    let before = store.path_for_id("cheeky-bell");
    assert!(before.to_string_lossy().ends_with(".ds4.kv"), "{before:?}");

    set_family(ModelFamily::Ds41);
    assert_eq!(family(), ModelFamily::Ds41);
    let after = store.path_for_id("cheeky-bell");
    assert!(after.to_string_lossy().ends_with(".ds41.kv"), "{after:?}");

    // One directory, two names: the whole point of the tag.
    assert_eq!(before.parent(), after.parent());
    assert_ne!(before, after);
    assert_eq!(family_ext(ModelFamily::Ds41), ".ds41.kv");

    // The tag is a tag rather than a yes/no, so every family must round-trip
    // through the global — including back to the unset default.
    set_family(ModelFamily::Ds4);
    assert_eq!(family(), ModelFamily::Ds4);
    assert_eq!(store.path_for_id("cheeky-bell"), before);

    // No live family can ever produce the retired Qwen extension, so a
    // `.qwn.kv` transcript left on disk can never be opened, renamed or swept
    // as one of theirs.
    for f in [ModelFamily::Ds4, ModelFamily::Ds41] {
        assert_ne!(family_ext(f), ".qwn.kv");
    }

    let _ = std::fs::remove_dir_all(&dir);
}
