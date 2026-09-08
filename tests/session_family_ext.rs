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

    set_family(ModelFamily::Qwen);
    assert_eq!(family(), ModelFamily::Qwen);
    let after = store.path_for_id("cheeky-bell");
    assert!(after.to_string_lossy().ends_with(".qwn.kv"), "{after:?}");

    // One directory, two names: the whole point of the tag.
    assert_eq!(before.parent(), after.parent());
    assert_ne!(before, after);
    assert_eq!(family_ext(ModelFamily::Qwen), ".qwn.kv");

    let _ = std::fs::remove_dir_all(&dir);
}
