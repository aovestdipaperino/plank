//! plank's user-level files, shared by plank and `pt`.
//!
//! - [`vectors`]: named directional-steering vectors in
//!   `~/.plank/models/vectors.json`, keyed by engine (or file) name, with the
//!   decoding the C engine needs and the merge `pt install` performs.
//! - [`profiles`]: profile manifests (name, version, steering) and installing
//!   a profile folder into `~/.plank/profiles`.
//! - [`source`]: `repo:path` references and fetching them from a local
//!   checkout or a git remote.
//! - [`home`]: where plank keeps all of this.
//!
//! Errors are [`Error`]; plank turns them into its own messages with
//! `to_string`.

mod error;
pub mod home;
pub mod profiles;
pub mod source;
pub mod vectors;

pub use error::Error;
