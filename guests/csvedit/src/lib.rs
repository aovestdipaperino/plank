//! A Turbo Vision CSV editor, run by plank as a `frame` component.
//!
//! `csv` and `doc` are pure. `editor` is the Turbo Vision application.
//! `frame` holds the Extism exports and is compiled only for wasm, so native
//! `cargo test` covers everything but the ABI glue.

pub mod csv;
pub mod doc;
