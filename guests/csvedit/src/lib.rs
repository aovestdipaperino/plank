//! A Turbo Vision CSV editor, run by plank as a `frame` component.
//!
//! `csv` and `doc` are pure. `editor` is the Turbo Vision application, with
//! its dialogs in `dialogs` and its screen conversion in `paint`. `frame`
//! holds the Extism exports and is compiled only for wasm, so native
//! `cargo test` covers everything but the ABI glue.

pub mod commands;
pub mod csv;
mod dialogs;
pub mod disk;
pub mod doc;
pub mod editor;
pub mod keys;
pub mod paint;
