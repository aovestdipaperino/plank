// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! [`Model`] and [`Session`] for a build without the C engine.
//!
//! Same signatures as the real ones, so a dependent compiles unchanged. No
//! value of either type can exist: [`Model::open`] always fails, and each
//! type holds an [`Infallible`], so every other method is statically
//! unreachable and nothing here references a C symbol.

use std::convert::Infallible;
use std::sync::Arc;

use crate::ffi::Ds4ThinkMode as ThinkMode;
use crate::{Error, Options, TokenScore, ffi};

/// A loaded model. Never constructed in this build.
#[derive(Debug)]
pub struct Model {
    never: Infallible,
}

#[allow(missing_docs, clippy::missing_errors_doc, clippy::unused_self)]
impl Model {
    /// Always fails: this build has no ds4 engine.
    ///
    /// # Errors
    /// Always.
    pub fn open(options: &Options) -> Result<Self, Error> {
        Err(Error::new(format!(
            "cannot open {}: built without the ds4 engine (it needs macOS and the ds4 C sources)",
            options.model.display()
        )))
    }

    #[must_use]
    pub fn as_raw(&self) -> *mut ffi::Ds4Engine {
        match self.never {}
    }

    #[must_use]
    pub fn ctx_size(&self) -> i32 {
        match self.never {}
    }

    #[must_use]
    pub fn name(&self) -> String {
        match self.never {}
    }

    #[must_use]
    pub fn eos(&self) -> i32 {
        match self.never {}
    }

    #[must_use]
    pub fn has_vision(&self) -> bool {
        match self.never {}
    }

    pub fn encode_chat(&self, _: &str, _: &str, _: ThinkMode) -> Result<Vec<i32>, Error> {
        match self.never {}
    }

    pub fn tokenize_rendered(&self, _: &str) -> Result<Vec<i32>, Error> {
        match self.never {}
    }

    #[must_use]
    pub fn token_bytes(&self, _: i32) -> Vec<u8> {
        match self.never {}
    }
}

/// An inference stream. Never constructed in this build.
#[derive(Debug)]
pub struct Session {
    never: Infallible,
}

#[allow(missing_docs, clippy::missing_errors_doc, clippy::unused_self)]
impl Session {
    pub fn new(model: &Arc<Model>, _: i32) -> Result<Self, Error> {
        match model.never {}
    }

    #[must_use]
    pub fn as_raw(&self) -> *mut ffi::Ds4Session {
        match self.never {}
    }

    #[must_use]
    pub fn model(&self) -> &Arc<Model> {
        match self.never {}
    }

    #[must_use]
    pub fn ctx(&self) -> i32 {
        match self.never {}
    }

    #[must_use]
    pub fn pos(&self) -> i32 {
        match self.never {}
    }

    pub fn invalidate(&mut self) {
        match self.never {}
    }

    pub fn sync(&mut self, _: &[i32]) -> Result<(), Error> {
        match self.never {}
    }

    pub fn sync_with_progress(
        &mut self,
        _: &[i32],
        _: &mut dyn FnMut(&str, i32, i32),
    ) -> Result<(), Error> {
        match self.never {}
    }

    pub fn eval(&mut self, _: i32) -> Result<(), Error> {
        match self.never {}
    }

    #[must_use]
    pub fn sample(&mut self, _: f32, _: i32, _: f32, _: f32, _: &mut u64) -> i32 {
        match self.never {}
    }

    #[must_use]
    pub fn top_logprobs(&self, _: usize) -> Vec<TokenScore> {
        match self.never {}
    }
}
