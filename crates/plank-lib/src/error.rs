//! The one error type of this crate.

use std::backtrace::Backtrace;
use std::fmt;
use std::path::PathBuf;

/// Anything that stops a vector or profile file from being read or written.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Message(String),
}

impl Error {
    /// Wraps an I/O failure together with the path that caused it.
    #[must_use]
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::new(ErrorKind::Io {
            path: path.into(),
            source,
        })
    }

    /// A failure described by `message`.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Message(message.into()))
    }

    fn new(kind: ErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }

    /// The captured backtrace, empty unless `RUST_BACKTRACE` is set.
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::Io { path, source } => write!(f, "{}: {source}", path.display()),
            ErrorKind::Message(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Io { source, .. } => Some(source),
            ErrorKind::Message(_) => None,
        }
    }
}
