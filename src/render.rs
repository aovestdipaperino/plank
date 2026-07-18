//! Assistant markdown rendering: streaming terminal renderer for model output.
//!
//! Port of the "Assistant Markdown Rendering" section of `ds4-ref/ds4_agent.c`.
//! The renderer handles only the cheap markdown cues that make terminal output
//! readable: `**bold**`, `*italic*`, inline code, and fenced code blocks with a
//! kilo-style keyword highlighter. It is a streaming parser, so it buffers only
//! ambiguous marker bytes long enough to decide whether they are formatting or
//! literal text. It also hides `<think>` tags and renders thinking text grey.

use std::fmt;
use std::io::Write;

// ---------------------------------------------------------------------------
// Tail capture
// ---------------------------------------------------------------------------

/// Ring buffer recording the last N output bytes plus a total byte count.
#[derive(Debug, Default)]
pub struct TailCapture {
    buf: Vec<u8>,
    cap: usize,
    start: usize,
    len: usize,
    total: u64,
}

impl TailCapture {
    /// Creates a capture that retains at most `cap` trailing bytes.
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self {
            buf: Vec::new(),
            cap,
            start: 0,
            len: 0,
            total: 0,
        }
    }

    /// Appends bytes, keeping only the most recent `cap` bytes.
    pub fn append(&mut self, s: &[u8]) {
        if s.is_empty() || self.cap == 0 {
            return;
        }
        if self.buf.is_empty() {
            self.buf = vec![0; self.cap];
        }
        self.total += s.len() as u64;

        if s.len() >= self.cap {
            self.buf.copy_from_slice(&s[s.len() - self.cap..]);
            self.start = 0;
            self.len = self.cap;
            return;
        }

        for &b in s {
            if self.len < self.cap {
                let pos = (self.start + self.len) % self.cap;
                self.buf[pos] = b;
                self.len += 1;
            } else {
                self.buf[self.start] = b;
                self.start = (self.start + 1) % self.cap;
            }
        }
    }

    /// Returns the captured bytes in order and resets the capture.
    pub fn take(&mut self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.len);
        for i in 0..self.len {
            out.push(self.buf[(self.start + i) % self.cap]);
        }
        let cap = self.cap;
        *self = Self::new(cap);
        out
    }

    /// Total number of bytes ever appended.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Number of bytes currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no bytes are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---------------------------------------------------------------------------
// Syntax highlighting tables
// ---------------------------------------------------------------------------

/// Highlight class assigned to a run of code-block bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Highlight {
    Normal,
    Comment,
    Keyword1,
    Keyword2,
    String,
    Number,
}

const SYNTAX_NUMBERS: u8 = 1 << 0;
const SYNTAX_STRINGS: u8 = 1 << 1;
const SYNTAX_BACKTICK_STRINGS: u8 = 1 << 2;
const SYNTAX_CASE_INSENSITIVE: u8 = 1 << 3;

/// One language entry of the poor man's code highlighter.
///
/// Keywords ending in `|` are secondary (type-like) keywords, following the
/// kilo convention of the C reference.
#[derive(Debug)]
pub struct Syntax {
    name: &'static str,
    aliases: &'static str,
    keywords: &'static [&'static str],
    singleline_comments: &'static [&'static str],
    multiline_start: Option<&'static str>,
    multiline_end: Option<&'static str>,
    flags: u8,
}

impl Syntax {
    /// Canonical language name of this entry.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }
}

static KW_GENERIC: &[&str] = &[
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "return",
    "try",
    "catch",
    "finally",
    "throw",
    "throws",
    "class",
    "struct",
    "enum",
    "interface",
    "trait",
    "impl",
    "fn",
    "func",
    "function",
    "def",
    "lambda",
    "let",
    "var",
    "const",
    "static",
    "public",
    "private",
    "protected",
    "import",
    "include",
    "from",
    "export",
    "package",
    "module",
    "namespace",
    "new",
    "delete",
    "async",
    "await",
    "yield",
    "match",
    "type",
    "true|",
    "false|",
    "null|",
    "nil|",
    "none|",
    "None|",
    "NULL|",
    "void|",
    "int|",
    "long|",
    "float|",
    "double|",
    "char|",
    "bool|",
    "string|",
    "String|",
    "usize|",
    "isize|",
    "u8|",
    "u16|",
    "u32|",
    "u64|",
    "i8|",
    "i16|",
    "i32|",
    "i64|",
];

static KW_C: &[&str] = &[
    "auto",
    "break",
    "case",
    "continue",
    "default",
    "do",
    "else",
    "enum",
    "extern",
    "for",
    "goto",
    "if",
    "register",
    "return",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "volatile",
    "while",
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "bitand",
    "bitor",
    "class",
    "compl",
    "constexpr",
    "const_cast",
    "decltype",
    "delete",
    "dynamic_cast",
    "explicit",
    "export",
    "false",
    "friend",
    "inline",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "reinterpret_cast",
    "static_assert",
    "static_cast",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typeid",
    "typename",
    "virtual",
    "xor",
    "xor_eq",
    "NULL|",
    "bool|",
    "char|",
    "const|",
    "double|",
    "float|",
    "int|",
    "long|",
    "short|",
    "signed|",
    "size_t|",
    "ssize_t|",
    "uint8_t|",
    "uint16_t|",
    "uint32_t|",
    "uint64_t|",
    "unsigned|",
    "void|",
];

static KW_PYTHON: &[&str] = &[
    "and", "as", "assert", "async", "await", "break", "case", "class", "continue", "def", "del",
    "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "match", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with",
    "yield", "False|", "None|", "True|", "bool|", "bytes|", "dict|", "float|", "int|", "list|",
    "object|", "set|", "str|", "tuple|",
];

static KW_JS: &[&str] = &[
    "async",
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "else",
    "export",
    "extends",
    "finally",
    "for",
    "from",
    "function",
    "get",
    "if",
    "import",
    "in",
    "instanceof",
    "let",
    "new",
    "of",
    "return",
    "set",
    "static",
    "super",
    "switch",
    "this",
    "throw",
    "try",
    "typeof",
    "var",
    "void",
    "while",
    "with",
    "yield",
    "abstract",
    "as",
    "declare",
    "enum",
    "implements",
    "interface",
    "keyof",
    "namespace",
    "private",
    "protected",
    "public",
    "readonly",
    "type",
    "any|",
    "boolean|",
    "false|",
    "never|",
    "null|",
    "number|",
    "string|",
    "symbol|",
    "true|",
    "undefined|",
    "unknown|",
    "void|",
];

static KW_JAVA: &[&str] = &[
    "abstract",
    "assert",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "interface",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "volatile",
    "while",
    "boolean|",
    "byte|",
    "char|",
    "double|",
    "false|",
    "float|",
    "int|",
    "long|",
    "null|",
    "short|",
    "true|",
    "void|",
];

static KW_CSHARP: &[&str] = &[
    "abstract",
    "as",
    "base",
    "break",
    "case",
    "catch",
    "checked",
    "class",
    "const",
    "continue",
    "default",
    "delegate",
    "do",
    "else",
    "enum",
    "event",
    "explicit",
    "extern",
    "finally",
    "fixed",
    "for",
    "foreach",
    "goto",
    "if",
    "implicit",
    "in",
    "interface",
    "internal",
    "is",
    "lock",
    "namespace",
    "new",
    "operator",
    "out",
    "override",
    "params",
    "private",
    "protected",
    "public",
    "readonly",
    "ref",
    "return",
    "sealed",
    "sizeof",
    "stackalloc",
    "static",
    "struct",
    "switch",
    "this",
    "throw",
    "try",
    "typeof",
    "unchecked",
    "unsafe",
    "using",
    "virtual",
    "volatile",
    "while",
    "async",
    "await",
    "get",
    "init",
    "record",
    "set",
    "var",
    "bool|",
    "byte|",
    "char|",
    "decimal|",
    "double|",
    "false|",
    "float|",
    "int|",
    "long|",
    "null|",
    "object|",
    "sbyte|",
    "short|",
    "string|",
    "true|",
    "uint|",
    "ulong|",
    "ushort|",
    "void|",
];

static KW_GO: &[&str] = &[
    "break",
    "case",
    "chan",
    "const",
    "continue",
    "default",
    "defer",
    "else",
    "fallthrough",
    "for",
    "func",
    "go",
    "goto",
    "if",
    "import",
    "interface",
    "map",
    "package",
    "range",
    "return",
    "select",
    "struct",
    "switch",
    "type",
    "var",
    "bool|",
    "byte|",
    "complex64|",
    "complex128|",
    "error|",
    "false|",
    "float32|",
    "float64|",
    "int|",
    "int8|",
    "int16|",
    "int32|",
    "int64|",
    "nil|",
    "rune|",
    "string|",
    "true|",
    "uint|",
    "uint8|",
    "uint16|",
    "uint32|",
    "uint64|",
    "uintptr|",
];

static KW_RUST: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
    "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
    "return", "self", "Self", "static", "struct", "super", "trait", "type", "unsafe", "use",
    "where", "while", "bool|", "char|", "false|", "f32|", "f64|", "i8|", "i16|", "i32|", "i64|",
    "i128|", "isize|", "str|", "String|", "true|", "u8|", "u16|", "u32|", "u64|", "u128|",
    "usize|",
];

static KW_SHELL: &[&str] = &[
    "case", "do", "done", "elif", "else", "esac", "fi", "for", "function", "if", "in", "select",
    "then", "time", "until", "while", "break", "continue", "return", "export", "local", "readonly",
    "source", "test", "true|", "false|", "echo|", "printf|", "cd|", "pwd|", "read|", "set|",
    "unset|", "shift|",
];

static KW_SQL: &[&str] = &[
    "add",
    "alter",
    "and",
    "as",
    "asc",
    "between",
    "by",
    "case",
    "check",
    "column",
    "constraint",
    "create",
    "delete",
    "desc",
    "distinct",
    "drop",
    "else",
    "end",
    "exists",
    "foreign",
    "from",
    "group",
    "having",
    "in",
    "index",
    "insert",
    "into",
    "is",
    "join",
    "key",
    "left",
    "like",
    "limit",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "primary",
    "references",
    "right",
    "select",
    "set",
    "table",
    "then",
    "union",
    "unique",
    "update",
    "values",
    "view",
    "where",
    "bigint|",
    "boolean|",
    "date|",
    "decimal|",
    "false|",
    "int|",
    "integer|",
    "numeric|",
    "real|",
    "text|",
    "timestamp|",
    "true|",
    "varchar|",
];

static KW_RUBY: &[&str] = &[
    "BEGIN", "END", "alias", "and", "begin", "break", "case", "class", "def", "defined?", "do",
    "else", "elsif", "end", "ensure", "for", "if", "in", "module", "next", "not", "or", "redo",
    "rescue", "retry", "return", "self", "super", "then", "undef", "unless", "until", "when",
    "while", "yield", "false|", "nil|", "true|",
];

static KW_PHP: &[&str] = &[
    "abstract",
    "and",
    "array",
    "as",
    "break",
    "callable",
    "case",
    "catch",
    "class",
    "clone",
    "const",
    "continue",
    "declare",
    "default",
    "die",
    "do",
    "echo",
    "else",
    "elseif",
    "empty",
    "enddeclare",
    "endfor",
    "endforeach",
    "endif",
    "endswitch",
    "endwhile",
    "eval",
    "exit",
    "extends",
    "final",
    "finally",
    "fn",
    "for",
    "foreach",
    "function",
    "global",
    "goto",
    "if",
    "implements",
    "include",
    "include_once",
    "instanceof",
    "insteadof",
    "interface",
    "isset",
    "list",
    "match",
    "namespace",
    "new",
    "or",
    "print",
    "private",
    "protected",
    "public",
    "readonly",
    "require",
    "require_once",
    "return",
    "static",
    "switch",
    "throw",
    "trait",
    "try",
    "unset",
    "use",
    "var",
    "while",
    "xor",
    "bool|",
    "false|",
    "float|",
    "int|",
    "null|",
    "string|",
    "true|",
    "void|",
];

static KW_SWIFT: &[&str] = &[
    "actor",
    "as",
    "associatedtype",
    "async",
    "await",
    "break",
    "case",
    "catch",
    "class",
    "continue",
    "default",
    "defer",
    "do",
    "else",
    "enum",
    "extension",
    "fallthrough",
    "for",
    "func",
    "guard",
    "if",
    "import",
    "in",
    "init",
    "inout",
    "is",
    "let",
    "nonisolated",
    "operator",
    "private",
    "protocol",
    "public",
    "repeat",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "subscript",
    "super",
    "switch",
    "throw",
    "throws",
    "try",
    "typealias",
    "var",
    "where",
    "while",
    "Any|",
    "Bool|",
    "Double|",
    "false|",
    "Float|",
    "Int|",
    "nil|",
    "String|",
    "true|",
    "Void|",
];

static KW_KOTLIN: &[&str] = &[
    "as",
    "break",
    "class",
    "continue",
    "do",
    "else",
    "false",
    "for",
    "fun",
    "if",
    "in",
    "interface",
    "is",
    "null",
    "object",
    "package",
    "return",
    "super",
    "this",
    "throw",
    "true",
    "try",
    "typealias",
    "typeof",
    "val",
    "var",
    "when",
    "while",
    "actual",
    "annotation",
    "by",
    "catch",
    "companion",
    "const",
    "constructor",
    "crossinline",
    "data",
    "enum",
    "expect",
    "external",
    "final",
    "finally",
    "import",
    "infix",
    "init",
    "inline",
    "inner",
    "internal",
    "lateinit",
    "noinline",
    "open",
    "operator",
    "out",
    "override",
    "private",
    "protected",
    "public",
    "reified",
    "sealed",
    "suspend",
    "tailrec",
    "vararg",
    "Any|",
    "Boolean|",
    "Byte|",
    "Char|",
    "Double|",
    "Float|",
    "Int|",
    "Long|",
    "Short|",
    "String|",
    "Unit|",
];

static KW_ZIG: &[&str] = &[
    "addrspace",
    "align",
    "allowzero",
    "and",
    "anyframe",
    "anytype",
    "asm",
    "async",
    "await",
    "break",
    "callconv",
    "catch",
    "comptime",
    "const",
    "continue",
    "defer",
    "else",
    "enum",
    "errdefer",
    "error",
    "export",
    "extern",
    "fn",
    "for",
    "if",
    "inline",
    "linksection",
    "noalias",
    "noinline",
    "nosuspend",
    "opaque",
    "or",
    "orelse",
    "packed",
    "pub",
    "resume",
    "return",
    "struct",
    "suspend",
    "switch",
    "test",
    "threadlocal",
    "try",
    "union",
    "unreachable",
    "usingnamespace",
    "var",
    "volatile",
    "while",
    "bool|",
    "false|",
    "f32|",
    "f64|",
    "i32|",
    "i64|",
    "null|",
    "true|",
    "u8|",
    "u16|",
    "u32|",
    "u64|",
    "usize|",
    "void|",
];

static KW_LUA: &[&str] = &[
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto", "if", "in",
    "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while",
];

static KW_HTML: &[&str] = &[
    "a", "body", "button", "div", "doctype", "form", "h1", "h2", "h3", "head", "html", "input",
    "label", "li", "link", "main", "meta", "ol", "option", "p", "script", "section", "select",
    "span", "style", "table", "tbody", "td", "th", "thead", "title", "tr", "ul", "class|", "href|",
    "id|", "name|", "rel|", "src|", "type|", "value|",
];

static KW_CSS: &[&str] = &[
    "align-items",
    "background",
    "border",
    "bottom",
    "color",
    "display",
    "flex",
    "font",
    "font-size",
    "gap",
    "grid",
    "height",
    "justify-content",
    "left",
    "margin",
    "max-width",
    "min-width",
    "padding",
    "position",
    "right",
    "top",
    "transform",
    "width",
    "z-index",
    "absolute|",
    "auto|",
    "block|",
    "flex|",
    "grid|",
    "hidden|",
    "inline|",
    "none|",
    "relative|",
    "solid|",
];

static SYNTAXES: &[Syntax] = &[
    Syntax {
        name: "generic",
        aliases: "text txt",
        keywords: KW_GENERIC,
        singleline_comments: &["//", "#"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS | SYNTAX_BACKTICK_STRINGS,
    },
    Syntax {
        name: "c",
        aliases: "c h cpp c++ cc cxx hpp hxx objc objective-c",
        keywords: KW_C,
        singleline_comments: &["//"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "python",
        aliases: "py python py3",
        keywords: KW_PYTHON,
        singleline_comments: &["#"],
        multiline_start: None,
        multiline_end: None,
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "javascript",
        aliases: "js jsx javascript typescript ts tsx node mjs cjs",
        keywords: KW_JS,
        singleline_comments: &["//"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS | SYNTAX_BACKTICK_STRINGS,
    },
    Syntax {
        name: "java",
        aliases: "java",
        keywords: KW_JAVA,
        singleline_comments: &["//"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "csharp",
        aliases: "cs c# csharp dotnet",
        keywords: KW_CSHARP,
        singleline_comments: &["//"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "go",
        aliases: "go golang",
        keywords: KW_GO,
        singleline_comments: &["//"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS | SYNTAX_BACKTICK_STRINGS,
    },
    Syntax {
        name: "rust",
        aliases: "rs rust",
        keywords: KW_RUST,
        singleline_comments: &["//"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "shell",
        aliases: "sh bash zsh shell fish ksh",
        keywords: KW_SHELL,
        singleline_comments: &["#"],
        multiline_start: None,
        multiline_end: None,
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS | SYNTAX_BACKTICK_STRINGS,
    },
    Syntax {
        name: "sql",
        aliases: "sql postgres mysql sqlite",
        keywords: KW_SQL,
        singleline_comments: &["--"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS | SYNTAX_CASE_INSENSITIVE,
    },
    Syntax {
        name: "ruby",
        aliases: "rb ruby",
        keywords: KW_RUBY,
        singleline_comments: &["#"],
        multiline_start: None,
        multiline_end: None,
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "php",
        aliases: "php",
        keywords: KW_PHP,
        singleline_comments: &["//", "#"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "swift",
        aliases: "swift",
        keywords: KW_SWIFT,
        singleline_comments: &["//"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "kotlin",
        aliases: "kt kts kotlin",
        keywords: KW_KOTLIN,
        singleline_comments: &["//"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "zig",
        aliases: "zig",
        keywords: KW_ZIG,
        singleline_comments: &["//"],
        multiline_start: None,
        multiline_end: None,
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "lua",
        aliases: "lua",
        keywords: KW_LUA,
        singleline_comments: &["--"],
        multiline_start: None,
        multiline_end: None,
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "html",
        aliases: "html htm xml svg",
        keywords: KW_HTML,
        singleline_comments: &[],
        multiline_start: Some("<!--"),
        multiline_end: Some("-->"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "css",
        aliases: "css scss sass",
        keywords: KW_CSS,
        singleline_comments: &[],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "json",
        aliases: "json jsonc",
        keywords: &[],
        singleline_comments: &["//"],
        multiline_start: Some("/*"),
        multiline_end: Some("*/"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "yaml",
        aliases: "yaml yml toml ini",
        keywords: &[],
        singleline_comments: &["#"],
        multiline_start: None,
        multiline_end: None,
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
    Syntax {
        name: "markdown",
        aliases: "md markdown",
        keywords: KW_GENERIC,
        singleline_comments: &[],
        multiline_start: Some("<!--"),
        multiline_end: Some("-->"),
        flags: SYNTAX_NUMBERS | SYNTAX_STRINGS,
    },
];

/// Looks up a syntax entry by language name or alias, defaulting to generic.
#[must_use]
pub fn syntax_for_lang(lang: &str) -> &'static Syntax {
    if !lang.is_empty() {
        for s in SYNTAXES {
            if s.name.eq_ignore_ascii_case(lang)
                || s.aliases
                    .split(' ')
                    .any(|a| !a.is_empty() && a.eq_ignore_ascii_case(lang))
            {
                return s;
            }
        }
    }
    &SYNTAXES[0]
}

/// Looks up a syntax entry from a file path's basename and extension.
#[must_use]
pub fn syntax_for_path(path: &str) -> &'static Syntax {
    if path.is_empty() {
        return syntax_for_lang("");
    }
    let base = path.rsplit('/').next().unwrap_or(path);
    if base.eq_ignore_ascii_case("Dockerfile") || base.eq_ignore_ascii_case("Makefile") {
        return syntax_for_lang("sh");
    }
    match base.rfind('.') {
        Some(dot) if dot + 1 < base.len() => syntax_for_lang(&base[dot + 1..]),
        _ => syntax_for_lang(""),
    }
}

fn syntax_separator(c: u8) -> bool {
    c == 0 || c.is_ascii_whitespace() || b",.()+-/*=~%[]{}<>:;!&|^?".contains(&c)
}

fn syntax_color(hl: Highlight) -> u16 {
    match hl {
        Highlight::Comment => 244,
        Highlight::Keyword1 => 214,
        Highlight::Keyword2 => 81,
        Highlight::String => 150,
        Highlight::Number => 203,
        Highlight::Normal => 252,
    }
}

fn keyword_len(kw: &str) -> (usize, bool) {
    if let Some(stripped) = kw.strip_suffix('|') {
        (stripped.len(), true)
    } else {
        (kw.len(), false)
    }
}

fn bytes_eq_ci(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.eq_ignore_ascii_case(y))
}

fn match_keyword(syn: &Syntax, rest: &[u8]) -> Option<(usize, Highlight)> {
    for kw in syn.keywords {
        let (klen, secondary) = keyword_len(kw);
        if rest.len() < klen {
            continue;
        }
        let kbytes = &kw.as_bytes()[..klen];
        let matched = if syn.flags & SYNTAX_CASE_INSENSITIVE != 0 {
            bytes_eq_ci(&rest[..klen], kbytes)
        } else {
            &rest[..klen] == kbytes
        };
        if !matched {
            continue;
        }
        let follow = rest.get(klen).copied().unwrap_or(0);
        if !syntax_separator(follow) {
            continue;
        }
        let hl = if secondary {
            Highlight::Keyword2
        } else {
            Highlight::Keyword1
        };
        return Some((klen, hl));
    }
    None
}

fn number_len(rest: &[u8]) -> usize {
    rest.iter()
        .take_while(|&&c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'+' | b'-'))
        .count()
}

// ---------------------------------------------------------------------------
// Token renderer
// ---------------------------------------------------------------------------

/// Options controlling the token renderer's output.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderOptions {
    /// Emit ANSI color and attribute sequences.
    pub use_color: bool,
    /// Interpret `<think>`/`</think>` tags and dim thinking text.
    pub format_thinking: bool,
    /// Interpret markdown cues (bold, italic, inline code, fences).
    pub format_markdown: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MdPending {
    None,
    Star,
    Backtick,
}

const UPTO_MARKER: &[u8] = b"[upto]";
const FENCE_LANG_MAX: usize = 31;

/// SGR for thinking text: a barely-visible dark gray (256-color index 238).
const THINK_GREY: &[u8] = b"\x1b[38;5;238m";

/// Streaming markdown-aware terminal renderer for assistant output.
///
/// Port of `agent_token_renderer`: bold/italic/inline code, fenced code
/// blocks with keyword highlighting, grey thinking text, and UTF-8-safe
/// byte-at-a-time streaming.
#[allow(clippy::struct_excessive_bools)]
pub struct TokenRenderer<W: Write> {
    sink: W,
    opts: RenderOptions,
    capture: Option<TailCapture>,

    in_think: bool,
    color_open: bool,
    last_output_newline: bool,
    wrote_visible_output: bool,

    md_bold: bool,
    md_italic: bool,
    md_inline_code: bool,
    md_code_block: bool,
    md_fence_info: bool,
    md_code_line_start: bool,
    md_code_in_ml_comment: bool,
    md_syntax_silent: bool,
    md_syntax_has_highlight: bool,
    md_pending: MdPending,
    md_pending_len: usize,
    md_syntax: Option<&'static Syntax>,
    md_fence_lang: String,
    md_code_line_prefix: Option<String>,
    md_code_line_prefix_color: Option<String>,
    md_code_highlight_upto: bool,
    md_code_line: Vec<u8>,

    pending: Vec<u8>,
    utf8_pending: [u8; 4],
    utf8_pending_len: usize,
    utf8_pending_need: usize,
}

impl<W: Write> fmt::Debug for TokenRenderer<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenRenderer")
            .field("opts", &self.opts)
            .field("in_think", &self.in_think)
            .field("md_code_block", &self.md_code_block)
            .field("wrote_visible_output", &self.wrote_visible_output)
            .field("last_output_newline", &self.last_output_newline)
            .finish_non_exhaustive()
    }
}

impl<W: Write> TokenRenderer<W> {
    /// Creates a renderer writing to `sink` with the given options.
    pub fn new(sink: W, opts: RenderOptions) -> Self {
        Self {
            sink,
            opts,
            capture: None,
            in_think: false,
            color_open: false,
            last_output_newline: false,
            wrote_visible_output: false,
            md_bold: false,
            md_italic: false,
            md_inline_code: false,
            md_code_block: false,
            md_fence_info: false,
            md_code_line_start: false,
            md_code_in_ml_comment: false,
            md_syntax_silent: false,
            md_syntax_has_highlight: false,
            md_pending: MdPending::None,
            md_pending_len: 0,
            md_syntax: None,
            md_fence_lang: String::new(),
            md_code_line_prefix: None,
            md_code_line_prefix_color: None,
            md_code_highlight_upto: false,
            md_code_line: Vec::new(),
            pending: Vec::new(),
            utf8_pending: [0; 4],
            utf8_pending_len: 0,
            utf8_pending_need: 0,
        }
    }

    /// Attaches a tail capture; output is recorded instead of written.
    pub fn set_capture(&mut self, capture: Option<TailCapture>) {
        self.capture = capture;
    }

    /// Detaches and returns the tail capture, if any.
    pub fn take_capture(&mut self) -> Option<TailCapture> {
        self.capture.take()
    }

    /// Returns `true` if any visible byte has been emitted.
    #[must_use]
    pub fn wrote_visible_output(&self) -> bool {
        self.wrote_visible_output
    }

    /// Returns `true` if the last emitted output byte was a newline.
    #[must_use]
    pub fn last_output_newline(&self) -> bool {
        self.last_output_newline
    }

    /// Sets thinking mode: grey text, markdown disabled.
    pub fn set_in_think(&mut self, in_think: bool) {
        self.in_think = in_think;
    }

    /// Streams a chunk of assistant text through the renderer.
    pub fn write(&mut self, text: &str) {
        self.write_bytes(text.as_bytes());
    }

    /// Streams raw bytes; UTF-8 sequences may be split across calls.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        if self.opts.format_thinking {
            self.process(bytes, false);
        } else {
            for &b in bytes {
                self.write_char(b);
            }
        }
    }

    /// Flushes pending state and emits the trailing blank line.
    pub fn finish(&mut self) {
        if self.opts.format_thinking {
            self.process(&[], true);
        }
        self.markdown_finish();
        self.flush_utf8();
        self.reset_color();
        if self.wrote_visible_output {
            if !self.last_output_newline {
                self.out(b"\n");
            }
            self.out(b"\n");
            self.last_output_newline = true;
        }
        let _ = self.sink.flush();
    }

    /// Emits a raw color escape, tracking whether a manual color is open.
    pub fn color(&mut self, seq: &str) {
        self.markdown_emit_pending_literals();
        self.flush_utf8();
        let reset = seq.is_empty() || seq == "\x1b[0m";
        if self.opts.use_color && !seq.is_empty() {
            self.out_str(seq);
        }
        self.color_open = self.opts.use_color && !reset;
    }

    /// Emits text verbatim, bypassing markdown but flushing pending state.
    pub fn plain(&mut self, s: &str) {
        self.markdown_emit_pending_literals();
        self.flush_utf8();
        self.out_str(s);
        if !s.is_empty() {
            self.wrote_visible_output = true;
            self.last_output_newline = s.ends_with('\n');
        }
    }

    /// Re-applies tracked text attributes after external output.
    pub fn restore_text_attrs(&mut self) {
        if !self.opts.use_color || !self.color_open || !self.has_text_attrs() {
            return;
        }
        self.set_text_attrs();
    }

    /// Enters code-block streaming mode with an explicit syntax.
    pub fn code_stream_begin(&mut self, syntax: &'static Syntax) {
        self.reset_color();
        self.md_code_block = true;
        self.md_inline_code = false;
        self.md_fence_info = false;
        self.md_code_line_start = true;
        self.md_code_in_ml_comment = false;
        self.md_syntax = Some(syntax);
        self.md_fence_lang.clear();
        self.md_code_line_prefix = None;
        self.md_code_line_prefix_color = None;
        self.md_code_highlight_upto = false;
        self.md_code_line.clear();
    }

    /// Sets a per-line prefix (and its color) restored on repaint.
    pub fn code_stream_set_prefix(&mut self, prefix: Option<&str>, color: Option<&str>) {
        self.md_code_line_prefix = prefix.map(str::to_owned);
        self.md_code_line_prefix_color = color.map(str::to_owned);
    }

    /// Enables highlighting of the literal `[upto]` marker in code lines.
    pub fn code_stream_set_upto_marker(&mut self, enabled: bool) {
        self.md_code_highlight_upto = enabled;
    }

    /// Ends code-block streaming mode, emitting any buffered line.
    pub fn code_stream_end(&mut self) {
        self.code_end();
    }

    // -- raw output ---------------------------------------------------------

    fn out(&mut self, s: &[u8]) {
        if let Some(c) = self.capture.as_mut() {
            c.append(s);
        } else {
            let _ = self.sink.write_all(s);
        }
    }

    fn out_str(&mut self, s: &str) {
        self.out(s.as_bytes());
    }

    fn set_grey(&mut self) {
        if self.opts.use_color {
            // Barely-visible dark gray so thinking text reads as background
            // muttering, clearly distinct from the assistant's real output.
            self.out(THINK_GREY);
        }
    }

    fn reset_color(&mut self) {
        if self.opts.use_color {
            self.out(b"\x1b[0m");
        }
        self.color_open = false;
    }

    fn has_text_attrs(&self) -> bool {
        self.in_think || self.md_bold || self.md_italic || self.md_inline_code || self.md_code_block
    }

    fn set_text_attrs(&mut self) {
        if !self.opts.use_color {
            return;
        }
        if self.in_think {
            self.set_grey();
            return;
        }
        if self.md_code_block {
            self.out(b"\x1b[38;5;75m");
            return;
        } else if self.md_inline_code {
            self.out(b"\x1b[36m");
        }
        if self.md_bold {
            self.out(b"\x1b[1m");
        }
        if self.md_italic {
            self.out(b"\x1b[3m");
        }
    }

    fn write_complete_char_raw(&mut self, s: &[u8]) {
        let styled = self.opts.use_color && self.has_text_attrs();
        if styled && !self.color_open {
            self.set_text_attrs();
            self.color_open = true;
        } else if !styled && self.color_open {
            self.reset_color();
        }
        self.out(s);
        if !s.is_empty() {
            self.wrote_visible_output = true;
        }
        self.last_output_newline = s == b"\n";
    }

    fn flush_utf8(&mut self) {
        if self.utf8_pending_len == 0 {
            return;
        }
        let buf: [u8; 4] = self.utf8_pending;
        let len = self.utf8_pending_len;
        self.write_complete_char_raw(&buf[..len]);
        self.utf8_pending_len = 0;
        self.utf8_pending_need = 0;
    }

    fn utf8_need(c: u8) -> usize {
        match c {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => 1,
        }
    }

    fn write_char_raw(&mut self, c: u8) {
        if self.utf8_pending_len > 0 {
            if c & 0xc0 == 0x80 && self.utf8_pending_len < self.utf8_pending.len() {
                self.utf8_pending[self.utf8_pending_len] = c;
                self.utf8_pending_len += 1;
                if self.utf8_pending_len == self.utf8_pending_need {
                    self.flush_utf8();
                }
                return;
            }
            self.flush_utf8();
        }

        let need = Self::utf8_need(c);
        if need == 1 {
            self.write_complete_char_raw(&[c]);
            return;
        }
        self.utf8_pending[0] = c;
        self.utf8_pending_len = 1;
        self.utf8_pending_need = need;
    }

    /// Writes one byte with markdown attributes temporarily disabled.
    fn write_plain_byte(&mut self, c: u8) {
        let (bold, italic, inline, block) = (
            self.md_bold,
            self.md_italic,
            self.md_inline_code,
            self.md_code_block,
        );
        self.md_bold = false;
        self.md_italic = false;
        self.md_inline_code = false;
        self.md_code_block = false;
        self.write_char_raw(c);
        self.md_bold = bold;
        self.md_italic = italic;
        self.md_inline_code = inline;
        self.md_code_block = block;
    }

    // -- syntax highlighting ------------------------------------------------

    fn syntax_write(&mut self, hl: Highlight, s: &[u8]) {
        if s.is_empty() {
            return;
        }
        if hl != Highlight::Normal {
            self.md_syntax_has_highlight = true;
        }
        if self.md_syntax_silent {
            return;
        }
        if self.opts.use_color && hl != Highlight::Normal {
            let seq = format!("\x1b[38;5;{}m", syntax_color(hl));
            self.out_str(&seq);
        }
        self.out(s);
        if self.opts.use_color && hl != Highlight::Normal {
            self.out(b"\x1b[0m");
        }
        self.wrote_visible_output = true;
        self.last_output_newline = false;
    }

    fn syntax_write_upto_marker(&mut self) {
        self.md_syntax_has_highlight = true;
        if self.md_syntax_silent {
            return;
        }
        if self.opts.use_color {
            self.out(b"\x1b[38;5;244m[");
            self.out(b"\x1b[1;38;5;177mupto");
            self.out(b"\x1b[38;5;244m]\x1b[0m");
        } else {
            self.out(UPTO_MARKER);
        }
        self.wrote_visible_output = true;
        self.last_output_newline = false;
    }

    #[allow(clippy::too_many_lines)]
    fn syntax_emit_line(&mut self, line: &[u8]) {
        let syn = self.md_syntax.unwrap_or_else(|| syntax_for_lang(""));
        let mut i = 0;
        let end = line.len();
        let mut prev_sep = true;
        let mut prev_hl = Highlight::Normal;

        while i < end {
            let rest = &line[i..];

            if self.md_code_highlight_upto && rest.starts_with(UPTO_MARKER) {
                self.syntax_write_upto_marker();
                i += UPTO_MARKER.len();
                prev_sep = true;
                prev_hl = Highlight::Normal;
                continue;
            }

            if self.md_code_in_ml_comment {
                if let Some(mce) = syn.multiline_end
                    && let Some(pos) = find_sub(rest, mce.as_bytes())
                {
                    let take = pos + mce.len();
                    let seg = &line[i..i + take];
                    self.syntax_write(Highlight::Comment, seg);
                    i += take;
                    self.md_code_in_ml_comment = false;
                    prev_sep = true;
                    prev_hl = Highlight::Comment;
                    continue;
                }
                let seg = &line[i..];
                self.syntax_write(Highlight::Comment, seg);
                return;
            }

            if syn
                .singleline_comments
                .iter()
                .any(|m| rest.starts_with(m.as_bytes()))
            {
                let seg = &line[i..];
                self.syntax_write(Highlight::Comment, seg);
                return;
            }

            if let (Some(mls), Some(mle)) = (syn.multiline_start, syn.multiline_end)
                && rest.starts_with(mls.as_bytes())
            {
                let body = &rest[mls.len()..];
                let take = if let Some(pos) = find_sub(body, mle.as_bytes()) {
                    mls.len() + pos + mle.len()
                } else {
                    self.md_code_in_ml_comment = true;
                    rest.len()
                };
                let seg = &line[i..i + take];
                self.syntax_write(Highlight::Comment, seg);
                i += take;
                prev_sep = false;
                prev_hl = Highlight::Comment;
                continue;
            }

            let c = rest[0];
            if syn.flags & SYNTAX_STRINGS != 0
                && (c == b'"'
                    || c == b'\''
                    || (syn.flags & SYNTAX_BACKTICK_STRINGS != 0 && c == b'`'))
            {
                let quote = c;
                let mut q = 1;
                while q < rest.len() {
                    if rest[q] == b'\\' && q + 1 < rest.len() {
                        q += 2;
                        continue;
                    }
                    q += 1;
                    if rest[q - 1] == quote {
                        break;
                    }
                }
                let seg = &line[i..i + q];
                self.syntax_write(Highlight::String, seg);
                i += q;
                prev_sep = false;
                prev_hl = Highlight::String;
                continue;
            }

            let number_start = c.is_ascii_digit() && (prev_sep || prev_hl == Highlight::Number)
                || (c == b'.' && i > 0 && prev_hl == Highlight::Number);
            if syn.flags & SYNTAX_NUMBERS != 0 && number_start {
                let nlen = number_len(rest);
                let seg = &line[i..i + nlen];
                self.syntax_write(Highlight::Number, seg);
                i += nlen;
                prev_sep = false;
                prev_hl = Highlight::Number;
                continue;
            }

            if prev_sep && let Some((klen, khl)) = match_keyword(syn, rest) {
                let seg = &line[i..i + klen];
                self.syntax_write(khl, seg);
                i += klen;
                prev_sep = false;
                prev_hl = khl;
                continue;
            }

            let seg = &line[i..=i];
            self.syntax_write(Highlight::Normal, seg);
            prev_sep = syntax_separator(c);
            prev_hl = Highlight::Normal;
            i += 1;
        }
    }

    // -- code block line buffering / repaint ---------------------------------

    fn terminal_cols() -> usize {
        // Deviation from the C reference: the sink is a generic writer, so we
        // cannot ioctl(TIOCGWINSZ); assume the classic 80-column default.
        80
    }

    fn code_line_can_repaint(&self) -> bool {
        if !self.opts.use_color || self.capture.is_some() || self.md_code_line.is_empty() {
            return false;
        }
        let cols = Self::terminal_cols();
        let prefix_len = self.md_code_line_prefix.as_ref().map_or(0, String::len);
        if cols <= 1 || prefix_len + self.md_code_line.len() >= cols {
            return false;
        }
        self.md_code_line
            .iter()
            .all(|&c| c == b'\r' || (0x20..0x80).contains(&c) && c != 0x1b)
    }

    fn code_write_line_prefix(&mut self) {
        let Some(prefix) = self.md_code_line_prefix.clone() else {
            return;
        };
        let color = self.md_code_line_prefix_color.clone();
        if self.opts.use_color
            && let Some(col) = &color
        {
            self.out_str(col);
        }
        self.out_str(&prefix);
        if self.opts.use_color && color.is_some() {
            self.out(b"\x1b[0m");
        }
        self.color_open = false;
    }

    /// Runs the highlighter silently to learn whether repainting would change
    /// the line, preserving multiline-comment state for the caller.
    fn code_scan_line(&mut self) -> (bool, bool) {
        let old_silent = self.md_syntax_silent;
        let old_highlight = self.md_syntax_has_highlight;
        let old_ml = self.md_code_in_ml_comment;

        self.md_syntax_silent = true;
        self.md_syntax_has_highlight = false;
        let line = std::mem::take(&mut self.md_code_line);
        self.syntax_emit_line(&line);
        self.md_code_line = line;
        let changed = self.md_syntax_has_highlight;
        let final_ml = self.md_code_in_ml_comment;

        self.md_code_in_ml_comment = old_ml;
        self.md_syntax_silent = old_silent;
        self.md_syntax_has_highlight = old_highlight;
        (changed, final_ml)
    }

    fn code_emit_buffered_line(&mut self, with_newline: bool) {
        let (changed, final_ml) = self.code_scan_line();
        let repaint = changed && self.code_line_can_repaint();
        if repaint {
            self.reset_color();
            self.out(b"\r\x1b[0K");
            self.code_write_line_prefix();
            let line = std::mem::take(&mut self.md_code_line);
            self.syntax_emit_line(&line);
            self.md_code_line = line;
        } else {
            self.md_code_in_ml_comment = final_ml;
        }
        self.md_code_line.clear();
        if with_newline {
            self.write_plain_byte(b'\n');
            self.wrote_visible_output = true;
            self.last_output_newline = true;
            self.md_code_line_start = true;
        }
    }

    fn code_byte(&mut self, c: u8) {
        if c == b'\n' {
            self.code_emit_buffered_line(true);
            return;
        }
        self.md_code_line.push(c);
        self.write_plain_byte(c);
        if c != b' ' && c != b'\t' && c != b'\r' {
            self.md_code_line_start = false;
        }
    }

    fn code_emit_backtick_literals(&mut self, count: usize) {
        for _ in 0..count {
            self.code_byte(b'`');
        }
    }

    fn code_begin(&mut self) {
        self.reset_color();
        self.md_code_block = true;
        self.md_inline_code = false;
        self.md_fence_info = true;
        self.md_code_line_start = true;
        self.md_code_in_ml_comment = false;
        self.md_syntax = Some(syntax_for_lang(""));
        self.md_fence_lang.clear();
        self.md_code_line_prefix = None;
        self.md_code_line_prefix_color = None;
        self.md_code_highlight_upto = false;
        self.md_code_line.clear();
    }

    fn code_end(&mut self) {
        let only_space = self
            .md_code_line
            .iter()
            .all(|&c| c == b' ' || c == b'\t' || c == b'\r');
        if !self.md_code_line.is_empty() && !only_space {
            self.code_emit_buffered_line(false);
        } else {
            self.md_code_line.clear();
        }
        self.md_code_block = false;
        self.md_inline_code = false;
        self.md_fence_info = false;
        self.md_code_line_start = true;
        self.md_code_in_ml_comment = false;
        self.md_syntax = None;
        self.md_fence_lang.clear();
        self.md_code_line_prefix = None;
        self.md_code_line_prefix_color = None;
    }

    // -- markdown state machine ----------------------------------------------

    fn markdown_clear_pending(&mut self) {
        self.md_pending = MdPending::None;
        self.md_pending_len = 0;
    }

    fn markdown_emit_pending_literals(&mut self) {
        let c = match self.md_pending {
            MdPending::Star => b'*',
            MdPending::Backtick => b'`',
            MdPending::None => return,
        };
        let count = self.md_pending_len;
        self.markdown_clear_pending();
        if self.md_code_block {
            if c == b'`' {
                self.code_emit_backtick_literals(count);
            } else {
                for _ in 0..count {
                    self.code_byte(c);
                }
            }
            return;
        }
        for _ in 0..count {
            self.write_char_raw(c);
        }
    }

    fn markdown_commit_backticks(&mut self) {
        let count = self.md_pending_len;
        self.markdown_clear_pending();
        if count >= 3 {
            for _ in 0..count {
                self.write_plain_byte(b'`');
            }
            if self.md_code_block {
                self.code_end();
            } else {
                self.code_begin();
            }
            return;
        }
        if self.md_code_block {
            self.code_emit_backtick_literals(count);
            return;
        }
        // Support both `code` and ``code``.
        self.md_inline_code = !self.md_inline_code;
    }

    fn markdown_feed(&mut self, c: u8) {
        if self.md_fence_info {
            if c == b'\n' {
                if self.md_code_block {
                    self.md_syntax = Some(syntax_for_lang(&self.md_fence_lang.clone()));
                }
                self.write_plain_byte(b'\n');
                self.md_fence_info = false;
            } else if self.md_code_block {
                if self.md_fence_lang.len() < FENCE_LANG_MAX
                    && (c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'+' | b'#'))
                {
                    self.md_fence_lang.push(char::from(c));
                }
                self.write_plain_byte(c);
            }
            return;
        }

        if self.md_pending == MdPending::Backtick {
            if c == b'`' {
                self.md_pending_len += 1;
                return;
            }
            self.markdown_commit_backticks();
            self.markdown_feed(c);
            return;
        }

        if self.md_pending == MdPending::Star {
            self.markdown_clear_pending();
            if !self.md_inline_code && !self.md_code_block && c == b'*' {
                self.md_bold = !self.md_bold;
                return;
            }
            if !self.md_inline_code
                && !self.md_code_block
                && (self.md_italic || !matches!(c, b' ' | b'\t' | b'\r' | b'\n'))
            {
                self.md_italic = !self.md_italic;
                self.markdown_feed(c);
                return;
            }
            self.write_char_raw(b'*');
            self.markdown_feed(c);
            return;
        }

        if c == b'`' && (!self.md_code_block || self.md_code_line_start) {
            self.md_pending = MdPending::Backtick;
            self.md_pending_len = 1;
            return;
        }
        if self.md_code_block {
            self.code_byte(c);
            return;
        }
        if !self.md_inline_code && c == b'*' {
            self.md_pending = MdPending::Star;
            self.md_pending_len = 1;
            return;
        }
        self.write_char_raw(c);
    }

    fn markdown_finish(&mut self) {
        // A closing code fence can be the final bytes of the reply; commit a
        // full fence instead of leaking the literal ``` marker.
        if self.md_pending == MdPending::Backtick && self.md_pending_len >= 3 {
            self.markdown_commit_backticks();
        } else {
            self.markdown_emit_pending_literals();
        }
        if self.md_code_block && !self.md_code_line.is_empty() {
            self.code_emit_buffered_line(false);
        }
        self.md_bold = false;
        self.md_italic = false;
        self.md_inline_code = false;
        self.md_code_block = false;
        self.md_fence_info = false;
        self.md_code_line_start = false;
        self.md_code_in_ml_comment = false;
        self.md_syntax = None;
        self.md_fence_lang.clear();
        self.md_code_line_prefix = None;
        self.md_code_line_prefix_color = None;
        self.md_code_highlight_upto = false;
        self.md_code_line = Vec::new();
    }

    fn write_char(&mut self, c: u8) {
        if !self.opts.format_markdown || self.in_think {
            self.markdown_emit_pending_literals();
            self.write_char_raw(c);
            return;
        }
        self.markdown_feed(c);
    }

    // -- think tag processing --------------------------------------------------

    /// Renders text while hiding `<think>` tags and dimming thinking text,
    /// holding back a partial control tag split across model tokens.
    fn process(&mut self, text: &[u8], finish: bool) {
        const THINK_OPEN: &[u8] = b"<think>";
        const THINK_CLOSE: &[u8] = b"</think>";

        let mut buf = std::mem::take(&mut self.pending);
        buf.extend_from_slice(text);

        let mut i = 0;
        while i < buf.len() {
            let cur = &buf[i..];
            if cur.starts_with(THINK_OPEN) {
                self.in_think = true;
                i += THINK_OPEN.len();
                continue;
            }
            if cur.starts_with(THINK_CLOSE) {
                self.in_think = false;
                self.reset_color();
                if !self.last_output_newline {
                    self.out(b"\n");
                }
                self.out(b"\n");
                self.last_output_newline = true;
                i += THINK_CLOSE.len();
                continue;
            }
            if !finish
                && cur[0] == b'<'
                && (is_partial_prefix(cur, THINK_OPEN) || is_partial_prefix(cur, THINK_CLOSE))
            {
                self.pending = cur.to_vec();
                return;
            }
            self.write_char(cur[0]);
            i += 1;
        }
    }
}

fn is_partial_prefix(p: &[u8], prefix: &[u8]) -> bool {
    p.len() < prefix.len() && prefix.starts_with(p)
}

fn find_sub(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn renderer(opts: RenderOptions) -> TokenRenderer<Vec<u8>> {
        TokenRenderer::new(Vec::new(), opts)
    }

    fn output(r: TokenRenderer<Vec<u8>>) -> String {
        String::from_utf8(r.sink).unwrap()
    }

    const COLOR_MD: RenderOptions = RenderOptions {
        use_color: true,
        format_thinking: false,
        format_markdown: true,
    };

    #[test]
    fn plain_text_passthrough() {
        let mut r = renderer(RenderOptions::default());
        r.write("hello world\n");
        assert!(r.wrote_visible_output());
        assert!(r.last_output_newline());
        r.finish();
        let out = output(r);
        assert!(out.starts_with("hello world\n"));
        assert!(!out.contains('\x1b'));
    }

    #[test]
    fn bold_and_italic_markers() {
        let mut r = renderer(COLOR_MD);
        r.write("a **bold** and *ital* b");
        r.finish();
        let out = output(r);
        assert!(out.contains("\x1b[1mbold"), "bold SGR missing: {out:?}");
        assert!(out.contains("\x1b[3mital"), "italic SGR missing: {out:?}");
        assert!(!out.contains('*'), "markers leaked: {out:?}");
    }

    #[test]
    fn inline_code_cyan() {
        let mut r = renderer(COLOR_MD);
        r.write("run `ls -la` now");
        r.finish();
        let out = output(r);
        assert!(out.contains("\x1b[36mls -la"), "cyan inline code: {out:?}");
        assert!(!out.contains('`'));
    }

    #[test]
    fn fenced_rust_block_keyword_highlighting() {
        let mut r = renderer(COLOR_MD);
        r.write("```rust\nfn main() { let x = 42; }\n```\n");
        r.finish();
        let out = output(r);
        // Repaint replaces the streamed line with highlighted text.
        assert!(out.contains("\r\x1b[0K"), "repaint missing: {out:?}");
        assert!(out.contains("\x1b[38;5;214mfn"), "kw1 'fn': {out:?}");
        assert!(out.contains("\x1b[38;5;214mlet"), "kw1 'let': {out:?}");
        assert!(out.contains("\x1b[38;5;203m42"), "number: {out:?}");
        // As in the C, code streams plain first; the fence markers stay visible.
        assert!(
            out.contains("```") && out.contains("rust\n"),
            "fence line: {out:?}"
        );
    }

    #[test]
    fn thinking_rendered_grey() {
        let mut r = renderer(RenderOptions {
            use_color: true,
            format_thinking: true,
            format_markdown: true,
        });
        r.write("<think>pondering</think>answer");
        r.finish();
        let out = output(r);
        assert!(
            out.contains("\x1b[38;5;238mpondering"),
            "grey think: {out:?}"
        );
        assert!(!out.contains("<think>"));
        assert!(!out.contains("</think>"));
        assert!(out.contains("answer"));
    }

    #[test]
    fn partial_think_tag_held_across_writes() {
        let mut r = renderer(RenderOptions {
            use_color: false,
            format_thinking: true,
            format_markdown: false,
        });
        r.write("<thi");
        r.write("nk>x</thi");
        r.write("nk>y");
        r.finish();
        let out = output(r);
        assert!(!out.contains('<'));
        assert!(out.contains('x'));
        assert!(out.contains('y'));
    }

    #[test]
    fn utf8_split_across_writes() {
        let mut r = renderer(RenderOptions::default());
        let euro = "€".as_bytes(); // three bytes
        r.write_bytes(&euro[..1]);
        r.write_bytes(&euro[1..2]);
        r.write_bytes(&euro[2..]);
        r.finish();
        let out = output(r);
        assert!(out.starts_with('€'), "utf-8 reassembly failed: {out:?}");
    }

    #[test]
    fn tail_capture_records_last_bytes() {
        let mut cap = TailCapture::new(8);
        cap.append(b"0123456789abcdef");
        assert_eq!(cap.total(), 16);
        assert_eq!(cap.len(), 8);
        let taken = cap.take();
        assert_eq!(taken, b"89abcdef");
        assert!(cap.is_empty());

        // Attached to a renderer, output goes to the capture, not the sink.
        let mut r = renderer(RenderOptions::default());
        r.set_capture(Some(TailCapture::new(64)));
        r.write("captured text");
        let mut got = r.take_capture().unwrap();
        assert_eq!(got.take(), b"captured text");
        assert!(r.sink.is_empty());
    }

    #[test]
    fn syntax_lookup_by_lang_and_path() {
        assert_eq!(syntax_for_lang("rs").name(), "rust");
        assert_eq!(syntax_for_lang("TypeScript").name(), "javascript");
        assert_eq!(syntax_for_lang("nosuchlang").name(), "generic");
        assert_eq!(syntax_for_path("src/main.rs").name(), "rust");
        assert_eq!(syntax_for_path("Dockerfile").name(), "shell");
        assert_eq!(syntax_for_path("a/b/Makefile").name(), "shell");
        assert_eq!(syntax_for_path("noext").name(), "generic");
    }
}
