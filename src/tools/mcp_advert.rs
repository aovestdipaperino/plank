// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Last-known-good MCP tool advertisements, cached per global server.
//!
//! Tier 1 of the KV hierarchy is keyed on the system prompt text, and that text
//! includes the tool schemas, resource-derived tool schemas, and server
//! instructions of every MCP server that completed a handshake. A single global
//! server failing to start therefore changes `fp1` and forces the most
//! expensive re-prefill there is.
//!
//! This module persists exactly what a server contributes to the prompt — and
//! nothing else — so a failed global server can be rendered from its last
//! successful handshake instead of vanishing. See
//! `docs/superpowers/specs/2026-07-26-global-mcp-advert-cache-design.md`.
//!
//! Scope is global servers only: project-local servers key Tier 2, which is
//! cheap to rebuild, and giving them records would let a project prompt
//! advertise a dead tool for no cache benefit.

use std::path::{Path, PathBuf};

use crate::tools::mcp::{Json, McpResource, McpTool, json_escape, json_parse};

/// Record format version. An unrecognized version parses as absent, so a
/// future format change degrades to pre-cache behavior rather than rendering
/// garbage into the system prompt.
const VERSION: f64 = 1.0;

/// Everything about one MCP server that reaches the system prompt.
#[derive(Debug, Clone)]
pub struct AdvertRecord {
    /// Server name as configured; the `mcp__<server>__<tool>` prefix.
    pub server: String,
    /// The `initialize` free-text instructions block, or empty.
    pub instructions: String,
    /// Advertised tools, in the order they are rendered.
    pub tools: Vec<McpTool>,
    /// Advertised resources. Not rendered directly, but their presence gates
    /// the `mcp_list_resources` / `mcp_read_resource` schemas.
    pub resources: Vec<McpResource>,
}

/// `McpTool` has no `PartialEq` of its own, so `AdvertRecord` compares its
/// prompt-bearing fields by hand rather than deriving.
impl PartialEq for AdvertRecord {
    fn eq(&self, other: &Self) -> bool {
        let tool_eq = |a: &McpTool, b: &McpTool| {
            a.name == b.name
                && a.description == b.description
                && a.schema_json == b.schema_json
                && a.primary == b.primary
        };
        self.server == other.server
            && self.instructions == other.instructions
            && self.resources == other.resources
            && self.tools.len() == other.tools.len()
            && self
                .tools
                .iter()
                .zip(other.tools.iter())
                .all(|(a, b)| tool_eq(a, b))
    }
}

/// Default record root: `~/.plank/mcp-advert`. `None` when `HOME` is unset.
///
/// Every other function in this module takes the root explicitly so tests can
/// use a temp directory without mutating the environment.
#[must_use]
pub fn default_root() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".plank").join("mcp-advert"))
}

/// File name for one server's record: the name with every character outside
/// `[A-Za-z0-9_-]` replaced by `_`, plus a short hash of the original name so
/// two distinct names cannot sanitize onto the same file.
#[must_use]
pub fn record_file_name(server: &str) -> String {
    // `.` is deliberately excluded from the safe set (not just `/`): letting it
    // through would allow a sanitized name like `..` to reach the filesystem.
    let safe: String = server
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let hash = crate::session::sha1_hex(server.as_bytes());
    format!("{safe}-{}.json", &hash[..8])
}

/// Serializes a record. Schema text is stored as a JSON string so its bytes
/// survive verbatim — the prompt embeds `schema_json` unmodified.
#[must_use]
pub fn to_json(rec: &AdvertRecord) -> String {
    let mut out = String::from("{\n  \"version\": 1,\n  \"server\": ");
    json_escape(&mut out, &rec.server);
    out.push_str(",\n  \"instructions\": ");
    json_escape(&mut out, &rec.instructions);
    out.push_str(",\n  \"tools\": [");
    for (i, t) in rec.tools.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("\n    {\"name\": ");
        json_escape(&mut out, &t.name);
        out.push_str(", \"description\": ");
        json_escape(&mut out, &t.description);
        out.push_str(", \"schema\": ");
        json_escape(&mut out, &t.schema_json);
        out.push_str(", \"primary\": ");
        out.push_str(if t.primary { "true" } else { "false" });
        out.push('}');
    }
    out.push_str("\n  ],\n  \"resources\": [");
    for (i, r) in rec.resources.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("\n    {\"uri\": ");
        json_escape(&mut out, &r.uri);
        out.push_str(", \"name\": ");
        json_escape(&mut out, &r.name);
        out.push_str(", \"mime\": ");
        json_escape(&mut out, &r.mime);
        out.push('}');
    }
    out.push_str("\n  ]\n}\n");
    out
}

/// Parses a record. `None` for unparsable text, an unrecognized `version`, or
/// a missing server name — all of which must degrade to "no record".
#[must_use]
pub fn parse(text: &str) -> Option<AdvertRecord> {
    let root = json_parse(text)?;
    if root.get("version") != Some(&Json::Num(VERSION)) {
        return None;
    }
    let server = root.str_or("server", "");
    if server.is_empty() {
        return None;
    }
    let mut tools = Vec::new();
    if let Some(Json::Arr(list)) = root.get("tools") {
        for t in list {
            let name = t.str_or("name", "");
            if name.is_empty() {
                return None;
            }
            tools.push(McpTool {
                name: name.to_string(),
                description: t.str_or("description", "").to_string(),
                schema_json: t.str_or("schema", "").to_string(),
                primary: matches!(t.get("primary"), Some(Json::Bool(true))),
            });
        }
    }
    let mut resources = Vec::new();
    if let Some(Json::Arr(list)) = root.get("resources") {
        for r in list {
            resources.push(McpResource {
                uri: r.str_or("uri", "").to_string(),
                name: r.str_or("name", "").to_string(),
                mime: r.str_or("mime", "").to_string(),
            });
        }
    }
    Some(AdvertRecord {
        server: server.to_string(),
        instructions: root.str_or("instructions", "").to_string(),
        tools,
        resources,
    })
}

/// Loads one server's record. A missing or corrupt file is simply absent.
#[must_use]
pub fn load(root: &Path, server: &str) -> Option<AdvertRecord> {
    let text = std::fs::read_to_string(root.join(record_file_name(server))).ok()?;
    parse(&text)
}

/// Writes a record, skipping the write when the bytes already match.
///
/// Returns whether anything was written, so a steady state does no disk IO.
#[must_use]
pub fn save(root: &Path, rec: &AdvertRecord) -> bool {
    let text = to_json(rec);
    let path = root.join(record_file_name(&rec.server));
    if std::fs::read_to_string(&path).is_ok_and(|old| old == text) {
        return false;
    }
    if std::fs::create_dir_all(root).is_err() {
        return false;
    }
    std::fs::write(&path, text).is_ok()
}

/// Deletes records for servers not in `keep`.
///
/// `keep` must come from the **global** config alone, and only when that config
/// actually parsed — see `global_config_names`. Files we did not create are
/// left alone.
pub fn prune(root: &Path, keep: &[String]) {
    let wanted: Vec<String> = keep.iter().map(|n| record_file_name(n)).collect();
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let is_json = Path::new(name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
        if !is_json || wanted.iter().any(|w| w == name) {
            continue;
        }
        let _ = std::fs::remove_file(entry.path());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, primary: bool) -> McpTool {
        McpTool {
            name: name.to_string(),
            description: "Line one.\nLine two \"quoted\".".to_string(),
            schema_json: "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}}"
                .to_string(),
            primary,
        }
    }

    fn record() -> AdvertRecord {
        AdvertRecord {
            server: "demo".to_string(),
            instructions: "Use demo\ncarefully.".to_string(),
            tools: vec![tool("alpha", true), tool("omega", false)],
            resources: vec![McpResource {
                uri: "file:///x".to_string(),
                name: "x".to_string(),
                mime: "text/plain".to_string(),
            }],
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "plank_advert_{tag}_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).expect("create temp root");
        root
    }

    // The whole feature rests on a record reproducing the prompt-bearing fields
    // exactly: `primary` decides full schema vs. directory entry, and
    // `schema_json` is raw text that must survive byte-for-byte.
    #[test]
    fn record_round_trips_every_prompt_bearing_field() {
        let rec = record();
        let parsed = parse(&to_json(&rec)).expect("parse");
        assert_eq!(parsed, rec);
        assert!(parsed.tools[0].primary);
        assert!(!parsed.tools[1].primary);
    }

    #[test]
    fn unrecognized_version_and_garbage_parse_as_absent() {
        let bumped = to_json(&record()).replace("\"version\": 1", "\"version\": 2");
        assert!(
            parse(&bumped).is_none(),
            "a future version must not be read"
        );
        assert!(parse("{not json").is_none());
        assert!(parse("{\"version\": 1}").is_none(), "no server name");
    }

    #[test]
    fn file_names_are_path_safe_and_collision_free() {
        let a = record_file_name("weird name/../x");
        assert!(
            Path::new(&a)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        );
        assert!(
            !a.contains('/') && !a.contains(".."),
            "must not escape the root: {a}"
        );
        // Sanitizing alone would map these two names onto one file; the hash
        // suffix is what keeps them apart.
        assert_ne!(record_file_name("a/b"), record_file_name("a_b"));
    }

    #[test]
    fn save_then_load_round_trips_and_skips_an_unchanged_write() {
        let root = temp_root("save");
        let rec = record();
        assert!(save(&root, &rec), "first save writes");
        assert_eq!(load(&root, "demo"), Some(rec.clone()));
        assert!(!save(&root, &rec), "an unchanged record must not rewrite");

        let mut changed = rec;
        changed.instructions = "different".to_string();
        assert!(save(&root, &changed), "a changed record writes");
        assert_eq!(load(&root, "demo"), Some(changed));

        assert_eq!(load(&root, "never_saved"), None);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn corrupt_record_on_disk_loads_as_absent() {
        let root = temp_root("corrupt");
        std::fs::write(root.join(record_file_name("demo")), "{truncated").expect("write");
        assert_eq!(load(&root, "demo"), None);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn prune_keeps_named_servers_and_drops_the_rest() {
        let root = temp_root("prune");
        for name in ["keep", "drop"] {
            let mut rec = record();
            rec.server = name.to_string();
            let _ = save(&root, &rec);
        }
        // An unrelated file must survive: the root is ours, but deleting
        // anything we did not recognize would be gratuitous.
        std::fs::write(root.join("notes.txt"), "hi").expect("write");

        prune(&root, &["keep".to_string()]);
        assert!(load(&root, "keep").is_some());
        assert!(load(&root, "drop").is_none());
        assert!(root.join("notes.txt").exists());
        std::fs::remove_dir_all(root).ok();
    }
}
