// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Grid stagings: CSV an MCP server hands to a WASM frame component.
//!
//! An MCP server may return, next to its text, an embedded resource whose URI
//! is `plank-frame://<component id>/<file>` with CSV text and a
//! `_meta.writeBack` naming the tool that takes an edited copy back. The MCP
//! client (`tools::mcp`) removes such items before the result reaches the
//! model and hands them over as [`GridStaging`]s; whether one is honoured, and
//! how, is decided later by the WASM side. The type lives here rather than in
//! the MCP client because it is a WASM concept that the client only carries.

/// The URI scheme that marks an MCP resource as a grid for a frame component.
pub const FRAME_SCHEME: &str = "plank-frame://";

/// The tool plank calls with an edited grid when its frame closes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteBack {
    /// Name of the server's write-back tool (for chatbgt, `apply_grid`).
    pub tool: String,
    /// The table the grid was exported from.
    pub table: String,
    /// The server's token for this export.
    pub grid: String,
}

/// One grid an MCP tool result asked plank to open in a frame component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridStaging {
    /// The MCP server that produced the resource.
    pub server: String,
    /// The component id named by the URI's first segment.
    pub component: String,
    /// The file name the CSV is to be written under on the component's RAM disk.
    pub file: String,
    /// The CSV text.
    pub csv: String,
    /// Where an edited copy goes back.
    pub write_back: WriteBack,
}

/// The write-back tool's arguments as a JSON object: `{"table", "grid", "csv"}`.
#[must_use]
pub fn write_back_arguments(write_back: &WriteBack, csv: &str) -> String {
    use crate::wasmreg::json_str;
    format!(
        "{{\"table\":{},\"grid\":{},\"csv\":{}}}",
        json_str(&write_back.table),
        json_str(&write_back.grid),
        json_str(csv)
    )
}

/// Splits `plank-frame://<component>/<file>` into its two parts.
///
/// `None` unless both are present and plain: the component is a non-empty id
/// without `/`, and the file is a single non-empty path segment without `/`,
/// `\` or `..`, so it can never name anything but a file at the root of the
/// component's RAM disk.
#[must_use]
pub fn parse_frame_uri(uri: &str) -> Option<(&str, &str)> {
    let rest = uri.strip_prefix(FRAME_SCHEME)?;
    let (component, file) = rest.split_once('/')?;
    if component.is_empty() || file.is_empty() || file == "." {
        return None;
    }
    if file.contains(['/', '\\']) || file.contains("..") {
        return None;
    }
    Some((component, file))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_back_arguments_name_the_table_token_and_csv() {
        let write_back = WriteBack {
            tool: "apply_grid".to_string(),
            table: "categories".to_string(),
            grid: "0badf00d".to_string(),
        };
        assert_eq!(
            write_back_arguments(&write_back, "#,name\n1,\"Food\"\n"),
            r##"{"table":"categories","grid":"0badf00d","csv":"#,name\n1,\"Food\"\n"}"##
        );
    }

    #[test]
    fn a_frame_uri_splits_into_component_and_file() {
        assert_eq!(
            parse_frame_uri("plank-frame://csvedit/accounts.csv"),
            Some(("csvedit", "accounts.csv"))
        );
    }

    #[test]
    fn another_scheme_is_not_a_frame_uri() {
        assert_eq!(parse_frame_uri("file:///csvedit/a.csv"), None);
        assert_eq!(parse_frame_uri("csvedit/a.csv"), None);
    }

    #[test]
    fn a_missing_component_or_file_is_refused() {
        assert_eq!(parse_frame_uri("plank-frame://"), None);
        assert_eq!(parse_frame_uri("plank-frame://csvedit"), None);
        assert_eq!(parse_frame_uri("plank-frame://csvedit/"), None);
        assert_eq!(parse_frame_uri("plank-frame:///a.csv"), None);
    }

    #[test]
    fn a_file_that_is_not_one_plain_segment_is_refused() {
        assert_eq!(parse_frame_uri("plank-frame://csvedit/sub/a.csv"), None);
        assert_eq!(parse_frame_uri("plank-frame://csvedit/..\\a.csv"), None);
        assert_eq!(parse_frame_uri("plank-frame://csvedit/a\\b.csv"), None);
        assert_eq!(parse_frame_uri("plank-frame://csvedit/.."), None);
        assert_eq!(parse_frame_uri("plank-frame://csvedit/a..csv"), None);
        assert_eq!(parse_frame_uri("plank-frame://csvedit/."), None);
    }
}
