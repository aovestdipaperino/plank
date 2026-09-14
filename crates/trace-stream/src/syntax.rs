//! Which tool-call dialect a model speaks.
//!
//! The C picks this per engine in `agent_tool_syntax_for_engine`
//! (`refs/ds4/ds4_agent.c`) and then uses it to choose a tools prompt, a
//! parser, and a syntax reminder. plank carries the two dialects it supports.

/// The tool-call dialect in force for a generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolSyntax {
    /// `DeepSeek` V4's DSML markers, the dialect plank was built around.
    #[default]
    Dsml,
    /// Qwen3.8-Flash-Next's `<tool_call>` / `<function=…>` / `<parameter=…>`.
    Qwen,
    /// `DeepSeek` V4.1's DSML markers: the same dialect with a leading space
    /// and a shorter outer tag name.
    Dsml41,
}

/// The tag spellings of one DSML dialect.
///
/// V4 and V4.1 differ only in these strings. Nothing outside this table may
/// name a tag, so adding a dialect cannot silently miss a site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsmlTags {
    pub start: &'static str,
    pub start_bar: &'static str,
    pub invoke: &'static str,
    pub param_close: &'static str,
    pub calls_name: &'static str,
    pub invoke_name: &'static str,
    pub param_name: &'static str,
}

impl ToolSyntax {
    /// Every dialect, in the order a stanza opener is matched against them.
    ///
    /// The openers are distinct strings, so the order is not load-bearing for
    /// correctness; V4 leads because it is the default and the spelling the
    /// published tools prompt teaches. Anything that has to consider *all*
    /// dialects iterates this, so adding one cannot silently miss a site.
    pub const ALL: [Self; 2] = [Self::Dsml, Self::Dsml41];

    /// The dialect for a model, by the name the engine reports.
    ///
    /// Keyed on the C's shape name (`DS4_MODEL_SHAPE_NAME`) rather than on the
    /// GGUF path, because that is what the engine actually detected from the
    /// file — a renamed or relocated model still resolves correctly.
    #[must_use]
    pub fn for_model_name(name: &str) -> Self {
        if name.starts_with("Qwen3.8") {
            Self::Qwen
        } else if name.starts_with("DeepSeek V4.1") {
            Self::Dsml41
        } else {
            Self::Dsml
        }
    }

    /// Whether a stanza in this dialect is delimited by plain XML-ish tags,
    /// which is also what decides how the C injects the tools prompt.
    #[must_use]
    pub fn is_xml_tool_call(self) -> bool {
        self == Self::Qwen
    }

    /// The DSML tag spellings for this dialect, or `None` for a dialect that
    /// is not DSML at all.
    #[must_use]
    pub fn dsml_tags(self) -> Option<DsmlTags> {
        match self {
            Self::Dsml => Some(DsmlTags {
                start: "<｜DSML｜tool_calls>",
                start_bar: "<｜DSML｜tool_calls｜",
                invoke: "<｜DSML｜invoke",
                param_close: "</｜DSML｜parameter>",
                calls_name: "tool_calls",
                invoke_name: "invoke",
                param_name: "parameter",
            }),
            Self::Dsml41 => Some(DsmlTags {
                start: "<｜DSML｜ calls>",
                start_bar: "<｜DSML｜ calls｜",
                invoke: "<｜DSML｜ invoke",
                param_close: "</｜DSML｜ parameter>",
                calls_name: " calls",
                invoke_name: " invoke",
                param_name: " parameter",
            }),
            Self::Qwen => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both Qwen shapes the C declares — the full model and the `mini` — must
    /// land on the Qwen dialect, and nothing else may.
    #[test]
    fn model_names_map_to_their_dialect() {
        assert_eq!(
            ToolSyntax::for_model_name("Qwen3.8 Flash Next"),
            ToolSyntax::Qwen
        );
        assert_eq!(
            ToolSyntax::for_model_name("Qwen3.8 Flash Next mini"),
            ToolSyntax::Qwen
        );
        for other in [
            "DeepSeek V4 Flash",
            "DeepSeek V4 Flash Vision Experimental",
            "DeepSeek V4 Pro",
            "GLM 5.2",
            "GLM 5.3 Flash",
            "",
        ] {
            assert_eq!(
                ToolSyntax::for_model_name(other),
                ToolSyntax::Dsml,
                "{other} is not Qwen"
            );
        }
    }

    #[test]
    fn v41_shape_name_selects_the_v41_dialect() {
        assert_eq!(
            ToolSyntax::for_model_name("DeepSeek V4.1 Flash"),
            ToolSyntax::Dsml41
        );
        assert_eq!(
            ToolSyntax::for_model_name("DeepSeek V4 Flash"),
            ToolSyntax::Dsml
        );
    }

    #[test]
    fn v41_tags_carry_the_leading_space() {
        let v4 = ToolSyntax::Dsml.dsml_tags().expect("dsml has tags");
        let v41 = ToolSyntax::Dsml41.dsml_tags().expect("dsml41 has tags");
        assert_eq!(v4.start, "<｜DSML｜tool_calls>");
        assert_eq!(v41.start, "<｜DSML｜ calls>");
        assert_eq!(v41.invoke, "<｜DSML｜ invoke");
        assert_eq!(v41.param_close, "</｜DSML｜ parameter>");
        assert_eq!(
            (v41.calls_name, v41.invoke_name, v41.param_name),
            (" calls", " invoke", " parameter")
        );
        assert!(ToolSyntax::Qwen.dsml_tags().is_none());
    }

    #[test]
    fn v41_is_not_an_xml_dialect() {
        assert!(!ToolSyntax::Dsml41.is_xml_tool_call());
    }

    #[test]
    fn only_qwen_is_an_xml_dialect() {
        assert!(ToolSyntax::Qwen.is_xml_tool_call());
        assert!(!ToolSyntax::Dsml.is_xml_tool_call());
        assert_eq!(ToolSyntax::default(), ToolSyntax::Dsml);
    }
}
