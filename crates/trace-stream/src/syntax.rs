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
}

impl ToolSyntax {
    /// The dialect for a model, by the name the engine reports.
    ///
    /// Keyed on the C's shape name (`DS4_MODEL_SHAPE_NAME`) rather than on the
    /// GGUF path, because that is what the engine actually detected from the
    /// file — a renamed or relocated model still resolves correctly.
    #[must_use]
    pub fn for_model_name(name: &str) -> Self {
        if name.starts_with("Qwen3.8") {
            Self::Qwen
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
    fn only_qwen_is_an_xml_dialect() {
        assert!(ToolSyntax::Qwen.is_xml_tool_call());
        assert!(!ToolSyntax::Dsml.is_xml_tool_call());
        assert_eq!(ToolSyntax::default(), ToolSyntax::Dsml);
    }
}
