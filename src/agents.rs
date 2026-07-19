//! Single-subagent sidechain support (issue #10, reduced scope).
//!
//! `/subagent <task>` runs a delegated task as a *fork* of the current
//! conversation: the framed task is appended to the live transcript, the
//! normal turn loop runs (tools included), and afterwards the fork is
//! truncated so only the subagent's final report — framed by
//! [`report_message`] — enters the parent conversation. Because the fork
//! shares the parent transcript prefix, the engine's per-turn common-prefix
//! sync reuses the parent KV cache on the way in and rolls the sidechain
//! back on the next real turn.
//!
//! One built-in general-purpose subagent only; named agent definitions and
//! agent teams are a separate feature (see the tracking issue).

/// Frames the delegated task as the sidechain's user turn.
#[must_use]
pub fn task_message(task: &str) -> String {
    format!(
        "<system-reminder>\n\
         You are now acting as a subagent, handling a task delegated from the \
         main conversation. Complete the task using your tools, then end with \
         a final report of your results — only that report is carried back \
         into the main conversation; everything else is discarded.\n\
         </system-reminder>\n\n\
         Task: {}",
        task.trim()
    )
}

/// Frames the subagent's final report for the parent conversation.
#[must_use]
pub fn report_message(task: &str, report: &str) -> String {
    format!(
        "<system-reminder>\n\
         A subagent completed the delegated task: {}\n\
         Its final report follows. This is background context from a sidechain \
         run; do not respond to it directly unless the user asks.\n\
         </system-reminder>\n\n\
         Subagent report:\n{}",
        task.trim(),
        report.trim()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_and_report_framing() {
        let task = task_message("count the tests\n");
        assert!(task.starts_with("<system-reminder>\n"));
        assert!(task.ends_with("Task: count the tests"));
        let report = report_message("count the tests", "There are 42.\n");
        assert!(report.contains("completed the delegated task: count the tests"));
        assert!(report.ends_with("Subagent report:\nThere are 42."));
    }
}
