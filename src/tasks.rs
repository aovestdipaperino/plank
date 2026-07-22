// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Model-visible task list (issue #35): structured working memory that
//! survives compaction.
//!
//! A task list is not a to-do widget; it is the structured part of the model's
//! plan, promoted out of the expendable transcript into session state so it
//! costs a fixed, small number of tokens per turn and is never summarized away
//! by [`crate::compact`]. The list lives on the [`crate::session::Session`]
//! next to the transcript, so it serializes with it and rides `/resume`,
//! `/checkpoint` rollback, and save/load for free.
//!
//! One `task` tool with an `op` argument (`add`, `update`, `list`) is the
//! entire model-facing surface — the smallest table that works, since every
//! tool costs a parity fixture entry (`tests/c_parity.rs`). The current list is
//! injected into every turn ([`TaskList::inject_block`]), so `list` is a
//! recovery path rather than the normal way to read state.

use std::fmt::Write as _;

/// Lifecycle status of one task. There is no `blocked` in the first cut:
/// blocking relationships need a dependency UI to be worth anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Not started.
    Pending,
    /// Actively being worked on; the spinner-friendly [`Task::active_form`]
    /// describes it while in flight.
    InProgress,
    /// Finished.
    Completed,
}

impl TaskStatus {
    /// The wire form used in serialization and the model-facing text.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            TaskStatus::Pending => "pending",
            TaskStatus::InProgress => "in_progress",
            TaskStatus::Completed => "completed",
        }
    }

    /// Parses a status word (case-insensitively); `None` for anything else.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pending" => Some(TaskStatus::Pending),
            "in_progress" | "in-progress" | "active" => Some(TaskStatus::InProgress),
            "completed" | "complete" | "done" => Some(TaskStatus::Completed),
            _ => None,
        }
    }
}

/// One task: an id, a subject, a status, and an optional spinner active form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task {
    /// Stable id, unique within a [`TaskList`], assigned on `add`.
    pub id: u32,
    /// One-line description of the work.
    pub subject: String,
    /// Current lifecycle status.
    pub status: TaskStatus,
    /// Present-tense form shown while the task is in flight (e.g. "Refactoring
    /// the parser"); optional.
    pub active_form: Option<String>,
}

/// An ordered list of [`Task`]s with monotonic id assignment.
///
/// Append-only within a session (there is no remove op): ids never repeat, so
/// the model can refer back to a task it created many turns ago.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskList {
    tasks: Vec<Task>,
    next_id: u32,
}

impl TaskList {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// True when no tasks have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Number of tasks in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// The tasks in insertion order.
    #[must_use]
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// The next id that would be assigned by [`add`](Self::add). Exposed so
    /// session persistence can round-trip monotonic id assignment.
    #[must_use]
    pub fn next_id(&self) -> u32 {
        self.next_id
    }

    /// Rebuilds a list from its persisted parts (see [`crate::session`]).
    /// `next_id` is clamped to at least the highest id present so a restored
    /// list can never mint a colliding id.
    #[must_use]
    pub fn from_parts(tasks: Vec<Task>, next_id: u32) -> Self {
        let max_id = tasks.iter().map(|t| t.id).max().unwrap_or(0);
        Self {
            next_id: next_id.max(max_id),
            tasks,
        }
    }

    /// Appends a pending task and returns its freshly minted id.
    pub fn add(&mut self, subject: impl Into<String>, active_form: Option<String>) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        self.tasks.push(Task {
            id,
            subject: subject.into(),
            status: TaskStatus::Pending,
            active_form: active_form.filter(|s| !s.trim().is_empty()),
        });
        id
    }

    /// Looks up a task by id.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&Task> {
        self.tasks.iter().find(|t| t.id == id)
    }

    /// Number of completed tasks.
    #[must_use]
    pub fn completed(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count()
    }

    /// `(completed, total)` counter shown in the status bar, or `None` when the
    /// list is empty (so an empty list renders no text and costs nothing).
    #[must_use]
    pub fn counter(&self) -> Option<(usize, usize)> {
        if self.tasks.is_empty() {
            None
        } else {
            Some((self.completed(), self.tasks.len()))
        }
    }

    /// True when the list is non-empty and every task is completed.
    #[must_use]
    pub fn all_done(&self) -> bool {
        !self.tasks.is_empty() && self.completed() == self.tasks.len()
    }

    /// Updates a task's status, subject, and/or active form. Returns `Ok(true)`
    /// when the task became `completed` by this call (so the UI can log the
    /// single completion line), `Ok(false)` otherwise. An unknown id is a clear
    /// error.
    ///
    /// # Errors
    ///
    /// Returns the id in the error when no task matches it.
    pub fn update(
        &mut self,
        id: u32,
        status: Option<TaskStatus>,
        subject: Option<String>,
        active_form: Option<String>,
    ) -> Result<bool, u32> {
        let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) else {
            return Err(id);
        };
        let was_completed = task.status == TaskStatus::Completed;
        if let Some(s) = status {
            task.status = s;
        }
        if let Some(s) = subject {
            task.subject = s;
        }
        if let Some(a) = active_form {
            let a = a.trim();
            task.active_form = if a.is_empty() {
                None
            } else {
                Some(a.to_string())
            };
        }
        Ok(task.status == TaskStatus::Completed && !was_completed)
    }

    /// The model-facing full listing (one task per line), or a clear
    /// no-tasks message.
    #[must_use]
    pub fn render_list(&self) -> String {
        if self.tasks.is_empty() {
            return "No tasks in the list.\n".to_string();
        }
        let mut out = String::new();
        for t in &self.tasks {
            let _ = writeln!(out, "- [{}] {}: {}", t.id, t.status.as_str(), t.subject);
        }
        out
    }

    /// The per-turn injection block, or `None` when the list is empty so an
    /// empty list adds nothing to the prompt. Regenerated every turn from
    /// session state, so it survives compaction and never accumulates in the
    /// transcript.
    #[must_use]
    pub fn inject_block(&self) -> Option<String> {
        if self.tasks.is_empty() {
            return None;
        }
        let mut out = String::from(
            "# Task list\n\nYour current tasks (manage with the `task` tool, \
             op=add|update|list):\n",
        );
        for t in &self.tasks {
            let _ = writeln!(out, "- [{}] {}: {}", t.id, t.status.as_str(), t.subject);
        }
        Some(out)
    }

    /// Rows for the contextual strip above the separator rule: the active
    /// (in-progress) task first, then up to two pending tasks, capped at three
    /// rows regardless of list length. Empty when nothing is in progress (a
    /// finished, or not-yet-started, list takes no space). Each row is
    /// `(text, is_active)`.
    #[must_use]
    pub fn strip_rows(&self) -> Vec<(String, bool)> {
        const MAX_ROWS: usize = 3;
        if !self
            .tasks
            .iter()
            .any(|t| t.status == TaskStatus::InProgress)
        {
            return Vec::new();
        }
        let mut rows = Vec::new();
        // The active row: the first in-progress task, shown in its active form
        // when it has one.
        if let Some(active) = self
            .tasks
            .iter()
            .find(|t| t.status == TaskStatus::InProgress)
        {
            let text = active
                .active_form
                .clone()
                .unwrap_or_else(|| active.subject.clone());
            rows.push((text, true));
        }
        // Then the next pending tasks, dimmed, filling up to the cap.
        for t in self
            .tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Pending)
        {
            if rows.len() >= MAX_ROWS {
                break;
            }
            rows.push((t.subject.clone(), false));
        }
        rows
    }
}

/// `task` tool: the model's one entry point to its task list. `op` selects
/// `add` (append a pending task, returns its id), `update` (change status,
/// subject, or active form; unknown id is an error), or `list` (dump the whole
/// list — a recovery path, since the list is injected each turn).
///
/// Subjects freshly completed by an `update` are pushed onto `completions` so
/// the UI can write the single dim completion line to the log.
pub fn tool_task(
    tasks: &mut TaskList,
    completions: &mut Vec<String>,
    call: &crate::dsml::ToolCall,
) -> String {
    let op = call
        .arg_value("op")
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match op.as_str() {
        "add" => {
            let subject = call.arg_value("subject").unwrap_or("").trim();
            if subject.is_empty() {
                return "Tool error: task add requires a non-empty 'subject'\n".to_string();
            }
            let active_form = call.arg_value("active_form").map(str::to_string);
            let id = tasks.add(subject, active_form);
            format!("Added task [{id}]: {subject}\n")
        }
        "update" => {
            let Some(id) = call
                .arg_value("id")
                .and_then(|s| s.trim().parse::<u32>().ok())
            else {
                return "Tool error: task update requires a numeric 'id'\n".to_string();
            };
            let status = match call.arg_value("status") {
                Some(s) if !s.trim().is_empty() => match TaskStatus::parse(s) {
                    Some(st) => Some(st),
                    None => {
                        return format!(
                            "Tool error: unknown status '{}'; use pending, in_progress, \
                             or completed\n",
                            s.trim()
                        );
                    }
                },
                _ => None,
            };
            let subject = call
                .arg_value("subject")
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            let active_form = call.arg_value("active_form").map(str::to_string);
            if status.is_none() && subject.is_none() && active_form.is_none() {
                return "Tool error: task update needs at least one of status, subject, \
                        or active_form\n"
                    .to_string();
            }
            match tasks.update(id, status, subject, active_form) {
                Ok(just_completed) => match tasks.get(id) {
                    Some(t) => {
                        if just_completed {
                            completions.push(t.subject.clone());
                        }
                        format!(
                            "Updated task [{}]: {} ({})\n",
                            t.id,
                            t.subject,
                            t.status.as_str()
                        )
                    }
                    // Unreachable: `update` returned Ok, so the id exists.
                    None => format!("Updated task [{id}]\n"),
                },
                Err(id) => format!("Tool error: no task with id {id}\n"),
            }
        }
        "list" => tasks.render_list(),
        "" => "Tool error: task requires 'op' set to add, update, or list\n".to_string(),
        other => {
            format!("Tool error: unknown task op '{other}'; use add, update, or list\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsml::{ToolArg, ToolCall};

    fn call(args: &[(&str, &str)]) -> ToolCall {
        ToolCall {
            name: "task".to_string(),
            args: args
                .iter()
                .map(|(k, v)| ToolArg {
                    name: (*k).to_string(),
                    value: (*v).to_string(),
                    is_string: true,
                })
                .collect(),
        }
    }

    #[test]
    fn add_appends_pending_and_returns_id() {
        let mut list = TaskList::new();
        let a = list.add("first", None);
        let b = list.add("second", Some("doing second".to_string()));
        assert_eq!((a, b), (1, 2));
        assert_eq!(list.len(), 2);
        assert_eq!(list.get(1).unwrap().status, TaskStatus::Pending);
        assert_eq!(
            list.get(2).unwrap().active_form.as_deref(),
            Some("doing second")
        );
    }

    #[test]
    fn update_changes_status_and_flags_completion() {
        let mut list = TaskList::new();
        list.add("a", None);
        assert_eq!(
            list.update(1, Some(TaskStatus::InProgress), None, None),
            Ok(false)
        );
        // First transition into completed reports true; a second stays false.
        assert_eq!(
            list.update(1, Some(TaskStatus::Completed), None, None),
            Ok(true)
        );
        assert_eq!(
            list.update(1, Some(TaskStatus::Completed), None, None),
            Ok(false)
        );
    }

    #[test]
    fn update_unknown_id_is_an_error() {
        let mut list = TaskList::new();
        list.add("a", None);
        assert_eq!(
            list.update(99, Some(TaskStatus::Completed), None, None),
            Err(99)
        );
    }

    #[test]
    fn counter_and_all_done_track_completion() {
        let mut list = TaskList::new();
        assert_eq!(list.counter(), None);
        assert!(!list.all_done());
        list.add("a", None);
        list.add("b", None);
        assert_eq!(list.counter(), Some((0, 2)));
        list.update(1, Some(TaskStatus::Completed), None, None)
            .unwrap();
        assert_eq!(list.counter(), Some((1, 2)));
        assert!(!list.all_done());
        list.update(2, Some(TaskStatus::Completed), None, None)
            .unwrap();
        assert!(list.all_done());
    }

    #[test]
    fn empty_list_injects_nothing() {
        assert_eq!(TaskList::new().inject_block(), None);
        assert!(TaskList::new().strip_rows().is_empty());
    }

    #[test]
    fn inject_block_lists_every_task() {
        let mut list = TaskList::new();
        list.add("read the spec", None);
        list.update(1, Some(TaskStatus::InProgress), None, None)
            .unwrap();
        list.add("write tests", None);
        let block = list.inject_block().unwrap();
        assert!(
            block.contains("- [1] in_progress: read the spec"),
            "{block}"
        );
        assert!(block.contains("- [2] pending: write tests"), "{block}");
    }

    #[test]
    fn strip_shows_active_first_then_pending_capped_at_three() {
        let mut list = TaskList::new();
        for i in 0..6 {
            list.add(format!("task {i}"), None);
        }
        // Nothing in progress: no strip.
        assert!(list.strip_rows().is_empty());
        list.update(
            3,
            Some(TaskStatus::InProgress),
            None,
            Some("doing task 2".to_string()),
        )
        .unwrap();
        let rows = list.strip_rows();
        assert_eq!(rows.len(), 3, "capped at three regardless of list length");
        assert_eq!(rows[0], ("doing task 2".to_string(), true));
        // The rest are pending subjects, not active.
        assert!(!rows[1].1 && !rows[2].1);
        assert_eq!(rows[1].0, "task 0");
        assert_eq!(rows[2].0, "task 1");
    }

    #[test]
    fn strip_vanishes_when_all_completed() {
        let mut list = TaskList::new();
        list.add("a", None);
        list.update(1, Some(TaskStatus::InProgress), None, None)
            .unwrap();
        assert_eq!(list.strip_rows().len(), 1);
        list.update(1, Some(TaskStatus::Completed), None, None)
            .unwrap();
        assert!(list.strip_rows().is_empty());
    }

    #[test]
    fn tool_add_then_update_then_list() {
        let mut list = TaskList::new();
        let mut done = Vec::new();
        let out = tool_task(
            &mut list,
            &mut done,
            &call(&[("op", "add"), ("subject", "do it")]),
        );
        assert_eq!(out, "Added task [1]: do it\n");
        assert!(done.is_empty(), "add logs no completion");

        let out = tool_task(
            &mut list,
            &mut done,
            &call(&[("op", "update"), ("id", "1"), ("status", "completed")]),
        );
        assert!(
            out.starts_with("Updated task [1]: do it (completed)"),
            "{out}"
        );
        assert_eq!(done, vec!["do it".to_string()], "completion recorded once");

        let out = tool_task(&mut list, &mut done, &call(&[("op", "list")]));
        assert_eq!(out, "- [1] completed: do it\n");
    }

    #[test]
    fn tool_errors_are_clear() {
        let mut list = TaskList::new();
        let mut done = Vec::new();
        assert!(
            tool_task(&mut list, &mut done, &call(&[]))
                .starts_with("Tool error: task requires 'op'")
        );
        assert!(
            tool_task(&mut list, &mut done, &call(&[("op", "frob")]))
                .starts_with("Tool error: unknown task op 'frob'")
        );
        assert!(
            tool_task(&mut list, &mut done, &call(&[("op", "add")]))
                .starts_with("Tool error: task add requires")
        );
        assert!(
            tool_task(
                &mut list,
                &mut done,
                &call(&[("op", "update"), ("id", "5"), ("status", "completed")])
            )
            .starts_with("Tool error: no task with id 5")
        );
        list.add("x", None);
        assert!(
            tool_task(
                &mut list,
                &mut done,
                &call(&[("op", "update"), ("id", "1"), ("status", "weird")])
            )
            .starts_with("Tool error: unknown status 'weird'")
        );
        assert!(
            tool_task(
                &mut list,
                &mut done,
                &call(&[("op", "update"), ("id", "1")])
            )
            .starts_with("Tool error: task update needs at least one")
        );
        assert!(done.is_empty());
    }
}
