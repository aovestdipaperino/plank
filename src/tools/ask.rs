//! The `ask` tool (issue #34): structured multiple-choice questions to the user.
//!
//! When a turn is genuinely ambiguous the model's only trained move is to emit
//! prose and stop; `ask` gives it a structured alternative — present two to four
//! labelled options and block until the user picks. This is a native plank tool
//! (not in the C-trained table), so like `glob`/`skill`/`task` its schema is
//! appended on top of the parity-locked base and mirrored in the provider
//! registry (`crate::sysprompt`).
//!
//! The tool itself is UI-agnostic: [`tool_ask`] parses and validates the call,
//! then delegates the actual prompting to an [`Asker`] carried on the
//! [`ToolContext`](crate::tools::ToolContext). The three front ends install
//! different askers — the Ratatui TUI an [`AskBridge`]-backed one that renders
//! into the input region, the plain REPL a stdin reader, and `--non-interactive`
//! none at all (fast-fail). The rendering-independent pieces (validation,
//! option-list navigation, result formatting) live here so they can be
//! unit-tested without a live terminal.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::dsml::ToolCall;

/// Minimum number of options `ask` accepts: a choice needs at least two arms.
pub const MIN_OPTIONS: usize = 2;
/// Upper bound on options; more than this stops reading as a bounded pick.
pub const MAX_OPTIONS: usize = 4;

/// One selectable choice: a short label and a one-line description of what
/// picking it means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskOption {
    /// The label returned as (part of) the tool result when chosen.
    pub label: String,
    /// One-line explanation shown beneath the label.
    pub description: String,
}

/// A fully-parsed, validated question ready to present to the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskRequest {
    /// The complete question, phrased as a question.
    pub question: String,
    /// Short UI-chip label (~12 chars).
    pub header: String,
    /// Between [`MIN_OPTIONS`] and [`MAX_OPTIONS`] choices.
    pub options: Vec<AskOption>,
    /// When true, more than one option may be selected.
    pub multi: bool,
}

/// What the user did with a presented [`AskRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AskOutcome {
    /// The user accepted a (possibly multi) selection; carries the chosen labels.
    Answered(Vec<String>),
    /// The user explicitly declined (Escape / empty answer).
    Declined,
    /// The turn was interrupted (Ctrl-C) while the question was up.
    Interrupted,
}

/// A front-end capable of presenting a question and returning the user's choice.
///
/// Installed on the [`ToolContext`](crate::tools::ToolContext) by whichever
/// front end is running; `None` there means non-interactive mode and
/// [`tool_ask`] fast-fails.
pub trait Asker: Send {
    /// Presents `req` and blocks until the user answers, declines, or interrupts.
    fn ask(&mut self, req: AskRequest) -> AskOutcome;
}

/// Parses and validates an `ask` tool call, then delegates presentation to the
/// context's [`Asker`]. Returns the model-visible tool result text.
///
/// Validation failures and the non-interactive fast-fail both return before any
/// blocking, matching the C `Tool error:` convention for the former.
#[must_use]
pub fn tool_ask(asker: Option<&mut Box<dyn Asker>>, call: &ToolCall) -> String {
    let question = call.arg_value("question").unwrap_or("").trim();
    if question.is_empty() {
        return "Tool error: ask requires a non-empty 'question'\n".to_string();
    }
    let header = call.arg_value("header").unwrap_or("").trim();
    if header.is_empty() {
        return "Tool error: ask requires a non-empty 'header'\n".to_string();
    }
    let options = match parse_options(call) {
        Ok(o) => o,
        Err(e) => return e,
    };
    let multi = crate::tools::parse_bool_default(call.arg_value("multi"), false);
    let req = AskRequest {
        question: question.to_string(),
        header: header.to_string(),
        options,
        multi,
    };
    // No interactive front end (`--non-interactive` / headless): there is no
    // user to ask, so tell the model to proceed rather than blocking forever.
    let Some(asker) = asker else {
        return "No interactive user is available to answer (non-interactive mode); \
                proceed using your best judgment.\n"
            .to_string();
    };
    format_result(&asker.ask(req))
}

/// Parses and validates the `options` argument into 2..=4 [`AskOption`]s.
///
/// Options arrive as a JSON array in the `options` argument — a list of
/// `{"label": "...", "description": "..."}` objects; the array carries the
/// structure DSML's flat string parameters cannot. A count outside
/// [`MIN_OPTIONS`]..=[`MAX_OPTIONS`] is a dispatch error.
fn parse_options(call: &ToolCall) -> Result<Vec<AskOption>, String> {
    let raw = call.arg_value("options").unwrap_or("").trim();
    if raw.is_empty() {
        return Err("Tool error: ask requires an 'options' JSON array\n".to_string());
    }
    let parsed: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| format!("Tool error: ask 'options' is not valid JSON: {e}\n"))?;
    let Some(arr) = parsed.as_array() else {
        return Err("Tool error: ask 'options' must be a JSON array\n".to_string());
    };
    let mut options = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let label = item
            .get("label")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .trim();
        if label.is_empty() {
            return Err(format!(
                "Tool error: ask option {} is missing a 'label'\n",
                i + 1
            ));
        }
        let description = item
            .get("description")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .trim();
        options.push(AskOption {
            label: label.to_string(),
            description: description.to_string(),
        });
    }
    if options.len() < MIN_OPTIONS || options.len() > MAX_OPTIONS {
        return Err(format!(
            "Tool error: ask needs {MIN_OPTIONS} to {MAX_OPTIONS} options, got {}\n",
            options.len()
        ));
    }
    Ok(options)
}

/// Renders an [`AskOutcome`] as the model-visible tool result text.
#[must_use]
pub fn format_result(outcome: &AskOutcome) -> String {
    match outcome {
        AskOutcome::Answered(labels) if labels.is_empty() => {
            // Multi-select accepted with nothing ticked reads as a decline.
            "User answered but selected no option.\n".to_string()
        }
        AskOutcome::Answered(labels) => format!("User selected: {}\n", labels.join(", ")),
        AskOutcome::Declined => {
            "User declined to answer; proceed using your best judgment.\n".to_string()
        }
        AskOutcome::Interrupted => "The question was interrupted by the user.\n".to_string(),
    }
}

/// Navigable selection state over a bounded option list.
///
/// Pure logic behind the interactive front ends: the cursor moves, and in
/// multi-select mode entries toggle. Kept free of any terminal dependency so it
/// is unit-testable and shared by every front end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskState {
    /// Highlighted row.
    pub cursor: usize,
    /// Per-option ticked flags (all false until toggled); only meaningful when
    /// [`multi`](Self::multi) is set.
    pub selected: Vec<bool>,
    /// Whether more than one option may be ticked.
    pub multi: bool,
}

impl AskState {
    /// A fresh state for `len` options, cursor on the first.
    #[must_use]
    pub fn new(len: usize, multi: bool) -> Self {
        Self {
            cursor: 0,
            selected: vec![false; len],
            multi,
        }
    }

    /// Number of options.
    #[must_use]
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    /// True when there are no options (never happens after validation).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// Moves the cursor up one row, saturating at the top.
    pub fn move_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    /// Moves the cursor down one row, saturating at the last option.
    pub fn move_down(&mut self) {
        if self.cursor + 1 < self.len() {
            self.cursor += 1;
        }
    }

    /// Toggles the option under the cursor (multi-select only; a no-op otherwise).
    pub fn toggle(&mut self) {
        if self.multi
            && let Some(flag) = self.selected.get_mut(self.cursor)
        {
            *flag = !*flag;
        }
    }

    /// The labels the current selection resolves to, given the option list.
    ///
    /// In single-select mode this is the highlighted option; in multi-select it
    /// is every ticked option, falling back to the highlighted one when none is
    /// ticked (Enter on a fresh list still commits a sensible choice).
    #[must_use]
    pub fn accept(&self, options: &[AskOption]) -> Vec<String> {
        if self.multi {
            let ticked: Vec<String> = self
                .selected
                .iter()
                .enumerate()
                .filter(|(_, on)| **on)
                .filter_map(|(i, _)| options.get(i).map(|o| o.label.clone()))
                .collect();
            if ticked.is_empty() {
                options
                    .get(self.cursor)
                    .map(|o| vec![o.label.clone()])
                    .unwrap_or_default()
            } else {
                ticked
            }
        } else {
            options
                .get(self.cursor)
                .map(|o| vec![o.label.clone()])
                .unwrap_or_default()
        }
    }
}

/// Rows the question panel needs: a header/question line, a blank spacer, one
/// row per option, and a key-hint line. Used by the TUI to size the panel so it
/// never overlaps the status bar. Bounded because the option count is bounded.
#[must_use]
pub fn panel_rows(options: usize) -> u16 {
    let rows = 2 + options + 1;
    u16::try_from(rows).unwrap_or(u16::MAX)
}

/// Resolves a plain-REPL answer line against a request.
///
/// Accepts option numbers (1-based) or a case-insensitive label prefix, comma-
/// separated when `multi`. An empty line (or one that resolves to nothing) is a
/// decline, matching the TUI's Escape. Only the first token is honoured in
/// single-select mode.
#[must_use]
pub fn parse_repl_answer(req: &AskRequest, line: &str) -> AskOutcome {
    let line = line.trim();
    if line.is_empty() {
        return AskOutcome::Declined;
    }
    let tokens: Vec<&str> = if req.multi {
        line.split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect()
    } else {
        vec![line]
    };
    let mut labels = Vec::new();
    for token in tokens {
        if let Some(label) = resolve_token(req, token)
            && !labels.contains(&label)
        {
            labels.push(label);
        }
    }
    if labels.is_empty() {
        AskOutcome::Declined
    } else {
        AskOutcome::Answered(labels)
    }
}

/// Resolves one answer token to an option label: a 1-based index first, then a
/// unique case-insensitive label prefix.
fn resolve_token(req: &AskRequest, token: &str) -> Option<String> {
    if let Ok(n) = token.parse::<usize>()
        && n >= 1
        && n <= req.options.len()
    {
        return Some(req.options[n - 1].label.clone());
    }
    let lower = token.to_ascii_lowercase();
    let mut matches = req
        .options
        .iter()
        .filter(|o| o.label.to_ascii_lowercase().starts_with(&lower));
    let first = matches.next()?;
    if matches.next().is_some() {
        // Ambiguous prefix: refuse rather than guess.
        return None;
    }
    Some(first.label.clone())
}

/// Shared slot backing an [`AskBridge`]: the worker parks a request here and
/// spins until the UI thread posts a response.
#[derive(Debug, Default)]
struct AskInner {
    request: Mutex<Option<AskRequest>>,
    response: Mutex<Option<AskOutcome>>,
    pending: AtomicBool,
}

/// Cross-thread rendezvous between the worker (which runs tool dispatch) and the
/// TUI event loop (which owns the terminal). The worker cannot touch the
/// terminal, so the [`BridgeAsker`] parks the request here and blocks; the UI
/// loop picks it up, renders the interactive panel, and posts the answer back.
#[derive(Debug, Clone, Default)]
pub struct AskBridge {
    inner: Arc<AskInner>,
}

impl AskBridge {
    /// A fresh, idle bridge.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Worker side: park `req` and block until the UI posts a response. Polls
    /// rather than condvar-waits to keep the type trivially `Send` and to bound
    /// latency at one tick; the worker has nothing else to do meanwhile.
    fn submit_and_wait(&self, req: AskRequest) -> AskOutcome {
        Self::set(&self.inner.request, Some(req));
        self.inner.pending.store(true, Ordering::SeqCst);
        loop {
            if let Some(outcome) = Self::take(&self.inner.response) {
                self.inner.pending.store(false, Ordering::SeqCst);
                return outcome;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// UI side: true while a question awaits an answer.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.inner.pending.load(Ordering::SeqCst)
    }

    /// UI side: take the parked request (leaving `pending` set until answered).
    #[must_use]
    pub fn take_request(&self) -> Option<AskRequest> {
        Self::take(&self.inner.request)
    }

    /// UI side: post the user's answer, unblocking the worker.
    pub fn respond(&self, outcome: AskOutcome) {
        Self::set(&self.inner.response, Some(outcome));
    }

    fn set<T>(m: &Mutex<Option<T>>, v: Option<T>) {
        *m.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = v;
    }

    fn take<T>(m: &Mutex<Option<T>>) -> Option<T> {
        m.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

/// [`Asker`] that hands questions to the TUI event loop over an [`AskBridge`].
#[derive(Debug)]
pub struct BridgeAsker(pub AskBridge);

impl Asker for BridgeAsker {
    fn ask(&mut self, req: AskRequest) -> AskOutcome {
        self.0.submit_and_wait(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(args: &[(&str, &str)]) -> ToolCall {
        crate::tools::test_call("ask", args)
    }

    fn opts(n: usize) -> String {
        let items: Vec<String> = (0..n)
            .map(|i| format!("{{\"label\":\"L{i}\",\"description\":\"D{i}\"}}"))
            .collect();
        format!("[{}]", items.join(","))
    }

    struct FixedAsker(AskOutcome);
    impl Asker for FixedAsker {
        fn ask(&mut self, _req: AskRequest) -> AskOutcome {
            self.0.clone()
        }
    }

    #[test]
    fn too_few_options_is_dispatch_error() {
        let c = call(&[("question", "q?"), ("header", "h"), ("options", &opts(1))]);
        let out = tool_ask(None, &c);
        assert!(out.starts_with("Tool error:"), "{out}");
    }

    #[test]
    fn too_many_options_is_dispatch_error() {
        let c = call(&[("question", "q?"), ("header", "h"), ("options", &opts(5))]);
        let out = tool_ask(None, &c);
        assert!(out.starts_with("Tool error:"), "{out}");
    }

    #[test]
    fn valid_count_bounds_are_accepted() {
        for n in MIN_OPTIONS..=MAX_OPTIONS {
            let c = call(&[("question", "q?"), ("header", "h"), ("options", &opts(n))]);
            // With no asker this fast-fails, but it must get past validation
            // (i.e. not a Tool error).
            let out = tool_ask(None, &c);
            assert!(!out.starts_with("Tool error:"), "n={n}: {out}");
        }
    }

    #[test]
    fn single_select_result() {
        let c = call(&[("question", "q?"), ("header", "h"), ("options", &opts(3))]);
        let mut asker: Box<dyn Asker> =
            Box::new(FixedAsker(AskOutcome::Answered(vec!["L1".to_string()])));
        let out = tool_ask(Some(&mut asker), &c);
        assert_eq!(out, "User selected: L1\n");
    }

    #[test]
    fn multi_select_result() {
        let c = call(&[
            ("question", "q?"),
            ("header", "h"),
            ("options", &opts(3)),
            ("multi", "true"),
        ]);
        let mut asker: Box<dyn Asker> = Box::new(FixedAsker(AskOutcome::Answered(vec![
            "L0".to_string(),
            "L2".to_string(),
        ])));
        let out = tool_ask(Some(&mut asker), &c);
        assert_eq!(out, "User selected: L0, L2\n");
    }

    #[test]
    fn declined_result() {
        let c = call(&[("question", "q?"), ("header", "h"), ("options", &opts(2))]);
        let mut asker: Box<dyn Asker> = Box::new(FixedAsker(AskOutcome::Declined));
        let out = tool_ask(Some(&mut asker), &c);
        assert!(out.contains("declined"), "{out}");
        assert!(!out.starts_with("Tool error:"));
    }

    #[test]
    fn non_interactive_fast_fails_without_blocking() {
        let c = call(&[("question", "q?"), ("header", "h"), ("options", &opts(2))]);
        let out = tool_ask(None, &c);
        assert!(out.contains("non-interactive"), "{out}");
        assert!(!out.starts_with("Tool error:"));
    }

    #[test]
    fn missing_question_or_header_errors() {
        let c = call(&[("header", "h"), ("options", &opts(2))]);
        assert!(tool_ask(None, &c).starts_with("Tool error:"));
        let c = call(&[("question", "q?"), ("options", &opts(2))]);
        assert!(tool_ask(None, &c).starts_with("Tool error:"));
    }

    #[test]
    fn state_navigation_saturates() {
        let mut s = AskState::new(3, false);
        s.move_up();
        assert_eq!(s.cursor, 0);
        s.move_down();
        s.move_down();
        s.move_down();
        assert_eq!(s.cursor, 2);
    }

    #[test]
    fn single_select_accept_uses_cursor() {
        let options = vec![
            AskOption {
                label: "a".into(),
                description: String::new(),
            },
            AskOption {
                label: "b".into(),
                description: String::new(),
            },
        ];
        let mut s = AskState::new(2, false);
        s.move_down();
        assert_eq!(s.accept(&options), vec!["b".to_string()]);
        // Toggle is a no-op in single-select.
        s.toggle();
        assert_eq!(s.accept(&options), vec!["b".to_string()]);
    }

    #[test]
    fn multi_select_accept_collects_ticked() {
        let options = vec![
            AskOption {
                label: "a".into(),
                description: String::new(),
            },
            AskOption {
                label: "b".into(),
                description: String::new(),
            },
            AskOption {
                label: "c".into(),
                description: String::new(),
            },
        ];
        let mut s = AskState::new(3, true);
        s.toggle(); // a
        s.move_down();
        s.move_down();
        s.toggle(); // c
        assert_eq!(s.accept(&options), vec!["a".to_string(), "c".to_string()]);
    }

    #[test]
    fn multi_select_accept_falls_back_to_cursor() {
        let options = vec![
            AskOption {
                label: "a".into(),
                description: String::new(),
            },
            AskOption {
                label: "b".into(),
                description: String::new(),
            },
        ];
        let s = AskState::new(2, true);
        assert_eq!(s.accept(&options), vec!["a".to_string()]);
    }

    fn req(multi: bool) -> AskRequest {
        AskRequest {
            question: "q?".into(),
            header: "h".into(),
            options: vec![
                AskOption {
                    label: "Alpha".into(),
                    description: String::new(),
                },
                AskOption {
                    label: "Beta".into(),
                    description: String::new(),
                },
                AskOption {
                    label: "Gamma".into(),
                    description: String::new(),
                },
            ],
            multi,
        }
    }

    #[test]
    fn repl_answer_by_number() {
        assert_eq!(
            parse_repl_answer(&req(false), "2"),
            AskOutcome::Answered(vec!["Beta".to_string()])
        );
    }

    #[test]
    fn repl_answer_by_label_prefix() {
        assert_eq!(
            parse_repl_answer(&req(false), "gam"),
            AskOutcome::Answered(vec!["Gamma".to_string()])
        );
    }

    #[test]
    fn repl_empty_declines() {
        assert_eq!(parse_repl_answer(&req(false), "  "), AskOutcome::Declined);
        assert_eq!(parse_repl_answer(&req(false), "9"), AskOutcome::Declined);
    }

    #[test]
    fn repl_multi_comma_list() {
        assert_eq!(
            parse_repl_answer(&req(true), "1, Gamma"),
            AskOutcome::Answered(vec!["Alpha".to_string(), "Gamma".to_string()])
        );
    }

    #[test]
    fn panel_rows_scales_with_options() {
        assert!(panel_rows(4) > panel_rows(2));
    }

    #[test]
    fn bridge_round_trips_answer() {
        let bridge = AskBridge::new();
        let worker = bridge.clone();
        let handle = std::thread::spawn(move || {
            let mut asker = BridgeAsker(worker);
            asker.ask(AskRequest {
                question: "q?".into(),
                header: "h".into(),
                options: vec![AskOption {
                    label: "L0".into(),
                    description: "d".into(),
                }],
                multi: false,
            })
        });
        // Spin until the worker parks the request.
        let req = loop {
            if let Some(req) = bridge.take_request() {
                break req;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(req.header, "h");
        assert!(bridge.is_pending());
        bridge.respond(AskOutcome::Answered(vec!["L0".to_string()]));
        let outcome = handle.join().expect("worker joins");
        assert_eq!(outcome, AskOutcome::Answered(vec!["L0".to_string()]));
        assert!(!bridge.is_pending());
    }
}
