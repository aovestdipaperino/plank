// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `/stats`: activity heatmap and headline usage figures computed from the
//! saved-session metadata `/insights` already caches. Every number is
//! deterministic; the model is never involved.

use std::collections::{BTreeMap, BTreeSet};

use crate::insights::SessionMeta;
use crate::session::SessionEntry;

/// Weeks the heatmap spans, the current week included.
pub const WEEKS: usize = 53;

const SECS_PER_DAY: i64 = 86_400;

/// Which sessions the figures below the heatmap describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    AllTime,
    Last7,
    Last30,
}

impl Scope {
    /// The scope after this one when `/stats` is re-issued with the panel open.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::AllTime => Self::Last7,
            Self::Last7 => Self::Last30,
            Self::Last30 => Self::AllTime,
        }
    }

    /// Parses the `/stats` argument: empty or `all`, `7`, `30`.
    #[must_use]
    pub fn parse(arg: &str) -> Option<Self> {
        match arg.trim() {
            "" | "all" => Some(Self::AllTime),
            "7" => Some(Self::Last7),
            "30" => Some(Self::Last30),
            _ => None,
        }
    }

    /// Panel title carrying the scope, so re-issuing `/stats` knows where it is.
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Self::AllTime => "stats",
            Self::Last7 => "stats 7d",
            Self::Last30 => "stats 30d",
        }
    }

    /// Inverse of [`Scope::title`].
    #[must_use]
    pub fn from_title(title: &str) -> Option<Self> {
        match title {
            "stats" => Some(Self::AllTime),
            "stats 7d" => Some(Self::Last7),
            "stats 30d" => Some(Self::Last30),
            _ => None,
        }
    }

    /// Human label in the scope row.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::AllTime => "All time",
            Self::Last7 => "Last 7 days",
            Self::Last30 => "Last 30 days",
        }
    }

    fn window_days(self) -> Option<i64> {
        match self {
            Self::AllTime => None,
            Self::Last7 => Some(7),
            Self::Last30 => Some(30),
        }
    }
}

/// Local calendar day (days since 1970-01-01 in local time) of a unix stamp.
#[must_use]
pub fn day_of(ts: u64, tz_offset: i64) -> i64 {
    (i64::try_from(ts)
        .unwrap_or(i64::MAX)
        .saturating_add(tz_offset))
    .div_euclid(SECS_PER_DAY)
}

/// Monday-first weekday (0 = Monday) of a day index; day 0 was a Thursday.
#[must_use]
pub fn weekday_mon0(day: i64) -> usize {
    usize::try_from((day + 3).rem_euclid(7)).unwrap_or(0)
}

/// Proleptic Gregorian (year, month, day) of a day index (Howard Hinnant's
/// `civil_from_days`).
#[must_use]
pub fn civil(day: i64) -> (i64, u32, u32) {
    let z = day + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
    let m = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).unwrap_or(1);
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Headline figures for one [`Scope`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Figures {
    pub sessions: usize,
    pub active_days: usize,
    /// Days the scope covers: the window, or all-time days since the first
    /// session, inclusive.
    pub span_days: i64,
    /// Day index of the day with the most approx tokens; ties go to the latest.
    pub most_active_day: Option<i64>,
    pub longest_session_secs: u64,
    pub longest_streak: usize,
    /// Consecutive active days ending today or yesterday.
    pub current_streak: usize,
    /// Model with the most sessions; ties go to the most approx tokens.
    pub favorite_model: Option<String>,
    pub approx_tokens: u64,
}

#[derive(Debug, Clone)]
struct Row {
    day: i64,
    tokens: u64,
    duration: u64,
    model: String,
}

/// Everything `/stats` shows, computed once per open or scope change.
#[derive(Debug, Clone)]
pub struct Stats {
    /// `WEEKS` columns, oldest first, each Monday-first; approx tokens per day.
    pub grid: Vec<[u64; 7]>,
    /// Day index of the first (oldest) column's Monday.
    pub grid_start: i64,
    pub today: i64,
    pub first_day: Option<i64>,
    rows: Vec<Row>,
}

/// Approx tokens of a transcript: four bytes per token, the usual rule of thumb.
fn approx_tokens(bytes: u64) -> u64 {
    bytes / 4
}

/// Display name of the model family a transcript filename is tagged with.
fn model_name(entry: &SessionEntry) -> String {
    let name = entry
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let stem = name.strip_suffix(".kv").unwrap_or(name);
    match stem.rsplit('.').next().filter(|_| stem.contains('.')) {
        Some("ds4") => "DeepSeek V4 Flash",
        Some("ds41") => "DeepSeek V4.1 Flash",
        Some("qwen") => "Qwen3.8 Flash Next",
        _ => "unknown",
    }
    .to_string()
}

impl Stats {
    #[must_use]
    pub fn build(
        metas: &[SessionMeta],
        entries: &[SessionEntry],
        now: u64,
        tz_offset: i64,
    ) -> Self {
        let models: BTreeMap<&str, String> = entries
            .iter()
            .map(|e| (e.id.as_str(), model_name(e)))
            .collect();
        let rows: Vec<Row> = metas
            .iter()
            .map(|m| Row {
                day: day_of(m.created_at, tz_offset),
                tokens: approx_tokens(m.bytes),
                duration: m.duration_secs(),
                model: models
                    .get(m.id.as_str())
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string()),
            })
            .filter(|r| r.day <= day_of(now, tz_offset)) // clock-skewed future stamps must not exist
            .collect();
        let today = day_of(now, tz_offset);
        let this_monday = today - i64::try_from(weekday_mon0(today)).unwrap_or(0);
        let grid_start = this_monday - i64::try_from((WEEKS - 1) * 7).unwrap_or(0);
        let mut grid = vec![[0u64; 7]; WEEKS];
        for r in &rows {
            if r.day < grid_start || r.day > today {
                continue;
            }
            let off = usize::try_from(r.day - grid_start).unwrap_or(0);
            grid[off / 7][off % 7] += r.tokens;
        }
        let first_day = rows.iter().map(|r| r.day).min();
        Self {
            grid,
            grid_start,
            today,
            first_day,
            rows,
        }
    }

    #[must_use]
    pub fn figures(&self, scope: Scope) -> Figures {
        let cutoff = scope.window_days().map(|w| self.today - w + 1);
        let rows: Vec<&Row> = self
            .rows
            .iter()
            .filter(|r| cutoff.is_none_or(|c| r.day >= c))
            .collect();
        let mut per_day: BTreeMap<i64, u64> = BTreeMap::new();
        let mut per_model: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
        for r in &rows {
            *per_day.entry(r.day).or_default() += r.tokens;
            let e = per_model.entry(r.model.as_str()).or_default();
            e.0 += 1;
            e.1 += r.tokens;
        }
        let days: BTreeSet<i64> = per_day.keys().copied().collect();
        let (longest_streak, current_streak) = streaks(&days, self.today);
        let span_days = match scope.window_days() {
            Some(w) => w,
            None => self.first_day.map_or(0, |f| self.today - f + 1),
        };
        Figures {
            sessions: rows.len(),
            active_days: days.len(),
            span_days,
            most_active_day: per_day
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(a.0.cmp(b.0)))
                .map(|(d, _)| *d),
            longest_session_secs: rows.iter().map(|r| r.duration).max().unwrap_or(0),
            longest_streak,
            current_streak,
            favorite_model: per_model
                .iter()
                .max_by(|a, b| a.1.cmp(b.1))
                .map(|(m, _)| (*m).to_string()),
            approx_tokens: rows.iter().map(|r| r.tokens).sum(),
        }
    }
}

/// `(longest, current)` runs of consecutive days; the current run must end
/// today or yesterday.
fn streaks(days: &BTreeSet<i64>, today: i64) -> (usize, usize) {
    let mut longest = 0;
    let mut run = 0;
    let mut prev: Option<i64> = None;
    let mut run_end = i64::MIN;
    for &d in days {
        run = if prev == Some(d - 1) { run + 1 } else { 1 };
        prev = Some(d);
        run_end = d;
        longest = longest.max(run);
    }
    let current = if run_end >= today - 1 { run } else { 0 };
    (longest, current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insights::SessionMeta;
    use crate::session::SessionEntry;
    use std::path::PathBuf;

    const DAY: u64 = 86_400;

    fn meta(id: &str, created_at: u64, bytes: u64, dur: u64) -> SessionMeta {
        SessionMeta {
            id: id.to_string(),
            created_at,
            last_used: created_at + dur,
            bytes,
            ..Default::default()
        }
    }

    fn entry(id: &str, ext: &str) -> SessionEntry {
        SessionEntry {
            id: id.to_string(),
            path: PathBuf::from(format!("/x/{id}.{ext}.kv")),
            ..Default::default()
        }
    }

    #[test]
    fn epoch_day_zero_is_a_thursday() {
        assert_eq!(day_of(0, 0), 0);
        assert_eq!(weekday_mon0(0), 3);
        assert_eq!(weekday_mon0(4), 0); // 1970-01-05 was a Monday
    }

    #[test]
    fn civil_dates_round_trip_known_days() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(19_723), (2024, 1, 1));
        assert_eq!(civil(20_713), (2026, 9, 17));
    }

    #[test]
    fn tz_offset_moves_a_late_evening_into_the_next_day() {
        // 23:30 UTC with +2h is 01:30 the next local day.
        assert_eq!(day_of(23 * 3600 + 1800, 7200), 1);
    }

    #[test]
    fn scope_cycles_and_parses() {
        assert_eq!(Scope::AllTime.next(), Scope::Last7);
        assert_eq!(Scope::Last7.next(), Scope::Last30);
        assert_eq!(Scope::Last30.next(), Scope::AllTime);
        assert_eq!(Scope::parse("7"), Some(Scope::Last7));
        assert_eq!(Scope::parse(" 30 "), Some(Scope::Last30));
        assert_eq!(Scope::parse("all"), Some(Scope::AllTime));
        assert_eq!(Scope::parse(""), Some(Scope::AllTime));
        assert_eq!(Scope::parse("x"), None);
        for s in [Scope::AllTime, Scope::Last7, Scope::Last30] {
            assert_eq!(Scope::from_title(s.title()), Some(s));
        }
        assert_eq!(Scope::from_title("toks"), None);
    }

    #[test]
    fn today_lands_in_the_last_column_at_its_weekday() {
        let now = 20_713 * DAY + 3600; // 2026-09-17, a Thursday
        let st = Stats::build(&[meta("a", now, 400, 60)], &[], now, 0);
        assert_eq!(st.grid.len(), 53);
        assert_eq!(st.grid[52][3], 100);
        assert_eq!(st.first_day, Some(20_713));
    }

    #[test]
    fn a_session_fifty_two_weeks_ago_lands_in_the_first_column() {
        let now = 20_713 * DAY;
        let then = now - 52 * 7 * DAY; // same weekday, 52 weeks back
        let st = Stats::build(&[meta("a", then, 40, 0)], &[], now, 0);
        assert_eq!(st.grid[0][3], 10);
    }

    #[test]
    fn a_session_older_than_the_grid_still_counts_in_figures() {
        let now = 20_713 * DAY;
        let old = now - 60 * 7 * DAY;
        let st = Stats::build(&[meta("a", old, 40, 0)], &[], now, 0);
        assert!(st.grid.iter().all(|c| c.iter().all(|&v| v == 0)));
        assert_eq!(st.figures(Scope::AllTime).sessions, 1);
        assert_eq!(st.figures(Scope::Last30).sessions, 0);
    }

    #[test]
    fn streaks_count_consecutive_days_and_a_gap_resets() {
        let today = 20_713;
        let now = today * DAY + 100;
        let metas: Vec<_> = [today - 6, today - 5, today - 4, today - 2, today - 1, today]
            .iter()
            .enumerate()
            .map(|(i, d)| meta(&format!("s{i}"), d * DAY, 4, 0))
            .collect();
        let f = Stats::build(&metas, &[], now, 0).figures(Scope::AllTime);
        assert_eq!(f.longest_streak, 3);
        assert_eq!(f.current_streak, 3);
        assert_eq!(f.active_days, 6);
        assert_eq!(f.span_days, 7);
    }

    #[test]
    fn a_streak_ending_yesterday_is_current_but_two_days_ago_is_not() {
        let today: i64 = 20_713;
        let now = today.cast_unsigned() * DAY;
        let m = |d: i64| meta(&format!("d{d}"), d.cast_unsigned() * DAY, 4, 0);
        let f = Stats::build(&[m(today - 2), m(today - 1)], &[], now, 0).figures(Scope::AllTime);
        assert_eq!(f.current_streak, 2);
        let f = Stats::build(&[m(today - 3), m(today - 2)], &[], now, 0).figures(Scope::AllTime);
        assert_eq!(f.current_streak, 0);
        assert_eq!(f.longest_streak, 2);
    }

    #[test]
    fn scopes_filter_sessions_but_not_the_heatmap() {
        let today = 20_713;
        let now = today * DAY + 10;
        let metas = [
            meta("a", (today - 40) * DAY, 400, 10),
            meta("b", (today - 10) * DAY, 800, 20),
            meta("c", (today - 1) * DAY, 1200, 30),
        ];
        let st = Stats::build(&metas, &[], now, 0);
        assert_eq!(st.figures(Scope::AllTime).sessions, 3);
        assert_eq!(st.figures(Scope::Last30).sessions, 2);
        assert_eq!(st.figures(Scope::Last7).sessions, 1);
        assert_eq!(st.figures(Scope::Last7).approx_tokens, 300);
        assert_eq!(st.figures(Scope::Last7).span_days, 7);
        assert_eq!(st.figures(Scope::Last30).span_days, 30);
        let lit: usize = st.grid.iter().flatten().filter(|&&v| v > 0).count();
        assert_eq!(lit, 3);
    }

    #[test]
    fn most_active_day_longest_session_and_favorite_model() {
        let today: i64 = 20_713;
        let now = today.cast_unsigned() * DAY;
        let metas = [
            meta("a", (today - 3).cast_unsigned() * DAY, 400, 3600),
            meta("b", (today - 3).cast_unsigned() * DAY, 400, 7200),
            meta("c", (today - 1).cast_unsigned() * DAY, 400, 60),
        ];
        let entries = [entry("a", "ds4"), entry("b", "ds41"), entry("c", "ds41")];
        let f = Stats::build(&metas, &entries, now, 0).figures(Scope::AllTime);
        assert_eq!(f.most_active_day, Some(today - 3));
        assert_eq!(f.longest_session_secs, 7200);
        assert_eq!(f.favorite_model.as_deref(), Some("DeepSeek V4.1 Flash"));
    }

    #[test]
    fn a_session_dated_in_the_future_is_ignored_everywhere() {
        let today: i64 = 20_713;
        let now = today.cast_unsigned() * DAY;
        let metas = [
            meta("future", (today + 5).cast_unsigned() * DAY, 4000, 60),
            meta("todays", today.cast_unsigned() * DAY, 400, 30),
        ];
        let st = Stats::build(&metas, &[], now, 0);
        let f = st.figures(Scope::AllTime);
        assert_eq!(f.sessions, 1);
        assert_eq!(f.most_active_day, Some(today));
        assert_eq!(f.current_streak, 1);
        assert_eq!(f.approx_tokens, 100);
    }

    #[test]
    fn an_empty_store_yields_zeroes() {
        let st = Stats::build(&[], &[], 20_713 * DAY, 0);
        let f = st.figures(Scope::AllTime);
        assert_eq!(f.sessions, 0);
        assert_eq!(f.span_days, 0);
        assert_eq!(f.most_active_day, None);
        assert_eq!(f.favorite_model, None);
        assert_eq!(st.first_day, None);
    }
}
