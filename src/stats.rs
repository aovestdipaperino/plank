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
        let today = day_of(now, tz_offset);
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
            .filter(|r| r.day <= today) // clock-skewed future stamps must not exist
            .collect();
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

/// Heatmap shades, dimmest first; the last is `THEME_GREEN`.
pub const SHADES: [u8; 4] = [22, 28, 71, 114];
const DIM: u8 = 238;
const MUTED: u8 = 245;
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

fn fg(color: bool, idx: u8, text: &str) -> String {
    if color {
        format!("\x1b[38;5;{idx}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn bold_green(color: bool, text: &str) -> String {
    if color {
        format!("\x1b[1;38;5;{}m{text}\x1b[0m", SHADES[3])
    } else {
        text.to_string()
    }
}

/// `k`/`m`/`b` with one decimal above 999.
#[must_use]
#[allow(clippy::cast_precision_loss)] // token counts are already approximate
pub fn fmt_count(n: u64) -> String {
    let f = n as f64;
    if n >= 1_000_000_000 {
        format!("{:.1}b", f / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}m", f / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", f / 1e3)
    } else {
        n.to_string()
    }
}

/// `Nd Nh Nm`, dropping leading zero units; never empty.
#[must_use]
pub fn fmt_duration(secs: u64) -> String {
    let d = secs / 86_400;
    let h = secs % 86_400 / 3_600;
    let m = secs % 3_600 / 60;
    if d > 0 {
        format!("{d}d {h}h {m}m")
    } else if h > 0 {
        format!("{h}h {m}m")
    } else {
        format!("{m}m")
    }
}

fn fmt_day(day: i64) -> String {
    let (_, m, d) = civil(day);
    format!("{} {d}", MONTHS[usize::try_from(m).unwrap_or(1) - 1])
}

/// Shade index for a day's tokens given the sorted non-zero daily totals.
///
/// The ranks are spread over `len - 1` so the busiest day always takes the
/// brightest shade: scaling by `len` leaves [`SHADES`]`[3]` unreachable below
/// four distinct totals, which is most of a new user's history.
fn shade_for(tokens: u64, sorted: &[u64]) -> usize {
    if sorted.len() < 2 || sorted.first() == sorted.last() {
        return 3;
    }
    let rank = sorted.partition_point(|&v| v < tokens);
    (rank * 3 / (sorted.len() - 1)).min(3)
}

fn render_grid(stats: &Stats, color: bool) -> Vec<String> {
    const GUTTER: &str = "     ";
    let mut sorted: Vec<u64> = stats
        .grid
        .iter()
        .flatten()
        .copied()
        .filter(|&v| v > 0)
        .collect();
    sorted.sort_unstable();
    let mut lines = Vec::with_capacity(9);

    let mut header = String::from(GUTTER);
    let mut last_month = 0;
    for (col, _) in stats.grid.iter().enumerate() {
        let monday = stats.grid_start + i64::try_from(col * 7).unwrap_or(0);
        let (_, m, _) = civil(monday);
        if m != last_month && header.len() <= GUTTER.len() + col * 2 {
            header.push_str(MONTHS[usize::try_from(m).unwrap_or(1) - 1]);
            header.push(' ');
            last_month = m;
        } else if header.len() < GUTTER.len() + col * 2 + 2 {
            while header.len() < GUTTER.len() + col * 2 + 2 {
                header.push(' ');
            }
        }
    }
    lines.push(header.trim_end().to_string());

    for row in 0..7 {
        let label = match row {
            0 => "Mon  ",
            2 => "Wed  ",
            4 => "Fri  ",
            _ => GUTTER,
        };
        let mut line = String::from(label);
        for (col, week) in stats.grid.iter().enumerate() {
            let day = stats.grid_start + i64::try_from(col * 7 + row).unwrap_or(0);
            let cell = if day > stats.today {
                " ".to_string()
            } else if week[row] == 0 {
                fg(color, DIM, "·")
            } else {
                fg(color, SHADES[shade_for(week[row], &sorted)], "■")
            };
            line.push_str(&cell);
            line.push(' ');
        }
        lines.push(line.trim_end().to_string());
    }
    lines
}

/// The `Less ■ ■ ■ ■ More` legend explaining the heatmap's shading.
fn render_legend(color: bool) -> String {
    const GUTTER: &str = "     ";
    let mut legend = String::from(GUTTER);
    legend.push_str("Less ");
    for s in SHADES {
        legend.push_str(&fg(color, s, "■"));
        legend.push(' ');
    }
    legend.push_str("More");
    legend
}

/// The whole `/stats` report as text, ANSI-colored when `color`, with the
/// panel footer hint when `hint`.
#[must_use]
pub fn render(stats: &Stats, scope: Scope, color: bool, hint: bool) -> String {
    let f = stats.figures(scope);
    let mut out = render_grid(stats, color);
    out.push(render_legend(color));
    out.push(String::new());

    let scopes: Vec<String> = [Scope::AllTime, Scope::Last7, Scope::Last30]
        .iter()
        .map(|s| {
            if *s == scope {
                bold_green(color, s.label())
            } else {
                fg(color, MUTED, s.label())
            }
        })
        .collect();
    out.push(scopes.join(&fg(color, MUTED, " · ")));
    out.push(String::new());

    let val = |s: &str| fg(color, SHADES[3], s);
    let dash = "-".to_string();
    let left = [
        format!(
            "Favorite model: {}",
            val(f.favorite_model.as_deref().unwrap_or("-"))
        ),
        format!("Sessions: {}", val(&f.sessions.to_string())),
        format!(
            "Active days: {}{}",
            val(&f.active_days.to_string()),
            fg(color, MUTED, &format!("/{}", f.span_days))
        ),
        format!(
            "Most active day: {}",
            val(&f.most_active_day.map_or(dash.clone(), fmt_day))
        ),
    ];
    let right = [
        format!(
            "Total tokens: {} {}",
            val(&fmt_count(f.approx_tokens)),
            fg(color, MUTED, "(approx)")
        ),
        format!(
            "Longest session: {}",
            val(&fmt_duration(f.longest_session_secs))
        ),
        format!(
            "Longest streak: {} days",
            val(&f.longest_streak.to_string())
        ),
        format!(
            "Current streak: {} days",
            val(&f.current_streak.to_string())
        ),
    ];
    // The right column starts past the widest left cell, never at a fixed
    // offset: "Favorite model: DeepSeek V4.1 Flash" overruns 34 columns and
    // would otherwise run straight into "Total tokens:".
    let col = left
        .iter()
        .map(|l| visible_len(l) + 2)
        .max()
        .unwrap_or(0)
        .max(34);
    for (l, r) in left.iter().zip(right.iter()) {
        out.push(format!("{l}{}{r}", " ".repeat(col - visible_len(l))));
    }
    out.push(fg(
        color,
        MUTED,
        "tokens are approximate (transcript bytes / 4)",
    ));
    if hint {
        out.push(String::new());
        out.push(fg(color, MUTED, "/stats cycles range · Esc closes"));
    }
    let mut s = out.join("\n");
    s.push('\n');
    s
}

/// Characters shown once ANSI escapes are dropped.
fn visible_len(s: &str) -> usize {
    let mut n = 0;
    let mut in_esc = false;
    for c in s.chars() {
        match (in_esc, c) {
            (true, 'm') => in_esc = false,
            (true, _) => {}
            (false, '\x1b') => in_esc = true,
            (false, _) => n += 1,
        }
    }
    n
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

    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn counts_and_durations_format_compactly() {
        assert_eq!(fmt_count(0), "0");
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(1_500), "1.5k");
        assert_eq!(fmt_count(34_300_000_000), "34.3b");
        assert_eq!(fmt_count(119_200_000), "119.2m");
        assert_eq!(fmt_duration(0), "0m");
        assert_eq!(fmt_duration(59), "0m");
        assert_eq!(fmt_duration(3_600 + 120), "1h 2m");
        assert_eq!(
            fmt_duration(31 * 86_400 + 20 * 3_600 + 4 * 60),
            "31d 20h 4m"
        );
    }

    #[test]
    fn plain_render_has_no_escapes_and_names_every_figure() {
        let today = 20_713;
        let now = today * DAY;
        let metas = [meta("a", (today - 1) * DAY, 4_000, 3_600)];
        let st = Stats::build(&metas, &[entry("a", "ds4")], now, 0);
        let out = render(&st, Scope::AllTime, false, true);
        assert!(!out.contains('\x1b'));
        for needle in [
            "Favorite model: DeepSeek V4 Flash",
            "Total tokens: 1.0k (approx)",
            "Sessions: 1",
            "Active days: 1/2",
            "Most active day: Sep 16",
            "Longest session: 1h 0m",
            "Longest streak: 1 days",
            "Current streak: 1 days",
            "tokens are approximate (transcript bytes / 4)",
            "/stats cycles range · Esc closes",
            "Less ",
            " More",
            "Mon",
            "Wed",
            "Fri",
        ] {
            assert!(out.contains(needle), "missing {needle:?} in:\n{out}");
        }
        assert!(!render(&st, Scope::AllTime, false, false).contains("Esc closes"));
    }

    #[test]
    fn the_active_scope_is_marked_and_the_others_listed() {
        let st = Stats::build(&[], &[], 20_713 * DAY, 0);
        let out = strip_ansi(&render(&st, Scope::Last7, true, false));
        assert!(out.contains("All time · Last 7 days · Last 30 days"));
        assert!(out.contains("Most active day: -"));
        assert!(out.contains("Favorite model: -"));
    }

    #[test]
    fn colored_render_uses_only_green_shades_and_dim_greys() {
        let today = 20_713;
        let now = today * DAY;
        let metas: Vec<_> = (0..8)
            .map(|i| meta(&format!("s{i}"), (today - i) * DAY, 400 * (i + 1), 0))
            .collect();
        let st = Stats::build(&metas, &[], now, 0);
        let out = render(&st, Scope::AllTime, true, true);
        let mut seen = std::collections::BTreeSet::new();
        for part in out.split("\x1b[38;5;").skip(1) {
            let n: u32 = part.split('m').next().unwrap().parse().unwrap();
            seen.insert(n);
        }
        for n in &seen {
            assert!(
                [22, 28, 71, 114, 238, 245].contains(n),
                "unexpected color index {n}"
            );
        }
        for shade in SHADES {
            assert!(seen.contains(&u32::from(shade)), "shade {shade} unused");
        }
    }

    #[test]
    fn a_single_active_day_gets_the_brightest_shade() {
        let today = 20_713;
        let st = Stats::build(&[meta("a", today * DAY, 40, 0)], &[], today * DAY, 0);
        let out = render(&st, Scope::AllTime, true, false);
        let grid = out
            .lines()
            .take_while(|l| !l.contains("Less"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(grid.contains(&format!("\x1b[38;5;{}m■", SHADES[3])));
        assert!(!grid.contains(&format!("\x1b[38;5;{}m■", SHADES[0])));
    }

    #[test]
    fn long_model_name_does_not_collide_with_the_right_column() {
        let today = 20_713;
        let now = today * DAY;
        let metas = [meta("a", (today - 1) * DAY, 4_000, 3_600)];
        let st = Stats::build(&metas, &[entry("a", "ds41")], now, 0);
        let out = render(&st, Scope::AllTime, false, true);
        assert!(out.contains("Favorite model: DeepSeek V4.1 Flash"));
        assert!(out.contains("  Total tokens:"), "columns collided:\n{out}");
    }

    #[test]
    fn busiest_day_is_always_the_brightest_shade() {
        for totals in [
            vec![10u64, 40],
            vec![10u64, 20, 40],
            vec![10u64, 20, 30, 40],
        ] {
            let today: i64 = 20_713;
            let now = today.cast_unsigned() * DAY;
            let metas: Vec<_> = totals
                .iter()
                .enumerate()
                .map(|(i, &t)| {
                    let day = today - i64::try_from(i).unwrap_or(0);
                    meta(&format!("s{i}"), day.cast_unsigned() * DAY, t * 4, 0)
                })
                .collect();
            let st = Stats::build(&metas, &[], now, 0);
            let out = render(&st, Scope::AllTime, true, false);
            let grid = out
                .lines()
                .take_while(|l| !l.contains("Less"))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                grid.contains(&format!("\x1b[38;5;{}m■", SHADES[3])),
                "brightest shade missing for {totals:?}:\n{grid}"
            );
            // The quietest day earns the dimmest shade; what it must never do
            // is share the brightest one with the busiest day.
            let brightest = grid.matches(&format!("\x1b[38;5;{}m■", SHADES[3])).count();
            assert_eq!(brightest, 1, "only the busiest day is brightest {totals:?}");
            assert!(
                grid.contains(&format!("\x1b[38;5;{}m■", SHADES[0])),
                "quietest day should take the dimmest shade in {totals:?}:\n{grid}"
            );
        }
    }

    #[test]
    fn month_header_never_drops_a_label_across_many_years() {
        for today in [18_000i64, 18_500, 19_200, 19_900, 20_400, 20_713] {
            let st = Stats::build(&[], &[], today.cast_unsigned() * DAY, 0);
            let out = render(&st, Scope::AllTime, false, false);
            let header = out.lines().next().unwrap();
            let months: Vec<&str> = header.split_whitespace().collect();
            assert!(
                months.len() == 12 || months.len() == 13,
                "today={today}: got {} labels: {months:?}",
                months.len()
            );
            for w in months.windows(2) {
                assert_ne!(
                    w[0], w[1],
                    "today={today}: adjacent duplicate in {months:?}"
                );
            }
            // Expected calendar sequence starting from the grid's first column.
            let this_monday = today - i64::try_from(weekday_mon0(today)).unwrap_or(0);
            let grid_start = this_monday - i64::try_from((WEEKS - 1) * 7).unwrap_or(0);
            let mut expected = Vec::new();
            let mut last_m = 0u32;
            for col in 0..WEEKS {
                let monday = grid_start + i64::try_from(col * 7).unwrap_or(0);
                let (_, m, _) = civil(monday);
                if m != last_m {
                    expected.push(MONTHS[usize::try_from(m).unwrap_or(1) - 1]);
                    last_m = m;
                }
            }
            assert_eq!(months, expected, "today={today}");
        }
    }

    #[test]
    fn month_labels_appear_in_calendar_order_across_a_year_boundary() {
        let today = 20_713; // 2026-09-17
        let st = Stats::build(&[], &[], today * DAY, 0);
        let out = render(&st, Scope::AllTime, false, false);
        let header = out.lines().next().unwrap();
        let months: Vec<&str> = header.split_whitespace().collect();
        assert_eq!(months.first(), Some(&"Sep"));
        assert_eq!(months.last(), Some(&"Sep"));
        let pos = |m: &str| months.iter().position(|x| *x == m).unwrap();
        assert!(pos("Dec") < pos("Jan"));
        assert!(pos("Jan") < pos("Feb"));
        assert_eq!(months.len(), 13);
    }
}
