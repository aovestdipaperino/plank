// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Destructive-git detection for the `bash` tool (`tools.gitGuard`).
//!
//! A pure, quote-aware reader of a shell command line that reports the first
//! git operation in it that throws work away or rewrites history: a hard
//! reset, a forced clean or push, `branch -D`, a checkout or restore that
//! overwrites the working tree, a rebase, `stash drop`, and a few more. The
//! bash tool asks the user (or refuses, headless) before running such a
//! command.
//!
//! This catches accidents, not adversaries. The Seatbelt sandbox remains the
//! containment boundary; a command built to hide a reset from this scanner
//! (a script file, an alias, an env-var-built command name) will get past it.
//! What it does resolve: quoting (`echo "git reset --hard"` is data), chains
//! (`a && git reset --hard`, `;`, `|`, `&`, newlines, subshells), command and
//! backtick substitution, `sh -c '...'` / `eval ...`, leading env assignments
//! and wrappers (`sudo`, `env`, `time`), and git's own global options
//! (`git -C dir -c k=v reset --hard`).

/// What the bash tool does when [`destructive`] flags a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Ask the user; with nobody to ask (headless) the command is refused.
    #[default]
    Ask,
    /// Refuse without asking.
    Block,
    /// Run it unchecked.
    Off,
}

impl Mode {
    /// Parses a `tools.gitGuard` value (`ask`, `block`, `off`), ignoring case.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ask" => Some(Self::Ask),
            "block" => Some(Self::Block),
            "off" => Some(Self::Off),
            _ => None,
        }
    }

    /// The settings spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Block => "block",
            Self::Off => "off",
        }
    }
}

/// How deep `sh -c` / `eval` / substitution nesting is followed.
const MAX_DEPTH: usize = 4;

/// Words that wrap or prefix a command without being it.
const WRAPPERS: &[&str] = &[
    "sudo", "command", "env", "nice", "nohup", "time", "exec", "builtin", "if", "then", "else",
    "elif", "do", "while", "until", "!", "{",
];

/// Shells whose `-c` argument is a command line of its own.
const SHELLS: &[&str] = &["sh", "bash", "zsh", "dash", "ksh"];

/// The reason the first destructive git operation in `cmd` is destructive, or
/// `None` when there is none.
#[must_use]
pub fn destructive(cmd: &str) -> Option<&'static str> {
    scan(cmd, 0)
}

fn scan(cmd: &str, depth: usize) -> Option<&'static str> {
    if depth > MAX_DEPTH {
        return None;
    }
    let lexed = lex(cmd);
    for sub in &lexed.subs {
        if let Some(why) = scan(sub, depth + 1) {
            return Some(why);
        }
    }
    lexed
        .segments
        .iter()
        .find_map(|words| segment(words, depth))
}

/// The segments of one command line and the bodies of its substitutions.
#[derive(Default)]
struct Lexed {
    /// Each simple command as its unquoted words.
    segments: Vec<Vec<String>>,
    /// `$(...)` and backtick bodies, scanned as command lines of their own.
    subs: Vec<String>,
}

/// Splits `cmd` into simple commands at unquoted `; & | ( )` and newlines,
/// removing quotes and backslashes from the words, and collects substitution
/// bodies (live inside double quotes, inert inside single quotes).
fn lex(cmd: &str) -> Lexed {
    let chars: Vec<char> = cmd.chars().collect();
    let mut out = Lexed::default();
    let mut words: Vec<String> = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let (mut single, mut double) = (false, false);
    let mut i = 0;
    let end_word = |words: &mut Vec<String>, word: &mut String, in_word: &mut bool| {
        if *in_word {
            words.push(std::mem::take(word));
            *in_word = false;
        }
    };
    while i < chars.len() {
        let c = chars[i];
        if single {
            if c == '\'' {
                single = false;
            } else {
                word.push(c);
            }
            i += 1;
            continue;
        }
        match c {
            '\\' => {
                in_word = true;
                if let Some(&next) = chars.get(i + 1) {
                    word.push(next);
                    i += 1;
                }
            }
            '\'' if !double => {
                single = true;
                in_word = true;
            }
            '"' => {
                double = !double;
                in_word = true;
            }
            '$' if chars.get(i + 1) == Some(&'(') => {
                let (body, next) = paren_body(&chars, i + 2);
                out.subs.push(body);
                i = next;
                continue;
            }
            '`' => {
                let close = chars[i + 1..].iter().position(|&ch| ch == '`');
                let stop = close.map_or(chars.len(), |p| i + 1 + p);
                out.subs.push(chars[i + 1..stop].iter().collect());
                i = stop + 1;
                continue;
            }
            c if double => {
                word.push(c);
                in_word = true;
            }
            c if c.is_whitespace() && c != '\n' => end_word(&mut words, &mut word, &mut in_word),
            ';' | '&' | '|' | '(' | ')' | '\n' => {
                end_word(&mut words, &mut word, &mut in_word);
                if !words.is_empty() {
                    out.segments.push(std::mem::take(&mut words));
                }
            }
            c => {
                word.push(c);
                in_word = true;
            }
        }
        i += 1;
    }
    end_word(&mut words, &mut word, &mut in_word);
    if !words.is_empty() {
        out.segments.push(words);
    }
    out
}

/// The text up to the `)` closing a `$(` whose body starts at `from`, and the
/// index just past it. An unterminated body runs to the end.
fn paren_body(chars: &[char], from: usize) -> (String, usize) {
    let mut depth = 1usize;
    let mut j = from;
    while j < chars.len() {
        match chars[j] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return (chars[from..j].iter().collect(), j + 1);
                }
            }
            _ => {}
        }
        j += 1;
    }
    (chars[from..].iter().collect(), chars.len())
}

/// Whether `w` is a leading `NAME=value` environment assignment.
fn is_env_assignment(w: &str) -> bool {
    w.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty()
            && !name.starts_with(|c: char| c.is_ascii_digit())
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

fn basename(w: &str) -> &str {
    w.rsplit('/').next().unwrap_or(w)
}

fn segment(words: &[String], depth: usize) -> Option<&'static str> {
    let mut i = 0;
    while let Some(w) = words.get(i) {
        if is_env_assignment(w) {
            i += 1;
        } else if WRAPPERS.contains(&basename(w)) {
            i += 1;
            // A wrapper's own flags (`sudo -n`, `env -i`) are not the command.
            while words.get(i).is_some_and(|f| f.starts_with('-')) {
                i += 1;
            }
        } else {
            break;
        }
    }
    let name = basename(words.get(i)?);
    let rest = &words[i + 1..];
    if name == "git" {
        return git(rest);
    }
    if name == "eval" {
        return scan(&rest.join(" "), depth + 1);
    }
    if SHELLS.contains(&name) {
        let at = rest
            .iter()
            .position(|w| w.starts_with('-') && !w.starts_with("--") && w.ends_with('c'))?;
        return scan(rest.get(at + 1)?, depth + 1);
    }
    None
}

/// Global options that take their value as the next word.
const GLOBAL_WITH_VALUE: &[&str] = &["-C", "-c", "--git-dir", "--work-tree", "--namespace"];

fn git(args: &[String]) -> Option<&'static str> {
    let mut i = 0;
    while let Some(a) = args.get(i) {
        if GLOBAL_WITH_VALUE.contains(&a.as_str()) {
            i += 2;
        } else if a.starts_with('-') {
            i += 1;
        } else {
            break;
        }
    }
    let sub = args.get(i)?.as_str();
    let rest: Vec<&str> = args[i + 1..].iter().map(String::as_str).collect();
    let o = Opts::new(&rest);
    match sub {
        "reset" if o.long("--hard") || o.long("--merge") => {
            Some("git reset --hard throws away uncommitted changes")
        }
        "clean"
            if (o.long("--force") || o.short('f')) && !(o.long("--dry-run") || o.short('n')) =>
        {
            Some("git clean -f deletes untracked files for good")
        }
        "push" if o.forces_push() => Some("this git push force-updates or deletes remote refs"),
        "branch"
            if o.short('D')
                || ((o.long("--force") || o.short('f'))
                    && (o.long("--delete") || o.short('d'))) =>
        {
            Some("git branch -D deletes a branch even when it is unmerged")
        }
        "checkout" if o.after_dashes || o.long("--force") || o.short('f') || o.arg(".") => {
            Some("git checkout overwrites uncommitted changes in the working tree")
        }
        "switch" if o.long("--discard-changes") || o.long("--force") || o.short('f') => {
            Some("git switch --discard-changes throws away uncommitted changes")
        }
        "restore"
            if o.long("--worktree") || o.short('W') || !(o.long("--staged") || o.short('S')) =>
        {
            Some("git restore overwrites uncommitted changes in the working tree")
        }
        "rebase"
            if ![
                "--abort",
                "--continue",
                "--skip",
                "--quit",
                "--show-current-patch",
            ]
            .iter()
            .any(|f| o.long(f)) =>
        {
            Some("git rebase rewrites history")
        }
        "stash" if o.first_arg().is_some_and(|a| a == "drop" || a == "clear") => {
            Some("git stash drop/clear discards stashed work for good")
        }
        "filter-branch" | "filter-repo" => Some("git filter-branch rewrites history"),
        "reflog"
            if o.first_arg()
                .is_some_and(|a| a == "expire" || a == "delete") =>
        {
            Some("git reflog expire/delete removes the recovery trail")
        }
        "gc" if o.long("--prune=now") => {
            Some("git gc --prune=now permanently drops unreachable commits")
        }
        "update-ref" if o.short('d') => Some("git update-ref -d deletes a ref"),
        _ => None,
    }
}

/// The words after a git subcommand, split at a bare `--` into flags and
/// pathspecs.
struct Opts<'a> {
    /// Words before any `--`.
    before: &'a [&'a str],
    /// Whether a `--` was present.
    after_dashes: bool,
}

impl<'a> Opts<'a> {
    fn new(rest: &'a [&'a str]) -> Self {
        match rest.iter().position(|w| *w == "--") {
            Some(p) => Self {
                before: &rest[..p],
                after_dashes: true,
            },
            None => Self {
                before: rest,
                after_dashes: false,
            },
        }
    }

    /// A long flag, exactly or as `--flag=value`.
    fn long(&self, flag: &str) -> bool {
        self.before
            .iter()
            .any(|w| *w == flag || w.strip_prefix(flag).is_some_and(|r| r.starts_with('=')))
    }

    /// A short flag, alone or inside a cluster (`-fd`).
    fn short(&self, c: char) -> bool {
        self.before
            .iter()
            .any(|w| w.starts_with('-') && !w.starts_with("--") && w[1..].contains(c))
    }

    /// A non-flag word equal to `s`.
    fn arg(&self, s: &str) -> bool {
        self.before.contains(&s)
    }

    /// The first word that is not a flag.
    fn first_arg(&self) -> Option<&str> {
        self.before.iter().copied().find(|w| !w.starts_with('-'))
    }

    /// `push` flags or refspecs that overwrite or delete remote refs.
    fn forces_push(&self) -> bool {
        self.long("--force")
            || self.long("--force-with-lease")
            || self.long("--mirror")
            || self.long("--delete")
            || self.short('f')
            || self.short('d')
            || self.before.iter().any(|w| {
                !w.starts_with('-') && (w.starts_with('+') || w.starts_with(':')) && w.len() > 1
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(cmd: &str) {
        assert!(destructive(cmd).is_some(), "should flag: {cmd}");
    }

    fn pass(cmd: &str) {
        assert_eq!(destructive(cmd), None, "should pass: {cmd}");
    }

    #[test]
    fn flags_the_destructive_operations() {
        hit("git reset --hard");
        hit("git reset --hard HEAD~3");
        hit("git reset --merge");
        hit("git clean -fd");
        hit("git clean --force -x");
        hit("git push --force origin main");
        hit("git push -f");
        hit("git push --force-with-lease");
        hit("git push origin +main");
        hit("git push origin :old-branch");
        hit("git push --delete origin old");
        hit("git branch -D feature");
        hit("git branch -df feature");
        hit("git checkout -- .");
        hit("git checkout .");
        hit("git checkout -f main");
        hit("git checkout main -- src/lib.rs");
        hit("git switch --discard-changes main");
        hit("git restore src/lib.rs");
        hit("git restore --worktree --staged src");
        hit("git rebase main");
        hit("git stash drop");
        hit("git stash clear");
        hit("git filter-branch --tree-filter x");
        hit("git reflog expire --expire=now --all");
        hit("git gc --prune=now");
        hit("git update-ref -d refs/heads/x");
    }

    #[test]
    fn leaves_the_safe_operations_alone() {
        pass("git status");
        pass("git reset");
        pass("git reset HEAD file");
        pass("git reset --soft HEAD~1");
        pass("git clean -n");
        pass("git clean -nfd");
        pass("git clean --dry-run -f");
        pass("git push origin main");
        pass("git push -u origin feature");
        pass("git branch -d merged");
        pass("git branch feature");
        pass("git checkout main");
        pass("git checkout -b topic");
        pass("git switch topic");
        pass("git restore --staged file");
        pass("git rebase --abort");
        pass("git rebase --continue");
        pass("git stash");
        pass("git stash pop");
        pass("git stash list");
        pass("git commit -am msg");
        pass("git log --oneline");
        pass("git diff -- .");
        pass("cargo build");
        pass("ls -la");
    }

    #[test]
    fn quoted_text_is_data_not_a_command() {
        pass(r#"echo "git reset --hard""#);
        pass("echo 'git push --force' > notes.txt");
        pass(r#"grep "git clean -fd" README.md"#);
        pass(r#"git commit -m "never run git reset --hard""#);
    }

    #[test]
    fn finds_a_command_inside_a_chain() {
        hit("cargo test && git reset --hard");
        hit("ls; git clean -fd");
        hit("make || git checkout .");
        hit("echo done | git reset --hard");
        hit("sleep 1 & git reset --hard");
        hit("echo a\ngit reset --hard");
        hit("(git reset --hard)");
        hit("if true; then git reset --hard; fi");
        hit("{ git reset --hard; }");
    }

    #[test]
    fn sees_through_prefixes_and_global_options() {
        hit("sudo git reset --hard");
        hit("FOO=1 git reset --hard");
        hit("env -i FOO=1 git reset --hard");
        hit("/usr/bin/git reset --hard");
        hit("git -C /tmp/repo reset --hard");
        hit("git -c core.x=1 --no-pager reset --hard");
        hit("git --git-dir=/x/.git reset --hard");
        hit("time git clean -fd");
    }

    #[test]
    fn follows_substitution_eval_and_shell_c() {
        hit("echo $(git reset --hard)");
        hit("echo `git clean -fd`");
        hit(r#"echo "$(git reset --hard)""#);
        hit("bash -c 'git reset --hard'");
        hit(r#"sh -c "cd x && git clean -fd""#);
        hit("bash -lc 'git push --force'");
        hit("eval git reset --hard");
        hit("eval 'git reset --hard'");
        pass("echo '$(git reset --hard)'");
        pass("bash -c 'git status'");
    }

    #[test]
    fn cannot_loop_on_deep_nesting() {
        let mut cmd = "git reset --hard".to_string();
        for _ in 0..10 {
            cmd = format!("bash -c '{}'", cmd.replace('\'', "'\\''"));
        }
        // Past the depth limit it gives up quietly rather than recursing.
        let _ = destructive(&cmd);
    }

    #[test]
    fn handles_odd_input_without_panicking() {
        for cmd in [
            "",
            " ",
            "git",
            "git -C",
            "git -c",
            "$(",
            "`",
            "'",
            "\"",
            "\\",
            "git push +",
        ] {
            let _ = destructive(cmd);
        }
        pass("git push origin +");
    }

    #[test]
    fn mode_parses_its_spellings() {
        assert_eq!(Mode::parse("ask"), Some(Mode::Ask));
        assert_eq!(Mode::parse(" BLOCK "), Some(Mode::Block));
        assert_eq!(Mode::parse("off"), Some(Mode::Off));
        assert_eq!(Mode::parse("sometimes"), None);
        assert_eq!(Mode::default(), Mode::Ask);
        for m in [Mode::Ask, Mode::Block, Mode::Off] {
            assert_eq!(Mode::parse(m.as_str()), Some(m));
        }
    }
}
