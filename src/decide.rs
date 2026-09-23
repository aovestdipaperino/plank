//! System-1 typed decisions: turning the logprobs of a handful of known
//! answer-letter tokens into a typed Rust value with a probability attached.
//!
//! Nothing here touches the engine. The engine's job is to produce one
//! logprob per answer letter; everything from there — the softmax, the
//! abstention rules, the mapping back onto a caller's enum — is pure, so it
//! is exercised against fixed numbers with no model present.
//!
//! Design: `docs/superpowers/specs/2026-09-23-system-1-typed-decisions-design.md`.

/// The answer letters, in option order. Four is the cap on a question's
/// options; every letter must tokenize to exactly one token on the loaded
/// family or the whole capability reports unsupported
/// (`Engine::supports_decide`).
pub const LETTERS: [&str; 4] = ["A", "B", "C", "D"];

/// Below this probability the top letter is not trusted and the verdict
/// abstains. Callers treat an abstention as "no answer", never as a "no".
pub const DEFAULT_ABSTAIN_FLOOR: f32 = 0.55;

/// Below this share of total probability mass sitting on the answer letters,
/// the model was trying to say something other than a letter and its ranking
/// among the letters means little.
pub const MIN_LETTER_MASS: f32 = 0.10;

/// One question and its answer options, in letter order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    pub text: String,
    pub options: Vec<String>,
}

impl Question {
    /// A yes/no question. `yes` is index 0, so `p` reads as "probability of
    /// yes" at every call site.
    #[must_use]
    pub fn boolean(text: &str) -> Self {
        Self::choice(text, &["yes", "no"])
    }

    /// A multiple-choice question. Options beyond [`LETTERS`] are dropped
    /// rather than silently mis-lettered.
    #[must_use]
    pub fn choice(text: &str, options: &[&str]) -> Self {
        Self {
            text: text.to_string(),
            options: options
                .iter()
                .take(LETTERS.len())
                .map(|s| (*s).to_string())
                .collect(),
        }
    }
}

/// An answer before it is mapped onto a caller's type: which option won, how
/// confident the model was, and what came second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawVerdict {
    pub index: usize,
    pub p: f32,
    pub runner_up: Option<(usize, f32)>,
    /// True when the answer must not be acted on — either the top letter fell
    /// below the floor, or the letters together held too little mass.
    pub abstained: bool,
}

/// Softmax over `logprobs` (one per option, in option order) and the
/// abstention rules.
///
/// `letter_mass` is the share of the model's total probability that sat on
/// the answer letters; pass `0.0` when it is unknown, which disables that
/// check. The softmax is max-shifted, so logprobs far below zero rank
/// correctly instead of underflowing to a uniform tie.
#[must_use]
pub fn score(logprobs: &[f32], floor: f32, letter_mass: f32) -> RawVerdict {
    let abstain = RawVerdict {
        index: 0,
        p: 0.0,
        runner_up: None,
        abstained: true,
    };
    if logprobs.is_empty() || logprobs.iter().any(|v| !v.is_finite()) {
        return abstain;
    }
    let max = logprobs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logprobs.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return abstain;
    }

    let mut order: Vec<(usize, f32)> = exps.iter().enumerate().map(|(i, e)| (i, e / sum)).collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));

    let (index, p) = order[0];
    let runner_up = order.get(1).copied();
    let thin = letter_mass > 0.0 && letter_mass < MIN_LETTER_MASS;
    RawVerdict {
        index,
        p,
        runner_up,
        abstained: p < floor || thin,
    }
}

/// A Rust type a decision can answer with. Implemented by plain enums: the
/// option labels are what the model is shown, in letter order, and
/// `from_index` maps the winning letter back.
pub trait Decision: Sized + Copy {
    const OPTIONS: &'static [&'static str];
    fn from_index(i: usize) -> Option<Self>;

    /// The question asking for this type.
    #[must_use]
    fn question(text: &str) -> Question {
        Question::choice(text, Self::OPTIONS)
    }
}

/// A typed answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict<T> {
    pub value: T,
    pub p: f32,
    pub runner_up: Option<(T, f32)>,
    pub abstained: bool,
}

/// Maps a [`RawVerdict`] onto `T`, or `None` when the winning index is
/// outside what `T` covers — which means the question and the type disagreed
/// about the option list, a programming error rather than a model failure.
#[must_use]
pub fn typed<T: Decision>(raw: &RawVerdict) -> Option<Verdict<T>> {
    let value = T::from_index(raw.index)?;
    let runner_up = match raw.runner_up {
        Some((i, p)) => Some((T::from_index(i)?, p)),
        None => None,
    };
    Some(Verdict {
        value,
        p: raw.p,
        runner_up,
        abstained: raw.abstained,
    })
}

/// Asks one typed question and maps the answer back onto `T`.
///
/// This is the call site shape every consumer uses, wrapping the untyped
/// [`Engine::decide`] underneath it.
///
/// An index outside what `T` covers is an engine or programming fault, not a
/// model one, so it surfaces as an error rather than a silent abstention.
///
/// # Errors
/// Propagates the engine's error, plus an internal error if the engine
/// returned an index outside what `T` covers.
///
/// [`Engine::decide`]: crate::engine::Engine::decide
pub fn decide_one<T: Decision>(
    engine: &mut dyn crate::engine::Engine,
    state: &str,
    text: &str,
) -> Result<Verdict<T>, crate::engine::EngineError> {
    let raw = engine.decide(state, &T::question(text))?;
    typed(&raw).ok_or_else(|| crate::engine::EngineError::new("verdict index outside the type"))
}

/// Renders the question suffix the engine evaluates after the state.
///
/// The trailing `Answer: ` is load-bearing: it leaves the model exactly one
/// token to place, which is the token whose distribution we read. Written as
/// separate pushes rather than one literal because a `\`-continued Rust
/// string literal would strip the indentation of model-facing text.
#[must_use]
pub fn render_question(q: &Question) -> String {
    let mut out = String::with_capacity(128 + q.text.len());
    out.push_str("\n\n");
    out.push_str(&q.text);
    out.push('\n');
    for (i, opt) in q.options.iter().enumerate() {
        let letter = LETTERS.get(i).copied().unwrap_or("?");
        out.push_str(letter);
        out.push_str(". ");
        out.push_str(opt);
        out.push('\n');
    }
    out.push_str("Reply with a single letter.\n");
    out.push_str("Answer: ");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Worthy {
        Yes,
        No,
    }

    impl Decision for Worthy {
        const OPTIONS: &'static [&'static str] = &["yes", "no"];
        fn from_index(i: usize) -> Option<Self> {
            match i {
                0 => Some(Self::Yes),
                1 => Some(Self::No),
                _ => None,
            }
        }
    }

    #[test]
    fn score_picks_the_highest_letter_and_normalises_over_letters_only() {
        // Two letters, second clearly ahead: logprobs -2.0 and -0.1.
        let v = score(&[-2.0, -0.1], DEFAULT_ABSTAIN_FLOOR, 0.0);
        assert_eq!(v.index, 1);
        assert!(v.p > 0.85, "p was {}", v.p);
        assert_eq!(v.runner_up.map(|(i, _)| i), Some(0));
        assert!(!v.abstained);
        // Softmax over just the letters, so the pair sums to 1.
        let (_, runner_p) = v.runner_up.unwrap();
        assert!((v.p + runner_p - 1.0).abs() < 1e-5);
    }

    #[test]
    fn score_abstains_when_the_top_letter_is_below_the_floor() {
        // Near-tie: both ~0.5, under the 0.55 floor.
        let v = score(&[-0.7, -0.7], DEFAULT_ABSTAIN_FLOOR, 0.0);
        assert!(v.abstained);
    }

    #[test]
    fn score_abstains_when_the_mass_sits_outside_the_letters() {
        // Letters are confident relative to each other, but together they
        // hold only 2% of the model's probability: it wanted to say something
        // else entirely, so the answer is not trustworthy.
        let v = score(&[-0.01, -5.0], DEFAULT_ABSTAIN_FLOOR, 0.02);
        assert!(v.abstained, "low letter mass must abstain");
    }

    #[test]
    fn score_is_not_fooled_by_large_negative_logprobs() {
        // Naive exp() without a max-shift underflows here; the shifted
        // softmax must still rank correctly.
        let v = score(&[-900.0, -899.0], DEFAULT_ABSTAIN_FLOOR, 0.0);
        assert_eq!(v.index, 1);
        assert!(v.p.is_finite());
    }

    #[test]
    fn score_of_an_empty_slice_abstains_rather_than_panicking() {
        let v = score(&[], DEFAULT_ABSTAIN_FLOOR, 0.0);
        assert!(v.abstained);
    }

    #[test]
    fn typed_maps_the_index_back_to_the_enum() {
        let raw = score(&[-0.05, -3.0], DEFAULT_ABSTAIN_FLOOR, 0.0);
        let v: Verdict<Worthy> = typed(&raw).expect("index 0 is in range");
        assert_eq!(v.value, Worthy::Yes);
        assert_eq!(v.runner_up.map(|(t, _)| t), Some(Worthy::No));
    }

    #[test]
    fn typed_rejects_an_index_the_enum_does_not_cover() {
        let raw = RawVerdict {
            index: 7,
            p: 1.0,
            runner_up: None,
            abstained: false,
        };
        assert!(typed::<Worthy>(&raw).is_none());
    }

    #[test]
    fn render_question_lists_the_options_as_letters_and_ends_ready_for_one() {
        let q = Question::boolean("Is this worth remembering?");
        let text = render_question(&q);
        assert!(text.contains("A. yes"));
        assert!(text.contains("B. no"));
        assert!(
            text.ends_with("Answer: "),
            "the suffix must leave the model one token to place, got {text:?}"
        );
    }

    #[test]
    fn a_question_is_capped_at_the_available_letters() {
        let q = Question::choice("pick", &["a", "b", "c", "d", "e"]);
        assert_eq!(q.options.len(), LETTERS.len(), "extra options are dropped");
    }
}
