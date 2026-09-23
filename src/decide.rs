//! System-1 typed decisions: turning the logprobs of a handful of known
//! answer-letter tokens into a typed Rust value with a probability attached.
//!
//! Nothing here touches the engine. The engine's job is to produce one
//! logprob per answer letter; everything from there — the softmax, the
//! abstention rules, the mapping back onto a caller's enum — is pure, so it
//! is exercised against fixed numbers with no model present.
//!
//! Design: `docs/superpowers/specs/2026-09-23-system-1-typed-decisions-design.md`.

/// The answer letters, in option order, as they are *displayed* to the model.
/// Four is the cap on a question's options.
///
/// What gets *scored* is [`letter_token_text`], not these — see there.
pub const LETTERS: [&str; 4] = ["A", "B", "C", "D"];

/// The text whose token is scored for option `i`: the letter with a **leading
/// space**.
///
/// This is not cosmetic, and it is the single easiest thing to get wrong here.
/// On the ds4 tokenizer `" A"` and `"A"` are different tokens (334 vs 35), and
/// after a prompt ending in `Answer:` the model puts essentially all of its
/// letter mass on `" A"`. Measured on `DeepSeek` V4 Flash: `" A"` at p=0.402,
/// `"A"` at p≈9e-8. Score the bare letter and every verdict abstains for want
/// of letter mass — the capability reports itself healthy and decides nothing.
///
/// Every scored variant must still be exactly one token on the loaded family,
/// or the whole capability reports unsupported (`Engine::supports_decide`).
#[must_use]
pub fn letter_token_text(i: usize) -> Option<String> {
    LETTERS.get(i).map(|l| format!(" {l}"))
}

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

    /// The same yes/no question with the letters swapped: `no` is A and `yes`
    /// is B. Only the letter-order calibration asks this (see [`OrderPair`]);
    /// the live gate always asks [`boolean`](Self::boolean), so its `p` keeps
    /// reading as "probability of yes".
    #[must_use]
    pub fn boolean_swapped(text: &str) -> Self {
        Self::choice(text, &["no", "yes"])
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
    /// Share of the model's total probability that sat on the answer letters,
    /// as passed to [`score`]; `0.0` when unknown. Carried on the verdict so a
    /// consumer that reads the probabilities themselves (the letter-order
    /// calibration, [`OrderPair::from_verdicts`]) can drop a ranking read off
    /// noise without re-deriving why `abstained` was set.
    pub letter_mass: f32,
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
        letter_mass: 0.0,
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
        letter_mass,
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
    // No trailing space: the scored token carries the leading space itself
    // (`letter_token_text`), and emitting one here too would leave the model
    // choosing between a double space and a token we do not score.
    out.push_str("Answer:");
    out
}

/// One yes/no question asked twice, once as [`Question::boolean`] (yes = A)
/// and once as [`Question::boolean_swapped`] (yes = B), reduced to the
/// probability of "yes" each time.
///
/// `letter_mass` says whether the model answered with a letter; it cannot say
/// why it picked the one it did. A model with a prior toward a letter as such
/// (Zheng et al., "Large Language Models Are Not Robust Multiple Choice
/// Selectors", ICLR 2024) can hold nearly all its mass on the letters and
/// still lean toward A partly because it is A. Swapping the order moves that
/// lean from "yes" to "no", so half the difference between the two asks is
/// the letter bias and their mean is the answer with the bias averaged out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderPair {
    /// P(yes) when yes was A — what the live gate reads.
    pub p_yes_ab: f32,
    /// P(yes) when yes was B.
    pub p_yes_ba: f32,
}

impl OrderPair {
    /// Builds a pair from the two verdicts, or `None` when either one carries
    /// no usable probability for "yes": a degenerate verdict with no runner-up
    /// (the early abstention `score` returns on bad logprobs), or letter mass
    /// too thin for the ranking to mean anything.
    #[must_use]
    pub fn from_verdicts(ab: &RawVerdict, ba: &RawVerdict) -> Option<Self> {
        Some(Self {
            p_yes_ab: p_of(ab, 0)?,
            p_yes_ba: p_of(ba, 1)?,
        })
    }

    /// The letter bias on this question, in probability: positive when the
    /// model leans toward A, whatever A means.
    #[must_use]
    pub fn bias(&self) -> f32 {
        (self.p_yes_ab - self.p_yes_ba) / 2.0
    }

    /// P(yes) with the letter bias averaged out.
    #[must_use]
    pub fn debiased(&self) -> f32 {
        f32::midpoint(self.p_yes_ab, self.p_yes_ba)
    }
}

/// The normalised probability of option `i` on a verdict, when it carries one.
fn p_of(v: &RawVerdict, i: usize) -> Option<f32> {
    if v.letter_mass > 0.0 && v.letter_mass < MIN_LETTER_MASS {
        return None;
    }
    let (ru_i, ru_p) = v.runner_up?;
    if v.index == i {
        Some(v.p)
    } else if ru_i == i {
        Some(ru_p)
    } else {
        None
    }
}

/// Fewer usable pairs than this and [`bias_report`] suggests no shift at all:
/// a mean over a handful of spans says more about those spans than about the
/// model.
pub const MIN_CALIBRATION_PAIRS: usize = 10;

/// What a calibration run measured, and the threshold shift it supports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasReport {
    /// Usable pairs the numbers below are computed over.
    pub pairs: usize,
    /// Mean P(yes) as the live gate asks it (yes = A).
    pub mean_ab: f32,
    /// Mean P(yes) with the letters swapped (yes = B).
    pub mean_ba: f32,
    /// Mean letter bias ([`OrderPair::bias`]); positive leans toward A.
    pub mean_bias: f32,
    /// Standard error of `mean_bias`; `0.0` below two pairs.
    pub stderr: f32,
    /// Pairs whose gate verdict at the threshold changes once the bias is
    /// averaged out: as-asked says one thing, debiased the other.
    pub flips: usize,
    /// Percentage points to add to the gate threshold for this family, so
    /// that comparing the as-asked P(yes) against it approximates comparing
    /// the debiased P(yes) against the unshifted threshold. `0` when the run
    /// is too small ([`MIN_CALIBRATION_PAIRS`]) or the bias is within two
    /// standard errors of zero.
    pub shift_pp: i32,
}

/// Summarises a calibration run against the gate threshold `threshold`
/// (a probability, `0.6` for the default 60 percent).
///
/// The shift is the mean bias itself, not something fitted to the flips:
/// the live gate reads `p_yes_ab`, and `p_yes_ab - bias` is the debiased
/// answer, so asking `p_yes_ab >= t + bias` is asking `debiased >= t`. The
/// flips are reported so the size of the effect is visible in verdicts, the
/// unit that matters, rather than only in probability.
#[must_use]
pub fn bias_report(pairs: &[OrderPair], threshold: f32) -> BiasReport {
    let n = pairs.len();
    if n == 0 {
        return BiasReport {
            pairs: 0,
            mean_ab: 0.0,
            mean_ba: 0.0,
            mean_bias: 0.0,
            stderr: 0.0,
            flips: 0,
            shift_pp: 0,
        };
    }
    // Accumulated in f64: the run is small, but the variance subtracts two
    // close quantities and f32 would lose the digits that decide the shift.
    #[allow(clippy::cast_precision_loss)] // a run is at most a few hundred pairs
    let nf = n as f64;
    let mean =
        |f: &dyn Fn(&OrderPair) -> f32| pairs.iter().map(|p| f64::from(f(p))).sum::<f64>() / nf;
    let mean_ab = mean(&|p| p.p_yes_ab);
    let mean_ba = mean(&|p| p.p_yes_ba);
    let mean_bias = mean(&OrderPair::bias);
    let stderr = if n < 2 {
        0.0
    } else {
        let var = pairs
            .iter()
            .map(|p| (f64::from(p.bias()) - mean_bias).powi(2))
            .sum::<f64>()
            / (nf - 1.0);
        (var / nf).sqrt()
    };
    let flips = pairs
        .iter()
        .filter(|p| (p.p_yes_ab >= threshold) != (p.debiased() >= threshold))
        .count();
    let significant = n >= MIN_CALIBRATION_PAIRS && mean_bias.abs() > 2.0 * stderr;
    // |mean_bias| <= 0.5, so the rounded percentage fits an i32 with room.
    #[allow(clippy::cast_possible_truncation)]
    let shift_pp = if significant {
        (mean_bias * 100.0).round() as i32
    } else {
        0
    };
    #[allow(clippy::cast_possible_truncation)] // probabilities, back to f32
    BiasReport {
        pairs: n,
        mean_ab: mean_ab as f32,
        mean_ba: mean_ba as f32,
        mean_bias: mean_bias as f32,
        stderr: stderr as f32,
        flips,
        shift_pp,
    }
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

    fn pair(ab: f32, ba: f32) -> OrderPair {
        OrderPair {
            p_yes_ab: ab,
            p_yes_ba: ba,
        }
    }

    #[test]
    fn boolean_swapped_puts_yes_on_b() {
        let q = Question::boolean_swapped("worth it?");
        assert_eq!(q.options, vec!["no".to_string(), "yes".to_string()]);
        let text = render_question(&q);
        assert!(text.contains("A. no\nB. yes\n"), "{text}");
    }

    #[test]
    fn order_pair_reads_yes_from_a_then_from_b() {
        // As asked: yes = A wins at 0.8. Swapped: yes = B, which lost at 0.4.
        let ab = score(&[-0.2, -1.6], 0.0, 0.9);
        let ba = score(&[-0.5, -0.9], 0.0, 0.9);
        let p = OrderPair::from_verdicts(&ab, &ba).expect("both usable");
        assert!((p.p_yes_ab - ab.p).abs() < 1e-6);
        assert!((p.p_yes_ba - ba.runner_up.unwrap().1).abs() < 1e-6);
        assert!(p.p_yes_ab > p.p_yes_ba, "{p:?}");
    }

    #[test]
    fn order_pair_bias_is_half_the_difference_and_debiased_the_mean() {
        let p = pair(0.8, 0.6);
        assert!((p.bias() - 0.1).abs() < 1e-6);
        assert!((p.debiased() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn order_pair_drops_thin_mass_and_degenerate_verdicts() {
        let good = score(&[-0.2, -1.6], 0.0, 0.9);
        let thin = score(&[-3.0, -4.0], 0.0, 0.05);
        let degenerate = score(&[], 0.0, 0.0);
        assert!(OrderPair::from_verdicts(&good, &thin).is_none());
        assert!(OrderPair::from_verdicts(&thin, &good).is_none());
        assert!(OrderPair::from_verdicts(&degenerate, &good).is_none());
    }

    #[test]
    fn order_pair_keeps_a_near_tie_the_gate_would_abstain_on() {
        // p just above one half sits under the abstain floor, so the live gate
        // would not act on it; the calibration still needs the number.
        let ab = score(&[-0.68, -0.71], DEFAULT_ABSTAIN_FLOOR, 0.9);
        assert!(ab.abstained);
        assert!(OrderPair::from_verdicts(&ab, &ab).is_some());
    }

    #[test]
    fn bias_report_suggests_the_mean_bias_when_it_is_significant() {
        let pairs: Vec<OrderPair> = (0..20)
            .map(|i| {
                let jitter = if i % 2 == 0 { 0.01 } else { -0.01 };
                pair(0.70 + jitter, 0.56 + jitter)
            })
            .collect();
        let r = bias_report(&pairs, 0.6);
        assert_eq!(r.pairs, 20);
        assert!((r.mean_bias - 0.07).abs() < 1e-4, "{r:?}");
        assert_eq!(r.shift_pp, 7);
        // Every as-asked 0.69/0.71 clears 0.6, every debiased 0.62/0.64 too.
        assert_eq!(r.flips, 0);
    }

    #[test]
    fn bias_report_counts_verdicts_the_bias_flips() {
        // 0.66 as asked clears 0.6; debiased (0.66 + 0.50) / 2 = 0.58 does not.
        let pairs = vec![pair(0.66, 0.50); 12];
        let r = bias_report(&pairs, 0.6);
        assert_eq!(r.flips, 12);
    }

    #[test]
    fn bias_report_suggests_nothing_from_a_small_run() {
        let pairs = vec![pair(0.9, 0.5); MIN_CALIBRATION_PAIRS - 1];
        let r = bias_report(&pairs, 0.6);
        assert!(r.mean_bias > 0.1);
        assert_eq!(r.shift_pp, 0, "too few pairs to act on");
    }

    #[test]
    fn bias_report_suggests_nothing_when_the_bias_is_noise() {
        // Symmetric scatter around zero bias.
        let pairs: Vec<OrderPair> = (0..20)
            .map(|i| {
                if i % 2 == 0 {
                    pair(0.8, 0.6)
                } else {
                    pair(0.6, 0.8)
                }
            })
            .collect();
        let r = bias_report(&pairs, 0.6);
        assert!(r.mean_bias.abs() < 1e-6);
        assert_eq!(r.shift_pp, 0);
    }

    #[test]
    fn bias_report_of_nothing_is_all_zero() {
        let r = bias_report(&[], 0.6);
        assert_eq!(r.pairs, 0);
        assert_eq!(r.shift_pp, 0);
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
            letter_mass: 0.0,
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
            text.ends_with("Answer:"),
            "no trailing space: the scored token carries it, got {text:?}"
        );
    }

    #[test]
    fn a_question_is_capped_at_the_available_letters() {
        let q = Question::choice("pick", &["a", "b", "c", "d", "e"]);
        assert_eq!(q.options.len(), LETTERS.len(), "extra options are dropped");
    }
}
