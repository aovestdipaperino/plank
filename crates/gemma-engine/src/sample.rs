//! Token sampling: greedy, or temperature with min-p and top-p filtering.
//!
//! The RNG is `SplitMix64`, so a seed reproduces a run exactly with no
//! dependency.

/// A seeded sampler over one step's logits.
#[derive(Debug, Clone)]
pub struct Sampler {
    state: u64,
}

impl Sampler {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, from the top 53 bits.
    // A 53-bit integer is exactly representable in an f64.
    #[allow(clippy::cast_precision_loss)]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Picks the next token.
    ///
    /// `greedy`, or a `temperature` of zero or less, takes the argmax.
    /// Otherwise the softmax of `logits / temperature` is filtered: tokens
    /// below `min_p` times the top probability go, then only the smallest
    /// most-likely prefix whose mass reaches `top_p` stays; one is drawn
    /// from what is left. Empty `logits` give token 0.
    pub fn sample(
        &mut self,
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        min_p: f32,
        greedy: bool,
    ) -> u32 {
        if greedy || temperature <= 0.0 || logits.len() < 2 {
            return argmax(logits);
        }
        let t = f64::from(temperature);
        let max = logits
            .iter()
            .copied()
            .filter(|l| !l.is_nan())
            .fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return argmax(logits);
        }
        let mut probs: Vec<(usize, f64)> = logits
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.is_nan())
            .map(|(i, &l)| (i, ((f64::from(l) - f64::from(max)) / t).exp()))
            .collect();
        let sum: f64 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in &mut probs {
            *p /= sum;
        }
        // The top probability is exp(0) / sum.
        let floor = f64::from(min_p) / sum;
        probs.retain(|&(_, p)| p >= floor);
        probs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut mass = 0.0;
        let mut keep = 0;
        for &(_, p) in &probs {
            mass += p;
            keep += 1;
            if mass >= f64::from(top_p) {
                break;
            }
        }
        probs.truncate(keep.max(1));
        let total: f64 = probs.iter().map(|(_, p)| p).sum();
        let mut r = self.next_f64() * total;
        for &(i, p) in &probs {
            if r < p {
                return id(i);
            }
            r -= p;
        }
        probs.last().map_or(0, |&(i, _)| id(i))
    }
}

fn id(i: usize) -> u32 {
    u32::try_from(i).unwrap_or(u32::MAX)
}

/// The first index of the largest logit, ignoring NaN.
fn argmax(logits: &[f32]) -> u32 {
    let mut best: Option<(usize, f32)> = None;
    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, b)| l > b) {
            best = Some((i, l));
        }
    }
    best.map_or(0, |(i, _)| id(i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax_and_seeded_sampling_is_deterministic() {
        let logits = [0.1, 3.0, 0.2, 2.9];
        assert_eq!(Sampler::new(1).sample(&logits, 0.8, 0.95, 0.0, true), 1);
        let a: Vec<u32> = {
            let mut s = Sampler::new(9);
            (0..20)
                .map(|_| s.sample(&logits, 1.0, 1.0, 0.0, false))
                .collect()
        };
        let b: Vec<u32> = {
            let mut s = Sampler::new(9);
            (0..20)
                .map(|_| s.sample(&logits, 1.0, 1.0, 0.0, false))
                .collect()
        };
        assert_eq!(a, b);
        // Tokens 0 and 2 still hold ~6.4% of the mass at top_p 1.0, so a
        // correct sampler draws one now and then: "dominate", not "only".
        let top2 = a.iter().filter(|&&t| t == 1 || t == 3).count();
        assert!(top2 >= 16, "top-2 dominate: {a:?}");
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let logits = [0.1, 3.0, 0.2, 2.9];
        assert_eq!(Sampler::new(3).sample(&logits, 0.0, 1.0, 0.0, false), 1);
    }

    #[test]
    fn min_p_and_top_p_drop_the_tail() {
        // p ~ [0.47, 0.47, 0.06]: min_p 0.5 drops the third; top_p 0.4 keeps
        // only the first.
        let logits = [2.0, 2.0, 0.0];
        let mut s = Sampler::new(5);
        for _ in 0..50 {
            assert_ne!(s.sample(&logits, 1.0, 1.0, 0.5, false), 2);
        }
        let mut s = Sampler::new(5);
        for _ in 0..50 {
            assert_eq!(s.sample(&logits, 1.0, 0.4, 0.0, false), 0);
        }
    }

    #[test]
    fn both_tokens_get_drawn_when_equally_likely() {
        let logits = [1.0, 1.0];
        let mut s = Sampler::new(11);
        let draws: Vec<u32> = (0..64)
            .map(|_| s.sample(&logits, 1.0, 1.0, 0.0, false))
            .collect();
        assert!(draws.contains(&0) && draws.contains(&1), "{draws:?}");
    }
}
