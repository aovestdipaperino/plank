// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! A native-Rust model behind [`Engine`], over gemma-engine's shared session.
//!
//! The transcript discipline is the ds4 engine's: the rendered transcript is
//! split into sections, the leading sections that match recorded spans keep
//! their tokens verbatim, and only the rest is rendered (with the family's
//! chat template) and tokenized. Unlike the ds4 KV, the native session's is
//! prefix-truncatable, so a prompt that diverges anywhere keeps the KV up to
//! that token and prefills only the remainder.
//!
//! What differs between model families (opening the file, the template, the
//! stop ids, the stream translation) sits behind [`NativeFamily`]; the
//! reconcile, the KV and warm walk and the [`Engine`] impl are shared here.

use std::path::Path;
use std::sync::Arc;

use gemma_engine::Device;
use gemma_engine::sample::Sampler;
use gemma_engine::session::Session;
use gemma_engine::template::{self, Kind};

use crate::ds4tokens::{SectionKey, SpanRole, TokenTranscript, parse_sections};
use crate::engine::{
    Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats, KvReuse, PrefillProgress,
    Prompt, ThinkMode, Utf8Stream,
};
use crate::kvcache::KVCache;

/// Tokens a prefill evaluates between progress reports (the session's own
/// chunk size, so each report is one forward): the default of
/// [`NativeFamily::prefill_chunk`].
const PREFILL_CHUNK: usize = 512;

/// The context used when the caller asks for none and the machine's memory
/// is unknown; otherwise [`default_ctx`] sizes it from the RAM.
const DEFAULT_CTX: usize = 32_768;

/// What a model family supplies to [`NativeEngine`].
pub trait NativeFamily: Sized + Send + std::fmt::Debug + 'static {
    /// The model type the shared session runs.
    type Model: gemma_engine::session::SessionModel + Send + Sync + std::fmt::Debug + 'static;
    /// Per-generation stream state turning sampled ids into text.
    type Translator: Default;
    /// Short tag for `PLANK_KV_DEBUG` lines ("gemma", "flint").
    const LABEL: &'static str;

    /// Opens the model at `path` on `device`, with rotary tables for `cap`
    /// positions.
    ///
    /// # Errors
    /// When the file does not load as this family's model.
    fn open(
        path: &Path,
        device: &Device,
        cap: usize,
    ) -> Result<(Arc<Self::Model>, Self), EngineError>;

    /// `(context_length, kv_bytes_per_token)` from the file header alone.
    ///
    /// # Errors
    /// When the header does not read as this family's.
    fn probe(path: &Path) -> Result<(usize, usize), EngineError>;

    /// Bytes of KV one token takes.
    fn kv_bytes_per_token(model: &Self::Model) -> usize;

    /// The model's display name.
    fn model_name(&self) -> String;

    /// BOS for the first span, when the family wants one.
    fn bos(&self) -> Option<u32>;

    /// One section as token ids (classify + render + encode).
    fn render(
        &self,
        kind: Kind,
        text: &str,
        prev: Option<Kind>,
        think: bool,
        trusted_len: usize,
    ) -> Vec<u32>;

    /// The ids the reply starts with after `last`.
    fn generation_prefix(&self, last: Option<Kind>, think: bool) -> Vec<u32>;

    /// Whether sampling `id` ends the reply. A stop token is neither recorded
    /// nor evaluated: the next section's rendering supplies it.
    fn is_stop(&self, id: u32) -> bool;

    /// Tokens `text` encodes to as plain text.
    fn count_tokens(&self, text: &str) -> usize;

    /// Text `id` contributes to the stream.
    fn translate(&self, tr: &mut Self::Translator, id: u32, utf8: &mut Utf8Stream) -> String;

    /// Applied to each step's logits before sampling (default: nothing).
    fn filter_logits(&self, _logits: &mut [f32], _greedy: bool) {}

    /// Tokens per prefill forward when the session holds `keys` positions.
    fn prefill_chunk(&self, _keys: usize) -> usize {
        PREFILL_CHUNK
    }
}

/// The context opened when the caller asks for none (`ctx <= 0`): the model's
/// own `context_length` when its KV passes the half-of-memory refusal in
/// [`NativeEngine::open`]; otherwise the KV a third of `ram` can hold
/// ([`crate::enginefit::NATIVE_KV_RAM_DIVISOR`]), in whole 1024-token steps,
/// never below [`crate::enginefit::NATIVE_MIN_CTX`].
///
/// The cache is f32 and keeps every position, so the window is what decides
/// the memory: 688 KB a token on Gemma 4 12B, 115 KB on E4B, 100 KB on Flint
/// (whose cache is f16).
/// On a 128 GB machine E4B's 131K and Flint's 262K open whole, while the
/// 12B's 131K (90 GB) would be refused, so it gets the third instead, which
/// leaves the weights and the rest of the system room. `ram` of 0 or
/// `usize::MAX` (not measurable) keeps the old 32K.
pub(crate) fn default_ctx(ram: usize, context_length: usize, kv_bytes_per_token: usize) -> usize {
    let fallback = DEFAULT_CTX.min(context_length);
    if ram == 0 || ram == usize::MAX || kv_bytes_per_token == 0 {
        return fallback;
    }
    if context_length.saturating_mul(kv_bytes_per_token) <= ram / 2 {
        return context_length;
    }
    let fits = ram / crate::enginefit::NATIVE_KV_RAM_DIVISOR / kv_bytes_per_token;
    (fits / 1024 * 1024)
        .max(crate::enginefit::NATIVE_MIN_CTX)
        .min(context_length)
}

/// The engine: one model, one live session, and the token transcript that
/// describes what the session's KV was built from.
#[derive(Debug)]
pub struct NativeEngine<F: NativeFamily> {
    /// The family's tokenizer, template and stream rules.
    pub(crate) family: F,
    pub(crate) model: Arc<F::Model>,
    pub(crate) session: Session<F::Model>,
    pub(crate) transcript: TokenTranscript,
    /// The template kind of each span, parallel to `transcript.spans()`.
    pub(crate) kinds: Vec<Kind>,
    pub(crate) think: ThinkMode,
    pub(crate) trusted_len: usize,
    /// `"metal"` or `"cpu"`: where the model runs.
    pub(crate) device_name: &'static str,
    /// Set by `warm_reset` until the walk's first append or sync: the window
    /// in which `kvtier` restores a tier checkpoint. A restore there keeps the
    /// warm buffer, because the walk goes on to append every tier again —
    /// restored ones included — and taking the checkpoint's own transcript
    /// would hold those tiers twice.
    pub(crate) warm_pending: bool,
}

/// The span role a template kind is recorded under.
fn role_of(kind: Kind) -> SpanRole {
    match kind {
        Kind::System => SpanRole::System,
        Kind::User | Kind::ToolResult => SpanRole::User,
        Kind::Assistant => SpanRole::Assistant,
    }
}

/// The section tag `template::classify` reads for a span role.
fn tag_of(role: SpanRole) -> &'static str {
    match role {
        SpanRole::System => "system",
        SpanRole::User => "user",
        SpanRole::Assistant => "assistant",
    }
}

fn to_i32(ids: &[u32]) -> Vec<i32> {
    ids.iter()
        .map(|&t| i32::try_from(t).unwrap_or(i32::MAX))
        .collect()
}

pub(crate) fn to_u32(ids: &[i32]) -> Vec<u32> {
    ids.iter().map(|&t| u32::try_from(t).unwrap_or(0)).collect()
}

fn count(n: usize) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

pub(crate) fn engine_error(e: gemma_engine::Error) -> EngineError {
    EngineError::new(e.0)
}

/// How the template renders: what [`Render::reconcile`] needs besides the
/// transcript it edits.
#[derive(Debug)]
pub(crate) struct Render<'a, F> {
    family: &'a F,
    think: bool,
    trusted_len: usize,
}

impl<'a, F: NativeFamily> Render<'a, F> {
    fn new(family: &'a F, think: ThinkMode, trusted_len: usize) -> Self {
        Self {
            family,
            think: !matches!(think, ThinkMode::Off),
            trusted_len,
        }
    }

    /// Renders one section after `kinds`' last span and appends it.
    fn push(
        &self,
        transcript: &mut TokenTranscript,
        kinds: &mut Vec<Kind>,
        role: &str,
        text: &str,
    ) {
        let prev = kinds.last().copied();
        let kind = template::classify(role, text, prev);
        let mut ids = Vec::new();
        if transcript.spans().is_empty()
            && let Some(bos) = self.family.bos()
        {
            ids.push(bos);
        }
        ids.extend(
            self.family
                .render(kind, text, prev, self.think, self.trusted_len),
        );
        transcript.push_span(role_of(kind), 0, text.to_owned(), &to_i32(&ids));
        kinds.push(kind);
    }

    /// Brings `transcript` in line with the rendered transcript `flat`: the
    /// leading spans whose (role, text) match keep their tokens, the rest are
    /// rendered from text.
    fn reconcile(&self, transcript: &mut TokenTranscript, kinds: &mut Vec<Kind>, flat: &str) {
        let sections = parse_sections(flat);
        let keys: Vec<SectionKey> = sections
            .iter()
            .filter_map(|(role, text)| {
                SpanRole::from_tag(role).map(|role| SectionKey {
                    role,
                    text: text.clone(),
                })
            })
            .collect();
        let keep = transcript.common_prefix(&keys);
        crate::engine::kv_debug(|| {
            format!(
                "{} reconcile: {} spans held, {} sections in, kept {keep}",
                F::LABEL,
                transcript.spans().len(),
                keys.len()
            )
        });
        transcript.truncate_spans(keep);
        kinds.truncate(keep);
        for (role, text) in sections.iter().skip(keep) {
            self.push(transcript, kinds, role, text);
        }
    }

    /// The tokens a generation after `kinds`' last span starts with.
    pub(crate) fn generation_prefix(&self, kinds: &[Kind]) -> Vec<u32> {
        self.family
            .generation_prefix(kinds.last().copied(), self.think)
    }
}

/// Bytes of RAM on this machine, or `usize::MAX` when it cannot be read (so
/// the KV check never refuses on a failed probe).
#[cfg(target_os = "macos")]
pub(crate) fn physical_memory() -> usize {
    let mut bytes: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: the name is NUL-terminated, `bytes` and `len` are valid out
    // pointers sized for `hw.memsize`'s u64, and no new value is set.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut bytes).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        usize::try_from(bytes).unwrap_or(usize::MAX)
    } else {
        usize::MAX
    }
}

/// Bytes of RAM on this machine, or `usize::MAX` when it cannot be read.
#[cfg(not(target_os = "macos"))]
pub(crate) fn physical_memory() -> usize {
    crate::platform::physical_memory()
        .and_then(|b| usize::try_from(b).ok())
        .unwrap_or(usize::MAX)
}

/// Metal when this machine has it, else the CPU.
fn pick_device() -> (Device, &'static str) {
    #[cfg(target_os = "macos")]
    if let Ok(d) = Device::new_metal(0) {
        return (d, "metal");
    }
    (Device::Cpu, "cpu")
}

impl<F: NativeFamily> NativeEngine<F> {
    /// Opens the family's GGUF at `path` on Metal (the CPU when there is no
    /// Metal device) with a `ctx`-token context, clamped to the model's; a
    /// `ctx` of zero or less means [`default_ctx`]: sized from the RAM.
    ///
    /// # Errors
    /// When the file does not load as this family's model, or when the KV for
    /// `ctx` tokens would take more than half the machine's memory.
    pub fn open(path: &Path, ctx: i32) -> Result<Self, EngineError> {
        let (device, name) = pick_device();
        Self::open_on(path, ctx, &device, name)
    }

    /// [`NativeEngine::open`] forced onto the CPU, for tests.
    ///
    /// # Errors
    /// As [`NativeEngine::open`].
    #[cfg(test)]
    pub fn open_on_cpu(path: &Path, ctx: i32) -> Result<Self, EngineError> {
        Self::open_on(path, ctx, &Device::Cpu, "cpu")
    }

    fn open_on(
        path: &Path,
        ctx: i32,
        device: &Device,
        device_name: &'static str,
    ) -> Result<Self, EngineError> {
        let cap = if let Some(c) = usize::try_from(ctx).ok().filter(|&c| c > 0) {
            c
        } else {
            let (context_length, kv_bytes_per_token) = F::probe(path)?;
            default_ctx(physical_memory(), context_length, kv_bytes_per_token)
        };
        // Rotary tables are built for the capped context only, which also
        // makes `context_length()` the clamped context.
        let (model, family) = F::open(path, device, cap)?;
        let ctx = gemma_engine::session::SessionModel::context_length(&*model);
        let need = F::kv_bytes_per_token(&model).saturating_mul(ctx);
        if need > physical_memory() / 2 {
            return Err(EngineError::new(format!(
                "a {ctx}-token context needs {} GB of KV; lower it with --ctx",
                need.div_ceil(1 << 30)
            )));
        }
        let session = Session::new(model.clone(), ctx);
        Ok(Self {
            family,
            model,
            session,
            transcript: TokenTranscript::new(),
            kinds: Vec::new(),
            think: ThinkMode::default(),
            trusted_len: 0,
            device_name,
            warm_pending: false,
        })
    }

    /// Bytes the KV cache takes at the full context this engine opened with:
    /// what the window actually costs in memory, for the startup line.
    #[must_use]
    pub fn kv_bytes_at_ctx(&self) -> usize {
        F::kv_bytes_per_token(&self.model).saturating_mul(self.session.ctx())
    }

    /// Where the model runs: `"metal"` or `"cpu"`.
    #[must_use]
    pub fn device_name(&self) -> &'static str {
        self.device_name
    }

    pub(crate) fn render(&self) -> Render<'_, F> {
        Render::new(&self.family, self.think, self.trusted_len)
    }

    /// Reconciles the token transcript with the rendered transcript `flat`.
    pub(crate) fn reconcile(&mut self, flat: &str) {
        Render::new(&self.family, self.think, self.trusted_len).reconcile(
            &mut self.transcript,
            &mut self.kinds,
            flat,
        );
    }

    /// Drops every recorded token, for a change that alters the system turn.
    fn invalidate(&mut self) {
        self.session.truncate(0);
        self.transcript = TokenTranscript::new();
        self.kinds.clear();
        self.warm_pending = false;
    }

    /// Prefills `toks` after the session's first `base` tokens (the session
    /// already holds exactly those), reporting progress per chunk. Returns the
    /// last logits, or `None` when interrupted.
    fn prefill(
        &mut self,
        base: usize,
        toks: &[u32],
        interrupt: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let total_abs = base + toks.len();
        on_event(EngineEvent::Prefill(PrefillProgress::primed(
            count(base),
            count(total_abs),
        )));
        let start = std::time::Instant::now();
        let mut total = count(total_abs);
        let mut last = None;
        let mut done = 0;
        while done < toks.len() {
            let n = self.family.prefill_chunk(base + done).max(1);
            let chunk = &toks[done..toks.len().min(done + n)];
            done += chunk.len();
            match self
                .session
                .prefill(chunk, interrupt)
                .map_err(engine_error)?
            {
                Some(l) => last = Some(l),
                None => return Ok(None),
            }
            on_event(EngineEvent::Prefill(PrefillProgress::from_absolute(
                count(base),
                count(self.session.tokens().len()),
                &mut total,
                start.elapsed().as_secs_f64(),
            )));
        }
        Ok(last)
    }

    /// Fails before touching the session when `need` tokens cannot fit.
    fn check_fits(&self, need: usize) -> Result<(), EngineError> {
        if need > self.session.ctx() {
            return Err(EngineError::new(format!(
                "context full: {need} tokens > {}",
                self.session.ctx()
            )));
        }
        Ok(())
    }

    /// Records a sampled reply: the generation prefix it followed and its ids,
    /// as a new assistant span — or as more of the last one when that is
    /// already an assistant reply (a resumed `/btw` pass).
    fn record_reply(&mut self, text: String, prefix: &[u32], reply: &[u32]) {
        if reply.is_empty() {
            return;
        }
        let mut ids = to_i32(prefix);
        ids.extend(to_i32(reply));
        if self.transcript.last_is_assistant() {
            self.transcript.extend_last_span(&text, &ids);
        } else {
            self.transcript
                .push_span(SpanRole::Assistant, 0, text, &ids);
            self.kinds.push(Kind::Assistant);
        }
    }
}

impl<F: NativeFamily> Engine for NativeEngine<F> {
    fn generate(
        &mut self,
        prompt: Prompt<'_>,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        self.warm_pending = false;
        self.reconcile(prompt.flat());
        let prefix = self.render().generation_prefix(&self.kinds);
        let mut full = to_u32(self.transcript.tokens());
        full.extend_from_slice(&prefix);
        self.check_fits(full.len())?;

        // The last prompt token is always evaluated: its logits pick the
        // first reply token.
        let mut common = common_prefix(self.session.tokens(), &full);
        if common == full.len() {
            common = common.saturating_sub(1);
        }
        let live = self.session.tokens().len();
        // Below a restored snapshot's sliding window the session empties.
        let common = self.session.truncate(common);
        crate::engine::kv_debug(|| {
            format!(
                "{} generate: prompt={} live={live} reused={common}",
                F::LABEL,
                full.len()
            )
        });
        let Some(mut logits) = self.prefill(common, &full[common..], interrupt, on_event)? else {
            return Ok(GenerationStats {
                interrupted: true,
                ctx_used: count(self.session.tokens().len()),
                ..GenerationStats::default()
            });
        };

        let mut sampler = Sampler::new(if opts.seed != 0 {
            opts.seed
        } else {
            0x2545_f491_4f6c_dd1d
        });
        let mut tr = F::Translator::default();
        let mut utf8 = Utf8Stream::default();
        let mut reply: Vec<u32> = Vec::new();
        let mut reply_text = String::new();
        let mut interrupted = false;
        let mut generated: i32 = 0;
        let start = std::time::Instant::now();
        let mut steady_mark: Option<(std::time::Instant, i32)> = None;
        let (max_tokens, bound) = crate::engine::pass_budget(
            opts.n_predict,
            count(self.session.ctx()) - count(self.session.tokens().len()),
            opts.ctx_reserve,
        );
        while generated < max_tokens {
            if steady_mark.is_none()
                && start.elapsed().as_secs_f64() >= crate::engine::STEADY_WARMUP_SECS
            {
                steady_mark = Some((std::time::Instant::now(), generated));
            }
            if interrupt() {
                interrupted = true;
                break;
            }
            let is_greedy = greedy();
            self.family.filter_logits(&mut logits, is_greedy);
            let t = sampler.sample(&logits, opts.temperature, opts.top_p, opts.min_p, is_greedy);
            if self.family.is_stop(t) {
                break;
            }
            let text = self.family.translate(&mut tr, t, &mut utf8);
            if !text.is_empty() {
                reply_text.push_str(&text);
                on_event(EngineEvent::Text(text));
            }
            reply.push(t);
            generated += 1;
            logits = self.session.step(t).map_err(engine_error)?;
        }
        let tail = utf8.flush();
        if !tail.is_empty() {
            reply_text.push_str(&tail);
            on_event(EngineEvent::Text(tail));
        }
        self.record_reply(reply_text, &prefix, &reply);

        let secs = start.elapsed().as_secs_f64();
        Ok(GenerationStats {
            generated,
            tps: if secs > 0.0 {
                f64::from(generated) / secs
            } else {
                0.0
            },
            steady_tps: crate::engine::rate_since(steady_mark, generated),
            ctx_used: count(self.session.tokens().len()),
            interrupted,
            limit: crate::engine::pass_limit(generated, max_tokens, bound, interrupted),
            usage: None,
            spec: crate::engine::SpecStats::default(),
            prefill_tokens: 0,
            prefill_secs: 0.0,
        })
    }

    fn emits_think_tags(&self) -> bool {
        true
    }

    /// The candle KV truncates to the common prefix in `generate`, so fork
    /// snapshots and ladder rungs would only serialise the whole f32 cache
    /// (about 114 KB per token on Gemma E4B) for nothing.
    fn kv_truncates_exactly(&self) -> bool {
        true
    }

    /// Truncates the session to its common prefix with `transcript`, the
    /// truncation the next `generate` would make, so a snapshot taken before
    /// that turn does not record a sidechain's tail. A restored session keeps
    /// only `window + SNAPSHOT_SLACK` positions per sliding layer, so a tail
    /// longer than the slack would otherwise turn the resumed turn into a
    /// rebuild from zero. Left alone during a warm walk, which owns the
    /// transcript until `warm_sync`.
    fn sync_to_prefix(&mut self, transcript: &str) {
        if self.warm_pending {
            return;
        }
        self.reconcile(transcript);
        let want = to_u32(self.transcript.tokens());
        let common = common_prefix(self.session.tokens(), &want);
        let kept = self.session.truncate(common);
        crate::engine::kv_debug(|| {
            format!(
                "{} sync_to_prefix: transcript={} common={common} kept={kept}",
                F::LABEL,
                want.len()
            )
        });
    }

    /// Reports the reusable prefix as the live end: `live == common`.
    ///
    /// The native KV truncates exactly, so a prompt that diverges behind the
    /// live end keeps every token up to the divergence and prefills only the
    /// rest. That is never the rebuild-from-zero shape
    /// `KvReuse::rebuilds_from_zero` describes (the ds4 sync's), which the
    /// agent answers by restoring a ladder rung or fork snapshot and by
    /// skipping prompt suggestions. Reporting the real session length would
    /// buy both for nothing.
    ///
    /// The exception is a session restored from a snapshot, whose sliding
    /// layers hold only their window plus `SNAPSHOT_SLACK`: diverging so far
    /// behind its end that [`Session::reusable`] is 0 empties it, so the probe
    /// reports the real length, the rebuild shape. So does a prompt sharing no
    /// token at all, which keeps nothing on any session.
    fn kv_reuse_probe(&mut self, transcript: &str, _think: ThinkMode) -> Option<KvReuse> {
        let render = self.render();
        let mut tokens = self.transcript.clone();
        let mut kinds = self.kinds.clone();
        render.reconcile(&mut tokens, &mut kinds, transcript);
        let mut full = to_u32(tokens.tokens());
        full.extend(render.generation_prefix(&kinds));
        let common = common_prefix(self.session.tokens(), &full);
        let live = if common > 0 && self.session.reusable(common) == common {
            common
        } else {
            self.session.tokens().len()
        };
        Some(KvReuse {
            live: count(live),
            common: count(common),
        })
    }

    fn set_think_mode(&mut self, mode: ThinkMode) {
        if mode != self.think {
            self.think = mode;
            self.invalidate();
        }
    }

    fn set_trusted_system_prefix(&mut self, len: usize) {
        if len != self.trusted_len {
            self.trusted_len = len;
            self.invalidate();
        }
    }

    fn count_tokens(&self, text: &str) -> i32 {
        count(self.family.count_tokens(text))
    }

    fn get_kv(&mut self) -> Option<KVCache> {
        Some(KVCache::new(
            self.session.snapshot().ok()?,
            self.transcript.clone(),
        ))
    }

    /// Restores the session and, outside a warm walk, the transcript the
    /// checkpoint was captured with. Inside one (right after `warm_reset`) the
    /// warm buffer stays: the walk re-appends every tier, and `warm_sync`
    /// matches it against the restored session's own tokens.
    fn set_kv(&mut self, cache: &KVCache) -> Result<(), EngineError> {
        self.session.restore(cache.kv()).map_err(engine_error)?;
        if self.warm_pending {
            return Ok(());
        }
        self.transcript = cache.transcript().clone();
        self.kinds.clear();
        for span in self.transcript.spans() {
            let kind =
                template::classify(tag_of(span.role), &span.text, self.kinds.last().copied());
            self.kinds.push(kind);
        }
        Ok(())
    }

    fn can_release_gpu(&self) -> bool {
        true
    }

    /// Places the system turn in the transcript. Nothing is prefilled until
    /// [`Engine::warm_sync`], so a checkpoint restored in between is not paid
    /// for twice.
    fn warm_reset(&mut self, system: &str) -> Result<(), EngineError> {
        self.transcript = TokenTranscript::new();
        self.kinds.clear();
        self.reconcile(&format!("[system]\n{system}\n"));
        self.warm_pending = true;
        Ok(())
    }

    fn warm_append(&mut self, text: Option<&str>) -> Result<(), EngineError> {
        self.warm_pending = false;
        if let Some(text) = text {
            // Trimmed as `parse_sections` trims, so the span matches the
            // section the next turn's transcript carries.
            Render::new(&self.family, self.think, self.trusted_len).push(
                &mut self.transcript,
                &mut self.kinds,
                "user",
                text.trim_end(),
            );
        }
        Ok(())
    }

    fn warm_sync(&mut self, on_event: &mut dyn FnMut(EngineEvent)) -> Result<bool, EngineError> {
        self.warm_pending = false;
        let want = to_u32(self.transcript.tokens());
        self.check_fits(want.len())?;
        let common = common_prefix(self.session.tokens(), &want);
        let common = self.session.truncate(common);
        if common == want.len() {
            return Ok(false);
        }
        self.prefill(common, &want[common..], &|| false, on_event)?;
        Ok(true)
    }

    fn ctx_size(&self) -> i32 {
        count(self.session.ctx())
    }

    fn model_name(&self) -> String {
        self.family.model_name()
    }

    fn is_local(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1 << 30;

    /// With RAM to spare the default is the file's own window; with less,
    /// the KV a third of RAM holds.
    #[test]
    fn the_default_context_reaches_the_files_window() {
        assert_eq!(default_ctx(128 * GIB, 262_144, 100_352), 262_144);
        assert_eq!(default_ctx(16 * GIB, 262_144, 100_352), 56_320);
    }
}
