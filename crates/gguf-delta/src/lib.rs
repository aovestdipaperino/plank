//! Compact weight deltas between same-layout GGUF model files.
//!
//! A `.ggd` file records the byte spans where a target GGUF differs from a
//! base GGUF of identical layout (same metadata, same tensor table), each as a
//! deflate-compressed byte-wise `(target - base) mod 256` difference. Quantized
//! weights that were nudged by a small edit compress far better this way than
//! the raw target bytes do.
//!
//! - [`write_delta`] creates a `.ggd` from a base and a target.
//! - [`read_header`] and [`read_chunks`] parse one.
//! - [`apply`] turns a byte copy of the base into the target, in place.
//! - [`find_base`] follows the link a delta carries to its base.
//! - The `ggd` binary (`ggd create`, `ggd info`) wraps the above.
//! - [`gguf::layout`] is the small GGUF header reader everything is built on.
//!
//! The format is documented in the crate README. Neither input file is ever
//! opened for writing; every chunk carries a hash of the base bytes it
//! replaces, so a delta applied onto the wrong base fails at the first chunk.

pub mod gguf;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use sha2::{Digest, Sha256};

pub use gguf::{Layout, TensorSpan};

/// File extension of a weight delta, without the dot.
pub const EXTENSION: &str = "ggd";

const MAGIC: &[u8; 4] = b"GGDL";
const VERSION: u32 = 1;
/// Flag bit: chunk payloads are deflate-compressed.
pub const FLAG_DEFLATE: u32 = 1;
/// Flag bit: `base_sha256` was computed at create time.
pub const FLAG_BASE_SHA: u32 = 2;
/// Comparison granularity within a tensor. A tensor edited end to end costs
/// one chunk of the whole tensor; a tensor with one changed row costs at most
/// this much.
pub const PIECE: u64 = 1 << 20;
/// Refuses header strings beyond this: the file is not a `.ggd`.
const MAX_STRING: u32 = 1 << 16;
/// Refuses chunk counts beyond this.
const MAX_CHUNKS: u64 = 1 << 24;

/// Byte offsets of the fixed header fields, for patching after the chunks are
/// written.
const OFF_FLAGS: u64 = 8;
const OFF_BASE_SHA: u64 = 4 + 4 + 4 + 8 + 8 + 32;
const OFF_TARGET_SHA: u64 = OFF_BASE_SHA + 32;

/// Everything that can go wrong here.
#[derive(Debug)]
pub enum Error {
    /// A read or write failed.
    Io(io::Error),
    /// The file is not a `.ggd` or GGUF this crate understands.
    Format(String),
    /// The inputs do not go together: different layouts, a base that does
    /// not hold the bytes a chunk expects.
    Mismatch(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Format(s) | Self::Mismatch(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Format(_) | Self::Mismatch(_) => None,
        }
    }
}

fn io_ctx<'a>(what: &'a str, path: &'a Path) -> impl FnOnce(io::Error) -> Error + 'a {
    move |e| {
        Error::Io(io::Error::new(
            e.kind(),
            format!("{what} {}: {e}", path.display()),
        ))
    }
}

/// Whether `path` names a weight delta rather than a model.
#[must_use]
pub fn is_delta_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(EXTENSION))
}

/// The fixed and annotating fields at the front of a `.ggd`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Header {
    /// `FLAG_*` bits.
    pub flags: u32,
    /// Size of the base (and target) file in bytes.
    pub base_size: u64,
    /// First byte of the tensor data region in the base.
    pub data_pos: u64,
    /// `sha256(base[0..data_pos])`: the metadata and tensor table.
    pub header_sha256: [u8; 32],
    /// `sha256` of the whole base, or zero when not computed.
    pub base_sha256: [u8; 32],
    /// `sha256` of the whole target.
    pub target_sha256: [u8; 32],
    /// Short human name of the edit, e.g. `abliterated`.
    pub label: String,
    /// The base as given at creation: absolute, or relative to the `.ggd`.
    pub base_path: PathBuf,
    /// The base's filename, for lookup when the path has moved.
    pub base_name: String,
    /// `general.name` from the base's metadata.
    pub base_general: String,
    /// `general.source.url` from the base's metadata, empty if absent.
    pub base_source_url: String,
    /// `general.source.revision` from the base's metadata, empty if absent.
    pub base_source_rev: String,
    /// The target's filename.
    pub target_name: String,
    /// Number of chunk records that follow the header.
    pub n_chunks: u64,
}

impl Header {
    /// Whether `base_sha256` carries a real hash.
    #[must_use]
    pub fn has_base_sha(&self) -> bool {
        self.flags & FLAG_BASE_SHA != 0
    }

    /// Whether payloads are deflate-compressed.
    #[must_use]
    pub fn is_deflated(&self) -> bool {
        self.flags & FLAG_DEFLATE != 0
    }
}

/// One changed span: where it goes and where its payload sits in the `.ggd`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInfo {
    /// Absolute byte offset in the model file.
    pub offset: u64,
    /// Uncompressed length.
    pub len: u64,
    /// First 8 bytes of `sha256(base[offset..offset+len])`.
    pub base_check: [u8; 8],
    /// Byte offset of the payload inside the `.ggd`.
    pub payload_pos: u64,
    /// Stored payload length.
    pub payload_len: u64,
}

/// Knobs for [`write_delta`].
#[derive(Debug, Clone, Default)]
pub struct CreateOptions {
    /// Label to record; [`default_label`] when `None`.
    pub label: Option<String>,
    /// Also hash the whole base (minutes for an 87 GB file).
    pub hash_base: bool,
}

/// What [`write_delta`] found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CreateReport {
    /// Tensors with at least one differing byte.
    pub tensors_changed: usize,
    /// Chunk records written.
    pub chunks: usize,
    /// Bytes covered by chunks.
    pub bytes_spanned: u64,
    /// Bytes that actually differ.
    pub bytes_changed: u64,
    /// Compressed payload bytes written.
    pub payload_bytes: u64,
    /// The label recorded.
    pub label: String,
}

/// `sha256(file[0..data_pos])`.
///
/// # Errors
/// Read failures.
pub fn header_sha256(path: &Path, data_pos: u64) -> Result<[u8; 32], Error> {
    let f = File::open(path).map_err(io_ctx("cannot open", path))?;
    let mut h = Sha256::new();
    hash_range(&f, &mut h, 0, data_pos).map_err(io_ctx("cannot read", path))?;
    Ok(h.finalize().into())
}

/// Hex `sha256` over a whole file.
///
/// # Errors
/// Read failures.
pub fn file_sha256_hex(path: &Path) -> Result<String, Error> {
    let f = File::open(path).map_err(io_ctx("cannot open", path))?;
    let len = f.metadata()?.len();
    let mut h = Sha256::new();
    hash_range(&f, &mut h, 0, len).map_err(io_ctx("cannot read", path))?;
    Ok(hex(&h.finalize()))
}

/// Lowercase hex of `bytes`.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn piece_len() -> usize {
    usize::try_from(PIECE).unwrap_or(1 << 20)
}

fn hash_range(f: &File, h: &mut Sha256, start: u64, end: u64) -> io::Result<()> {
    let mut buf = vec![0u8; piece_len()];
    let mut pos = start;
    while pos < end {
        let n = usize::try_from((end - pos).min(PIECE)).unwrap_or(buf.len());
        f.read_exact_at(&mut buf[..n], pos)?;
        h.update(&buf[..n]);
        pos += n as u64;
    }
    Ok(())
}

/// The label a delta gets when none is given: the parts of the target's stem
/// that the base's stem does not have, lowercased and joined with `-`;
/// `delta` when nothing remains.
#[must_use]
pub fn default_label(base: &Path, target: &Path) -> String {
    let stem = |p: &Path| {
        p.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let b = stem(base);
    let t = stem(target);
    let bp: Vec<&str> = b.split(['-', '_', '.']).collect();
    let extra: Vec<&str> = t
        .split(['-', '_', '.'])
        .filter(|p| !p.is_empty() && !bp.contains(p))
        .collect();
    let label: String = extra
        .join("-")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect::<String>()
        .to_ascii_lowercase()
        .trim_matches('-')
        .to_owned();
    if label.is_empty() {
        "delta".to_owned()
    } else {
        label
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    let len = u32::try_from(s.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&s.as_bytes()[..s.len().min(len as usize)]);
}

/// Serializes `h`. Returns the bytes and the offset of `n_chunks` in them.
fn encode_header(h: &Header) -> (Vec<u8>, usize) {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&h.flags.to_le_bytes());
    out.extend_from_slice(&h.base_size.to_le_bytes());
    out.extend_from_slice(&h.data_pos.to_le_bytes());
    out.extend_from_slice(&h.header_sha256);
    out.extend_from_slice(&h.base_sha256);
    out.extend_from_slice(&h.target_sha256);
    put_str(&mut out, &h.label);
    put_str(&mut out, &h.base_path.to_string_lossy());
    put_str(&mut out, &h.base_name);
    put_str(&mut out, &h.base_general);
    put_str(&mut out, &h.base_source_url);
    put_str(&mut out, &h.base_source_rev);
    put_str(&mut out, &h.target_name);
    let n_pos = out.len();
    out.extend_from_slice(&h.n_chunks.to_le_bytes());
    (out, n_pos)
}

struct Reader<'a> {
    f: &'a File,
    pos: u64,
}

impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let mut b = [0u8; N];
        self.f
            .read_exact_at(&mut b, self.pos)
            .map_err(|e| Error::Format(format!("truncated .ggd: {e}")))?;
        self.pos += N as u64;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.bytes::<4>()?))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.bytes::<8>()?))
    }
    fn string(&mut self) -> Result<String, Error> {
        let len = self.u32()?;
        if len > MAX_STRING {
            return Err(Error::Format("absurd string length in .ggd header".into()));
        }
        let mut b = vec![0u8; len as usize];
        self.f
            .read_exact_at(&mut b, self.pos)
            .map_err(|e| Error::Format(format!("truncated .ggd header: {e}")))?;
        self.pos += u64::from(len);
        String::from_utf8(b).map_err(|_| Error::Format("non-UTF-8 string in .ggd header".into()))
    }
}

/// Reads the header of the `.ggd` at `path`, returning it with the file
/// offset of the first chunk record.
///
/// # Errors
/// Not a `.ggd`, an unsupported version, or a truncated file.
pub fn read_header(path: &Path) -> Result<(Header, u64), Error> {
    let f = File::open(path).map_err(io_ctx("cannot open", path))?;
    read_header_from(&f)
}

fn read_header_from(f: &File) -> Result<(Header, u64), Error> {
    let mut r = Reader { f, pos: 0 };
    if &r.bytes::<4>()? != MAGIC {
        return Err(Error::Format("not a GGUF weight delta (bad magic)".into()));
    }
    let version = r.u32()?;
    if version != VERSION {
        return Err(Error::Format(format!("unsupported .ggd version {version}")));
    }
    let mut h = Header {
        flags: r.u32()?,
        base_size: r.u64()?,
        data_pos: r.u64()?,
        header_sha256: r.bytes()?,
        base_sha256: r.bytes()?,
        target_sha256: r.bytes()?,
        ..Header::default()
    };
    h.label = r.string()?;
    h.base_path = PathBuf::from(r.string()?);
    h.base_name = r.string()?;
    h.base_general = r.string()?;
    h.base_source_url = r.string()?;
    h.base_source_rev = r.string()?;
    h.target_name = r.string()?;
    h.n_chunks = r.u64()?;
    if h.n_chunks > MAX_CHUNKS {
        return Err(Error::Format("absurd chunk count in .ggd header".into()));
    }
    Ok((h, r.pos))
}

/// Reads the chunk table that follows the header at `first`.
///
/// # Errors
/// A truncated file or a chunk pointing outside the base.
pub fn read_chunks(path: &Path, h: &Header, first: u64) -> Result<Vec<ChunkInfo>, Error> {
    let f = File::open(path).map_err(io_ctx("cannot open", path))?;
    read_chunks_from(&f, h, first)
}

fn read_chunks_from(f: &File, h: &Header, first: u64) -> Result<Vec<ChunkInfo>, Error> {
    let file_len = f.metadata()?.len();
    let mut r = Reader { f, pos: first };
    let mut out = Vec::with_capacity(usize::try_from(h.n_chunks).unwrap_or(0));
    for i in 0..h.n_chunks {
        let offset = r.u64()?;
        let len = r.u64()?;
        let base_check = r.bytes::<8>()?;
        let payload_len = r.u64()?;
        let payload_pos = r.pos;
        if offset < h.data_pos || offset.checked_add(len).is_none_or(|e| e > h.base_size) {
            return Err(Error::Format(format!(
                "chunk {i} points outside the base model"
            )));
        }
        if payload_pos
            .checked_add(payload_len)
            .is_none_or(|e| e > file_len)
        {
            return Err(Error::Format(format!(
                "chunk {i} payload runs past the end of the .ggd"
            )));
        }
        out.push(ChunkInfo {
            offset,
            len,
            base_check,
            payload_pos,
            payload_len,
        });
        r.pos = payload_pos + payload_len;
    }
    Ok(out)
}

fn check_of(bytes: &[u8]) -> [u8; 8] {
    let d: [u8; 32] = Sha256::digest(bytes).into();
    d[..8].try_into().expect("8 of 32")
}

/// The base path to record: just the filename when the base sits in the
/// same directory as the delta, so the pair can move together; otherwise the
/// base as given.
fn recorded_base_path(base: &Path, out: &Path) -> PathBuf {
    let dir_of = |p: &Path| {
        let d = p.parent().unwrap_or(Path::new("."));
        let d = if d.as_os_str().is_empty() {
            Path::new(".")
        } else {
            d
        };
        fs::canonicalize(d).ok()
    };
    let same_dir = matches!((dir_of(base), dir_of(out)), (Some(a), Some(b)) if a == b);
    if same_dir {
        base.file_name()
            .map_or_else(|| base.to_path_buf(), PathBuf::from)
    } else {
        base.to_path_buf()
    }
}

/// Writes the delta from `base` to `target` into `out`.
///
/// Both inputs are opened read-only. `out` is written through `<out>.part`
/// and renamed into place; nothing partial is left on failure.
///
/// # Errors
/// Files of different sizes or tensor tables, a non-GGUF input, or I/O.
pub fn write_delta(
    base: &Path,
    target: &Path,
    out: &Path,
    opts: &CreateOptions,
) -> Result<CreateReport, Error> {
    let bf = File::open(base).map_err(io_ctx("cannot open base", base))?;
    let tf = File::open(target).map_err(io_ctx("cannot open target", target))?;
    let base_size = bf.metadata()?.len();
    let target_size = tf.metadata()?.len();
    if base_size != target_size {
        return Err(Error::Mismatch(format!(
            "base is {base_size} bytes but target is {target_size}: not the same layout"
        )));
    }
    let layout = gguf::layout(base)?;
    let header_sha = header_sha256(base, layout.data_pos)?;
    if header_sha != header_sha256(target, layout.data_pos)? {
        return Err(Error::Mismatch(
            "base and target have different metadata or tensor tables".into(),
        ));
    }
    let name_of = |p: &Path| {
        p.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let label = opts
        .label
        .clone()
        .unwrap_or_else(|| default_label(base, target));
    let mut header = Header {
        flags: FLAG_DEFLATE | if opts.hash_base { FLAG_BASE_SHA } else { 0 },
        base_size,
        data_pos: layout.data_pos,
        header_sha256: header_sha,
        label: label.clone(),
        base_path: recorded_base_path(base, out),
        base_name: name_of(base),
        base_general: gguf::string_value(base, "general.name").unwrap_or_default(),
        base_source_url: gguf::string_value(base, "general.source.url").unwrap_or_default(),
        base_source_rev: gguf::string_value(base, "general.source.revision").unwrap_or_default(),
        target_name: name_of(target),
        ..Header::default()
    };

    let part = out.with_extension(format!("{EXTENSION}.part"));
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).map_err(io_ctx("cannot create", dir))?;
    }
    match write_delta_body(&bf, &tf, &layout, &mut header, opts, &part) {
        Ok(report) => {
            fs::rename(&part, out).map_err(io_ctx("cannot rename", &part))?;
            Ok(CreateReport { label, ..report })
        }
        Err(e) => {
            let _ = fs::remove_file(&part);
            Err(e)
        }
    }
}

/// A run of differing pieces within one tensor: start offset, base bytes,
/// target bytes.
type Run = Option<(u64, Vec<u8>, Vec<u8>)>;

fn flush_run(run: &mut Run, of: &mut File, report: &mut CreateReport) -> Result<(), Error> {
    let Some((offset, base_bytes, target_bytes)) = run.take() else {
        return Ok(());
    };
    let diff: Vec<u8> = base_bytes
        .iter()
        .zip(&target_bytes)
        .map(|(b, t)| t.wrapping_sub(*b))
        .collect();
    report.bytes_changed += diff.iter().filter(|d| **d != 0).count() as u64;
    report.bytes_spanned += diff.len() as u64;
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(6));
    enc.write_all(&diff)?;
    let payload = enc.finish()?;
    let mut rec = Vec::with_capacity(32);
    rec.extend_from_slice(&offset.to_le_bytes());
    rec.extend_from_slice(&(diff.len() as u64).to_le_bytes());
    rec.extend_from_slice(&check_of(&base_bytes));
    rec.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    of.write_all(&rec)?;
    of.write_all(&payload)?;
    report.payload_bytes += payload.len() as u64;
    report.chunks += 1;
    Ok(())
}

fn write_delta_body(
    bf: &File,
    tf: &File,
    layout: &Layout,
    header: &mut Header,
    opts: &CreateOptions,
    part: &Path,
) -> Result<CreateReport, Error> {
    let mut of = File::create(part).map_err(io_ctx("cannot create", part))?;
    let (hdr_bytes, n_pos) = encode_header(header);
    of.write_all(&hdr_bytes)?;

    let mut target_hash = Sha256::new();
    let mut base_hash = opts.hash_base.then(Sha256::new);
    // The header region and any gap before the first tensor are hashed but
    // never diffed: they were proven equal by the caller.
    let first_tensor = layout
        .tensors
        .first()
        .map_or(header.base_size, |t| t.offset);
    hash_range(tf, &mut target_hash, 0, first_tensor)?;
    if let Some(h) = base_hash.as_mut() {
        hash_range(bf, h, 0, first_tensor)?;
    }

    let piece = piece_len();
    let mut bb = vec![0u8; piece];
    let mut tb = vec![0u8; piece];
    let mut report = CreateReport::default();
    let mut run: Run = None;

    for t in &layout.tensors {
        let mut tensor_changed = false;
        let mut pos = t.offset;
        let end = t.offset + t.len;
        while pos < end {
            let n = usize::try_from((end - pos).min(PIECE)).unwrap_or(piece);
            bf.read_exact_at(&mut bb[..n], pos)?;
            tf.read_exact_at(&mut tb[..n], pos)?;
            target_hash.update(&tb[..n]);
            if let Some(h) = base_hash.as_mut() {
                h.update(&bb[..n]);
            }
            if bb[..n] == tb[..n] {
                flush_run(&mut run, &mut of, &mut report)?;
            } else {
                tensor_changed = true;
                match run.as_mut() {
                    Some((_, b, tt)) => {
                        b.extend_from_slice(&bb[..n]);
                        tt.extend_from_slice(&tb[..n]);
                    }
                    None => run = Some((pos, bb[..n].to_vec(), tb[..n].to_vec())),
                }
            }
            pos += n as u64;
        }
        // Pieces never cross a tensor boundary.
        flush_run(&mut run, &mut of, &mut report)?;
        if tensor_changed {
            report.tensors_changed += 1;
        }
    }

    header.n_chunks = report.chunks as u64;
    header.target_sha256 = target_hash.finalize().into();
    if let Some(h) = base_hash {
        header.base_sha256 = h.finalize().into();
    }
    of.seek(SeekFrom::Start(n_pos as u64))?;
    of.write_all(&header.n_chunks.to_le_bytes())?;
    of.seek(SeekFrom::Start(OFF_BASE_SHA))?;
    of.write_all(&header.base_sha256)?;
    of.seek(SeekFrom::Start(OFF_TARGET_SHA))?;
    of.write_all(&header.target_sha256)?;
    of.seek(SeekFrom::Start(OFF_FLAGS))?;
    of.write_all(&header.flags.to_le_bytes())?;
    of.sync_all()?;
    Ok(report)
}

/// Applies the chunks of the `.ggd` at `delta` onto `clone`, which must hold
/// the base's bytes and becomes the target. `h` and `first` come from
/// [`read_header`].
///
/// Each chunk's base bytes are checked before they are replaced, so a wrong
/// base fails at the first chunk. The clone may then be partly patched; the
/// caller decides whether to keep it.
///
/// # Errors
/// A base mismatch, a corrupt payload, or I/O.
pub fn apply(delta: &Path, h: &Header, first: u64, clone: &Path) -> Result<(), Error> {
    let df = File::open(delta).map_err(io_ctx("cannot open", delta))?;
    let chunks = read_chunks_from(&df, h, first)?;
    let cf = OpenOptions::new()
        .read(true)
        .write(true)
        .open(clone)
        .map_err(io_ctx("cannot open for writing", clone))?;
    for (i, c) in chunks.iter().enumerate() {
        let len = usize::try_from(c.len).map_err(|_| Error::Format("chunk too large".into()))?;
        let mut base = vec![0u8; len];
        cf.read_exact_at(&mut base, c.offset)?;
        if check_of(&base) != c.base_check {
            return Err(Error::Mismatch(format!(
                "chunk {i} at offset {}: the base does not hold the bytes this delta was cut against",
                c.offset
            )));
        }
        let plen = usize::try_from(c.payload_len)
            .map_err(|_| Error::Format("payload too large".into()))?;
        let mut payload = vec![0u8; plen];
        df.read_exact_at(&mut payload, c.payload_pos)?;
        let diff = if h.is_deflated() {
            let mut out = Vec::with_capacity(len);
            DeflateDecoder::new(&payload[..])
                .read_to_end(&mut out)
                .map_err(|e| Error::Format(format!("chunk {i} payload is corrupt: {e}")))?;
            out
        } else {
            payload
        };
        if diff.len() != len {
            return Err(Error::Format(format!(
                "chunk {i} inflates to {} bytes, want {len}",
                diff.len()
            )));
        }
        for (b, d) in base.iter_mut().zip(&diff) {
            *b = b.wrapping_add(*d);
        }
        cf.write_all_at(&base, c.offset)?;
    }
    cf.sync_all()?;
    Ok(())
}

/// Whether the file at `candidate` has the size and header the delta was cut
/// against. `Err` explains why not, in a few words fit for a list.
///
/// # Errors
/// `Mismatch` for a missing file, a wrong size, or a different header; `Io`
/// for anything else.
pub fn check_base(h: &Header, candidate: &Path) -> Result<(), Error> {
    let meta = match fs::metadata(candidate) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(Error::Mismatch("missing".into()));
        }
        Err(e) => return Err(Error::Io(e)),
    };
    if !meta.is_file() {
        return Err(Error::Mismatch("not a file".into()));
    }
    if meta.len() != h.base_size {
        return Err(Error::Mismatch(format!(
            "wrong size ({} bytes, want {})",
            meta.len(),
            h.base_size
        )));
    }
    if header_sha256(candidate, h.data_pos)? != h.header_sha256 {
        return Err(Error::Mismatch("different metadata or tensor table".into()));
    }
    Ok(())
}

/// Finds the base model `h` describes.
///
/// Candidates in order: `base_path` as recorded (a relative one against the
/// delta's directory), a file named `base_name` beside the delta, then each of
/// `extra`. The first whose size and header hash match (see [`check_base`])
/// wins.
///
/// # Errors
/// `Mismatch` listing every candidate tried and why it was rejected.
pub fn find_base(h: &Header, delta: &Path, extra: &[PathBuf]) -> Result<PathBuf, Error> {
    let dir = delta
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let mut candidates: Vec<PathBuf> = Vec::new();
    if !h.base_path.as_os_str().is_empty() {
        if h.base_path.is_absolute() {
            candidates.push(h.base_path.clone());
        } else {
            candidates.push(dir.join(&h.base_path));
        }
    }
    if !h.base_name.is_empty() {
        candidates.push(dir.join(&h.base_name));
    }
    candidates.extend(extra.iter().cloned());
    candidates.dedup();
    let mut reasons = Vec::new();
    for c in &candidates {
        match check_base(h, c) {
            Ok(()) => return Ok(c.clone()),
            Err(why) => reasons.push(format!("  {}: {why}", c.display())),
        }
    }
    Err(Error::Mismatch(format!(
        "cannot find the base model for {} ({}):\n{}",
        delta.display(),
        h.base_name,
        reasons.join("\n")
    )))
}

/// A human summary of a delta: header fields and the chunk list, with tensor
/// names when the base's `layout` is given.
#[must_use]
pub fn describe(h: &Header, chunks: &[ChunkInfo], layout: Option<&Layout>) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "  label:          {}", h.label);
    let _ = writeln!(s, "  base path:      {}", h.base_path.display());
    let _ = writeln!(s, "  base name:      {}", h.base_name);
    let _ = writeln!(s, "  base model:     {}", h.base_general);
    if !h.base_source_url.is_empty() {
        let _ = writeln!(
            s,
            "  base source:    {} @ {}",
            h.base_source_url, h.base_source_rev
        );
    }
    let _ = writeln!(s, "  base size:      {} bytes", h.base_size);
    let _ = writeln!(s, "  header sha256:  {}", hex(&h.header_sha256));
    if h.has_base_sha() {
        let _ = writeln!(s, "  base sha256:    {}", hex(&h.base_sha256));
    }
    let _ = writeln!(s, "  target name:    {}", h.target_name);
    let _ = writeln!(s, "  target sha256:  {}", hex(&h.target_sha256));
    let spanned: u64 = chunks.iter().map(|c| c.len).sum();
    let payload: u64 = chunks.iter().map(|c| c.payload_len).sum();
    let _ = writeln!(
        s,
        "  chunks:         {} spanning {spanned} bytes, {payload} bytes compressed",
        chunks.len()
    );
    for c in chunks {
        let name = layout
            .and_then(|l| l.tensor_at(c.offset))
            .map_or_else(String::new, |t| format!("  {}", t.name));
        let _ = writeln!(s, "    @{:>14} {:>12} bytes{name}", c.offset, c.len);
    }
    s
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::testing::Gguf;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gguf-delta-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// A base with three tensors: a small one, a 3 MiB one (so pieces
    /// matter), and another small one.
    fn base_builder() -> Gguf {
        let big: Vec<u8> = (0..3u32 << 20).map(|i| (i % 251) as u8).collect();
        Gguf::default()
            .str_val("general.architecture", "deepseek4")
            .str_val("general.name", "Tiny Test Model")
            .str_val("general.source.url", "https://example.invalid/tiny")
            .tensor("blk.0.a", &[64], 0, &[7u8; 64])
            .tensor("blk.1.big", &[3 << 20], 0, &big)
            .tensor("blk.2.c", &[100], 0, &[9u8; 100])
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, bytes).unwrap();
        p
    }

    fn range(t: &TensorSpan) -> std::ops::Range<usize> {
        t.offset as usize..(t.offset + t.len) as usize
    }

    #[test]
    fn round_trip_restores_the_target_exactly() {
        let d = dir("roundtrip");
        let base_bytes = base_builder().bytes();
        let base = write(&d, "tiny.gguf", &base_bytes);
        let layout = gguf::layout(&base).unwrap();
        let big = &layout.tensors[1];
        let mut target_bytes = base_bytes.clone();
        // One byte in the second MiB of the big tensor, and the whole of the
        // last small tensor.
        let edit = (big.offset + PIECE + 17) as usize;
        target_bytes[edit] = target_bytes[edit].wrapping_add(1);
        let c = &layout.tensors[2];
        target_bytes[range(c)].fill(42);
        let target = write(&d, "tiny-Abliterated.gguf", &target_bytes);
        let out = d.join("tiny.ggd");
        let report = write_delta(&base, &target, &out, &CreateOptions::default()).unwrap();
        assert_eq!(report.tensors_changed, 2);
        assert_eq!(report.chunks, 2);
        assert_eq!(report.label, "abliterated");
        assert_eq!(report.bytes_changed, 1 + c.len);

        let (h, first) = read_header(&out).unwrap();
        assert_eq!(h.base_size, base_bytes.len() as u64);
        assert_eq!(
            h.base_path,
            PathBuf::from("tiny.gguf"),
            "same dir: bare filename"
        );
        assert_eq!(h.base_name, "tiny.gguf");
        assert_eq!(h.base_general, "Tiny Test Model");
        assert_eq!(h.base_source_url, "https://example.invalid/tiny");
        assert_eq!(h.target_name, "tiny-Abliterated.gguf");
        assert!(!h.has_base_sha());
        assert!(h.is_deflated());
        let target_sha: [u8; 32] = Sha256::digest(&target_bytes).into();
        assert_eq!(h.target_sha256, target_sha);
        let chunks = read_chunks(&out, &h, first).unwrap();
        assert_eq!(chunks[0].offset, big.offset + PIECE);
        assert_eq!(chunks[0].len, PIECE, "one piece, not the whole tensor");
        assert_eq!(chunks[1].offset, c.offset);
        assert_eq!(chunks[1].len, c.len);
        assert!(chunks[1].payload_len < c.len, "a constant fill compresses");

        let clone = write(&d, "clone.gguf", &base_bytes);
        apply(&out, &h, first, &clone).unwrap();
        assert_eq!(fs::read(&clone).unwrap(), target_bytes);
        assert_eq!(fs::read(&base).unwrap(), base_bytes, "base untouched");
        assert!(check_base(&h, &base).is_ok());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn adjacent_pieces_coalesce_and_identical_files_yield_no_chunks() {
        let d = dir("coalesce");
        let base_bytes = base_builder().bytes();
        let base = write(&d, "b.gguf", &base_bytes);
        let layout = gguf::layout(&base).unwrap();
        let big = &layout.tensors[1];
        let mut t = base_bytes.clone();
        t[(big.offset + 5) as usize] ^= 0xff;
        t[(big.offset + PIECE + 5) as usize] ^= 0xff;
        let target = write(&d, "t.gguf", &t);
        let out = d.join("d.ggd");
        let opts = CreateOptions {
            label: Some("x".into()),
            hash_base: true,
        };
        let r = write_delta(&base, &target, &out, &opts).unwrap();
        assert_eq!(r.chunks, 1);
        assert_eq!(r.bytes_spanned, 2 * PIECE);
        assert_eq!(r.bytes_changed, 2);
        let (h, _) = read_header(&out).unwrap();
        assert!(h.has_base_sha());
        let base_sha: [u8; 32] = Sha256::digest(&base_bytes).into();
        assert_eq!(h.base_sha256, base_sha);
        assert_eq!(h.label, "x");

        // A file against itself: no chunks, and no label to derive.
        let out2 = d.join("none.ggd");
        let r = write_delta(&base, &base, &out2, &CreateOptions::default()).unwrap();
        assert_eq!(r.chunks, 0);
        assert_eq!(r.label, "delta");
        let (h, first) = read_header(&out2).unwrap();
        assert_eq!(h.n_chunks, 0);
        let clone = write(&d, "clone.gguf", &base_bytes);
        apply(&out2, &h, first, &clone).unwrap();
        assert_eq!(fs::read(clone).unwrap(), base_bytes);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn different_layouts_are_refused_at_create() {
        let d = dir("layouts");
        let base = write(&d, "b.gguf", &base_builder().bytes());
        let other = write(
            &d,
            "o.gguf",
            &Gguf::default()
                .str_val("general.architecture", "deepseek4")
                .tensor("x", &[8], 0, &[1u8; 8])
                .bytes(),
        );
        let e =
            write_delta(&base, &other, &d.join("x.ggd"), &CreateOptions::default()).unwrap_err();
        assert!(
            matches!(&e, Error::Mismatch(m) if m.contains("not the same layout")),
            "{e}"
        );
        // Same size, different tensor table.
        let mut renamed = base_builder().bytes();
        let pos = renamed.windows(7).position(|w| w == b"blk.0.a").unwrap();
        renamed[pos..pos + 7].copy_from_slice(b"blk.0.z");
        let renamed = write(&d, "r.gguf", &renamed);
        let e =
            write_delta(&base, &renamed, &d.join("y.ggd"), &CreateOptions::default()).unwrap_err();
        assert!(
            e.to_string()
                .contains("different metadata or tensor tables"),
            "{e}"
        );
        assert!(!d.join("y.ggd.part").exists(), "no partial output");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_base_with_altered_bytes_fails_the_chunk_check() {
        let d = dir("altered");
        let base_bytes = base_builder().bytes();
        let base = write(&d, "b.gguf", &base_bytes);
        let layout = gguf::layout(&base).unwrap();
        let c = &layout.tensors[2];
        let mut t = base_bytes.clone();
        t[c.offset as usize] ^= 1;
        let target = write(&d, "t.gguf", &t);
        let out = d.join("d.ggd");
        write_delta(&base, &target, &out, &CreateOptions::default()).unwrap();
        let (h, first) = read_header(&out).unwrap();
        // A "clone" that is not really the base, inside the chunk.
        let mut bad = base_bytes.clone();
        bad[(c.offset + 3) as usize] ^= 0x80;
        let clone = write(&d, "clone.gguf", &bad);
        let e = apply(&out, &h, first, &clone).unwrap_err();
        assert!(
            matches!(&e, Error::Mismatch(m) if m.contains("does not hold the bytes")),
            "{e}"
        );
        assert_eq!(
            fs::read(&clone).unwrap(),
            bad,
            "nothing written before the check"
        );
        // check_base only sees the header, so it still passes; a short file
        // and a different header do not.
        assert!(check_base(&h, &clone).is_ok());
        let short = write(&d, "short.gguf", b"GGUF");
        assert!(
            matches!(check_base(&h, &short), Err(Error::Mismatch(m)) if m.contains("wrong size"))
        );
        assert!(
            matches!(check_base(&h, &d.join("nope.gguf")), Err(Error::Mismatch(m)) if m == "missing")
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn base_path_is_recorded_as_given_when_dirs_differ() {
        let d = dir("recorded");
        let elsewhere = dir("recorded-elsewhere");
        let base_bytes = base_builder().bytes();
        let base = write(&d, "b.gguf", &base_bytes);
        let target = write(&d, "t.gguf", &base_bytes);
        let out = elsewhere.join("d.ggd");
        write_delta(&base, &target, &out, &CreateOptions::default()).unwrap();
        let (h, _) = read_header(&out).unwrap();
        assert_eq!(h.base_path, base);
        let _ = fs::remove_dir_all(&d);
        let _ = fs::remove_dir_all(&elsewhere);
    }

    #[test]
    fn find_base_follows_the_link_then_siblings_then_extras() {
        let d = dir("find");
        let elsewhere = dir("find-elsewhere");
        let base_bytes = base_builder().bytes();
        let base = write(&d, "b.gguf", &base_bytes);
        let target = write(&d, "t.gguf", &base_bytes);
        let out = elsewhere.join("d.ggd");
        write_delta(&base, &target, &out, &CreateOptions::default()).unwrap();
        let (h, _) = read_header(&out).unwrap();
        // 1. The recorded link.
        assert_eq!(find_base(&h, &out, &[]).unwrap(), base);
        // 2. Link broken, sibling by name.
        fs::rename(&base, elsewhere.join("b.gguf")).unwrap();
        assert_eq!(find_base(&h, &out, &[]).unwrap(), elsewhere.join("b.gguf"));
        // 3. Neither; an extra candidate. A wrong-size sibling is skipped
        //    with a reason, and the missing link is named too.
        fs::rename(elsewhere.join("b.gguf"), d.join("moved.gguf")).unwrap();
        write(&elsewhere, "b.gguf", b"short");
        let e = find_base(&h, &out, &[]).unwrap_err().to_string();
        assert!(e.contains("wrong size"), "{e}");
        assert!(e.contains("missing"), "{e}");
        assert_eq!(
            find_base(&h, &out, &[d.join("moved.gguf")]).unwrap(),
            d.join("moved.gguf")
        );
        // A relative link resolves against the delta's directory.
        let mut rel = h.clone();
        rel.base_path = Path::new("..")
            .join(d.file_name().unwrap())
            .join("moved.gguf");
        assert_eq!(
            find_base(&rel, &out, &[]).unwrap(),
            elsewhere.join(&rel.base_path)
        );
        let _ = fs::remove_dir_all(&d);
        let _ = fs::remove_dir_all(&elsewhere);
    }

    #[test]
    fn default_labels() {
        let p = |s: &str| PathBuf::from(s);
        assert_eq!(
            default_label(
                &p("DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"),
                &p(
                    "DeepSeek-V4-Flash-Vision-Exp-Abliterated-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"
                )
            ),
            "abliterated"
        );
        assert_eq!(default_label(&p("a.gguf"), &p("a.gguf")), "delta");
        assert_eq!(
            default_label(&p("m.gguf"), &p("m_uncensored_v2.gguf")),
            "uncensored-v2"
        );
    }

    #[test]
    fn describe_names_base_metadata_and_tensors() {
        let d = dir("describe");
        let base_bytes = base_builder().bytes();
        let base = write(&d, "b.gguf", &base_bytes);
        let layout = gguf::layout(&base).unwrap();
        let mut t = base_bytes;
        t[layout.tensors[2].offset as usize] ^= 1;
        let target = write(&d, "t.gguf", &t);
        let out = d.join("d.ggd");
        write_delta(&base, &target, &out, &CreateOptions::default()).unwrap();
        let (h, first) = read_header(&out).unwrap();
        let chunks = read_chunks(&out, &h, first).unwrap();
        let text = describe(&h, &chunks, Some(&layout));
        assert!(text.contains("Tiny Test Model"), "{text}");
        assert!(text.contains("blk.2.c"), "{text}");
        assert!(!describe(&h, &chunks, None).contains("blk.2.c"));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn junk_is_not_a_delta() {
        let d = dir("junk");
        let j = write(&d, "j.ggd", b"GGUF not a delta");
        assert!(matches!(read_header(&j), Err(Error::Format(m)) if m.contains("bad magic")));
        let short = write(&d, "s.ggd", b"GG");
        assert!(matches!(read_header(&short), Err(Error::Format(_))));
        assert!(is_delta_path(Path::new("x.GGD")));
        assert!(!is_delta_path(Path::new("x.gguf")));
        let _ = fs::remove_dir_all(&d);
    }
}
