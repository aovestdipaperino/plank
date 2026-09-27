# gguf-delta

Weight deltas between two GGUF model files that share a layout, as used by
the [plank](https://github.com/aovestdipaperino/plank) coding agent to load a
derived checkpoint (an abliterated variant, a fine-tune re-quantized with the
same recipe) from the base model plus a small `.ggd` file.

Two files of the same quantization recipe often differ in a few dozen tensors
and share the rest byte for byte. A `.ggd` records only the changed spans, as
a byte-wise `(target - base) mod 256` difference that deflate compresses to a
fraction of its size: a rank-1 residual edit moves most int8 weights by 0 or
±1, so the difference is mostly zeros and small numbers even though the raw
target bytes look random. On an 87 GB DeepSeek V4 Flash checkpoint with 33
edited `Q8_0` tensors, the 1.18 GB of changed spans become a 440 MB delta.

The delta is self-describing. It carries the base's size, a hash of its
header (metadata and tensor table), the base's filename and its `general.*`
metadata strings, plus the target's whole-file hash. There is no path: a delta
always sits beside its base and is resolved as `<delta dir>/<base_name>`, so
the pair keeps working wherever it is moved together.

```rust
use gguf_delta::{write_delta, apply, read_header, CreateOptions};
use std::path::Path;

// Create.
let report = write_delta(
    Path::new("base.gguf"),
    Path::new("base-abliterated.gguf"),
    Path::new("abliterated.ggd"),
    &CreateOptions::default(),
)?;
println!("{} tensors changed, {} bytes compressed", report.tensors_changed, report.payload_bytes);

// Apply: `clone.gguf` must start as a byte copy of the base (an APFS or
// reflink clone is ideal); it becomes the target.
let (header, first_chunk) = read_header(Path::new("abliterated.ggd"))?;
apply(Path::new("abliterated.ggd"), &header, first_chunk, Path::new("clone.gguf"))?;
# Ok::<(), gguf_delta::Error>(())
```

Neither input is ever opened for writing. Every chunk carries a hash of the
base bytes it replaces, so applying a delta onto the wrong base fails at the
first chunk instead of producing a corrupt model.

## Other languages

The sibling crate `gguf-delta-ffi` exposes this API as a C library
(`include/gguf_delta.h`) and a `ctypes`-based Python package, so a `.ggd` can
be opened and materialized into a full GGUF from C or Python without a Rust
toolchain at run time.

## Command line

```sh
cargo install gguf-delta            # installs `ggd`
ggd create base.gguf base-abliterated.gguf abliterated.ggd [--label NAME] [--hash-base]
ggd info abliterated.ggd
```

`create` streams both files once and prints what changed; `info` prints the
header, whether the base is found beside the delta, and the chunk list with
tensor names.

## Format

Little-endian, streamed: header, then chunks in file order.

```
magic            4   b"GGDL"
version          u32 1
flags            u32 bit 0: payloads deflate-compressed; bit 1: base_sha256 present
base_size        u64
data_pos         u64 tensor data start in the base
header_sha256    32  sha256(base[0 .. data_pos])
base_sha256      32  sha256 of the whole base, or zero
target_sha256    32  sha256 of the whole target
label            str
base_name        str
base_general     str general.name
base_source_url  str general.source.url
base_source_rev  str general.source.revision
target_name      str
n_chunks         u64
chunk[n_chunks]:
  offset         u64 absolute byte offset in the model
  len            u64 uncompressed length
  base_check     8   first 8 bytes of sha256(base[offset .. offset+len])
  payload_len    u64
  payload        bytes
```

`str` is `u32 len + UTF-8`. Chunks are 1 MiB pieces within a tensor, coalesced
when adjacent and never crossing a tensor boundary.

## Features

- `testing`: an in-memory GGUF builder (`gguf_delta::testing::Gguf`) for tests
  of code that consumes this crate.

License: MIT.
