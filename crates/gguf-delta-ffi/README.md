# gguf-delta-ffi

C ABI for [gguf-delta](../gguf-delta): open, inspect and materialize `.ggd`
weight deltas from C, from Python through `ctypes`, or from anything with a
foreign function interface. One shared library, one header, one Python
module that needs no compiler.

```sh
cargo build --release -p gguf-delta-ffi
# -> target/release/libgguf_delta.{dylib,so}  and  libgguf_delta.a
```

## C

The header is `include/gguf_delta.h`. Open a delta, find its base beside it,
materialize a full GGUF:

```c
ggd_delta *d = ggd_open("model-abliterated.ggd", err, sizeof err);
ggd_materialize(d, "build/model-abliterated.gguf", err, sizeof err);
ggd_close(d);
```

`examples/load.c` is the complete program:

```sh
cc -I crates/gguf-delta-ffi/include crates/gguf-delta-ffi/examples/load.c \
   -L target/release -lgguf_delta -o load
./load model-abliterated.ggd build/model-abliterated.gguf
```

Every fallible call takes an error buffer and returns `NULL` or `-1` on
failure with a message in the buffer. Strings returned by the accessors are
owned by the handle until `ggd_close`.

## Python

`python/gguf_delta` is a pure-Python package over the same library. Point it
at the shared object with `GGUF_DELTA_LIB`, or put the library beside the
package, or install it where the system loader finds it.

```python
from gguf_delta import Delta

with Delta("model-abliterated.ggd") as d:
    print(d.label, d.base_model, len(d.chunks))
    path = d.materialize("build/model-abliterated.gguf")
# hand `path` to llama.cpp or any GGUF loader
```

`gguf_delta.create(base, target, out)` writes a delta and returns a report.
Tests:

```sh
GGUF_DELTA_LIB=target/release/libgguf_delta.dylib \
  python3 -m unittest discover crates/gguf-delta-ffi/python/tests
```

## What "load" means here

Inference engines memory-map a GGUF, so a delta cannot be applied in memory
without touching the engine. `materialize` instead clones the base beside the
delta (an APFS `clonefile` on macOS, instant and block-sharing; a reflink or
copy elsewhere), patches the clone in place with per-chunk verification of
the base bytes, and gives you a path. The originals are never written.

License: MIT.
