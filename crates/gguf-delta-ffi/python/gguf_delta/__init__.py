"""Load .ggd weight deltas from Python.

A ``.ggd`` records where a GGUF differs from a same-layout base GGUF. This
module wraps the ``gguf_delta`` shared library through ``ctypes``, so there
is nothing to compile on the Python side::

    from gguf_delta import Delta

    d = Delta("model-abliterated.ggd")        # base must sit beside it
    print(d.label, d.base_name, len(d.chunks))
    path = d.materialize("build/model-abliterated.gguf")
    # hand `path` to llama.cpp, or anything else that loads GGUF

The library is found from ``GGUF_DELTA_LIB`` if set, then beside this file,
then through the system loader (``libgguf_delta.dylib`` / ``.so``). Build it
with ``cargo build --release -p gguf-delta-ffi``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Chunk", "Delta", "DeltaError", "Report", "create", "version"]

_ERR = 4096


class DeltaError(Exception):
    """Anything the library refused: bad file, missing base, wrong base, I/O."""


@dataclass(frozen=True)
class Chunk:
    offset: int
    length: int
    payload_length: int


@dataclass(frozen=True)
class Report:
    tensors_changed: int
    chunks: int
    bytes_spanned: int
    bytes_changed: int
    payload_bytes: int


class _CReport(ctypes.Structure):
    _fields_ = [(f, ctypes.c_uint64) for f in Report.__dataclass_fields__]


def _candidates() -> list[str]:
    names = {
        "darwin": ["libgguf_delta.dylib"],
        "win32": ["gguf_delta.dll"],
    }.get(sys.platform, ["libgguf_delta.so"])
    out: list[str] = []
    env = os.environ.get("GGUF_DELTA_LIB")
    if env:
        out.append(env)
    here = Path(__file__).resolve().parent
    for n in names:
        out.append(str(here / n))
    found = ctypes.util.find_library("gguf_delta")
    if found:
        out.append(found)
    out.extend(names)
    return out


def _load() -> ctypes.CDLL:
    errors = []
    for c in _candidates():
        try:
            return ctypes.CDLL(c)
        except OSError as e:  # noqa: PERF203
            errors.append(f"{c}: {e}")
    raise DeltaError(
        "cannot load the gguf_delta shared library; set GGUF_DELTA_LIB or build it "
        "with `cargo build --release -p gguf-delta-ffi`. Tried:\n  " + "\n  ".join(errors)
    )


_lib: ctypes.CDLL | None = None


def _l() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = _load()
        P, S, U64, I = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_int
        CS, CB = ctypes.c_char_p, ctypes.c_char_p
        lib.ggd_version.restype = CS
        lib.ggd_open.restype = P
        lib.ggd_open.argtypes = [CS, CB, S]
        lib.ggd_close.argtypes = [P]
        for name in (
            "ggd_label",
            "ggd_base_name",
            "ggd_base_model",
            "ggd_base_source_url",
            "ggd_base_source_revision",
            "ggd_target_name",
            "ggd_target_sha256",
        ):
            fn = getattr(lib, name)
            fn.restype = CS
            fn.argtypes = [P]
        lib.ggd_base_size.restype = U64
        lib.ggd_base_size.argtypes = [P]
        lib.ggd_chunk_count.restype = S
        lib.ggd_chunk_count.argtypes = [P]
        lib.ggd_chunk.restype = I
        lib.ggd_chunk.argtypes = [P, S, ctypes.POINTER(U64), ctypes.POINTER(U64), ctypes.POINTER(U64)]
        lib.ggd_find_base.restype = I
        lib.ggd_find_base.argtypes = [P, CB, S, CB, S]
        lib.ggd_check_base.restype = I
        lib.ggd_check_base.argtypes = [P, CS, CB, S]
        lib.ggd_materialize.restype = I
        lib.ggd_materialize.argtypes = [P, CS, CB, S]
        lib.ggd_apply.restype = I
        lib.ggd_apply.argtypes = [P, CS, CB, S]
        lib.ggd_create.restype = I
        lib.ggd_create.argtypes = [CS, CS, CS, CS, I, ctypes.POINTER(_CReport), CB, S]
        _lib = lib
    return _lib


def _p(path: str | os.PathLike[str]) -> bytes:
    return os.fsencode(os.fspath(path))


def _err() -> ctypes.Array:
    return ctypes.create_string_buffer(_ERR)


def version() -> str:
    """Version of the shared library."""
    return _l().ggd_version().decode()


class Delta:
    """An opened ``.ggd``. Use as a context manager or call :meth:`close`."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        err = _err()
        self._h = _l().ggd_open(_p(path), err, _ERR)
        if not self._h:
            raise DeltaError(err.value.decode(errors="replace"))

    def close(self) -> None:
        if getattr(self, "_h", None):
            _l().ggd_close(self._h)
            self._h = None

    def __enter__(self) -> Delta:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _s(self, name: str) -> str:
        return getattr(_l(), name)(self._h).decode()

    @property
    def label(self) -> str:
        return self._s("ggd_label")

    @property
    def base_name(self) -> str:
        """Filename of the base; it is expected beside the ``.ggd``."""
        return self._s("ggd_base_name")

    @property
    def base_model(self) -> str:
        """``general.name`` of the base."""
        return self._s("ggd_base_model")

    @property
    def base_source_url(self) -> str:
        return self._s("ggd_base_source_url")

    @property
    def base_source_revision(self) -> str:
        return self._s("ggd_base_source_revision")

    @property
    def target_name(self) -> str:
        return self._s("ggd_target_name")

    @property
    def target_sha256(self) -> str:
        """Hex SHA-256 of the full target the delta reproduces."""
        return self._s("ggd_target_sha256")

    @property
    def base_size(self) -> int:
        return int(_l().ggd_base_size(self._h))

    @property
    def chunks(self) -> list[Chunk]:
        lib = _l()
        n = lib.ggd_chunk_count(self._h)
        out = []
        off, ln, pl = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
        for i in range(n):
            lib.ggd_chunk(self._h, i, off, ln, pl)
            out.append(Chunk(off.value, ln.value, pl.value))
        return out

    def find_base(self) -> Path:
        """The base beside the delta, verified by size and header hash."""
        buf = ctypes.create_string_buffer(_ERR)
        err = _err()
        if _l().ggd_find_base(self._h, buf, _ERR, err, _ERR) != 0:
            raise DeltaError(err.value.decode(errors="replace"))
        return Path(os.fsdecode(buf.value))

    def check_base(self, base: str | os.PathLike[str]) -> None:
        """Raise unless ``base`` has the size and header the delta expects."""
        err = _err()
        if _l().ggd_check_base(self._h, _p(base), err, _ERR) != 0:
            raise DeltaError(err.value.decode(errors="replace"))

    def materialize(self, out: str | os.PathLike[str]) -> Path:
        """Clone the base beside the delta to ``out`` and apply the delta.

        Returns ``out`` as a :class:`Path`, now a complete GGUF identical to
        the original target. On APFS the clone is instant and shares blocks
        with the base; elsewhere it is a reflink or a copy.
        """
        err = _err()
        if _l().ggd_materialize(self._h, _p(out), err, _ERR) != 0:
            raise DeltaError(err.value.decode(errors="replace"))
        return Path(out)

    def apply(self, clone: str | os.PathLike[str]) -> None:
        """Patch ``clone``, which must hold the base's bytes, in place."""
        err = _err()
        if _l().ggd_apply(self._h, _p(clone), err, _ERR) != 0:
            raise DeltaError(err.value.decode(errors="replace"))

    def __repr__(self) -> str:
        return f"Delta({str(self.path)!r}, label={self.label!r}, base={self.base_name!r})"


def create(
    base: str | os.PathLike[str],
    target: str | os.PathLike[str],
    out: str | os.PathLike[str],
    label: str | None = None,
    hash_base: bool = False,
) -> Report:
    """Write the delta from ``base`` to ``target`` into ``out``."""
    rep = _CReport()
    err = _err()
    rc = _l().ggd_create(
        _p(base),
        _p(target),
        _p(out),
        label.encode() if label is not None else None,
        1 if hash_base else 0,
        ctypes.byref(rep),
        err,
        _ERR,
    )
    if rc != 0:
        raise DeltaError(err.value.decode(errors="replace"))
    return Report(**{f: int(getattr(rep, f)) for f in Report.__dataclass_fields__})
