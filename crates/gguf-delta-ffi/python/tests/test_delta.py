"""Round trip through the ctypes binding on a tiny synthetic GGUF pair.

Run with the shared library built::

    cargo build --release -p gguf-delta-ffi
    GGUF_DELTA_LIB=target/release/libgguf_delta.dylib python3 -m unittest discover crates/gguf-delta-ffi/python/tests
"""

import hashlib
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gguf_delta  # noqa: E402


def tiny_gguf(tensors: list[tuple[str, bytes]], name: str = "Tiny") -> bytes:
    """GGUF v3 with one string metadata key and the given tensors."""
    def s(x: str) -> bytes:
        return struct.pack("<Q", len(x)) + x.encode()

    kv = s("general.name") + struct.pack("<I", 8) + s(name)
    infos, data = b"", b""
    for tname, payload in tensors:
        rel = (len(data) + 31) // 32 * 32
        data = data.ljust(rel, b"\0") + payload
        infos += s(tname) + struct.pack("<I", 1) + struct.pack("<Q", len(payload)) + struct.pack("<I", 0) + struct.pack("<Q", rel)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", 1) + kv + infos
    return head.ljust((len(head) + 31) // 32 * 32, b"\0") + data


class DeltaTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            base = tiny_gguf([("a", b"\x01" * 64), ("b", b"\x02" * 64)])
            target = base[:-64] + b"\x09" * 64
            (d / "tiny.gguf").write_bytes(base)
            (d / "tiny-edit.gguf").write_bytes(target)

            rep = gguf_delta.create(d / "tiny.gguf", d / "tiny-edit.gguf", d / "tiny.ggd")
            self.assertEqual(rep.chunks, 1)
            self.assertEqual(rep.bytes_changed, 64)
            self.assertEqual(rep.tensors_changed, 1)

            with gguf_delta.Delta(d / "tiny.ggd") as delta:
                self.assertEqual(delta.label, "edit")
                self.assertEqual(delta.base_name, "tiny.gguf")
                self.assertEqual(delta.base_model, "Tiny")
                self.assertEqual(delta.base_size, len(base))
                self.assertEqual(delta.target_sha256, hashlib.sha256(target).hexdigest())
                self.assertEqual([c.length for c in delta.chunks], [64])
                self.assertEqual(delta.find_base(), d / "tiny.gguf")
                delta.check_base(d / "tiny.gguf")
                with self.assertRaises(gguf_delta.DeltaError):
                    delta.check_base(d / "tiny.ggd")

                out = delta.materialize(d / "build" / "tiny-edit.gguf")
                self.assertEqual(out.read_bytes(), target)
                self.assertEqual((d / "tiny.gguf").read_bytes(), base, "base untouched")

                clone = d / "clone.gguf"
                clone.write_bytes(base)
                delta.apply(clone)
                self.assertEqual(clone.read_bytes(), target)

            with self.assertRaises(gguf_delta.DeltaError):
                gguf_delta.Delta(d / "tiny.gguf")
            self.assertTrue(gguf_delta.version())


if __name__ == "__main__":
    unittest.main()
