"""GGUF header reader (format v2/v3, ggml's gguf.md): metadata and tensor infos only, the weights are never read.
Gives the exact tensor shapes and storage types of a model file."""
from __future__ import annotations

import struct
from pathlib import Path

from .model import Model, Tensor

GGML_TYPES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1", 10: "q2_k",
              11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k", 15: "q8_k", 20: "iq4_nl", 23: "iq4_xs", 30: "bf16"}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _R:
    def __init__(self, f):
        self.f = f

    def get(self, fmt: str):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(n))[0]

    def string(self) -> str:
        return self.f.read(self.get("<Q")).decode("utf-8", "replace")

    def value(self, t: int):
        if t in _SCALAR:
            return self.get(_SCALAR[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.get("<I"), self.get("<Q")
            if et in _SCALAR and n > 4096:                    # long arrays (vocabulary scores): skip the bytes
                self.f.seek(n * struct.calcsize(_SCALAR[et]), 1); return f"<{n} values>"
            if et == 8 and n > 4096:
                for _ in range(n):
                    self.f.seek(self.get("<Q"), 1)
                return f"<{n} strings>"
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"GGUF value type {t}")


def read(path: str | Path) -> tuple[dict, list[Tensor]]:
    with open(path, "rb") as f:
        r = _R(f)
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = r.get("<I")
        if version < 2:
            raise ValueError(f"{path}: GGUF v{version} not supported")
        n_tensors, n_kv = r.get("<Q"), r.get("<Q")
        meta = {"gguf.version": version}
        for _ in range(n_kv):
            k = r.string()
            meta[k] = r.value(r.get("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = tuple(r.get("<Q") for _ in range(r.get("<I")))
            t = r.get("<I")
            r.get("<Q")                                       # offset
            tensors.append(Tensor(name, dims, GGML_TYPES.get(t, f"type{t}")))
    return meta, tensors


def model(path: str | Path) -> Model:
    meta, tensors = read(path)
    arch = meta.get("general.architecture", "llama")
    g = lambda k, d=None: meta.get(f"{arch}.{k}", d)
    d, heads = g("embedding_length"), g("attention.head_count")
    emb = next(t for t in tensors if t.name == "token_embd.weight")
    hd = g("attention.key_length") or d // heads
    return Model(name=meta.get("general.name", Path(path).stem), arch=arch, d_model=d, layers=g("block_count"),
                 heads=heads, kv_heads=g("attention.head_count_kv", heads), head_dim=hd,
                 ffn=g("feed_forward_length"), vocab=emb.shape[1], tensors=tensors, source=f"GGUF header {Path(path).name}")
