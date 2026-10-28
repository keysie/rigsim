"""A model as the engine sees it: architecture dimensions and the tensor list with storage types.

Two sources, in this order: the GGUF file's own header (gguf.py — exact tensor shapes and types), else the published
architecture configuration (ARCHS below, from each model's config.json on its official repository) with the
quantiser's documented type rules (FILE_TYPES). Every model names its source."""
from __future__ import annotations

from dataclasses import dataclass, field

from .quant import bytes_for


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]          # ggml order: (columns = input width, rows = output width)
    type: str

    @property
    def n(self) -> int:
        out = 1
        for s in self.shape:
            out *= s
        return out

    @property
    def bytes(self) -> float:
        return bytes_for(self.type, self.n)


@dataclass
class Model:
    name: str
    arch: str                       # llama (llama 1/2/3, TinyLlama), gemma
    d_model: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    tensors: list[Tensor] = field(default_factory=list)
    source: str = ""

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    def tensor(self, name: str) -> Tensor | None:
        return next((t for t in self.tensors if t.name == name), None)

    @property
    def weight_bytes(self) -> float:
        return sum(t.bytes for t in self.tensors)


# Published configurations (config.json of the official repository). head_dim = d_model / heads unless stated.
ARCHS = {
    "llama-2-7b": dict(arch="llama", d_model=4096, layers=32, heads=32, kv_heads=32, ffn=11008, vocab=32000, tied=False,
                       source="meta-llama/Llama-2-7b-hf config.json"),
    "llama-3.2-1b": dict(arch="llama", d_model=2048, layers=16, heads=32, kv_heads=8, ffn=8192, vocab=128256, tied=True,
                         source="meta-llama/Llama-3.2-1B config.json"),
    "llama-3.1-8b": dict(arch="llama", d_model=4096, layers=32, heads=32, kv_heads=8, ffn=14336, vocab=128256, tied=False,
                         source="meta-llama/Llama-3.1-8B config.json"),
    "tinyllama-1.1b": dict(arch="llama", d_model=2048, layers=22, heads=32, kv_heads=4, ffn=5632, vocab=32000, tied=False,
                           source="TinyLlama/TinyLlama-1.1B-Chat-v1.0 config.json"),
    "gemma-2b": dict(arch="gemma", d_model=2048, layers=18, heads=8, kv_heads=1, head_dim=256, ffn=16384, vocab=256000,
                     tied=True, source="google/gemma-2b config.json"),
}

# llama.cpp's quantiser (llama-quant.cpp, llama_tensor_get_type): for a "mostly X" file every 2-D weight is X except
# the output projection, kept at Q6_K for the 4- and 5-bit types; with tied embeddings the token embedding is that
# output tensor. Norms stay f32. The GGUF header, when available, replaces these rules.
FILE_TYPES = {"q4_0": dict(weights="q4_0", output="q6_k"), "q8_0": dict(weights="q8_0", output="q8_0"),
              "q4_k_m": dict(weights="q4_k", output="q6_k"), "f16": dict(weights="f16", output="f16")}


def from_config(key: str, file_type: str) -> Model:
    a = dict(ARCHS[key])
    ft = FILE_TYPES[file_type.lower()]
    hd = a.pop("head_dim", a["d_model"] // a["heads"])
    tied, src = a.pop("tied"), a.pop("source")
    m = Model(name=key, head_dim=hd, source=f"{src}; types from llama.cpp's quantiser rules ({file_type})", **a)
    d, kv, q = m.d_model, m.kv_dim, m.heads * hd
    w = ft["weights"]
    t = [Tensor("token_embd.weight", (d, m.vocab), ft["output"] if tied else w)]
    for i in range(m.layers):
        p = f"blk.{i}."
        t += [Tensor(p + "attn_norm.weight", (d,), "f32"),
              Tensor(p + "attn_q.weight", (d, q), w), Tensor(p + "attn_k.weight", (d, kv), w),
              Tensor(p + "attn_v.weight", (d, kv), w), Tensor(p + "attn_output.weight", (q, d), w),
              Tensor(p + "ffn_norm.weight", (d,), "f32"),
              Tensor(p + "ffn_gate.weight", (d, m.ffn), w), Tensor(p + "ffn_up.weight", (d, m.ffn), w),
              Tensor(p + "ffn_down.weight", (m.ffn, d), w)]
    t.append(Tensor("output_norm.weight", (d,), "f32"))
    if not tied:
        t.append(Tensor("output.weight", (d, m.vocab), ft["output"]))
    m.tensors = t
    return m
