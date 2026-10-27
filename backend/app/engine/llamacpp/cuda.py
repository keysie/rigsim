"""llama.cpp, CUDA backend: one forward pass as operations (proposal §6), llama-bench on top of it.

Kernel paths (ggml-cuda, mid-2025 builds): a quantised weight times few columns (<= MMVQ_MAX_BATCH) goes through
MMVQ — the activations quantised to q8_1, a dot-product kernel streaming the weight once (memory-bound); more columns
go through MMQ — int8 tensor-core tiles on q8_1 activations (compute-bound at prompt sizes). f16/f32 weights use
cuBLAS. Attention is FlashAttention (llama-bench's default on CUDA): a vector kernel streaming the KV cache for one
token, tensor-core tiles for a prompt. Everything else is small elementwise kernels, one per graph node after
ggml-cuda's fusions.

The kernel factors (the engine's speed against the reference: cuBLAS for a class, a plain stream read for bandwidth)
are 1.0 and marked unmeasured until per-kernel measurements exist (llama.cpp's test-backend-ops perf mode). They are
not fitted to llama-bench results: those validate."""
from __future__ import annotations

from dataclasses import dataclass

from ...hw.execute import Op, StepResult, execute
from ...hw.machine import Machine
from ..model import Model

RUNTIME = "cuda-12.8"
MMVQ_MAX_BATCH = 8                      # ggml-cuda: columns up to which MMVQ is used for quantised weights
QUANTISED = {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq4_nl", "iq4_xs"}
Q8_1_BYTES_PER_VALUE = 36 / 32          # activations quantised for MMVQ / MMQ: 32 int8 + fp16 scale + fp16 sum
ACT_BYTES = 4                           # activations between kernels are f32
KV_BYTES = 2                            # KV cache f16 (llama-bench default)


@dataclass
class Factor:
    value: float = 1.0
    basis: str = "unmeasured (no per-kernel measurement yet)"


# one factor per operation kind: compute against the class's reference library, memory against a stream read
FACTORS = {
    "mmvq": dict(memory=Factor()),
    "mmq": dict(compute=Factor(), memory=Factor()),
    "cublas": dict(compute=Factor(), memory=Factor()),
    "fa_vec": dict(memory=Factor()),
    "fa_mma": dict(compute=Factor(), memory=Factor()),
    "elementwise": dict(memory=Factor()),
}


def _op(name: str, kind: str, path: str, *, op_class=None, ops=0.0, bytes_=0.0, kernels=1) -> Op:
    f = FACTORS[path]
    cf, mf = f.get("compute", Factor()), f.get("memory", Factor())
    return Op(name, kind, op_class, ops, bytes_, kernels, cf.value, mf.value,
              factor_basis=f"{path}: compute {cf.basis}; memory {mf.basis}")


def matmul(name: str, w_type: str, cols: int, rows: int, n: int, w_bytes: float) -> Op:
    """Weight (cols x rows, stored as w_type) times n activation columns."""
    ops = 2.0 * n * cols * rows
    if w_type in QUANTISED and n <= MMVQ_MAX_BATCH:
        b = w_bytes + n * cols * (ACT_BYTES + Q8_1_BYTES_PER_VALUE) + n * rows * ACT_BYTES
        return _op(name, "matmul", "mmvq", ops=ops, bytes_=b, kernels=2)        # quantise + dot product
    if w_type in QUANTISED:
        b = w_bytes + n * cols * (ACT_BYTES + Q8_1_BYTES_PER_VALUE) + n * rows * ACT_BYTES
        return _op(name, "matmul", "mmq", op_class="int8_tensor", ops=ops, bytes_=b, kernels=2)
    cls = "fp16_tensor_fp32acc" if w_type in ("f16", "bf16") else "tf32_tensor"
    b = w_bytes + n * (cols + rows) * ACT_BYTES
    return _op(name, "matmul", "cublas", op_class=cls, ops=ops, bytes_=b, kernels=1)


def elementwise(name: str, bytes_: float, kernels: int = 1) -> Op:
    return _op(name, "elementwise", "elementwise", bytes_=bytes_, kernels=kernels)


def attention(m: Model, n: int, depth: int) -> Op:
    """FlashAttention over the cache: n new tokens after `depth` cached ones, causal."""
    kv_len = depth + n
    kv_bytes = 2 * kv_len * m.kv_dim * KV_BYTES
    q = m.heads * m.head_dim
    io = 2 * n * q * ACT_BYTES                                   # read Q, write the output
    ops = 4.0 * m.heads * m.head_dim * n * (depth + (n + 1) / 2)  # QK^T and PV, causal half of the new block
    if n == 1:
        return _op("attention", "attention", "fa_vec", ops=ops, bytes_=kv_bytes + io)
    return _op("attention", "attention", "fa_mma", op_class="fp16_tensor_fp32acc", ops=ops, bytes_=kv_bytes + io)


def forward(m: Model, n: int, depth: int, n_outputs: int = 1) -> list[Op]:
    """One forward pass of n tokens after `depth` cached ones; logits for n_outputs tokens."""
    d, kv, q, ff = m.d_model, m.kv_dim, m.heads * m.head_dim, m.ffn
    ops = [elementwise("token embedding", n * d * ACT_BYTES * 2, kernels=2 if m.arch == "gemma" else 1)]
    for i in range(m.layers):
        p = f"blk.{i}."
        t = {k: m.tensor(p + k + ".weight") for k in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")}
        ops.append(elementwise(f"{p}attn_norm", n * d * ACT_BYTES * 2 + d * 4))
        for k, rows in (("attn_q", q), ("attn_k", kv), ("attn_v", kv)):
            ops.append(matmul(p + k, t[k].type, d, rows, n, t[k].bytes))
        ops.append(elementwise(f"{p}rope", n * (q + kv) * ACT_BYTES * 2, kernels=2))
        ops.append(elementwise(f"{p}kv store", n * kv * (ACT_BYTES + KV_BYTES) * 2, kernels=2))
        ops.append(attention(m, n, depth))
        ops.append(matmul(p + "attn_output", t["attn_output"].type, q, d, n, t["attn_output"].bytes))
        ops.append(elementwise(f"{p}residual", n * d * ACT_BYTES * 3))
        ops.append(elementwise(f"{p}ffn_norm", n * d * ACT_BYTES * 2 + d * 4))
        for k in ("ffn_gate", "ffn_up"):
            ops.append(matmul(p + k, t[k].type, d, ff, n, t[k].bytes))
        ops.append(elementwise(f"{p}glu", n * ff * ACT_BYTES * 3))
        ops.append(matmul(p + "ffn_down", t["ffn_down"].type, ff, d, n, t["ffn_down"].bytes))
        ops.append(elementwise(f"{p}residual", n * d * ACT_BYTES * 3))
    out = m.tensor("output.weight") or m.tensor("token_embd.weight")
    ops.append(elementwise("output_norm", n_outputs * d * ACT_BYTES * 2 + d * 4))
    ops.append(matmul("output", out.type, d, m.vocab, n_outputs, out.bytes))
    return ops


def step(machine: Machine, gid: str, m: Model, n: int, depth: int, n_outputs: int = 1) -> StepResult:
    return execute(machine, gid, forward(m, n, depth, n_outputs), RUNTIME, graphs=(n <= MMVQ_MAX_BATCH))


def llama_bench(machine: Machine, gid: str, m: Model, test: str, n_ubatch: int = 512) -> dict:
    """llama-bench's tests: ppN = N prompt tokens in ubatches of n_ubatch (one output), tgN = N tokens one at a time
    after an empty context. Returns tokens/s and where the time went."""
    kind, n = test[:2], int(test[2:])
    steps = []
    if kind == "pp":
        done = 0
        while done < n:
            b = min(n_ubatch, n - done)
            steps.append(step(machine, gid, m, b, done)); done += b
    elif kind == "tg":
        steps = [step(machine, gid, m, 1, k) for k in range(n)]
    else:
        raise ValueError(test)
    total = sum(s.seconds for s in steps)
    share: dict[str, float] = {}
    for s in steps:
        for k, v in s.share().items():
            share[k] = share.get(k, 0.0) + v * s.seconds / total
    first = steps[0]
    return dict(test=test, tokens_per_s=n / total, seconds=total, share=share, load_class=first.load_class,
                clock_mhz=first.clock_hz / 1e6, vram_gbs=first.vram_bps / 1e9, basis=first.basis)
