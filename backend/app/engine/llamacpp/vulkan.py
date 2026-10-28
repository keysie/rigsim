"""llama.cpp, Vulkan backend (ggml-vulkan): the same graph as CUDA (cuda.forward), its own kernels.

Kernel paths (ggml-vulkan, mid-2025 builds): a quantised weight times few columns goes through the mul_mat_vec
shaders, which dequantise the weight while streaming it once (memory-bound; no separate activation quantisation
kernel); more columns go through the matrix shaders, which dequantise into fp16 tiles and multiply on the matrix
units through cooperative matrices (fp16 in, fp32 accumulation) where the driver offers them. Attention is
flash attention, vector-wise for one token, on cooperative matrices for a prompt.

As for CUDA, the kernel factors against their references (a stream read; the fp16 tensor rate) are 1.0 and
unmeasured: per-kernel timings (test-backend-ops) set them, llama-bench validates. The per-dispatch cost is the
hardware layer's (measured by vkprobe's launch mode where available, else its placeholder)."""
from __future__ import annotations

from ...hw.execute import StepResult, execute
from ...hw.machine import Machine
from ..model import Model
from . import cuda

RUNTIME = "vulkan"
MMV_MAX_COLS = 8                        # ggml-vulkan: columns up to which the mul_mat_vec shaders are used
FACTORS = {"mmv": dict(memory=cuda.Factor()), "mm_coopmat": dict(compute=cuda.Factor(), memory=cuda.Factor()),
           "fa_vec": dict(memory=cuda.Factor()), "fa_coopmat": dict(compute=cuda.Factor(), memory=cuda.Factor())}


def _op(name, kind, path, **kw):
    f = FACTORS[path]
    cf, mf = f.get("compute", cuda.Factor()), f.get("memory", cuda.Factor())
    from ...hw.execute import Op
    return Op(name, kind, kw.get("op_class"), kw.get("ops", 0.0), kw.get("bytes_", 0.0), kw.get("kernels", 1),
              cf.value, mf.value, factor_basis=f"vulkan {path}: compute {cf.basis}; memory {mf.basis}")


# Without matrix units (Intel Gen9 / Xe-LP iGPUs) the matrix and attention shaders run on the vector units in fp16.
VECTOR = {"mm_coopmat": ("mm_vector", "fp16"), "fa_coopmat": ("fa_vector", "fp16")}
FACTORS.update({"mm_vector": dict(compute=cuda.Factor(), memory=cuda.Factor()),
                "fa_vector": dict(compute=cuda.Factor(), memory=cuda.Factor())})


def matmul(name: str, w_type: str, cols: int, rows: int, n: int, w_bytes: float, matrix_units: bool = True):
    ops = 2.0 * n * cols * rows
    b = w_bytes + n * cols * cuda.ACT_BYTES + n * rows * cuda.ACT_BYTES
    if n <= MMV_MAX_COLS:
        return _op(name, "matmul", "mmv", ops=ops, bytes_=b, kernels=1)
    path, cls = ("mm_coopmat", "fp16_tensor_fp32acc") if matrix_units else VECTOR["mm_coopmat"]
    return _op(name, "matmul", path, op_class=cls, ops=ops, bytes_=b, kernels=1)


def attention(m: Model, n: int, depth: int, matrix_units: bool = True):
    a = cuda.attention(m, n, depth)
    if n == 1:
        return _op("attention", "attention", "fa_vec", ops=a.ops, bytes_=a.bytes)
    path, cls = ("fa_coopmat", "fp16_tensor_fp32acc") if matrix_units else VECTOR["fa_coopmat"]
    return _op("attention", "attention", path, op_class=cls, ops=a.ops, bytes_=a.bytes)


def step(machine: Machine, gid: str, m: Model, n: int, depth: int, n_outputs: int = 1) -> StepResult:
    mu = machine.devices[gid].spec.get("matrix_units", True)
    mm = lambda *a: matmul(*a, matrix_units=mu)
    at = lambda *a: attention(*a, matrix_units=mu)
    return execute(machine, gid, cuda.forward(m, n, depth, n_outputs, matmul=mm, attention=at), RUNTIME, graphs=False)


def llama_bench(machine: Machine, gid: str, m: Model, test: str, n_ubatch: int = 512) -> dict:
    """llama-bench ppN / tgN on the Vulkan backend (same test structure as cuda.llama_bench)."""
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
    return dict(test=test, tokens_per_s=n / total, seconds=total, share=share)
