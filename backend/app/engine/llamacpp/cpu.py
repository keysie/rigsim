"""llama.cpp v0.7.0 (b11669, 620d222) on the CPU, as the stick builds it: GGML_NATIVE=OFF — AVX2, FMA, F16C, BMI2;
no AVX-512, no AVX-VNNI — with weight repacking (GGML_CPU_REPACK) and OpenMP. The same graph as the GPU backends
(cuda.forward); this module brings the CPU's kernels, and every kernel is work on the core budgets
(docs/overhaul/cpu-compute.md) plus transfers through the graph.

What the code does (source read at v0.7.0; instruction mixes counted from the hot loops of the objects compiled with
the stick's flags — g++/gcc 13, -O3 -mavx2 -mfma -mf16c -mbmi2 — by tools/kernel_mix.py, classes as cpu_uarch):

  threads      llama-bench's default common_cpu_get_num_math(): P-cores on hybrid Intel parts, else physical
               cores; the run records the number (n_threads). One thread per core, filling the core groups in order
               (P-cores first; the first core die first); ggml gives every thread an equal share of each kernel.
  Q4_0 weights with a row count divisible by 8 are repacked at load time to q4_0_8x8 (repack.cpp, AVX2 branch).
  matmul       1. the activations are quantised to Q8_0, four rows interleaved (ggml_quantize_mat_q8_0_4x8;
                  one loop iteration: 4 rows x 32 values)
               2. four columns at a time: gemm_q4_b32_8x8_q8_0_lut_avx (one iteration: 4 columns x 8 rows x 32);
                  the remaining columns: gemv_q4_b32_8x8_q8_0_lut_avx (one iteration: 8 rows x 32)
  Q6_K weights (the output layer of these models) are not repacked on x86: activations quantised to Q8_K, then
               ggml_vec_dot_q6_K_q8_K per row and column (one iteration: 128 values); llamafile's sgemm has no Q6_K
  attention    flash attention on f16 K/V (flash_attn auto resolves to on for the CPU): per query head and cached
               position a ggml_vec_dot_f16 over the head (one iteration: 32 values) and a ggml_vec_mad_f16 into the
               accumulator (vec.h: per 32 values 4 loads, 4 conversions in, 4 FMAs, 4 conversions out, 4 stores —
               read from the source, not compiled: it is inlined into ops.cpp); the per-position softmax update is
               scalar code, not counted
  elementwise  norms, rope, residuals, SwiGLU, KV stores: few instructions per value against memory traffic; taken
               as 2 loads, 1 FMA-class, 1 ALU, 1 store per 8 values (an estimate, stated)
  barriers     every graph node ends in a barrier across the threads; its cost is not identified (core-to-core
               latency) and not counted

Memory: weights stream from the system memory (no cache hits: far larger than any cache); activations are served by
the caches (hits = 1, they are small and just written); the KV cache hits when it fits the last-level cache,
else not — the engine's statement, from the graph's L3 capacity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...hw.activity import Step, Transfer, Work, solve_step
from ...hw.machine import Machine
from ...hw.primitives import Cache
from ..model import Model
from . import cuda

VERSION = "v0.7.0 (b11669)"
# hot-loop instruction mixes per iteration (tools/kernel_mix.py on the objects of v0.7.0)
MIX = {
    "gemv_q4_0_8x8": dict(load=16, vec_alu=44, shuffle=29, vec_mul=16, fma=2, store=2, scalar=3),
    "gemm_q4_0_8x8": dict(load=84, vec_alu=102, shuffle=63, vec_mul=64, fma=8, store=48, scalar=4),
    "quantize_q8_0_4x8": dict(load=35, vec_alu=75, shuffle=35, fma=48, store=10, scalar=16),
    "vec_dot_q6_K_q8_K": dict(load=36, vec_alu=46, shuffle=18, vec_mul=17, fma=2, store=1, scalar=5),
    "vec_dot_f16": dict(load=8, shuffle=8, fma=4, scalar=4),
    "vec_mad_f16": dict(load=4, shuffle=8, fma=4, store=4),
    "elementwise": dict(load=2, fma=1, vec_alu=1, store=1),
}
PER_ITER = {"gemv_q4_0_8x8": 256, "gemm_q4_0_8x8": 1024, "quantize_q8_0_4x8": 128, "vec_dot_q6_K_q8_K": 128,
            "vec_dot_f16": 32, "vec_mad_f16": 32, "elementwise": 8}
SCALAR_ISSUE = "scalar"            # counts against the issue width only


@dataclass
class CpuOp:
    name: str
    works: list = field(default_factory=list)        # (kernel, iterations) run one after another
    ram_bytes: float = 0.0                           # weights from the system memory
    cached_bytes: float = 0.0                        # activations, served by the caches
    kv_bytes: float = 0.0                            # the KV cache


def _iters(kernel: str, values: float) -> tuple[str, float]:
    return kernel, values / PER_ITER[kernel]


def matmul(name: str, w_type: str, cols: int, rows: int, n: int, w_bytes: float) -> CpuOp:
    act = n * cols * 4 + n * rows * 4
    if w_type == "q4_0" and rows % 8 == 0:
        n4, rest = (n // 4) * 4, n % 4
        works = [_iters("quantize_q8_0_4x8", n * cols)]
        if n4:
            works.append(("gemm_q4_0_8x8", (n4 / 4) * (rows / 8) * (cols / 32)))
        if rest:
            works.append(("gemv_q4_0_8x8", rest * (rows / 8) * (cols / 32)))
        return CpuOp(name, works, ram_bytes=w_bytes, cached_bytes=act)
    if w_type == "q6_k":
        works = [_iters("quantize_q8_0_4x8", n * cols), ("vec_dot_q6_K_q8_K", n * rows * cols / 128)]
        return CpuOp(name, works, ram_bytes=w_bytes, cached_bytes=act)
    raise NotImplementedError(f"{name}: {w_type} weights on the CPU backend are not modelled yet")


def attention(m: Model, n: int, depth: int) -> CpuOp:
    ctx = depth + (n + 1) / 2                        # mean cached positions a query sees (causal)
    pairs = m.heads * n * ctx * m.head_dim
    kv = 2 * m.kv_dim * (depth + n) * cuda.KV_BYTES
    return CpuOp("attention", [_iters("vec_dot_f16", pairs), _iters("vec_mad_f16", pairs)],
                 cached_bytes=n * (m.heads * m.head_dim) * 4 * 2, kv_bytes=kv)


def elementwise(name: str, bytes_: float, kernels: int = 1) -> CpuOp:
    return CpuOp(name, [_iters("elementwise", bytes_ / 4 / 2)], cached_bytes=bytes_)


def _ops(m: Model, n: int, depth: int, n_outputs: int) -> list[CpuOp]:
    return cuda.forward(m, n, depth, n_outputs, matmul=matmul, attention=attention, elementwise=elementwise)


def placement(machine: Machine, groups: list[str], threads: int) -> dict[str, int]:
    out, left = {}, threads
    for g in groups:
        n = min(left, machine.nodes[g].count)
        if n:
            out[g] = n
        left -= n
    return out


def _steps(machine: Machine, place: dict[str, int], op: CpuOp, kv_hits: float, memory: str) -> list[Step]:
    """The op's kernels one after another (a barrier between them), each thread an equal share; the last kernel
    carries the op's memory traffic."""
    total = sum(place.values())
    steps = []
    for k, (kernel, iters) in enumerate(op.works):
        acts = []
        for g, n in place.items():
            f = n / total
            acts.append(Work(f"{op.name}.{kernel}.{g}", g, "", iters * f, mix=MIX[kernel], units=n))
            if k == len(op.works) - 1:
                if op.ram_bytes:
                    acts.append(Transfer(f"{op.name}.{g}.w", memory, g, op.ram_bytes * f, units=n))
                if op.cached_bytes:
                    acts.append(Transfer(f"{op.name}.{g}.a", memory, g, op.cached_bytes * f, units=n, hits=1.0))
                if op.kv_bytes:
                    acts.append(Transfer(f"{op.name}.{g}.kv", memory, g, op.kv_bytes * f, units=n, hits=kv_hits))
        steps.append(Step(f"{op.name} {kernel}", acts))
    return steps


def llc_bytes(machine: Machine) -> float:
    return max([n.capacity_b for n in machine.nodes.values() if isinstance(n, Cache)] or [0.0])


def llama_bench(machine: Machine, groups: list[str], m: Model, test: str, threads: int, memory: str = "ram",
                n_ubatch: int = 512) -> dict:
    """llama-bench ppN / tgN on the CPU backend with the run's thread count."""
    place = placement(machine, groups, threads)
    threads = sum(place.values())
    llc = llc_bytes(machine)
    kind, n = test[:2], int(test[2:])
    cache: dict = {}
    total, share = 0.0, {}

    def run(ntok: int, depth: int) -> float:
        kv_total = 2 * m.kv_dim * (depth + ntok) * cuda.KV_BYTES * m.layers
        kv_hits = 1.0 if kv_total <= llc else 0.0
        t = 0.0
        for op in _ops(m, ntok, depth, 1):
            for st in _steps(machine, place, op, kv_hits, memory):
                r = solve_step(machine, st, cache)
                t += r.seconds
                for a, sec in r.alone_s.items():
                    key = "memory" if a.endswith((".w", ".a", ".kv")) else "compute"
                    share[key] = share.get(key, 0.0) + sec
        return t

    if kind == "pp":
        done = 0
        while done < n:
            b = min(n_ubatch, n - done)
            total += run(b, done); done += b
    elif kind == "tg":
        # tokens differ only in the cached depth; every token is simulated
        for k in range(n):
            total += run(1, k)
    else:
        raise ValueError(test)
    s = sum(share.values()) or 1.0
    return dict(test=test, tokens_per_s=n / total, seconds=total, threads=threads, placement=place,
                share={k: v / s for k, v in share.items()})
