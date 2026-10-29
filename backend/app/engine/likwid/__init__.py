"""likwid-bench's streaming kernels (the stick's RAM ladder) as an engine: steps of activities on the machine's graph.
It knows the graph's core groups and memory by name, nothing about the hardware.

  load_sse, load_avx   N threads read a buffer far larger than the caches (-w S0:<size>:N): reads, memory -> cores
  store_mem_avx        N threads write it with non-temporal stores, straight to memory: writes, cores -> memory
  store_avx            ordinary stores: the cache reads each line before it is written (write-allocate) — that is
                       the caches' behaviour and needs a cache block in the graph: not modelled yet

Thread placement as likwid's S0 domain: one thread per physical core in the order the cores are listed (P-cores
before E-cores on hybrid Intel parts, the first core die before the second on Ryzen); threads beyond the physical
cores are SMT siblings and add no memory requests of their own. One step = what the active cores have in flight at
once (each core's miss queue, a property of the cores in the graph); steps follow one another. Where a queue is not
identified: the whole stream as one step (optimistic, stated). likwid counts the bytes the program moves.
"""
from __future__ import annotations

from ...hw.activity import Step, Transfer, solve_step
from ...hw.machine import Machine

LINE = 64
KERNELS = {"load_sse": "read", "load_avx": "read", "load_avx512": "read", "store_mem_avx": "write",
           "store_mem_sse": "write", "store_mem": "write"}
NOT_YET = {"store_avx": "ordinary stores read each line first (write-allocate): needs a cache block in the graph",
           "store_sse": "ordinary stores read each line first (write-allocate): needs a cache block in the graph"}


def placement(m: Machine, core_groups: list[str], threads: int) -> dict[str, int]:
    """Threads per core group, one per physical core in list order; SMT siblings beyond that add nothing."""
    out, left = {}, threads
    for g in core_groups:
        n = min(left, m.nodes[g].count)
        if n:
            out[g] = n
        left -= n
    return out


def bandwidth_bps(m: Machine, core_groups: list[str], kernel: str, threads: int, memory: str = "ram",
                  total_bytes: float = 4e9) -> tuple[float, str]:
    """Program bytes per second for a kernel at a thread count, and how the steps were cut."""
    direction = KERNELS[kernel]
    active = placement(m, core_groups, threads)
    queues = {g: m.nodes[g].outstanding for g in active}
    known = all(q is not None for q in queues.values())
    acts, step_bytes = [], 0.0
    for g, n in active.items():
        b = n * queues[g] * LINE if known else total_bytes * n / sum(active.values())
        step_bytes += b
        acts.append(Transfer(g, memory, g, b, units=n) if direction == "read" else Transfer(g, g, memory, b, units=n))
    t = solve_step(m, Step(f"{kernel} x{threads}", acts)).seconds
    how = (f"{threads} threads on {', '.join(f'{n} {g}' for g, n in active.items())}; steps of "
           f"{step_bytes / 1024:.0f} KiB in flight" if known else
           f"{threads} threads; one step for the whole stream (miss queues not identified)")
    return step_bytes / t, how
