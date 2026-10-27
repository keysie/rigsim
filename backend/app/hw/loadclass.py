"""Load class of a state on a device, from its operation mix (one pass, no feedback): each operation's time at the
nominal clock is the larger of compute and memory time (roofline); the state is "compute" if compute-bound
operations take at least half of that time, else "memory". Nominal clock and bandwidth: the spec values."""
from __future__ import annotations

from .gpus import GPUS
from .rates import achievable


def classify(model: str, runtime: str, items: list[tuple[str, float, float]]) -> tuple[str, float]:
    """items: (op_class, ops, bytes from VRAM). Returns (load class, compute-bound share of the time)."""
    spec = GPUS[model]
    clock = spec["boost_mhz"] * 1e6
    bw = spec["vram_gbs"] * 1e9
    t_compute = t_memory = 0.0
    for op_class, ops, bytes_ in items:
        rate, _ = achievable(model, runtime, op_class, clock)
        tc, tm = ops / rate, bytes_ / bw
        if tc >= tm:
            t_compute += tc
        else:
            t_memory += tm
    total = t_compute + t_memory
    share = t_compute / total if total else 0.0
    return ("compute" if share >= 0.5 else "memory"), share
