"""The hardware's answer to an engine step (proposal §6): operations in, duration out.

An engine states what one step does on one device as a list of operations (class, amount, bytes from device
memory, kernel count, and its own kernel factors against the reference). This module alone turns them into time:
per operation the larger of compute time and memory time, plus a fixed cost per kernel; operations run one after
another (a single stream, as in one forward pass). It also picks the state's load class from the mix (one pass, no
feedback) and reports which resource set the pace, with the provenance of every rate used."""
from __future__ import annotations

from dataclasses import dataclass, field

from .gpu import set_load_class
from .loadclass import classify
from .machine import Machine
from .rates import achievable, launch_s


@dataclass
class Op:
    name: str
    kind: str                         # matmul, attention, elementwise, ... (reporting only)
    op_class: str | None = None       # compute rate class (hw rates); None: the compute is not modelled
    ops: float = 0.0
    bytes: float = 0.0                # moved to or from device memory
    kernels: int = 1
    compute_factor: float = 1.0       # the engine's kernel vs the reference library for op_class
    memory_factor: float = 1.0        # the engine's kernel vs a plain streaming read
    factor_basis: str = "unmeasured"


@dataclass
class OpTime:
    op: Op
    compute_s: float
    memory_s: float
    launch_s: float

    @property
    def seconds(self) -> float:
        return max(self.compute_s, self.memory_s) + self.launch_s

    @property
    def bound(self) -> str:
        work = max(self.compute_s, self.memory_s)
        if self.launch_s >= work:
            return "launch"
        return "compute" if self.compute_s >= self.memory_s else "memory"


@dataclass
class StepResult:
    seconds: float
    ops: list[OpTime]
    load_class: str
    clock_hz: float
    vram_bps: float
    basis: dict = field(default_factory=dict)

    def share(self) -> dict[str, float]:
        """Share of the step's time per bound (compute / memory / launch)."""
        out: dict[str, float] = {}
        for o in self.ops:
            out[o.bound] = out.get(o.bound, 0.0) + o.seconds
        return {k: v / self.seconds for k, v in out.items()} if self.seconds else out


def execute(m: Machine, gid: str, ops: list[Op], runtime: str, graphs: bool = True) -> StepResult:
    dev = m.devices[gid]
    items = [(o.op_class or "fp32", o.ops if o.op_class else 0.0, o.bytes) for o in ops]
    cls, _ = classify(dev.model, runtime, items)
    st = set_load_class(m, gid, cls)
    clock, bw = st["sm_clock_hz"], st["vram_bps"]
    k_s, k_basis = launch_s(dev.model, runtime, graphs)
    basis = dict(clock=st["sm_clock_basis"], vram=st["vram_basis"], launch=k_basis, rates={})
    times = []
    for o in ops:
        tc = 0.0
        if o.op_class and o.ops:
            rate, how = achievable(dev.model, runtime, o.op_class, clock)
            basis["rates"][o.op_class] = how
            tc = o.ops / (rate * o.compute_factor)
        tm = o.bytes / (bw * o.memory_factor) if o.bytes else 0.0
        times.append(OpTime(o, tc, tm, o.kernels * k_s))
    return StepResult(sum(t.seconds for t in times), times, cls, clock, bw, basis)
