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
from .activity import Fixed, Step, Transfer, Work, solve_step
from .machine import Machine, Transfer as Flow, solve_transfers
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
    """An engine's ops on one device, each as a step of activities on the machine's graph (activity.py): its kernel
    launch (fixed), its operations on the device's units (work) and its bytes from the device's memory (a transfer
    through the graph — the card's VRAM, or the system's channels for an iGPU), running at the same time. Steps run
    one after another. The device's state (load class: clock, VRAM per the curves or the spec law) is set first."""
    dev = m.devices[gid]
    items = [(o.op_class or "fp32", o.ops if o.op_class else 0.0, o.bytes) for o in ops]
    cls, _ = classify(dev.model, runtime, items)
    st = set_load_class(m, gid, cls)
    clock = st["sm_clock_hz"]
    sm = m.nodes[f"{gid}.sm"]
    k_s, k_basis = launch_s(dev.model, runtime, graphs)
    flow = solve_transfers(m, [Flow("device", dev.memory, f"{gid}.sm")])["device"]
    basis = dict(clock=st["sm_clock_basis"], vram=f"{st['vram_basis']}; graph flow {dev.memory} -> {gid}.sm, "
                 f"bound by {flow['bottleneck']}", launch=k_basis, rates={})
    cache: dict = {}
    times = []
    for i, o in enumerate(ops):
        acts = []
        if o.op_class and o.ops:
            rate, how = achievable(dev.model, runtime, o.op_class, clock)
            basis["rates"][o.op_class] = how
            peak = sm.rate(o.op_class)
            acts.append(Work(f"w{i}", f"{gid}.sm", o.op_class, o.ops, efficiency=rate / peak * o.compute_factor))
        if o.bytes:
            acts.append(Transfer(f"t{i}", dev.memory, f"{gid}.sm", o.bytes / o.memory_factor))
        step = Step(o.name, acts, [Fixed(f"k{i}", gid, o.kernels * k_s, k_basis)])
        r = solve_step(m, step, cache)
        times.append(OpTime(o, r.alone_s.get(f"w{i}", 0.0), r.alone_s.get(f"t{i}", 0.0), r.fixed_s))
    return StepResult(sum(t.seconds for t in times), times, cls, clock, flow["rate_bps"], basis)
