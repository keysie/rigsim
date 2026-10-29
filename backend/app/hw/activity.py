"""The one way an engine asks the hardware model anything: activities on the machine's graph.

  Transfer  bytes from a storage to a compute (its units read them), from a compute to a storage (they write) or
            between storages — a flow through the graph's links, routers and storages, sharing them max-min fairly
            with every other flow that runs. Every transfer first waits its path's latency — the storage's access
            time, links and routers — which is the hardware's, never the engine's choice. An engine whose accesses
            depend on each other (a pointer chase) expresses that as steps one after another.
            The engine states one more thing: how its operations access the storage — 'sequential' (one stream:
            the storage positions once) or 'random' (requests of request_bytes each, every one positioned anew).
            Reading and writing speeds, latencies, seek and the read/write switch are the storage's (primitives).
  Work      operations of one class on a compute node — its units at its clock, at the efficiency the engine's
            kernel reaches (the engine's software fact; the peak is the node's)
  Fixed     a fixed time spent on a node before the step's activities (a kernel launch, a dispatch, a sync)

A Step is a set of activities that run at the same time (plus its fixed costs, paid first); an engine describes its
work as a sequence of steps, one after another. What runs in parallel belongs in one step — how much an engine puts
into a step is its model of the software, not a property of this interface. solve_step runs them on the graph: rates are solved jointly, and whenever an activity
finishes the others are re-solved with what it frees (progressive filling). Nothing in here knows a GPU, a CPU or an
engine: what a part can do is only ever its model in the graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .machine import Machine, Transfer as Flow, solve_transfers
from .primitives import Compute, Storage

EPS = 1e-15


@dataclass
class Transfer:
    id: str
    src: str
    dst: str
    bytes: float
    units: int | None = None          # units of the compute at either end (None: all)
    pattern: str = "sequential"       # 'sequential' | 'random' — the one access property an engine states
    request_bytes: float | None = None  # size of one operation's request (random: each positions anew)


@dataclass
class Work:
    id: str
    compute: str
    op_class: str
    ops: float
    efficiency: float = 1.0           # what the engine's kernel reaches of the node's peak for this class
    units: int | None = None
    basis: str = ""


@dataclass
class Fixed:
    id: str
    node: str
    seconds: float
    basis: str = ""


@dataclass
class Step:
    label: str
    concurrent: list = field(default_factory=list)      # Transfer | Work
    fixed: list = field(default_factory=list)           # Fixed


@dataclass
class StepTime:
    label: str
    seconds: float
    fixed_s: float
    finish_s: dict                    # activity id -> when it finished (after the fixed costs)
    alone_s: dict                     # activity id -> its time had it run alone (what bounds the step)
    bottleneck: dict                  # activity id -> what limited it (last solve it took part in)


def _work_rates(m: Machine, works: list[Work]) -> dict[str, tuple[float, str]]:
    """Works on the same compute node share its units equally."""
    by_node: dict[str, list[Work]] = {}
    for w in works:
        by_node.setdefault(w.compute, []).append(w)
    out = {}
    for node, ws in by_node.items():
        c = m.nodes[node]
        if not isinstance(c, Compute):
            raise TypeError(f"work on {node}: not a compute node")
        for w in ws:
            peak = c.rate(w.op_class, w.units)
            out[w.id] = (peak * w.efficiency / len(ws), f"compute:{node}")
    return out


def _rates(m: Machine, acts: list, cache: dict) -> dict[str, tuple[float, str]]:
    ts = [a for a in acts if isinstance(a, Transfer)]
    ws = [a for a in acts if isinstance(a, Work)]
    out = _work_rates(m, ws)
    if ts:
        key = tuple(sorted((t.src, t.dst, t.units) for t in ts))
        if len({(t.src, t.dst, t.units) for t in ts}) == len(ts) and key in cache:
            solved = {t.id: cache[key][(t.src, t.dst, t.units)] for t in ts}
        else:
            res = solve_transfers(m, [Flow(t.id, t.src, t.dst, t.units) for t in ts])
            solved = {t.id: (res[t.id]["rate_bps"], res[t.id]["bottleneck"]) for t in ts}
            if len({(t.src, t.dst, t.units) for t in ts}) == len(ts):
                cache[key] = {(t.src, t.dst, t.units): solved[t.id] for t in ts}
        out.update(solved)
    return out


def _ends(m: Machine, a: Transfer) -> list[tuple[str, str]]:
    """(storage member, direction) a transfer touches: its source is read, its destination written."""
    out = []
    for end, direction in ((a.src, "read"), (a.dst, "write")):
        for mem in m.interleaved.get(end, [end]):
            if isinstance(m.nodes.get(mem), Storage):
                out.append((mem, direction))
    return out


def _storage_overheads(m: Machine, acts: list) -> dict[str, float]:
    """Device time a step costs each storage beyond moving its bytes, with the best batching inside the step: every
    read, then every write — one switch if both happen — and positioning: once per sequential transfer, once per
    request of a random one (spread over an interleaved set's members). The solver never batches across steps."""
    dirs: dict[str, set] = {}
    seeks: dict[str, float] = {}
    for a in acts:
        if not isinstance(a, Transfer):
            continue
        ends = _ends(m, a)
        for mem, direction in ends:
            dirs.setdefault(mem, set()).add(direction)
            per_member = len([e for e in ends if e[1] == direction])
            if a.pattern == "random":
                n = max(1.0, a.bytes / a.request_bytes) if a.request_bytes else 1.0
                seeks[mem] = seeks.get(mem, 0.0) + n / max(per_member, 1)
            else:
                seeks[mem] = seeks.get(mem, 0.0) + 1.0
    out = {}
    for mem, ds in dirs.items():
        st = m.nodes[mem]
        o = (st.turnaround_s or 0.0) * (1 if len(ds) == 2 else 0) + (st.seek_s or 0.0) * seeks.get(mem, 0.0)
        if o:
            out[mem] = o
    return out


def _latency(m: Machine, a, cache: dict) -> float:
    """A transfer's path latency (storage access + links + routers), from the graph. Where a part's latency is not
    identified (e.g. a card's VRAM access time, not on spec sheets) it adds nothing — non-binding, like any
    unidentified capacity."""
    if not isinstance(a, Transfer):
        return 0.0
    key = ("latency", a.src, a.dst)
    if key not in cache:
        writes = isinstance(m.nodes.get(a.src), Compute)
        comp, store = (a.src, a.dst) if writes else (a.dst, a.src)
        try:
            # positioning is charged as the storage's device time for the step (_storage_overheads), not twice
            cache[key] = (m.idle_latency_s(comp, store, "write" if writes else "read", seek=False)
                          if isinstance(m.nodes.get(comp), Compute) else 0.0)
        except (ValueError, KeyError):
            cache[key] = 0.0
    return cache[key]


def _amount(a) -> float:
    return a.bytes if isinstance(a, Transfer) else a.ops


def solve_step(m: Machine, step: Step, cache: dict | None = None) -> StepTime:
    cache = {} if cache is None else cache
    fixed = sum(f.seconds for f in step.fixed)
    left = {a.id: _amount(a) for a in step.concurrent}
    acts = {a.id: a for a in step.concurrent if left[a.id] > 0}
    finish = {a.id: 0.0 for a in step.concurrent if left[a.id] <= 0}
    alone, neck = {}, {}
    # a transfer starts after its path's latency and after the device time its storages spend on positioning and
    # switching direction for this step (the batching inside the step)
    over = _storage_overheads(m, list(acts.values()))
    start = {a.id: _latency(m, a, cache) + max([over.get(mem, 0.0) for mem, _ in _ends(m, a)] or [0.0])
             if isinstance(a, Transfer) else 0.0 for a in acts.values()}
    for a in acts.values():
        r, b = _rates(m, [a], cache)[a.id]
        alone[a.id] = start[a.id] + (_amount(a) / r if r > 0 else float("inf"))
        neck[a.id] = b
    t = 0.0
    waiting = dict(acts)
    acts = {}
    while acts or waiting:
        for i in [i for i in waiting if start[i] <= t + 1e-18]:
            acts[i] = waiting.pop(i)
        rates = _rates(m, list(acts.values()), cache) if acts else {}
        dt = min([(left[i] / rates[i][0] if rates[i][0] > 0 else float("inf")) for i in acts]
                 + [start[i] - t for i in waiting])
        if dt == float("inf"):
            raise ValueError(f"step {step.label}: an activity cannot progress (zero rate): "
                             f"{[i for i in acts if rates[i][0] <= 0]}")
        t += dt
        for i in list(acts):
            left[i] -= rates[i][0] * dt
            neck[i] = rates[i][1]
            if left[i] <= EPS * max(1.0, _amount(acts[i])) * 1e6:
                finish[i] = t
                del acts[i]
    for a in step.concurrent:
        alone.setdefault(a.id, 0.0)
    return StepTime(step.label, fixed + t, fixed, finish, alone, neck)
