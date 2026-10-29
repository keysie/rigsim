"""The one way an engine asks the hardware model anything: activities on the machine's graph.

  Transfer  bytes from a storage to a compute (its units read them), from a compute to a storage (they write) or
            between storages — a flow through the graph's links, routers and storages, sharing them max-min fairly
            with every other flow that runs. Every transfer first waits its path's latency — the storage's access
            time, links and routers — which is the hardware's, never the engine's choice. An engine whose accesses
            depend on each other (a pointer chase) expresses that as steps one after another.
            The engine states one more thing: how its operations access the storage — 'sequential' (one stream:
            the storage positions once) or 'random' (requests of request_bytes each, every one positioned anew).
            Reading and writing speeds, latencies, seek and the read/write switch are the storage's (primitives).
            Two more statements about the software: `hits`, the share served by the cache nearest the compute on
            the path (an assumption, like a kernel's efficiency: hits are never computed), and `non_temporal`,
            writes that bypass the caches (streaming stores). What a miss or a write costs is the caches' policy.
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
from .primitives import Cache, Compute, Storage

EPS = 1e-15


@dataclass
class Transfer:
    id: str
    src: str
    dst: str
    bytes: float
    units: int | None = None          # units of the compute at either end (None: all)
    pattern: str = "sequential"       # 'sequential' | 'random' — how the operations access the storage
    hits: float = 0.0                 # share served by the nearest cache on the path (the engine's assumption)
    non_temporal: bool = False        # writes that bypass the caches (streaming stores)
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


def _nearest_cache(m: Machine, compute: str, storage: str) -> str | None:
    """The cache closest to the compute on the path to a storage (or to the first member of an interleaved set)."""
    target = m.interleaved.get(storage, [storage])[0]
    try:
        hops = m.path(compute, target)
    except ValueError:
        return None
    return next((n for _, n in hops if isinstance(m.nodes.get(n), Cache)), None)


def _expand(m: Machine, a) -> list:
    """A transfer as the flows the hardware makes of it, given the caches on its path and the engine's hit share:
    a read's hits come from the cache, its misses from the storage; a write's hits go to the cache; a write miss goes
    to the storage directly when non-temporal or when the cache does not allocate, else the cache first reads the line
    from the storage (write-allocate) and — write-back or write-through — the line is written to the storage."""
    if not isinstance(a, Transfer):
        return [a]
    writes = isinstance(m.nodes.get(a.src), Compute)
    comp, store = (a.src, a.dst) if writes else (a.dst, a.src)
    c = _nearest_cache(m, comp, store) if isinstance(m.nodes.get(comp), Compute) else None
    if c is None:
        return [a]
    h = min(max(a.hits, 0.0), 1.0)
    hit, miss = a.bytes * h, a.bytes * (1 - h)
    part = lambda suffix, src, dst, b: Transfer(f"{a.id}{suffix}", src, dst, b, a.units, a.pattern, a.request_bytes)
    out = []
    if not writes:
        if hit:
            out.append(part(".hit", c, comp, hit))
        if miss:
            out.append(part("", store, comp, miss))
        return out
    if hit:
        out.append(part(".hit", comp, c, hit))
    if miss:
        cache = m.nodes[c]
        if a.non_temporal or not cache.write_allocate:
            out.append(part("", comp, store, miss))
        else:
            out.append(part(".allocate", store, c, miss))           # the line is read before it is written
            out.append(part("", c, store, miss))                     # and reaches the storage (evicted or through)
    return out


def solve_step(m: Machine, step: Step, cache: dict | None = None) -> StepTime:
    """Solves a step; transfers are first turned into the flows the caches on their paths make of them."""
    parts, owner = [], {}
    for a in step.concurrent:
        for p in _expand(m, a):
            parts.append(p); owner[p.id] = a.id
    if all(owner[p.id] == p.id for p in parts) and len(parts) == len(step.concurrent):
        return _solve_flat(m, step, cache)
    r = _solve_flat(m, Step(step.label, parts, step.fixed), cache)
    finish, alone, neck = {}, {}, {}
    for pid, oid in owner.items():
        finish[oid] = max(finish.get(oid, 0.0), r.finish_s.get(pid, 0.0))
        neck.setdefault(oid, r.bottleneck.get(pid, ""))
    for a in step.concurrent:
        mine = [p for p in parts if owner[p.id] == a.id]
        alone[a.id] = _solve_flat(m, Step(a.id, mine), cache).seconds if mine else 0.0
    return StepTime(step.label, r.seconds, r.fixed_s, finish, alone, neck)


def _solve_flat(m: Machine, step: Step, cache: dict | None = None) -> StepTime:
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
