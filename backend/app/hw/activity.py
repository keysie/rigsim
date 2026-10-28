"""The one way an engine asks the hardware model anything: activities on the machine's graph.

  Transfer  bytes from a storage to a compute (its units read them) or to another storage — a flow through the
            graph's links, routers and storages, sharing them max-min fairly with every other flow that runs
  Work      operations of one class on a compute node — its units at its clock, at the efficiency the engine's
            kernel reaches (the engine's software fact; the peak is the node's)
  Fixed     a fixed time spent on a node before the step's activities (a kernel launch, a dispatch, a sync)

A Step is a set of activities that run at the same time (plus its fixed costs, paid first); an engine describes its
work as a sequence of steps. solve_step runs them on the graph: rates are solved jointly, and whenever an activity
finishes the others are re-solved with what it frees (progressive filling). Nothing in here knows a GPU, a CPU or an
engine: what a part can do is only ever its model in the graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .machine import Machine, Transfer as Flow, solve_transfers
from .primitives import Compute

EPS = 1e-15


@dataclass
class Transfer:
    id: str
    src: str
    dst: str
    bytes: float
    units: int | None = None          # receiving units of a compute destination (None: all)


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


def _amount(a) -> float:
    return a.bytes if isinstance(a, Transfer) else a.ops


def solve_step(m: Machine, step: Step, cache: dict | None = None) -> StepTime:
    cache = {} if cache is None else cache
    fixed = sum(f.seconds for f in step.fixed)
    left = {a.id: _amount(a) for a in step.concurrent}
    acts = {a.id: a for a in step.concurrent if left[a.id] > 0}
    finish = {a.id: 0.0 for a in step.concurrent if left[a.id] <= 0}
    alone, neck = {}, {}
    for a in acts.values():
        r, b = _rates(m, [a], cache)[a.id]
        alone[a.id] = _amount(a) / r if r > 0 else float("inf")
        neck[a.id] = b
    t = 0.0
    while acts:
        rates = _rates(m, list(acts.values()), cache)
        dt = min((left[i] / rates[i][0] if rates[i][0] > 0 else float("inf")) for i in acts)
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
