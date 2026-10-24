"""Plain dataclasses with light validation, so the same code runs under Pyodide in the browser
(no third-party packages) and behind the optional FastAPI dev server."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


class ValidationError(ValueError):
    pass


def _coerce(cls, data: Any):
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        raise ValidationError(f"{cls.__name__}: expected an object, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    try:
        obj = cls(**kwargs)
    except TypeError as e:
        raise ValidationError(f"{cls.__name__}: {e}") from e
    return obj


@dataclass
class CpuSlot:
    cpu_id: str
    enabled: bool = True
    ram_gb: int = 128
    ram_mts: int = 3200
    ram_channels: int = 4


@dataclass
class PcieDevice:
    id: str
    kind: str                       # gpu | ssd
    model_id: str
    cpu: int = 0
    link_gen: int = 4
    link_width: int = 16
    link_type: str = "pcie"              # pcie | tb3 | tb4 | usb4 | usb4v2 | tb5
    power_w: float | None = None         # GPU power cap in W; None = the card's default limit
    power_w: int | None = None           # GPU power cap in watts (None = the card's default limit)
    capacity_tb: float | None = None


@dataclass
class Machine:
    cpus: list[CpuSlot]
    devices: list[PcieDevice] = field(default_factory=list)
    nvlinks: list[tuple[str, str]] = field(default_factory=list)
    os: str = "linux"

    @classmethod
    def model_validate(cls, data: Any) -> "Machine":
        m = _coerce(cls, data)
        if not isinstance(m.cpus, list) or not 1 <= len(m.cpus) <= 2:
            raise ValidationError("machine needs one or two CPUs")
        m.cpus = [_coerce(CpuSlot, c) for c in m.cpus]
        m.devices = [_coerce(PcieDevice, d) for d in (m.devices or [])]
        for d in m.devices:
            if d.kind not in ("gpu", "ssd"):
                raise ValidationError(f"{d.id}: kind must be gpu or ssd")
            if d.link_type not in ("pcie", "tb3", "tb4", "usb4", "usb4v2", "tb5"):
                raise ValidationError(f"{d.id}: unknown link type '{d.link_type}'")
        m.nvlinks = [tuple(p) for p in (m.nvlinks or [])]
        if m.os not in ("linux", "windows"):
            raise ValidationError("os must be linux or windows")
        return m


@dataclass
class EngineSpec:
    id: str
    engine: str
    version: str
    model_id: str
    name: str = ""
    args: dict = field(default_factory=dict)
    gpus: list[str] = field(default_factory=list)
    cpu: int = 0
    cpu_cores: int = 4
    storage: str | None = None
    benchmarks: dict = field(default_factory=dict)

    @classmethod
    def model_validate(cls, data: Any) -> "EngineSpec":
        e = _coerce(cls, data)
        e.args = dict(e.args or {})
        e.gpus = list(e.gpus or [])
        e.benchmarks = dict(e.benchmarks or {})
        return e


@dataclass
class Roles:
    coordinator: str | None = None
    maker: str | None = None
    sub: str | None = None
    guard: str | None = None
    judge: str | None = None
    reranker: str | None = None


@dataclass
class Workload:
    """Turnstone-shaped workload; every range is sampled per scenario."""
    maker_turns: tuple = (6, 30)
    answer_median: tuple = (200, 700)
    tool_calls_mean: tuple = (0.8, 2.0)
    p_delegate: tuple = (0.0, 0.4)
    fanout: tuple = (1, 4)
    tool_result_median: tuple = (500, 3000)
    tool_exec_median_s: tuple = (0.5, 5.0)
    p_approval: tuple = (0.0, 0.5)
    sub_turns: tuple = (3, 12)
    sub_answer_median: tuple = (200, 500)
    sub_tool_calls_mean: tuple = (0.8, 1.6)
    sub_final_tokens: tuple = (300, 1500)
    task_tokens: tuple = (200, 1500)
    maker_sys: int = 11000
    sub_sys: int = 5500
    window: int = 32768
    compact_at: float = 0.8
    visible_frac: float = 0.1
    replay_thinking: bool = False
    tasks_per_ws: int = 3
    notes: list = field(default_factory=list, repr=False, compare=False)
    work_scale: float = 1.0          # lever: how much work one task is, relative to the ranges (turns, tool calls, outputs)
    reruns: float = 1.0              # lever: how often a task is run before it counts (1 = once; 1.5 = half the tasks twice)
    work_scale: float = 1.0        # multiplies turns per task: how much agent work one task takes (model-dependent)
    reruns_per_task: float = 1.0   # how often a task is redone before it counts (1 = never)

    @classmethod
    def model_validate(cls, data: Any) -> "Workload":
        w = _coerce(cls, data or {})
        for f in fields(cls):
            v = getattr(w, f.name)
            if isinstance(v, list):
                setattr(w, f.name, tuple(v))
        w.notes = sanitize_workload(w)
        return w


# per-field limits: (low, high). Ranges are (min, max) pairs; reversed pairs are swapped, missing or
# non-numeric parts fall back to the default, everything is clamped so one stray digit cannot run for hours.
_LIMITS = {
    "maker_turns": (1, 200), "answer_median": (1, 32000), "tool_calls_mean": (0.0, 20.0), "p_delegate": (0.0, 1.0),
    "fanout": (1, 16), "tool_result_median": (1, 30000), "tool_exec_median_s": (0.0, 3600.0), "p_approval": (0.0, 1.0),
    "sub_turns": (1, 200), "sub_answer_median": (1, 32000), "sub_tool_calls_mean": (0.0, 20.0),
    "sub_final_tokens": (1, 32000), "task_tokens": (1, 100000),
    "maker_sys": (0, 200000), "sub_sys": (0, 200000), "window": (2048, 2_000_000), "compact_at": (0.1, 0.99),
    "visible_frac": (0.0, 1.0), "tasks_per_ws": (1, 20), "work_scale": (0.1, 10.0), "reruns": (1.0, 5.0),
}


def _num(v, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if (isinstance(v, bool) or f != f or abs(f) == float("inf")) else f


def sanitize_workload(w: "Workload") -> list[str]:
    """Repairs the workload in place and returns what it changed, for the result's warnings."""
    notes: list[str] = []
    for f in fields(Workload):
        if f.name in _LIMITS:
            cur = getattr(w, f.name)
            dflt = f.default
            lo, hi = _LIMITS[f.name]
            is_int = isinstance(dflt, int) and not isinstance(dflt, bool) or (isinstance(dflt, tuple) and isinstance(dflt[0], int))
            if isinstance(dflt, tuple):
                a = _num(cur[0] if isinstance(cur, (tuple, list)) and len(cur) > 0 else None, dflt[0])
                b = _num(cur[1] if isinstance(cur, (tuple, list)) and len(cur) > 1 else None, dflt[1])
                if a > b:
                    a, b = b, a
                    notes.append(f"{f.name}: minimum was above maximum; swapped")
                a2, b2 = min(max(a, lo), hi), min(max(b, lo), hi)
                if (a2, b2) != (a, b):
                    notes.append(f"{f.name}: limited to {lo}–{hi}")
                new = (int(round(a2)), int(round(b2))) if is_int else (a2, b2)
            else:
                v = _num(cur, dflt)
                v2 = min(max(v, lo), hi)
                if v2 != v or v != cur:
                    notes.append(f"{f.name}: {cur!r} replaced by {int(round(v2)) if is_int else v2}")
                new = int(round(v2)) if is_int else v2
            setattr(w, f.name, new)
    return notes


@dataclass
class SimRequest:
    machine: Machine
    engines: list[EngineSpec]
    roles: Roles
    workload: Workload = field(default_factory=Workload)
    concurrency: list[int] = field(default_factory=lambda: [1, 2, 4])
    scenarios: int = 12
    seed: int = 1

    @classmethod
    def model_validate(cls, data: Any) -> "SimRequest":
        if not isinstance(data, dict):
            raise ValidationError("request must be an object")
        r = cls(machine=Machine.model_validate(data.get("machine")),
                engines=[EngineSpec.model_validate(e) for e in data.get("engines", [])],
                roles=_coerce(Roles, data.get("roles") or {}),
                workload=Workload.model_validate(data.get("workload") or {}),
                concurrency=[int(c) for c in data.get("concurrency", [1, 2, 4]) if int(c) > 0][:8] or [1],
                scenarios=max(1, min(60, int(data.get("scenarios", 12)))),
                seed=int(data.get("seed", 1)))
        if len(r.engines) > 24:
            raise ValidationError("at most 24 engines")
        return r


def to_dict(obj) -> dict:
    return asdict(obj)
