"""Achievable rates per (runtime and version, operation class) on a compute (proposal §7).

The hardware answers what the vendor's reference library reaches (cuBLAS / cuBLASLt matrix products, a plain copy
for bandwidth), as a share of the spec peak. Lookup order: a measurement for this GPU model; else one inherited
from a measured model of the same architecture (per clock and per unit, so it scales with clock and SM count);
else the spec peak times one global default efficiency, marked unmeasured. Engines multiply by their own kernel
factor (per engine version and operation class) — not here."""
from __future__ import annotations

from dataclasses import dataclass

from .gpus import GPUS

DEFAULT_EFFICIENCY = 0.7     # the only global constant: used only where nothing is measured, and flagged as such


@dataclass
class Measured:
    model: str
    runtime: str          # e.g. "cuda-12.8"
    op_class: str
    efficiency: float     # achieved / spec peak at the measured clock
    source: str


MEASUREMENTS: list[Measured] = []          # filled from low-level GPU data (roadmap step 2)


def achievable(model: str, runtime: str, op_class: str, clock_hz: float) -> tuple[float, str]:
    """Ops per second on the whole GPU at `clock_hz`, and the provenance: measured / inherited / unmeasured."""
    spec = GPUS[model]
    peak = spec["ops_per_clk_per_sm"][op_class] * spec["sms"] * clock_hz
    for m in MEASUREMENTS:
        if m.model == model and m.runtime == runtime and m.op_class == op_class:
            return peak * m.efficiency, f"measured ({m.source})"
    for m in MEASUREMENTS:
        if GPUS[m.model]["arch"] == spec["arch"] and m.runtime == runtime and m.op_class == op_class:
            return peak * m.efficiency, f"inherited from {m.model} ({m.source})"
    return peak * DEFAULT_EFFICIENCY, "unmeasured (spec peak x default efficiency)"


def op_time_s(ops: float, rate_ops: float, bytes_: float, bandwidth_bps: float, launch_s: float = 0.0) -> float:
    """One operation: the larger of compute and memory time, plus a fixed launch cost (proposal §6)."""
    return max(ops / rate_ops if rate_ops else 0.0, bytes_ / bandwidth_bps if bandwidth_bps else 0.0) + launch_s
