"""Achievable rates per (runtime and version, operation class) on a compute (proposal §7).

The hardware answers what the vendor's reference library reaches (cuBLAS / cuBLASLt matrix products, a plain copy
for bandwidth), as a share of the spec peak. Lookup order: a measurement for this GPU model; else one inherited
from a measured model of the same architecture (per clock and per unit, so it scales with clock and SM count);
else the spec peak times one global default efficiency, marked unmeasured. Engines multiply by their own kernel
factor (per engine version and operation class) — not here."""
from __future__ import annotations

from dataclasses import dataclass

from .gpus import GPUS

DEFAULT_EFFICIENCY = 0.7     # used only where nothing is measured, and flagged as such
# Time one small kernel takes on the GPU when kernels are queued back to back (CUDA graph replay; the stick's
# gpu-launch question measures it per card and runtime). Placeholder of the right order (a few microseconds) until
# measured; flagged wherever it is used. Not fitted to any end-to-end benchmark.
DEFAULT_LAUNCH_S = 2.5e-6


@dataclass
class Measured:
    model: str
    runtime: str          # e.g. "cuda-12.8"
    op_class: str
    efficiency: float     # achieved / spec peak at the measured clock
    source: str


MEASUREMENTS: list[Measured] = []          # filled from low-level GPU data (roadmap step 2)


@dataclass
class MeasuredLaunch:
    model: str
    runtime: str
    graphs: bool
    seconds: float
    source: str


LAUNCHES: list[MeasuredLaunch] = []        # from the stick's / bench's gpu-launch question


def launch_s(model: str, runtime: str, graphs: bool = True) -> tuple[float, str]:
    """Per-kernel time of tiny kernels back to back: measured for this model, else inherited from a card of the same
    architecture (launch cost does not scale with units), else the placeholder."""
    for exact in (True, False):
        for m in LAUNCHES:
            same = m.model == model if exact else GPUS[m.model]["arch"] == GPUS[model]["arch"]
            if same and m.runtime == runtime and m.graphs == graphs:
                return m.seconds, f"{'measured' if exact else 'inherited from ' + m.model} ({m.source})"
    return DEFAULT_LAUNCH_S, "unmeasured placeholder (gpu-launch not yet measured)"


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
