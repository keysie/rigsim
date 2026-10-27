"""A GPU as a bundle of primitives (proposal §5): SMs (compute), VRAM (storage), copy engine (router) and its PCIe
port (link to the slot's router). Power-limited: the SM clock and the VRAM bandwidth come from the card's power
curves for the state's load class at its power limit; without a curve, from the spec (marked unmeasured)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .gpus import GPUS
from .links import pcie_bps
from .machine import Machine
from .power import LOAD_CLASSES, PowerCurve
from .primitives import Compute, Link, Router, Storage


@dataclass
class GpuDevice:
    id: str
    model: str
    power_limit_w: float
    curves: dict[tuple[str, str], PowerCurve] = field(default_factory=dict)   # (primitive, load class) -> curve
    load_class: str = "compute"

    @property
    def spec(self) -> dict:
        return GPUS[self.model]

    def _lookup(self, primitive: str, spec_value: float) -> tuple[float, str]:
        own = self.curves.get((primitive, self.load_class))
        if own:
            how = own.basis if own.covers(self.power_limit_w) else f"{own.basis}, extrapolated flat"
            return own.value(self.power_limit_w), f"{how} ({own.source})"
        other = next((self.curves[(primitive, c)] for c in LOAD_CLASSES if (primitive, c) in self.curves), None)
        if other:
            return other.value(self.power_limit_w), f"other load class's curve ({other.source})"
        return spec_value, "unmeasured (spec)"

    def sm_clock_hz(self) -> tuple[float, str]:
        mhz, how = self._lookup("sm_clock", self.spec["boost_mhz"])
        return mhz * 1e6, how

    def vram_bps(self) -> tuple[float, str]:
        gbs, how = self._lookup("vram_bandwidth", self.spec["vram_gbs"])
        return gbs * 1e9, how


def add_gpu(m: Machine, gid: str, model: str, slot_router: str, *, slot_gen: int, slot_lanes: int,
            power_limit_w: float | None = None, curves: dict | None = None) -> GpuDevice:
    """Adds the card's primitives and its PCIe link to `slot_router`. The link runs at the lower of the card's and
    the slot's generation and width (negotiated)."""
    spec = GPUS[model]
    dev = GpuDevice(gid, model, power_limit_w or spec["tdp_w"], dict(curves or {}))
    m.add(Compute(f"{gid}.sm", clock_hz=0.0, ops_per_clock=dict(spec["ops_per_clk_per_sm"]), count=spec["sms"]))
    m.add(Storage(f"{gid}.vram", capacity_b=spec["vram_gb"] * 1e9, bandwidth_bps=0.0, access_s=None))
    m.add(Router(f"{gid}.copy"))
    m.connect(Link(f"{gid}.vram_bus", f"{gid}.vram", f"{gid}.copy", None, duplex=False))   # VRAM bandwidth binds
    m.connect(Link(f"{gid}.sm_bus", f"{gid}.sm", f"{gid}.copy", None))
    gen, lanes = min(spec["pcie_gen"], slot_gen), min(spec["pcie_lanes"], slot_lanes)
    m.connect(Link(f"{gid}.pcie", f"{gid}.copy", slot_router, pcie_bps(gen, lanes), latency_s=0.0, duplex=True))
    m.devices = getattr(m, "devices", {})
    m.devices[gid] = dev
    set_load_class(m, gid, dev.load_class)
    return dev


def set_load_class(m: Machine, gid: str, load_class: str) -> dict:
    """Applies a state's load class to the card: SM clock and VRAM bandwidth from the matching curves.
    Returns the values with their provenance."""
    if load_class not in LOAD_CLASSES:
        raise ValueError(load_class)
    dev = m.devices[gid]
    dev.load_class = load_class
    clock, how_c = dev.sm_clock_hz()
    bw, how_b = dev.vram_bps()
    m.nodes[f"{gid}.sm"].clock_hz = clock
    m.nodes[f"{gid}.vram"].bandwidth_bps = bw
    return dict(load_class=load_class, sm_clock_hz=clock, sm_clock_basis=how_c, vram_bps=bw, vram_basis=how_b)
