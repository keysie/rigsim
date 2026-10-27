"""A GPU as a bundle of primitives (proposal §5): SMs (compute), VRAM (storage), copy engine (router) and its PCIe
port (link to the slot's router). Power-limited: each primitive's scalable parameter follows its own curve against
the card's power limit (SM clock; VRAM bandwidth)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .gpus import GPUS
from .links import pcie_bps
from .machine import Machine
from .power import PowerCurve
from .primitives import Compute, Link, Router, Storage


@dataclass
class GpuDevice:
    id: str
    model: str
    power_limit_w: float
    curves: dict[str, PowerCurve] = field(default_factory=dict)      # "sm_clock", "vram_bandwidth"

    @property
    def spec(self) -> dict:
        return GPUS[self.model]

    @property
    def limit_ratio(self) -> float:
        return min(1.0, self.power_limit_w / self.spec["tdp_w"])

    def factor(self, primitive: str) -> float:
        return self.curves.get(primitive, PowerCurve()).factor(self.limit_ratio)

    def unmeasured(self) -> list[str]:
        return [p for p in ("sm_clock", "vram_bandwidth") if not self.curves.get(p, PowerCurve()).measured]


def add_gpu(m: Machine, gid: str, model: str, slot_router: str, *, slot_gen: int, slot_lanes: int,
            power_limit_w: float | None = None, curves: dict[str, PowerCurve] | None = None) -> GpuDevice:
    """Adds the card's primitives and its PCIe link to `slot_router`. The link runs at the lower of the card's and
    the slot's generation and width (negotiated)."""
    spec = GPUS[model]
    dev = GpuDevice(gid, model, power_limit_w or spec["tdp_w"], dict(curves or {}))
    clock = spec["boost_mhz"] * 1e6 * dev.factor("sm_clock")
    m.add(Compute(f"{gid}.sm", clock_hz=clock, ops_per_clock=dict(spec["ops_per_clk_per_sm"]), count=spec["sms"]))
    m.add(Storage(f"{gid}.vram", capacity_b=spec["vram_gb"] * 1e9,
                  bandwidth_bps=spec["vram_gbs"] * 1e9 * dev.factor("vram_bandwidth"), access_s=None))
    m.add(Router(f"{gid}.copy"))
    m.connect(Link(f"{gid}.vram_bus", f"{gid}.vram", f"{gid}.copy", None, duplex=False))   # VRAM bandwidth binds
    m.connect(Link(f"{gid}.sm_bus", f"{gid}.sm", f"{gid}.copy", None))
    gen, lanes = min(spec["pcie_gen"], slot_gen), min(spec["pcie_lanes"], slot_lanes)
    m.connect(Link(f"{gid}.pcie", f"{gid}.copy", slot_router, pcie_bps(gen, lanes), latency_s=0.0, duplex=True))
    m.devices = getattr(m, "devices", {})
    m.devices[gid] = dev
    return dev
