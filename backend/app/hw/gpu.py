"""A GPU as a bundle of primitives built from its documented architecture (proposal §5, spec-sheet first).

  {id}.sm        compute: every SM of the card (GPC split not modelled: not documented per card)
  {id}.xbar      router: the crossbar between SMs, L2, memory controllers and the copy engines
  {id}.l2        storage: capacity from the spec; bandwidth and access time not published (None, non-binding)
  {id}.mc<i>     one link per 32-bit memory controller (pins x data rate), to its VRAM slice {id}.vram<i>
  {id}.vram      the interleaved set of the slices (addresses spread over every controller)
  {id}.copy      router: the copy engines and the PCIe interface, linked to the slot's router at the negotiated
                 generation and width

SM clock and VRAM bandwidth per state: from a measured curve (the user's override) where one exists for the state's
load class, else the spec-sheet law (power.spec_clock_mhz) and the spec VRAM bandwidth x the global VRAM efficiency.
Every value says where it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import constants as K
from .gpus import GPUS, channel_bandwidth_gbs
from .links import pcie_bps
from .machine import Machine
from .power import LOAD_CLASSES, PowerCurve, spec_clock_mhz
from .primitives import Compute, Link, Router, Storage


@dataclass
class GpuDevice:
    id: str
    model: str
    power_limit_w: float
    curves: dict[tuple[str, str], PowerCurve] = field(default_factory=dict)   # user overrides: (primitive, class)
    load_class: str = "compute"
    memory: str = ""                  # the storage its data lives in: '<id>.vram', or the system's 'ram' (iGPU)

    @property
    def spec(self) -> dict:
        return GPUS[self.model]

    def _curve(self, primitive: str) -> tuple[float, str] | None:
        own = self.curves.get((primitive, self.load_class))
        if own:
            how = own.basis if own.covers(self.power_limit_w) else f"{own.basis}, extrapolated flat"
            return own.value(self.power_limit_w), f"{how} ({own.source})"
        other = next((self.curves[(primitive, c)] for c in LOAD_CLASSES if (primitive, c) in self.curves), None)
        if other:
            return other.value(self.power_limit_w), f"other load class's curve ({other.source})"
        return None

    def sm_clock_hz(self) -> tuple[float, str]:
        c = self._curve("sm_clock")
        if c:
            return c[0] * 1e6, c[1]
        if self.spec.get("fixed_clock_mhz"):               # an integrated GPU: its clock as the run recorded it
            return self.spec["fixed_clock_mhz"] * 1e6, "the run's own clock record (integrated GPU)"
        return spec_clock_mhz(self.spec, self.power_limit_w, self.load_class) * 1e6, "spec-sheet law (global constants)"

    def vram_bps(self) -> tuple[float, str]:
        c = self._curve("vram_bandwidth")
        if c:
            return c[0] * 1e9, c[1]
        if self.spec.get("integrated"):
            return 0.0, "the system's memory: a flow through the machine's graph"
        how = "spec x global VRAM efficiency" + ("" if K.VRAM_EFFICIENCY_MEASURED else " (unmeasured)")
        return self.spec["mem_controllers"] * channel_bandwidth_gbs(self.spec) * 1e9 * K.VRAM_EFFICIENCY, how


def add_gpu(m: Machine, gid: str, model: str, slot_router: str, *, slot_gen: int, slot_lanes: int,
            power_limit_w: float | None = None, curves: dict | None = None) -> GpuDevice:
    """Adds the card's documented structure and its PCIe link to `slot_router` (negotiated: the lower of card and
    slot, in generation and in width)."""
    spec = GPUS[model]
    dev = GpuDevice(gid, model, power_limit_w or spec["tdp_w"], dict(curves or {}))
    n = spec["mem_controllers"]
    m.add(Compute(f"{gid}.sm", clock_hz=0.0, ops_per_clock=dict(spec["ops_per_clk_per_sm"]), count=spec["sms"]))
    m.add(Router(f"{gid}.xbar"))
    m.add(Router(f"{gid}.copy"))
    m.add(Storage(f"{gid}.l2", capacity_b=spec["l2_mb"] * 2**20, bandwidth_bps=None, access_s=None))
    m.connect(Link(f"{gid}.sm_bus", f"{gid}.sm", f"{gid}.xbar", None))
    m.connect(Link(f"{gid}.l2_bus", f"{gid}.l2", f"{gid}.xbar", None))
    m.connect(Link(f"{gid}.copy_bus", f"{gid}.copy", f"{gid}.xbar", None))
    chan = channel_bandwidth_gbs(spec) * 1e9
    for i in range(n):
        m.add(Storage(f"{gid}.vram{i}", capacity_b=spec["vram_gb"] * 1e9 / n, bandwidth_bps=0.0, access_s=None))
        m.connect(Link(f"{gid}.mc{i}", f"{gid}.vram{i}", f"{gid}.xbar", chan, duplex=False))
    m.interleave(f"{gid}.vram", [f"{gid}.vram{i}" for i in range(n)])
    gen, lanes = min(spec["pcie_gen"], slot_gen), min(spec["pcie_lanes"], slot_lanes)
    m.connect(Link(f"{gid}.pcie", f"{gid}.copy", slot_router, pcie_bps(gen, lanes), latency_s=0.0, duplex=True))
    dev.memory = f"{gid}.vram"
    m.devices = getattr(m, "devices", {})
    m.devices[gid] = dev
    set_load_class(m, gid, dev.load_class)
    return dev


def add_igpu(m: Machine, gid: str, model: str, host_router: str, memory: str = "ram") -> GpuDevice:
    """An integrated GPU: its execution units on the CPU package's router, reading the SAME memory as the cores
    (`memory`, the package's interleaved channels) — so CPU and iGPU share its bandwidth in the graph. The graphics
    interface to the ring is not documented: non-binding."""
    spec = GPUS[model]
    dev = GpuDevice(gid, model, spec["tdp_w"])
    m.add(Compute(f"{gid}.sm", clock_hz=0.0, ops_per_clock=dict(spec["ops_per_clk_per_sm"]), count=spec["sms"]))
    m.add(Router(f"{gid}.gti"))
    m.connect(Link(f"{gid}.eu_bus", f"{gid}.sm", f"{gid}.gti", None))
    m.connect(Link(f"{gid}.ring", f"{gid}.gti", host_router, None))
    dev.memory = memory
    m.devices = getattr(m, "devices", {})
    m.devices[gid] = dev
    set_load_class(m, gid, dev.load_class)
    return dev


def set_load_class(m: Machine, gid: str, load_class: str) -> dict:
    """Applies a state's load class to the card: SM clock and VRAM bandwidth for it, with their provenance."""
    if load_class not in LOAD_CLASSES:
        raise ValueError(load_class)
    dev = m.devices[gid]
    dev.load_class = load_class
    clock, how_c = dev.sm_clock_hz()
    bw, how_b = dev.vram_bps()
    m.nodes[f"{gid}.sm"].clock_hz = clock
    if not dev.spec.get("integrated"):                # an iGPU's memory belongs to the system: not the card's to set
        n = dev.spec["mem_controllers"]
        for i in range(n):
            m.nodes[f"{gid}.vram{i}"].bandwidth_bps = bw / n
    return dict(load_class=load_class, sm_clock_hz=clock, sm_clock_basis=how_c, vram_bps=bw, vram_basis=how_b)
