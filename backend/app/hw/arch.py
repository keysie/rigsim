"""Package topologies per architecture (proposal §5: architecture differences are topology, from spec sheets).

Device-specific values come from cpus.py (vendor documentation); the rest are the global constants in
constants.py — nothing is fitted per system. Per-core bandwidth follows Little's law from the core's documented
capacity for outstanding cache misses: misses x 64 bytes / idle latency; where that capacity is not sourced yet, the
per-core link is not identified (non-binding, and results that cross it say so)."""
from __future__ import annotations

from . import constants as K
from . import dram
from .machine import Machine, Read
from . import cpu_uarch
from .primitives import Cache, Compute, Link, PowerBudget, Router, Storage

LINE_B = 64


def access_s(mts: float, cl: int | None) -> float:
    """Idle DRAM access: CAS time (CL cycles at the memory clock, MT/s / 2) + the global fixed term."""
    cas = cl * 2 / (mts * 1e6) if cl else 0.0
    return cas + K.FIXED_LATENCY_NS * 1e-9


def FIXED_LATENCY_S() -> float:
    return K.FIXED_LATENCY_NS * 1e-9


def _core(gid: str, spec: dict | None, core_type: str, n: int, clock_hz: float | None, budget: bool) -> Compute:
    """A core group: its miss queue, and — with a clock — its budgets from the microarchitecture's documentation."""
    uarch = ((spec or {}).get("core_uarch") or {})
    u = cpu_uarch.lookup(uarch.get(core_type) if isinstance(uarch, dict) else uarch)
    c = Compute(gid, clock_hz=clock_hz or 0.0, count=n, outstanding=_queue(spec, core_type))
    if u and clock_hz:
        c.ops_per_clock, c.issue_per_clock, c.energy_j = dict(u["rates"]), u["issue"], dict(u["energy"])
        c.budget = "pkg.power" if budget else None
    return c


def _l3(m: Machine, cid: str, size: float, router: str) -> str:
    m.add(Cache(cid, capacity_b=size, bandwidth_bps=None, access_s=None, backing="ram", write_allocate=True,
                write_back=True))
    m.connect(Link(f"{cid}.port", cid, router, None))
    return cid


def _queue(spec: dict | None, core_type: str) -> int | None:
    return ((spec or {}).get("outstanding_misses_per_core") or {}).get(core_type)


def core_link_from_queue(spec: dict | None, core_type: str, latency_s: float) -> float | None:
    q = ((spec or {}).get("outstanding_misses_per_core") or {}).get(core_type)
    return q * LINE_B / latency_s if q else None


# JEDEC: a DDR4 channel is one 64-bit bus; a DDR5 DIMM channel is two independent 32-bit sub-channels, each with its
# own command bus, banks and refresh.
SUBCHANNELS = {"ddr4": 1, "ddr5": 2}


def _channels(m: Machine, router: str, mts: float, channels: int, eff_dram: float, cl: int | None = None,
              mem_type: str | None = None) -> list[str]:
    """Each (sub-)channel its own simplex link and storage. DDR4: 8 bytes per transfer per channel; DDR5: two
    sub-channels of 4 bytes each per channel. Unknown type: one 64-bit channel (the same peak)."""
    k = SUBCHANNELS.get((mem_type or "").lower(), 1)
    per = mts * 1e6 * 8 / k
    names = []
    for c in range(channels):
        for s in range(k):
            n = f"ch{c}" if k == 1 else f"ch{c}.sub{s}"
            w = dram.write_timing((mem_type or "").lower(), mts, cl)
            m.add(Storage(n, capacity_b=16 * 2**30 / k, bandwidth_bps=per * eff_dram, access_s=access_s(mts, cl),
                          write_access_s=(FIXED_LATENCY_S() + w["cwl_ns"] * 1e-9) if w else None,
                          turnaround_s=w["turnaround_ns"] * 1e-9 if w else None))
            m.connect(Link(f"bus.{n}" if k > 1 else f"bus{c}", n, router, per, duplex=False))
            names.append(n)
    m.interleave("ram", names)
    return names


def domain(spec: dict) -> dict:
    """One NUMA domain of a split CPU (Intel SNC, AMD NPS): its share of cores, chiplets and channels."""
    if spec.get("snc"):
        k = spec["snc"]
        return dict(spec, cores={t: n // k for t, n in spec["cores"].items()}, channels=spec["channels"] // k)
    if spec.get("nps"):
        k = spec["nps"]
        return dict(spec, ccds=spec["ccds"] // k, channels=spec["channels"] // k)
    return spec


def add_cpu(m: Machine, spec: dict | None, mts: float, channels: int, eff_dram: float | None = None,
            fclk_mhz: float | None = None, eff_link: float | None = None, core_link_bps: float | None = None,
            cl: int | None = None, mem_type: str | None = None, density_gbit: int | None = None,
            trefi_ns: float | None = None, trfc_ns: float | None = None,
            l3_bytes: float | None = None, clock_hz: float | None = None,
            power: tuple[float | None, str] | None = None) -> tuple[str, list[str]]:
    """Adds the CPU package and its memory channels to an existing machine. Returns the package's I/O router (where
    PCIe root ports and an integrated GPU attach: 'pkg', or 'iod' on chiplet parts) and the core groups. Unknown
    CPU: one generic core group, monolithic. Efficiencies default to the global constants.
    `l3_bytes` (the machine's own report): a last-level cache between the cores and the package (per core die on
    chiplet parts), write-back and write-allocate — x86's write-back memory type (Intel SDM vol. 3A, memory types;
    AMD APM vol. 2, memory types). Its speeds and latency are not identified here: non-binding.
    `clock_hz` (the machine's maximum turbo clock) and `power` (watts, basis) make the cores budgets
    (docs/overhaul/cpu-compute.md): issue width and per-class rates from cpu_uarch, energy per instruction where
    calibrated, drawing on one package power budget. `core_link_bps`: what one
    core sustains (its own link to its router); None = from the documented miss queue (Little's law) where sourced,
    else not identified."""
    if eff_dram is None:
        t = (mem_type or "").lower()
        eff_dram = (dram.efficiency(t, mts, density_gbit, trefi_ns, trfc_ns)[0] if t in dram.JEDEC
                    else K.DRAM_READ_EFFICIENCY)
    eff_link = K.CHIPLET_LINK_EFFICIENCY if eff_link is None else eff_link
    lat = access_s(mts, cl)
    groups: list[str] = []
    if power is not None:
        m.add(PowerBudget("pkg.power", power[0], power[1]))
    if spec and spec["topology"] == "chiplet" and fclk_mhz:
        io = "iod"
        m.add(Router(io))
        _channels(m, io, mts, channels, eff_dram, cl, mem_type)
        for d in range(spec["ccds"]):
            r = f"ccd{d}"
            m.add(Router(r))
            m.connect(Link(f"ifop{d}", r, io, spec["ccd_read_bytes_per_fclk"] * fclk_mhz * 1e6 * eff_link, duplex=False))
            near = _l3(m, f"{r}.l3", l3_bytes / spec["ccds"], r) if l3_bytes else r
            for t, n in spec["cores_per_ccd"].items():
                g = f"{r}.{t}"
                m.add(_core(g, spec, t, n, clock_hz, power is not None))
                bps = core_link_bps if core_link_bps is not None else core_link_from_queue(spec, t, lat)
                m.connect(Link(f"{g}.link", g, near, bps, per_unit=True))
                groups.append(g)
    else:
        io = "pkg"
        m.add(Router(io))
        _channels(m, io, mts, channels, eff_dram, cl, mem_type)
        near = _l3(m, "l3", l3_bytes, io) if l3_bytes else io
        for t, n in ((spec or {}).get("cores") or {"generic": 8}).items():
            m.add(_core(t, spec, t, n, clock_hz, power is not None))
            bps = core_link_bps if core_link_bps is not None else core_link_from_queue(spec, t, lat)
            m.connect(Link(f"{t}.link", t, near, bps, per_unit=True))
            groups.append(t)
    return io, groups


def build(spec: dict | None, mts: float, channels: int, eff_dram: float | None = None, fclk_mhz: float | None = None,
          eff_link: float | None = None, core_link_bps: float | None = None, cl: int | None = None,
          mem_type: str | None = None, density_gbit: int | None = None, trefi_ns: float | None = None,
          trfc_ns: float | None = None) -> tuple[Machine, list[str]]:
    """A machine holding only the CPU and its memory (the yardsticks use this). Returns it and the core groups."""
    m = Machine()
    _, groups = add_cpu(m, spec, mts, channels, eff_dram, fclk_mhz, eff_link, core_link_bps, cl, mem_type,
                        density_gbit, trefi_ns, trfc_ns)
    return m, groups


def all_cores_read_gbs(m: Machine, groups: list[str]) -> float:
    res = m.solve_reads([Read(g, g, "ram") for g in groups])
    return sum(v["rate_bps"] for v in res.values()) / 1e9
