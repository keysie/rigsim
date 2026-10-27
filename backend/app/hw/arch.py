"""Package topologies per architecture (proposal §5: architecture differences are topology, from spec sheets).

Device-specific values come from cpus.py (vendor documentation); the rest are the global constants in
constants.py — nothing is fitted per system. Per-core bandwidth follows Little's law from the core's documented
capacity for outstanding cache misses: misses x 64 bytes / idle latency; where that capacity is not sourced yet, the
per-core link is not identified (non-binding, and results that cross it say so)."""
from __future__ import annotations

from . import constants as K
from .machine import Machine, Read
from .primitives import Compute, Link, Router, Storage

LINE_B = 64


def access_s(mts: float, cl: int | None) -> float:
    """Idle DRAM access: CAS time (CL cycles at the memory clock, MT/s / 2) + the global fixed term."""
    cas = cl * 2 / (mts * 1e6) if cl else 0.0
    return cas + K.FIXED_LATENCY_NS * 1e-9


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
            m.add(Storage(n, capacity_b=16 * 2**30 / k, bandwidth_bps=per * eff_dram, access_s=access_s(mts, cl)))
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


def build(spec: dict | None, mts: float, channels: int, eff_dram: float | None = None, fclk_mhz: float | None = None,
          eff_link: float | None = None, core_link_bps: float | None = None, cl: int | None = None,
          mem_type: str | None = None) -> tuple[Machine, list[str]]:
    """Returns the machine and the core groups that read. Unknown CPU: one generic core group, monolithic.
    Efficiencies default to the global constants. `core_link_bps`: what one core sustains (its own link to its
    router); None = from the documented miss queue (Little's law) where sourced, else not identified."""
    eff_dram = K.DRAM_READ_EFFICIENCY if eff_dram is None else eff_dram
    eff_link = K.CHIPLET_LINK_EFFICIENCY if eff_link is None else eff_link
    lat = access_s(mts, cl)
    m = Machine()
    groups: list[str] = []
    if spec and spec["topology"] == "chiplet" and fclk_mhz:
        m.add(Router("iod"))
        _channels(m, "iod", mts, channels, eff_dram, cl, mem_type)
        for d in range(spec["ccds"]):
            r = f"ccd{d}"
            m.add(Router(r))
            m.connect(Link(f"ifop{d}", r, "iod", spec["ccd_read_bytes_per_fclk"] * fclk_mhz * 1e6 * eff_link, duplex=False))
            for t, n in spec["cores_per_ccd"].items():
                g = f"{r}.{t}"
                m.add(Compute(g, clock_hz=0.0, count=n))
                bps = core_link_bps if core_link_bps is not None else core_link_from_queue(spec, t, lat)
                m.connect(Link(f"{g}.link", g, r, bps, per_unit=True))
                groups.append(g)
    else:
        m.add(Router("pkg"))
        _channels(m, "pkg", mts, channels, eff_dram, cl, mem_type)
        for t, n in ((spec or {}).get("cores") or {"generic": 8}).items():
            m.add(Compute(t, clock_hz=0.0, count=n))
            bps = core_link_bps if core_link_bps is not None else core_link_from_queue(spec, t, lat)
            m.connect(Link(f"{t}.link", t, "pkg", bps, per_unit=True))
            groups.append(t)
    return m, groups


def all_cores_read_gbs(m: Machine, groups: list[str]) -> float:
    res = m.solve_reads([Read(g, g, "ram") for g in groups])
    return sum(v["rate_bps"] for v in res.values()) / 1e9
