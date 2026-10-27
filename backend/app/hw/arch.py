"""Package topologies per architecture (proposal §5: architecture differences are topology, guided by spec sheets)."""
from __future__ import annotations

from .machine import Machine, Read
from .primitives import Compute, Link, Router, Storage


def _channels(m: Machine, router: str, mts: float, channels: int, eff_dram: float) -> list[str]:
    """Each channel its own simplex link and storage (8 bytes per transfer per channel)."""
    per = mts * 1e6 * 8
    names = []
    for c in range(channels):
        n = f"ch{c}"
        m.add(Storage(n, capacity_b=16 * 2**30, bandwidth_bps=per * eff_dram, access_s=80e-9))
        m.connect(Link(f"bus{c}", n, router, per, duplex=False))
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


def build(spec: dict | None, mts: float, channels: int, eff_dram: float, fclk_mhz: float | None = None,
          eff_link: float = 1.0, core_link_bps: float | None = None) -> tuple[Machine, list[str]]:
    """Returns the machine and the core groups that read. Unknown CPU: one generic core group, monolithic.
    `core_link_bps`: what one core sustains (its own link to its router); None = not identified."""
    m = Machine()
    groups: list[str] = []
    if spec and spec["topology"] == "chiplet" and fclk_mhz:
        m.add(Router("iod"))
        _channels(m, "iod", mts, channels, eff_dram)
        for d in range(spec["ccds"]):
            r = f"ccd{d}"
            m.add(Router(r))
            m.connect(Link(f"ifop{d}", r, "iod", spec["ccd_read_bytes_per_fclk"] * fclk_mhz * 1e6 * eff_link, duplex=False))
            for t, n in spec["cores_per_ccd"].items():
                g = f"{r}.{t}"
                m.add(Compute(g, clock_hz=0.0, count=n))
                m.connect(Link(f"{g}.link", g, r, core_link_bps, per_unit=True))
                groups.append(g)
    else:
        m.add(Router("pkg"))
        _channels(m, "pkg", mts, channels, eff_dram)
        for t, n in ((spec or {}).get("cores") or {"generic": 8}).items():
            m.add(Compute(t, clock_hz=0.0, count=n))
            m.connect(Link(f"{t}.link", t, "pkg", core_link_bps, per_unit=True))
            groups.append(t)
    return m, groups


def all_cores_read_gbs(m: Machine, groups: list[str]) -> float:
    res = m.solve_reads([Read(g, g, "ram") for g in groups])
    return sum(v["rate_bps"] for v in res.values()) / 1e9
