"""Derives every global constant in hw/constants.py from its data (proposal §5, spec-sheet first).

Rule: one value per measured system (so a source with many points on one system counts once), then the median over
systems. Each derivation uses the yardsticks' identification code only to read a system's value off its data; the
model itself then runs with the medians (validate_globals.py). A test fails if constants.py differs from this.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from .. import constants as K
from .. import dram
from ..curves import load as load_curves
from ..gpus import GPUS
from . import igorslab_arrowlake as igor
from . import tud_ladders as tud
from . import ramdb
from . import validate_ramdb as vr
from . import vsc_mlc as vsc


def per_system() -> dict[str, list[tuple[str, float]]]:
    """(system, value) per constant."""
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for node in vsc.NODES:
        p = vsc.fit(node)
        out["DRAM_READ_EFFICIENCY"].append((f"VSC {node} (MLC)", p["ram_eff"]))
        out["FIXED_LATENCY_NS"].append((f"VSC {node} (MLC)", p["fixed_s"] * 1e9))
        out["SOCKET_LINK_LATENCY_NS"].append((f"VSC {node} (MLC)", p["socket_latency_s"] * 1e9))
    clx, rome = tud.fit_clx(), tud.fit_rome()
    out["DRAM_READ_EFFICIENCY"] += [("TUD Cascade Lake (likwid)", clx["eff_dram"]), ("TUD Rome (likwid)", rome["eff_dram"])]
    out["DRAM_READ_EFFICIENCY"].append(("igor'sLAB Core Ultra 7 265K (MLC, median of 5 configs)",
                                        statistics.median(igor.eff_from(n) for n in igor.CONFIGS)))
    out["CHIPLET_LINK_EFFICIENCY"].append(("TUD Rome (likwid)", rome["eff_link"]))
    # scheduling share: the value that reproduces each DRAM-bound system's efficiency with its JEDEC timings
    def s_for(t, mts, dens, meas, trefi=None):
        lo, hi = 0.0, 8.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if dram.efficiency(t, mts, dens, trefi, same_bg_share=mid)[0] > meas else (lo, mid)
        return (lo + hi) / 2
    for node, n in vsc.NODES.items():
        out["DRAM_SAME_BANK_GROUP_SHARE"].append((f"VSC {node} (MLC)", s_for("ddr4", n["mts"], 8, vsc.fit(node)["ram_eff"])))
    out["DRAM_SAME_BANK_GROUP_SHARE"].append(("TUD Cascade Lake (likwid)", s_for("ddr4", 2933, 8, clx["eff_dram"])))
    out["DRAM_SAME_BANK_GROUP_SHARE"].append(("TUD Rome (likwid)", s_for("ddr4", 3200, 8, rome["eff_dram"])))
    out["DRAM_SAME_BANK_GROUP_SHARE"].append(("igor'sLAB Core Ultra 7 265K (MLC, median of configs)", statistics.median(
        s_for("ddr5", mts, 16, igor.eff_from(c)) for c, (mts, _, _, _) in igor.CONFIGS.items())))
    hw = [p for p in ramdb.points() if p["source"].startswith("hwluxx") and p["trust"] == "medium"]
    out["DRAM_SAME_BANK_GROUP_SHARE"].append(("hardwareluxx Core Ultra 9 285K (median of configs)", statistics.median(
        s_for("ddr5", p["mts"], p["density_gbit"], p["read_gbs"] / (p["mts"] * 8 * p["channels"] / 1000)) for p in hw)))
    seen = set()
    for r in vr.run():
        if r["fitted"] and r["fitted_param"] == "link" and r["source"] not in seen:
            seen.add(r["source"])
            out["CHIPLET_LINK_EFFICIENCY"].append((f"RAM db {r['source']} ({r['cpu']})", r["param"]))
    # GPU: sustained clock factor from compute-heavy points, memory gain from pairs at the same limit
    data = load_curves()
    for c in data["curves"]:
        spec = GPUS[c["model"]]
        pts = dict(c["points"])
        for w, mhz in pts.get("compute", []):
            k = mhz / (spec["boost_mhz"] * (w / spec["tdp_w"]) ** K.CLOCK_POWER_EXPONENT)
            out["SUSTAINED_CLOCK_FACTOR"].append((f"{c['id']} compute {w:g} W", k))
        comp = dict(pts.get("compute", []))
        for w, mhz in pts.get("memory", []):
            if w in comp:
                out["MEMORY_LOAD_CLOCK_GAIN"].append((f"{c['id']} {w:g} W", mhz / comp[w]))
    return out


def derive() -> dict[str, float]:
    return {k: statistics.median(v for _, v in rows) for k, rows in per_system().items()}


if __name__ == "__main__":
    for k, rows in per_system().items():
        med = statistics.median(v for _, v in rows)
        print(f"{k} = {med:.4g}   (constants.py: {getattr(K, k):.4g})")
        for name, v in rows:
            print(f"    {v:8.4g}  {name}")
