"""Derives every global constant in hw/constants.py from its data (proposal §5, spec-sheet first).

Rule: one value per measured system (so a source with many points on one system counts once), then the median over
systems. Each derivation uses the yardsticks' identification code only to read a system's value off its data; the
model itself then runs with the medians (validate_globals.py). A test fails if constants.py differs from this.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from .. import constants as K
from ..curves import load as load_curves
from ..gpus import GPUS
from . import igorslab_arrowlake as igor
from . import tud_ladders as tud
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
