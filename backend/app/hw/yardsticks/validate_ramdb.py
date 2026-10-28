"""Validation against the RAM database, within one source's sequence (same system, one change per step).

Per group: the free parameter (fabric-link efficiency where the chiplet link binds, else RAM efficiency) is fitted
on the group's first point; every other point is a prediction. For chiplet parts the RAM efficiency is inherited
within the architecture from the measured single-DIMM point (TechPowerUp 9800X3D: 0.959), as the proposal's
"inherit within the architecture" rule says.
"""
from __future__ import annotations

from collections import defaultdict

from .. import arch
from ..cpus import lookup
from . import ramdb



def predict(p: dict, eff_dram: float | None, eff_link: float) -> float:
    """eff_dram None: from the JEDEC mechanism with the point's memory type, density and stated timings."""
    m, groups = arch.build(lookup(p["cpu"]), p["mts"], p["channels"], eff_dram, p["fclk"], eff_link,
                           mem_type=p.get("mem_type"), density_gbit=p.get("density_gbit"), trefi_ns=p.get("trefi_ns"))
    return arch.all_cores_read_gbs(m, groups)


def _bisect(f, target, lo=0.05, hi=1.2):
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def run() -> list[dict]:
    rows = []
    groups = defaultdict(list)
    for p in ramdb.points():
        groups[(p["source"], p["group"])].append(p)
    for (src, grp), pts in groups.items():
        pts.sort(key=lambda p: (p["step"] or 0, p["id"]))
        cpu = pts[0]["cpu"]
        spec = lookup(cpu)
        chiplet = bool(spec and spec["topology"] == "chiplet" and all(p["fclk"] for p in pts))
        base = next((p for p in pts if p["channels"] == 2), pts[0])
        if chiplet:
            eff_dram = None                                  # the JEDEC mechanism (dram.py)
            eff_link = _bisect(lambda x: predict(base, eff_dram, x), base["read_gbs"])
        else:
            eff_link = 1.0
            eff_dram = _bisect(lambda x: predict(base, x, 1.0), base["read_gbs"])
        for p in pts:
            model = predict(p, eff_dram, eff_link)
            calib = p is base
            rows.append(dict(p, model_gbs=model, error=model / p["read_gbs"] - 1, fitted=calib,
                             fitted_param=("link" if chiplet else "dram"),
                             param=eff_link if chiplet else (eff_dram or 0.0), held_out=(len(pts) > 1 and not calib)))
    return rows


if __name__ == "__main__":
    for r in run():
        tag = "fit " if r["fitted"] else ("pred" if r["held_out"] else "only")
        print(f"{r['id']:8s} {r['trust']:6s} {str(r['cpu'])[:18]:18s} {r['mts']:5d} ch{r['channels']} FCLK {str(r['fclk'] or '-'):>5s} "
              f"meas {r['read_gbs']:6.1f} model {r['model_gbs']:6.1f} {tag} {100 * r['error']:+6.1f} %  "
              f"[{r['fitted_param']} eff {r['param']:.3f}] {(r['change'] or '')[:24]}")
