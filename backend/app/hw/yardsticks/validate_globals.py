"""Every yardstick against the model with the global constants only (proposal §5, spec-sheet first).

No value is fitted per system here. A row is `calibration` if its system contributed to a constant's median
(fit_globals.py), else `held_out`. Where the model misses, the miss is the finding: the tests name each one and its
suspected cause, so it is looked at in the model, not patched with a per-device value.
"""
from __future__ import annotations

from .. import arch
from .. import constants as K
from ..cpus import CPUS
from ..curves import load as load_curves
from ..gpus import GPUS
from ..machine import Read
from ..power import spec_clock_mhz
from . import igorslab_arrowlake as igor
from . import phoronix_ddr5 as pho
from . import ramdb
from . import tud_ladders as tud
from . import vsc_mlc as vsc

GB = 1e9


def _row(source, point, measured, model, unit, trust, role):
    return dict(source=source, point=point, measured=measured, model=model, unit=unit, trust=trust, role=role,
                error=model / measured - 1)


def cpu_rows() -> list[dict]:
    rows = []
    for name, n in vsc.NODES.items():
        o = vsc.observe(vsc.build(n, K.FIXED_LATENCY_NS * 1e-9, K.SOCKET_LINK_LATENCY_NS * 1e-9, K.DRAM_READ_EFFICIENCY))
        mm = n["measured"]
        for key, unit in (("lat_local_ns", "ns"), ("lat_remote_ns", "ns"), ("bw_local_gbs", "GB/s"), ("bw_system_gbs", "GB/s")):
            meas = vsc.mean(mm[key]) if isinstance(mm[key], tuple) else mm[key]
            rows.append(_row("VSC MLC", f"{name} {key}", meas, o[key], unit, "high", "calibration"))
    for kernel, d in tud.CLX.items():
        for cores, meas in d.items():
            spec = arch.domain(CPUS["Xeon Gold 6248"])
            m, g = arch.build(spec, 2933, spec["channels"])
            model = m.solve_reads([Read("x", g[0], "ram", units=cores)])["x"]["rate_bps"] / GB
            rows.append(_row("TUD ladders", f"Cascade Lake {kernel} {cores} cores", meas, model, "GB/s", "high", "calibration"))
    for kernel, d in tud.ROME.items():
        for cores, meas in d.items():
            spec = arch.domain(CPUS["EPYC 7702"])
            m, g = arch.build(spec, 3200, spec["channels"], fclk_mhz=spec["fclk_nominal_mhz"])
            reads = [Read("a", g[0], "ram", units=min(cores, 8))] + ([Read("b", g[1], "ram", units=cores - 8)] if cores > 8 else [])
            model = sum(v["rate_bps"] for v in m.solve_reads(reads).values()) / GB
            rows.append(_row("TUD ladders", f"Rome {kernel} {cores} cores", meas, model, "GB/s", "high", "calibration"))
    for cfg, (mts, rd, _, _) in igor.CONFIGS.items():
        rows.append(_row("igor'sLAB MLC", f"265K {cfg}", rd, igor.read_gbs(mts, K.DRAM_READ_EFFICIENCY), "GB/s", "medium",
                         "calibration"))
    for kernel in ("copy",):
        for mts, v in zip(pho.SPEEDS, pho.STREAM[kernel][0]):
            rows.append(_row("Phoronix STREAM", f"12900K {kernel} {mts}", v / 1000, pho.observe(mts, K.DRAM_READ_EFFICIENCY),
                             "GB/s", "medium", "held_out"))
    from .validate_ramdb import lookup                   # same CPU lookup the RAM database uses
    for p in ramdb.points():
        m, groups = arch.build(lookup(p["cpu"]), p["mts"], p["channels"], fclk_mhz=p["fclk"], mem_type=p["mem_type"])
        model = arch.all_cores_read_gbs(m, groups)
        role = "calibration" if lookup(p["cpu"]) and lookup(p["cpu"])["topology"] == "chiplet" else "held_out"
        rows.append(_row(f"RAM db {p['source']}", f"{p['id']} {p['cpu']} {p['mts']} ch{p['channels']}", p["read_gbs"],
                         model, "GB/s", p["trust"], role))
    return rows


def gpu_rows() -> list[dict]:
    data = load_curves()
    rows = []
    for c in data["curves"]:
        spec = GPUS[c["model"]]
        for cls, pts in c["points"].items():
            for w, mhz in pts:
                rows.append(_row(c["id"], f"{c['model']} {cls} {w:g} W", mhz, spec_clock_mhz(spec, w, cls), "MHz",
                                 c["trust"], "calibration"))
    for v in data["validation"]:
        spec = GPUS[v["model"]]
        for p in v["points"]:
            rows.append(_row(v["id"], f"{p['card']} {p['load_class']} {p['limit_w']:g} W", p["sm_mhz"],
                             spec_clock_mhz(spec, p["limit_w"], p["load_class"]), "MHz", v["trust"], "held_out"))
    return rows


def run() -> list[dict]:
    return cpu_rows() + gpu_rows()


TOLERANCE = {"high": 0.10, "medium": 0.10, "low": 0.15}


if __name__ == "__main__":
    for r in run():
        flag = "" if abs(r["error"]) <= TOLERANCE[r["trust"]] else "  <-- outside"
        print(f"{r['role'][:5]:5s} {r['trust']:6s} {r['source'][:26]:26s} {r['point'][:44]:44s} meas {r['measured']:8.1f} "
              f"model {r['model']:8.1f} {r['unit']:4s} {100 * r['error']:+6.1f} %{flag}")
