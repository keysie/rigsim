"""Yardstick 4: core ladders on one NUMA domain (Velten, Schöne, Ilsche, Hackenberg, TU Dresden, ICPE 2022).

Configuration verified against the paper's Table 1 (ar5iv.labs.arxiv.org/html/2204.03290): 2 x Xeon Gold 6248,
HPE DL360 Gen10, 12 x 32 GiB DDR4-2933, SNC on, Ubuntu 18.04, cores pinned to 1.6 GHz, uncore 2.4 GHz; 2 x EPYC
7702, GIGABYTE MZ62-HD0-00, 16 x 32 GiB DDR4-3200, NPS4, cores at 2.0 GHz, nominal FCLK 1467 MHz. BenchIT load-only
kernels, maximum of 120 samples per point. The bandwidth values are read from the paper's heat-map figures via the
research report (docs/overhaul/yardsticks.md): medium trust.

What it tests: the per-core link (identified here for the first time) and saturation, on a monolithic sub-NUMA
cluster and on an AMD quadrant with two chiplet links. Fitted on the AVX ladder: per-core bandwidth (1 core), the
cap (plateau). Held out: the AVX points in between, and the SSE ladder's plateaus (the caps cannot depend on the
instruction width; only the per-core value is refitted on SSE's single-core point).
"""
from __future__ import annotations

from .. import arch
from ..cpus import CPUS
from ..machine import Read

SOURCE = "Velten et al., 'Memory Performance of AMD EPYC Rome and Intel Cascade Lake SP Server Processors', ICPE 2022"
GB = 1e9
# cores -> GB/s, one NUMA domain, RAM read (load-only)
CLX = {"avx512": {1: 11.5, 2: 22.8, 4: 43.3, 8: 60.4, 10: 60.7},
       "sse": {1: 10.7, 2: 21.9, 4: 43.8, 8: 60.2, 10: 59.5}}
ROME = {"avx": {1: 21.0, 2: 34.9, 3: 36.8, 4: 36.4, 8: 35.8, 16: 42.8},       # 1-8 on one CCD, 16 on two
        "sse": {1: 17.7, 2: 32.4, 3: 36.8, 4: 36.6, 8: 35.7, 16: 42.7}}


def clx_read(cores: int, core_gbs: float, eff_dram: float) -> float:
    spec = arch.domain(CPUS["Xeon Gold 6248"])
    m, groups = arch.build(spec, 2933, spec["channels"], eff_dram, core_link_bps=core_gbs * GB)
    return m.solve_reads([Read("x", groups[0], "ram", units=cores)])["x"]["rate_bps"] / GB


def rome_read(cores: int, core_gbs: float, eff_link: float, eff_dram: float) -> float:
    spec = arch.domain(CPUS["EPYC 7702"])
    m, groups = arch.build(spec, 3200, spec["channels"], eff_dram, spec["fclk_nominal_mhz"], eff_link, core_gbs * GB)
    if cores <= 8:                                     # the paper fills one CCD first (cores 0-7), then the second
        reads = [Read("a", groups[0], "ram", units=cores)]
    else:
        reads = [Read("a", groups[0], "ram", units=8), Read("b", groups[1], "ram", units=cores - 8)]
    return sum(v["rate_bps"] for v in m.solve_reads(reads).values()) / GB


def _bisect(f, target, lo, hi, n=70):
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def fit_clx():
    d = CLX["avx512"]
    core = d[1]                                        # one core alone is core-link bound (far below the channels)
    eff = _bisect(lambda e: clx_read(10, core, e), d[10], 0.1, 1.0)
    return dict(core_gbs=core, eff_dram=eff)


def fit_rome():
    d = ROME["avx"]
    core = d[1]
    eff_link = _bisect(lambda e: rome_read(3, core, e, 1.0), d[3], 0.1, 1.0)     # one CCD's plateau: its link
    eff_dram = _bisect(lambda e: rome_read(16, core, eff_link, e), d[16], 0.1, 1.0)   # two CCDs: the channels
    return dict(core_gbs=core, eff_link=eff_link, eff_dram=eff_dram)


def report() -> list[dict]:
    rows = []
    c, r = fit_clx(), fit_rome()
    for isa, d in CLX.items():
        core = d[1]                                    # SSE: only the per-core value is refitted
        for n, meas in d.items():
            model = clx_read(n, core, c["eff_dram"])
            fitted = (isa == "avx512" and n in (1, 10)) or (isa == "sse" and n == 1)
            rows.append(dict(cpu="Xeon Gold 6248 (1 SNC)", isa=isa, cores=n, meas=meas, model=model, fitted=fitted))
    for isa, d in ROME.items():
        core = d[1]
        for n, meas in d.items():
            model = rome_read(n, core, r["eff_link"], r["eff_dram"])
            fitted = (isa == "avx" and n in (1, 3, 16)) or (isa == "sse" and n == 1)
            rows.append(dict(cpu="EPYC 7702 (1 NPS4 node)", isa=isa, cores=n, meas=meas, model=model, fitted=fitted))
    for x in rows:
        x["error"] = x["model"] / x["meas"] - 1
    return rows


if __name__ == "__main__":
    c, r = fit_clx(), fit_rome()
    print(f"CLX: per-core {c['core_gbs']} GB/s, RAM efficiency {c['eff_dram']:.3f}")
    print(f"Rome: per-core {r['core_gbs']} GB/s, chiplet-link efficiency {r['eff_link']:.3f} "
          f"(of 32 B x 1467 MHz = 46.9 GB/s), RAM efficiency {r['eff_dram']:.3f}")
    for x in report():
        print(f"  {x['cpu']:24s} {x['isa']:6s} {x['cores']:2d} cores  meas {x['meas']:5.1f}  model {x['model']:5.1f}  "
              f"{'fit ' if x['fitted'] else 'pred'} {100 * x['error']:+5.1f} %")
