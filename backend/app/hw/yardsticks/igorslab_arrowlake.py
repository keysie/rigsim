"""Yardstick 3 (candidate): Core Ultra 7 265K, Intel MLC 3.12, five memory configurations on one system.

Source: igor'sLAB, "Arrow Lake DDR5 in detail: memory clock, uncore, timings" (2026), page 2. Each step changes one
variable: transfer rate (6000 -> 7800 with absolute primary timings kept), the uncore clock (memory controller and
fabric), or the secondary timings (tRAS, tRFC, tREFI, tWR, tWTR, tRTP, tRRD). Single runs, no spread published;
DIMM count INFERRED from the stated theoretical bandwidth (96.0 GB/s at 6000 = 2 DIMMs x 8 bytes).

The model's claims tested: bandwidth = channels x rate x 8 bytes x a constant efficiency (timings and uncore do not
enter bandwidth); the controller/fabric term enters latency only.
"""
from __future__ import annotations

from .phoronix_ddr5 import build as _build_two_dimm   # same topology: one package, 2 DIMMs x 2 sub-channels
from ..machine import Read

SOURCE = "igorslab.de/en/arrow-lake-ddr5-detail-memory-clock-uncore-timings-core-ultra-7-265k-two-platforms/2/"
CONFIGS = {   # name: (MT/s, all reads GB/s, 1:1 reads/writes GB/s, idle latency ns)
    "jedec5600": (5600, 75.97, 61.15, 106.7),
    "xmp6000": (6000, 90.01, 75.28, 87.2),
    "xmp7800": (7800, 117.76, 92.75, 88.6),
    "xmp7800_uncore": (7800, 118.20, 92.59, 77.0),
    "manual7800_uncore": (7800, 122.35, 102.42, 78.1),
}


def read_gbs(mts: int, eff: float) -> float:
    m = _build_two_dimm(mts, eff)
    res = m.solve_reads([Read("p", "pcores", "ram"), Read("e", "ecores", "ram")])
    return sum(v["rate_bps"] for v in res.values()) / 1e9


def eff_from(name: str) -> float:
    mts, rd, _, _ = CONFIGS[name]
    return rd / read_gbs(mts, 1.0)


def predicted_error(fit_on: str, name: str) -> float:
    mts, rd, _, _ = CONFIGS[name]
    return read_gbs(mts, eff_from(fit_on)) / rd - 1


if __name__ == "__main__":
    for n, (mts, rd, rw, lat) in CONFIGS.items():
        print(f"{n:18s} {mts} MT/s  reads {rd:6.2f} GB/s = {rd / read_gbs(mts, 1.0):.3f} of peak  1:1 {rw:6.2f}  idle {lat} ns  "
              f"model error (efficiency from xmp7800): {100 * predicted_error('xmp7800', n):+.1f} %")
