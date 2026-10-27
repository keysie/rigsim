"""Yardstick 2: one Linux system at ten DDR5 speeds (Phoronix, OpenBenchmarking result 2202236-PTS-DDR5600092).

Intel Core i9-12900K (8 P + 8 E cores), ASUS ROG STRIX Z690-E, 2 x 16 GB G.SKILL Trident Z5 DDR5, Ubuntu 22.04,
STREAM 2013-01-17 (-O3 -march=native -fopenmp), N = 5 runs per speed. Timings per speed: not published.

What it tests: the model's claim that bandwidth = channels x transfer rate x 8 bytes x a constant efficiency.
STREAM's kernels also write, and how it counts bytes differs from the traffic on the RAM; that ratio is unknown but
the same at every speed, so it is absorbed into the efficiency fitted at one speed. Latency is not tested here.
"""
from __future__ import annotations

from ..machine import Machine, Read
from ..primitives import Compute, Link, Router, Storage

GB = 1e9
SOURCE = "openbenchmarking.org/result/2202236-PTS-DDR5600092 (Phoronix, 'DDR5-6000 Memory Performance On Linux', 2022)"
SPEEDS = [3000, 3200, 3600, 3800, 4200, 4600, 5000, 5400, 5600, 6000]
# MB/s (1e6 bytes), mean of 5 runs, and standard error
STREAM = {
    "copy":  ([36664.8, 39557.5, 46664.8, 50460.3, 51380.5, 56475.2, 67568.0, 72668.6, 74907.3, 80288.4],
              [114.49, 127.54, 78.80, 112.86, 24.30, 80.47, 32.63, 28.85, 20.60, 3.82]),
    "triad": ([31320.0, 34225.7, 40943.6, 44333.4, 45855.2, 50670.0, 61356.9, 65960.8, 68141.3, 72979.8],
              [220.31, 149.15, 97.73, 48.56, 26.45, 19.78, 20.15, 20.13, 17.55, 16.41]),
}
DIMMS, SUBCHANNELS_PER_DIMM, BYTES_PER_SUBCHANNEL = 2, 2, 4      # DDR5: two 32-bit sub-channels per DIMM


def build(mts: int, eff: float) -> Machine:
    m = Machine()
    m.add(Router("pkg"))
    for group, count in (("p", 8), ("e", 8)):                   # core groups; per-core link not identified here
        m.add(Compute(f"{group}cores", clock_hz=0.0, count=count))
        m.connect(Link(f"{group}link", f"{group}cores", "pkg", None, per_unit=True))
    sub = mts * 1e6 * BYTES_PER_SUBCHANNEL
    names = []
    for d in range(DIMMS):
        for c in range(SUBCHANNELS_PER_DIMM):
            n = f"dimm{d}.sub{c}"
            m.add(Storage(n, capacity_b=16 * 2**30 / SUBCHANNELS_PER_DIMM, bandwidth_bps=sub * eff, access_s=80e-9))
            m.connect(Link(f"bus.{n}", n, "pkg", sub, duplex=False))
            names.append(n)
    m.interleave("ram", names)
    return m


def observe(mts: int, eff: float) -> float:
    """All cores streaming from RAM (both core groups at once), GB/s."""
    res = build(mts, eff).solve_reads([Read("p", "pcores", "ram"), Read("e", "ecores", "ram")])
    return sum(v["rate_bps"] for v in res.values()) / GB


def peak_gbs(mts: int) -> float:
    return DIMMS * SUBCHANNELS_PER_DIMM * BYTES_PER_SUBCHANNEL * mts * 1e6 / GB


def fit_at(kernel: str, mts: int) -> float:
    """Efficiency that makes the model reproduce one speed (linear in eff, so a ratio suffices)."""
    i = SPEEDS.index(mts)
    return STREAM[kernel][0][i] / 1000 / observe(mts, 1.0)


def fit_least_squares(kernel: str) -> float:
    vals = [v / 1000 for v in STREAM[kernel][0]]
    model1 = [observe(s, 1.0) for s in SPEEDS]
    return sum(a * b for a, b in zip(vals, model1)) / sum(b * b for b in model1)


def errors(kernel: str, eff: float) -> list[float]:
    return [observe(s, eff) / (v / 1000) - 1 for s, v in zip(SPEEDS, STREAM[kernel][0])]


if __name__ == "__main__":
    for k in STREAM:
        e6 = fit_at(k, 6000); ls = fit_least_squares(k)
        print(f"== STREAM {k}: efficiency fitted at 6000 MT/s {e6:.3f}, least squares over all {ls:.3f}")
        se = [s / v for s, v in zip(STREAM[k][1], STREAM[k][0])]
        print(f"   measured / peak: " + ", ".join(f"{s}: {v / 1000 / peak_gbs(s):.3f}" for s, v in zip(SPEEDS, STREAM[k][0])))
        print(f"   model error, fit at 6000:  " + ", ".join(f"{s}: {100 * x:+.1f}%" for s, x in zip(SPEEDS, errors(k, e6))))
        print(f"   model error, least squares: " + ", ".join(f"{s}: {100 * x:+.1f}%" for s, x in zip(SPEEDS, errors(k, ls))))
        print(f"   standard error of the measurement: {100 * min(se):.2f}–{100 * max(se):.2f} %")
