"""Yardstick 1: Intel MLC v3.10 on dual-socket Xeon nodes, VSC Leuven (docs/overhaul/yardsticks.md).

Configuration from the source unless marked INFERRED. Free parameters (4 per node) are fitted through the model;
everything else is validation. Per-core link bandwidth and router capacity are not identifiable from these runs
(every bandwidth test uses all cores) and stay None.
"""
from __future__ import annotations

from ..machine import Machine, Read
from ..primitives import Compute, Link, Router, Storage

GB = 1e9          # MLC reports MB/s with 1 MB = 1e6 bytes
NS = 1e-9

NODES = {
    "skylake": dict(cpu="2 x Xeon Gold 6140", cores=18, ram_gib=192,
                    channels=6, mts=2666, cl=19,          # INFERRED: 1 DIMM per channel at the CPU's max speed, JEDEC RDIMM CL
                    socket_links=3, socket_gts=10.4,       # spec sheet: UPI links per CPU and rate; board wiring unknown
                    measured=dict(lat_local_ns=(90.2, 89.1), lat_remote_ns=(144.3, 143.2),
                                  bw_local_gbs=(104.3622, 106.0284), bw_remote_gbs=(34.2906, 34.2833),
                                  bw_system_gbs=211.1942)),
    "cascadelake": dict(cpu="2 x Xeon Gold 6240", cores=18, ram_gib=192, channels=6, mts=2933, cl=21,
                        socket_links=3, socket_gts=10.4,
                        measured=dict(lat_local_ns=(80.8, 81.5), lat_remote_ns=(140.4, 141.6),
                                      bw_local_gbs=(113.0737, 112.8107), bw_remote_gbs=(34.3055, 34.3222),
                                      bw_system_gbs=225.6308)),
    "icelake": dict(cpu="2 x Xeon Platinum 8360Y", cores=36, ram_gib=256, channels=8, mts=3200, cl=22,
                    socket_links=3, socket_gts=11.2,
                    measured=dict(lat_local_ns=(89.2, 89.0), lat_remote_ns=(142.7, 142.9),
                                  bw_local_gbs=(175.6909, 175.9126), bw_remote_gbs=(55.6768, 55.6923),
                                  bw_system_gbs=351.7621)),
}
SOURCE = "docs.vscentrum.be: Memory bandwidth and latency measurements (Genius, wICE), MLC v3.10"


def cas_s(n):
    return n["cl"] * 2 / (n["mts"] * 1e6)                    # CL cycles at the memory clock (MT/s / 2)


def channel_peak_bps(n):
    return n["channels"] * n["mts"] * 1e6 * 8


def socket_peak_bps(n):
    return n["socket_links"] * n["socket_gts"] * 1e9 * 2       # per direction: 2 bytes per transfer per link


def build(n, fixed_s, socket_latency_s, ram_eff, socket_eff) -> Machine:
    m = Machine()
    for s in (0, 1):
        m.add(Router(f"s{s}.router"))
        m.add(Compute(f"s{s}.cores", clock_hz=0.0, count=n["cores"]))
        m.add(Storage(f"s{s}.ram", capacity_b=n["ram_gib"] / 2 * 2**30, bandwidth_bps=channel_peak_bps(n) * ram_eff,
                      access_s=cas_s(n) + fixed_s))
        m.connect(Link(f"s{s}.corelink", f"s{s}.cores", f"s{s}.router", None, per_unit=True))   # not identifiable here
        m.connect(Link(f"s{s}.channels", f"s{s}.ram", f"s{s}.router", channel_peak_bps(n), duplex=False))
    m.connect(Link("socket", "s0.router", "s1.router", socket_peak_bps(n) * socket_eff, latency_s=socket_latency_s))
    return m


def observe(m: Machine) -> dict:
    """What MLC reports, computed by the model."""
    one = lambda reads: m.solve_reads(reads)
    local = one([Read("x", "s0.cores", "s0.ram")])["x"]["rate_bps"]
    remote = one([Read("x", "s0.cores", "s1.ram")])["x"]["rate_bps"]
    system = sum(v["rate_bps"] for v in one([Read("a", "s0.cores", "s0.ram"), Read("b", "s1.cores", "s1.ram")]).values())
    return dict(lat_local_ns=m.idle_latency_s("s0.cores", "s0.ram") / NS,
                lat_remote_ns=m.idle_latency_s("s0.cores", "s1.ram") / NS,
                bw_local_gbs=local / GB, bw_remote_gbs=remote / GB, bw_system_gbs=system / GB)


def _bisect(f, target, lo, hi, n=80):
    """Monotonic increasing f: find x in [lo, hi] with f(x) = target."""
    for _ in range(n):
        mid = (lo + hi) / 2
        if f(mid) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def mean(t):
    return sum(t) / len(t)


def fit(name: str) -> dict:
    """The four free parameters, each fitted through the model to one measured value (means of the two sockets)."""
    n, meas = NODES[name], NODES[name]["measured"]
    p = dict(fixed_s=60e-9, socket_latency_s=50e-9, ram_eff=0.8, socket_eff=0.8)
    obs = lambda **kw: observe(build(n, **{**p, **kw}))
    p["fixed_s"] = _bisect(lambda x: obs(fixed_s=x)["lat_local_ns"], mean(meas["lat_local_ns"]), 0, 300e-9)
    p["socket_latency_s"] = _bisect(lambda x: obs(socket_latency_s=x)["lat_remote_ns"], mean(meas["lat_remote_ns"]), 0, 300e-9)
    p["ram_eff"] = _bisect(lambda x: obs(ram_eff=x)["bw_local_gbs"], mean(meas["bw_local_gbs"]), 0.01, 1.0)
    p["socket_eff"] = _bisect(lambda x: obs(socket_eff=x)["bw_remote_gbs"], mean(meas["bw_remote_gbs"]), 0.01, 1.0)
    return p


def report() -> list[dict]:
    rows = []
    for name, n in NODES.items():
        p = fit(name)
        o = observe(build(n, **p))
        rows.append(dict(node=name, params=p, model=o, measured=n["measured"]))
    return rows


if __name__ == "__main__":
    for r in report():
        p, o, meas = r["params"], r["model"], r["measured"]
        print(f"== {r['node']}: fixed {p['fixed_s']/NS:.1f} ns, socket link {p['socket_latency_s']/NS:.1f} ns, "
              f"RAM eff {p['ram_eff']:.3f}, socket-link eff {p['socket_eff']:.3f}")
        sysm = meas["bw_system_gbs"]
        print(f"   VALIDATION system read bandwidth: model {o['bw_system_gbs']:.1f} vs measured {sysm:.1f} GB/s "
              f"({100 * (o['bw_system_gbs'] / sysm - 1):+.1f} %)")
