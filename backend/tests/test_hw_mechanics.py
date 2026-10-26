"""Hardware layer mechanics, checked against cases computed by hand."""
from app.hw import Compute, Link, Machine, Read, Router, Storage, max_min_fair_share

GB = 1e9


def test_fair_share_equal_split_and_freed_capacity():
    # two flows share a 10 GB/s link; one is capped at 2 -> the other gets the rest (8), not half
    rates, why = max_min_fair_share({"l": 10 * GB}, {"a": (1, ["l"], 2 * GB), "b": (1, ["l"], None)})
    assert abs(rates["a"] - 2 * GB) < 1 and abs(rates["b"] - 8 * GB) < 1
    assert why["a"] == "own cap" and why["b"] == "l"


def test_fair_share_is_weighted_per_unit():
    # a group of 3 units and a single unit share 8 GB/s: 6 and 2
    rates, _ = max_min_fair_share({"l": 8 * GB}, {"g": (3, ["l"], None), "s": (1, ["l"], None)})
    assert abs(rates["g"] - 6 * GB) < 1 and abs(rates["s"] - 2 * GB) < 1


def two_socket(per_core_bps=None):
    m = Machine()
    for s in (0, 1):
        m.add(Router(f"s{s}.router"))
        m.add(Compute(f"s{s}.cores", clock_hz=2e9, count=4))
        m.add(Storage(f"s{s}.ram", capacity_b=64 * GB, bandwidth_bps=40 * GB, access_s=80e-9))
        m.connect(Link(f"s{s}.corelink", f"s{s}.cores", f"s{s}.router", per_core_bps, per_unit=True))
        m.connect(Link(f"s{s}.channels", f"s{s}.ram", f"s{s}.router", 50 * GB, duplex=False))
    m.connect(Link("socket", "s0.router", "s1.router", 20 * GB, latency_s=60e-9, duplex=True))
    return m


def test_latency_adds_along_the_path():
    m = two_socket()
    assert abs(m.idle_latency_s("s0.cores", "s0.ram") - 80e-9) < 1e-15
    assert abs(m.idle_latency_s("s0.cores", "s1.ram") - 140e-9) < 1e-15


def test_saturation_emerges_from_per_core_links():
    # 4 cores x 6 GB/s = 24 < 40: core-bound; 4 x 15 = 60 > 40: RAM-bound
    r = two_socket(6 * GB).solve_reads([Read("x", "s0.cores", "s0.ram")])["x"]
    assert abs(r["rate_bps"] - 24 * GB) < 1 and r["bottleneck"] == "own cap"
    r = two_socket(15 * GB).solve_reads([Read("x", "s0.cores", "s0.ram")])["x"]
    assert abs(r["rate_bps"] - 40 * GB) < 1 and r["bottleneck"] == "storage:s0.ram"
    r = two_socket(6 * GB).solve_reads([Read("x", "s0.cores", "s0.ram", units=2)])["x"]
    assert abs(r["rate_bps"] - 12 * GB) < 1


def test_duplex_directions_are_independent_and_remote_reads_share_the_socket_link():
    m = two_socket(15 * GB)
    both = m.solve_reads([Read("a", "s0.cores", "s1.ram"), Read("b", "s1.cores", "s0.ram")])
    assert all(abs(v["rate_bps"] - 20 * GB) < 1 for v in both.values())          # one direction each
    same = m.solve_reads([Read("a", "s0.cores", "s1.ram", units=2), Read("b", "s0.cores", "s1.ram", units=2)])
    assert abs(sum(v["rate_bps"] for v in same.values()) - 20 * GB) < 1         # same direction: shared


def test_unidentified_parameters_are_reported():
    r = two_socket(None).solve_reads([Read("x", "s0.cores", "s0.ram")])["x"]
    assert "s0.corelink" in r["unidentified_on_path"] and "s0.router" in r["unidentified_on_path"]
