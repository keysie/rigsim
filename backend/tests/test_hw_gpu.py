"""GPU layer mechanics, checked against values computed by hand."""
import pytest

from app.hw import Compute, Link, Machine, Router, Storage, Transfer, solve_transfers
from app.hw import rates
from app.hw.gpu import add_gpu
from app.hw.links import pcie_bps
from app.hw.power import PowerCurve

GB = 1e9


def test_pcie_capacity_per_generation():
    assert abs(pcie_bps(3, 16) / GB - 15.754) < 0.01          # 8 GT/s x 128/130 / 8 x 16
    assert abs(pcie_bps(4, 16) / GB - 31.508) < 0.01
    assert abs(pcie_bps(3, 4) / GB - 3.938) < 0.01


def box():
    """The reference box's shape: X299 CPU (48 lanes Gen3), two x16 slots on the CPU, one x4 slot behind the
    chipset, whose uplink (DMI 3.0) is x4 Gen3; 4 channels DDR4-3200; an NVMe on the chipset too."""
    m = Machine()
    m.add(Router("cpu")); m.add(Router("pch"))
    m.add(Compute("cores", clock_hz=3.7e9, count=10))
    m.connect(Link("corelink", "cores", "cpu", None, per_unit=True))
    for c in range(4):
        m.add(Storage(f"ch{c}", capacity_b=32 * GB, bandwidth_bps=25.6 * GB * 0.85, access_s=80e-9))
        m.connect(Link(f"bus{c}", f"ch{c}", "cpu", 25.6 * GB, duplex=False))
    m.interleave("ram", [f"ch{c}" for c in range(4)])
    m.connect(Link("dmi", "pch", "cpu", pcie_bps(3, 4)))
    m.add(Storage("nvme", capacity_b=2e12, bandwidth_bps=3.5 * GB, access_s=80e-6))
    m.connect(Link("nvme.link", "nvme", "pch", pcie_bps(3, 4)))
    g1 = add_gpu(m, "gpu1", "rtx-3090", "cpu", slot_gen=3, slot_lanes=16, power_limit_w=250)
    g2 = add_gpu(m, "gpu2", "rtx-3090", "cpu", slot_gen=3, slot_lanes=16, power_limit_w=250)
    g0 = add_gpu(m, "gpu0", "rtx-3060-12g", "pch", slot_gen=3, slot_lanes=4)
    return m


def test_host_to_card_copy_runs_at_the_negotiated_link():
    r = solve_transfers(box(), [Transfer("x", "ram", "gpu1.vram")])["x"]
    assert abs(r["rate_bps"] - pcie_bps(3, 16)) < 1 and r["bottleneck"].startswith("link:gpu1.pcie")


def test_two_cards_on_cpu_lanes_copy_independently():
    res = solve_transfers(box(), [Transfer("a", "ram", "gpu1.vram"), Transfer("b", "ram", "gpu2.vram")])
    assert all(abs(v["rate_bps"] - pcie_bps(3, 16)) < 1 for v in res.values())


def test_chipset_card_shares_the_uplink_with_the_nvme_in_the_same_direction():
    alone = solve_transfers(box(), [Transfer("x", "ram", "gpu0.vram")])["x"]["rate_bps"]
    assert abs(alone - pcie_bps(3, 4)) < 1
    same = solve_transfers(box(), [Transfer("x", "gpu0.vram", "ram"), Transfer("y", "nvme", "ram")])   # both upward
    assert abs(same["x"]["rate_bps"] + same["y"]["rate_bps"] - pcie_bps(3, 4)) < 1
    assert abs(same["x"]["rate_bps"] - same["y"]["rate_bps"]) < 1
    opposite = solve_transfers(box(), [Transfer("x", "ram", "gpu0.vram"), Transfer("y", "nvme", "ram")])
    assert abs(opposite["x"]["rate_bps"] - pcie_bps(3, 4)) < 1 and abs(opposite["y"]["rate_bps"] - 3.5 * GB) < 1   # duplex


def test_card_to_card_goes_through_the_cpu_router():
    r = solve_transfers(box(), [Transfer("x", "gpu1.vram", "gpu2.vram")])["x"]
    assert abs(r["rate_bps"] - pcie_bps(3, 16)) < 1


def test_sms_read_vram_at_its_bandwidth():
    r = solve_transfers(box(), [Transfer("x", "gpu1.vram", "gpu1.sm")])["x"]
    assert abs(r["rate_bps"] - 936.2 * GB) < 1e3 and r["bottleneck"] == "storage:gpu1.vram"


def test_power_curve_applies_per_primitive():
    csv = "power_w,value\n220,1605\n250,1740\n300,1830\n350,1860\n"          # an SM-clock curve (made up for the test)
    sm = PowerCurve.from_csv(csv, tdp_w=350)
    m = Machine(); m.add(Router("cpu"))
    dev = add_gpu(m, "g", "rtx-3090", "cpu", slot_gen=4, slot_lanes=16, power_limit_w=250, curves={"sm_clock": sm})
    assert abs(m.nodes["g.sm"].clock_hz - 1695e6 * 1740 / 1860) < 1e3
    assert m.nodes["g.vram"].bandwidth_bps == 936.2 * GB                     # no VRAM curve: flat, unmeasured
    assert dev.unmeasured() == ["vram_bandwidth"]


def test_router_cpu_cycles_compete():
    """A copy path that needs CPU cycles (bounce buffer): 8 cores x 3 GHz at 2 cycles/byte = 12 GB/s for all
    copies through it together."""
    m = Machine()
    m.add(Compute("cores", clock_hz=3e9, count=8))
    m.add(Router("bounce", cpu="cores", cycles_per_byte=2.0)); m.add(Router("host"))
    m.add(Storage("a", capacity_b=1e12, bandwidth_bps=100 * GB, access_s=1e-6))
    m.add(Storage("b", capacity_b=1e12, bandwidth_bps=100 * GB, access_s=1e-6))
    m.add(Storage("c", capacity_b=1e12, bandwidth_bps=100 * GB, access_s=1e-6))
    m.connect(Link("la", "a", "host", 50 * GB)); m.connect(Link("hb", "host", "bounce", 50 * GB))
    m.connect(Link("lb", "bounce", "b", 50 * GB)); m.connect(Link("lc", "bounce", "c", 50 * GB))
    one = solve_transfers(m, [Transfer("x", "a", "b")])["x"]
    two = solve_transfers(m, [Transfer("x", "a", "b"), Transfer("y", "a", "c")])
    assert abs(one["rate_bps"] - 12 * GB) < 1 and one["bottleneck"] == "cycles:cores"
    assert abs(sum(v["rate_bps"] for v in two.values()) - 12 * GB) < 1


def test_achievable_rate_provenance():
    clock = 1695e6
    rate, how = rates.achievable("rtx-3090", "cuda-12.8", "int8_tensor", clock)
    assert how.startswith("unmeasured") and abs(rate - 2048 * 82 * clock * rates.DEFAULT_EFFICIENCY) < 1
    rates.MEASUREMENTS.append(rates.Measured("rtx-3090", "cuda-12.8", "int8_tensor", 0.8, "test"))
    try:
        assert rates.achievable("rtx-3090", "cuda-12.8", "int8_tensor", clock)[1].startswith("measured")
        rate60, how60 = rates.achievable("rtx-3060-12g", "cuda-12.8", "int8_tensor", 1777e6)
        assert how60.startswith("inherited") and abs(rate60 - 2048 * 28 * 1777e6 * 0.8) < 1
        assert rates.achievable("rtx-4090", "cuda-12.8", "int8_tensor", 2520e6)[1].startswith("unmeasured")
    finally:
        rates.MEASUREMENTS.clear()


def test_roofline_op_time():
    # 1 GFLOP at 100 TFLOP/s = 10 us; 1 GB at 1 TB/s = 1 ms -> memory-bound, plus 5 us launch
    assert abs(rates.op_time_s(1e9, 100e12, 1e9, 1e12, 5e-6) - 1.005e-3) < 1e-12


def test_unidentified_vram_latency_is_reported():
    m = box()
    with pytest.raises(ValueError):
        m.idle_latency_s("gpu1.sm", "gpu1.vram")
