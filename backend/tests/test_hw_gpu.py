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
    """12 controllers x 32 bits x 19.5 Gbps = 936.0 GB/s (the vendor quotes 936.2), spread over every slice."""
    r = solve_transfers(box(), [Transfer("x", "gpu1.vram", "gpu1.sm")])["x"]
    assert abs(r["rate_bps"] - 936.0 * GB) < 1e3
    assert r["bottleneck"].startswith(("storage:gpu1.vram", "link:gpu1.mc"))


def test_vram_is_one_slice_per_documented_controller():
    m = box()
    assert len(m.interleaved["gpu1.vram"]) == 12 and len(m.interleaved["gpu0.vram"]) == 6   # GA102 / GA106
    assert abs(m.links["gpu0.mc0"].bandwidth_bps - 4 * 15e9) < 1                            # 32 bits x 15 Gbps
    assert m.nodes["gpu1.l2"].capacity_b == 6 * 2**20 and m.nodes["gpu1.l2"].bandwidth_bps is None


def test_clock_comes_from_the_state_load_class_curve():
    from app.hw.gpu import set_load_class
    curves = {("sm_clock", "compute"): PowerCurve.from_csv("power_w,value\n220,1250\n300,1670", source="t"),
              ("sm_clock", "memory"): PowerCurve.from_csv("power_w,value\n220,1500\n300,1770", source="t")}
    m = Machine(); m.add(Router("cpu"))
    add_gpu(m, "g", "rtx-3090", "cpu", slot_gen=4, slot_lanes=16, power_limit_w=260, curves=curves)
    st = set_load_class(m, "g", "compute")
    assert abs(m.nodes["g.sm"].clock_hz - 1460e6) < 1e3 and st["sm_clock_basis"].startswith("measured")
    set_load_class(m, "g", "memory")
    assert abs(m.nodes["g.sm"].clock_hz - 1635e6) < 1e3
    assert abs(st["vram_bps"] - 936.0 * GB) < 1e3 and "unmeasured" in st["vram_basis"]   # no VRAM curve: spec


def test_clock_fallbacks_are_labelled():
    from app.hw.gpu import set_load_class
    m = Machine(); m.add(Router("cpu"))
    add_gpu(m, "a", "rtx-3090", "cpu", slot_gen=4, slot_lanes=16)
    assert set_load_class(m, "a", "compute")["sm_clock_basis"].startswith("spec-sheet law")
    from app.hw import constants as K                                    # at TDP: sustained factor x spec boost
    assert abs(m.nodes["a.sm"].clock_hz - K.SUSTAINED_CLOCK_FACTOR * 1695e6) < 1
    one = {("sm_clock", "compute"): PowerCurve(points=[(450, 2733)], basis="inferred", source="t")}
    add_gpu(m, "b", "rtx-4090", "cpu", slot_gen=4, slot_lanes=16, power_limit_w=300, curves=one)
    st = set_load_class(m, "b", "compute")
    assert "extrapolated" in st["sm_clock_basis"] and m.nodes["b.sm"].clock_hz == 2733e6
    assert "other load class" in set_load_class(m, "b", "memory")["sm_clock_basis"]


def test_spec_sheet_clock_law():
    from app.hw import constants as K
    from app.hw.gpus import GPUS
    from app.hw.power import spec_clock_mhz
    g = GPUS["rtx-3090"]
    full = spec_clock_mhz(g, 350, "compute")
    assert abs(full - K.SUSTAINED_CLOCK_FACTOR * 1695) < 1e-9
    assert abs(spec_clock_mhz(g, 350 / 8, "compute") - full / 2) < 1e-9        # (1/8)^(1/3) = 1/2
    assert abs(spec_clock_mhz(g, 250, "memory") / spec_clock_mhz(g, 250, "compute") - K.MEMORY_LOAD_CLOCK_GAIN) < 1e-12


def test_load_class_from_the_operation_mix():
    from app.hw.loadclass import classify
    big_gemm = ("fp16_tensor", 2 * 8192 ** 3, 3 * 8192 ** 2 * 2)          # high arithmetic intensity
    gemv = ("fp16_tensor", 2 * 8192 ** 2, 8192 ** 2 * 2)                   # 1 op per byte
    assert classify("rtx-3090", "cuda-12.8", [big_gemm])[0] == "compute"
    assert classify("rtx-3090", "cuda-12.8", [gemv] * 10)[0] == "memory"
    cls, share = classify("rtx-3090", "cuda-12.8", [big_gemm, gemv])
    assert cls == "compute" and 0.5 < share < 1.0


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
