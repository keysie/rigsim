"""Steps of activities on the graph: concurrent flows share, finishing ones free capacity, work and fixed costs."""
import pytest

from app.hw.activity import Fixed, Step, Transfer, Work, solve_step
from app.hw.machine import Machine
from app.hw.primitives import Compute, Link, Router, Storage


def small():
    m = Machine()
    m.add(Router("r"))
    m.add(Storage("mem", capacity_b=1e12, bandwidth_bps=100e9, access_s=None))
    m.connect(Link("bus", "mem", "r", 100e9, duplex=False))
    for c in ("a", "b"):
        m.add(Compute(c, clock_hz=1e9, ops_per_clock={"fp32": 10}, count=4))
        m.connect(Link(f"{c}.link", c, "r", None))
    return m


def test_concurrent_transfers_share_and_finishing_frees_capacity():
    m = small()
    r = solve_step(m, Step("two reads", [Transfer("x", "mem", "a", 10e9), Transfer("y", "mem", "b", 30e9)]))
    # both at 50 GB/s until x is done (0.2 s), then y alone at 100 GB/s for its remaining 20 GB (0.2 s)
    assert r.finish_s["x"] == pytest.approx(0.2) and r.finish_s["y"] == pytest.approx(0.4)
    assert r.alone_s["y"] == pytest.approx(0.3) and r.seconds == pytest.approx(0.4)


def test_work_and_transfer_overlap_fixed_cost_first():
    m = small()
    step = Step("kernel", [Work("w", "a", "fp32", 20e9, efficiency=0.5), Transfer("t", "mem", "a", 10e9)],
                [Fixed("launch", "a", 0.01)])
    r = solve_step(m, step)
    assert r.alone_s["w"] == pytest.approx(1.0) and r.alone_s["t"] == pytest.approx(0.1)   # 40 Gops/s x 0.5
    assert r.seconds == pytest.approx(1.01) and r.fixed_s == pytest.approx(0.01)


def test_two_works_on_one_node_share_its_units():
    m = small()
    r = solve_step(m, Step("two", [Work("w1", "a", "fp32", 20e9), Work("w2", "a", "fp32", 20e9)]))
    assert r.seconds == pytest.approx(1.0)                                              # 40 Gops/s shared


def test_cpu_cores_and_igpu_share_the_system_memory(tmp_path):
    from app.system import build_system
    from tests.test_system import machine_run
    s = build_system(machine_run(tmp_path))
    gb = 8e9
    alone_cpu = solve_step(s.m, Step("cpu", [Transfer(g, "ram", g, gb) for g in s.groups]))
    both = solve_step(s.m, Step("cpu+igpu", [Transfer(g, "ram", g, gb) for g in s.groups]
                                 + [Transfer("igpu", "ram", "igpu.sm", gb)]))
    assert both.finish_s[s.groups[0]] > alone_cpu.finish_s[s.groups[0]]                # the iGPU takes its share


def test_latency_and_writes():
    m = small()
    m.nodes["mem"].access_s = 80e-9
    m.links["bus"].latency_s = 5e-9
    r = solve_step(m, Step("read", [Transfer("x", "mem", "a", 64, units=1)]))
    assert r.seconds == pytest.approx(85e-9 + 64 / 100e9)        # the hardware's latency, then the line — always
    both = solve_step(m, Step("rw", [Transfer("r", "mem", "a", 10e9), Transfer("w", "a", "mem", 10e9)]))
    assert both.seconds == pytest.approx(0.2)                                     # one simplex bus: shared


def test_mlc_engine_on_the_machine(tmp_path):
    from app.engine import mlc
    from app.system import build_system
    from tests.test_system import machine_run
    s = build_system(machine_run(tmp_path))
    reads = mlc.bandwidth_bps(s.m, s.groups, "all_reads")
    assert reads == pytest.approx(mlc.bandwidth_bps(s.m, s.groups, "1to1_rw"))      # no turnaround in the graph yet
    assert 70e-9 < mlc.idle_latency_s(s.m, s.groups[0]) < 100e-9
