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
    reads, how = mlc.bandwidth_bps(s.m, s.groups, "all_reads")
    mixed, _ = mlc.bandwidth_bps(s.m, s.groups, "1to1_rw")
    assert mixed < reads and "512 lines in flight" in how            # each step turns the DDR5 bus once (JEDEC)
    from app.hw import dram
    w = dram.write_timing("ddr5", 5600, None)
    assert w["cwl"] == w["cl"] - 2 and 25 < w["turnaround_ns"] < 40
    assert s.m.nodes["p"].outstanding == 48 and s.m.nodes["e"].outstanding == 16
    assert 70e-9 < mlc.idle_latency_s(s.m, s.groups[0]) < 100e-9


def storage_machine(**kw):
    m = Machine()
    m.add(Router("r"))
    m.add(Storage("disk", capacity_b=1e12, bandwidth_bps=None, access_s=None, **kw))
    m.connect(Link("bus", "disk", "r", None))
    m.add(Compute("c", clock_hz=1e9, count=1))
    m.connect(Link("c.link", "c", "r", None))
    return m


def test_writing_slower_than_reading_shares_the_device_at_its_own_speeds():
    m = storage_machine(read_bps=100e6, write_bps=50e6)
    assert solve_step(m, Step("w", [Transfer("w", "c", "disk", 50e6)])).seconds == pytest.approx(1.0)
    both = solve_step(m, Step("rw", [Transfer("r", "disk", "c", 100e6), Transfer("w", "c", "disk", 50e6)]))
    assert both.seconds == pytest.approx(2.0)                   # 1 s of reading + 1 s of writing on one device


def test_small_mixed_steps_pay_the_switch_every_step_large_ones_once():
    m = storage_machine(read_bps=1e9, write_bps=1e9, turnaround_s=1e-6)
    small = Step("3r1w", [Transfer("r", "disk", "c", 3 * 64e3), Transfer("w", "c", "disk", 64e3)])
    t_small = 1000 * solve_step(m, small).seconds                              # the engine's 1000 small steps
    big = Step("big", [Transfer("r", "disk", "c", 1000 * 3 * 64e3), Transfer("w", "c", "disk", 1000 * 64e3)])
    t_big = solve_step(m, big).seconds
    data = 1000 * 4 * 64e3 / 1e9
    assert t_big == pytest.approx(data + 1e-6) and t_small == pytest.approx(data + 1000 * 1e-6)
    only_reads = Step("r", [Transfer("r", "disk", "c", 64e3)])
    assert solve_step(m, only_reads).seconds == pytest.approx(64e3 / 1e9)       # no switch without a write


def test_seek_once_sequential_per_request_random():
    m = storage_machine(read_bps=200e6, write_bps=200e6, read_access_s=0.0, seek_s=8e-3)   # an HDD
    seq = solve_step(m, Step("seq", [Transfer("r", "disk", "c", 400e6)])).seconds
    rnd = solve_step(m, Step("rnd", [Transfer("r", "disk", "c", 400e6, pattern="random", request_bytes=4096)])).seconds
    assert seq == pytest.approx(2.0 + 8e-3)                     # positioned once, then the data
    n = 400e6 / 4096
    assert rnd == pytest.approx(2.0 + n * 8e-3)                 # positioned for every request


def test_storage_shorthand_keeps_everything_else_unchanged():
    s = Storage("x", 1, bandwidth_bps=10.0, access_s=1.0, write_bps=5.0)
    assert (s.bps("read"), s.bps("write"), s.access("write")) == (10.0, 5.0, 1.0)


def test_likwid_engine_threads_and_directions(tmp_path):
    from app.engine import likwid
    from app.system import build_system
    from tests.test_system import machine_run
    s = build_system(machine_run(tmp_path))
    assert likwid.placement(s.m, s.groups, 10) == {"p": 8, "e": 2} and likwid.placement(s.m, s.groups, 40) == {"p": 8, "e": 8}
    one, how = likwid.bandwidth_bps(s.m, s.groups, "load_avx", 1)
    eight, _ = likwid.bandwidth_bps(s.m, s.groups, "load_avx", 8)
    assert one < eight and "1 threads on 1 p" in how and "3 KiB in flight" in how      # 48 lines of one P-core
    nt, _ = likwid.bandwidth_bps(s.m, s.groups, "store_mem_avx", 8)
    assert nt > 0


def cache_machine(write_allocate=True):
    from app.hw.primitives import Cache
    m = Machine()
    m.add(Router("r"))
    m.add(Storage("mem", capacity_b=1e12, bandwidth_bps=100e9, access_s=None))
    m.connect(Link("bus", "mem", "r", None))
    m.add(Cache("l3", capacity_b=32e6, bandwidth_bps=1000e9, access_s=None, backing="mem",
                write_allocate=write_allocate, write_back=True))
    m.connect(Link("l3.port", "l3", "r", None))
    m.add(Compute("c", clock_hz=1e9, count=8))
    m.connect(Link("c.link", "c", "l3", None))
    return m


def test_cache_hits_are_stated_misses_and_writes_follow_the_policy():
    m = cache_machine()
    read = lambda h: solve_step(m, Step("r", [Transfer("x", "mem", "c", 10e9, hits=h)])).seconds
    assert read(0.0) == pytest.approx(0.1) and read(1.0) == pytest.approx(0.01) and read(0.5) == pytest.approx(0.05)
    # an ordinary write miss: the line is read first (write-allocate), then written back — twice the memory traffic
    assert solve_step(m, Step("w", [Transfer("w", "c", "mem", 10e9)])).seconds == pytest.approx(0.2)
    # streaming stores bypass the cache: once
    assert solve_step(m, Step("nt", [Transfer("w", "c", "mem", 10e9, non_temporal=True)])).seconds == pytest.approx(0.1)
    m2 = cache_machine(write_allocate=False)
    assert solve_step(m2, Step("w", [Transfer("w", "c", "mem", 10e9)])).seconds == pytest.approx(0.1)


def test_l3_in_the_cpu_graph(tmp_path):
    from app.system import build_system
    from tests.test_system import machine_run
    run = machine_run(tmp_path)
    (run / "inventory" / "lscpu.txt").write_text("L3 cache:                                30 MiB (1 instance)\n")
    s = build_system(run)
    l3 = s.m.nodes["l3"]
    assert l3.capacity_b == 30 * 2**20 and l3.write_allocate and l3.write_back and l3.backing == "ram"
    assert [n for _, n in s.m.path("p", "pkg")] == ["l3", "pkg"] and "L3 30 MiB" in s.labels["l3"]


def budget_machine(watts=None, energy=None):
    from app.hw.primitives import PowerBudget
    m = Machine()
    if watts is not None:
        m.add(PowerBudget("pkg.power", watts))
    m.add(Compute("core", clock_hz=4e9, count=6, ops_per_clock=dict(fma=2, shuffle=1, vec_alu=3),
                  issue_per_clock=4, energy_j=energy or {}, budget="pkg.power" if watts is not None else None))
    return m


def test_cpu_budgets_issue_ports_and_power():
    m = budget_machine()
    one = lambda mix, units, n=1e9: solve_step(m, Step("k", [Work("w", "core", "", n, mix=mix, units=units)])).seconds
    assert one({"fma": 1}, 1) == pytest.approx(1e9 / (4e9 * 2))                 # one core, port-bound: 2 FMA/clock
    assert one({"shuffle": 27, "vec_alu": 28}, 1) == pytest.approx(1e9 * 27 / 4e9)    # the single shuffle port binds
    assert one({"vec_alu": 3, "fma": 2, "shuffle": 1}, 1) == pytest.approx(1e9 * 6 / (4 * 4e9))   # issue width binds
    assert one({"fma": 1}, 6) == pytest.approx(1e9 / (6 * 4e9 * 2))              # six cores, no power budget
    p = budget_machine(watts=60.0, energy={"fma": 2e-9})
    six = solve_step(p, Step("k", [Work("w", "core", "", 1e10, mix={"fma": 1}, units=6)]))
    assert six.seconds == pytest.approx(1e10 * 2e-9 / 60.0) and six.bottleneck["w"] == "power:pkg.power"
    single = solve_step(p, Step("k", [Work("w", "core", "", 1e9, mix={"fma": 1}, units=1)]))
    assert single.bottleneck["w"] == "fma:core#w"                               # one core alone: its clock binds


def test_cpu_budgets_on_a_published_machine():
    """i7-8700: one core FMA-bound at its 4.6 GHz turbo; six cores throttled by the 65 W budget (calibrated)."""
    import glob
    from pathlib import Path
    from app.system import build_system
    runs = glob.glob("/home/claude/bench-repo/results/runs/*e963")
    if not runs:
        pytest.skip("published run not available")
    s = build_system(Path(runs[0]))
    sgemm = {"fma": 1, "load": 0.5}
    gflops = lambda units: 16e9 / solve_step(s.m, Step("k", [Work("w", "core", "", 1e9, mix=sgemm, units=units)])).seconds / 1e9
    assert gflops(1) == pytest.approx(4.6 * 32, rel=1e-6)                     # 147 GFLOP/s (137 measured)
    assert 450 < gflops(6) < 560                                              # power-bound (471 measured)
    assert s.m.nodes["pkg.power"].watts == 65.0
