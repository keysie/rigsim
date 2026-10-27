"""GPU database (low-level parts) against the spec table — validation only, no fitting."""
from app.hw.yardsticks import gpudb
from app.hw.yardsticks.validate_gpudb import run


def test_only_dense_results_are_compared():
    db = {e["id"]: e for e in gpudb.load()["entries"]}
    assert all(gpudb.val(db[p["id"]]["operation"]["sparse"]) is False for p in gpudb.tensor_points())


def test_per_sm_tensor_rates_match_the_spec_table():
    """RTX 3070 Ti (consumer Ampere, same SM as the 3090 and 3060), instruction-level mma microbenchmarks: every
    dense precision reaches 98-100 % of the per-SM rate in the spec table."""
    rows = [r for r in run() if r["kind"] == "per_sm_clock"]
    assert len(rows) >= 10 and {r["op_class"] for r in rows} >= {"fp16_tensor", "fp16_tensor_fp32acc", "tf32_tensor",
                                                                 "int8_tensor", "int4_tensor"}
    assert all(0.975 < r["ratio"] <= 1.001 for r in rows)


def test_finding_4090_runs_above_its_boost_clock():
    """Finding, not a target: whole-GPU throughput is 7.7-8.3 % above SMs x boost clock x per-SM rate, the same at
    every precision — the per-SM rates hold, the sustained clock under load (~2.73 GHz) exceeds the spec boost
    (2.52 GHz). Open: where the clock at full power limit comes from (spec boost or a measured value)."""
    rows = [r for r in run() if r["model"] == "rtx-4090"]
    ratios = [r["ratio"] for r in rows]
    assert len(rows) >= 4 and all(1.07 < x < 1.09 for x in ratios) and max(ratios) - min(ratios) < 0.01


def test_trust_rule():
    t = {e["id"]: gpudb.trust(e) for e in gpudb.load()["entries"]}
    assert t["mma-sun-rtx3070ti-d-fp16fp16m16n8k16"] == "medium" and t["gemm-nvf-01"] == "low"
