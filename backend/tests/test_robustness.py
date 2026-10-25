"""Regression tests from exploratory testing in a real browser: inputs that used to hang the tab, blank every
panel or crash the simulation. Each case below was reproduced before it was fixed."""
import copy
import glob
import json
import time

import pytest

from app import api, perf
from app.schemas import EngineSpec

BOX = {"os": "linux", "cpus": [{"cpu_id": "i9-10900x", "enabled": True, "ram_gb": 128, "ram_mts": 3200, "ram_channels": 4}],
       "devices": [{"id": "gpu-1", "kind": "gpu", "model_id": "rtx-3090", "link_gen": 3, "link_width": 16, "power_w": 250},
                   {"id": "gpu-2", "kind": "gpu", "model_id": "rtx-3090", "link_gen": 3, "link_width": 16, "power_w": 250},
                   {"id": "gpu-3", "kind": "gpu", "model_id": "rtx-3090", "link_gen": 3, "link_width": 8, "power_w": 250}], "nvlinks": []}
STRATA = {"id": "s", "engine": "strata", "version": "0.1.43.1", "model_id": "qwen3.8-flash-next", "gpus": ["gpu-1", "gpu-2"], "cpu_cores": 6, "args": {"batch": 4}}
VLLM = {"id": "v", "engine": "vllm", "version": "0.31.0", "model_id": "qwen3.8-27b", "gpus": ["gpu-3"], "cpu_cores": 1,
        "args": {"tensor_parallel": "1", "max_num_seqs": 4}}
ROLES = {"maker": "s", "sub": "v"}


def derive(engines):
    return api.engines_derive({"machine": BOX, "engines": engines})["engines"]


def test_huge_batch_is_clamped_not_run():
    e = copy.deepcopy(STRATA); e["args"]["batch"] = 99_999_999
    t = time.time(); out = derive([e])["s"]
    assert time.time() - t < 3
    assert not out["errors"] and any("batch" in n and "using 8" in n for n in out["notes"])


@pytest.mark.parametrize("field,value", [("max_num_seqs", 0), ("max_num_seqs", -5), ("max_num_seqs", 0.5), ("max_model_len", 0), ("max_model_len", None)])
def test_vllm_numbers_never_throw(field, value):
    e = copy.deepcopy(VLLM); e["args"][field] = value
    out = derive([e, STRATA])
    assert out["v"]["decode_total"] and out["s"]["decode_total"]       # the bad field does not take the other engine with it
    assert any(field in n for n in out["v"]["notes"])


def test_one_broken_engine_does_not_blank_the_others(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(perf, "_vllm", boom)
    out = derive([VLLM, STRATA])
    assert out["v"]["errors"] and "internal error" in out["v"]["errors"][0]
    assert out["s"]["decode_total"]


def test_non_positive_measured_values_are_ignored():
    e = copy.deepcopy(STRATA); e["benchmarks"] = {"decode_1": -5, "prefill": float("nan"), "kv_room_tokens": 0}
    out = derive([e])["s"]
    assert out["decode_total"]["1"] > 0 and "measured" not in " ".join(out["notes"])


def simulate(**over):
    req = {"machine": BOX, "engines": [STRATA, VLLM], "roles": ROLES, "workload": {}, "concurrency": [1], "scenarios": 2}
    req.update(over)
    return api.api_simulate(req)


@pytest.mark.parametrize("field", ["maker_turns", "answer_median", "tool_calls_mean", "p_delegate", "fanout", "tool_result_median",
                                   "tool_exec_median_s", "p_approval", "sub_turns", "sub_answer_median", "sub_tool_calls_mean",
                                   "sub_final_tokens", "task_tokens"])
def test_reversed_missing_and_negative_ranges_are_repaired(field):
    for v in ([5, 1], [None, None], [-3, -1], [0, 0]):
        r = simulate(workload={field: v})
        assert not r["errors"], (field, v, r["errors"])
        assert r["results"]


@pytest.mark.parametrize("field", ["maker_sys", "sub_sys", "window", "compact_at", "visible_frac", "tasks_per_ws", "work_scale", "reruns"])
def test_scalar_workload_values_are_repaired(field):
    for v in (None, 0, -1, 1_000_000.0):
        t = time.time(); r = simulate(workload={field: v})
        assert not r["errors"] and r["results"], (field, v)
        assert time.time() - t < 20, (field, v)


def test_repairs_are_reported():
    r = simulate(workload={"maker_turns": [30, 6], "work_scale": -1})
    text = " ".join(r["warnings"])
    assert "maker_turns" in text and "swapped" in text and "work_scale" in text


def test_a_huge_range_is_limited_to_something_that_finishes():
    t = time.time(); r = simulate(workload={"fanout": [1e6, 1e6], "maker_turns": [1e6, 1e6]})
    assert r["results"] and time.time() - t < 30


RECS = [json.load(open(f)) for f in glob.glob("../measurements/vllm-*.json")]


def mutated(path, value):
    r = copy.deepcopy(RECS[0]); o = r
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = value
    return r


@pytest.mark.parametrize("path,value", [
    (["results", "decode_1"], 0), (["results", "decode_1"], float("nan")), (["results", "decode_1"], True), (["results", "decode_1"], -5),
    (["results", "decode_1"], "39"), (["method", "runs"], 0), (["method", "runs"], -3), (["method", "runs"], "abc"),
    (["method", "streams"], 0), (["method", "streams"], -3), (["method", "streams"], None), (["method", "context"], -3),
    (["method", "date"], "tomorrow"), (["method", "date"], "2026-13-45"), (["method", "date"], "2099-01-01"),
    (["method", "submitter"], "a@b.com"), (["method", "submitter"], "http://x"), (["method", "submitter"], "x" * 100),
    (["format"], "something-else"),
])
def test_bad_measurement_records_are_rejected_without_exceptions(path, value):
    v = api.measurement_validate(mutated(path, value))
    assert not v["ok"], (path, value)


def test_fx8350_can_hold_a_gpu():
    m = {"os": "linux", "cpus": [{"cpu_id": "fx-8350", "enabled": True, "ram_gb": 32, "ram_mts": 1866, "ram_channels": 2}],
         "devices": [{"id": "gpu-1", "kind": "gpu", "model_id": "rtx-3060-12g", "link_gen": 2, "link_width": 16}], "nvlinks": []}
    assert api.machine_validate(m)["ok"]
