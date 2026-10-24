"""The performance model must land near the one measured box (3 x RTX 3090 + 3060, i9-10900X)."""
import pytest

from app.machine import validate
from app.perf import derive
from app.schemas import CpuSlot, EngineSpec, Machine, PcieDevice

BOX = Machine(cpus=[CpuSlot(cpu_id="i9-10900x", ram_gb=128, ram_mts=3200, ram_channels=4)],
              devices=[PcieDevice(id="gpu-1", kind="gpu", model_id="rtx-3090", link_gen=3, link_width=16, power_w=250),
                       PcieDevice(id="gpu-2", kind="gpu", model_id="rtx-3090", link_gen=3, link_width=16, power_w=250),
                       PcieDevice(id="gpu-3", kind="gpu", model_id="rtx-3090", link_gen=3, link_width=8, power_w=250),
                       PcieDevice(id="gpu-4", kind="gpu", model_id="rtx-3060-12g", link_gen=3, link_width=4)])
INFO = validate(BOX)


def near(x, target, tol=0.3):
    assert abs(x - target) <= tol * target, f"{x} not within {tol:.0%} of {target}"


def strata(gpus, **args):
    return derive(EngineSpec(id="s", engine="strata", version="0.1.43.1", model_id="qwen3.8-flash-next",
                             gpus=gpus, cpu_cores=9, args=args), INFO)


def test_strata_one_card_matches_pass_2():
    p = strata(["gpu-1"], batch=1)
    assert not p.errors
    near(p.decode_total[1], 95)          # measured 91-99 tok/s
    near(p.prefill_rate, 1900)           # measured 1,880-2,000 tok/s
    assert 0.8 <= p.memory["hit"] <= 0.95


def test_strata_split_matches_pass_1():
    p = strata(["gpu-1", "gpu-2"], batch=4)
    assert not p.errors
    near(p.decode_total[1], 125)         # 105-135 tok/s
    near(p.decode_total[4], 175)         # ~175 tok/s total with 4 slots
    near(p.prefill_rate, 3300, 0.35)     # ~3,300 tok/s shared


def test_strata_three_cards_prefill():
    p = strata(["gpu-1", "gpu-2", "gpu-3"], batch=1)
    near(p.prefill_rate, 4000)           # +25 % over two cards


def test_strata_needs_room():
    p = derive(EngineSpec(id="s", engine="strata", version="0.1.43.1", model_id="qwen3.8-flash-next",
                          gpus=["gpu-4"], cpu_cores=9, args={"batch": 8, "context": 131072}), INFO)
    assert p.errors


def test_vllm_27b_awq_on_a_3090():
    p = derive(EngineSpec(id="v", engine="vllm", version="0.31.0", model_id="qwen3.8-27b", gpus=["gpu-3"],
                          args={"tensor_parallel": "1", "max_num_seqs": 4}), INFO)
    assert not p.errors
    near(p.speed_per_request[1], 39)     # 38-41 tok/s
    near(p.prefill_rate, 1050, 0.35)     # 1,000-1,100 tok/s
    near(p.memory["kv_room_tokens"], 21000, 0.4)


def test_vllm_tp2_over_pcie():
    p = derive(EngineSpec(id="v", engine="vllm", version="0.31.0", model_id="qwen3.8-27b", gpus=["gpu-1", "gpu-2"],
                          args={"tensor_parallel": "2", "max_num_seqs": 8}), INFO)
    assert not p.errors
    near(p.speed_per_request[1], 50, 0.35)   # 42-55 tok/s flat to 6


def test_vllm_refuses_what_does_not_fit():
    p = derive(EngineSpec(id="v", engine="vllm", version="0.31.0", model_id="llama-3.3-70b", gpus=["gpu-4"],
                          args={"tensor_parallel": "1"}), INFO)
    assert p.errors and "does not fit" in p.errors[0]


def test_llama_27b_three_slots_fit_a_3090():
    p = derive(EngineSpec(id="l", engine="llama.cpp", version="b11552", model_id="qwen3.8-27b", gpus=["gpu-3"],
                          args={"parallel": 3, "ctx_size": 32768, "quantization": "iq3_s"}), INFO)
    assert not p.errors
    near(p.speed_per_request[3], 29, 0.4)    # 24-35 tok/s with MTP at 2-3 slots


def test_guard_and_reranker():
    g = derive(EngineSpec(id="g", engine="llama.cpp", version="b11552", model_id="ministral-3-3b", gpus=["gpu-4"],
                          args={"parallel": 1, "quantization": "q8_0"}), INFO)
    near(g.prefill_rate, 4500, 0.4)          # ~5,000 tok/s measured
    r = derive(EngineSpec(id="r", engine="llama.cpp", version="b11552", model_id="bge-reranker-v2-m3", gpus=[],
                          cpu_cores=4, args={"n_gpu_layers": "0"}), INFO)
    near(40 / r.decode_total[1], 25, 0.4)    # 23 s per tool search on the CPU


def test_hosted_engine():
    p = derive(EngineSpec(id="c", engine="hosted-api", version="provider", model_id="deepseek-v4.1-flash"), INFO)
    assert p.kind == "hosted" and p.hosted_prices["price_in"] == 0.5
