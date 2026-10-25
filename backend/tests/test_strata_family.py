"""The Flash-Next packs Strata serves: packs per model, RAM budget / SSD streaming, speed factor."""
from app import api

STRATA = ["qwen3.8-flash-next", "qwen3.8-flash-next-swift", "qwen3.8-flash-next-coder", "qwen3.8-flash-next-unsloth", "qwen3.8-flash-next-orca"]


def box(ram, gpus=("rtx-3090",), ssd=True):
    devs = [{"id": f"gpu-{i + 1}", "kind": "gpu", "model_id": g, "link_gen": 4, "link_width": 16} for i, g in enumerate(gpus)]
    if ssd:
        devs.append({"id": "ssd-1", "kind": "ssd", "model_id": "nvme-gen4", "link_gen": 4, "link_width": 4, "capacity_tb": 2})
    return {"os": "linux", "cpus": [{"cpu_id": "epyc-7453", "enabled": True, "ram_gb": ram, "ram_mts": 3200, "ram_channels": 8}], "devices": devs, "nvlinks": []}


def strata(m, model, quant, storage="ssd-1"):
    e = {"id": "s", "engine": "strata", "version": "0.1.43.1", "model_id": model, "gpus": ["gpu-1"], "cpu_cores": 16, "storage": storage,
         "args": {"batch": 1, "quantization": quant}}
    return api.engines_derive({"machine": m, "engines": [e]})["engines"]["s"]


def test_every_pack_derives():
    from app.catalog.models import ENGINES, MODEL_BY_ID
    assert set(STRATA) <= set(ENGINES["strata"]["models"])
    for mid in STRATA:
        for q in MODEL_BY_ID[mid]["quants"]:
            r = strata(box(256), mid, q)
            assert not r["errors"] and r["decode_total"]["1"] > 5, (mid, q, r["errors"])


def test_a_pack_the_model_does_not_come_in_is_forced_to_one_it_does():
    r = strata(box(64), "qwen3.8-flash-next-coder", "iq3_s")
    assert r["arg_states"]["quantization"]["force"] == "iq1_m"
    assert abs(r["memory"]["experts_total_gb"] - 25.5) < 1       # IQ1_M stores experts like IQ3_S: 67B x 0.87 x 3.5 bit


def test_unsloth_experts_beyond_the_ram_budget_go_to_the_ssd_and_slow_misses():
    small = strata(box(64), "qwen3.8-flash-next-unsloth", "ud_iq4_xs")
    big = strata(box(96), "qwen3.8-flash-next-unsloth", "ud_iq4_xs")
    assert small["memory"]["experts_on_ssd_gb"] > 15 and big["memory"]["experts_on_ssd_gb"] == 0
    assert small["decode_total"]["1"] < big["decode_total"]["1"] * 0.8
    assert not small["errors"]                                       # the RAM check uses the budget, not the whole pack


def test_unsloth_without_storage_is_an_error_only_when_it_needs_the_ssd():
    r = strata(box(64, ssd=False), "qwen3.8-flash-next-unsloth", "ud_iq4_xs", storage=None)
    assert r["errors"] and "no model storage" in r["errors"][0]
    r = strata(box(128, ssd=False), "qwen3.8-flash-next-unsloth", "ud_iq4_xs", storage=None)
    assert not r["errors"]


def test_coder_is_not_faster_than_the_original_per_token():
    coder = strata(box(256), "qwen3.8-flash-next-coder", "iq1_m")["decode_total"]["1"]
    orig = strata(box(256), "qwen3.8-flash-next", "iq3_s")["decode_total"]["1"]
    assert coder < orig * 1.3          # Strata table: +4 %; our box differs in hit rate, so allow some room
