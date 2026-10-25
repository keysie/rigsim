"""Rules between arguments, hardware and model (app/argrules.py), checked through derive()."""
from app import api

M2 = {"os": "linux", "cpus": [{"cpu_id": "ryzen9-7950x", "enabled": True, "ram_gb": 128, "ram_mts": 5200, "ram_channels": 2}],
      "devices": [{"id": "gpu-1", "kind": "gpu", "model_id": "rtx-4090", "link_gen": 4, "link_width": 16},
                  {"id": "gpu-2", "kind": "gpu", "model_id": "rtx-3090", "link_gen": 4, "link_width": 8}], "nvlinks": []}


def derive(engine):
    return api.engines_derive({"machine": M2, "engines": [engine]})["engines"][engine["id"]]


def strata(gpus, **args):
    return derive({"id": "s", "engine": "strata", "version": "0.1.43.1", "model_id": "qwen3.8-flash-next", "gpus": gpus, "cpu_cores": 12, "args": args})


def test_strata_pipeline_windows_only_for_a_solo_split():
    r = strata(["gpu-1", "gpu-2"], batch=4, pipeline_windows="2")
    assert r["arg_states"]["pipeline_windows"]["disabled"] and r["arg_states"]["pipeline_windows"]["force"] == "0"
    r = strata(["gpu-1"], batch=1, pipeline_windows="2")
    assert r["arg_states"]["pipeline_windows"]["disabled"]                 # one card: nothing to pipeline
    r = strata(["gpu-1", "gpu-2"], batch=1, pipeline_windows="2")
    assert not r["arg_states"].get("pipeline_windows", {}).get("disabled")


def test_strata_batch_mtp_needs_serial_groups_and_mtp():
    r = strata(["gpu-1", "gpu-2"], batch=4, batch_mtp=True)                # pipelined by default
    assert r["arg_states"]["batch_mtp"]["force"] is False
    r = strata(["gpu-1", "gpu-2"], batch=4, batch_mtp=True, batch_groups="1")
    assert not r["arg_states"].get("batch_mtp", {}).get("disabled")
    r = strata(["gpu-1", "gpu-2"], batch=4, batch_mtp=True, batch_groups="1", mtp=False)
    assert r["arg_states"]["batch_mtp"]["disabled"]


def test_strata_parking_needs_both_values_and_follows_the_conversation_size_not_the_slots():
    # 8192 MiB at the default 32K context: a full conversation is ~0.66 GB, so 12 fit — whatever the batch size
    forced = {b: strata(["gpu-1"], batch=b, parking_mib=8192)["arg_states"]["parking_slots"]["force"] for b in (1, 2, 4, 8)}
    assert len(set(forced.values())) == 1 and 10 <= forced[1] <= 14, forced
    r = strata(["gpu-1"], batch=2, parking_slots=6)
    assert 3000 <= r["arg_states"]["parking_mib"]["force"] <= 5120        # 6 x ~0.66 GB, rounded up to a GiB


def test_strata_peer_device_excludes_split_options():
    r = strata(["gpu-1", "gpu-2"], batch=1, multi_gpu="peer-device")
    assert all(r["arg_states"][k]["disabled"] for k in ("batch_groups", "pipeline_windows", "trim_stage_weights"))
    assert r["decode_total"] and "peer-device" in " ".join(r["notes"])


def test_strata_groups_must_divide_batch():
    r = strata(["gpu-1", "gpu-2"], batch=3, batch_groups="2")
    assert r["arg_states"]["batch_groups"]["force"] == "auto"


def test_vllm_fp8_needs_ada_or_newer():
    r = derive({"id": "v", "engine": "vllm", "version": "0.31.0", "model_id": "qwen3.8-27b", "gpus": ["gpu-2"],
                "args": {"tensor_parallel": "1", "quantization": "fp8", "kv_cache_dtype": "fp8"}})
    assert r["arg_states"]["quantization"]["force"] == "awq4" and r["arg_states"]["kv_cache_dtype"]["force"] == "fp16"
    r = derive({"id": "v", "engine": "vllm", "version": "0.31.0", "model_id": "qwen3.8-27b", "gpus": ["gpu-1"],
                "args": {"tensor_parallel": "1", "quantization": "fp8"}})
    assert r["arg_states"].get("quantization", {}).get("force") is None


def test_vllm_parallelism_matches_the_gpus():
    r = derive({"id": "v", "engine": "vllm", "version": "0.31.0", "model_id": "qwen3.8-27b", "gpus": ["gpu-1", "gpu-2"],
                "args": {"tensor_parallel": "1"}})
    assert r["arg_states"]["tensor_parallel"]["force"] == "2"
    assert r["errors"] == ["tensor parallel needs identical GPUs"] and "identical" in r["arg_states"]["tensor_parallel"]["reason"]
    r = derive({"id": "v", "engine": "vllm", "version": "0.31.0", "model_id": "qwen3.8-27b", "gpus": ["gpu-1", "gpu-2"],
                "args": {"tensor_parallel": "1", "pipeline_parallel": "2"}})
    assert not r["errors"] and "pipeline parallel 2" in " ".join(r["notes"])
    r = derive({"id": "v", "engine": "vllm", "version": "0.31.0", "model_id": "qwen3.8-27b", "gpus": ["gpu-1"],
                "args": {"tensor_parallel": "1", "pipeline_parallel": "2"}})
    assert r["arg_states"]["pipeline_parallel"]["force"] == "1"


def test_vllm_speculative_needs_a_draft_head():
    r = derive({"id": "v", "engine": "vllm", "version": "0.31.0", "model_id": "qwen3-32b", "gpus": ["gpu-1"], "args": {"tensor_parallel": "1", "speculative": True}})
    assert r["arg_states"]["speculative"]["disabled"] and r["arg_states"]["speculative"]["force"] is False


def test_llama_rules():
    l = {"id": "l", "engine": "llama.cpp", "version": "b11552", "model_id": "llama-3.1-8b", "gpus": ["gpu-1"], "cpu_cores": 8}
    r = derive({**l, "args": {"cache_type": "q8_0", "flash_attn": False}})
    assert r["arg_states"]["cache_type"]["force"] == "f16"
    r = derive({**l, "args": {"ubatch": 4096, "batch": 1024}})
    assert r["arg_states"]["ubatch"]["force"] == 1024
    r = derive({**l, "args": {"mtp": True, "cpu_moe": True}})
    assert r["arg_states"]["mtp"]["force"] is False and r["arg_states"]["cpu_moe"]["force"] is False      # dense model, no MTP head
    r = derive({**l, "args": {}})
    assert r["arg_states"]["threads"]["disabled"] and r["arg_states"]["split_mode"]["disabled"]


def test_llama_cpu_moe_makes_a_120b_fit_on_one_card():
    r = derive({"id": "l", "engine": "llama.cpp", "version": "b11552", "model_id": "gpt-oss-120b", "gpus": ["gpu-1"], "cpu_cores": 16,
                "args": {"quantization": "mxfp4", "parallel": 1, "ctx_size": 16384, "cpu_moe": True}})
    assert not r["errors"] and 8 < r["decode_total"]["1"] < 60
    r = derive({"id": "l", "engine": "llama.cpp", "version": "b11552", "model_id": "gpt-oss-120b", "gpus": ["gpu-1"], "cpu_cores": 16,
                "args": {"quantization": "mxfp4", "parallel": 1, "ctx_size": 16384}})
    assert r["errors"]


def test_every_schema_argument_is_consumed_or_ruled():
    """No dead arguments: each must appear in perf.py or argrules.py by name."""
    import pathlib
    from app.catalog.models import ENGINES
    code = "".join(pathlib.Path(f).read_text() for f in ("app/perf.py", "app/argrules.py", "app/sim/engines.py", "app/sim/run.py"))
    for e, d in ENGINES.items():
        for v, vd in d["versions"].items():
            for a in vd["args"]:
                assert f'"{a["name"]}"' in code, f"{e} {v}: argument {a['name']} is in the schema but used nowhere"
