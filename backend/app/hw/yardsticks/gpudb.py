"""The GPU benchmark database (data/gpu-bench-data.json, schema v1) — low-level parts for the hardware layer.

Trust: medium for papers and PR tables; low for forum posts, user logs, community scoreboards, secondary copies and
anything overclocked or anecdotal. Rejected: entries above the theoretical peak (except where that is explained),
2:4-sparse results compared with dense peaks. Application-level families (llm_inference, hashing, mining,
molecular_dynamics, transformer_training) wait for engines (roadmap steps 4-5).
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "gpu-bench-data.json"
LOW = {"forum_post", "user_log", "community_scoreboard", "secondary_copy", "secondary", "overclocked", "anecdotal",
       "user_profile", "vendor_profile", "excerpt_partial"}
LOW_LEVEL = {"gemm", "gemv", "tensor_core_microbenchmark", "vector_compute"}
# precision (in / accumulate) -> our operation class
OP_CLASS = {("FP16", "FP32"): "fp16_tensor_fp32acc", ("FP16", "FP16"): "fp16_tensor", ("TF32", "FP32"): "tf32_tensor",
            ("INT8", "INT32"): "int8_tensor", ("INT4", "INT32"): "int4_tensor"}
MODEL_ID = {"RTX 3090": "rtx-3090", "RTX 3060": "rtx-3060-12g", "RTX 3070 Ti": "rtx-3070-ti", "RTX 4090": "rtx-4090"}


def load() -> dict:
    return json.loads(DATA.read_text())


def val(f):
    return f.get("value") if isinstance(f, dict) else f


def trust(e: dict) -> str:
    return "low" if set(e.get("quality") or []) & LOW else "medium"


def tensor_points() -> list[dict]:
    """Dense tensor-core results for GPUs in our spec table: per-SM per-clock (FMA/clk/SM, clock-independent) or
    whole-GPU throughput (TFLOPS / TOPS, at an unstated clock)."""
    out = []
    for e in load()["entries"]:
        if e["family"] not in LOW_LEVEL:
            continue
        model = MODEL_ID.get(val(e["gpu"]["model"]))
        op = e["operation"]
        cls = OP_CLASS.get((val(op.get("precision_in")), val(op.get("precision_acc"))))
        if not model or not cls or val(op.get("sparse")) is not False:
            continue
        per_sm = [m["value"] for m in e["measurements"] if m["metric"].startswith("throughput_per_sm")]
        whole = [m for m in e["measurements"] if m["metric"] in ("throughput", "throughput_max") and m["unit"] in ("TFLOPS", "TOPS")]
        if per_sm:
            out.append(dict(id=e["id"], model=model, op_class=cls, kind="per_sm_clock", value=max(per_sm) * 2,   # FMA -> ops
                            trust=trust(e), source=e["source_id"]))
        for m in whole:
            out.append(dict(id=e["id"], model=model, op_class=cls, kind="throughput", value=m["value"] * 1e12,
                            trust=trust(e), source=e["source_id"]))
    return out
