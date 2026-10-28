"""Validation of the GPU spec table against the low-level GPU data (no fitting).

per_sm_clock points test ops_per_clk_per_sm directly (clock-independent). Throughput points test the whole-GPU
peak at the spec boost clock: SMs x boost x ops per clock per SM."""
from __future__ import annotations

from ..gpus import GPUS
from . import gpudb


def run() -> list[dict]:
    rows = []
    for p in gpudb.tensor_points():
        spec = GPUS[p["model"]]
        per = spec["ops_per_clk_per_sm"][p["op_class"]]
        model = per if p["kind"] == "per_sm_clock" else per * spec["sms"] * spec["boost_mhz"] * 1e6
        rows.append(dict(p, model_value=model, ratio=p["value"] / model))
    return rows


if __name__ == "__main__":
    for r in run():
        unit = "ops/clk/SM" if r["kind"] == "per_sm_clock" else "TOPS"
        scale = 1 if r["kind"] == "per_sm_clock" else 1e12
        print(f"{r['id'][:34]:34s} {r['model']:12s} {r['op_class']:20s} {r['trust']:6s} measured {r['value'] / scale:8.1f} "
              f"spec {r['model_value'] / scale:8.1f} {unit:10s} measured/spec {r['ratio']:.3f}")
