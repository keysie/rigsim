#!/usr/bin/env python3
"""Turn a fetched (masked) run into rigsim validation / calibration data. Trust: high (own controlled runs,
repeats, host-side sampling) — still marked per point with its spread.

  to_rigsim.py RUNDIR [--out FILE]

gpu-clock-power -> SM / memory clock per card, load class and busy share vs power limit (hw/power.py curves)
gpu-gemm        -> achieved ops/s and the share of SMs x measured clock x per-SM rate (hw/rates.py MEASUREMENTS)
cpu-mem-ladder  -> read bandwidth per kernel, thread count and core clock (per-core link, RAM efficiency)
anything else   -> one row per result, medians over repeats
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MODEL_IDS = {"NVIDIA GeForce RTX 3090": "rtx-3090", "NVIDIA GeForce RTX 3060": "rtx-3060-12g",
             "NVIDIA GeForce RTX 4090": "rtx-4090", "NVIDIA GeForce RTX 3070 Ti": "rtx-3070-ti"}
PREC_CLASS = {"fp32": "fp32", "tf32": "tf32_tensor", "fp16_fp32acc": "fp16_tensor_fp32acc", "fp16": "fp16_tensor",
              "int8": "int8_tensor", "bf16": "fp16_tensor_fp32acc"}       # BF16 runs at the FP16/FP32-acc rate


def rows(rundir: Path) -> list[dict]:
    out = []
    for line in (rundir / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["rc"] == 0 and not r.get("hot") and r.get("results"):
            out.append(r)
    return out


def spread(vals: list[float]) -> dict:
    vals = sorted(vals)
    return dict(n=len(vals), median=statistics.median(vals), min=vals[0], max=vals[-1])


def spec_table():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
        from app.hw.gpus import GPUS
        return GPUS
    except Exception:                                                  # noqa: BLE001 - converter works without it
        return {}


def clock_power(rs: list[dict], models: dict) -> dict:
    groups = defaultdict(list)
    for r in rs:
        p = r["params"]
        s = r["samples"].get(p["gpu"], {})
        if "clocks.sm" not in s:
            continue
        groups[(p["gpu"], p["load"], float(p["duty"]), p["power_w"])].append(
            (s["clocks.sm"]["median"], s.get("clocks.mem", {}).get("median"), s.get("power.draw", {}).get("median"),
             r["results"][0].get("busy_share")))
    curves: dict = {}
    for (gpu, load, duty, watts), v in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], float(kv[0][3]))):
        curves.setdefault(gpu, dict(model=models.get(gpu), points=[]))["points"].append(dict(
            load_class=load, duty=duty, limit_w=float(watts), sm_mhz=spread([x[0] for x in v]),
            mem_mhz=spread([x[1] for x in v if x[1] is not None]) if any(x[1] for x in v) else None,
            draw_w=spread([x[2] for x in v if x[2] is not None]) if any(x[2] for x in v) else None,
            busy_share=statistics.median([x[3] for x in v if x[3] is not None]) if any(x[3] for x in v) else None))
    return dict(kind="gpu-power-curves", trust="high", cards=curves)


def gemm(rs: list[dict], models: dict) -> dict:
    gpus = spec_table()
    groups = defaultdict(list)
    for r in rs:
        p, res = r["params"], r["results"][0]
        if res.get("unsupported"):
            groups[(p["gpu"], p["prec"], p["shape"])].append(None); continue
        clock = r["samples"].get(p["gpu"], {}).get("clocks.sm", {}).get("median")
        groups[(p["gpu"], p["prec"], p["shape"])].append((res["ops_per_s"], clock, res.get("cudart")))
    out = []
    for (gpu, prec, shape), v in sorted(groups.items()):
        model = models.get(gpu)
        if all(x is None for x in v):
            out.append(dict(card=gpu, model=model, prec=prec, shape=shape, unsupported=True)); continue
        v = [x for x in v if x is not None]
        ops = spread([x[0] for x in v])
        clock = statistics.median([x[1] for x in v if x[1]]) if any(x[1] for x in v) else None
        eff = None
        spec = gpus.get(model or "")
        cls = PREC_CLASS.get(prec)
        if spec and clock and cls in spec["ops_per_clk_per_sm"]:
            eff = ops["median"] / (spec["sms"] * clock * 1e6 * spec["ops_per_clk_per_sm"][cls])
        out.append(dict(card=gpu, model=model, prec=prec, op_class=cls, shape=shape, ops_per_s=ops, sm_mhz=clock,
                        share_of_clock_peak=eff, cudart=v[0][2]))
    return dict(kind="gemm-rates", trust="high", measurements=out)


def ladder(rs: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rs:
        p, res = r["params"], r["results"][0]
        if res.get("bytes_per_s"):
            groups[(p["kernel"], int(p["threads"]), p.get("cpu_mhz"))].append(res["bytes_per_s"] / 1e9)
    return dict(kind="core-ladder", trust="high", points=[
        dict(kernel=k, threads=t, cpu_mhz=c, read_gbs=spread(v)) for (k, t, c), v in sorted(groups.items(), key=str)])


def generic(rs: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rs:
        groups[json.dumps(r["params"], sort_keys=True)].append(r["results"])
    return dict(kind="results", trust="high", rows=[dict(params=json.loads(k), results=v) for k, v in sorted(groups.items())])


def convert(rundir: Path) -> dict:
    rs = rows(rundir)
    base = json.loads((rundir / "baseline.json").read_text()) if (rundir / "baseline.json").exists() else {}
    models = {g["name"]: MODEL_IDS.get(g["model"], g["model"]) for g in base.get("gpus", [])}
    q = rs[0]["question"] if rs else None
    data = {"gpu-clock-power": lambda: clock_power(rs, models), "gpu-gemm": lambda: gemm(rs, models),
            "cpu-mem-ladder": lambda: ladder(rs)}.get(q, lambda: generic(rs))()
    return dict(question=q, source=rundir.name, items=len(rs), **data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    text = json.dumps(convert(Path(a.rundir)), indent=1)
    Path(a.out).write_text(text + "\n") if a.out else print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
