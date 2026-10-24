"""Community measurement records: validation, aggregation keys and clusters.

A record bundles the simulator's own machine and engine JSON with how it was measured and what came out,
so it is reproducible by construction. Records live in measurements/*.json in the repository; the build
copies them into the site, where the browser build aggregates them and offers matching clusters next to
the derived values. Records never change the model's constants by themselves.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st

from .catalog.cpus import CPU_BY_ID
from .catalog.gpus import GPU_BY_ID
from .catalog.models import ENGINES, MODEL_BY_ID, QUANTS
from .machine import validate
from .perf import derive
from .schemas import EngineSpec, Machine, ValidationError

RESULT_KEYS = ("decode_1", "decode_2", "decode_3", "decode_4", "decode_6", "decode_8", "prefill", "kv_room_tokens", "hit")
METHOD_FIELDS = {"tool": str, "runs": int, "streams": int, "context": int, "power_limit_w": (int, float), "driver": str,
                 "date": str}
OPTIONAL_METHOD = ("notes", "submitter", "raw_sha256", "engine_build")


def record_key(rec: dict) -> dict:
    eng = rec["engine"]
    used = [d for d in rec["machine"].get("devices", []) if d.get("kind") == "gpu" and d["id"] in eng.get("gpus", [])]
    gpus = sorted(d["model_id"] for d in used)
    caps = sorted((d.get("power_w") or rec["method"].get("power_limit_w") or 0) for d in used)
    quant = str(eng.get("args", {}).get("quantization", _default_quant(eng)))
    return {"gpus": gpus, "engine": eng["engine"], "model": eng["model_id"], "quant": quant,
            "streams": int(rec["method"].get("streams", 1)), "power_w": [int(c) for c in caps]}


def _default_quant(eng):
    e = ENGINES.get(eng.get("engine"), {})
    v = e.get("versions", {}).get(eng.get("version"), {})
    for a in v.get("args", []):
        if a["name"] == "quantization":
            return a["default"]
    return "n/a"


def record_id(rec: dict) -> str:
    k = record_key(rec)
    h = hashlib.sha256(json.dumps([k, rec["method"].get("date", ""), rec["results"]], sort_keys=True).encode()).hexdigest()[:10]
    gpu = (k["gpus"][0] if k["gpus"] else "cpu") + (f"x{len(k['gpus'])}" if len(k["gpus"]) > 1 else "")
    return f"{k['engine']}-{k['model']}-{k['quant']}-{gpu}-s{k['streams']}-{h}"


def validate_record(rec: dict) -> dict:
    """Returns {ok, errors, warnings, id, key, derived}; derived = what the model says for the same setup."""
    errors, warnings = [], []
    for part in ("machine", "engine", "method", "results"):
        if part not in rec or not isinstance(rec[part], dict):
            errors.append(f"missing section '{part}'")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}
    try:
        m = Machine.model_validate(rec["machine"])
        info = validate(m)
        errors += info["errors"]
        spec = EngineSpec.model_validate(rec["engine"])
    except ValidationError as e:
        return {"ok": False, "errors": [str(e)], "warnings": warnings}
    for c in m.cpus:
        if c.cpu_id not in CPU_BY_ID:
            errors.append(f"unknown CPU '{c.cpu_id}'")
    for d in m.devices:
        if d.kind == "gpu" and d.model_id not in GPU_BY_ID:
            errors.append(f"unknown GPU '{d.model_id}'")
    if spec.model_id not in MODEL_BY_ID:
        errors.append(f"unknown model '{spec.model_id}'")
    q = str(spec.args.get("quantization", "")) if spec.args else ""
    if q and q not in QUANTS:
        errors.append(f"unknown quantization '{q}'")
    for f, t in METHOD_FIELDS.items():
        v = rec["method"].get(f)
        if v in (None, ""):
            warnings.append(f"method.{f} missing (lowers the trust score)")
        elif not isinstance(v, t):
            errors.append(f"method.{f} should be {t if isinstance(t, type) else 'a number'}")
    if int(rec["method"].get("runs", 0) or 0) < 3:
        warnings.append("fewer than 3 runs: medians are not meaningful")
    res = rec["results"]
    if not any(k in res and res[k] for k in RESULT_KEYS):
        errors.append("results has none of " + ", ".join(RESULT_KEYS))
    for k, v in res.items():
        if k not in RESULT_KEYS:
            warnings.append(f"results.{k} is not a known field and will be ignored")
        elif not isinstance(v, (int, float)) or v < 0:
            errors.append(f"results.{k} must be a non-negative number")
    text = json.dumps(rec.get("method", {})) + json.dumps(rec.get("notes", ""))
    for bad in ("@", "http://", "https://"):
        if bad in text:
            warnings.append(f"free text contains '{bad}': keep records free of addresses and links")
    derived = None
    if not errors:
        spec_clean = EngineSpec.model_validate({**rec["engine"], "benchmarks": {}})
        pf = derive(spec_clean, info)
        if pf.errors:
            errors += [f"the simulator cannot build this engine: {e}" for e in pf.errors]
        else:
            derived = {"decode_1": pf.decode_total.get(1), "prefill": pf.prefill_rate,
                       "decode_table": {str(k): v for k, v in pf.decode_total.items()}}
            gpus = [info["_devices"][g] for g in spec.gpus if g in info["_devices"]]
            bw = sum(g["spec"]["mem_bw_gbs"] for g in gpus)
            model = MODEL_BY_ID[spec.model_id]
            if bw and model["params_b"] and not model["experts"]:
                bpw = QUANTS.get(q or _default_quant(rec["engine"]), 4.5)
                ideal = bw / (model["params_b"] * bpw / 8)        # 100 % of the bus, weights only
                d1 = res.get("decode_1")
                if d1 and d1 > ideal * 1.15:
                    errors.append(f"decode_1 {d1} tok/s exceeds the memory-bus bound of ~{ideal:.0f} tok/s for these cards")
            if derived["decode_1"] and res.get("decode_1") and res["decode_1"] > 3 * derived["decode_1"]:
                warnings.append("decode_1 is more than 3× the model's estimate; check the engine and quant fields")
            if derived["prefill"] and res.get("prefill") and res["prefill"] > 4 * derived["prefill"]:
                warnings.append("prefill is more than 4× the model's estimate")
    out = {"ok": not errors, "errors": errors, "warnings": warnings}
    if not errors:
        out["id"] = record_id(rec)
        out["key"] = record_key(rec)
        out["derived"] = derived
    return out


def trust_score(rec: dict) -> float:
    m = rec.get("method", {})
    have = sum(1 for f in METHOD_FIELDS if m.get(f) not in (None, ""))
    s = have / len(METHOD_FIELDS)
    if int(m.get("runs", 0) or 0) >= 3:
        s += 0.15
    if m.get("raw_sha256"):
        s += 0.1
    return round(min(1.0, s), 2)


def aggregate(records: list[dict]) -> list[dict]:
    """Clusters by key: count, median and spread per result field, trust, and the model's error."""
    groups: dict[str, dict] = {}
    for rec in records:
        v = validate_record(rec)
        if not v["ok"]:
            continue
        k = json.dumps(v["key"], sort_keys=True)
        g = groups.setdefault(k, {"key": v["key"], "n": 0, "values": {}, "trust": [], "derived": v["derived"], "cpus": set(), "os": set(), "ids": []})
        g["n"] += 1
        g["ids"].append(v["id"])
        g["trust"].append(trust_score(rec))
        g["cpus"].add(rec["machine"]["cpus"][0]["cpu_id"])
        g["os"].add(rec["machine"].get("os", "linux"))
        for f in RESULT_KEYS:
            if f in rec["results"] and rec["results"][f]:
                g["values"].setdefault(f, []).append(float(rec["results"][f]))
    out = []
    for g in groups.values():
        summary = {}
        for f, vals in g["values"].items():
            med = st.median(vals)
            spread = (max(vals) - min(vals)) / med if med and len(vals) > 1 else 0.0
            summary[f] = {"median": round(med, 2 if f == "hit" else 1), "n": len(vals), "spread": round(spread, 2)}
        model_err = None
        if g["derived"] and g["derived"].get("decode_1") and "decode_1" in summary:
            model_err = round((g["derived"]["decode_1"] - summary["decode_1"]["median"]) / summary["decode_1"]["median"], 3)
        out.append({"key": g["key"], "n": g["n"], "results": summary, "trust": round(st.mean(g["trust"]), 2),
                    "cpus": sorted(g["cpus"]), "os": sorted(g["os"]), "ids": g["ids"], "model_decode_1": g["derived"]["decode_1"] if g["derived"] else None,
                    "model_prefill": g["derived"]["prefill"] if g["derived"] else None, "model_error_decode_1": model_err})
    out.sort(key=lambda c: (-c["n"], c["key"]["engine"], c["key"]["model"]))
    return out


def match(records_or_clusters: list[dict], machine: dict, engine: dict) -> list[dict]:
    """Clusters whose key matches this engine on this machine (any streams), most measured first."""
    rec = {"machine": machine, "engine": engine, "method": {"streams": 1}, "results": {"decode_1": 1}}
    try:
        k = record_key(rec)
    except (KeyError, TypeError):
        return []
    exact, near = [], []
    for c in records_or_clusters:
        ck = c["key"]
        same_model = ck["engine"] == k["engine"] and ck["model"] == k["model"]
        if same_model and ck["gpus"] == k["gpus"] and ck["quant"] == k["quant"] and ck.get("power_w", []) == k.get("power_w", []):
            exact.append({**c, "near": None})
        elif same_model and ck["gpus"] == k["gpus"] and ck["quant"] == k["quant"]:
            exact.append({**c, "near": f"other power cap ({', '.join(f'{w} W' if w else 'default' for w in ck.get('power_w', []))})"})
        elif same_model and ck["gpus"] == k["gpus"]:
            near.append({**c, "near": f"other quant ({ck['quant']})"})
        elif same_model and ck["quant"] == k["quant"] and set(ck["gpus"]) == set(k["gpus"]):
            near.append({**c, "near": "same cards, different count"})
        elif same_model and ck["quant"] == k["quant"]:
            near.append({**c, "near": "other cards (" + ", ".join(ck["gpus"]) + ")"})
    return exact if exact else near[:4]
