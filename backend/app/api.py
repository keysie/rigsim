"""Function-level API shared by the browser build (Pyodide) and the dev server."""
from __future__ import annotations

import json

from .calibration import SOURCES, check_claims, data_points
from .catalog.cpus import CPUS, FIELDS as CPU_FIELDS, RAM_SIZES_GB, RAM_SPEEDS
from .catalog.gpus import GPUS, GPU_FIELDS, PCIE_LANE_GBS, SSDS, SSD_FIELDS
from .catalog.models import ENGINES, MODELS, QUANTS
from .catalog.prebuilts import PREBUILTS
from .machine import validate
from . import measurements as meas
from .perf import derive
from .schemas import EngineSpec, Machine, SimRequest, ValidationError
from .sim.run import simulate, trace


def catalog():
    from .catalog.cpus import CPU_BY_ID
    from .catalog.gpus import GPU_BY_ID
    return {"cpus": [CPU_BY_ID[r[0]] for r in CPUS], "gpus": [GPU_BY_ID[r[0]] for r in GPUS],
            "ssds": [dict(zip(SSD_FIELDS, r)) for r in SSDS], "ram_speeds": RAM_SPEEDS, "ram_sizes_gb": RAM_SIZES_GB,
            "pcie_lane_gbs": PCIE_LANE_GBS, "prebuilts": PREBUILTS, "soc_ram_gb": __import__("app.catalog.cpus", fromlist=["SOC_RAM_GB"]).SOC_RAM_GB,
            "soc_gpu_for": __import__("app.catalog.gpus", fromlist=["SOC_GPU_FOR"]).SOC_GPU_FOR, "models": MODELS, "quants": QUANTS, "engines": ENGINES}


def machine_validate(body):
    info = validate(Machine.model_validate(body))
    return {k: v for k, v in info.items() if not k.startswith("_")}


def engines_derive(body):
    info = validate(Machine.model_validate(body["machine"]))
    specs = [EngineSpec.model_validate(e) for e in body.get("engines", [])]
    perfs = {s.id: derive(s, info) for s in specs}
    # one engine per card, cores not oversubscribed: sharing is out of scope (README), so it is an error here too
    users: dict = {}
    for s in specs:
        for g in s.gpus:
            users.setdefault(g, []).append(s)
    for g, es in users.items():
        if len(es) > 1:
            for s in es:
                perfs[s.id].errors.append(f"{g} is also used by {', '.join((o.name or o.id) for o in es if o.id != s.id)}: one engine per card — sharing a GPU is out of scope")
    by_cpu: dict = {}
    for s in specs:
        if s.engine != "hosted-api":
            by_cpu.setdefault(s.cpu, []).append(s)
    for ci, es in by_cpu.items():
        have = int(info["cpus"].get(ci, {}).get("cores", 0) or 0)
        total = sum(int(s.cpu_cores or 0) for s in es)
        if have and total > have:
            for s in es:
                perfs[s.id].errors.append(f"CPU {ci + 1}: the engines on it ask for {total} cores, it has {have} — cores are not shared")
    out = {s.id: perfs[s.id].public() for s in specs}
    return {"machine": {k: v for k, v in info.items() if not k.startswith("_")}, "engines": out}


def api_simulate(body):
    return simulate(SimRequest.model_validate(body))


def api_trace(body):
    req = SimRequest.model_validate(body["request"])
    return trace(req, int(body.get("concurrency", 2)), int(body.get("scenario", 0)))


def calibration():
    return {"sources": SOURCES, "points": data_points(), "claims": check_claims()}


_RECORDS: list[dict] = []
_CLUSTERS: list[dict] = []


def load_records(body):
    """The browser hands over the records shipped with the site; the dev server reads measurements/."""
    global _RECORDS, _CLUSTERS
    _RECORDS = list(body.get("records", []))
    _CLUSTERS = meas.aggregate(_RECORDS)
    return {"records": len(_RECORDS), "clusters": len(_CLUSTERS)}


def community():
    return {"clusters": _CLUSTERS, "records": len(_RECORDS)}


def community_match(body):
    return {"clusters": meas.match(_CLUSTERS, body["machine"], body["engine"])}


def measurement_validate(body):
    return meas.validate_record(body)


ROUTES = {"/api/measurements/load": load_records, "/api/community": community, "/api/community/match": community_match,
          "/api/measurements/validate": measurement_validate,"/api/catalog": catalog, "/api/machine/validate": machine_validate, "/api/engines/derive": engines_derive,
          "/api/simulate": api_simulate, "/api/trace": api_trace, "/api/calibration": calibration}


def call(path: str, body_json: str | None = None) -> str:
    """Entry point for the browser: JSON in, JSON out, errors as {"error": ...}."""
    fn = ROUTES.get(path)
    if fn is None:
        return json.dumps({"error": f"unknown route {path}"})
    try:
        body = json.loads(body_json) if body_json else None
        return json.dumps(fn(body) if body is not None else fn())
    except ValidationError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:  # noqa: BLE001 - surface anything to the UI
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
