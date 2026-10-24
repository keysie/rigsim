"""Optional dev server: the same functions as the browser build, behind FastAPI."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import api
from .schemas import ValidationError

app = FastAPI(title="rigsim", version="0.2.0")


def _run(fn, body=None):
    try:
        return fn(body) if body is not None else fn()
    except ValidationError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/catalog")
def catalog():
    return api.catalog()


@app.get("/api/calibration")
def calibration():
    return api.calibration()


@app.post("/api/machine/validate")
def machine_validate(body: dict):
    return _run(api.machine_validate, body)


@app.post("/api/engines/derive")
def engines_derive(body: dict):
    return _run(api.engines_derive, body)


@app.post("/api/simulate")
def simulate(body: dict):
    return _run(api.api_simulate, body)


@app.post("/api/trace")
def trace(body: dict):
    return _run(api.api_trace, body)


@app.get("/api/community")
def community():
    _load_measurements_once()
    return api.community()


@app.post("/api/community/match")
def community_match(body: dict):
    _load_measurements_once()
    return _run(api.community_match, body)


@app.post("/api/measurements/validate")
def measurement_validate(body: dict):
    return _run(api.measurement_validate, body)


_LOADED = False


def _load_measurements_once():
    global _LOADED
    if _LOADED:
        return
    import json
    folder = Path(__file__).resolve().parent.parent.parent / "measurements"
    records = []
    for f in sorted(folder.glob("*.json")):
        if f.name != "schema.json":
            records.append(json.loads(f.read_text()))
    api.load_records({"records": records})
    _LOADED = True


STATIC = Path(__file__).resolve().parent.parent / "static"
if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = STATIC / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(STATIC / "index.html")
