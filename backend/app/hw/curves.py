"""Power curves from data/gpu-power-curves.json, as {model: {(primitive, load class): PowerCurve}} for calibration."""
from __future__ import annotations

import json
from pathlib import Path

from .power import PowerCurve

DATA = Path(__file__).parent / "data" / "gpu-power-curves.json"


def load() -> dict:
    return json.loads(DATA.read_text())


def calibration_curves() -> dict[str, dict[tuple[str, str], PowerCurve]]:
    out: dict[str, dict] = {}
    for c in load()["curves"]:
        if c.get("use") != "calibration":
            continue
        for cls, pts in c["points"].items():
            out.setdefault(c["model"], {})[(c["primitive"], cls)] = PowerCurve(
                points=[tuple(p) for p in pts], basis=c.get("basis", "measured"), source=c["id"])
    return out
