"""rigsim's predictions for a measured run of the rigsim-benchmark stick (contract: rigsim-benchmark's
predictions/README.md). Run from backend/:

  python3 -m app.predict --run <run folder> --measurements <measurements.json> --out <out.json>

The machine is built from the run's own inventory: GPUs by nvidia-smi name and memory size (catalogue: hw/gpus.py),
power limits from nvidia-smi, PCIe link from inventory/pcie.json. What the model covers is predicted; everything
else comes back as unsupported with the reason. No measurement of these runs went into rigsim's constants, so no
prediction is marked as calibration."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .engine.llamacpp import cuda
from .engine.model import ARCHS, FILE_TYPES, from_config
from .hw.execute import Op, execute
from .hw.gpu import add_gpu
from .hw.gpus import GPUS
from .hw.machine import Machine
from .hw.primitives import Router

# nvidia-smi names -> catalogue, told apart by memory where one name covers several cards
GPU_NAMES = [(re.compile(r"RTX 3090(?! Ti)"), None, "rtx-3090"), (re.compile(r"RTX 3080(?! Ti)"), (9000, 11000), "rtx-3080"),
             (re.compile(r"RTX 3070 Ti"), None, "rtx-3070-ti"), (re.compile(r"RTX 3060(?! Ti)"), (11000, 13000), "rtx-3060-12g"),
             (re.compile(r"RTX 4090"), None, "rtx-4090")]
MODELS = {"llama-2-7b": "llama-2-7b", "tinyllama-1.1b": "tinyllama-1.1b", "llama-3.2-1b": "llama-3.2-1b",
          "llama-3.1-8b": "llama-3.1-8b", "gemma-2b": "gemma-2b"}


def catalogue_id(name: str, mib: float | None) -> str | None:
    for rx, mem, mid in GPU_NAMES:
        if rx.search(name) and (mem is None or mib is None or mem[0] <= mib <= mem[1]):
            return mid
    return None


def nvidia_gpus(run: Path) -> list[dict]:
    """[{index, name, mib, pci}] in nvidia-smi order (= the CUDA order the stick uses)."""
    try:
        hw = json.loads((run / "run.json").read_text()).get("hardware", {})
    except (OSError, ValueError):
        return []
    out = []
    for g in hw.get("gpus", []):
        parts = [x.strip() for x in g.split(",")]
        if len(parts) >= 3 and re.search(r"[0-9a-fA-F]+:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d", parts[-1]):
            m = re.search(r"([\d.]+)\s*MiB", parts[1])
            bus = parts[-1].lower()
            bus = re.sub(r"^0*([0-9a-f]{4}):", r"\1:", bus) if len(bus.split(":")[0]) > 4 else bus
            out.append(dict(index=len(out), name=parts[0], mib=float(m.group(1)) if m else None, pci=bus[-12:]))
    return out


def power_limits(run: Path) -> dict[str, float]:
    """bus (0000:bb:dd.f) -> current power limit in W, from `nvidia-smi -q` in the inventory."""
    f = run / "inventory" / "nvidia-smi.txt"
    out, bus = {}, None
    for line in (f.read_text(errors="replace").splitlines() if f.exists() else []):
        m = re.match(r"^GPU ([0-9A-Fa-f]+:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d)", line)
        if m:
            bus = m.group(1).lower()[-12:]
        m = re.match(r"\s*(?:Current )?Power Limit\s*:\s*([\d.]+) W", line)
        if bus and m and bus not in out:
            out[bus] = float(m.group(1))
    return out


def pcie_slot(run: Path, pci: str) -> tuple[int, int]:
    """Generation and width the card can link at: the lower of its own and its port's maximum (pcie.json, 1.2+)."""
    try:
        devs = json.loads((run / "inventory" / "pcie.json").read_text())["devices"]
    except (OSError, ValueError, KeyError):
        return 4, 16
    gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
    d = devs.get(pci, {})
    links = [d.get("link") or {}] + [(devs.get(d.get("parent", ""), {}).get("link") or {})]
    g = [gen.get(str(l.get("max_speed", "")).split(" ")[0]) for l in links]
    w = [int(l["max_width"]) for l in links if str(l.get("max_width", "")).isdigit()]
    g = [x for x in g if x]
    return (min(g) if g else 4), (min(w) if w else 16)


def machine_for(run: Path, gpu: dict, model_id: str) -> Machine:
    m = Machine(); m.add(Router("host"))
    gen, lanes = pcie_slot(run, gpu["pci"])
    add_gpu(m, "gpu0", model_id, "host", slot_gen=gen, slot_lanes=lanes, power_limit_w=power_limits(run).get(gpu["pci"]))
    return m


def model_for(stem: str):
    m = re.match(r"^(.+?)-(q\d_[0-9a-z_]+|f16)$", stem)
    if not m or m.group(1) not in MODELS or m.group(2) not in FILE_TYPES:
        return None
    return from_config(MODELS[m.group(1)], m.group(2))


def predict(run: Path, measurements: list[dict]) -> tuple[list[dict], list[dict]]:
    gpus = {f"gpu{g['index']}": g for g in nvidia_gpus(run)}
    preds, unsup = [], []
    for meas in measurements:
        key, c = meas["key"], meas.get("conditions", {})
        parts = key.split("/")
        try:
            if parts[0] == "llama-bench" and parts[1] == "cuda":
                g = gpus.get(parts[2])
                mid = g and catalogue_id(g["name"], g["mib"])
                mdl = model_for(parts[3])
                if not g:
                    unsup.append(dict(key=key, reason="device not in the run's GPU list")); continue
                if not mid:
                    unsup.append(dict(key=key, reason=f"GPU not in rigsim's catalogue: {g['name']}")); continue
                if not mdl:
                    unsup.append(dict(key=key, reason=f"model architecture not described: {parts[3]}")); continue
                r = cuda.llama_bench(machine_for(run, g, mid), "gpu0", mdl, parts[4])
                preds.append(dict(key=key, value=r["tokens_per_s"], unit="tok/s", calibration=False,
                                  basis=f"llama.cpp CUDA engine, {mid}, kernel factors unmeasured; "
                                        + ", ".join(f"{k} {v:.0%}" for k, v in r["share"].items())))
            elif parts[0] == "llama-bench":
                unsup.append(dict(key=key, reason=f"llama.cpp {parts[1]} backend not modelled yet"))
            elif parts[0] == "gpu-vram" and len(parts) == 3 and parts[2] == "read":
                g = gpus.get(parts[1])
                mid = g and catalogue_id(g["name"], g["mib"])
                if not mid:
                    unsup.append(dict(key=key, reason="GPU not in rigsim's catalogue" + (f": {g['name']}" if g else ""))); continue
                m = machine_for(run, g, mid)
                r = execute(m, "gpu0", [Op("stream read", "memory", bytes=4e9)], cuda.RUNTIME)
                preds.append(dict(key=key, value=r.vram_bps / 1e9, unit="GB/s", calibration=False,
                                  basis=f"{mid}: {r.basis['vram']}"))
            else:
                unsup.append(dict(key=key, reason="not modelled yet"))
        except Exception as e:                       # one value failing must not lose the others
            unsup.append(dict(key=key, reason=f"rigsim error: {type(e).__name__}: {e}"))
    return preds, unsup


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    meas = json.loads(a.measurements.read_text())["measurements"]
    preds, unsup = predict(a.run, meas)
    a.out.write_text(json.dumps(dict(predictions=preds, unsupported=unsup), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
