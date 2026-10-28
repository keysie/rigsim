"""rigsim's predictions for a measured run of the rigsim-benchmark stick (contract: rigsim-benchmark's
predictions/README.md). Run from backend/:

  python3 -m app.predict --run <run folder> --measurements <measurements.json> --out <out.json>

The machine is built from the run's own inventory: GPUs by nvidia-smi name and memory size (catalogue: hw/gpus.py),
power limits from nvidia-smi, PCIe link from inventory/pcie.json. What the model covers is predicted; everything
else comes back as unsupported with the reason. No measurement of these runs went into rigsim's constants, so no
prediction is marked as calibration. CPU memory (MLC idle latency, all-core read bandwidth): the CPU from the
catalogue (cpus.py; Intel Core 6th-13th gen from intel_client.py, AMD Ryzen 1000-7000 from amd_client.py, both
with the inventory's core counts), the memory from
dmidecode (type, configured speed, populated channels) and the SPD dump (tAA, die density)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .engine import mlc
from .engine.llamacpp import cuda, vulkan
from .engine.model import ARCHS, FILE_TYPES, from_config
from .hw.execute import Op, execute
from .hw.machine import Read
from .hw.gpu import add_gpu
from .hw.gpus import GPUS
from .hw.machine import Machine
from .hw.primitives import Router

from .inventory import *  # noqa: F401,F403  (the run-reading functions; tests and older callers use them here)
from .inventory import model_for
from .system import System, build_system


def cpu_memory_machine(run: Path):
    """(machine, core groups, parameters, why-not): the run's machine, for the CPU-and-memory questions."""
    s = build_system(run)
    return (None, None, None, s.cpu_why) if s.cpu_why else (s.m, s.groups, s.cpu, None)


def _basis_engine(name: str, s: System, gid: str, r: dict) -> str:
    g = s.gpus[gid]
    return (f"llama.cpp {name} engine on {g['model']} ({gid}, {g['pci']}), kernel factors unmeasured; "
            + ", ".join(f"{k} {v:.0%}" for k, v in r["share"].items()))


def predict(run: Path, measurements: list[dict]) -> tuple[list[dict], list[dict]]:
    """Every prediction is a question to ONE graph of the run's machine (system.build_system)."""
    s = build_system(run)
    preds, unsup = [], []

    def gpu_for(backend: str, dev: str, c: dict) -> tuple[str | None, str | None]:
        if backend == "cuda" or dev.startswith("gpu"):
            gid = s.cuda.get(int(dev[3:])) if dev[3:].isdigit() else None
        else:
            gid = s.by_pci.get(c.get("pci", ""))
            if not gid:
                return None, f"Vulkan device not a catalogued GPU ({c.get('gpu') or c.get('pci') or 'unknown'})"
        if not gid:
            return None, "device not in the run's GPU list"
        if not s.gpus[gid]["model"]:
            return None, f"GPU not in rigsim's catalogue: {s.gpus[gid]['name']}"
        if gid not in s.m.devices:
            return None, "GPU not in the machine's graph"
        return gid, None

    for meas in measurements:
        key, c = meas["key"], meas.get("conditions", {})
        parts = key.split("/")
        try:
            if parts[0] == "llama-bench" and parts[1] in ("cuda", "vulkan"):
                gid, why = gpu_for(parts[1], parts[2], c)
                mdl = model_for(parts[3])
                if why:
                    unsup.append(dict(key=key, reason=why)); continue
                if not mdl:
                    unsup.append(dict(key=key, reason=f"model architecture not described: {parts[3]}")); continue
                eng = cuda if parts[1] == "cuda" else vulkan
                r = eng.llama_bench(s.m, gid, mdl, parts[4])
                preds.append(dict(key=key, value=r["tokens_per_s"], unit="tok/s", calibration=False,
                                  basis=_basis_engine("CUDA" if eng is cuda else "Vulkan", s, gid, r)))
            elif parts[0] == "llama-bench":
                unsup.append(dict(key=key, reason=f"llama.cpp {parts[1]} backend not modelled yet"))
            elif parts[0] == "gpu-vram" and len(parts) == 3 and parts[2] in ("read", "write", "copy"):
                gid, why = gpu_for("cuda", parts[1], c)
                if why:
                    unsup.append(dict(key=key, reason=why)); continue
                r = execute(s.m, gid, [Op("stream", "memory", bytes=4e9)], cuda.RUNTIME)
                note = "" if parts[2] == "read" else "; read and write treated alike (bus turnaround not modelled)"
                preds.append(dict(key=key, value=r.vram_bps / 1e9, unit="GB/s", calibration=False,
                                  basis=f"{s.gpus[gid]['model']}: {r.basis['vram']}{note}"))
            elif parts[0] == "mlc" and (key == "mlc/idle_latency" or (parts[1] == "max_bandwidth" and parts[2] in mlc.MIX)):
                if s.cpu_why:
                    unsup.append(dict(key=key, reason=s.cpu_why)); continue
                basis = ", ".join(f"{k} {v}" for k, v in s.cpu.items() if v is not None and k != "name")
                if key == "mlc/idle_latency":
                    cas = "CAS + fixed term" if s.cpu["cl"] else "fixed term only (CAS unknown: no SPD dump)"
                    preds.append(dict(key=key, value=mlc.idle_latency_s(s.m, s.groups[0]) * 1e9, unit="ns",
                                      calibration=False, basis=f"MLC engine: dependent reads {s.groups[0]} <- ram, "
                                                               f"path latency in the graph, {cas}; {basis}"))
                else:
                    note = "" if parts[2] == "all_reads" else f"; {mlc.NOT_IN_GRAPH}"
                    preds.append(dict(key=key, value=mlc.bandwidth_bps(s.m, s.groups, parts[2]) / 1e9, unit="GB/s",
                                      calibration=False, basis=f"MLC engine: every core group at once, {parts[2]}; "
                                                               f"{basis}{note}"))
            elif parts[0] == "mlc":
                unsup.append(dict(key=key, reason="this MLC test is not modelled yet"))
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
