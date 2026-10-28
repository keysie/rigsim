"""rigsim's predictions for a measured run of the rigsim-benchmark stick (contract: rigsim-benchmark's
predictions/README.md). Run from backend/:

  python3 -m app.predict --run <run folder> --measurements <measurements.json> --out <out.json>

The machine is built from the run's own inventory: GPUs by nvidia-smi name and memory size (catalogue: hw/gpus.py),
power limits from nvidia-smi, PCIe link from inventory/pcie.json. What the model covers is predicted; everything
else comes back as unsupported with the reason. No measurement of these runs went into rigsim's constants, so no
prediction is marked as calibration. CPU memory (MLC idle latency, all-core read bandwidth): the CPU from the
catalogue (cpus.py, Intel Core 6th-13th gen from intel_client.py with the inventory's core counts), the memory from
dmidecode (type, configured speed, populated channels) and the SPD dump (tAA, die density)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .engine.llamacpp import cuda
from .engine.model import ARCHS, FILE_TYPES, from_config
from .hw import arch, intel_client
from .hw.cpus import lookup as cpu_lookup
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


# --- CPU and memory from the inventory ------------------------------------------------------------------------------
def cpu_name(run: Path) -> str:
    try:
        return json.loads((run / "run.json").read_text()).get("hardware", {}).get("cpu", "")
    except (OSError, ValueError):
        return ""


def core_counts(run: Path) -> dict[str, int]:
    """Cores by type from `lscpu -e`: a core with two hardware threads is a P-core (Hyper-Threading), one thread an
    E-core — on hybrid generations; spec() folds them into one type on the others."""
    f = run / "inventory" / "lscpu-e.txt"
    per_core: dict[str, int] = {}
    for line in (f.read_text(errors="replace").splitlines()[1:] if f.exists() else []):
        cols = line.split()
        if len(cols) >= 4 and cols[0].isdigit():
            key = f"{cols[2]}:{cols[3]}"                  # socket:core
            per_core[key] = per_core.get(key, 0) + 1
    p = sum(1 for n in per_core.values() if n >= 2)
    return {"p": p, "e": len(per_core) - p} if per_core else {}


def dimms(run: Path) -> list[dict]:
    """Populated memory devices from dmidecode -t memory: type, configured speed, size, locators, rank."""
    f = run / "inventory" / "dmidecode.txt"
    out, cur = [], None
    for line in (f.read_text(errors="replace").splitlines() if f.exists() else []):
        if line.strip() == "Memory Device":
            cur = {}; out.append(cur); continue
        if cur is None or ":" not in line:
            continue
        k, _, v = line.strip().partition(":")
        cur[k.strip()] = v.strip()
    res = []
    for d in out:
        size = d.get("Size", "")
        if not size or "No Module" in size or "Not Installed" in size:
            continue
        sp = re.search(r"(\d+)", d.get("Configured Memory Speed", "") or d.get("Speed", ""))
        res.append(dict(type=d.get("Type", ""), mts=int(sp.group(1)) if sp else None, size=size,
                        locator=d.get("Locator", ""), bank=d.get("Bank Locator", ""), rank=d.get("Rank", "")))
    return res


def spd_timing(run: Path) -> dict:
    """From the first SPD dump (inventory/spd-*.hex, serial bytes zeroed): tAAmin (ns) and die density (Gb), JEDEC
    base-profile values (DDR4: bytes 24 + 123, density byte 4 bits 3:0; DDR5: bytes 30-31 in ps, byte 4 bits 4:0)."""
    for f in sorted((run / "inventory").glob("spd-*.hex")):
        lines = [l for l in f.read_text().splitlines() if l and not l.startswith("#")]
        b = bytes.fromhex("".join(lines).replace(" ", ""))
        head = f.read_text().splitlines()[0]
        try:
            if "spd5118" in head and len(b) > 31:
                return dict(taa_ns=int.from_bytes(b[30:32], "little") / 1000,
                            density_gbit={1: 4, 2: 8, 3: 12, 4: 16, 5: 24, 6: 32, 7: 48, 8: 64}.get(b[4] & 0x1F), source=f.name)
            if "ee1004" in head and len(b) > 123:
                ftb = b[123] - 256 if b[123] > 127 else b[123]
                return dict(taa_ns=b[24] * 0.125 + ftb / 1000,
                            density_gbit={4: 4, 5: 8, 6: 16, 7: 32}.get(b[4] & 0x0F), source=f.name)
        except (ValueError, IndexError):
            continue
    return {}


def cpu_memory_machine(run: Path):
    """(machine, core groups, parameters, why-not) for the CPU and its memory, from the run's inventory."""
    name = cpu_name(run)
    spec = cpu_lookup(name) or intel_client.spec(name, core_counts(run))
    if not spec:
        return None, None, None, f"CPU not in rigsim's catalogue: {name or 'unknown'}"
    ds = dimms(run)
    if not ds:
        return None, None, None, "no DIMM data in the inventory (stick before 0.11.0: dmidecode read nothing)"
    t = ds[0]["type"].lower()
    mts = min(d["mts"] for d in ds if d["mts"]) if any(d["mts"] for d in ds) else None
    if t not in ("ddr4", "ddr5") or not mts:
        return None, None, None, f"memory not identified (type {ds[0]['type']!r}, speed {mts})"
    def channel(d: dict) -> str | None:
        """The channel letter a board names in its locators: 'Controller0-ChannelA', 'CHANNEL B', 'DDR5-A2', 'DIMM_B1'."""
        for txt in (d["bank"], d["locator"]):
            m = re.search(r"(?i)channel\s*([a-d])\b", txt) or re.search(r"(?i)(?:^|[-_ ])([a-d])\d$", txt)
            if m:
                return m.group(1).upper()
        return None
    chans = {c for c in map(channel, ds) if c}
    channels = min(spec["channels"], len(chans) if chans else len(ds))
    spd = spd_timing(run)
    cl = round(spd["taa_ns"] * mts / 2000) if spd.get("taa_ns") else None
    m, groups = arch.build(spec, mts, channels, cl=cl, mem_type=t, density_gbit=spd.get("density_gbit"))
    params = dict(cpu=spec.get("arch"), cores=spec.get("cores"), mem=f"{t.upper()}-{mts}", channels=channels,
                  cl=cl, density_gbit=spd.get("density_gbit"), channels_from="locators" if chans else "DIMM count")
    return m, groups, params, None


def predict(run: Path, measurements: list[dict]) -> tuple[list[dict], list[dict]]:
    gpus = {f"gpu{g['index']}": g for g in nvidia_gpus(run)}
    preds, unsup = [], []
    cpu_m = None
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
            elif key in ("mlc/idle_latency", "mlc/max_bandwidth/all_reads"):
                if cpu_m is None:
                    cpu_m = cpu_memory_machine(run)
                m, groups, params, why = cpu_m
                if why:
                    unsup.append(dict(key=key, reason=why)); continue
                basis = ", ".join(f"{k} {v}" for k, v in params.items() if v is not None)
                if key == "mlc/idle_latency":
                    cas = "CAS + fixed term" if params["cl"] else "fixed term only (CAS unknown: no SPD dump)"
                    preds.append(dict(key=key, value=arch.access_s(int(params["mem"].split("-")[1]), params["cl"]) * 1e9,
                                      unit="ns", calibration=False, basis=f"{cas}; {basis}"))
                else:
                    preds.append(dict(key=key, value=arch.all_cores_read_gbs(m, groups), unit="GB/s", calibration=False,
                                      basis=f"block model, JEDEC DRAM efficiency; {basis}"))
            elif parts[0] == "mlc":
                unsup.append(dict(key=key, reason="write traffic not modelled yet (reads only)"))
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
