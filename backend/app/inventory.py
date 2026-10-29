"""What a rigsim-benchmark run says about its machine, read from the run folder (format 1.x): GPUs (nvidia-smi
list, power limits, the PCIe tree), CPU (name, cores by type, L3 slices), memory (dmidecode, SPD), the iGPU's
clock. Facts only; system.py turns them into the machine's graph."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .engine.model import FILE_TYPES, from_config

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


def lscpu_topology(run: Path) -> dict:
    """Physical cores and L3 slices (core complexes) from `lscpu -e` (column L1d:L1i:L2:L3)."""
    f = run / "inventory" / "lscpu-e.txt"
    cores, l3 = set(), set()
    for line in (f.read_text(errors="replace").splitlines()[1:] if f.exists() else []):
        cols = line.split()
        if len(cols) >= 5 and cols[0].isdigit():
            cores.add(f"{cols[2]}:{cols[3]}")
            l3.add(f"{cols[2]}:{cols[4].split(':')[-1]}")
    return dict(cores=len(cores), l3_slices=len(l3))


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


def igpu_clock_mhz(run: Path, pci: str) -> float | None:
    """The highest graphics clock sampled for this device while the run measured (samples-drm.jsonl)."""
    f = run / "samples-drm.jsonl"
    best = None
    for line in (f.read_text(errors="replace").splitlines() if f.exists() else []):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("pci", "").lower().endswith(pci.lower()[-7:]) and r.get("gt_mhz"):
            best = max(best or 0, float(r["gt_mhz"]))
    return best


def gpu_names(run: Path) -> list[str]:
    try:
        return json.loads((run / "run.json").read_text()).get("hardware", {}).get("gpus", [])
    except (OSError, ValueError):
        return []


def max_clock_hz(run: Path) -> float | None:
    """The highest clock the machine reports for its cores (lscpu 'CPU max MHz')."""
    f = run / "inventory" / "lscpu.txt"
    m = re.search(r"^CPU max MHz:\s*([\d.]+)", f.read_text(errors="replace"), re.M) if f.exists() else None
    return float(m.group(1)) * 1e6 if m else None


def l3_bytes(run: Path) -> float | None:
    """The machine's total L3 (lscpu: 'L3 cache: 64 MiB (2 instances)')."""
    f = run / "inventory" / "lscpu.txt"
    m = re.search(r"^L3 cache:\s*([\d.]+)\s*(KiB|MiB|GiB|K|M)", f.read_text(errors="replace"), re.M) if f.exists() else None
    if not m:
        return None
    return float(m.group(1)) * {"K": 2**10, "KiB": 2**10, "M": 2**20, "MiB": 2**20, "GiB": 2**30}[m.group(2)]


def pcie_devices(run: Path) -> dict:
    """inventory/pcie.json (format 1.2+): every PCI device with parent, path, port type, link."""
    try:
        return json.loads((run / "inventory" / "pcie.json").read_text()).get("devices", {})
    except (OSError, ValueError):
        return {}
