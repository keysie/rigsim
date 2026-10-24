"""Plausibility checks for parts submitted to the catalog. Pure consistency: a part that passes is not
necessarily real, but an invented one has to be internally consistent and inside the envelope of its
vendor and year to get through."""
from __future__ import annotations

import difflib
import re

from .catalog.cpus import CPU_BY_ID, CPUS, FIELDS as CPU_FIELDS, RAM_SPEEDS
from .catalog.gpus import GPU_BY_ID, GPUS, GPU_FIELDS

VRAM_SIZES = {2, 3, 4, 6, 8, 10, 11, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 141, 180, 192, 256, 288}
GPU_VENDORS = {"NVIDIA": ("geforce", "rtx", "gtx", "quadro", "titan", "tesla", "a100", "h100", "h200", "b200", "l40", "a40", "a10", "a30"),
               "AMD": ("radeon", "instinct", "rx ", "pro ", "mi"), "Intel": ("arc", "data center", "gaudi", "max")}
CPU_VENDORS = {"Intel": ("core", "xeon", "pentium", "celeron", "ultra"), "AMD": ("ryzen", "epyc", "threadripper", "fx", "athlon")}
SOURCE_DOMAINS = ("nvidia.com", "amd.com", "intel.com")


def _dup(name: str, names: list[str], errors: list[str], warnings: list[str]) -> None:
    norm = re.sub(r"\s+", " ", name.lower()).strip()
    low = [re.sub(r"\s+", " ", n.lower()).strip() for n in names]
    if norm in low:
        errors.append(f"name '{name}' is already in the catalog")
        return
    m = difflib.get_close_matches(norm, low, n=1, cutoff=0.9)
    if m:
        warnings.append(f"name is close to '{m[0]}': make sure this is a different part, not a duplicate")


def check_gpu(spec: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for f in GPU_FIELDS + ["mem_bus_bits", "mem_gbps"]:
        if f not in spec:
            errors.append(f"missing field '{f}'")
    if errors:
        return errors, warnings
    name, vendor, year = str(spec["name"]), str(spec["vendor"]), int(spec["year"])
    if vendor not in GPU_VENDORS:
        errors.append(f"vendor must be one of {sorted(GPU_VENDORS)}")
    elif not any(k in name.lower() for k in GPU_VENDORS[vendor]):
        warnings.append(f"name '{name}' does not look like a {vendor} product line")
    if not 2010 <= year <= 2027:
        errors.append("year must be between 2010 and 2027")
    if spec["id"] in GPU_BY_ID:
        errors.append(f"id '{spec['id']}' exists")
    _dup(name, [g[1] for g in GPUS], errors, warnings)
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", str(spec["id"])):
        errors.append("id must be lowercase letters, digits, dots and dashes")
    if int(spec["vram_gb"]) not in VRAM_SIZES:
        errors.append(f"vram_gb {spec['vram_gb']} is not a size that exists ({sorted(VRAM_SIZES)})")
    bw = float(spec["mem_bw_gbs"])
    calc = float(spec["mem_bus_bits"]) * float(spec["mem_gbps"]) / 8
    if calc <= 0 or abs(bw - calc) / calc > 0.03:
        errors.append(f"mem_bw_gbs {bw} does not match bus x rate / 8 = {calc:.0f} GB/s")
    if not 50 <= bw <= 9000:
        errors.append("mem_bw_gbs outside 50–9,000 GB/s")
    tf = float(spec["fp16_tflops"])
    env = {2010: 20, 2016: 60, 2018: 200, 2020: 400, 2022: 1000, 2024: 2500}
    cap = max(v for y, v in env.items() if year >= y)
    if tf > cap:
        errors.append(f"fp16_tflops {tf} is above what any {year} card reaches (~{cap})")
    if tf > bw * 0.6:
        warnings.append("fp16 TFLOPS per GB/s of bandwidth is unusually high for a graphics card")
    if int(spec["pcie_gen"]) > (3 if year < 2020 else 4 if year < 2022 else 5):
        errors.append("pcie_gen is newer than the card's year allows")
    if int(spec["pcie_lanes"]) not in (4, 8, 16):
        errors.append("pcie_lanes must be 4, 8 or 16")
    if float(spec["nvlink_gbs"]) > 0 and vendor != "NVIDIA" and vendor != "AMD":
        errors.append("nvlink_gbs only on NVIDIA (NVLink) or AMD (Infinity Fabric) parts")
    if not 30 <= float(spec["tdp_w"]) <= 1500:
        errors.append("tdp_w outside 30–1,500 W")
    if str(spec["form"]) not in ("pcie", "sxm", "oam", "soc"):
        errors.append("form must be pcie, sxm, oam or soc")
    if int(spec["slots"]) not in (0, 1, 2, 3, 4):
        errors.append("slots must be 0–4")
    return errors, warnings


def check_cpu(spec: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for f in CPU_FIELDS:
        if f not in spec:
            errors.append(f"missing field '{f}'")
    if errors:
        return errors, warnings
    name, vendor, year = str(spec["name"]), str(spec["vendor"]), int(spec["year"])
    if vendor not in CPU_VENDORS:
        errors.append(f"vendor must be one of {sorted(CPU_VENDORS)}")
    elif not any(k in name.lower() for k in CPU_VENDORS[vendor]):
        warnings.append(f"name '{name}' does not look like a {vendor} product line")
    if not 2010 <= year <= 2027:
        errors.append("year must be between 2010 and 2027")
    if spec["id"] in CPU_BY_ID:
        errors.append(f"id '{spec['id']}' exists")
    _dup(name, [c[1] for c in CPUS], errors, warnings)
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", str(spec["id"])):
        errors.append("id must be lowercase letters, digits, dots and dashes")
    cores, threads = int(spec["cores"]), int(spec["threads"])
    if not 1 <= cores <= 256 or threads not in (cores, 2 * cores):
        errors.append("threads must equal cores or 2 x cores, cores 1–256")
    if int(spec["mem_channels"]) not in (1, 2, 4, 6, 8, 12):
        errors.append("mem_channels must be 1, 2, 4, 6, 8 or 12")
    if spec["mem_type"] not in RAM_SPEEDS or int(spec["max_mem_mts"]) not in RAM_SPEEDS[spec["mem_type"]]:
        errors.append(f"max_mem_mts must be a standard {spec.get('mem_type')} speed")
    lanes = int(spec["pcie_lanes"])
    if not 0 <= lanes <= 160:
        errors.append("pcie_lanes outside 0–160")
    if int(spec["pcie_gen"]) > (3 if year < 2019 else 4 if year < 2021 else 5):
        errors.append("pcie_gen is newer than the part's year allows")
    if int(spec["sockets"]) not in (1, 2, 4, 8):
        errors.append("sockets must be 1, 2, 4 or 8")
    isa = str(spec["isa"])
    if isa not in ("sse", "avx", "avx2", "avx512", "amx"):
        errors.append("isa must be sse, avx, avx2, avx512 or amx")
    if isa == "avx512" and year < 2016:
        errors.append("AVX-512 did not exist before 2016")
    if isa == "amx" and (vendor != "Intel" or year < 2023):
        errors.append("AMX only on Intel Xeon from 2023")
    if bool(spec["avx512"]) != (isa in ("avx512", "amx")):
        errors.append("avx512 flag must agree with isa")
    e = int(spec["e_cores"])
    if e and (vendor != "Intel" or year < 2021):
        errors.append("efficiency cores only on Intel from 2021")
    if e >= cores:
        errors.append("e_cores must be fewer than cores")
    if not 1.5 <= float(spec["all_core_ghz"]) <= 6.5:
        errors.append("all_core_ghz outside 1.5–6.5")
    if not 15 <= float(spec["tdp_w"]) <= 700:
        errors.append("tdp_w outside 15–700 W")
    return errors, warnings


def check_source(url: str, name: str, fetch: bool = False) -> tuple[list[str], list[str]]:
    """The spec-sheet URL must be on the vendor's own domain; with fetch=True the page
    must exist and mention the part's name."""
    errors, warnings = [], []
    m = re.match(r"https?://([^/]+)/", url + "/")
    host = m.group(1).lower() if m else ""
    if not any(host == d or host.endswith("." + d) for d in SOURCE_DOMAINS):
        errors.append(f"source must be on one of {', '.join(SOURCE_DOMAINS)} (got '{host or url}')")
        return errors, warnings
    if fetch:
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rigsim-intake/1"})
            text = urllib.request.urlopen(req, timeout=20).read(2_000_000).decode("utf-8", "replace").lower()
        except Exception as e:  # noqa: BLE001
            errors.append(f"source page could not be fetched: {e}")
            return errors, warnings
        key = re.sub(r"\s+", " ", name.lower())
        tokens = [t for t in re.split(r"[^a-z0-9]+", key) if len(t) >= 2 and t not in ("geforce", "radeon", "core", "nvidia", "amd", "intel", "gb")]
        if not all(t in text for t in tokens):
            errors.append(f"source page does not mention '{name}' (looked for {', '.join(tokens)})")
    return errors, warnings
