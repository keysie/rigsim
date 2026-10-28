"""Intel integrated graphics of the Core desktop processors, 6th to 13th generation (Gen9 HD 510/530 .. Xe-LP
UHD 770), as a GPU of the hardware layer — from Intel's architecture documentation; nothing fitted.

Per execution unit (EU) and clock, from Intel's architecture papers: Gen9 / Gen9.5 ("The Compute Architecture of
Intel Processor Graphics Gen9"): two SIMD-4 FPUs with FMA = 16 fp32 FLOPs, 32 fp16; Gen12 Xe-LP (Intel Xe
architecture, Xe-LP): 8-wide ALUs, 16 fp32 FLOPs, 32 fp16, 64 int8 ops (DP4A). No matrix (XMX) units on either.
EU counts per product, from Intel's product pages: HD 510 / 610 12, HD 530 / 630 and UHD 630 24, UHD 710 16,
UHD 730 24, UHD 750 32, UHD 770 32.

The memory is the system's: the iGPU reads the DDR channels through the CPU's ring and memory controller. Its
"VRAM" is built from the machine's memory as the run records it — channels, data rate and the JEDEC DRAM efficiency
(dram.py) — as 32-bit units (a DDR4 channel is two, a DDR5 channel its two sub-channels). The ring / graphics
interface capacity is not documented: non-binding. The clock the GPU ran at is the run's own record (the
highest graphics clock sampled while measuring), not a spec-sheet law — the maximum differs between SKUs and is
not on the run's other records."""
from __future__ import annotations

import re

from . import dram
from .gpus import GPUS

GEN9 = {"fp32": 16, "fp16": 32}
XE_LP = {"fp32": 16, "fp16": 32, "int8": 64}
PRODUCTS = {               # marketing name -> (architecture, EUs, rates per EU per clock)
    "HD Graphics 510": ("gen9", 12, GEN9), "HD Graphics 530": ("gen9", 24, GEN9),
    "HD Graphics 610": ("gen9.5", 12, GEN9), "HD Graphics 630": ("gen9.5", 24, GEN9),
    "UHD Graphics 610": ("gen9.5", 12, GEN9), "UHD Graphics 630": ("gen9.5", 24, GEN9),
    "UHD Graphics 710": ("xe-lp", 16, XE_LP), "UHD Graphics 730": ("xe-lp", 24, XE_LP),
    "UHD Graphics 750": ("xe-lp", 32, XE_LP), "UHD Graphics 770": ("xe-lp", 32, XE_LP),
}
SOURCE = ("Intel Gen9 compute architecture paper and Xe architecture (Xe-LP) documentation: rates per EU; EU counts "
          "from Intel's product pages; memory = the machine's DDR channels with JEDEC DRAM efficiency; clock as sampled "
          "in the run")


def product(name: str) -> str | None:
    """'Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770] [8086:a780]' -> 'UHD Graphics 770'."""
    for p in sorted(PRODUCTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(p)}\b", name or ""):
            return p
    return None


def register(name: str, clock_mhz: float, mem_type: str, mts: float, channels: int,
             density_gbit: int | None = None) -> str | None:
    """Adds this machine's iGPU to the GPU catalogue (GPUS) under a key naming its memory, and returns the key."""
    prod = product(name)
    if not prod or not clock_mhz or not mts:
        return None
    arch, eus, rates = PRODUCTS[prod]
    t = mem_type.lower()
    eff, _ = dram.efficiency(t, mts, density_gbit) if t in dram.JEDEC else (1.0, [])
    key = f"intel-{prod.lower().replace(' ', '-')}@{t}-{int(mts)}x{channels}"
    GPUS[key] = dict(arch=arch, die=prod, sms=eus, boost_mhz=clock_mhz, fixed_clock_mhz=clock_mhz, tdp_w=1.0,
                     vram_gb=0, vram_gbs=channels * mts * 8 / 1000, base_mhz=None, gpcs=None, l2_mb=0,
                     mem_type=mem_type, mem_controllers=channels * 2, mem_gbps=mts / 1000 * eff, copy_engines=None,
                     pcie_gen=4, pcie_lanes=16, integrated=True, matrix_units=False,
                     ops_per_clk_per_sm={"fp32": rates["fp32"], "fp16": rates["fp16"], **({"int8": rates["int8"]} if "int8" in rates else {})},
                     arch_source=f"{prod}: {eus} EUs, {arch}", source=SOURCE)
    return key
