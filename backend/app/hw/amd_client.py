"""AMD Ryzen desktop processors, Ryzen 1000 to 7000 (Summit Ridge .. Raphael, 2017-2022, the age range of Intel's
6th-13th gen): what the block model needs, per family, from AMD's documentation (proposal §5; nothing fitted).

Two topologies:
  - monolithic: one die with cores and memory controller — Summit / Pinnacle Ridge (one "Zeppelin" die, two core
    complexes on it) and every APU (Raven Ridge, Picasso, Renoir, Cezanne);
  - chiplet: one to two core dies (CCD) linked to an I/O die with the memory controller — Matisse (Zen 2), Vermeer
    (Zen 3), Raphael (Zen 4). Each CCD link reads 32 bytes per fabric clock (the value cpus.py already uses for
    Zen 2-5, AMD / Chips and Cheese).
How many CCDs and cores a given SKU has is read from the machine (the L3 slices in `lscpu -e`: Zen 2 has two core
complexes with their own L3 per CCD, Zen 3 and 4 one), not from a SKU table. Two DDR channels throughout.

The fabric clock is not visible from Linux; the default rule AMD documents is used and stated in each prediction:
Zen 2 / Zen 3 run FCLK = memory clock (1:1) up to DDR4-3600 (1800 MHz); above, 1800 MHz is assumed. Zen 4 runs
1733 MHz at DDR5-5200 and up to 2000 MHz with DDR5-6000 profiles (AMD, R. Hallock, Aug 2022; igor'sLAB: in
between on Auto, e.g. ~1900 MHz at 5600) — linear in between here.

Sources: AMD product pages (system memory specification): Ryzen 1000 DDR4-2667, Ryzen 2000 DDR4-2933, Raven Ridge
and Picasso APUs DDR4-2933, Ryzen 3000 / 4000G / 5000 DDR4-3200, Ryzen 7000 DDR5-5200; core architecture per family.
"""
from __future__ import annotations

import math
import re

FAMILIES = {
    "summitridge": dict(core="zen", topology="monolithic", mem=dict(ddr4=2667)),
    "pinnacleridge": dict(core="zenplus", topology="monolithic", mem=dict(ddr4=2933)),
    "ravenridge": dict(core="zen", topology="monolithic", mem=dict(ddr4=2933)),
    "picasso": dict(core="zenplus", topology="monolithic", mem=dict(ddr4=2933)),
    "matisse": dict(core="zen2", topology="chiplet", ccx_per_ccd=2, mem=dict(ddr4=3200)),
    "renoir": dict(core="zen2", topology="monolithic", mem=dict(ddr4=3200)),
    "vermeer": dict(core="zen3", topology="chiplet", ccx_per_ccd=1, mem=dict(ddr4=3200)),
    "cezanne": dict(core="zen3", topology="monolithic", mem=dict(ddr4=3200)),
    "raphael": dict(core="zen4", topology="chiplet", ccx_per_ccd=1, mem=dict(ddr5=5200)),
}
CHANNELS = 2
CCD_READ_BYTES_PER_FCLK = 32
MODEL = re.compile(r"Ryzen [3579](?: PRO)? (\d{4})([A-Z0-9]*)")
MONOLITHIC_5000 = {5500, 5700}                     # Cezanne without graphics (no G suffix)
SOURCE = ("AMD product pages (memory per family, two channels, core architecture); CCD link 32 B/FCLK read; core and "
          "CCD counts from the machine's own inventory")


def family(cpu: str) -> str | None:
    """Family from a desktop CPU name ('AMD Ryzen 7 5800X 8-Core Processor', 'AMD Ryzen 5 5600G with Radeon
    Graphics'). Mobile parts (H, HS, HX, U suffixes) and Threadripper are not covered."""
    m = MODEL.search(cpu or "")
    if not m or "Threadripper" in cpu:
        return None
    num, suf = int(m.group(1)), m.group(2)
    if re.match(r"^(H|HS|HX|U)", suf):
        return None
    apu = suf.startswith("G")
    series = num // 1000
    if series == 1:
        return "summitridge"
    if series == 2:
        return "ravenridge" if apu else "pinnacleridge"
    if series == 3:
        return "picasso" if apu else "matisse"
    if series == 4:
        return "renoir"                            # 4x00G, and the 4100 / 4500 without graphics
    if series == 5:
        return "cezanne" if apu or (num in MONOLITHIC_5000 and not suf) else "vermeer"
    if series == 7 and not apu:
        return "raphael"
    return None


# TDP per desktop model (AMD product pages); the socket's sustained limit is PPT = 1.35 x TDP (AMD's documented
# AM4 / AM5 package power tracking). Models not listed: not specified.
TDP_W = {1800: 95, 1700: 65, 1600: 65, 1500: 65, 1400: 65, 1300: 65, 1200: 65,
         2700: 65, 2600: 65, 2400: 65, 2200: 65,
         3950: 105, 3900: 105, 3800: 105, 3700: 65, 3600: 65, 3500: 65, 3400: 65, 3300: 65, 3200: 65, 3100: 65,
         4600: 65, 4650: 65, 4700: 65, 4500: 65, 4100: 65,
         5950: 105, 5900: 105, 5800: 105, 5700: 65, 5600: 65, 5500: 65,
         7950: 170, 7900: 170, 7800: 120, 7700: 105, 7600: 105}
X_TDP = {1800: 95, 1700: 95, 1600: 95, 1500: 65, 2700: 105, 2600: 95, 3700: 65, 3600: 95, 3800: 105,
         5700: 65, 5600: 65, 7900: 170, 7700: 105, 7600: 105}
PPT_FACTOR = 1.35


def package_power_w(cpu: str) -> tuple[float | None, str]:
    """PPT (W) from the model's TDP, and where it comes from; the user may scale it."""
    m = MODEL.search(cpu or "")
    if not m or not family(cpu):
        return None, "package power not specified for this CPU"
    num, suf = int(m.group(1)), m.group(2)
    nonx = {7900: 65, 7700: 65, 7600: 65}
    if suf.startswith("X3D"):
        tdp = 120 if num // 1000 == 7 else 105
    elif suf.startswith("X"):
        tdp = X_TDP.get(num, TDP_W.get(num))
    elif suf.startswith("GE"):
        tdp = 35
    else:
        tdp = nonx.get(num, TDP_W.get(num))
    if not tdp:
        return None, f"TDP of the Ryzen {num}{suf} not in the table"
    return tdp * PPT_FACTOR, f"AMD: TDP {tdp} W, PPT = 1.35 x TDP"


def fclk_mhz(fam: str, mts: float) -> float | None:
    """AMD's default fabric clock for this memory speed (module docstring); None for monolithic parts."""
    core = FAMILIES[fam]["core"]
    if FAMILIES[fam]["topology"] != "chiplet":
        return None
    if core in ("zen2", "zen3"):
        return min(mts / 2, 1800.0)
    if mts <= 5200:
        return 1733.0
    return min(2000.0, 1733.0 + (mts - 5200) / 800 * 267)


def spec(cpu: str, cores: int, l3_slices: int) -> dict | None:
    """A cpus.py-style entry, with the core count and L3 slices (core complexes) the machine reports."""
    fam = family(cpu)
    if not fam or not cores:
        return None
    f = FAMILIES[fam]
    base = dict(arch=fam, family=fam, core_uarch=f["core"], channels=CHANNELS, memory_support=f["mem"], source=SOURCE)
    if f["topology"] == "monolithic":
        return dict(base, topology="monolithic", cores={f["core"]: cores})
    ccds = max(1, math.ceil(max(l3_slices, 1) / f["ccx_per_ccd"]))
    return dict(base, topology="chiplet", ccds=ccds, cores_per_ccd={f["core"]: cores // ccds},
                ccd_read_bytes_per_fclk=CCD_READ_BYTES_PER_FCLK)
