"""DRAM read efficiency from JEDEC timings (proposal §5, spec-sheet first).

Per (sub-)channel, two documented losses:
  - bank-group gaps: a line takes `burst` clocks on the bus, but two reads to the same bank group must be tCCD_L =
    max(min clocks, 5 ns) apart; the controller avoids that, but not always: back-to-back same-group reads happen at
    DRAM_SAME_BANK_GROUP_SHARE x the random rate (1 / bank groups) — the one global constant here.
  - refresh (all-bank, the JEDEC default; a controlled experiment supports it, docs/overhaul/yardsticks.md): the
    channel stalls tRFC every tREFI.
Stated timings (a source's or the user's tREFI / tRFC) replace the JEDEC defaults. Where tRFC is not known (a
density JEDEC lists as TBD and nothing stated), the refresh term is not identified and the result says so.

Sources: JESD79-4 (DDR4) and JESD79-5 (DDR5) as published in vendor datasheets and Micron's DDR5 white paper:
DDR4 BL8 on 64 bits (4 clocks per line), 4 bank groups, tCCD_L max(5 nCK, 5 ns) from 2400 MT/s, tREFI 7.8 us,
tRFC1 260 / 350 / 550 ns (4 / 8 / 16 Gb); DDR5 BL16 on 32 bits (8 clocks per line), 8 bank groups, tCCD_L
max(8 nCK, 5 ns), tREFI 3.9 us, tRFC1 195 / 295 ns (8 / 16 Gb). Temperatures below 85 °C.
"""
from __future__ import annotations

import math

from . import constants as K

JEDEC = {
    "ddr4": dict(bank_groups=4, burst_nck=4, ccd_l_min_nck=5, ccd_l_ns=5.0, trefi_ns=7800.0,
                 trfc_ns={4: 260.0, 8: 350.0, 16: 550.0}),
    "ddr5": dict(bank_groups=8, burst_nck=8, ccd_l_min_nck=8, ccd_l_ns=5.0, trefi_ns=3900.0,
                 trfc_ns={8: 195.0, 16: 295.0}),
}


def efficiency(mem_type: str, mts: float, density_gbit: int | None = None, trefi_ns: float | None = None,
               trfc_ns: float | None = None, same_bg_share: float | None = None) -> tuple[float, list[str]]:
    """Sustained read bandwidth / peak for one (sub-)channel, and notes on what was not identified."""
    j = JEDEC[mem_type.lower()]
    s = K.DRAM_SAME_BANK_GROUP_SHARE if same_bg_share is None else same_bg_share
    tck = 2000.0 / mts                                       # ns per memory clock (MT/s / 2)
    ccd_l = max(j["ccd_l_min_nck"], math.ceil(j["ccd_l_ns"] / tck - 1e-9))
    gap = max(0, ccd_l - j["burst_nck"])
    cmd = j["burst_nck"] / (j["burst_nck"] + s / j["bank_groups"] * gap)
    notes = []
    trfc = trfc_ns if trfc_ns is not None else j["trfc_ns"].get(density_gbit)
    trefi = trefi_ns if trefi_ns is not None else j["trefi_ns"]
    if trfc is None:
        notes.append("refresh not identified (tRFC unknown for this density)")
        ref = 1.0
    else:
        ref = 1.0 - trfc / trefi
    return cmd * ref, notes


def clocks_to_ns(clocks: float, mts: float) -> float:
    """Timings stated in memory clocks (BIOS / OC tools) to ns."""
    return clocks * 2000.0 / mts
