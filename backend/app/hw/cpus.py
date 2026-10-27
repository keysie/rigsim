"""CPU models: what the topology needs, from spec sheets and architecture documentation (proposal §5).

Each entry names its source. `ccd_read_bytes_per_fclk` is the read width of one chiplet's link to the I/O die per
fabric clock (AMD Zen 2-5 desktop, architecture documentation); None for monolithic or tiled parts whose internal
link capacity is not documented (modelled as non-binding).
"""
from __future__ import annotations

CPUS = {
    "Ryzen 7 7800X3D": dict(arch="zen4", topology="chiplet", ccds=1, cores_per_ccd={"zen4": 8}, channels=2,
                            ccd_read_bytes_per_fclk=32,
                            source="AMD product page (8 cores, 1 CCD, 2 ch DDR5); Zen 4 IFOP 32 B/clk read (AMD, Chips and Cheese)"),
    "Ryzen 7 9800X3D": dict(arch="zen5", topology="chiplet", ccds=1, cores_per_ccd={"zen5": 8}, channels=2,
                            ccd_read_bytes_per_fclk=32,
                            source="AMD product page (8 cores, 1 CCD, 2 ch DDR5); Zen 5 IFOP 32 B/clk read (AMD, Chips and Cheese)"),
    "Ryzen 9 9950X3D": dict(arch="zen5", topology="chiplet", ccds=2, cores_per_ccd={"zen5": 8}, channels=2,
                            ccd_read_bytes_per_fclk=32, source="AMD product page (16 cores, 2 CCDs, 2 ch DDR5)"),
    "Core Ultra 9 285K": dict(arch="arrowlake", topology="monolithic", cores={"p": 8, "e": 16}, channels=2,
                              source="Intel ARK (8 P + 16 E, 2 ch DDR5); compute-to-SoC tile link not documented"),
    "Core Ultra 7 265K": dict(arch="arrowlake", topology="monolithic", cores={"p": 8, "e": 12}, channels=2,
                              source="Intel ARK (8 P + 12 E, 2 ch DDR5); compute-to-SoC tile link not documented"),
    "Ryzen 7 5800H": dict(arch="zen3-apu", topology="monolithic", cores={"zen3": 8}, channels=2,
                          source="AMD product page (8 cores, monolithic APU, 2 ch DDR4)"),
}


def lookup(name: str | None) -> dict | None:
    if not name:
        return None
    for k, v in CPUS.items():
        if name.startswith(k):
            return v
    return None
