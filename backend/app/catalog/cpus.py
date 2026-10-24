"""CPU catalog. Values are public spec-sheet figures, rounded; the memory speed is the
highest officially supported JEDEC speed. `pcie_lanes` counts CPU lanes only (chipset lanes
are not modelled). `sockets` is the number of CPUs the part can be paired as."""

CPUS = [
    # id, name, vendor, year, cores, threads, mem channels, max MT/s, mem type, PCIe gen, lanes, sockets, TDP W, AVX-512,
    # vector ISA, efficiency cores, all-core GHz under vector load
    ("i7-2600k", "Core i7-2600K", "Intel", 2011, 4, 8, 2, 1333, "DDR3", 2, 16, 1, 95, False, "avx", 0, 3.5),
    ("i7-3930k", "Core i7-3930K", "Intel", 2011, 6, 12, 4, 1600, "DDR3", 3, 40, 1, 130, False, "avx", 0, 3.6),
    ("i7-4790k", "Core i7-4790K", "Intel", 2014, 4, 8, 2, 1600, "DDR3", 3, 16, 1, 88, False, "avx2", 0, 4.2),
    ("i7-5960x", "Core i7-5960X", "Intel", 2014, 8, 16, 4, 2133, "DDR4", 3, 40, 1, 140, False, "avx2", 0, 3.3),
    ("i7-6700k", "Core i7-6700K", "Intel", 2015, 4, 8, 2, 2133, "DDR4", 3, 16, 1, 91, False, "avx2", 0, 4.0),
    ("i7-6950x", "Core i7-6950X", "Intel", 2016, 10, 20, 4, 2400, "DDR4", 3, 40, 1, 140, False, "avx2", 0, 3.4),
    ("i7-8700k", "Core i7-8700K", "Intel", 2017, 6, 12, 2, 2666, "DDR4", 3, 16, 1, 95, False, "avx2", 0, 4.3),
    ("i9-7980xe", "Core i9-7980XE", "Intel", 2017, 18, 36, 4, 2666, "DDR4", 3, 44, 1, 165, True, "avx512", 0, 3.4),
    ("i9-9900k", "Core i9-9900K", "Intel", 2018, 8, 16, 2, 2666, "DDR4", 3, 16, 1, 95, False, "avx2", 0, 4.7),
    ("i9-10900x", "Core i9-10900X", "Intel", 2019, 10, 20, 4, 2933, "DDR4", 3, 48, 1, 165, True, "avx512", 0, 3.5),
    ("i9-10980xe", "Core i9-10980XE", "Intel", 2019, 18, 36, 4, 2933, "DDR4", 3, 48, 1, 165, True, "avx512", 0, 3.4),
    ("i9-11900k", "Core i9-11900K", "Intel", 2021, 8, 16, 2, 3200, "DDR4", 4, 20, 1, 125, True, "avx512", 0, 4.6),
    ("i7-12700-ddr4", "Core i7-12700 (DDR4 board)", "Intel", 2021, 12, 20, 2, 3200, "DDR4", 5, 20, 1, 65, False, "avx2", 4, 4.5),
    ("i9-12900k", "Core i9-12900K", "Intel", 2021, 16, 24, 2, 4800, "DDR5", 5, 20, 1, 125, False, "avx2", 8, 4.9),
    ("i9-13900k", "Core i9-13900K", "Intel", 2022, 24, 32, 2, 5600, "DDR5", 5, 20, 1, 125, False, "avx2", 16, 5.3),
    ("i9-14900k", "Core i9-14900K", "Intel", 2023, 24, 32, 2, 5600, "DDR5", 5, 20, 1, 125, False, "avx2", 16, 5.4),
    ("ultra9-285k", "Core Ultra 9 285K", "Intel", 2024, 24, 24, 2, 6400, "DDR5", 5, 24, 1, 125, False, "avx2", 16, 5.2),
    ("xeon-e5-2690v4", "Xeon E5-2690 v4", "Intel", 2016, 14, 28, 4, 2400, "DDR4", 3, 40, 2, 135, False, "avx2", 0, 3.0),
    ("xeon-e5-2696v4", "Xeon E5-2696 v4", "Intel", 2016, 22, 44, 4, 2400, "DDR4", 3, 40, 2, 150, False, "avx2", 0, 2.6),
    ("xeon-gold-6248", "Xeon Gold 6248", "Intel", 2019, 20, 40, 6, 2933, "DDR4", 3, 48, 2, 150, True, "avx512", 0, 2.9),
    ("xeon-w-3175x", "Xeon W-3175X", "Intel", 2018, 28, 56, 6, 2666, "DDR4", 3, 48, 1, 255, True, "avx512", 0, 3.1),
    ("xeon-w9-3495x", "Xeon w9-3495X", "Intel", 2023, 56, 112, 8, 4800, "DDR5", 5, 112, 1, 350, True, "amx", 0, 3.5),
    ("xeon-8480plus", "Xeon Platinum 8480+", "Intel", 2023, 56, 112, 8, 4800, "DDR5", 5, 80, 2, 350, True, "amx", 0, 3.0),
    ("xeon-6980p", "Xeon 6980P", "Intel", 2024, 128, 256, 12, 6400, "DDR5", 5, 96, 2, 500, True, "amx", 0, 2.9),
    ("fx-8350", "FX-8350 (990FX board: chipset lanes)", "AMD", 2012, 8, 8, 2, 1866, "DDR3", 2, 32, 1, 125, False, "avx", 0, 4.0),
    ("ryzen7-1800x", "Ryzen 7 1800X", "AMD", 2017, 8, 16, 2, 2667, "DDR4", 3, 20, 1, 95, False, "avx2", 0, 3.7),
    ("ryzen9-3950x", "Ryzen 9 3950X", "AMD", 2019, 16, 32, 2, 3200, "DDR4", 4, 20, 1, 105, False, "avx2", 0, 3.9),
    ("ryzen9-5950x", "Ryzen 9 5950X", "AMD", 2020, 16, 32, 2, 3200, "DDR4", 4, 20, 1, 105, False, "avx2", 0, 4.1),
    ("ryzen9-7950x", "Ryzen 9 7950X", "AMD", 2022, 16, 32, 2, 5200, "DDR5", 5, 24, 1, 170, True, "avx512", 0, 5.0),
    ("ryzen9-9950x3d", "Ryzen 9 9950X3D", "AMD", 2025, 16, 32, 2, 5600, "DDR5", 5, 24, 1, 170, True, "avx512", 0, 5.1),
    ("ryzen9-9950x", "Ryzen 9 9950X", "AMD", 2024, 16, 32, 2, 5600, "DDR5", 5, 24, 1, 170, True, "avx512", 0, 5.1),
    ("tr-1950x", "Threadripper 1950X", "AMD", 2017, 16, 32, 4, 2667, "DDR4", 3, 60, 1, 180, False, "avx2", 0, 3.6),
    ("tr-3960x", "Threadripper 3960X", "AMD", 2019, 24, 48, 4, 3200, "DDR4", 4, 64, 1, 280, False, "avx2", 0, 4.0),
    ("tr-3970x", "Threadripper 3970X", "AMD", 2019, 32, 64, 4, 3200, "DDR4", 4, 64, 1, 280, False, "avx2", 0, 3.9),
    ("tr-3990x", "Threadripper 3990X", "AMD", 2020, 64, 128, 4, 3200, "DDR4", 4, 64, 1, 280, False, "avx2", 0, 3.5),
    ("tr-7980x", "Threadripper 7980X", "AMD", 2023, 64, 128, 4, 5200, "DDR5", 5, 48, 1, 350, True, "avx512", 0, 4.6),
    ("tr-pro-7995wx", "Threadripper PRO 7995WX", "AMD", 2023, 96, 192, 8, 5200, "DDR5", 5, 128, 1, 350, True, "avx512", 0, 4.2),
    ("tr-pro-9995wx", "Threadripper PRO 9995WX", "AMD", 2025, 96, 192, 8, 6400, "DDR5", 5, 128, 1, 350, True, "avx512", 0, 4.5),
    ("epyc-7601", "EPYC 7601", "AMD", 2017, 32, 64, 8, 2667, "DDR4", 3, 128, 2, 180, False, "avx2", 0, 2.7),
    ("epyc-7742", "EPYC 7742", "AMD", 2019, 64, 128, 8, 3200, "DDR4", 4, 128, 2, 225, False, "avx2", 0, 3.0),
    ("epyc-7453", "EPYC 7453", "AMD", 2021, 28, 56, 8, 3200, "DDR4", 4, 128, 2, 225, False, "avx2", 0, 3.2),
    ("epyc-7313p", "EPYC 7313P", "AMD", 2021, 16, 32, 8, 3200, "DDR4", 4, 128, 1, 155, False, "avx2", 0, 3.5),
    ("epyc-7763", "EPYC 7763", "AMD", 2021, 64, 128, 8, 3200, "DDR4", 4, 128, 2, 280, False, "avx2", 0, 3.2),
    ("epyc-9654", "EPYC 9654", "AMD", 2022, 96, 192, 12, 4800, "DDR5", 5, 128, 2, 360, True, "avx512", 0, 3.4),
    ("epyc-9754", "EPYC 9754", "AMD", 2023, 128, 256, 12, 4800, "DDR5", 5, 128, 2, 360, True, "avx512", 0, 3.0),
    ("epyc-9965", "EPYC 9965", "AMD", 2024, 192, 384, 12, 6000, "DDR5", 5, 128, 2, 500, True, "avx512", 0, 3.3),
    # systems on a chip with unified memory (fixed bandwidth below in SOC_RAM_BW)
    ("gb10", "NVIDIA GB10 (DGX Spark)", "NVIDIA", 2025, 20, 20, 8, 8533, "LPDDR5X", 5, 4, 1, 140, False, "neon", 10, 3.6),
    ("apple-m4-max", "Apple M4 Max", "Apple", 2024, 16, 16, 8, 8533, "LPDDR5X", 4, 8, 1, 90, False, "neon", 4, 4.0),
    ("apple-m3-ultra", "Apple M3 Ultra", "Apple", 2025, 32, 32, 8, 8533, "LPDDR5X", 4, 16, 1, 270, False, "neon", 8, 3.8),
    ("apple-m2-ultra", "Apple M2 Ultra (Mac Pro)", "Apple", 2023, 24, 24, 8, 6400, "LPDDR5", 4, 16, 1, 295, False, "neon", 8, 3.5),
    ("apple-m1-ultra", "Apple M1 Ultra", "Apple", 2022, 20, 20, 8, 6400, "LPDDR5", 4, 16, 1, 215, False, "neon", 4, 3.2),
    ("apple-m5-max", "Apple M5 Max", "Apple", 2026, 18, 18, 8, 8533, "LPDDR5X", 5, 8, 1, 100, False, "neon", 0, 4.2),
    ("apple-m5-ultra", "Apple M5 Ultra", "Apple", 2026, 36, 36, 8, 8533, "LPDDR5X", 5, 16, 1, 300, False, "neon", 0, 4.0),
    ("amd-ai-max-395", "Ryzen AI Max+ 395 (Strix Halo)", "AMD", 2025, 16, 32, 8, 8000, "LPDDR5X", 4, 16, 1, 120, True, "avx512", 0, 4.5),
    ("epyc-9575f", "EPYC 9575F", "AMD", 2024, 64, 128, 12, 6000, "DDR5", 5, 128, 2, 400, True, "avx512", 0, 4.5),
]

FIELDS = ["id", "name", "vendor", "year", "cores", "threads", "mem_channels", "max_mem_mts", "mem_type",
          "pcie_gen", "pcie_lanes", "sockets", "tdp_w", "avx512", "isa", "e_cores", "all_core_ghz"]

# per-core expert-weight throughput, GB/s per GHz, by vector ISA (fitted: see tests/validation.py)
ISA_GBS_PER_GHZ = {"sse": 0.12, "avx": 0.18, "avx2": 0.30, "avx512": 0.86, "amx": 0.95, "neon": 0.45}
E_CORE_WEIGHT = 0.4
OS_FACTOR = {"linux": 1.0, "windows": 0.7}   # Windows: WDDM and non-pinned host paths slow CPU/PCIe expert traffic

CPU_BY_ID = {row[0]: dict(zip(FIELDS, row)) for row in CPUS}

# unified-memory parts: the pool bandwidth is a property of the package, not of DIMM slots
SOC_RAM_BW = {"gb10": 273, "apple-m4-max": 546, "apple-m3-ultra": 819, "apple-m2-ultra": 800, "apple-m1-ultra": 800,
              "apple-m5-max": 614, "apple-m5-ultra": 1228, "amd-ai-max-395": 256}
SOC_RAM_GB = {"gb10": [128], "apple-m4-max": [36, 48, 64, 128], "apple-m3-ultra": [96, 192, 256, 512],
              "apple-m2-ultra": [64, 128, 192], "apple-m1-ultra": [64, 128], "apple-m5-max": [48, 64, 128],
              "apple-m5-ultra": [96, 256, 512], "amd-ai-max-395": [32, 64, 128]}

RAM_SPEEDS = {"DDR3": [1333, 1600, 1866, 2133], "DDR4": [2133, 2400, 2666, 2933, 3200, 3600, 4000],
              "LPDDR5": [5500, 6400], "LPDDR5X": [7500, 8000, 8533], "DDR5": [4800, 5200, 5600, 6000, 6400, 7200, 8000]}
RAM_SIZES_GB = [16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]


# community-proposed parts (catalog/proposed.json): usable in the app, tagged as unverified, never calibration
def _load_proposed():
    import json
    from pathlib import Path
    try:
        data = json.loads((Path(__file__).resolve().parent / "proposed.json").read_text())
    except (OSError, ValueError):
        return
    for spec in data.get("cpus", []):
        if spec.get("id") in CPU_BY_ID:
            continue
        row = tuple(spec.get(f) for f in FIELDS)
        CPUS.append(row)
        d = dict(zip(FIELDS, row))
        d["proposed"] = True
        CPU_BY_ID[d["id"]] = d


_load_proposed()
