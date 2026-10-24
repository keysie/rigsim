"""GPU catalog. Memory bandwidth and FP16 tensor throughput (dense, no sparsity) are
spec-sheet figures, rounded. `nvlink_gbs` is the per-GPU peer bandwidth of a bridge or
fabric (0 = none). `slots` is the cooler width in expansion slots."""

GPUS = [
    # id, name, vendor, year, VRAM GB, mem BW GB/s, FP16 TFLOPS, PCIe gen, lanes, NVLink GB/s, slots, TDP W, form
    ("gtx-680", "GeForce GTX 680", "NVIDIA", 2012, 2, 192, 3.1, 3, 16, 0, 2, 195, "pcie"),
    ("gtx-780ti", "GeForce GTX 780 Ti", "NVIDIA", 2013, 3, 336, 5.0, 3, 16, 0, 2, 250, "pcie"),
    ("gtx-980ti", "GeForce GTX 980 Ti", "NVIDIA", 2015, 6, 336, 6.1, 3, 16, 0, 2, 250, "pcie"),
    ("gtx-1080ti", "GeForce GTX 1080 Ti", "NVIDIA", 2017, 11, 484, 11.3, 3, 16, 0, 2, 250, "pcie"),
    ("titan-v", "Titan V", "NVIDIA", 2017, 12, 653, 110, 3, 16, 0, 2, 250, "pcie"),
    ("titan-rtx", "TITAN RTX", "NVIDIA", 2018, 24, 672, 130, 3, 16, 100, 2, 280, "pcie"),
    ("rtx-2080ti", "GeForce RTX 2080 Ti", "NVIDIA", 2018, 11, 616, 108, 3, 16, 50, 2, 250, "pcie"),
    # integrated GPUs of unified-memory systems (form "soc": bandwidth = the pool, no PCIe link)
    ("gb10-gpu", "GB10 Blackwell iGPU (DGX Spark)", "NVIDIA", 2025, 128, 273, 31, 5, 0, 0, 0, 100, "soc"),
    ("m4-max-gpu", "Apple M4 Max 40-core GPU", "Apple", 2024, 128, 546, 34, 4, 0, 0, 0, 60, "soc"),
    ("m3-ultra-gpu", "Apple M3 Ultra 80-core GPU", "Apple", 2025, 512, 819, 54, 4, 0, 0, 0, 180, "soc"),
    ("m2-ultra-gpu", "Apple M2 Ultra 76-core GPU", "Apple", 2023, 192, 800, 27, 4, 0, 0, 0, 200, "soc"),
    ("strix-halo-gpu", "Radeon 8060S (Strix Halo iGPU)", "AMD", 2025, 96, 256, 30, 4, 0, 0, 0, 100, "soc"),
    ("rtx-3060-12g", "GeForce RTX 3060 12 GB", "NVIDIA", 2021, 12, 360, 51, 4, 16, 0, 2, 170, "pcie"),
    ("rtx-3080-10g", "GeForce RTX 3080 10 GB", "NVIDIA", 2020, 10, 760, 119, 4, 16, 0, 2, 320, "pcie"),
    ("rtx-3090", "GeForce RTX 3090", "NVIDIA", 2020, 24, 936, 142, 4, 16, 112, 3, 350, "pcie"),
    ("rtx-3090ti", "GeForce RTX 3090 Ti", "NVIDIA", 2022, 24, 1008, 160, 4, 16, 112, 3, 450, "pcie"),
    ("rtx-5060ti-16g", "GeForce RTX 5060 Ti 16 GB", "NVIDIA", 2025, 16, 448, 95, 5, 8, 0, 2, 180, "pcie"),
    ("rtx-4060ti-16g", "GeForce RTX 4060 Ti 16 GB", "NVIDIA", 2023, 16, 288, 88, 4, 8, 0, 2, 165, "pcie"),
    ("rtx-4070ti-super", "GeForce RTX 4070 Ti Super", "NVIDIA", 2024, 16, 672, 176, 4, 16, 0, 3, 285, "pcie"),
    ("rtx-4080", "GeForce RTX 4080", "NVIDIA", 2022, 16, 717, 195, 4, 16, 0, 3, 320, "pcie"),
    ("rtx-4090", "GeForce RTX 4090", "NVIDIA", 2022, 24, 1008, 330, 4, 16, 0, 3, 450, "pcie"),
    ("rtx-5070", "GeForce RTX 5070", "NVIDIA", 2025, 12, 672, 124, 5, 16, 0, 2, 250, "pcie"),
    ("rtx-5080", "GeForce RTX 5080", "NVIDIA", 2025, 16, 960, 225, 5, 16, 0, 2, 360, "pcie"),
    ("rtx-5090", "GeForce RTX 5090", "NVIDIA", 2025, 32, 1792, 420, 5, 16, 0, 2, 575, "pcie"),
    ("quadro-rtx-8000", "Quadro RTX 8000", "NVIDIA", 2018, 48, 672, 130, 3, 16, 100, 2, 295, "pcie"),
    ("rtx-4000-ada", "RTX 4000 Ada", "NVIDIA", 2023, 20, 360, 106, 4, 16, 0, 1, 130, "pcie"),
    ("rtx-a4000", "RTX A4000", "NVIDIA", 2021, 16, 448, 76, 4, 16, 0, 1, 140, "pcie"),
    ("rtx-a5000", "RTX A5000", "NVIDIA", 2021, 24, 768, 111, 4, 16, 112, 2, 230, "pcie"),
    ("rtx-a6000", "RTX A6000", "NVIDIA", 2020, 48, 768, 155, 4, 16, 112, 2, 300, "pcie"),
    ("rtx-6000-ada", "RTX 6000 Ada", "NVIDIA", 2022, 48, 960, 364, 4, 16, 0, 2, 300, "pcie"),
    ("rtx-pro-5000-bw", "RTX PRO 5000 Blackwell", "NVIDIA", 2025, 48, 1344, 380, 5, 16, 0, 2, 300, "pcie"),
    ("rtx-pro-6000-bw", "RTX PRO 6000 Blackwell", "NVIDIA", 2025, 96, 1792, 500, 5, 16, 0, 2, 600, "pcie"),
    ("v100-32g", "Tesla V100 32 GB (PCIe)", "NVIDIA", 2018, 32, 900, 112, 3, 16, 0, 2, 250, "pcie"),
    ("a100-40g", "A100 40 GB (PCIe)", "NVIDIA", 2020, 40, 1555, 312, 4, 16, 600, 2, 250, "pcie"),
    ("a100-80g", "A100 80 GB (PCIe)", "NVIDIA", 2021, 80, 1935, 312, 4, 16, 600, 2, 300, "pcie"),
    ("h100-80g-pcie", "H100 80 GB (PCIe)", "NVIDIA", 2022, 80, 2000, 756, 5, 16, 600, 2, 350, "pcie"),
    ("h100-80g-sxm", "H100 80 GB (SXM)", "NVIDIA", 2022, 80, 3350, 990, 5, 16, 900, 0, 700, "sxm"),
    ("h200-141g", "H200 141 GB (SXM)", "NVIDIA", 2024, 141, 4800, 990, 5, 16, 900, 0, 700, "sxm"),
    ("b200-180g", "B200 180 GB (SXM)", "NVIDIA", 2025, 180, 8000, 2250, 5, 16, 1800, 0, 1000, "sxm"),
    ("rx-6800xt", "Radeon RX 6800 XT", "AMD", 2020, 16, 512, 83, 4, 16, 0, 2, 300, "pcie"),
    ("rx-7900xt", "Radeon RX 7900 XT", "AMD", 2022, 20, 800, 103, 4, 16, 0, 3, 315, "pcie"),
    ("rx-7900xtx", "Radeon RX 7900 XTX", "AMD", 2022, 24, 960, 123, 4, 16, 0, 3, 355, "pcie"),
    ("rx-9070xt", "Radeon RX 9070 XT", "AMD", 2025, 16, 640, 195, 5, 16, 0, 2, 304, "pcie"),
    ("radeon-ai-pro-r9700", "Radeon AI PRO R9700", "AMD", 2025, 32, 640, 191, 5, 16, 0, 2, 300, "pcie"),
    ("mi210", "Instinct MI210", "AMD", 2022, 64, 1600, 181, 4, 16, 300, 2, 300, "pcie"),
    ("mi250x", "Instinct MI250X", "AMD", 2021, 128, 3200, 383, 4, 16, 400, 0, 560, "oam"),
    ("mi300x", "Instinct MI300X", "AMD", 2023, 192, 5300, 1300, 5, 16, 896, 0, 750, "oam"),
    ("arc-a770", "Arc A770 16 GB", "Intel", 2022, 16, 560, 138, 4, 16, 0, 2, 225, "pcie"),
    ("rtx-3070-8g", "GeForce RTX 3070 8 GB", "NVIDIA", 2020, 8, 448, 81, 4, 16, 0, 2, 220, "pcie"),
]

GPU_FIELDS = ["id", "name", "vendor", "year", "vram_gb", "mem_bw_gbs", "fp16_tflops", "pcie_gen", "pcie_lanes",
              "nvlink_gbs", "slots", "tdp_w", "form"]
GPU_BY_ID = {row[0]: dict(zip(GPU_FIELDS, row)) for row in GPUS}

SSDS = [
    # id, name, PCIe gen, lanes, seq read GB/s, random read MB/s (4K QD32), capacities TB
    ("sata-ssd", "SATA SSD", 0, 0, 0.55, 400, [1, 2, 4]),
    ("nvme-gen3", "NVMe Gen3 x4", 3, 4, 3.5, 2500, [1, 2, 4]),
    ("nvme-gen4", "NVMe Gen4 x4", 4, 4, 7.0, 4500, [1, 2, 4, 8]),
    ("nvme-gen5", "NVMe Gen5 x4", 5, 4, 14.0, 6000, [1, 2, 4, 8]),
    ("nvme-u2-ent", "Enterprise U.2 Gen4 x4", 4, 4, 6.8, 5500, [4, 8, 16, 30]),
]
SSD_FIELDS = ["id", "name", "pcie_gen", "pcie_lanes", "seq_read_gbs", "rand_read_mbs", "capacities_tb"]
SSD_BY_ID = {row[0]: dict(zip(SSD_FIELDS, row)) for row in SSDS}

def kernel_efficiency(gpu: dict) -> float:
    """Older architectures run today's quantised kernels less efficiently (no bf16, smaller
    caches, slower tensor cores): Pascal ~0.35, Turing ~0.55, Ampere and later 1.0. Calibrated
    on Strata community reports (2 x TITAN RTX, 2 x P40) and llama.cpp tables."""
    if gpu["vendor"] == "NVIDIA":
        if gpu["year"] <= 2016:
            return 0.35
        if gpu["year"] <= 2018 and gpu["id"] not in ("v100-32g",):
            return 0.55
        return 1.0
    if gpu["vendor"] == "AMD":
        return 0.8
    return 0.7


# which SoC an integrated GPU belongs to; the GPU shares that CPU's memory pool
SOC_GPU_FOR = {"gb10-gpu": "gb10", "m4-max-gpu": "apple-m4-max", "m3-ultra-gpu": "apple-m3-ultra",
               "m2-ultra-gpu": "apple-m2-ultra", "strix-halo-gpu": "amd-ai-max-395"}

# external links for eGPU enclosures and USB4/Thunderbolt SSDs: usable bandwidth per direction, GB/s
EXTERNAL_LINKS = {"tb3": 2.8, "tb4": 3.0, "usb4": 3.0, "usb4v2": 6.5, "tb5": 9.0}

# PCIe bandwidth per lane and direction, GB/s (after encoding)
PCIE_LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


# community-proposed parts (catalog/proposed.json): usable in the app, tagged as unverified, never calibration
def _load_proposed():
    import json
    from pathlib import Path
    try:
        data = json.loads((Path(__file__).resolve().parent / "proposed.json").read_text())
    except (OSError, ValueError):
        return
    for spec in data.get("gpus", []):
        if spec.get("id") in GPU_BY_ID:
            continue
        row = tuple(spec.get(f) for f in GPU_FIELDS)
        GPUS.append(row)
        d = dict(zip(GPU_FIELDS, row))
        d["proposed"] = True
        GPU_BY_ID[d["id"]] = d


_load_proposed()
