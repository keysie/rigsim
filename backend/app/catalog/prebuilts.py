"""Prebuilt systems: one click fills the machine page; everything stays editable afterwards."""


def _soc(cpu_id, gpu_id, ram_gb, mts, extra=None):
    return {"os": "linux", "cpus": [{"cpu_id": cpu_id, "enabled": True, "ram_gb": ram_gb, "ram_mts": mts, "ram_channels": 8}],
            "devices": [{"id": "gpu-1", "kind": "gpu", "model_id": gpu_id, "cpu": 0, "link_gen": 4, "link_width": 16, "link_type": "pcie"},
                        {"id": "ssd-1", "kind": "ssd", "model_id": "nvme-gen4", "cpu": 0, "link_gen": 4, "link_width": 4, "link_type": "pcie", "capacity_tb": 2}]
            + (extra or []), "nvlinks": []}


PREBUILTS = [
    dict(id="dgx-spark", name="NVIDIA DGX Spark", note="GB10, 128 GB unified at 273 GB/s; one TB4-class port for an eGPU",
         machine=_soc("gb10", "gb10-gpu", 128, 8533)),
    dict(id="mac-studio-m4-max", name="Mac Studio M4 Max (128 GB)", note="546 GB/s unified; external GPUs are not supported by macOS, TB5 SSDs are",
         machine=_soc("apple-m4-max", "m4-max-gpu", 128, 8533)),
    dict(id="mac-studio-m3-ultra", name="Mac Studio M3 Ultra (512 GB)", note="819 GB/s unified; the big-model unified box",
         machine=_soc("apple-m3-ultra", "m3-ultra-gpu", 512, 8533)),
    dict(id="mac-pro-m2-ultra", name="Mac Pro M2 Ultra (192 GB)", note="800 GB/s unified; PCIe slots exist but take no GPUs under macOS",
         machine=_soc("apple-m2-ultra", "m2-ultra-gpu", 192, 6400)),
    dict(id="framework-desktop", name="Framework Desktop (Ryzen AI Max+ 395, 128 GB)", note="Strix Halo, 256 GB/s unified",
         machine=_soc("amd-ai-max-395", "strix-halo-gpu", 128, 8000)),
    dict(id="gaming-4090", name="Gaming desktop, one RTX 4090", note="Ryzen 9 9950X, 2-channel DDR5-5600, Gen5 NVMe",
         machine={"os": "linux", "cpus": [{"cpu_id": "ryzen9-9950x", "enabled": True, "ram_gb": 96, "ram_mts": 5600, "ram_channels": 2}],
                  "devices": [{"id": "gpu-1", "kind": "gpu", "model_id": "rtx-4090", "cpu": 0, "link_gen": 4, "link_width": 16, "link_type": "pcie"},
                              {"id": "ssd-1", "kind": "ssd", "model_id": "nvme-gen5", "cpu": 0, "link_gen": 5, "link_width": 4, "link_type": "pcie", "capacity_tb": 2}], "nvlinks": []}),
    dict(id="dual-3090", name="Used-parts workstation, two RTX 3090", note="Ryzen 9 7950X, both cards at Gen4 x8",
         machine={"os": "linux", "cpus": [{"cpu_id": "ryzen9-7950x", "enabled": True, "ram_gb": 128, "ram_mts": 5200, "ram_channels": 2}],
                  "devices": [{"id": "gpu-1", "kind": "gpu", "model_id": "rtx-3090", "cpu": 0, "link_gen": 4, "link_width": 8, "link_type": "pcie"},
                              {"id": "gpu-2", "kind": "gpu", "model_id": "rtx-3090", "cpu": 0, "link_gen": 4, "link_width": 8, "link_type": "pcie"},
                              {"id": "ssd-1", "kind": "ssd", "model_id": "nvme-gen4", "cpu": 0, "link_gen": 4, "link_width": 4, "link_type": "pcie", "capacity_tb": 2}], "nvlinks": []}),
]
