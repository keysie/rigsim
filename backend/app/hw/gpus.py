"""GPU models from vendor whitepapers / spec sheets only (proposal §5, spec-sheet first). Each entry names its source.

Per-SM rates per clock are derived from the published dense peaks (peak / (SMs x boost clock)). The memory system is
the documented one: `mem_controllers` 32-bit controllers, each driving its own slice of the VRAM at `mem_gbps` per
pin; VRAM bandwidth is derived from it (vram_gbs below is the vendor's own figure, kept as a cross-check). GPCs and
the L2 size are recorded as documented; nothing is assumed about how SMs split over GPCs on a given card (it varies
from chip to chip) or about L2 bandwidth (not published). `copy_engines` is not on spec sheets: the driver reports
it at run time (cudaDeviceProp.asyncEngineCount); None until read from a device.
"""
from __future__ import annotations

GPUS = {
    "rtx-3090": dict(arch="ampere", die="GA102", sms=82, boost_mhz=1695, tdp_w=350, vram_gb=24, vram_gbs=936.2,
                     base_mhz=1395, gpcs=7, l2_mb=6, mem_type="GDDR6X", mem_controllers=12, mem_gbps=19.5, copy_engines=None,
                     pcie_gen=4, pcie_lanes=16,
                     ops_per_clk_per_sm={"fp32": 256, "tf32_tensor": 256, "fp16_tensor_fp32acc": 512, "fp16_tensor": 1024,
                                         "int8_tensor": 2048, "int4_tensor": 4096},
                     arch_source="GA102 whitepaper: 7 GPCs, 82 SMs, 6 MB L2, 12 x 32-bit controllers, 24 GB GDDR6X at 19.5 Gbps; 1395 MHz base",
                     source="NVIDIA Ampere GA102 whitepaper (82 SMs, 1695 MHz boost, 936 GB/s; dense tensor peaks "
                            "71 / 142 / 284 TFLOPS-TOPS)"),
    "rtx-3060-12g": dict(arch="ampere", die="GA106", sms=28, boost_mhz=1777, tdp_w=170, vram_gb=12, vram_gbs=360.0,
                         base_mhz=1320, gpcs=3, l2_mb=3, mem_type="GDDR6", mem_controllers=6, mem_gbps=15.0, copy_engines=None,
                     pcie_gen=4, pcie_lanes=16,
                         ops_per_clk_per_sm={"fp32": 256, "tf32_tensor": 256, "fp16_tensor_fp32acc": 512, "fp16_tensor": 1024,
                                         "int8_tensor": 2048, "int4_tensor": 4096},
                         arch_source="spec sheet: 3 GPCs, 28 SMs, 3 MB L2, 192-bit GDDR6 at 15 Gbps; 1320 MHz base",
                     source="NVIDIA spec sheet (28 SMs, 1777 MHz boost, 360 GB/s); per-SM rates as GA102 (same SM)"),
    "rtx-3080": dict(arch="ampere", die="GA102", sms=68, boost_mhz=1710, tdp_w=320, vram_gb=10, vram_gbs=760.3,
                     base_mhz=1440, gpcs=6, l2_mb=5, mem_type="GDDR6X", mem_controllers=10, mem_gbps=19.0, copy_engines=None,
                     pcie_gen=4, pcie_lanes=16,
                     ops_per_clk_per_sm={"fp32": 256, "tf32_tensor": 256, "fp16_tensor_fp32acc": 512, "fp16_tensor": 1024,
                                         "int8_tensor": 2048, "int4_tensor": 4096},
                     arch_source="GA102 whitepaper, RTX 3080 column: 6 GPCs, 68 SMs, 5 MB L2, 10 x 32-bit controllers, "
                                 "10 GB GDDR6X at 19 Gbps; 1440 MHz base",
                     source="NVIDIA Ampere GA102 whitepaper (RTX 3080 10 GB: 68 SMs, 1710 MHz boost, 760 GB/s, 320 W); "
                            "per-SM rates as GA102 (same SM)"),
    "rtx-3070-ti": dict(arch="ampere", die="GA104", sms=48, boost_mhz=1770, tdp_w=290, vram_gb=8, vram_gbs=608.3,
                        base_mhz=1575, gpcs=6, l2_mb=4, mem_type="GDDR6X", mem_controllers=8, mem_gbps=19.0, copy_engines=None,
                     pcie_gen=4, pcie_lanes=16,
                        ops_per_clk_per_sm={"fp32": 256, "tf32_tensor": 256, "fp16_tensor_fp32acc": 512, "fp16_tensor": 1024,
                                            "int8_tensor": 2048, "int4_tensor": 4096},
                        arch_source="spec sheet: 6 GPCs, 48 SMs, 4 MB L2, 256-bit GDDR6X at 19 Gbps; 1575 MHz base",
                     source="NVIDIA spec sheet (48 SMs, 1770 MHz boost, 608 GB/s); per-SM rates as GA102 (same SM)"),
    "rtx-4090": dict(arch="ada", die="AD102", sms=128, boost_mhz=2520, tdp_w=450, vram_gb=24, vram_gbs=1008.0,
                     base_mhz=2235, gpcs=11, l2_mb=72, mem_type="GDDR6X", mem_controllers=12, mem_gbps=21.0, copy_engines=None,
                     pcie_gen=4, pcie_lanes=16,
                     ops_per_clk_per_sm={"fp32": 256, "tf32_tensor": 256, "fp16_tensor_fp32acc": 512, "fp16_tensor": 1024,
                                         "int8_tensor": 2048, "int4_tensor": 4096},
                     arch_source="AD102 whitepaper: 11 GPCs enabled, 128 SMs, 72 MB L2, 384-bit GDDR6X at 21 Gbps; 2235 MHz base",
                     source="NVIDIA Ada AD102 whitepaper (128 SMs, 2520 MHz boost, 1008 GB/s; dense tensor peaks "
                            "165 / 330 / 661)"),
}


def vram_bandwidth_gbs(spec: dict) -> float:
    """Peak VRAM bandwidth from the documented memory system: controllers x 32 bits x data rate per pin."""
    return spec["mem_controllers"] * 32 * spec["mem_gbps"] / 8


def channel_bandwidth_gbs(spec: dict) -> float:
    return 32 * spec["mem_gbps"] / 8
