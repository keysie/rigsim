"""GPU models from vendor whitepapers / spec sheets. Per-SM rates per clock are derived from the published dense
peaks (peak / (SMs x boost clock)); each entry names its source. Power curves are not spec values: they come from
measurements (or a user CSV) and default to flat, unmeasured."""
from __future__ import annotations

GPUS = {
    "rtx-3090": dict(arch="ampere", die="GA102", sms=82, boost_mhz=1695, tdp_w=350, vram_gb=24, vram_gbs=936.2,
                     pcie_gen=4, pcie_lanes=16,
                     ops_per_clk_per_sm={"fp32": 256, "fp16_tensor_fp32acc": 512, "fp16_tensor": 1024, "int8_tensor": 2048},
                     source="NVIDIA Ampere GA102 whitepaper (82 SMs, 1695 MHz boost, 936 GB/s; dense tensor peaks "
                            "71 / 142 / 284 TFLOPS-TOPS)"),
    "rtx-3060-12g": dict(arch="ampere", die="GA106", sms=28, boost_mhz=1777, tdp_w=170, vram_gb=12, vram_gbs=360.0,
                         pcie_gen=4, pcie_lanes=16,
                         ops_per_clk_per_sm={"fp32": 256, "fp16_tensor_fp32acc": 512, "fp16_tensor": 1024, "int8_tensor": 2048},
                         source="NVIDIA spec sheet (28 SMs, 1777 MHz boost, 360 GB/s); per-SM rates as GA102 (same SM)"),
    "rtx-4090": dict(arch="ada", die="AD102", sms=128, boost_mhz=2520, tdp_w=450, vram_gb=24, vram_gbs=1008.0,
                     pcie_gen=4, pcie_lanes=16,
                     ops_per_clk_per_sm={"fp32": 256, "fp16_tensor_fp32acc": 512, "fp16_tensor": 1024, "int8_tensor": 2048},
                     source="NVIDIA Ada AD102 whitepaper (128 SMs, 2520 MHz boost, 1008 GB/s; dense tensor peaks "
                            "165 / 330 / 661)"),
}
