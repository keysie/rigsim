from app.partcheck import check_cpu, check_gpu, check_source

GOOD_GPU = {"id": "rtx-3070ti-8g", "name": "GeForce RTX 3070 Ti 8 GB", "vendor": "NVIDIA", "year": 2021, "vram_gb": 8,
            "mem_bw_gbs": 608, "fp16_tflops": 87, "pcie_gen": 4, "pcie_lanes": 16, "nvlink_gbs": 0, "slots": 2, "tdp_w": 290,
            "form": "pcie", "mem_bus_bits": 256, "mem_gbps": 19}
GOOD_CPU = {"id": "i5-12400", "name": "Core i5-12400", "vendor": "Intel", "year": 2022, "cores": 6, "threads": 12,
            "mem_channels": 2, "max_mem_mts": 4800, "mem_type": "DDR5", "pcie_gen": 5, "pcie_lanes": 20, "sockets": 1,
            "tdp_w": 65, "avx512": False, "isa": "avx2", "e_cores": 0, "all_core_ghz": 4.0}


def test_consistent_parts_pass():
    assert check_gpu(GOOD_GPU)[0] == []
    assert check_cpu(GOOD_CPU)[0] == []


def test_invented_gpu_fails_on_arithmetic_and_envelopes():
    bad = {**GOOD_GPU, "id": "troll", "name": "GeForce RTX 9999", "mem_bw_gbs": 5000, "fp16_tflops": 9000, "vram_gb": 13}
    errs = check_gpu(bad)[0]
    assert any("bus x rate" in e for e in errs) and any("vram_gb" in e for e in errs) and any("fp16_tflops" in e for e in errs)


def test_cpu_rules():
    assert any("threads" in e for e in check_cpu({**GOOD_CPU, "threads": 7})[0])
    assert any("AVX-512" in e for e in check_cpu({**GOOD_CPU, "isa": "avx512", "avx512": True, "year": 2014, "pcie_gen": 3})[0])
    assert any("already in the catalog" in e for e in check_cpu({**GOOD_CPU, "id": "x", "name": "Core i9-13900K"})[0])


def test_source_domain():
    assert check_source("https://www.nvidia.com/en-us/geforce/", "RTX 3070 Ti")[0] == []
    assert check_source("https://pastebin.com/abc", "RTX 3070 Ti")[0]
    assert check_source("https://www.techpowerup.com/gpu-specs/", "RTX 3070 Ti")[0]
