"""app.predict: rigsim's predictions for a rigsim-benchmark run (contract in rigsim-benchmark/predictions/README.md)."""
import json

from app import predict as P


def fake_run(tmp_path, gpus=("NVIDIA GeForce RTX 3080, 10240 MiB, 00000000:01:00.0",)):
    run = tmp_path / "run"; (run / "inventory").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(dict(hardware=dict(gpus=list(gpus)))))
    (run / "inventory" / "nvidia-smi.txt").write_text("GPU 00000000:01:00.0\n    Power Readings\n        Current Power Limit : 250.00 W\n")
    (run / "inventory" / "pcie.json").write_text(json.dumps(dict(devices={
        "0000:01:00.0": dict(parent="0000:00:01.0", link=dict(max_speed="16.0 GT/s PCIe", max_width="16")),
        "0000:00:01.0": dict(parent="", link=dict(max_speed="8.0 GT/s PCIe", max_width="8"))})))
    return run


def meas(*keys):
    return [dict(key=k, value=1.0, unit="tok/s" if k.startswith("llama") else "GB/s", conditions={}) for k in keys]


def test_inventory_parsing(tmp_path):
    run = fake_run(tmp_path)
    assert P.nvidia_gpus(run) == [dict(index=0, name="NVIDIA GeForce RTX 3080", mib=10240.0, pci="0000:01:00.0")]
    assert P.power_limits(run) == {"0000:01:00.0": 250.0}
    assert P.pcie_slot(run, "0000:01:00.0") == (3, 8)                    # the port limits it: gen 3 x8
    assert P.catalogue_id("NVIDIA GeForce RTX 3080", 10240) == "rtx-3080"
    assert P.catalogue_id("NVIDIA GeForce RTX 3080 Ti", 12288) is None
    assert P.catalogue_id("NVIDIA GeForce RTX 3060", 8192) is None        # the 8 GB 3060 is another card


def test_predictions_and_reasons(tmp_path):
    run = fake_run(tmp_path)
    preds, unsup = P.predict(run, meas("llama-bench/cuda/gpu0/llama-2-7b-q4_0/tg128", "gpu-vram/gpu0/read",
                                       "llama-bench/vulkan/dev0/llama-2-7b-q4_0/tg128", "llama-bench/cuda/gpu0/mistral-7b-q4_0/tg128",
                                       "mlc/idle_latency"))
    by = {p["key"]: p for p in preds}
    assert 100 < by["llama-bench/cuda/gpu0/llama-2-7b-q4_0/tg128"]["value"] < 200
    assert 600 < by["gpu-vram/gpu0/read"]["value"] < 800 and all(p["calibration"] is False for p in preds)
    reasons = {u["key"]: u["reason"] for u in unsup}
    assert "not a catalogued GPU" in reasons["llama-bench/vulkan/dev0/llama-2-7b-q4_0/tg128"]
    assert "model architecture not described" in reasons["llama-bench/cuda/gpu0/mistral-7b-q4_0/tg128"]
    assert "CPU not in rigsim's catalogue" in reasons["mlc/idle_latency"]


def test_unknown_gpu_and_cli(tmp_path):
    run = fake_run(tmp_path, gpus=("NVIDIA GeForce RTX 5090, 32607 MiB, 00000000:01:00.0",))
    mf, of = tmp_path / "m.json", tmp_path / "o.json"
    mf.write_text(json.dumps(dict(version="1", measurements=meas("llama-bench/cuda/gpu0/llama-2-7b-q4_0/tg128"))))
    assert P.main(["--run", str(run), "--measurements", str(mf), "--out", str(of)]) == 0
    out = json.loads(of.read_text())
    assert out["predictions"] == [] and "not in rigsim's catalogue: NVIDIA GeForce RTX 5090" in out["unsupported"][0]["reason"]


def cpu_run(tmp_path, cpu="13th Gen Intel(R) Core(TM) i7-13700K", mem="DDR5", mts=5600, ndimm=2):
    run = tmp_path / "cpu"; (run / "inventory").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(dict(hardware=dict(cpu=cpu, gpus=[]))))
    rows = ["CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE MAXMHZ MINMHZ MHZ"]
    cpu_id = 0
    for core in range(16):                                   # 8 P-cores with two threads, 8 E-cores with one
        for _ in range(2 if core < 8 else 1):
            rows.append(f"{cpu_id} 0 0 {core} {core}:{core}:{core}:0 yes 5400.0 800.0 800.0"); cpu_id += 1
    (run / "inventory" / "lscpu-e.txt").write_text("\n".join(rows) + "\n")
    dev = ("Memory Device\n\tSize: 16 GB\n\tLocator: {loc}\n\tBank Locator: Controller0-Channel{ch}\n\tType: {mem}\n"
           "\tSpeed: 5600 MT/s\n\tRank: 1\n\tConfigured Memory Speed: {mts} MT/s\n\n")
    text = "".join(dev.format(loc=f"DIMM {i}", ch="AB"[i % 2], mem=mem, mts=mts) for i in range(ndimm))
    text += "Memory Device\n\tSize: No Module Installed\n\tLocator: DIMM 9\n\n"
    (run / "inventory" / "dmidecode.txt").write_text(text)
    spd = bytearray(1024); spd[4] = 0x04                     # 16 Gb dies
    spd[30:32] = (16000).to_bytes(2, "little")               # tAAmin 16.0 ns
    (run / "inventory" / "spd-0-0050.hex").write_text("# spd5118, 1024 bytes, serial bytes zeroed\n" +
        "\n".join(spd[i:i + 16].hex(" ") for i in range(0, 1024, 16)) + "\n")
    return run


def test_intel_generations():
    from app.hw import intel_client as I
    assert I.generation("13th Gen Intel(R) Core(TM) i7-13700K") == (13, "i7", 13700)
    assert I.generation("Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz") == (6, "i7", 6700)
    assert I.generation("Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz")[0] == 9
    assert I.generation("AMD Ryzen 7 5800X 8-Core Processor") is None
    assert I.memory_support(13, "i5", 13400) == dict(ddr5=4800, ddr4=3200)            # below the 13600: 4800
    assert I.memory_support(10, "i5", 10400) == dict(ddr4=2666)
    s = I.spec("Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz", {"p": 6, "e": 0})
    assert s["cores"] == {"core": 6} and s["channels"] == 2 and s["arch"] == "coffeelake-s"
    assert I.spec("12th Gen Intel(R) Core(TM) i5-12400", {"p": 6, "e": 0})["cores"] == {"p": 6}


def test_inventory_cpu_and_memory(tmp_path):
    run = cpu_run(tmp_path)
    assert P.core_counts(run) == {"p": 8, "e": 8}
    assert [d["mts"] for d in P.dimms(run)] == [5600, 5600]                            # the empty slot left out
    assert P.spd_timing(run)["taa_ns"] == 16.0 and P.spd_timing(run)["density_gbit"] == 16
    m, groups, params, why = P.cpu_memory_machine(run)
    assert why is None and params["channels"] == 2 and params["cl"] == 45 and params["mem"] == "DDR5-5600"
    preds, unsup = P.predict(run, [dict(key="mlc/idle_latency", value=64.0, unit="ns"),
                                   dict(key="mlc/max_bandwidth/all_reads", value=89.0, unit="GB/s"),
                                   dict(key="mlc/max_bandwidth/3to1_rw", value=80.0, unit="GB/s")])
    by = {p["key"]: p["value"] for p in preds}
    assert 70 < by["mlc/idle_latency"] < 100                   # 16 ns CAS + the global fixed term
    assert 60 < by["mlc/max_bandwidth/all_reads"] < 89.6        # below the 2 x 44.8 GB/s peak
    assert "write traffic" in unsup[0]["reason"]


def test_cpu_memory_reasons(tmp_path):
    run = cpu_run(tmp_path)
    (run / "inventory" / "dmidecode.txt").write_text("")
    assert "no DIMM data" in P.cpu_memory_machine(run)[3]
    run2 = cpu_run(tmp_path / "b", cpu="AMD Ryzen Threadripper 3970X 32-Core Processor")
    assert "CPU not in rigsim's catalogue" in P.cpu_memory_machine(run2)[3]
    one = cpu_run(tmp_path / "c", ndimm=1)
    assert P.cpu_memory_machine(one)[2]["channels"] == 1



def test_channel_names_boards_use(tmp_path):
    run = cpu_run(tmp_path)
    text = (run / "inventory" / "dmidecode.txt").read_text()
    for a, b in (("DDR5-A2", "DDR5-B2"), ("DIMM_A1", "DIMM_B1")):
        t = text.replace("Bank Locator: Controller0-ChannelA", "Bank Locator: BANK 0").replace(
            "Bank Locator: Controller0-ChannelB", "Bank Locator: BANK 0").replace("DIMM 0", a).replace("DIMM 1", b)
        (run / "inventory" / "dmidecode.txt").write_text(t)
        assert P.cpu_memory_machine(run)[2]["channels_from"] == "locators"


def amd_run(tmp_path, cpu, cores=12, l3=4, threads=2, mem="DDR4", mts=3200):
    run = cpu_run(tmp_path, cpu=cpu, mem=mem, mts=mts)
    rows = ["CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE MAXMHZ MINMHZ MHZ"]
    n = 0
    for core in range(cores):
        for _ in range(threads):
            rows.append(f"{n} 0 0 {core} {core}:{core}:{core}:{core * l3 // cores} yes 4600.0 2200.0 2200.0"); n += 1
    (run / "inventory" / "lscpu-e.txt").write_text("\n".join(rows) + "\n")
    (run / "inventory" / "spd-0-0050.hex").unlink()
    return run


def test_ryzen_families():
    from app.hw import amd_client as A
    fam = lambda c: A.family(c)
    assert fam("AMD Ryzen 7 1800X Eight-Core Processor") == "summitridge"
    assert fam("AMD Ryzen 5 2400G with Radeon Vega Graphics") == "ravenridge"
    assert fam("AMD Ryzen 9 3900X 12-Core Processor") == "matisse" and fam("AMD Ryzen 5 3400G with Radeon Vega Graphics") == "picasso"
    assert fam("AMD Ryzen 7 5700 8-Core Processor") == "cezanne" and fam("AMD Ryzen 7 5700X 8-Core Processor") == "vermeer"
    assert fam("AMD Ryzen 9 7950X 16-Core Processor") == "raphael"
    assert fam("AMD Ryzen 7 5800H with Radeon Graphics") is None and fam("AMD Ryzen Threadripper 3970X") is None
    assert A.fclk_mhz("matisse", 3200) == 1600 and A.fclk_mhz("vermeer", 4000) == 1800
    assert A.fclk_mhz("raphael", 5200) == 1733 and A.fclk_mhz("raphael", 6000) == 2000 and A.fclk_mhz("cezanne", 3200) is None
    s = A.spec("AMD Ryzen 9 3900X 12-Core Processor", 12, 4)                    # Zen 2: two core complexes per CCD
    assert s["ccds"] == 2 and s["cores_per_ccd"] == {"zen2": 6}
    assert A.spec("AMD Ryzen 9 5950X 16-Core Processor", 16, 2)["ccds"] == 2       # Zen 3: one per CCD
    assert A.spec("AMD Ryzen 7 5800X 8-Core Processor", 8, 1)["ccds"] == 1


def test_ryzen_predictions(tmp_path):
    run = amd_run(tmp_path, "AMD Ryzen 9 3900X 12-Core Processor")
    assert P.lscpu_topology(run) == dict(cores=12, l3_slices=4)
    m, groups, params, why = P.cpu_memory_machine(run)
    assert why is None and params["fclk_mhz"].startswith("1600") and params["cores"]["ccds"] == 2
    preds, _ = P.predict(run, [dict(key="mlc/max_bandwidth/all_reads", value=40.0, unit="GB/s")])
    assert 30 < preds[0]["value"] < 51.2                                             # below 2 x 25.6 GB/s peak
    apu = amd_run(tmp_path / "apu", "AMD Ryzen 5 5600G with Radeon Graphics", cores=6, l3=1)
    assert P.cpu_memory_machine(apu)[2]["cpu"] == "cezanne" and P.cpu_memory_machine(apu)[2]["fclk_mhz"] is None



def test_vram_write_and_copy(tmp_path):
    run = fake_run(tmp_path)
    preds, unsup = P.predict(run, [dict(key=f"gpu-vram/gpu0/{op}", value=1.0, unit="GB/s", conditions={})
                                   for op in ("read", "write", "copy")])
    assert not unsup and len({round(p["value"], 3) for p in preds}) == 1
    assert "bus turnaround not modelled" in next(p for p in preds if p["key"].endswith("copy"))["basis"]


def test_vulkan_engine_by_pci(tmp_path):
    run = fake_run(tmp_path)
    meas = [dict(key="llama-bench/vulkan/dev0/llama-2-7b-q4_0/tg128", value=140.0, unit="tok/s",
                 conditions=dict(pci="0000:01:00.0", gpu="NVIDIA GeForce RTX 3080")),
            dict(key="llama-bench/vulkan/dev1/llama-2-7b-q4_0/tg128", value=5.8, unit="tok/s",
                 conditions=dict(pci="0000:00:02.0", gpu="Intel(R) Graphics (RPL-S)"))]
    preds, unsup = P.predict(run, meas)
    assert len(preds) == 1 and 100 < preds[0]["value"] < 200 and "Vulkan engine" in preds[0]["basis"]
    assert "not a catalogued GPU (Intel(R) Graphics (RPL-S))" in unsup[0]["reason"]
    from app.engine.llamacpp import vulkan
    from app.engine.model import from_config
    ops = vulkan.cuda.forward(from_config("tinyllama-1.1b", "q4_0"), 1, 0, matmul=vulkan.matmul, attention=vulkan.attention)
    q = next(o for o in ops if o.name == "blk.0.attn_q")
    assert q.factor_basis.startswith("vulkan mmv") and q.kernels == 1         # dequantising mat-vec, no quantise kernel


def test_intel_igpu_from_the_run(tmp_path):
    from app.hw import intel_igpu as I
    assert I.product("Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770] [8086:a780] (rev 04)") == "UHD Graphics 770"
    assert I.product("Intel Corporation HD Graphics 630") == "HD Graphics 630" and I.product("NVIDIA GeForce RTX 3080") is None
    run = cpu_run(tmp_path)
    hw = json.loads((run / "run.json").read_text())
    hw["hardware"]["gpus"] = ["Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770] [8086:a780] (rev 04)"]
    (run / "run.json").write_text(json.dumps(hw))
    (run / "samples-drm.jsonl").write_text("".join(json.dumps(dict(t=i, pci="0000:00:02.0", gt_mhz=v)) + "\n"
                                                    for i, v in enumerate([300, 1600, 1550, 0])))
    assert P.igpu_clock_mhz(run, "0000:00:02.0") == 1600
    meas = [dict(key="llama-bench/vulkan/dev1/llama-2-7b-q4_0/tg128", value=5.8, unit="tok/s",
                 conditions=dict(pci="0000:00:02.0", gpu="Intel(R) Graphics (RPL-S)"))]
    preds, unsup = P.predict(run, meas)
    assert not unsup and "(igpu, 0000:00:02.0)" in preds[0]["basis"]
    spec = __import__("app.hw.gpus", fromlist=["GPUS"]).GPUS[next(k for k in __import__("app.hw.gpus", fromlist=["GPUS"]).GPUS if k.startswith("intel-uhd-graphics-770@ddr5-5600"))]
    assert spec["mem_controllers"] == 4 and spec["matrix_units"] is False and spec["sms"] == 32
