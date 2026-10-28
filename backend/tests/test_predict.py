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
    assert "vulkan backend not modelled" in reasons["llama-bench/vulkan/dev0/llama-2-7b-q4_0/tg128"]
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
    run2 = cpu_run(tmp_path / "b", cpu="AMD Ryzen 5 3600 6-Core Processor")
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
