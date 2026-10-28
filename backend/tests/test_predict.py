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
    assert reasons["mlc/idle_latency"] == "not modelled yet"


def test_unknown_gpu_and_cli(tmp_path):
    run = fake_run(tmp_path, gpus=("NVIDIA GeForce RTX 5090, 32607 MiB, 00000000:01:00.0",))
    mf, of = tmp_path / "m.json", tmp_path / "o.json"
    mf.write_text(json.dumps(dict(version="1", measurements=meas("llama-bench/cuda/gpu0/llama-2-7b-q4_0/tg128"))))
    assert P.main(["--run", str(run), "--measurements", str(mf), "--out", str(of)]) == 0
    out = json.loads(of.read_text())
    assert out["predictions"] == [] and "not in rigsim's catalogue: NVIDIA GeForce RTX 5090" in out["unsupported"][0]["reason"]
