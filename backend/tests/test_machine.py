from app.machine import validate
from app.schemas import CpuSlot, Machine, PcieDevice


def gpu(i, model="rtx-3090", gen=4, width=16, cpu=0):
    return PcieDevice(id=f"gpu-{i}", kind="gpu", model_id=model, cpu=cpu, link_gen=gen, link_width=width)


def test_lane_budget_is_enforced():
    m = Machine(cpus=[CpuSlot(cpu_id="i9-13900k")], devices=[gpu(1), gpu(2)])   # 20 lanes, 32 asked
    r = validate(m)
    assert not r["ok"] and any("PCIe lanes" in e for e in r["errors"])


def test_link_is_capped_by_cpu_and_device():
    m = Machine(cpus=[CpuSlot(cpu_id="i9-10900x", ram_mts=2933)], devices=[gpu(1, gen=5), gpu(2, "rtx-4060ti-16g")])
    r = validate(m)
    assert r["ok"]
    assert r["devices"]["gpu-1"]["link_gen"] == 3                # CPU is Gen3
    assert r["devices"]["gpu-2"]["link_width"] == 8              # the 4060 Ti is x8
    assert r["cpus"][0]["lanes_used"] == 24


def test_second_cpu_needs_a_two_socket_part():
    m = Machine(cpus=[CpuSlot(cpu_id="ryzen9-7950x"), CpuSlot(cpu_id="ryzen9-7950x")])
    assert any("second socket" in e for e in validate(m)["errors"])
    m = Machine(cpus=[CpuSlot(cpu_id="epyc-9654", ram_channels=12, ram_mts=4800),
                      CpuSlot(cpu_id="epyc-9654", ram_channels=12, ram_mts=4800)],
                devices=[gpu(1, gen=5, cpu=1)])
    r = validate(m)
    assert r["ok"] and r["cpus"][1]["lanes_used"] == 16 and r["cpus"][1]["ram_bw_gbs"] > 300


def test_nvlink_rules():
    m = Machine(cpus=[CpuSlot(cpu_id="tr-pro-7995wx", ram_channels=8, ram_mts=5200)],
                devices=[gpu(1), gpu(2), gpu(3, "rtx-4090")], nvlinks=[("gpu-1", "gpu-2")])
    assert validate(m)["ok"]
    m.nvlinks = [("gpu-2", "gpu-3")]
    assert any("same model" in e for e in validate(m)["errors"])
    m.devices = [gpu(1, "rtx-4090"), gpu(2, "rtx-4090")]
    m.nvlinks = [("gpu-1", "gpu-2")]
    assert any("no NVLink" in e for e in validate(m)["errors"])
    m.devices = [gpu(1), gpu(2), gpu(3)]
    m.nvlinks = [("gpu-1", "gpu-3")]
    assert any("adjacent" in e for e in validate(m)["errors"])


def test_ram_channels_are_checked():
    m = Machine(cpus=[CpuSlot(cpu_id="i9-13900k", ram_channels=4)])
    assert any("memory channels" in e for e in validate(m)["errors"])


def test_prebuilts_validate_and_soc_rules():
    from app.catalog.prebuilts import PREBUILTS
    from app.machine import validate
    from app.schemas import Machine
    for pb in PREBUILTS:
        info = validate(Machine.model_validate(pb["machine"]))
        assert info["ok"], (pb["id"], info["errors"])
    spark = PREBUILTS[0]["machine"]
    info = validate(Machine.model_validate(spark))
    assert list(info["cpus"].values())[0]["ram_bw_gbs"] == 273.0
    assert info["_devices"]["gpu-1"]["link_type"] == "soc"
    assert info["_devices"]["gpu-1"]["link_bw_gbs"] == 273.0
    # the iGPU cannot sit on a different CPU
    wrong = {**spark, "cpus": [{"cpu_id": "ryzen9-7950x", "enabled": True, "ram_gb": 64, "ram_mts": 5200, "ram_channels": 2}]}
    assert not validate(Machine.model_validate(wrong))["ok"]


def test_external_links_skip_lanes_and_cap_bandwidth():
    from app.catalog.prebuilts import PREBUILTS
    from app.machine import validate
    from app.schemas import Machine
    m = dict(PREBUILTS[0]["machine"])
    m["devices"] = m["devices"] + [{"id": "gpu-2", "kind": "gpu", "model_id": "rtx-3090", "cpu": 0,
                                    "link_gen": 3, "link_width": 16, "link_type": "tb4"}]
    info = validate(Machine.model_validate(m))
    assert info["ok"]
    assert list(info["cpus"].values())[0]["lanes_used"] == 4          # SSD only: the GB10's port is integrated
    assert info["_devices"]["gpu-2"]["link_bw_gbs"] == 3.0
    assert any("integrated port" in w for w in info["warnings"])
    # the same eGPU on a 2019 HEDT CPU needs a controller card: 4 lanes
    old = {"os": "linux", "cpus": [{"cpu_id": "i9-10900x", "enabled": True, "ram_gb": 64, "ram_mts": 3200, "ram_channels": 4}],
           "devices": [{"id": "gpu-1", "kind": "gpu", "model_id": "rtx-3090", "cpu": 0, "link_gen": 3, "link_width": 16, "link_type": "tb3"}], "nvlinks": []}
    info2 = validate(Machine.model_validate(old))
    assert list(info2["cpus"].values())[0]["lanes_used"] == 4
    assert any("add-in card, 4 CPU lanes" in w for w in info2["warnings"])
