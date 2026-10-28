"""ONE graph per run: CPU, memory, PCIe tree, every GPU at its port, the iGPU on the shared memory."""
import json
import shutil
from pathlib import Path

import pytest

from app import system as S
from app.hw.machine import Read
from tests.test_predict import cpu_run

RUN = Path("/home/claude/bench-repo/results/runs/20261028T184331Z-2ad20e343d87-0f60")


def machine_run(tmp_path):
    run = cpu_run(tmp_path)
    hw = json.loads((run / "run.json").read_text())
    hw["hardware"]["gpus"] = ["NVIDIA GeForce RTX 3080, 10240 MiB, 00000000:01:00.0",
                              "Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770] [8086:a780] (rev 04)"]
    (run / "run.json").write_text(json.dumps(hw))
    (run / "samples-drm.jsonl").write_text(json.dumps(dict(t=0, pci="0000:00:02.0", gt_mhz=1600)) + "\n")
    (run / "inventory" / "pcie.json").write_text(json.dumps(dict(devices={
        "0000:00:01.0": dict(parent="", path=["0000:00:01.0"], port_type="root port", name="Raptor Lake PCIe 5.0 Graphics Port",
                             link=dict(max_speed="32.0 GT/s PCIe", max_width="16")),
        "0000:01:00.0": dict(parent="0000:00:01.0", path=["0000:00:01.0", "0000:01:00.0"], port_type="endpoint",
                             link=dict(max_speed="16.0 GT/s PCIe", max_width="16")),
        "0000:00:1c.0": dict(parent="", path=["0000:00:1c.0"], port_type="root port", name="Raptor Lake PCH PCI Express Root Port",
                             link=dict(max_speed="8.0 GT/s PCIe", max_width="4"))})))
    return run


def test_one_graph_holds_the_whole_machine(tmp_path):
    s = S.build_system(machine_run(tmp_path))
    m = s.m
    assert s.io == "pkg" and s.groups == ["p", "e"] and set(m.devices) == {"gpu0", "igpu"}
    assert m.path("gpu0.copy", "pkg")[-1][1] == "pkg"                                 # the card hangs off the CPU...
    assert any(l.id == "gpu0.pcie" and l.b == "pcie:0000:00:01.0" for l in m.links.values())   # ...at its own port
    assert m.devices["igpu"].memory == "ram" and m.devices["gpu0"].memory == "gpu0.vram"
    # the iGPU and the cores read the SAME channels: together they share what one alone gets
    alone = m.solve_reads([Read("g", "igpu.sm", "ram")])["g"]["rate_bps"]
    both = m.solve_reads([Read("g", "igpu.sm", "ram")] + [Read(g, g, "ram") for g in s.groups])
    cpu_alone = sum(v["rate_bps"] for v in m.solve_reads([Read(g, g, "ram") for g in s.groups]).values())
    total = sum(v["rate_bps"] for v in both.values())
    assert both["g"]["rate_bps"] < alone and total <= max(alone, cpu_alone) * 1.0001


def test_diagram_names_the_parts_an_owner_knows(tmp_path):
    s = S.build_system(machine_run(tmp_path))
    text = S.mermaid(s)
    for part in ("i7-13700K — package", "DDR5-5600 channel A", "RTX 3080 — 68 SMs", "UHD Graphics 770 — 32 EUs at 1600 MHz",
                 "root port 00:01.0", "system memory DDR5-5600, 2 channels"):
        assert part in text, part
    d = S.describe(s)
    assert any(e["label"].endswith("GB/s") for e in d["edges"]) and "gpu0.vram" in {n["id"] for n in d["nodes"]}
    assert not any(n["id"].startswith("gpu0.vram") and n["id"] != "gpu0.vram" for n in d["nodes"])   # slices drawn as one


def test_chipset_port_goes_through_the_chipset(tmp_path):
    run = machine_run(tmp_path)
    devs = json.loads((run / "inventory" / "pcie.json").read_text())
    devs["devices"]["0000:01:00.0"].update(parent="0000:00:1c.0", path=["0000:00:1c.0", "0000:01:00.0"])
    (run / "inventory" / "pcie.json").write_text(json.dumps(devs))
    s = S.build_system(run)
    assert [n for _, n in s.m.path("gpu0.copy", "pkg")] == ["pcie:0000:00:1c.0", "chipset", "pkg"]
    assert "x4" in s.labels["gpu0.copy"]                                              # the slot limits the link


@pytest.mark.skipif(not RUN.exists(), reason="published run not available")
def test_the_published_run_is_one_machine():
    s = S.build_system(RUN)
    assert set(s.m.devices) == {"gpu0", "igpu"} and s.cpu["mem"] == "DDR5-6200" and "pcie:0000:00:01.0" in s.m.nodes
