"""The on-box queue against a simulated host (nvidia-smi, Docker, cpufreq sysfs): every run must leave the host as
it found it. Run: python3 -m pytest -q bench/hw/tests"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "runner"))
sys.path.insert(0, str(HERE / "convert"))
import archive  # noqa: E402
import hwq  # noqa: E402
import to_rigsim  # noqa: E402

DIGEST = "registry.example/ai/rigsim-hwbench/cuda@sha256:" + "a" * 64


class FakeHost:
    def __init__(self):
        self.gpus = [dict(index=0, bus="00000000:17:00.0", model="NVIDIA GeForce RTX 3060", limit=140.0, lo=100.0, hi=170.0, dflt=170.0),
                     dict(index=1, bus="00000000:65:00.0", model="NVIDIA GeForce RTX 3090", limit=250.0, lo=100.0, hi=350.0, dflt=350.0)]
        self.containers = {"c" * 64: dict(name="strata", running=True, image="i1"),
                           "d" * 64: dict(name="caddy", running=True, image="i2"),
                           "e" * 64: dict(name="old-exited", running=False, image="i1")}
        self.images = {"i1", "i2", "i-unused"}
        self.volumes = {"v1"}
        self.log: list[list[str]] = []
        self.pl_history: list[tuple[int, float]] = []
        self.on_run = None

    def __call__(self, cmd, timeout=None, check=False):
        self.log.append(cmd)
        out, rc = "", 0
        j = " ".join(cmd)
        if cmd[0] == "nvidia-smi" and cmd[1].startswith("--query-gpu"):
            out = "\n".join(f"{g['index']}, {g['bus']}, {g['model']}, {g['limit']:.2f}, {g['lo']:.2f}, {g['hi']:.2f}, {g['dflt']:.2f}"
                            for g in self.gpus)
        elif cmd[0] == "nvidia-smi" and "-pl" in cmd:
            i, w = int(cmd[cmd.index("-i") + 1]), float(cmd[cmd.index("-pl") + 1])
            self.gpus[i]["limit"] = w; self.pl_history.append((i, w))
        elif j.startswith("docker ps -aq --no-trunc --filter label="):
            out = "\n".join(c for c, v in self.containers.items() if v.get("label"))
        elif j == "docker ps -aq --no-trunc":
            out = "\n".join(self.containers)
        elif j == "docker ps -q --no-trunc":
            out = "\n".join(c for c, v in self.containers.items() if v["running"])
        elif j == "docker images -q --no-trunc":
            out = "\n".join(sorted(self.images))
        elif j.startswith("docker volume ls -q --filter dangling=true"):
            out = "\n".join(sorted(self.volumes))
        elif j == "docker volume ls -q":
            out = "\n".join(sorted(self.volumes))
        elif cmd[:3] == ["docker", "inspect", "-f"]:
            out = "/" + self.containers[cmd[-1]]["name"]
        elif cmd[:2] == ["docker", "stop"]:
            self.containers[cmd[-1]]["running"] = False
        elif cmd[:2] == ["docker", "start"]:
            self.containers[cmd[-1]]["running"] = True
        elif cmd[:3] == ["docker", "image", "inspect"]:
            rc = 0 if cmd[-1] in self.images else 1
        elif cmd[:2] == ["docker", "pull"]:
            self.images.add(cmd[-1])
        elif cmd[:2] == ["docker", "run"]:
            if self.on_run:
                self.on_run(cmd)
            img_at = next(i for i, x in enumerate(cmd) if "@sha256:" in x)
            out = json.dumps(dict(mode="fake", argv=cmd[img_at + 1:], limits={g["index"]: g["limit"] for g in self.gpus}))
        elif cmd[:3] == ["docker", "rm", "-f"]:
            self.containers.pop(cmd[-1], None)
        elif cmd[:2] == ["docker", "rmi"]:
            self.images.discard(cmd[-1])
        elif cmd[:3] == ["docker", "volume", "rm"]:
            self.volumes.discard(cmd[-1])
        elif cmd[:2] == ["sh", "-c"]:
            out = "\n".join(v["image"] for v in self.containers.values())
        return subprocess.CompletedProcess(cmd, rc, out, "")


@pytest.fixture
def host(monkeypatch, tmp_path):
    h = FakeHost()
    monkeypatch.setattr(hwq, "SH", h)
    monkeypatch.setattr(hwq.time, "sleep", lambda s: None)
    monkeypatch.setattr(hwq.shutil, "which", lambda x: None)        # no real sampler
    return h


@pytest.fixture
def sysfs(tmp_path):
    root = tmp_path / "cpu"
    for c in range(2):
        d = root / f"cpu{c}" / "cpufreq"; d.mkdir(parents=True)
        for k, v in dict(scaling_governor="powersave", scaling_min_freq="1200000", scaling_max_freq="4700000",
                         cpuinfo_min_freq="1200000", cpuinfo_max_freq="4700000").items():
            (d / k).write_text(v)
    return str(root)


def plan(tmp_path, **over) -> str:
    body = {
        "question": "t-gemm", "asks": "x", "feeds": "y", "image": "cuda", "gpus": "item", "exclusive": True,
        "repeats": 2, "settle_s": 0, "max_attempts": 2,
        "grid": {"gpu": ["@all"], "power_w": ["default"], "prec": ["fp16", "int8"]},
        "args": {"reps": 10, "cmd": ["probe", "gemm", "--prec", "{prec}", "--reps", "{reps}"]},
        "smoke": {"grid": {"prec": ["fp16"]}, "args": {"reps": 1}},
    }
    body.update(over)
    lines = []
    for k, v in body.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {json.dumps(v)}")
    for k, v in body.items():
        if isinstance(v, dict):
            lines.append(f"[{k}]")
            for kk, vv in v.items():
                lines.append(f"{kk} = " + (json.dumps(vv) if not isinstance(vv, dict) else
                                           "{ " + ", ".join(f"{a} = {json.dumps(b)}" for a, b in vv.items()) + " }"))
    p = tmp_path / "q.toml"; p.write_text("\n".join(lines) + "\n")
    return str(p)


def cfg(tmp_path):
    return dict(images={"cuda": DIGEST}, gpu_names={"0000:17:00.0": "3060", "0000:65:00.0": "3090-gigabyte"},
                keep_running=["caddy"], testdir_parent=str(tmp_path / "disk"), min_free_gb=0)


def as_found(h: FakeHost):
    assert [g["limit"] for g in h.gpus] == [140.0, 250.0]
    assert h.containers["c" * 64]["running"] and h.containers["d" * 64]["running"]


# --- plans ---------------------------------------------------------------------------------------------------------
def test_every_question_in_the_repo_is_valid():
    for q in sorted((HERE / "questions").glob("*.toml")):
        p = hwq.load_plan(str(q))
        assert hwq.expand(p, False) and hwq.expand(p, True), q.name
        assert len(hwq.expand(p, True)) <= 4, f"{q.name}: smoke runs stay short"
        hwq.command_for(hwq.expand(p, False)[0])


def test_expand_with_gpus_and_smoke(tmp_path):
    p = hwq.load_plan(plan(tmp_path))
    gpus = [dict(name="3060"), dict(name="3090-gigabyte")]
    assert len(hwq.expand(p, False, gpus)) == 2 * 1 * 2 * 2
    sm = hwq.expand(p, True, gpus)
    assert len(sm) == 2 and hwq.command_for(sm[0]) == ["probe", "gemm", "--prec", "fp16", "--reps", "1"]


def test_unpinned_image_is_refused(tmp_path, host):
    c = cfg(tmp_path); c["images"]["cuda"] = "registry.example/x:latest"
    with pytest.raises(ValueError):
        hwq.run(plan(tmp_path), tmp_path / "run", c, False, False, lambda m: None)


# --- a run ---------------------------------------------------------------------------------------------------------
def test_run_measures_every_item_and_leaves_the_host_as_found(tmp_path, host):
    rd = tmp_path / "run"
    assert hwq.run(plan(tmp_path), rd, cfg(tmp_path), False, False, lambda m: None) == 0
    rows = [json.loads(l) for l in (rd / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 2 * 2 and all(r["rc"] == 0 for r in rows)
    # during an item: its card at the item's limit ('default' = 170 / 350 W), the other card at its baseline
    r = next(r for r in rows if r["params"]["gpu"] == "3090-gigabyte")
    assert r["results"][0]["limits"] == {"0": 140.0, "1": 350.0}
    run_cmds = [c for c in host.log if c[:2] == ["docker", "run"]]
    assert all("--network" in c and c[c.index("--network") + 1] == "none" for c in run_cmds)
    assert all(c[c.index("--gpus") + 1] in ("device=0", "device=1") for c in run_cmds)
    assert not host.containers["c" * 64].get("label")                 # strata was stopped and started again
    assert ["docker", "stop", "-t", "60", "c" * 64] in host.log
    assert ["docker", "stop", "-t", "60", "d" * 64] not in host.log  # caddy is in keep_running
    as_found(host)
    assert json.loads((rd / "restore.json").read_text())["problems"] == []


def test_a_resumed_run_does_not_repeat_done_items(tmp_path, host):
    rd = tmp_path / "run"
    hwq.run(plan(tmp_path), rd, cfg(tmp_path), False, False, lambda m: None)
    n = sum(1 for c in host.log if c[:2] == ["docker", "run"])
    hwq.run(plan(tmp_path), rd, cfg(tmp_path), False, False, lambda m: None)
    assert sum(1 for c in host.log if c[:2] == ["docker", "run"]) == n
    as_found(host)


def test_baseline_is_kept_from_before_the_first_item(tmp_path, host):
    """A run interrupted with a card at 350 W must not record 350 W as its baseline when it resumes."""
    rd = tmp_path / "run"
    def interrupt(cmd):
        raise hwq.Stop()
    host.on_run = interrupt
    assert hwq.run(plan(tmp_path), rd, cfg(tmp_path), False, False, lambda m: None) == 130
    as_found(host)                                                   # restored on the way out
    host.on_run = None
    host.gpus[1]["limit"] = 350.0                                    # something left it there
    hwq.run(plan(tmp_path), rd, cfg(tmp_path), False, False, lambda m: None)
    assert host.gpus[1]["limit"] == 250.0                            # the original baseline wins


def test_points_outside_a_cards_range_are_skipped(tmp_path, host):
    rd = tmp_path / "run"
    p = plan(tmp_path, grid={"gpu": ["@all"], "power_w": [200], "prec": ["fp16"]}, repeats=1)
    hwq.run(p, rd, cfg(tmp_path), False, False, lambda m: None)
    st = json.loads((rd / "state.json").read_text())
    skipped = [k for k, v in st.items() if v["status"] == "skipped"]
    assert len(skipped) == 1 and "gpu=3060" in skipped[0]           # 200 W > the 3060's 170 W
    as_found(host)


def test_failed_items_are_retried_then_marked_failed(tmp_path, host):
    rd = tmp_path / "run"
    real = host.__call__
    def failing(cmd, timeout=None, check=False):
        if cmd[:2] == ["docker", "run"]:
            host.log.append(cmd)
            return subprocess.CompletedProcess(cmd, 3, "", "cuBLAS status 13")
        return real(cmd, timeout, check)
    hwq.SH = failing
    try:
        hwq.run(plan(tmp_path, repeats=1), rd, cfg(tmp_path), False, False, lambda m: None)
    finally:
        hwq.SH = host
    st = json.loads((rd / "state.json").read_text())
    assert all(v["status"] == "failed" and v["attempts"] == 2 for v in st.values())
    as_found(host)


def test_cpu_clock_is_pinned_per_item_and_restored(tmp_path, host, sysfs):
    rd = tmp_path / "run"
    seen = []
    host.on_run = lambda cmd: seen.append(Path(sysfs, "cpu0/cpufreq/scaling_max_freq").read_text())
    p = plan(tmp_path, gpus="none", image="cuda", grid={"cpu_mhz": [2400, 9999]}, repeats=1,
             args={"cmd": ["cpuprobe", "latency"]})
    hwq.run(p, rd, cfg(tmp_path), False, False, lambda m: None, root=sysfs)
    assert seen == ["2400000"]                                       # 9999 MHz is outside the range: skipped
    for c in ("cpu0", "cpu1"):
        d = Path(sysfs, c, "cpufreq")
        assert (d / "scaling_governor").read_text() == "powersave"
        assert (d / "scaling_min_freq").read_text() == "1200000" and (d / "scaling_max_freq").read_text() == "4700000"


def test_cleanup_removes_only_what_the_run_created(tmp_path, host):
    rd = tmp_path / "run"
    p = plan(tmp_path, testdir=True, repeats=1)
    hwq.run(p, rd, cfg(tmp_path), True, True, lambda m: None)
    host.containers["f" * 64] = dict(name="rigsim-hwbench-x", running=False, image=DIGEST, label=True)
    host.volumes.add("v-new")
    testdir = json.loads((rd / "testdirs.json").read_text())[0]
    assert Path(testdir).exists()
    hwq.cleanup(rd, lambda m: None)
    assert "f" * 64 not in host.containers and DIGEST not in host.images and "v-new" not in host.volumes
    assert {"i1", "i2", "i-unused"} <= host.images and "v1" in host.volumes and "e" * 64 in host.containers
    assert not Path(testdir).exists()


# --- archive and conversion ------------------------------------------------------------------------------------------
def test_archive_masks_and_refuses_leftovers(tmp_path):
    d = tmp_path / "fetch"; d.mkdir()
    (d / "baseline.json").write_text(json.dumps(dict(
        gpus=[dict(bus="0000:65:00.0", name="3090-gigabyte", uuid="GPU-12345678-1234-1234-1234-123456789abc")],
        note="ssh ai-server at 100.64.1.2 from /home/alice/AI-SERVER, nic aa:bb:cc:dd:ee:ff, other 0000:b3:00.0")))
    assert archive.main([str(d), "--gpu-names", '{"0000:65:00.0": "3090-gigabyte"}', "--hosts", '["ai-server"]']) == 0
    text = (tmp_path / "fetch-public" / "baseline.json").read_text()
    for bad in ("100.64.1.2", "alice", "aa:bb:cc", "ai-server", "0000:65:00.0", "0000:b3:00.0", "GPU-1234"):
        assert bad not in text, bad
    assert "3090-gigabyte" in text and (tmp_path / "fetch-public.tar.gz").exists()
    assert archive.leftovers("host 10.1.2.3", []) and not archive.leftovers("cuda 12.8.1 driver 615.71.09", [])


def test_converter_clock_power(tmp_path):
    d = tmp_path / "r"; d.mkdir()
    (d / "baseline.json").write_text(json.dumps(dict(gpus=[dict(name="3090-gigabyte", model="NVIDIA GeForce RTX 3090")])))
    rows = []
    for w, mhz in ((220, 1250), (250, 1570)):
        for rep in range(2):
            rows.append(dict(question="gpu-clock-power", rc=0, hot=[], params=dict(gpu="3090-gigabyte", load="compute", duty=1.0, power_w=w),
                             results=[dict(busy_share=0.99)], samples={"3090-gigabyte": {"clocks.sm": {"median": mhz + rep * 10},
                                                                                         "power.draw": {"median": w - 3}}}))
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = to_rigsim.convert(d)
    pts = out["cards"]["3090-gigabyte"]["points"]
    assert out["trust"] == "high" and out["cards"]["3090-gigabyte"]["model"] == "rtx-3090"
    assert [p["limit_w"] for p in pts] == [220.0, 250.0] and pts[0]["sm_mhz"]["median"] == 1255


def test_converter_gemm_share_of_clock_peak(tmp_path):
    d = tmp_path / "r"; d.mkdir()
    (d / "baseline.json").write_text(json.dumps(dict(gpus=[dict(name="g", model="NVIDIA GeForce RTX 3090")])))
    ops = 0.9 * 82 * 1500e6 * 1024                                  # 90 % of the FP16 clock peak at 1500 MHz
    rows = [dict(question="gpu-gemm", rc=0, hot=[], params=dict(gpu="g", prec="fp16", shape="4096x4096x4096"),
                 results=[dict(ops_per_s=ops, cudart=12080)], samples={"g": {"clocks.sm": {"median": 1500}}}),
            dict(question="gpu-gemm", rc=0, hot=[], params=dict(gpu="g", prec="int8", shape="1x4096x4096"),
                 results=[dict(unsupported=True)], samples={})]
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    m = {(x["prec"], x["shape"]): x for x in to_rigsim.convert(d)["measurements"]}
    assert abs(m[("fp16", "4096x4096x4096")]["share_of_clock_peak"] - 0.9) < 1e-9
    assert m[("int8", "1x4096x4096")]["unsupported"]


@pytest.mark.skipif(shutil.which("ansible-playbook") is None, reason="ansible not installed")
def test_playbooks_parse():
    for pb, extra in (("hwbench.yml", ["-e", "question=x", "-e", "target=localhost"]),
                      ("images.yml", ["-e", "build_host=localhost", "-e", "probes=cuda"])):
        r = subprocess.run(["ansible-playbook", "--syntax-check", "-i", "localhost,", str(HERE / "ansible" / pb), *extra],
                           capture_output=True, text=True, env={"ANSIBLE_CONFIG": str(HERE / "ansible" / "ansible.cfg"),
                                                                "PATH": "/usr/local/bin:/usr/bin:/bin"})
        assert r.returncode == 0, r.stderr
