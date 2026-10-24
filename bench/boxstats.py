#!/usr/bin/env python3
"""Box stats while a benchmark runs (2026-10-20): one sample every --every seconds into a long CSV (ts,key,value),
until SIGTERM. Host side, stdlib + nvidia-smi only. run-stages.sh starts it for the whole run; benchreport.py averages
it over each item's time window, so every result comes with what the machine was doing meanwhile.

Keys: gpu<i>.util  gpu<i>.mem_mb  gpu<i>.power_w  gpu<i>.temp_c  gpu<i>.sm_mhz  gpu<i>.pcie_rx_mbs  gpu<i>.pcie_tx_mbs
      cpu.busy_pct  cpu.iowait_pct  cpu.max_core_pct  cpu.cores_over_90  cpu.core<n>_pct (each logical CPU, pass 4)
      ctr.<container>.cpu_cores  (cores a running container used, from its cgroup; pass 4)  ram.used_gb  ram.avail_gb  ram.cached_gb  swap.used_gb
      disk.<dev>.read_mbs  disk.<dev>.write_mbs   (whole NVMe/SATA disks only)
"""
import argparse
import json
import csv
import re
import signal
import subprocess
import sys
import time

STOP = False


def gpus():
    out = {}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu,"
                            "clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
        for line in q.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 6 and p[0].isdigit():
                for k, v in zip(("util", "mem_mb", "power_w", "temp_c", "sm_mhz"), p[1:6]):
                    out[f"gpu{p[0]}.{k}"] = v
        # PCIe throughput (MB/s): one dmon sample
        d = subprocess.run(["nvidia-smi", "dmon", "-s", "t", "-c", "1"], capture_output=True, text=True, timeout=15).stdout
        for line in d.splitlines():
            p = line.split()
            if p and p[0].isdigit() and len(p) >= 3:
                out[f"gpu{p[0]}.pcie_rx_mbs"], out[f"gpu{p[0]}.pcie_tx_mbs"] = p[1], p[2]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return out


def cpu_times():
    f = open("/proc/stat").readline().split()[1:]
    v = [int(x) for x in f]
    idle, iowait = v[3], v[4]
    return sum(v), idle, iowait


def core_times():
    """Per logical CPU: (total, idle+iowait) — for the busiest core (Strata's expert threads can pin a few)."""
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            p = line.split(); v = [int(x) for x in p[1:]]
            out[p[0]] = (sum(v), v[3] + v[4])
    return out


_CTR = {"t": 0.0, "ids": {}}


def containers():
    """Running containers' cgroup CPU usage (µs) by name; the id → name map is refreshed every 30 s."""
    if time.time() - _CTR["t"] > 30:
        try:
            out = subprocess.run(["docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"], capture_output=True,
                                 text=True, timeout=15).stdout
            _CTR["ids"] = dict(line.split(None, 1) for line in out.splitlines() if " " in line)
        except (OSError, subprocess.TimeoutExpired):
            _CTR["ids"] = {}
        _CTR["t"] = time.time()
    use = {}
    for cid, name in _CTR["ids"].items():
        for path in (f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/cpu.stat", f"/sys/fs/cgroup/docker/{cid}/cpu.stat"):
            try:
                for line in open(path):
                    if line.startswith("usage_usec"):
                        use[name.strip()] = int(line.split()[1])
                break
            except OSError:
                continue
    return use


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) / 1048576           # kB -> GiB
    return {"ram.used_gb": round(m["MemTotal"] - m["MemAvailable"], 2), "ram.avail_gb": round(m["MemAvailable"], 2),
            "ram.cached_gb": round(m.get("Cached", 0), 2), "swap.used_gb": round(m["SwapTotal"] - m["SwapFree"], 2)}


DISK = re.compile(r"^(nvme\d+n\d+|sd[a-z]+)$")


def disks():
    out = {}
    for line in open("/proc/diskstats"):
        p = line.split()
        if len(p) > 10 and DISK.match(p[2]):
            out[p[2]] = (int(p[5]), int(p[9]))           # sectors read, written (512 B)
    return out


MEMBW = "/run/ai-membw/latest.json"     # the ai-membw sampler's latest pcm-memory sample (2026-10-21), if it runs


def membw():
    try:
        d = json.load(open(MEMBW))
    except (OSError, ValueError):
        return {}
    if time.time() - float(d.get("t") or 0) > 30:     # stale: the sampler stopped
        return {}
    return {f"membw.{k}": d[k] for k in ("read_mbs", "write_mbs") if isinstance(d.get(k), (int, float))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=5.0)
    a = ap.parse_args()

    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    c0, d0, t0, k0, u0 = cpu_times(), disks(), time.time(), core_times(), containers()
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        while not STOP:
            time.sleep(a.every)
            now = time.time(); c1, d1 = cpu_times(), disks()
            rows = dict(gpus()); rows.update(meminfo()); rows.update(membw())
            tot = c1[0] - c0[0]
            if tot > 0:
                rows["cpu.busy_pct"] = round(100 * (tot - (c1[1] - c0[1]) - (c1[2] - c0[2])) / tot, 1)
                rows["cpu.iowait_pct"] = round(100 * (c1[2] - c0[2]) / tot, 1)
            k1 = core_times()
            per = {c: 100 * (k1[c][0] - k0[c][0] - (k1[c][1] - k0[c][1])) / (k1[c][0] - k0[c][0])
                   for c in k1 if c in k0 and k1[c][0] > k0[c][0]}
            busy = list(per.values())
            for c, b in per.items():
                rows[f"cpu.core{c[3:]}_pct"] = round(b, 1)
            u1 = containers()
            for name, us in u1.items():
                if name in u0 and now > t0:
                    rows[f"ctr.{name}.cpu_cores"] = round((us - u0[name]) / 1e6 / (now - t0), 2)
            u0 = u1
            if busy:
                rows["cpu.max_core_pct"] = round(max(busy), 1)
                rows["cpu.cores_over_90"] = sum(1 for b in busy if b > 90)
            k0 = k1
            dt = now - t0
            for dev, (r, wr) in d1.items():
                if dev in d0 and dt > 0:
                    rows[f"disk.{dev}.read_mbs"] = round((r - d0[dev][0]) * 512 / 1e6 / dt, 1)
                    rows[f"disk.{dev}.write_mbs"] = round((wr - d0[dev][1]) * 512 / 1e6 / dt, 1)
            c0, d0, t0 = c1, d1, now
            ts = round(now, 1)
            w.writerows([ts, k, v] for k, v in rows.items())
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
