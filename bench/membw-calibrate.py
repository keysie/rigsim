#!/usr/bin/env python3
"""Memory bandwidth calibration (2026-10-21, user): where is 100 % on this box? Compiles membw_probe.c, runs each
kernel (read, write, copy, triad) at 1 … all threads while the ai-membw sampler (pcm-memory) measures, and takes the
highest pcm-measured total as the peak — the same counter the board and the bench use, so their percentages are
against what the box really reaches, not the 102.4 GB/s on paper. Run with the box otherwise idle (pass 2 stops the
other engines first). Host side, stdlib + gcc.

  membw-calibrate.py --out DIR [--cache CDIR] [--seconds 16] [--threads 1,2,4,6,8,10,20] [--mib 1024]
  membw-calibrate.py --check CDIR              reuse or measure? exit 0 = "reuse …", 3 = "measure: <reason>"
  membw-calibrate.py --reuse CDIR --out DIR    copy the cached calibration to DIR if the hardware is unchanged (else 3)

Writes DIR/membw-calibrate.csv (kernel, threads, counted GB/s, pcm read/write/total GB/s), DIR/membw-peak.json
{"peak_total_gbs", "peak_read_gbs", "best": {...}, "theoretical_gbs": 102.4, "measured_at"} and DIR/membw-hardware.json
(the fingerprint below). With --cache, a successful calibration is also kept in CDIR (git-ignored, under the bench log
dir) for later runs.

Reuse (user 2026-10-22): a calibration is reused only while the hardware fingerprint is unchanged — board, BIOS
version and date, CPU model, logical CPUs and physical cores (SMT on/off), and every DIMM (locator, size, configured
speed, part number; dmidecode, root) plus the RAM total in GiB. A BIOS memory profile change shows as a different
configured speed. MEMBW_HW_JSON=<file> replaces the fingerprint (tests).
"""
import argparse, csv, json, os, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = Path(os.environ.get("AI_MEMBW_FILE", "/run/ai-membw/latest.json"))


def latest():
    try:
        d = json.loads(SAMPLE.read_text())
        return d if time.time() - float(d.get("t") or 0) < 15 else None
    except (OSError, ValueError):
        return None


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def hardware():
    """What a calibration depends on (the same board/BIOS/CPU/DIMM fields the collector's hardware section reads)."""
    if os.environ.get("MEMBW_HW_JSON"):
        return json.loads(Path(os.environ["MEMBW_HW_JSON"]).read_text())
    dmi = lambda f: _read(f"/sys/class/dmi/id/{f}")
    cpuinfo = _read("/proc/cpuinfo")
    cores, phys, cid = set(), "0", "0"
    for line in cpuinfo.splitlines() + [""]:
        if line.startswith("physical id"):
            phys = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            cid = line.split(":", 1)[1].strip()
        elif not line.strip():
            cores.add((phys, cid))
    mem_kb = next((int(l.split()[1]) for l in _read("/proc/meminfo").splitlines() if l.startswith("MemTotal:")), 0)
    hw = {"board": dmi("board_name"), "board_vendor": dmi("board_vendor"), "bios": dmi("bios_version"),
          "bios_date": dmi("bios_date"),
          "cpu": next((l.split(":", 1)[1].strip() for l in cpuinfo.splitlines() if l.startswith("model name")), ""),
          "logical_cpus": os.cpu_count(), "physical_cores": len(cores), "ram_gib": round(mem_kb / 1048576)}
    try:
        exe = shutil.which("dmidecode") or "/usr/sbin/dmidecode"     # not on a non-root PATH
        out = subprocess.run([exe, "-t", "17"], capture_output=True, text=True, timeout=30)
        rc, out = out.returncode, out.stdout
    except (OSError, subprocess.TimeoutExpired):
        rc, out = 1, ""
    dimms, cur = [], {}
    for line in (out.splitlines() if rc == 0 else []) + ["Memory Device"]:
        if line.startswith("Memory Device"):
            if cur.get("Size") and "No Module" not in cur.get("Size", ""):
                dimms.append({"locator": cur.get("Locator", ""), "size": cur.get("Size", ""),
                              "speed": cur.get("Configured Memory Speed", cur.get("Configured Clock Speed", "")),
                              "part": cur.get("Part Number", "").strip()})
            cur = {}
        elif ":" in line and line.startswith("\t") and not line.startswith("\t\t"):
            k, v = line.strip().split(":", 1); cur[k.strip()] = v.strip()
    hw["dimms"] = dimms if rc == 0 else "dmidecode unavailable"
    return hw


def check(cache: Path):
    """(True, why) when the cached calibration can be reused, else (False, reason)."""
    need = [cache / n for n in ("membw-peak.json", "membw-calibrate.csv", "hardware.json")]
    if not all(f.exists() for f in need):
        return False, f"no previous calibration in {cache}"
    try:
        old, peak = json.loads(need[2].read_text()), json.loads(need[0].read_text())
    except ValueError:
        return False, "the cached calibration is unreadable"
    if not peak.get("peak_total_gbs"):
        return False, "the cached calibration has no peak"
    now = hardware()
    if isinstance(now.get("dimms"), str) or isinstance(old.get("dimms"), str):
        return False, "the DIMMs cannot be read (dmidecode) — measuring to be safe"
    diff = [k for k in sorted(set(old) | set(now)) if old.get(k) != now.get(k)]
    if diff:
        return False, "hardware changed since the last calibration: " + ", ".join(
            f"{k} {old.get(k)!r} -> {now.get(k)!r}" if k != "dimms" else "dimms" for k in diff)
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(peak.get("measured_at") or 0))
    return True, f"calibration of {when} ({peak['peak_total_gbs']} GB/s), hardware unchanged"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="", help="keep a successful calibration here for later runs")
    ap.add_argument("--check", default="", metavar="CDIR", help="only say whether CDIR's calibration can be reused")
    ap.add_argument("--reuse", default="", metavar="CDIR", help="copy CDIR's calibration to --out if still valid")
    ap.add_argument("--seconds", type=float, default=16)      # >= 2 pcm samples (5 s) fully inside each run
    ap.add_argument("--threads", default="")
    ap.add_argument("--mib", type=int, default=1024)
    ap.add_argument("--kernels", default="read,write,copy,triad")
    a = ap.parse_args()
    if a.check or a.reuse:
        ok, why = check(Path(a.check or a.reuse))
        if not ok:
            print(f"measure: {why}")
            return 3
        if a.reuse:
            if not a.out:
                ap.error("--reuse needs --out")
            out, cache = Path(a.out), Path(a.reuse)
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(cache / "membw-calibrate.csv", out / "membw-calibrate.csv")
            shutil.copy(cache / "hardware.json", out / "membw-hardware.json")
            peak = json.loads((cache / "membw-peak.json").read_text())
            peak["reused_at"] = int(time.time())          # measured_at stays the real measurement's time
            (out / "membw-peak.json").write_text(json.dumps(peak, indent=1) + "\n")
            print(f"reused: {why}")
        else:
            print(f"reuse: {why}")
        return 0
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for _ in range(30):                           # the sampler writes every 5 s; give it a few rounds
        if latest():
            break
        time.sleep(1)
    if not latest():
        try:
            age = f"{time.time() - float(json.loads(SAMPLE.read_text()).get('t') or 0):.0f} s old"
        except (OSError, ValueError):
            age = "missing"
        print(f"membw-calibrate: no fresh pcm-memory sample ({SAMPLE}: {age}; ai-membw not running or its samples"
              " arrive in bursts) — cannot calibrate", file=sys.stderr)
        return 2
    exe = Path(tempfile.mkdtemp()) / "membw_probe"
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", str(HERE / "membw_probe.c"), "-o", str(exe)], check=True)
    ncpu = os.cpu_count() or 1
    threads = [int(x) for x in a.threads.split(",")] if a.threads else sorted({1, 2, 4, 6, 8, ncpu // 2, ncpu})
    rows = []
    for th in threads:
        for k in a.kernels.split(","):
            samples, stop = [], threading.Event()

            def poll():
                seen = 0
                while not stop.is_set():
                    d = latest()
                    if d and d["t"] != seen and d["t"] > t0 + 5.5:     # samples fully inside the run
                        seen = d["t"]; samples.append(d)
                    time.sleep(0.5)
            t0 = time.time()
            th_ = threading.Thread(target=poll); th_.start()
            p = subprocess.run([str(exe), str(a.mib), str(a.seconds), k], capture_output=True, text=True,
                               env={**os.environ, "OMP_NUM_THREADS": str(th), "OMP_PROC_BIND": "spread"})
            stop.set(); th_.join()
            line = (p.stdout.strip().splitlines() or [""])[-1].split(",")
            best = max(samples, key=lambda d: d.get("total_mbs") or 0, default={})
            g = lambda key: round(best[key] / 1000, 2) if best.get(key) is not None else None
            r = {"kernel": k, "threads": th, "counted_gbs": float(line[2]) if len(line) > 2 else None,
                 "pcm_read_gbs": g("read_mbs"), "pcm_write_gbs": g("write_mbs"), "pcm_total_gbs": g("total_mbs"),
                 "pcm_samples": len(samples)}
            rows.append(r)
            pcm = (f"pcm {r['pcm_total_gbs']} GB/s (read {r['pcm_read_gbs']}, write {r['pcm_write_gbs']})"
                   if samples else "pcm – (no sample fully inside the run)")        # never a 0.0 that looks measured
            print(f"{k:5} {th:2} threads: counted {r['counted_gbs']} GB/s, {pcm}", flush=True)
    with open(out / "membw-calibrate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    measured = [r for r in rows if r["pcm_total_gbs"] is not None]
    if not measured:
        print("membw-calibrate: no pcm-memory sample inside any run — no peak", file=sys.stderr)
        return 1
    best = max(measured, key=lambda r: r["pcm_total_gbs"])
    peak = {"peak_total_gbs": best["pcm_total_gbs"], "peak_read_gbs": max(r["pcm_read_gbs"] or 0 for r in measured),
            "best": best, "runs_measured": len(measured), "runs": len(rows),
            "theoretical_gbs": 102.4, "measured_at": int(time.time())}
    (out / "membw-peak.json").write_text(json.dumps(peak, indent=1) + "\n")
    hw = hardware()
    (out / "membw-hardware.json").write_text(json.dumps(hw, indent=1) + "\n")
    if a.cache and best["pcm_total_gbs"] > 0:            # keep it for later runs (reused while the hardware stays)
        c = Path(a.cache); c.mkdir(parents=True, exist_ok=True)
        for n in ("membw-peak.json", "membw-calibrate.csv"):
            shutil.copy(out / n, c / n)
        (c / "hardware.json").write_text(json.dumps(hw, indent=1) + "\n")
    print(f"peak: {peak['peak_total_gbs']} GB/s total, {peak['peak_read_gbs']} GB/s read "
          f"({100 * peak['peak_total_gbs'] / 102.4:.0f} % of 102.4; {len(measured)} of {len(rows)} runs had a sample)")
    return 0 if best["pcm_total_gbs"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
