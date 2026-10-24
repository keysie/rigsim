#!/usr/bin/env python3
"""One status line per bench plan for ops/bench.yml's wait loop (user 2026-10-21: show that it progresses).

  benchprogress.py LOGDIR --since EPOCH PLAN [PLAN ...]

LOGDIR = run-stages' log dir (plan state dirs are LOGDIR/<plan>; a PLAN containing / is a state dir); --since = the run's start (older state is not this
run's). Prints: elapsed, per plan done/skipped/failed of total, the running item and for how long, the last event, and
PAUSED when the pause file is set. Before the first plan starts it shows the last line of last-run.txt (calibration).
"""
import argparse
import json
import os
import time
from pathlib import Path


def mins(s):
    s = int(max(0, s))
    return f"{s // 3600}:{s % 3600 // 60:02d} h" if s >= 3600 else f"{s // 60} min"


def last_line(f: Path) -> str:
    try:
        lines = [l for l in f.read_text(errors="replace").splitlines() if l.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("plans", nargs="*")
    ap.add_argument("--since", type=float, default=0)
    a = ap.parse_intermixed_args(argv)
    d, now = Path(a.logdir), time.time()
    out = [f"{mins(now - a.since)} elapsed" if a.since else "running"]
    pause = Path(os.environ.get("BENCH_PAUSE_FILE") or d / "PAUSE")
    started = False
    for p in a.plans:
        sd = Path(p) if "/" in p else d / p            # a plan name (state dir LOGDIR/<plan>) or a state dir
        p = sd.name
        try:
            if (sd / "state.json").stat().st_mtime < a.since:
                out.append(f"{p}: waiting"); continue
            items = json.loads((sd / "state.json").read_text())["items"]
        except (OSError, ValueError, KeyError):
            out.append(f"{p}: waiting"); continue
        started = True
        c = {}
        for i in items:
            c[i["status"]] = c.get(i["status"], 0) + 1
        s = f"{p}: {c.get('done', 0)} done"
        s += "".join(f", {c[k]} {k}" for k in ("skipped", "failed") if c.get(k))
        s += f" of {len(items)}"
        run = [i for i in items if i["status"] == "running"]
        if run:
            s += f" | now {run[0]['id'].split('/effort=')[0]} ({mins(now - run[0].get('started', now))})"
        ev = last_line(sd / "events.log")
        if ev:
            s += f" | last: {ev[11:19]} {ev[20:][:110]}"
        out.append(s)
    if not started:
        ll = last_line(d / "last-run.txt") or last_line(d / "membw-calibrate.csv")
        if ll:
            out.append(f"stage: {ll[:120]}")
    if pause.exists():
        out.append("PAUSED (the running item finishes, then nothing new starts; resume: ops/bench-control.yml -e bench_control=resume)")
    print(" || ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
