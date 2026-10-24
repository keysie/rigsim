#!/usr/bin/env python3
"""Bench queue — runs a whole benchmark plan unattended, one item after another, resumable. Stdlib only.

  benchq.py plan   PLAN.toml            expand + validate, print the queue (changes nothing)
  benchq.py run    PLAN.toml [--retry-failed] [--only GLOB] [--smoke] [--fresh]   execute (resumes where it stopped;
                                        --fresh starts over, --smoke always does)
  benchq.py status PLAN.toml            progress, current item, failures
  benchq.py report PLAN.toml            summary tables (benchreport.py) -> stdout + report.md
  benchq.py reset  PLAN.toml GLOB       put matching items back to pending

A plan = variants (one service + its overrides) x workloads (what to measure) x the list-valued knobs of each
workload (effort, sampling profile, reasoning replay). Items run grouped by variant, so each engine restarts
once per variant. Every item gets up to `max_attempts` tries; a variant that does not come up skips its items
and the run goes on. Baselines are restored at the end, after Ctrl-C and on SIGTERM.

--smoke (2026-10-20): the same sequence — every variant starts (model switches included), every workload runs — but
each workload only with the first value of each knob, 1 task, 1 repeat, short answers (`[smoke]` in the plan, defaults
below), a separate state dir (<name>-smoke). Validates the whole chain in minutes instead of hours.
A variant whose `model` is not in the HF cache (MODELS_DIR/hf) is skipped with that reason before any restart.

Unattended: systemd-run --unit=ai-bench --collect --same-dir python3 infra/bench/benchq.py run PLAN.toml
            journalctl -u ai-bench -f        # follow;  systemctl stop ai-bench  # stops cleanly, resumable

State lives in <state_dir>/ (default infra/scripts/logs/bench/<plan name>/): state.json, results.jsonl,
throughput.csv, events.log, status.json (for the dashboard collector), overrides/, report.md.
Hooks for tests: BENCH_SVC (service script, default bench-svc.sh next to this file), BENCH_LOCAL=1 (paths are
host paths, no container mounts), BENCH_NOTIFY_URL (Uptime Kuma push URL; optional).
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
INFRA = HERE.parent
SVC = os.environ.get("BENCH_SVC", str(HERE / "bench-svc.sh"))
LOCAL = os.environ.get("BENCH_LOCAL") == "1"
LIST_KNOBS = ("effort", "sampling", "replay_reasoning", "fill", "shape")
SHAPES = {"direct": None, "turnstone-maker": "tasks/turnstone-shape/maker.json",
          "turnstone-sub": "tasks/turnstone-shape/sub.json"}       # agentloop: request shape (pass 2)
SVC_URL = {"strata": "http://strata:8080/v1", "vllm-sub": "http://vllm-sub:8000/v1",
           "vllm-maker-b": "http://vllm-maker-b:8000/v1", "llama-judge": "http://llama-judge:8080/v1",
           "llama-guard": "http://llama-guard:8080/v1", "llama-rerank": "http://llama-rerank:8080/v1"}
_stop = False


# ================================================================ plan
def load_plan(path: str) -> dict:
    p = Path(path).resolve()
    plan = tomllib.loads(p.read_text())
    plan["_path"] = str(p)
    plan.setdefault("name", p.stem)
    d = plan.setdefault("defaults", {})
    d.setdefault("repeats", 1); d.setdefault("timeout_s", 3600); d.setdefault("ready_timeout_s", 1800)
    d.setdefault("max_attempts", 2); d.setdefault("recreate", "always")
    g = plan.setdefault("guard", {})
    g.setdefault("max_gpu_temp_c", 85); g.setdefault("cooldown_s", 180); g.setdefault("min_free_gb", 2)
    plan.setdefault("effort", {}).setdefault("key", "reasoning_effort")
    plan.setdefault("sampling", {}).setdefault("engine", {})
    return plan


SMOKE_DEFAULTS = {"tasks": 1, "max_tokens": 512, "timeout_s": 600, "tps_streams": "1", "tps_prompt_tokens": 256,
                  "tps_max_tokens": 64}


def smoke(plan: dict) -> dict:
    """The plan as a cheap smoke run: same variants and workloads, first knob value each, 1 task, 1 repeat, short
    answers, own state dir. `[smoke]` in the plan overrides SMOKE_DEFAULTS."""
    p = copy.deepcopy(plan)
    s = {**SMOKE_DEFAULTS, **plan.get("smoke", {})}
    p["name"] = plan["name"] + "-smoke"
    if p.get("state_dir"):
        p["state_dir"] = str(p["state_dir"]).rstrip("/") + "-smoke"
    p["defaults"].update(repeats=1, max_attempts=1, timeout_s=s["timeout_s"])
    if any(v.get("smoke") for v in p["variant"]):          # one variant per mechanism, chosen in the plan
        p["variant"] = [v for v in p["variant"] if v.get("smoke")]
        for w in p.get("workload", []):                      # smoke_variants: where a workload runs in smoke
            if w.get("smoke_variants"):
                w["variants"] = w["smoke_variants"]
        ids = [v["id"] for v in p["variant"]]                # workloads left without a variant drop out
        p["workload"] = [w for w in p.get("workload", [])
                         if any(fnmatch.fnmatch(i, pat) for i in ids for pat in _aslist(w.get("variants", "*")))]
    p["workload"] = [w for w in p.get("workload", []) if not w.get("smoke_skip")]   # slow, nothing new to exercise
    for w in p.get("workload", []):
        for knob in LIST_KNOBS:
            if isinstance(w.get(knob), list) and w[knob]:
                w[knob] = w[knob][:1]
        if w["kind"] == "throughput" and "fill" in w:      # depth: keep the queueing (slots+N), short answers
            w["max_tokens"] = s["tps_max_tokens"]
        elif w["kind"] == "throughput" and w.get("duration_s"):        # sustained: keep its streams, one minute
            w.update(duration_s=60, max_tokens=s["tps_max_tokens"], prompt_tokens=s["tps_prompt_tokens"])
        elif w["kind"] == "throughput":
            w.update(streams=s["tps_streams"], prompt_tokens=s["tps_prompt_tokens"], max_tokens=s["tps_max_tokens"])
        elif w["kind"] == "guard":                              # pass 4: one small call, once (user 2026-10-23)
            w.update(sizes=[min(_aslist(w.get("sizes", [512])))], conc=[1], repeats=1, max_tokens=256)
        elif w["kind"] == "judge":                              # one evaluation with one tool turn
            w.update(history=[1024], tool_turns=[1], conc=[1], repeats=1, read_tokens=256, max_tokens=256)
        elif w["kind"] == "rerank":
            w.update(shapes=_aslist(w.get("shapes", []))[:1], conc=[1], repeats=1,
                     **({"duration_s": 3} if w.get("duration_s") else {}))
        elif w["kind"] == "agentloop":
            w.update(agents="1", turns=3, max_tokens=min(int(w.get("max_tokens", 8192)), s["max_tokens"]))
        elif w["kind"] == "longtask":
            if w.get("fills"):
                w["fills"] = [min(float(x), 0.1) for x in w["fills"]]
            if w.get("stagger_s"):
                w["stagger_s"] = min(float(w["stagger_s"]), 5)
            w.update(max_turns=int(w.get("smoke_max_turns", 1)), until_fill=0, fill=min(float(w.get("fill", 0)), 0.05),
                     max_tokens=min(int(w.get("max_tokens", 8192)), s["max_tokens"], 256))
        else:
            w.update(limit=s["tasks"], repeats=1, streams=1, max_tokens=min(int(w.get("max_tokens", 16384)), s["max_tokens"]))
        w.pop("timeout_s", None)
    return p


_ROLE_VARS = ("GPU_MAKER_A", "GPU_MAKER_B", "GPU_SUB", "GPU_SMALL")


def real_gpu(i: int) -> int:
    """A plan's GPU number is the card's ROLE (0 maker-a, 1 maker-b, 2 sub, 3 small — as gen_phase1.GPU_ENV); the live
    nvidia-smi index comes from infra/.env (role box_env renders it from the PCI order; the 3060 is index 0 since
    2026-10-22). Without a .env: the same number."""
    env = INFRA / ".env"
    if 0 <= i < len(_ROLE_VARS) and env.is_file():
        for line in env.read_text().splitlines():
            k, _, val = line.partition("=")
            if k.strip() == _ROLE_VARS[i] and val.split("#")[0].strip().isdigit():
                return int(val.split("#")[0].strip())
    return i


def models_dir() -> Path:
    """MODELS_DIR from infra/.env (the box), else the default layout."""
    env = INFRA / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "MODELS_DIR" and v.strip():
                return Path(v.split("#")[0].strip().strip('"\''))
    return Path(os.environ.get("MODELS_DIR", "/srv/ai/models"))


def model_missing(v: dict) -> str:
    """Reason a variant's own model cannot load (vLLM, from the HF cache), or ''."""
    m = v.get("model")
    if not m or v.get("service") not in ("vllm-sub", "vllm-maker-b") or m.startswith("/") or LOCAL:
        return ""
    d = models_dir() / "hf" / "hub" / ("models--" + m.strip("/").replace("/", "--")) / "snapshots"
    ok = d.is_dir() and any(any(s.glob("*.safetensors")) or any(s.glob("*.json")) for s in d.iterdir())
    return "" if ok else f"model {m} not in the HF cache ({d.parent}) — download it first (HF1)"


def validate(plan: dict) -> list[str]:
    errs, ids = [], set()
    variants = plan.get("variant", [])
    if not variants:
        errs.append("no [[variant]]")
    for v in variants:
        if v.get("id") in ids:
            errs.append(f"duplicate variant id {v.get('id')}")
        ids.add(v.get("id"))
        if v.get("service") not in SVC_URL:
            errs.append(f"variant {v.get('id')}: unknown service {v.get('service')!r}")
        if v.get("service") != "strata" and any(k.startswith("strata_") for k in v):
            errs.append(f"variant {v['id']}: strata_* keys only for service strata")
        if v.get("gpu_power_w") is not None and not (100 <= int(v["gpu_power_w"]) <= 350):
            errs.append(f"variant {v['id']}: gpu_power_w must be 100–350 W (a 3090's range)")
        if v.get("ladder") and v["ladder"] not in (plan.get("ladder") or {}):
            errs.append(f"variant {v['id']}: ladder {v['ladder']!r} needs a [ladder.{v['ladder']}] section")
        if v.get("expert_cache_frac") is not None and v.get("ladder_base") not in ids:
            errs.append(f"variant {v['id']}: expert_cache_frac needs ladder_base = an earlier variant")
        if v.get("expert_cache_from") and v["expert_cache_from"] not in (plan.get("ladder") or {}):
            errs.append(f"variant {v['id']}: expert_cache_from names no ladder")
        if v.get("only_if_failed") and v["only_if_failed"] not in ids:
            errs.append(f"variant {v['id']}: only_if_failed {v['only_if_failed']!r} must name an earlier variant")
    if plan.get("defaults", {}).get("on_end", "restore") not in ("restore", "stop"):
        errs.append("[defaults] on_end must be restore | stop")
    for lvl, txt in plan["effort"].get("prefix", {}).items():
        if str(txt).startswith("FILL"):
            errs.append(f"[effort.prefix] {lvl} is still a placeholder (copy the sentence from the chat template)")
    if str(plan["effort"].get("key", "")).startswith("FILL"):
        errs.append("[effort] key is still a placeholder")
    wids = set()
    for w in plan.get("workload", []):
        if w.get("id") in wids:
            errs.append(f"duplicate workload id {w.get('id')}")
        wids.add(w.get("id"))
        if w.get("kind") not in ("tasks", "multiturn", "throughput", "agentloop", "longtask", "guard", "judge", "rerank"):
            errs.append(f"workload {w.get('id')}: kind must be tasks | multiturn | throughput | agentloop | longtask")
        if w.get("kind") == "longtask" and not (HERE / w.get("tasks", "-")).is_file():
            errs.append(f"workload {w['id']}: longtask needs tasks = a task file (tasks/longtask-*.json)")
        if w.get("kind") == "agentloop":
            for s in _aslist(w.get("shape", "direct")):
                if s not in SHAPES:
                    errs.append(f"workload {w['id']}: shape {s!r} (direct | turnstone-maker | turnstone-sub)")
                elif SHAPES[s] and not (HERE / SHAPES[s]).is_file():
                    errs.append(f"workload {w['id']}: {SHAPES[s]} missing (tasks/turnstone-shape/make_shape.py)")
            if not w.get("corpus"):
                errs.append(f"workload {w['id']}: agentloop needs corpus")
        if w.get("kind") in ("tasks", "multiturn", "longtask"):
            t = (HERE / w.get("tasks", "")).resolve()
            if not t.is_file():
                errs.append(f"workload {w['id']}: tasks file {w.get('tasks')!r} not found (relative to infra/bench)")
            elif HERE not in t.parents:
                errs.append(f"workload {w['id']}: tasks file must live under infra/bench")
        for s in _aslist(w.get("sampling", "engine")):
            if s not in plan["sampling"]:
                errs.append(f"workload {w['id']}: sampling profile {s!r} not defined under [sampling.{s}]")
        for e in _aslist(w.get("effort", "default")):
            kind, _, lvl = e.partition(":")
            if e == "medium":
                errs.append(f"workload {w['id']}: effort 'medium' is ambiguous — 'default' sends nothing (the template's "
                            "own default, xhigh on Qwen3.8), 'template:medium' sends medium")
            elif e != "default" and kind not in ("template", "prefix"):
                errs.append(f"workload {w['id']}: effort {e!r} (default | template:<lvl> | prefix:<lvl>)")
            if kind == "prefix" and not plan["effort"].get("prefix", {}).get(lvl):
                errs.append(f"workload {w['id']}: prefix:{lvl} needs [effort.prefix] {lvl} = \"<verbatim sentence>\"")
        if not any(fnmatch.fnmatch(v.get("id", ""), pat) for v in variants for pat in _aslist(w.get("variants", "*"))):
            errs.append(f"workload {w['id']}: matches no variant")
    return errs


def _aslist(x):
    return list(x) if isinstance(x, list) else [x]


def expand(plan: dict) -> list[dict]:
    """Items grouped by variant (plan order), then workload order, then knob combinations."""
    items = []
    for v in plan["variant"]:
        for w in plan.get("workload", []):
            if not any(fnmatch.fnmatch(v["id"], pat) for pat in _aslist(w.get("variants", "*"))):
                continue
            combos = [{}]
            for knob in LIST_KNOBS:
                if knob == "replay_reasoning" and w["kind"] != "multiturn":
                    continue
                if knob in ("effort", "sampling") and w["kind"] == "throughput":
                    continue
                if knob == "fill" and (w["kind"] != "throughput" or "fill" not in w):
                    continue
                if knob == "shape" and w["kind"] != "agentloop":
                    continue
                default = {"effort": "default", "sampling": "engine", "replay_reasoning": False, "fill": None,
                           "shape": "direct"}[knob]
                combos = [dict(c, **{knob: val}) for c in combos for val in _aslist(w.get(knob, default))]
            for c in combos:
                parts = [v["id"], w["id"]] + [f"{k}={c[k]}" for k in LIST_KNOBS if k in c]
                items.append({"id": "/".join(parts), "variant": v["id"], "workload": w["id"], "knobs": c,
                              "status": "pending", "attempts": 0})
    return items


# ================================================================ state
class State:
    def __init__(self, plan: dict):
        base = Path(plan.get("state_dir") or (INFRA / "scripts" / "logs" / "bench" / plan["name"]))
        if not base.is_absolute():
            base = (Path(plan["_path"]).parent / base).resolve()
        self.dir = base
        (self.dir / "overrides").mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "state.json"
        self.items: list[dict] = []
        fresh = expand(plan)
        old = {}
        if self.file.exists():
            old = {i["id"]: i for i in json.loads(self.file.read_text())["items"]}
        for it in fresh:                         # keep progress of items that still exist in the plan
            o = old.get(it["id"])
            self.items.append({**it, **{k: o[k] for k in ("status", "attempts", "runs", "last_error", "started",
                                                          "finished", "duration_s") if o and k in o}})
        for it in self.items:                    # an item that was running when we died starts over
            if it["status"] == "running":
                it["status"] = "pending"
        self.save()

    def save(self):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"items": self.items}, indent=1))
        tmp.replace(self.file)

    def counts(self):
        c = {}
        for i in self.items:
            c[i["status"]] = c.get(i["status"], 0) + 1
        return c


def event(st: State, msg: str, notify: str | None = None):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(st.dir / "events.log", "a") as f:
        f.write(line + "\n")
    url = os.environ.get("BENCH_NOTIFY_URL")
    if url and notify:
        q = urllib.parse.urlencode({"status": "down" if notify == "down" else "up", "msg": msg[:200]})
        try:
            urllib.request.urlopen(f"{url}{'&' if '?' in url else '?'}{q}", timeout=10).read()
        except OSError:
            pass


def pause_file() -> Path:
    """Pause between items (user 2026-10-21): while this file exists no new item starts; the running one finishes.
    run-stages.sh exports BENCH_PAUSE_FILE (its log dir); `benchq.py pause|resume` and ops/bench-control.yml set it."""
    return Path(os.environ.get("BENCH_PAUSE_FILE") or (INFRA / "scripts" / "logs" / "bench" / "PAUSE"))


def wait_while_paused(st: State, plan: dict, nxt: str):
    if not pause_file().exists() or _stop:
        return
    t0 = time.time()
    event(st, f"paused before {nxt} (engines idle; resume: benchq.py resume)", "down")
    while pause_file().exists() and not _stop:
        write_status(st, plan, f"PAUSED before {nxt}")
        time.sleep(5)
    if not _stop:
        event(st, f"resumed after {int(time.time() - t0) // 60} min", "up")


def write_status(st: State, plan: dict, current: str = ""):
    s = {"plan": plan["name"], "updated": int(time.time()), "current": current, "counts": st.counts(),
         "total": len(st.items)}
    (st.dir / "status.json").write_text(json.dumps(s))


# ================================================================ overrides
def _patch_args(args: list[str], v: dict, prefix: str = "") -> list[str]:
    """args_set {flag: value} replaces '--f=x' or '--f x' (or adds); args_remove [flag]; args_add [str]."""
    args = list(args)
    for flag in v.get(f"{prefix}args_remove", []):
        out, skip = [], False
        for i, a in enumerate(args):
            if skip:
                skip = False; continue
            if a == flag:
                nxt = args[i + 1] if i + 1 < len(args) else ""
                skip = bool(nxt) and not nxt.startswith("-")
                continue
            if a.startswith(flag + "="):
                continue
            out.append(a)
        args = out
    for flag, val in v.get(f"{prefix}args_set", {}).items():
        val = str(val)
        for i, a in enumerate(args):
            if a.startswith(flag + "="):
                args[i] = f"{flag}={val}"; break
            if a == flag:
                if val == "":
                    break
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    args[i + 1] = val
                else:
                    args.insert(i + 1, val)
                break
        else:
            args += [flag] if val == "" else [flag, val]
    return args + [str(a) for a in v.get(f"{prefix}args_add", [])]


def build_override(st: State, v: dict) -> tuple[Path | None, str]:
    """Write a compose override for the variant. Returns (path or None for 'no change', short hash)."""
    svc = v["service"]
    ov: dict = {}
    if "command" in v:                            # full replacement (e.g. llama.cpp in the vllm-sub slot)
        ov["command"] = [str(a) for a in v["command"]]
    elif any(k in v for k in ("args_set", "args_add", "args_remove", "model")):
        base = json.loads(subprocess.run([SVC, "base", svc], check=True, capture_output=True, text=True).stdout)
        cmd = list(base.get("command") or [])
        if v.get("model"):
            if "-m" in cmd:                       # llama.cpp
                cmd[cmd.index("-m") + 1] = v["model"]
            else:                                 # vLLM: first positional
                pos = next((i for i, a in enumerate(cmd) if not a.startswith("-")), None)
                cmd = cmd[:pos] + [v["model"]] + cmd[pos + 1:] if pos is not None else [v["model"]] + cmd
        ov["command"] = _patch_args(cmd, v)
    if v.get("env"):
        ov["environment"] = {k: str(x) for k, x in v["env"].items()}
    if any(k.startswith("strata_") for k in v):
        src = INFRA / "strata" / "strata-maker.json"
        cfg = json.loads(Path(os.environ.get("BENCH_STRATA_CONFIG", src)).read_text())
        cfg["args"] = _patch_args(cfg["args"], v, "strata_")
        for k, x in v.get("strata_config_set", {}).items():
            cfg[k] = x
        sp = st.dir / "overrides" / f"{_safe(v['id'])}.strata.json"
        sp.write_text(json.dumps(cfg, indent=1))
        ov["volumes"] = [f"{sp}:/data/config/strata-maker.json:ro"]
    if v.get("image"):
        ov["image"] = v["image"]
    for k, x in (v.get("compose") or {}).items():  # any other service key (deploy, healthcheck, …)
        if k == "volumes" and "volumes" in ov:
            ov["volumes"] = ov["volumes"] + list(x)
        else:
            ov[k] = x
    if not ov:
        return None, "baseline"
    # JSON is valid YAML for compose; keys in compose_replace get compose's !override tag (replace, not merge —
    # e.g. deploy.resources.reservations.devices for another GPU set)
    rep = set(v.get("compose_replace") or [])
    body = ",\n".join(f'  {json.dumps(k)}: {"!override " if k in rep else ""}{json.dumps(x)}' for k, x in ov.items())
    text = '{"services": {' + json.dumps(svc) + ": {\n" + body + "\n}}}\n"
    h = hashlib.sha256(text.encode()).hexdigest()[:10]
    p = st.dir / "overrides" / f"{_safe(v['id'])}.yaml"
    p.write_text(text)
    return p, h


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# ================================================================ execution
def mount(path: Path, st: State) -> str:
    """Host path -> path inside the runner container (bench dir /b, scripts /s, state dir /out)."""
    if LOCAL:
        return str(path)
    path = path.resolve()
    for host, cont in ((st.dir, "/out"), (HERE, "/b"), (INFRA / "scripts", "/s")):
        if path == host or host in path.parents:
            return cont + "/" + str(path.relative_to(host)) if path != host else cont
    raise SystemExit(f"{path} is not reachable from the runner container")


def item_argv(plan: dict, st: State, it: dict, v: dict, w: dict) -> list[str]:
    svc = v["service"]
    url = os.environ.get("BENCH_URL_OVERRIDE") or SVC_URL[svc]
    keyenv = "STRATA_API_KEY" if svc == "strata" else "LOCAL_API_KEY"
    d = plan["defaults"]
    if w["kind"] == "throughput":
        fill = it["knobs"].get("fill")
        ptok = int(fill * v["ctx"]) - int(w.get("max_tokens", 512)) - 512 if fill else int(w.get("prompt_tokens", 2000))
        return [mount(INFRA / "scripts" / "bench.py", st), "--url", url, "--key-env", keyenv,
                "--streams", streams_for(w, v), "--prompt-tokens", str(max(ptok, 256)),
                "--max-tokens", str(w.get("max_tokens", 512)), "--csv", mount(st.dir / "throughput.csv", st),
                "--label", it["id"], "--rounds", str(w.get("rounds", 2))] + (
                    ["--corpus", w["corpus"]] if w.get("corpus") else []) + (
                    ["--duration", str(w["duration_s"])] if w.get("duration_s") else [])
    k = it["knobs"]
    pf = st.dir / "prefixes.json"
    if w["kind"] == "longtask":
        pf.write_text(json.dumps(plan["effort"].get("prefix", {})))
        return [mount(HERE / "benchw.py", st), "longtask", "--url", url, "--key-env", keyenv, "--item", it["id"],
                "--out", mount(st.dir / "results.jsonl", st), "--tasks", mount(HERE / w["tasks"], st),
                "--corpus", w.get("corpus", ""), "--fill", str(w.get("fill", 0)), "--ctx", str(v.get("ctx") or 131072),
                "--streams", str(w.get("conversations") or v.get("slots") or 1), "--max-turns", str(w.get("max_turns", 3)),
                "--until-fill", str(w.get("until_fill", 0)), "--min-tps", str(w.get("min_tps", 0)),
                *(["--think-back"] if w.get("think_back") else []),
                *(["--fills", ",".join(str(x) for x in w["fills"])] if w.get("fills") else []),
                "--stagger", str(w.get("stagger_s", 0)),
                *(["--trace", mount(st.dir / "trace.jsonl", st)] if w.get("trace") else []),
                "--effort", k.get("effort", "default"), "--effort-key", plan["effort"]["key"],
                "--prefix-file", mount(pf, st), "--sampling", json.dumps(plan["sampling"][k.get("sampling", "engine")]),
                "--max-tokens", str(w.get("max_tokens", 8192)), "--timeout", str(w.get("request_timeout_s", 3600)),
                "--attempt", str(it["runs"])]
    if w["kind"] in ("guard", "judge", "rerank"):              # pass 4: the small models' call shapes
        j = lambda x: ",".join(str(y) for y in _aslist(x))
        argv = [mount(HERE / "benchw.py", st), w["kind"], "--url", url, "--key-env", keyenv, "--item", it["id"],
                "--out", mount(st.dir / "results.jsonl", st), "--corpus", w.get("corpus", ""),
                "--conc", j(w.get("conc", [1])), "--repeats", str(w.get("repeats", d["repeats"])),
                "--sampling", json.dumps(plan["sampling"][k.get("sampling", "engine")]),
                "--max-tokens", str(w.get("max_tokens", 512)), "--timeout", str(w.get("request_timeout_s", 600)),
                "--attempt", str(it["runs"])]
        if w.get("chat_kwargs"):
            argv += ["--chat-kwargs", json.dumps(w["chat_kwargs"])]
        if w["kind"] == "guard":
            argv += ["--sizes", j(w.get("sizes", [512]))]
        elif w["kind"] == "judge":
            argv += ["--history", j(w.get("history", [4096])), "--tool-turns", j(w.get("tool_turns", [0])),
                     "--read-tokens", str(w.get("read_tokens", 2000)), "--judge-timeout", str(w.get("judge_timeout_s", 120))]
        else:
            argv += ["--shapes", j(w.get("shapes", [])), "--duration", str(w.get("duration_s", 0))]
        return argv
    if w["kind"] == "agentloop":
        pf.write_text(json.dumps(plan["effort"].get("prefix", {})))
        shape = SHAPES[k.get("shape", "direct")]
        return [mount(HERE / "benchw.py", st), "agentloop", "--url", url, "--key-env", keyenv, "--item", it["id"],
                "--out", mount(st.dir / "results.jsonl", st), "--corpus", w["corpus"],
                "--shape", mount(HERE / shape, st) if shape else "direct",
                "--agents", streams_for({"streams": w.get("agents", "1")}, v).split(",")[-1],
                "--turns", str(w.get("turns", 12)), "--tool-tokens", str(w.get("tool_tokens", 2000)),
                "--ctx", str(v.get("ctx") or 32768), "--chars-per-token", str(w.get("chars_per_token", 3.0)),
                "--effort", k.get("effort", "default"), "--effort-key", plan["effort"]["key"],
                "--prefix-file", mount(pf, st), "--sampling", json.dumps(plan["sampling"][k.get("sampling", "engine")]),
                "--max-tokens", str(w.get("max_tokens", 8192)), "--timeout", str(w.get("request_timeout_s", 1800)),
                "--attempt", str(it["runs"])]
    pf.write_text(json.dumps(plan["effort"].get("prefix", {})))
    argv = [mount(HERE / "benchw.py", st), w["kind"], "--url", url, "--key-env", keyenv, "--item", it["id"],
            "--limit", str(w.get("limit", 0)),
            "--tasks", mount(HERE / w["tasks"], st), "--out", mount(st.dir / "results.jsonl", st),
            "--effort", k.get("effort", "default"), "--effort-key", plan["effort"]["key"],
            "--prefix-file", mount(pf, st), "--sampling", json.dumps(plan["sampling"][k.get("sampling", "engine")]),
            "--max-tokens", str(w.get("max_tokens", 16384)), "--streams", str(w.get("streams", 1)),
            "--repeats", str(w.get("repeats", d["repeats"])), "--timeout", str(w.get("request_timeout_s", 1800)),
            "--attempt", str(it["runs"])]
    if w.get("only_level"):
        argv += ["--only-level", w["only_level"]]
    if k.get("replay_reasoning"):
        argv.append("--replay-reasoning")
    if w.get("keep_text"):
        argv.append("--keep-text")
    return argv


def streams_for(w: dict, v: dict) -> str:
    """`streams` as given ("1,2,4"), or relative to the variant's slots: "slots+2" (one level), "ramp+2" (1 … slots+2),
    and lists of those ("slots+0,slots+2": decode at depth without and with queueing)."""
    out: list[str] = []
    for s in str(w.get("streams", "1,2,3,4")).split(","):
        s = s.strip()
        for key in ("slots", "ramp"):
            if s.startswith(key):
                top = int(v["slots"]) + int(s[len(key):] or 0)
                s = str(top) if key == "slots" else ",".join(str(i) for i in range(1, top + 1))
                break
        out += [x for x in s.split(",") if x and x not in out]
    return ",".join(out)


def gpu_hot(plan: dict, v: dict | None = None) -> str:
    """Hottest of the GPUs this variant uses (variant `gpus`, else plan `gpus`, else all) above the limit."""
    mine = (v or {}).get("gpus", plan.get("gpus"))
    try:
        out = subprocess.run([SVC, "gpu"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if mine is not None and parts and parts[0].isdigit() and int(parts[0]) not in [real_gpu(m) for m in mine]:
            continue
        if len(parts) >= 2 and parts[1].replace(".", "").isdigit() and float(parts[1]) > plan["guard"]["max_gpu_temp_c"]:
            return f"GPU {parts[0]} at {parts[1]} °C"
    return ""


def free_gb(path: Path) -> float:
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def run_item(plan, st, it, v, w) -> tuple[str, str]:
    """Returns (status, error). status: done | failed | hot | stopped."""
    argv = item_argv(plan, st, it, v, w)
    timeout = w.get("timeout_s", plan["defaults"]["timeout_s"])
    log = open(st.dir / "items.log", "a")
    log.write(f"\n=== {it['id']} {time.strftime('%F %T')}\n"); log.flush()
    p = subprocess.Popen([SVC, "net", str(st.dir), "--"] + argv, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    t0, last_temp = time.time(), 0.0
    while p.poll() is None:
        time.sleep(1)
        why = ""
        if _stop:
            why = "stopped"
        elif time.time() - t0 > timeout:
            why = f"timeout after {timeout}s"
        elif time.time() - last_temp > 10:
            last_temp = time.time()
            hot = gpu_hot(plan, v)
            why = f"hot: {hot}" if hot else ""
        if why:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            log.close()
            return ("stopped" if why == "stopped" else "hot" if why.startswith("hot") else "failed"), why
    log.close()
    return ("done", "") if p.returncode == 0 else ("failed", f"exit {p.returncode} (see items.log)")


# ---- ladders (pass 2, 2026-10-21): variants stepping one setting down, stopped by measured limits ----------------
# variant keys: ladder = "<name>" (member), expert_cache_frac + ladder_base (cache = frac × the base's reported expert
# slots), expert_cache_from = "<name>" (the cache of that ladder's last step that stayed within the limits).
# plan [ladder.<name>]: stop_membw_pct (RAM bandwidth, % of the calibrated peak), stop_tps_below (writing speed, median
# of the item's turns), consecutive (that many slow steps in a row). The remaining steps are skipped with the reason.
_SLOTS_RE = re.compile(r"expert cache \((\d+) experts")


def _ladders(st):
    f = st.dir / "ladders.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def _save_ladders(st, lad):
    (st.dir / "ladders.json").write_text(json.dumps(lad, indent=1))


def reported_slots(st, vid):
    try:
        m = _SLOTS_RE.search((st.dir / f"up-{_safe(vid)}.log").read_text())
        return int(m.group(1)) if m else None
    except OSError:
        return None


def resolve_cache(st, v, lad):
    """The variant with its expert cache filled in, or (v, reason it cannot run)."""
    frac, src = v.get("expert_cache_frac"), v.get("expert_cache_from")
    if frac is None and not src:
        return v, ""
    if src:
        slots = (lad.get(src) or {}).get("last_ok_slots")
        if not slots:
            return v, f"no step of ladder {src} stayed within its limits"
    else:
        base = reported_slots(st, v.get("ladder_base", ""))
        if not base:
            return v, f"{v.get('ladder_base')} did not report its expert cache"
        slots = max(64, int(base * frac))
    v = copy.deepcopy(v)
    v.setdefault("strata_args_set", {})["--expert-cache"] = str(slots)
    v["_cache_slots"] = slots
    return v, ""


def membw_pct(t0, t1):
    """p90 of (read + write) during [t0, t1] as % of the calibrated peak (run-stages: BENCH_MEMBW_PEAK)."""
    try:
        peak = json.loads(Path(os.environ["BENCH_MEMBW_PEAK"]).read_text())["peak_total_gbs"]
        by_t = {}
        for line in open(os.environ["BENCH_BOXSTATS"]):
            ts, k, val = line.strip().split(",")
            if k in ("membw.read_mbs", "membw.write_mbs") and t0 <= float(ts) <= t1:
                by_t[ts] = by_t.get(ts, 0) + float(val)
    except (KeyError, OSError, ValueError):
        return None
    vals = sorted(by_t.values())
    return round(100 * vals[int(0.9 * (len(vals) - 1))] / 1000 / peak, 1) if vals and peak else None


def ladder_eval(plan, st, it, v, lad, slots):
    name = v["ladder"]
    L = (plan.get("ladder") or {}).get(name, {})
    s = lad.setdefault(name, {"slow": 0, "steps": []})
    rows = []
    try:
        for line in (st.dir / "results.jsonl").read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("item") == it["id"] and r.get("attempt", 1) == it.get("runs", 1):
                rows.append(r)
    except OSError:
        pass
    tps = [r.get("decode_tps") or r.get("engine_decode_tps") for r in rows if r.get("decode_tps") or r.get("engine_decode_tps")]
    med = round(statistics.median(tps), 1) if tps else 0.0
    bw = membw_pct(it["started"], it["finished"])
    step = {"variant": v["id"], "slots": slots, "tps_median": med, "membw_pct": bw}
    s["steps"].append(step)
    why = ""
    gain = L.get("stop_gain_below_pct")                  # pass 4: stop once adding no longer helps (user 2026-10-22)
    if gain is not None and med > 0:                     # (no writing measured: not a step of the ladder)
        best = s.get("best_tps", 0.0)
        if med > best * (1 + gain / 100):
            s["flat"] = 0
        else:
            s["flat"] = s.get("flat", 0) + 1
        s["best_tps"] = max(best, med)
        step["best_tps"] = s["best_tps"]
        if s["flat"] >= L.get("consecutive", 2):
            why = (f"writing gained less than {gain} % on the best so far ({s['best_tps']} tok/s) on {s['flat']} "
                   f"steps in a row (last {med})")
    if why:
        pass
    elif bw is not None and bw >= L.get("stop_membw_pct", 101):
        why = f"RAM bandwidth {bw} % >= {L['stop_membw_pct']} % at {slots} expert slots"
    elif med < L.get("stop_tps_below", 0):
        s["slow"] += 1
        if s["slow"] >= L.get("consecutive", 2):
            why = f"writing speed below {L['stop_tps_below']} tok/s on {s['slow']} steps in a row (last {med})"
    else:
        s["slow"] = 0
        if slots:
            s["last_ok_slots"] = slots
    if why:
        s["stopped"] = why
    _save_ladders(st, lad)
    return step, why


FAIL_PAT = re.compile(r"needs |error|out of memory|not enough|cannot|invalid|unsupported|exceeds|traceback", re.I)


def start_failure(log: str) -> str:
    """The engine's own reason from a failed start's log: the first line that names a problem (not the status line)."""
    for line in log.splitlines():
        body = line.split("|", 1)[-1].strip()
        if FAIL_PAT.search(body) and not body.startswith(("Container ", "ready after")):
            return body[:240]
    first = log.strip().splitlines()[0] if log.strip() else ""
    return first[:240] or "no engine output"


def after_logs(st: State, v: dict) -> None:
    """Append the engine's log lines from under load to up-<variant>.log (spec acceptance, preemption, errors); the raw
    tail goes to log-<variant>.txt (line formats for the board's log patterns)."""
    r = subprocess.run([SVC, "logs", v["service"], str(st.dir / f"log-{_safe(v['id'])}.txt")], capture_output=True, text=True)
    if r.stdout.strip():
        with open(st.dir / f"up-{_safe(v['id'])}.log", "a") as f:
            f.write("\n--- after the workloads ---\n" + r.stdout)


def cmd_run(plan: dict, args) -> int:
    global _stop
    st = State(plan)
    lock = open(st.dir / "lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another benchq run holds the lock for this plan"); return 2
    for it in st.items:
        if args.retry_failed and it["status"] in ("failed", "skipped"):
            it.update(status="pending", attempts=0)
    skip = [g for g in (getattr(args, "skip", "") or "").split(",") if g.strip()]
    for it in st.items:                             # bench_skip (user 2026-10-21): left out of this run on purpose
        if it["status"] == "pending" and any(fnmatch.fnmatch(it["id"], g.strip()) for g in skip):
            it.update(status="skipped", last_error=f"skipped by user (bench_skip {','.join(skip)})")
    st.save()

    def on_signal(*_):
        global _stop
        _stop = True
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)

    variants = {v["id"]: v for v in plan["variant"]}
    workloads = {w["id"]: w for w in plan.get("workload", [])}
    todo = [i for i in st.items if i["status"] == "pending"
            and (not args.only or fnmatch.fnmatch(i["id"], args.only))]
    event(st, f"run {plan['name']}: {len(todo)} items pending of {len(st.items)}", "up")
    touched: set[str] = set()
    stopped: list[str] = []                         # other services stopped for the current variant, in order
    current_variant, variant_ok, skip_reason = None, False, "variant did not start"
    cache_of: dict = {}                             # variant → resolved expert-cache slots (ladders)
    power_set = False                               # a variant's gpu_power_w is active (restored when it ends)
    started_ok = {i["variant"] for i in st.items if i["status"] == "done"}     # for only_if_failed (resumable)
    tried = started_ok | {i["variant"] for i in st.items if str(i.get("last_error", "")).startswith("did not start")}
    rc = 0
    try:
        for it in todo:
            wait_while_paused(st, plan, it["id"])
            if _stop:
                break
            if free_gb(st.dir) < plan["guard"]["min_free_gb"]:
                event(st, f"disk below {plan['guard']['min_free_gb']} GB free — stopping", "down"); rc = 1; break
            v, w = variants[it["variant"]], workloads[it["workload"]]
            if it["variant"] != current_variant:
                prev = variants.get(current_variant) if current_variant else None
                if power_set:                             # the previous variant's power limit back to the declared one
                    subprocess.run([SVC, "power", "restore"]); power_set = False
                    event(st, "GPU power limit back to the declared value")
                if prev and prev["id"] in started_ok and prev["service"] in touched:
                    after_logs(st, prev)          # what the engine logged under load (acceptance, preemption, errors)
                want_stopped = list(v.get("stop") or [])
                back = [s for s in stopped if s not in want_stopped]
                # the previous variant's own service goes back first when it moves away or when a stopped service
                # returns (it may hold that service's GPU: Strata on 3 cards, vLLM over #1+#2)
                if prev and prev["service"] in touched and (prev["service"] != v["service"] or back):
                    subprocess.run([SVC, "restore", prev["service"]]); touched.discard(prev["service"])
                for s in reversed(back):                  # reverse stop order: dependencies first
                    event(st, f"{s}: back (stopped for {prev['id'] if prev else '?'})")
                    subprocess.run([SVC, "restore", s]); stopped.remove(s)
                for s in want_stopped:
                    if s not in stopped:
                        if s in touched:
                            touched.discard(s)
                        event(st, f"{s}: stopped for {v['id']}")
                        subprocess.run([SVC, "stop", s]); stopped.append(s)
                current_variant = it["variant"]
                dep = v.get("only_if_failed")
                if dep and (dep in started_ok or dep not in tried):
                    why = f"not needed: {dep} started" if dep in started_ok else f"not needed: {dep} was not tried"
                    event(st, f"variant {v['id']}: skipped — {why}")
                    variant_ok, skip_reason = False, why
                    it.update(status="skipped", last_error=why); st.save(); continue
                missing = model_missing(v)
                if missing:
                    event(st, f"variant {v['id']}: skipped — {missing}", None)
                    variant_ok, skip_reason = False, missing
                    it.update(status="skipped", last_error=missing); st.save(); continue
                lad = _ladders(st)
                ln = v.get("ladder")
                if ln and (lad.get(ln) or {}).get("stopped"):
                    why = f"ladder {ln} stopped: {lad[ln]['stopped']}"
                    event(st, f"variant {v['id']}: skipped — {why}")
                    variant_ok, skip_reason = False, why
                    it.update(status="skipped", last_error=why); st.save(); continue
                v, why = resolve_cache(st, v, lad)
                if why:
                    event(st, f"variant {v['id']}: skipped — {why}")
                    variant_ok, skip_reason = False, why
                    it.update(status="skipped", last_error=why); st.save(); continue
                cache_of[v["id"]] = v.get("_cache_slots")
                if v.get("gpu_power_w"):                  # pass 3 Q6: one card at another limit for this variant
                    g = str(real_gpu((v.get("gpus") or [0])[0]))   # the role → the live index (.env)
                    r_ = subprocess.run([SVC, "power", g, str(v["gpu_power_w"])], capture_output=True, text=True)
                    power_set = True
                    event(st, f"GPU {g} power limit {v['gpu_power_w']} W for {v['id']}"
                              + ("" if r_.returncode == 0 else f" — FAILED: {(r_.stdout + r_.stderr).strip()[:120]}"))
                skip_reason = "variant did not start"
                ov, h = build_override(st, v)
                if ov is None and plan["defaults"]["recreate"] == "if-changed" and v["service"] not in touched:
                    variant_ok = True
                else:
                    event(st, f"variant {v['id']} ({v['service']}, {h}): starting")
                    r = subprocess.run([SVC, "up", v["service"], str(ov) if ov else "-",
                                        str(v.get("ready_timeout_s", plan["defaults"]["ready_timeout_s"]))],
                                       capture_output=True, text=True)
                    touched.add(v["service"])
                    tried.add(v["id"])
                    variant_ok = r.returncode == 0
                    (st.dir / f"up-{_safe(v['id'])}.log").write_text(r.stdout + r.stderr)
                    if variant_ok:
                        started_ok.add(v["id"])
                    else:
                        skip_reason = "did not start: " + start_failure(r.stdout + r.stderr)
                    event(st, f"variant {v['id']}: {'ready' if variant_ok else 'FAILED to start — ' + skip_reason[15:]}",
                          None if variant_ok else "down")
            if not variant_ok:
                it.update(status="skipped", last_error=skip_reason); st.save(); continue
            it.update(status="running", started=int(time.time()), runs=it.get("runs", 0) + 1)
            it["attempts"] += 1; st.save()
            write_status(st, plan, it["id"])
            status, err = run_item(plan, st, it, v, w)
            dur = int(time.time()) - it["started"]
            if status == "done":
                it.update(status="done", finished=int(time.time()), duration_s=dur, last_error="")
                event(st, f"done {it['id']} ({dur}s)", "up")
                if v.get("ladder") and w.get("ladder_eval"):
                    slots = cache_of.get(v["id"]) or reported_slots(st, v["id"])
                    step, why = ladder_eval(plan, st, it, v, _ladders(st), slots)
                    it["ladder"] = step
                    event(st, f"ladder {v['ladder']}: {v['id']} — {slots} expert slots, writing {step['tps_median']} "
                              f"tok/s, RAM bandwidth {step['membw_pct']} %" + (f" → STOP: {why}" if why else ""))
            elif status == "stopped":
                it.update(status="pending"); it["attempts"] -= 1
            else:
                if status == "hot":
                    event(st, f"{it['id']}: {err} — cooling {plan['guard']['cooldown_s']}s", "down")
                    time.sleep(plan["guard"]["cooldown_s"])
                retry = it["attempts"] < plan["defaults"]["max_attempts"]
                it.update(status="pending" if retry else "failed", last_error=err)
                event(st, f"{'retry later' if retry else 'FAILED'} {it['id']}: {err}", None if retry else "down")
                if retry:
                    todo.append(it)            # re-queued at the end; its variant restarts if needed then
            st.save(); write_status(st, plan)
    finally:
        if power_set:
            subprocess.run([SVC, "power", "restore"])
            event(st, "GPU power limit back to the declared value")
        if current_variant in started_ok and variants[current_variant]["service"] in touched:
            after_logs(st, variants[current_variant])
        on_end = plan["defaults"].get("on_end", "restore")
        for svc in sorted(touched):                 # own services first (they may hold a stopped one's GPU) …
            # on_end = "stop": leave the slot empty (llama.cpp in Strata's slot while a parallel plan still holds
            # the guard's GPU — Strata would size itself against it); ops/bench.yml restores guard, then Strata
            subprocess.run([SVC, "stop" if on_end == "stop" else "restore", svc])
        for svc in reversed(stopped):               # … then what was stopped, dependencies first
            subprocess.run([SVC, "restore", svc])
        write_status(st, plan)
        c = st.counts()
        msg = f"run {plan['name']} {'stopped' if _stop else 'finished'}: {c}"
        event(st, msg, "down" if c.get("failed") else "up")
    if not _stop and not rc:
        subprocess.run([sys.executable, str(HERE / "benchreport.py"), str(st.dir)], stdout=subprocess.DEVNULL)
    return 130 if _stop else (1 if st.counts().get("failed") else rc)


# ================================================================ cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "status", "report", "reset", "pause", "resume"])
    ap.add_argument("plan", nargs="?")
    ap.add_argument("glob", nargs="?")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--skip", default="", help="run: mark pending items matching these globs (comma-separated) skipped")
    ap.add_argument("--only", default=None, help="run only items whose id matches this glob")
    ap.add_argument("--smoke", action="store_true", help="cheap run of the same sequence (see the module doc); always fresh")
    ap.add_argument("--fresh", action="store_true", help="run: forget earlier progress and results of this plan first")
    a = ap.parse_args(argv)
    if a.cmd in ("pause", "resume"):
        f = pause_file()
        if a.cmd == "pause":
            f.parent.mkdir(parents=True, exist_ok=True); f.write_text(time.strftime("%F %T\n"))
            print(f"paused: the running item finishes, no new one starts ({f})")
        else:
            f.unlink(missing_ok=True); print("resumed")
        return 0
    if not a.plan:
        ap.error("this command needs a plan")
    plan = load_plan(a.plan)
    if a.smoke:
        plan = smoke(plan)
    errs = validate(plan)
    if errs:
        print("plan errors:\n  " + "\n  ".join(errs)); return 2
    if a.cmd == "plan":
        items = expand(plan)
        for it in items:
            print(it["id"])
        nv = len({i["variant"] for i in items})
        print(f"\n{len(items)} items, {nv} variant restarts (plus restores)")
        return 0
    if a.cmd == "run":
        if a.smoke or a.fresh:                  # a smoke run always starts over; a real one only when asked
            d = Path(plan.get("state_dir") or (INFRA / "scripts" / "logs" / "bench" / plan["name"]))
            if not d.is_absolute():
                d = (Path(plan["_path"]).parent / d).resolve()
            for f in ("state.json", "results.jsonl", "throughput.csv", "status.json", "report.md"):
                (d / f).unlink(missing_ok=True)
        return cmd_run(plan, a)
    st = State(plan)
    if a.cmd == "status":
        c = st.counts()
        print(f"{plan['name']}: {c}  (state: {st.dir})")
        s = st.dir / "status.json"
        if s.exists():
            j = json.loads(s.read_text())
            print(f"current: {j.get('current') or '-'}   updated {time.strftime('%F %T', time.localtime(j['updated']))}")
        for it in st.items:
            if it["status"] in ("failed", "skipped"):
                print(f"  {it['status']:<8} {it['id']}: {it.get('last_error', '')}")
        return 0
    if a.cmd == "reset":
        if not a.glob:
            print("reset needs a glob"); return 2
        n = 0
        for it in st.items:
            if fnmatch.fnmatch(it["id"], a.glob) and it["status"] != "running":
                it.update(status="pending", attempts=0); n += 1
        st.save(); print(f"{n} items reset to pending"); return 0
    return subprocess.run([sys.executable, str(HERE / "benchreport.py"), str(st.dir)]).returncode


if __name__ == "__main__":
    sys.exit(main())
