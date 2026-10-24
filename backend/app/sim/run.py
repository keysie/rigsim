"""Run the Turnstone-shaped workload over the configured machine, engines and roles."""
from __future__ import annotations

import random
import statistics as st

from ..machine import validate
from ..perf import derive
from ..schemas import SimRequest
from .core import Sim
from .engines import BatchEngine, StrataEngine
from .workload import Counter, make_scripts, sample_scenario, workstream

ROLES = ("coordinator", "maker", "sub", "guard", "judge", "reranker")


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def build_engines(req: SimRequest, info):
    perfs, errors = {}, []
    for spec in req.engines:
        pf = derive(spec, info)
        perfs[spec.id] = pf
        for e in pf.errors:
            errors.append(f"{spec.name or spec.id}: {e}")
    return perfs, errors


def make_sim_engines(perfs, replay):
    out = {}
    for eid, pf in perfs.items():
        if pf.errors:
            continue
        out[eid] = StrataEngine(pf, replay_thinking=replay) if pf.kind == "strata" else BatchEngine(pf)
    return out


def run_once(req, perfs, scenario, seed, n_ws, trace=None):
    sim_engines = make_sim_engines(perfs, req.workload.replay_thinking)
    roles = {}
    for r in ROLES:
        eid = getattr(req.roles, r)
        if eid and eid in sim_engines:
            roles[r] = sim_engines[eid]
    if "maker" not in roles:
        raise ValueError("the maker role needs an engine")
    if "sub" not in roles:
        roles["sub"] = roles["maker"]
    engines = {r: [e] for r, e in roles.items()}
    s = dict(scenario)
    s["_has_guard"] = "guard" in roles
    s["_has_judge"] = "judge" in roles
    specs = {e.id: e for e in req.engines}
    jspec = specs.get(req.roles.judge or "")
    s["guard_out"] = 60
    # non-reasoning judges answer in a short JSON verdict; reasoning models think first
    s["judge_out"] = 40 if (jspec and any(k in jspec.model_id for k in ("granite", "ministral", "phi"))) else 150
    sim = Sim(engines, t_max=48 * 3600.0, trace=trace)
    ids = Counter()
    log, tasks = [], []
    scripts = make_scripts(seed, s, n_ws)
    for w in range(n_ws):
        sim.spawn(workstream(s, ids, w, scripts[w], log, tasks, lambda: sim.now, {k: True for k in roles}))
    sim.run()
    T = sim.now or 1e-9
    durs = [t["end"] - t["start"] for t in tasks]
    m = {"tasks": len(tasks), "makespan_s": T, "tasks_per_h": len(tasks) / T * 3600,
         "task_med_s": pct(durs, .5), "task_p90_s": pct(durs, .9), "stopped": sim.stopped, "task_log": tasks}
    m["attempts"] = sum(t.get("attempts", 1) for t in tasks)
    eng_stats = {}
    for eid, e in sim_engines.items():
        js = [j for j in sim.calls if j.engine == eid and j.t_done is not None]
        waits = [(j.t_admit if j.t_admit is not None else j.t_done) - j.t_submit for j in js]
        tin = sum(j.prefill_tokens for j in js)
        tout = sum(j.req.out for j in js)
        pf = e.perf
        watts = pf.watts_idle + (pf.watts_busy - pf.watts_idle) * (e.busy_time / T)      # average draw of its parts
        es = {"util": e.busy_time / T, "mean_active": e.work_time / T, "calls": len(js),
              "wait_med": pct(waits, .5), "wait_p90": pct(waits, .9),
              "reread_tokens": getattr(e, "reread_tokens", 0), "prefill_tokens": getattr(e, "prefill_total", 0),
              "skipped": sum(1 for j in sim.calls if j.engine == eid and j.skipped),
              "tokens_in": tin, "tokens_out": tout, "tps_in": tin / T, "tps_out": tout / T, "tps": (tin + tout) / T,
              "watts": 0.0 if pf.kind == "hosted" else watts, "watts_busy": pf.watts_busy}
        if isinstance(e, BatchEngine) and e.hosted:
            pr = e.perf.hosted_prices
            es["cost_usd"] = (e.tokens_in * pr["price_in"] + e.tokens_cached * pr["price_cached"]
                              + e.tokens_out * pr["price_out"]) / 1e6
        eng_stats[eid] = es
    kinds = {}
    for k in ("maker", "sub", "guard", "judge", "rerank", "coordinator", "compact"):
        js = [j for j in sim.calls if j.req.kind == k and j.t_done is not None]
        if js:
            kinds[k] = {"calls": len(js), "per_min": len(js) / T * 60,
                        "prompt_p50": pct([j.req.context for j in js], .5), "prompt_p90": pct([j.req.context for j in js], .9),
                        "wait_p90": pct([(j.t_admit if j.t_admit is not None else j.t_done) - j.t_submit for j in js], .9),
                        "dur_p90": pct([j.t_done - j.t_submit for j in js], .9)}
    return m, eng_stats, kinds


def simulate(req: SimRequest) -> dict:
    info = validate(req.machine)
    out = {"machine": {k: v for k, v in info.items() if not k.startswith("_")}, "engines": {}, "errors": [], "results": []}
    perfs, errors = build_engines(req, info)
    out["engines"] = {eid: pf.public() for eid, pf in perfs.items()}
    out["errors"] = info["errors"] + errors
    # resource sharing warnings
    gpu_users = {}
    for eid, pf in perfs.items():
        for g in pf.resources.get("gpus", []):
            gpu_users.setdefault(g, []).append(eid)
    shared = {g: u for g, u in gpu_users.items() if len(u) > 1}
    warnings = list(info["warnings"])
    names = {e.id: (e.name or e.id) for e in req.engines}
    for g, u in shared.items():
        used = sum(float(perfs[e].memory.get(g, {}).get("used_gb", 0)) for e in u)
        vram = float(info["_devices"][g]["spec"]["vram_gb"])
        if used > vram:
            out["errors"].append(f"{g}: {', '.join(names[e] for e in u)} need {used:.1f} GB together, the card has {vram:.0f} GB")
        else:
            warnings.append(f"{g} is shared by {', '.join(names[e] for e in u)} ({used:.1f} of {vram:.0f} GB): "
                            "their kernels compete; the simulator sums the utilisation but does not slow either engine")
    if out["errors"]:
        out["warnings"] = warnings
        return out
    rng = random.Random(req.seed)
    scenarios = [sample_scenario(rng, req.workload) for _ in range(max(1, req.scenarios))]
    for n in req.concurrency:
        runs = []
        for si, sc in enumerate(scenarios):
            try:
                runs.append(run_once(req, perfs, sc, req.seed * 1000 + si, n))
            except ValueError as e:
                out["errors"].append(str(e))
                out["warnings"] = warnings
                return out
        ms = [r[0] for r in runs]
        res = {"concurrency": n,
               "tasks_per_h": {"p10": pct([m["tasks_per_h"] for m in ms], .1), "p50": pct([m["tasks_per_h"] for m in ms], .5),
                               "p90": pct([m["tasks_per_h"] for m in ms], .9)},
               "task_p90_min": pct([m["task_p90_s"] / 60 for m in ms], .5),
               "attempts_per_task": st.median(m["attempts"] / max(1, m["tasks"]) for m in ms),
               "task_med_min": pct([m["task_med_s"] / 60 for m in ms], .5),
               "engines": {}, "kinds": {}, "gpus": {}, "bottlenecks": []}
        for eid in perfs:
            vals = [r[1][eid] for r in runs if eid in r[1]]
            if not vals:
                continue
            res["engines"][eid] = {k: st.median(v[k] for v in vals) for k in vals[0]}
        for k in ("maker", "sub", "guard", "judge", "rerank", "coordinator", "compact"):
            vals = [r[2][k] for r in runs if k in r[2]]
            if vals:
                res["kinds"][k] = {kk: st.median(v[kk] for v in vals) for kk in vals[0]}
        for g, users in gpu_users.items():
            res["gpus"][g] = min(1.0, sum(res["engines"].get(u, {}).get("util", 0) for u in users))
        res["machine"] = _machine_totals(req, info, perfs, res)
        res["bottlenecks"] = _bottlenecks(req, perfs, res, shared)
        out["results"].append(res)
    out["warnings"] = warnings
    return out


def trace(req: SimRequest, concurrency: int, scenario_index: int = 0) -> dict:
    """One scenario at one concurrency, with every submit/admit/done event, for live replay."""
    info = validate(req.machine)
    perfs, errors = build_engines(req, info)
    errors = info["errors"] + errors
    if errors:
        return {"errors": errors, "events": []}
    rng = random.Random(req.seed)
    scenarios = [sample_scenario(rng, req.workload) for _ in range(max(1, req.scenarios))]
    sc = scenarios[min(scenario_index, len(scenarios) - 1)]
    events = []
    m, eng_stats, kinds = run_once(req, perfs, sc, req.seed * 1000 + scenario_index, concurrency, trace=events.append)
    resources = {}
    for eid, pf in perfs.items():
        resources[eid] = {"gpus": pf.resources.get("gpus", []), "cpu": pf.resources.get("cpu"), "cores": pf.resources.get("cores", 0),
                          "limits": pf.limits, "slots": pf.slots, "kind": pf.kind, "hosted": pf.kind == "hosted"}
    return {"errors": [], "concurrency": concurrency, "scenario": {k: v for k, v in sc.items() if not k.startswith("_")},
            "makespan_s": m["makespan_s"], "tasks": m["task_log"], "events": events, "resources": resources,
            "engines": eng_stats, "gpus": [d.id for d in req.machine.devices if d.kind == "gpu"],
            "cpus": {str(i): {"name": c["name"], "cores": c["cores"], "ram_bw_gbs": c["ram_bw_gbs"]} for i, c in info["cpus"].items()}}


def _machine_totals(req, info, perfs, res) -> dict:
    """The headline numbers: tokens per second served by the local engines (prompt + output), watts the
    machine draws on average (GPUs at idle or cap by busy share, CPU by its busy cores, 40 W for the rest),
    tokens per second per watt, and one utilisation figure: the busy share of every GPU and of the CPU
    pool, averaged over the parts (an idle 3060 counts as much as an idle 3090 — it is a parked asset)."""
    local = {eid: e for eid, e in res["engines"].items() if perfs[eid].kind != "hosted"}
    tps_in = sum(e["tps_in"] for e in local.values())
    tps_out = sum(e["tps_out"] for e in local.values())
    devices = info["_devices"]
    gpu_watts, parts = 0.0, []
    for gid, d in devices.items():
        if d["kind"] != "gpu":
            continue
        busy = res["gpus"].get(gid, 0.0)
        cap = d.get("power_w") or d["spec"]["tdp_w"]
        gpu_watts += d.get("idle_w", 0.07 * d["spec"]["tdp_w"]) + (cap - d.get("idle_w", 0.07 * d["spec"]["tdp_w"])) * busy
        parts.append(busy)
    cpu_busy = 0.0
    for eid, e in local.items():
        cores = perfs[eid].resources.get("cores", 0) or 0
        if cores and not perfs[eid].resources.get("gpus"):
            cpu_busy += e["util"] * cores
    cpu_cores = sum(c["cores"] for c in info["cpus"].values()) or 1
    cpu_share = min(1.0, cpu_busy / cpu_cores)
    cpu_tdp = sum(c.get("tdp_w", 0) for c in info["cpus"].values())
    cpu_watts = cpu_tdp * (0.15 + 0.65 * cpu_share)
    parts.append(cpu_share)
    watts = gpu_watts + cpu_watts + 40.0
    return {"tps_in": tps_in, "tps_out": tps_out, "tps": tps_in + tps_out, "watts": watts,
            "tps_per_w": (tps_in + tps_out) / watts if watts else 0.0, "out_tps_per_w": tps_out / watts if watts else 0.0,
            "utilisation": sum(parts) / len(parts) if parts else 0.0, "gpu_busy": {g: res["gpus"].get(g, 0.0) for g in res["gpus"]},
            "cpu_busy": cpu_share, "attempts_per_task": res.get("attempts_per_task", 1.0)}


def _bottlenecks(req, perfs, res, shared):
    """Rank what limits throughput at this concurrency, in plain words."""
    items = []
    roles = {r: getattr(req.roles, r) for r in ROLES}
    name = {s.id: (s.name or s.id) for s in req.engines}
    for eid, es in res["engines"].items():
        pf = perfs[eid]
        role_names = [r for r, e in roles.items() if e == eid]
        label = f"{name[eid]} ({', '.join(role_names)})" if role_names else name[eid]
        u = es["util"]
        if u >= 0.75:
            lim = max(pf.limits.items(), key=lambda kv: kv[1])[0] if pf.limits else "engine"
            items.append((u, f"{label} is busy {u:.0%} of the time; inside it, {lim.replace('_', ' ')} takes the "
                             f"largest share of a step. Requests wait {es['wait_p90']:.0f} s (p90)."))
        elif es["wait_p90"] > 10:
            items.append((0.5 + min(0.4, es["wait_p90"] / 200),
                          f"{label}: requests queue {es['wait_p90']:.0f} s (p90) although it is only {u:.0%} busy — "
                          "admissions are serial or slots are too few."))
        if es.get("skipped", 0) > 0:
            items.append((0.6, f"{label} refused {es['skipped']:.0f} requests over its {pf.window} window; those ran unguarded or unjudged."))
        if es.get("reread_tokens", 0) > 0.5 * max(1, es.get("prefill_tokens", 1)):
            items.append((0.45, f"{label} re-reads {es['reread_tokens'] / max(1, es['prefill_tokens']):.0%} of its prompt tokens: "
                                "history is not kept (no parking, slot reuse broken without thinking replay, or too few slots)."))
    mk = roles.get("maker")
    if mk and mk in res["engines"]:
        u = res["engines"][mk]["util"]
        if u < 0.35:
            items.append((0.4, f"The maker engine idles {1 - u:.0%} of the time: it waits for sub-agents, the judge and tools "
                               "(fork-join). More maker capacity will not help; faster sub-makers or fewer approvals will."))
    for g, users in shared.items():
        if res["gpus"].get(g, 0) > 0.9:
            items.append((0.7, f"{g} runs {', '.join(name[u] for u in users)} together and is saturated: "
                               "in reality both slow down."))
    items.sort(key=lambda t: -t[0])
    return [t[1] for t in items[:6]]
