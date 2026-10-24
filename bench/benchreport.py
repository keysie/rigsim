#!/usr/bin/env python3
"""Summarise a bench state dir (results.jsonl, throughput.csv, state.json) as markdown. Stdlib only.

  benchreport.py STATE_DIR [--by level|shape|task]     -> stdout and STATE_DIR/report.md

Tasks: per item (variant / workload / knobs) and group: requests, errors, pass rate of checked answers,
truncated (finish=length), reasoning tokens median / p90, think time and total time medians, decode tok/s.
Multiturn: turns reached, context high-water (max prompt tokens), first vs later TTFT, cached tokens, final pass.
Agent loops (pass 2): parallel agents, context growth per turn, cache reuse, TTFT/decode later turns, tool-call share.
Throughput: aggregate tok/s per stream count, one row per item.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stx
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path


def q(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * p
    lo = int(k)
    return vals[lo] if lo + 1 >= len(vals) else vals[lo] + (vals[lo + 1] - vals[lo]) * (k - lo)


def fmt(x, nd=0):
    if x is None:
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tasks_section(rows, by):
    g = OrderedDict()
    for r in rows:
        g.setdefault((r["item"], r.get(by, "") if by else ""), []).append(r)
    out = []
    for (item, sub), rs in g.items():
        ok = [r for r in rs if r.get("ok")]
        checked = [r for r in ok if r.get("passed") is not None]
        tps = [r["completion_tokens"] / (r["total_s"] - (r["ttft_s"] or 0))
               for r in ok if r.get("completion_tokens") and r.get("total_s") and r["total_s"] > (r["ttft_s"] or 0)]
        rt = [r.get("reasoning_tokens") for r in ok]
        src = {r.get("reasoning_src") for r in ok}
        out.append([item + (f" · {sub}" if by else ""), len(rs), len(rs) - len(ok),
                    f"{sum(1 for r in checked if r['passed'])}/{len(checked)}" if checked else "–",
                    sum(1 for r in ok if r.get("finish") == "length"),
                    fmt(q(rt, .5)) + ("*" if src == {"chunks"} else ""), fmt(q(rt, .9)),
                    fmt(q([r.get("think_s") for r in ok], .5), 1), fmt(q([r.get("total_s") for r in ok], .5), 1),
                    fmt(q(tps, .5), 1)])
    return table(["item" + (f" · {by}" if by else ""), "req", "err", "passed", "trunc", "think tok p50",
                  "p90", "think s p50", "total s p50", "tok/s p50"], out) + \
        "\n\n`*` reasoning tokens counted from stream chunks (the engine reported no reasoning_tokens)."


def multiturn_section(rows):
    runs = defaultdict(list)
    for r in rows:
        runs[(r["item"], r["task"], r["rep"])].append(r)
    per_item = OrderedDict()
    for (item, _t, _rep), rs in runs.items():
        rs.sort(key=lambda r: r["turn"])
        per_item.setdefault(item, []).append(rs)
    out = []
    for item, convs in per_item.items():
        turns = [len(c) for c in convs]
        hw = [max((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in c) for c in convs]
        first = [c[0].get("ttft_s") for c in convs]
        later = [r.get("ttft_s") for c in convs for r in c[1:]]
        cached = sum(r.get("cached_tokens") or 0 for c in convs for r in c)
        trunc = sum(1 for c in convs if c[-1].get("finish") == "length")
        final = [c[-1].get("passed") for c in convs if c[-1].get("passed") is not None]
        out.append([item, len(convs), fmt(q(turns, .5), 1), fmt(max(hw) if hw else None), trunc,
                    fmt(q(first, .5), 2), fmt(q(later, .5), 2), cached or "–",
                    f"{sum(final)}/{len(final)}" if final else "–"])
    return table(["item", "convs", "turns p50", "context high-water", "trunc", "TTFT 1st", "TTFT later",
                  "cached tok", "final passed"], out)


def sustained_section(path):
    """Sustained load (label …/t=<minutes>): rounds, aggregate and steady decode tok/s in the first vs the last three
    rounds, change — throttling or a heat-soaked card shows as a drop (box stats: temperatures and clocks)."""
    runs = OrderedDict()
    for r in csv.DictReader(open(path)):
        if "/t=" in r["label"]:
            runs.setdefault((r["label"].rsplit("/t=", 1)[0], r["streams"]), []).append(r)
    out = []
    for (item, streams), rs in runs.items():
        rs.sort(key=lambda r: float(r["label"].rsplit("/t=", 1)[1]))
        f = lambda rr, k: q([float(x[k]) for x in rr if x.get(k) not in (None, "")], .5)   # noqa: E731
        a0, a1 = f(rs[:3], "aggregate_tps"), f(rs[-3:], "aggregate_tps")
        out.append([item, streams, len(rs), rs[-1]["label"].rsplit("/t=", 1)[1] + " min", fmt(a0, 1), fmt(a1, 1),
                    f"{(a1 - a0) / a0:+.0%}" if a0 and a1 is not None else "–",
                    fmt(f(rs[:3], "decode_tps_steady"), 1), fmt(f(rs[-3:], "decode_tps_steady"), 1),
                    sum(int(x["errors"]) for x in rs)])
    return table(["item", "clients", "rounds", "last round at", "aggregate first", "aggregate last", "change",
                  "steady first", "steady last", "errors"], out) if out else ""


def longtask_section(rows, d):
    """Per item: conversations, turns, context first/last turn, writing speed (median, slowest turn), TTFT of the first
    turn (the long prefix), Strata's expert hit and MTP acceptance; the ladder's verdict if the item had one."""
    lad = {}
    try:
        for name, s in json.loads((d / "ladders.json").read_text()).items():
            for st in s.get("steps", []):
                lad[st["variant"]] = (name, st, s.get("stopped", ""))
    except (OSError, ValueError):
        pass
    per = OrderedDict()
    for r in rows:
        per.setdefault(r["item"], []).append(r)
    dur = {}                                           # an item's wall time (older runs: no per-row timestamps)
    try:
        dur = {i["id"]: i.get("duration_s") for i in json.loads((d / "state.json").read_text())["items"]}
    except (OSError, ValueError, KeyError):
        pass
    out = []
    for item, rs in per.items():
        convs = {r["conv"] for r in rs}
        # query to completion (user 2026-10-22): tokens written / (first request sent -> last answer done), with every
        # prompt read, queue wait and scheduling pause inside; from the rows' timestamps, else the item's duration
        ts = [r for r in rs if r.get("t_start") and r.get("t_end")]
        span = (max(r["t_end"] for r in ts) - min(r["t_start"] for r in ts)) if ts else (dur.get(item) or 0)
        written = sum(r.get("completion_tokens") or 0 for r in rs)
        read = sum((r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0) for r in rs)
        first = [r for r in rs if r["turn"] == 0]
        tps = [r.get("decode_tps") for r in rs if r.get("decode_tps")]
        hit = [r.get("expert_hit") for r in rs if r.get("expert_hit") is not None]
        do = sum(r.get("drafts_offered") or 0 for r in rs); da = sum(r.get("drafts_accepted") or 0 for r in rs)
        last = max(((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in rs), default=0)
        v = item.split("/")[0]
        lv = lad.get(v)
        out.append([item.split("/")[0] + "/" + item.split("/")[1], len(convs), len(rs) // max(len(convs), 1),
                    fmt(q([r.get("prompt_tokens") for r in first], .5), 0), f"{last:,}",
                    fmt(q(tps, .5), 1), fmt(min(tps) if tps else None, 1), fmt(q([r.get("ttft_s") for r in first], .5), 1),
                    fmt(sum(hit) / len(hit), 3) if hit else "–", fmt(da / do, 2) if do else "–",
                    fmt(written / span, 1) if span else "–", f"{read:,}",
                    (f"{lv[1].get('membw_pct')} % RAM bw" + (" · STOP" if lv[2] and lv[1] is lv[1] else ""))
                    if lv else "–"])
    out_t = table(["item", "convs", "turns", "prompt 1st", "context max", "writing med", "writing min", "TTFT 1st s",
                   "expert hit", "drafts acc", "written / wall", "prompt read", "ladder"], out)
    out_t += ("\n\n*writing med / min*: per answer, tokens ÷ (last token − first token), wall clock — every pause inside an "
              "answer counts. *written / wall*: all answers' tokens ÷ (first request sent → last answer done), prompt reads "
              "and waits included — the figure to compare across stream counts. *prompt read*: tokens not reused.")
    # pass 3: history reuse on follow-up turns (reused / prompt), their TTFT, and whether the thinking was sent back
    reuse = []
    for item, rs in per.items():
        fu = [r for r in rs if r["turn"] > 0 and r.get("prompt_tokens")]
        if not fu:
            continue
        sh = [(r.get("cached_tokens") or 0) / r["prompt_tokens"] for r in fu]
        full = sum(1 for x in sh if x >= 0.9)
        reuse.append([item.split("/")[0] + "/" + item.split("/")[1], "yes" if rs[0].get("think_back") else "no",
                      len(fu), fmt(q(sh, .5) * 100, 0) + " %", f"{full} of {len(fu)}",
                      fmt(q([r.get("ttft_s") for r in fu], .5), 1), fmt(max(r.get("ttft_s") or 0 for r in fu), 1)])
    if reuse:
        out_t += "\n\n**History reuse on follow-up turns** (reused / prompt; \"≥ 90 %\" = the slot or the parking cache had " \
                 "the conversation):\n\n" + table(["item", "thinking back", "follow-ups", "reused med", "≥ 90 %",
                                                    "TTFT med s", "TTFT max s"], reuse)
    tr = trace_section(d)
    if tr:
        out_t += "\n\n**Slot trace** (1 s samples of Strata's live view):\n\n" + tr
    return out_t


def trace_section(d):
    """trace.jsonl (pass 3): per item the seconds sampled, how often a prompt read and an answer ran at the same time
    (the interleaving pass 2 showed), the most requests waiting or outside the slots, and the conversation cache."""
    f = d / "trace.jsonl"
    if not f.exists():
        return ""
    per = OrderedDict()
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        per.setdefault(r.get("item", "?"), []).append(r)
    out = []
    for item, rs in per.items():
        def states(r):                        # a read on the solo path shows as the live state, not as a slot
            s = {x.get("state") for x in r.get("slots") or []}
            return s | ({"reading"} if r.get("state") == "reading" else set())
        both = sum(1 for r in rs if {"reading", "decoding"} <= states(r))
        cc = next((r.get("cc") for r in reversed(rs) if r.get("cc")), {}) or {}
        out.append([item.split("/")[0] + "/" + item.split("/")[1], len(rs), f"{100 * both / max(len(rs), 1):.0f} %",
                    max((r.get("waiting") or 0) for r in rs), max((r.get("outside") or 0) for r in rs),
                    ", ".join(f"{k} {cc[k]}" for k in ("enabled", "parks", "restores", "evictions", "parked") if k in cc) or "–"])
    return table(["item", "seconds", "read + write at once", "waiting max", "outside slots max", "conversation cache"],
                 out)


def agentloop_section(rows):
    """Per item: agents, turns reached, context at the first and the last turn (prompt tokens: Turnstone's system
    prompt and tool schemas show in the first), share of prompt tokens served from cache after turn 0, TTFT and decode
    speed first vs later turns, share of turns that called a tool."""
    per = OrderedDict()
    for r in rows:
        per.setdefault(r["item"], defaultdict(list))[r["agent"]].append(r)
    out = []
    for item, agents in per.items():
        convs = [sorted(rs, key=lambda r: r["turn"]) for rs in agents.values()]
        turns = [len(c) for c in convs]
        p0 = [c[0].get("prompt_tokens") for c in convs]
        pn = [c[-1].get("prompt_tokens") for c in convs]
        later = [r for c in convs for r in c[1:]]
        ptok = sum((r.get("prompt_tokens") or 0) for r in later)
        cached = sum((r.get("cached_tokens") or 0) for r in later)
        tools = sum(1 for c in convs for r in c if r.get("n_tool_calls"))
        n = sum(turns)
        out.append([item, len(convs), fmt(q(turns, .5), 1), fmt(q(p0, .5), 0), fmt(q(pn, .5), 0),
                    f"{cached / ptok:.0%}" if ptok else "–",
                    fmt(q([c[0].get("ttft_s") for c in convs], .5), 2), fmt(q([r.get("ttft_s") for r in later], .5), 2),
                    fmt(q([r.get("decode_tps") for r in later], .5), 1),
                    f"{tools / n:.0%}" if n else "–", sum(1 for c in convs if c[-1].get("finish") == "length"),
                    sum(1 for c in convs for r in c if not r.get("ok", True))])
    return table(["item", "agents", "turns p50", "prompt tok 1st", "prompt tok last", "cached (later turns)",
                  "TTFT 1st", "TTFT later", "decode later", "tool turns", "trunc", "errors"], out) + (
        "\n\ncached: llama.cpp reports reused tokens (cache_n), vLLM prompt_tokens_details; Strata only per request "
        "in its /metrics (engine-side table).")


def throughput_section(path):
    rows = [r for r in csv.DictReader(open(path)) if "/depth/" not in r["label"] and "/t=" not in r["label"]]
    if not rows:
        return ""
    t = OrderedDict()
    ss = sorted({int(r["streams"]) for r in rows})
    for r in rows:
        t.setdefault(r["label"], {})[int(r["streams"])] = r["aggregate_tps"]
    return table(["item"] + [f"{s} streams" for s in ss], [[k] + [d.get(s, "–") for s in ss] for k, d in t.items()])


def depth_section(path):
    """Depth items (label …/depth/fill=…): prompt speed from the fastest first token, queueing from the slowest."""
    rows = [r for r in csv.DictReader(open(path)) if "/depth/" in r["label"]]
    out = []
    for r in rows:
        pt, tmin = float(r.get("prompt_tokens") or 0), float(r.get("ttft_s_min") or 0)
        out.append([r["label"].replace("/depth/", " "), r["streams"], r.get("prompt_tokens", ""), r.get("ttft_s_min", "–"),
                    r.get("ttft_s_max", "–"), r.get("prefill_tps_slot") or (fmt(pt / tmin) + "*" if tmin else "–"),
                    r.get("prefill_tps_agg") or "–", r.get("decode_tps_steady") or "–", r["decode_tps_med"],
                    r["decode_tps_min"], r["aggregate_tps"], r.get("prompt_source") or "–", r["errors"]])
    return table(["variant fill", "clients", "prompt tok", "TTFT min s", "TTFT max s (queue)", "prefill tok/s per slot",
                  "prefill tok/s all", "decode steady", "decode med", "decode min", "decode all", "prompts", "errors"],
                 out) + (
        "\n\nprefill per slot: engine-timed (* = from the fastest first token where the engine does not time it); "
        "all = every prompt token ÷ time until the last request's first token. decode steady: per stream, only while "
        "every request of the round was decoding (no prefill of others in the window); med/min include that overlap. "
        "prompts: corpus = real text, filler = the old repeated sentence.") if out else ""


ENGINE_COLS = [("prefill_tps_slot", "prefill/slot"), ("prefill_tps_agg", "prefill all"), ("cache_hit", "prefix hit"),
               ("spec_accept", "draft accept"), ("kv_usage_max", "KV max"), ("waiting_max", "queue max"),
               ("preemptions", "preempt"), ("expert_hit", "expert hit"), ("pcie_share", "PCIe share")]


def engine_section(path):
    """What the engines reported per level (engine_metrics.py): caches, drafts, queue, KV, experts."""
    rows = list(csv.DictReader(open(path)))
    if not rows or "cache_hit" not in rows[0]:
        return ""
    out = [[r["label"], r["streams"]] + [r.get(k) if r.get(k) not in (None, "", "None") else "–" for k, _ in ENGINE_COLS]
           for r in rows]
    return table(["item", "clients"] + [h for _, h in ENGINE_COLS], out)


def starts_section(d):
    """What the engine made of each variant's settings (bench-svc up: ready time + key log lines)."""
    out = []
    for f in sorted(d.glob("up-*.log")):
        lines = [l.strip() for l in f.read_text(errors="replace").splitlines() if l.strip()]
        ready = next((l for l in lines if l.startswith("ready after") or l in ("not running", "crash loop")
                      or l.startswith("timeout")), "")
        keys = [l[-160:] for l in lines if l is not ready][-6:]
        out.append(f"- **{f.stem[3:]}** — {ready or 'no ready line'}" + "".join(f"\n  - `{k}`" for k in keys))
    return "\n".join(out)


def box_section(d, items):
    """Box stats averaged over each item's window (boxstats.csv: ts,key,value)."""
    import os
    f = Path(os.environ.get("BENCH_BOXSTATS") or d.parent / "boxstats.csv")
    if not f.is_file() or not items:
        return ""
    samples = defaultdict(list)                       # key -> [(ts, value)]
    for row in csv.reader(open(f)):
        try:
            samples[row[1]].append((float(row[0]), float(row[2])))
        except (ValueError, IndexError):
            continue
    gpus = sorted({int(k[3:k.index(".")]) for k in samples if k.startswith("gpu") and "." in k})
    head = ["item", "s"]
    for g in gpus:
        head += [f"GPU{g} util %", f"GPU{g} mem GB", f"GPU{g} W", f"GPU{g} PCIe rx MB/s"]
    head += ["CPU busy %", "iowait %", "RAM used GB", "disk read MB/s"]
    cores = sorted(k for k in samples if k.startswith("cpu.core") and k.endswith("_pct"))   # pass 4
    ctrs = sorted(k for k in samples if k.startswith("ctr.") and k.endswith(".cpu_cores"))
    if cores:
        head += ["cores > 50 %"]
    if ctrs:
        head += ["busiest container (cores)"]
    mbw = "membw.read_mbs" in samples                  # pcm-memory via ai-membw (2026-10-21)
    if mbw:
        head += ["RAM read GB/s", "RAM write GB/s"]
    rows = []
    for it in items:
        a, b = it.get("started"), it.get("finished")
        if not a or not b:
            continue

        def agg(key, how="avg"):
            v = [x for t, x in samples.get(key, []) if a <= t <= b + 5]
            if not v:
                return "–"
            return fmt(max(v) if how == "max" else sum(v) / len(v), 1)
        r = [it["id"], b - a]
        for g in gpus:
            mem = agg(f"gpu{g}.mem_mb", "max")
            r += [agg(f"gpu{g}.util"), fmt(float(mem) / 1024, 1) if mem != "–" else "–", agg(f"gpu{g}.power_w"),
                  agg(f"gpu{g}.pcie_rx_mbs")]
        reads = [k for k in samples if k.startswith("disk.") and k.endswith(".read_mbs")]
        rd = [agg(k) for k in reads]
        r += [agg("cpu.busy_pct"), agg("cpu.iowait_pct"), agg("ram.used_gb", "max"),
              fmt(sum(float(x) for x in rd if x != "–"), 1) if rd else "–"]
        if cores:
            busy = defaultdict(int); n = defaultdict(int)
            for k in cores:
                for ts, x in samples[k]:
                    if a <= ts <= b + 5:
                        busy[ts] += x > 50; n[ts] += 1
            r.append(fmt(sum(busy.values()) / len(busy), 1) if busy else "–")
        if ctrs:
            avg = {k[4:-10]: float(agg(k)) for k in ctrs if agg(k) != "–"}
            top = max(avg, key=avg.get) if avg else None
            r.append(f"{top} {avg[top]:.2f}" if top else "–")
        if mbw:
            gb = lambda k: (lambda v: "–" if v == "–" else fmt(float(v) / 1000, 1))(agg(k))   # noqa: E731
            r += [gb("membw.read_mbs"), gb("membw.write_mbs")]
        rows.append(r)
    return table(head, rows) if rows else ""


def small_section(rows):
    """Pass 4: the small models' envelopes — one line per item and call shape (guard: size × calls at once; judge:
    history × tool turns × calls at once; rerank: shape × calls at once / closed loop)."""
    out = []
    g = defaultdict(list)
    for r in rows:
        k = r["kind"]
        key = ((r["item"], r["size"], r["conc"]) if k == "guard" else
               (r["item"], r["history"], r["tool_turns"], r["conc"]) if k == "judge" else
               (r["item"], r["shape"], r["docs"], r["doc_tokens"], r["query_tokens"], r["conc"], r.get("closed_loop_s", 0)))
        g[(k,) + key].append(r)
    for kind, head in (
            ("guard", ["item", "tool output tok", "at once", "calls", "ok", "JSON ok", "prompt tok", "first token p50 s",
                       "verdict p50 s", "verdict p95 s", "prompt read tok/s"]),
            ("judge", ["item", "history tok", "tool turns", "at once", "evals", "ok", "JSON ok", "tool calls made",
                       "written tok", "thinking tok", "verdict p50 s", "verdict p95 s", "over 120 s", "reuse turns 2+"]),
            ("rerank", ["item", "shape", "docs", "doc tok", "query tok", "at once", "calls", "ok", "p50 s", "p95 s",
                        "calls/s", "docs/s"])):
        lines = []
        for key in sorted(k for k in g if k[0] == kind):
            rs = g[key]; ok = [r for r in rs if r.get("ok")]
            tot = [r.get("total_s") for r in ok]
            if kind == "guard":
                rate = [r["prompt_tokens"] / r["ttft_s"] for r in ok if r.get("prompt_tokens") and r.get("ttft_s")]
                lines.append(list(key[1:]) + [len(rs), len(ok), sum(bool(r.get("json_ok")) for r in ok),
                             fmt(q([r.get("prompt_tokens") for r in ok], .5)), fmt(q([r.get("ttft_s") for r in ok], .5), 2),
                             fmt(q(tot, .5), 2), fmt(q(tot, .95), 2), fmt(q(rate, .5))])
            elif kind == "judge":
                reuse = []
                for r in ok:
                    for c, pt in list(zip(r.get("cached_per_turn") or [], r.get("prompt_per_turn") or []))[1:]:
                        if c is not None and pt:
                            reuse.append(c / pt)
                lines.append(list(key[1:]) + [len(rs), len(ok), sum(bool(r.get("json_ok")) for r in ok),
                             fmt(q([r.get("tool_calls_made") for r in ok], .5)),
                             fmt(q([r.get("completion_tokens") for r in ok], .5)),
                             fmt(q([r.get("reasoning_tokens") for r in ok], .5)), fmt(q(tot, .5), 1), fmt(q(tot, .95), 1),
                             sum(bool(r.get("over_timeout")) for r in rs),
                             f"{round(100 * q(reuse, .5))} %" if reuse else "–"])
            else:
                span = (max(r["t_end"] for r in rs) - min(r["t_start"] for r in rs)) if rs else 0
                lines.append(list(key[1:6]) + [len(rs), len(ok), fmt(q(tot, .5), 3), fmt(q(tot, .95), 3),
                             fmt(len(ok) / span, 2) if span else "–", fmt(len(ok) * key[3] / span, 0) if span else "–"])
        if lines:
            out += [f"\n**{kind}**\n", table(head, lines)]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state_dir")
    ap.add_argument("--by", choices=["level", "shape", "task"], default="level")
    a = ap.parse_args(argv)
    d = Path(a.state_dir)
    parts = [f"# Bench report — {d.name}\n"]
    sf = d / "state.json"
    if sf.exists():
        items = json.loads(sf.read_text())["items"]
        c = defaultdict(int)
        for i in items:
            c[i["status"]] += 1
        parts.append("Status: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
        bad = [i for i in items if i["status"] in ("failed", "skipped")]
        if bad:
            parts.append("\n" + "\n".join(f"- **{i['status']}** {i['id']}: {i.get('last_error', '')}" for i in bad))
    rows = []
    rf = d / "results.jsonl"
    if rf.exists():
        rows = [json.loads(l) for l in rf.read_text().splitlines() if l.strip()]
        latest = {}
        for r in rows:                       # an interrupted attempt leaves partial rows: keep the latest only
            latest[r["item"]] = max(latest.get(r["item"], 0), r.get("attempt", 1))
        rows = [r for r in rows if r.get("attempt", 1) == latest[r["item"]]]
    small = [r for r in rows if r.get("kind") in ("guard", "judge", "rerank")]
    rows = [r for r in rows if r.get("kind") not in ("guard", "judge", "rerank")]
    single = [r for r in rows if "turn" not in r]
    multi = [r for r in rows if "turn" in r and "agent" not in r and "conv" not in r]
    longt = [r for r in rows if "conv" in r]
    loops = [r for r in rows if "agent" in r]
    if small:
        parts += ["\n## Small models — envelopes (pass 4)\n", small_section(small)]
    if single:
        parts += ["\n## Tasks\n", tasks_section(single, a.by)]
    if multi:
        parts += ["\n## Multi-turn (agent-loop stand-in)\n", multiturn_section(multi)]
    if longt:
        parts += ["\n## Long tasks (one growing conversation per slot; pass 2)\n", longtask_section(longt, d)]
    if loops:
        parts += ["\n## Agent loops (parallel agents, growing context; direct vs Turnstone's request shape)\n",
                  agentloop_section(loops)]
    if (d / "throughput.csv").exists():
        dep = depth_section(d / "throughput.csv")
        if dep:
            parts += ["\n## Depth (prompts at a share of the slot context; clients = slots, then slots + 2)\n", dep]
        su = sustained_section(d / "throughput.csv")
        if su:
            parts += ["\n## Sustained load (first vs last rounds)\n", su]
        tp = throughput_section(d / "throughput.csv")
        if tp:
            parts += ["\n## Throughput (aggregate tok/s)\n", tp]
        en = engine_section(d / "throughput.csv")
        if en:
            parts += ["\n## Engine-side (per level: caches, drafts, queue, KV, Strata's experts)\n", en]
    if sf.exists():
        bx = box_section(d, json.loads(sf.read_text())["items"])
        if bx:
            parts += ["\n## Box during each item (averages; memory = peak)\n", bx]
    st = starts_section(d)
    if st:
        parts += ["\n## Variant starts (ready time, what the engine reported)\n", st]
    text = "\n".join(parts) + "\n"
    (d / "report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
