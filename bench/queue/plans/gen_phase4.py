#!/usr/bin/env python3
"""Writes the benchmark pass 4 plans (docs/bench-plan.md "Pass 4", user 2026-10-22): the envelopes of everything that
might share a card or the CPU with Strata, each measured ALONE on an exclusive box (combined runs come later, once the
envelopes are known). Every other engine is stopped first; RAM bandwidth is calibrated (or reused) first.

  p4-cpu.toml    A — Strata's CPU need at high expert-cache hit rates: deep 128K fence-cat (deep-think) with
                 --pool-workers 1, 2, 3, … until writing stops improving (ladder: under +3 % on the best so far on two
                 steps in a row), then the default as a check. One card (#1, cache auto, hit ~0.91) and two cards
                 (#1+#2, today's split, hit ~0.99). Strata pins its host loop to core 0 and worker n to core n.
  p4-small.toml  B — guard (Ministral 3 3B), judge (Granite 4.2 8B) and reranker (bge-reranker-v2-m3) alone, each on its
                 candidate cards (3060 and a 3090) and the reranker also on the CPU:
                 guard   tool outputs 0.5 / 2 / 8 / 16 / 28K tokens, 1 and 2 at once (Turnstone's max_concurrency);
                         KV 4 x 8K (today), 2 x 16K, 1 x 32K, 4 slots sharing 32K (--kv-unified)
                 judge   history 4K / 12K, 0 / 2 / 5 forced read-only tool turns (3K-token reads), 1 and 2 at once;
                         thinking off / low / full (Granite's default; Turnstone sends nothing) on today's KV shape,
                         low only on 1 x 32K and 2 slots sharing 32K
                 rerank  Turnstone's call shapes (<= 50 documents) + the merge-audit sweep (64 per call, closed loop);
                         CPU threads 1 / 2 / 4 / 6 / 8 / 10 (4 slots) and 4 threads with 1 / 2 slots (today: 2), and
                         on the 3060 and a 3090 (CUDA build, all layers on the GPU)

The 3060's index comes from GPU_SMALL in the box's .env, the 3090 used is #2 (GPU_MAKER_B, the guard's card today).
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_phase1 as p1  # noqa: E402
import gen_phase2 as p2  # noqa: E402

K = p1.K
CHECK = "--check" in sys.argv
STOP_ALL = p2.EXCLUSIVE + ["llama-judge"]
CUDA = "ghcr.io/ggml-org/llama.cpp:${LLAMACPP_CUDA_TAG}"
CARD = {"3060": "${GPU_SMALL}", "3090": "${GPU_MAKER_B:-1}"}


def dev(card):
    return {"deploy": {"resources": {"reservations": {"devices": [
        {"driver": "nvidia", "device_ids": [CARD[card]], "capabilities": ["gpu"]}]}}}}


# ------------------------------------------------------------------------------------------------ A: Strata's CPU need
WORKERS = list(range(1, 10))                 # 10 physical cores: core 0 is the host loop, workers on 1 … 9


def strata_steps():
    out = []
    for lay in ("1c", "2c"):
        for n in WORKERS + [0]:              # 0 = Strata's default (every physical core but the host's): the check
            if lay == "1c":
                v, _ = p2.one_card(1, window=32 * K, mtp=True, vid=f"p4-{lay}-w{n or 'def'}", smoke=(n == 1))
            else:
                v, _ = p2.reference()
                v["id"] = f"p4-{lay}-w{n or 'def'}"
            if n:
                v["strata_args_set"] = {**v.get("strata_args_set", {}), "--pool-workers": str(n)}
                v["ladder"] = f"cpu{lay}"
            v["stop"] = STOP_ALL
            note = (f"{'one card #1' if lay == '1c' else 'two cards #1+#2'}: "
                    + (f"{n} pool worker{'s' if n > 1 else ''} (cores 1–{n})" if n else "default workers (check)"))
            out.append((v, note))
    return out


LADDERS = """
# pass 4 (user 2026-10-22): add pool workers until writing stops improving — under +3 % on the best so far on two steps
[ladder.cpu1c]
stop_gain_below_pct = 3
consecutive = 2

[ladder.cpu2c]
stop_gain_below_pct = 3
consecutive = 2
"""
DEEP = {"id": "deep-think", "kind": "longtask", "tasks": "tasks/longtask-cat.json", "corpus": p1.CORPUS,
        "max_turns": 3, "max_tokens": 8192, "effort": ["template:low"], "request_timeout_s": 3600, "trace": True,
        "fill": 0.7, "think_back": True, "min_tps": 10, "timeout_s": 7200, "ladder_eval": True,
        "variants": ["p4-*"]}


# ------------------------------------------------------------------------------------------------ B: the small models
def guard(card, ctx, slots, unified=False, smoke=False):
    vid = f"guard-{card}-" + (f"{ctx // K}k-u{slots}" if unified else f"{ctx // slots // K}kx{slots}")
    v = {"id": vid, "service": "llama-guard", "ctx": ctx // (1 if unified else slots), "slots": slots,
         "args_set": {"-c": str(ctx), "-np": str(slots)},
         "args_add": ["--metrics"] + (["--kv-unified"] if unified else []),
         "compose": dev(card), "compose_replace": ["deploy"], "stop": STOP_ALL}
    if smoke:
        v["smoke"] = True
    return v, f"guard on the {card}: " + (f"{slots} slots sharing {ctx // K}K" if unified
                                         else f"{slots} x {ctx // slots // K}K" + (" (today)" if slots == 4 else ""))


def judge(card, ctx, slots, unified=False, smoke=False):
    vid = f"judge-{card}-" + (f"{ctx // K}k-u{slots}" if unified else f"{ctx // slots // K}kx{slots}")
    v = {"id": vid, "service": "llama-judge", "ctx": ctx // (1 if unified else slots), "slots": slots,
         "args_set": {"-c": str(ctx), "-np": str(slots)},
         "args_add": ["--metrics"] + (["--kv-unified"] if unified else []),
         "compose": dev(card), "compose_replace": ["deploy"], "stop": STOP_ALL}
    if smoke:
        v["smoke"] = True
    return v, f"judge on the {card}: " + (f"{slots} slots sharing {ctx // K}K" if unified
                                         else f"{slots} x {ctx // slots // K}K" + (" (today)" if slots == 2 else ""))


def rerank_cpu(threads, slots=4, smoke=False):
    v = {"id": f"rerank-cpu-t{threads}" + ("" if slots == 4 else f"-np{slots}"), "service": "llama-rerank",
         "slots": slots, "args_set": {"-t": str(threads), "-np": str(slots)}, "args_add": ["--metrics"],
         "stop": STOP_ALL}
    if smoke:
        v["smoke"] = True
    return v, f"reranker on the CPU: {threads} threads, {slots} slots" + (" (today)" if (threads, slots) == (4, 2) else "")


def rerank_gpu(card, smoke=False):
    v = {"id": f"rerank-{card}", "service": "llama-rerank", "slots": 4, "image": CUDA,
         "args_set": {"-np": str(4)}, "args_remove": ["-t"], "args_add": ["--metrics", "-ngl", "99"],
         "compose": {**dev(card), "environment": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}}, "stop": STOP_ALL}
    if smoke:
        v["smoke"] = True
    return v, f"reranker on the {card} (CUDA build, every layer on the GPU)"


SMALL = ([guard(c, 32 * K, s, smoke=(c == "3060" and s == 4)) for c in ("3060", "3090") for s in (4, 2, 1)]
         + [guard(c, 32 * K, 4, unified=True) for c in ("3060", "3090")]
         + [judge(c, 32 * K, s, smoke=(c == "3060" and s == 2)) for c in ("3060", "3090") for s in (2, 1)]
         + [judge(c, 32 * K, 2, unified=True) for c in ("3060", "3090")]
         + [rerank_cpu(t, smoke=(t == 4)) for t in (1, 2, 4, 6, 8, 10)]
         + [rerank_cpu(4, 2), rerank_cpu(4, 1)]
         + [rerank_gpu("3060", smoke=True), rerank_gpu("3090")])

OFF = {"enable_thinking": False}
LOW = {"enable_thinking": True, "low_effort": True, "reasoning_effort": "low"}   # Granite 4.2: either spelling
SHAPES = ["memory:50:40:30", "tools:50:150:30", "web:30:100:20", "longquery:50:40:400"]
JUDGE = {"kind": "judge", "corpus": p1.CORPUS, "max_tokens": 2048, "read_tokens": 3000, "judge_timeout_s": 120,
         "request_timeout_s": 600}
SMALL_W = [
    {"id": "guard", "kind": "guard", "corpus": p1.CORPUS, "variants": ["guard-*"], "sizes": [512, 2048, 8192, 16384,
     28000], "conc": [1, 2], "repeats": 3, "max_tokens": 512, "request_timeout_s": 120, "timeout_s": 3600},
    dict(JUDGE, id="judge-off", variants=["judge-*-16kx2"], chat_kwargs=OFF, smoke_skip=True, history=[4096, 12288],
         tool_turns=[0, 2, 5], conc=[1, 2], repeats=2, timeout_s=3600),
    dict(JUDGE, id="judge-low", variants=["judge-*"], chat_kwargs=LOW, history=[4096, 12288],
         tool_turns=[0, 2, 5], conc=[1, 2], repeats=2, timeout_s=5400),
    # full thinking (Granite's own default; the engine runs with thinking off by default since 2026-10-22)
    dict(JUDGE, id="judge-default", variants=["judge-*-16kx2"], chat_kwargs={"enable_thinking": True}, smoke_skip=True,
         history=[4096, 12288], tool_turns=[0, 2, 5],
         conc=[1], repeats=1, timeout_s=5400),
    {"id": "rerank-light", "kind": "rerank", "corpus": p1.CORPUS, "variants": ["rerank-cpu-t1", "rerank-cpu-t2"],
     "shapes": SHAPES, "conc": [1], "repeats": 1, "request_timeout_s": 600, "timeout_s": 3600},
    {"id": "rerank", "kind": "rerank", "corpus": p1.CORPUS, "smoke_skip": True,
     "variants": ["rerank-cpu-t4*", "rerank-cpu-t6", "rerank-cpu-t8", "rerank-cpu-t10", "rerank-3060", "rerank-3090"],
     "shapes": SHAPES, "conc": [1, 4], "repeats": 2, "request_timeout_s": 600, "timeout_s": 3600},
    {"id": "rerank-sweep", "kind": "rerank", "corpus": p1.CORPUS, "variants": ["rerank-*"],
     "shapes": ["sweep:64:40:40"], "conc": [1, 4], "duration_s": 60, "request_timeout_s": 600, "timeout_s": 1800},
]


def main():
    p1.CHECK = CHECK
    p1.STALE.clear()
    files = [
        p1.write("p4-cpu", "pass 4 A: Strata's CPU need at high hit rates (pool-worker ladder)", [0, 1], 2400,
                 [copy.deepcopy(x) for x in strata_steps()], [copy.deepcopy(DEEP)], on_end="stop", extra=LADDERS),
        p1.write("p4-small", "pass 4 B: guard, judge and reranker alone on each candidate card / the CPU",
                 [0, 1, 2, 3], 900, [copy.deepcopy(x) for x in SMALL], [copy.deepcopy(w) for w in SMALL_W],
                 on_end="stop"),
    ]
    if CHECK:
        if p1.STALE:
            print(f"FAIL: {', '.join(p1.STALE)} differ from gen_phase4.py — run it and commit", file=sys.stderr)
            return 1
        print("ok: bench pass-4 plans are generated from gen_phase4.py")
        return 0
    for f in files:
        print(f"wrote {f.relative_to(HERE.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
