#!/usr/bin/env python3
"""Writes the benchmark pass 2 plan (docs/bench-plan.md "Pass 2", user 2026-10-21): how low can Strata go?

The box is exclusive: guard, sub-maker and the CPU reranker stopped (run-stages --membw stops them first and then
calibrates the RAM bandwidth: membw-calibrate.py → the real 100 %), Turnstone drained by ops/bench.yml. Strata runs
alone on card #1, always at 128K context or more, KV int8 streamed to RAM.

  windows        one conversation + MTP: KV window on the card 16K / 32K / 64K / 96K / the whole 128K; 256K context
  streams        2 / 3 / 4 conversations without MTP (window 32K) — MTP with several adds ~5 % (pass 1)
  ladder mtp1    one + MTP, window 32K, expert cache = 80 … 10 % of what Strata takes on its own (auto)
  ladder nomtp2  two without MTP, window 32K, 65 … 20 %
  stop rule      a ladder stops at RAM bandwidth >= 90 % of the calibrated peak, or writing below 15 tok/s on two
                 steps in a row (user); the remaining steps are skipped with the reason
  grow           the task chain from an empty context until it is 90 % full: at auto and at the ladder's last step
                 that stayed within the limits
  reference      the production layout (#1+#2) with one conversation + MTP, same workloads

Workloads: deep (a real-text prefix filling 70 % of the context, then three turns of tasks/longtask-cat.json — an
animated SVG of a cat on a fence at night, rewritten in full each turn; answers up to 8K tokens) and grow.
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_phase1 as p1  # noqa: E402

K = p1.K
CHECK = "--check" in sys.argv
EXCLUSIVE = ["llama-guard", "vllm-sub", "llama-rerank"]
BASE1, BASE2 = "strata1-b1-128k-w32k-mtp", "strata1-b2-128k-w32k"


def one_card(slots, ctx=128 * K, window=32 * K, mtp=True, frac=None, base=None, ladder=None, cache_from=None,
             vid=None, smoke=False, note=""):
    v, _ = p1.strata(slots, ctx, stream=window is not None, mtp=mtp, resident=window or 32 * K)
    w = f"w{window // K}k" if window else "full"
    v["id"] = vid or (f"strata1-b{slots}-{ctx // K}k-{w}{'-mtp' if mtp else ''}"
                      + (f"-cache{round(frac * 100)}pct" if frac else ""))
    # one card: no layer split (nor its options), no reserve for the guard, no wait for the guard on #2
    v["strata_args_remove"] = v.get("strata_args_remove", []) + ["--trim-stage-weights", "--remote-expert-opt",
                                                                  "--vram-reserve-later-mib"]
    v.update(gpus=[0], env={"STRATA_WAIT_GPU_USED": ""}, strata_config_set={"gpu": [0], "layer_split": "auto"},
             compose=p1.devices([0]), compose_replace=["deploy"])
    if frac:
        v.update(expert_cache_frac=frac, ladder_base=base)
    if ladder:
        v["ladder"] = ladder
    if cache_from:
        v["expert_cache_from"] = cache_from
    if smoke:
        v["smoke"] = True
    return v, note


def reference():
    v, _ = p1.strata(1, 128 * K, stream=True)
    v["id"] = "ref-2cards-b1-128k-w32k-mtp"
    v["strata_args_remove"] = v.get("strata_args_remove", []) + ["--vram-reserve-later-mib"]   # no guard on #2 now
    v["env"] = {"STRATA_WAIT_GPU_USED": ""}
    return v, "reference: today's layout on #1+#2, guard stopped"


MTP1 = [0.8, 0.65, 0.5, 0.4, 0.3, 0.22, 0.15, 0.1]
NOMTP2 = [0.65, 0.45, 0.3, 0.2]
VARIANTS = (
    [one_card(1, window=32 * K, ladder="mtp1", smoke=True, note="ladder mtp1 base: expert cache auto; smoke"),
     one_card(1, window=16 * K), one_card(1, window=64 * K), one_card(1, window=96 * K),
     one_card(1, window=None, note="the whole 128K KV on the card"),
     one_card(1, ctx=256 * K, window=32 * K, note="256K context"),
     one_card(2, window=32 * K, mtp=False, ladder="nomtp2", note="ladder nomtp2 base"),
     one_card(3, window=32 * K, mtp=False), one_card(4, window=32 * K, mtp=False, smoke=True, note="smoke: 4 × 128K")]
    + [one_card(1, frac=f, base=BASE1, ladder="mtp1", smoke=f == 0.3, note="smoke: capped cache" if f == 0.3 else "")
       for f in MTP1]
    + [one_card(2, mtp=False, frac=f, base=BASE2, ladder="nomtp2") for f in NOMTP2]
    + [one_card(1, cache_from="mtp1", vid="strata1-b1-128k-w32k-mtp-lastok",
                note="the ladder's last step within the limits, grown to 90 %"),
       reference()]
)
DEEP = {"id": "deep", "kind": "longtask", "tasks": "tasks/longtask-cat.json", "corpus": p1.CORPUS, "fill": 0.7,
        "max_turns": 3, "max_tokens": 8192, "min_tps": 10, "effort": ["template:low"], "request_timeout_s": 3600,
        "timeout_s": 7200}
VARIANT_IDS = [v["id"] for v, _ in VARIANTS]
WORKLOADS = [
    dict(DEEP, variants=[i for i in VARIANT_IDS if i.endswith("-lastok") is False], ladder_eval=True),
    dict(DEEP, id="grow", fill=0, until_fill=0.9, max_turns=40, min_tps=0, timeout_s=10800,
         variants=[BASE1, "strata1-b1-128k-w32k-mtp-lastok"]),
]
LADDERS = """
# ---------------------------------------------------------------------------------------------------- ladders
[ladder.mtp1]                # user 2026-10-21: stop at 90 % of the calibrated RAM bandwidth or below 15 tok/s twice
stop_membw_pct = 90
stop_tps_below = 15
consecutive = 2

[ladder.nomtp2]
stop_membw_pct = 90
stop_tps_below = 15
consecutive = 2
"""


def main():
    p1.CHECK = CHECK
    p1.STALE.clear()
    vs = []
    for v, n in VARIANTS:
        v = copy.deepcopy(v)
        v["stop"] = EXCLUSIVE
        vs.append((v, n))
    f = p1.write("p2-strata1", "pass 2: Strata alone on card #1 (how low can it go)", [0, 1], 2400, vs,
                 [copy.deepcopy(w) for w in WORKLOADS], on_end="stop", extra=LADDERS)
    if CHECK:
        if p1.STALE:
            print(f"FAIL: {', '.join(p1.STALE)} differ from gen_phase2.py — run it and commit", file=sys.stderr)
            return 1
        print("ok: bench pass-2 plan is generated from gen_phase2.py")
        return 0
    print(f"wrote {f.relative_to(HERE.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
