#!/usr/bin/env python3
"""Writes the benchmark pass 3 plan (docs/bench-plan.md "Pass 3", user 2026-10-21; re-scoped the same day): Strata on
one card with ONE slot — several slots do not pay on one card (Strata's setup: IQ3_S "needs 32 GB or a split"; pass 2:
two slots ~62 tok/s together against ~90 alone), so the production shape is one slot + parking.

The box exclusive as in pass 2 (every other engine stopped, RAM calibrated first), Strata alone on card #1, 128K, KV
int8 with a 32K window, expert cache auto, effort low. Every workload writes a 1-second trace of Strata's live view
(trace.jsonl); rows carry start / first-token / end times; the whole engine log is kept per variant.

  Q1 thinking sent back   deep-think here vs pass 2's deep on the same configurations (1 slot + MTP, 2 slots)
  Q2 slot / queue view    from the trace (every workload)
  Q3 several conversations on one slot   queue4 (4 conversations, fill 0.5): no parking vs 8192 MiB / 4
  Q4 mixed lengths        mixed4 (5 / 30 / 70 / 30 %, 60 s apart) on one slot, no parking vs parking
  Q5 helper threads       --aux-cpus auto
  Q6 power                80 % prefix + 8K answers at 220 / 250 / 300 W on card #1 (user 2026-10-21)
  Q7 prefill chunk        --prefill 4096 / auto (<= 8192) / auto:16384 / auto:32768 (user 2026-10-21)
  Q8 CPU scaling          5 % of the auto cache, --pool-workers 4 / 6 / 8 / default (user 2026-10-21)
  Q9 several slots        --batch-mtp + parking: two slots on queue4; one / two / three slots on par3 (the same three
                          conversations at once), compared by tokens written / wall time (user 2026-10-22)
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
PARK = {"--conversation-cache-mib": "8192", "--conversation-cache-slots": "4"}


def card(slots, mtp, vid, args=None, power=None, add=None, window=32 * K, smoke=False, note=""):
    frac = 0.05 if "-c5" in vid else None        # Q8: the cache as a share of the base's auto slots
    v, _ = p2.one_card(slots, window=window, mtp=mtp, vid=vid, smoke=smoke, note=note,
                       frac=frac, base="p3-b1-mtp" if frac else None)
    if args:
        v["strata_args_set"] = {**v.get("strata_args_set", {}), **args}
    if add:
        v["strata_args_add"] = v.get("strata_args_add", []) + add
    if power:
        v["gpu_power_w"] = power
    return v, note


VARIANTS = [
    card(1, True, "p3-b1-mtp", smoke=True, note="one stream + MTP (pass 2's base, no parking); smoke"),
    card(1, True, "p3-b1-mtp-aux", args={"--aux-cpus": "auto"}, note="Q5: helper threads on the SMT siblings"),
    card(1, True, "p3-b1-mtp-300w", power=300, note="Q6: card #1 at 300 W (user's OK 2026-10-21)"),
    card(1, True, "p3-b1-mtp-220w", power=220, note="Q6: card #1 at 220 W (user 2026-10-21)"),
    # Q7 (user 2026-10-21): prefill chunk size — each chunk streams every expert it routes to once, so the chunk sets
    # the bytes per token over PCIe; auto takes up to 8192 (what it can borrow from the expert cache)
    card(1, True, "p3-b1-mtp-pf4k", args={"--prefill": "4096"}, note="Q7: fixed 4K prefill chunks"),
    card(1, True, "p3-b1-mtp-pf16k", args={"--prefill": "auto:16384"}, note="Q7: auto up to 16K chunks"),
    card(1, True, "p3-b1-mtp-pf32k", args={"--prefill": "auto:32768"}, note="Q7: auto up to 32K chunks"),
    # Q8 (user 2026-10-21): does writing scale with the CPU? Pass 2 at 22 % of the auto cache: hit 0.67, 62 tok/s; at
    # ~1,040 experts still ~50 tok/s with RAM bandwidth ~22 % of the peak — the misses look compute-bound. The sweep at
    # 5 % (user: the CPU's share is largest there): --pool-workers 4 / 6 / 8 / default (every physical core but the
    # host loop's, the host joining the drain).
    *[card(1, True, f"p3-b1-mtp-c5-w{n}", args={"--pool-workers": str(n)}, note=f"Q8: 5 % cache, {n} CPU workers")
      for n in (4, 6, 8)],
    card(1, True, "p3-b1-mtp-c5", note="Q8: 5 % cache, default workers"),
    # Q3 (re-scoped 2026-10-21, user): one slot serving several conversations — the production shape on one card
    card(1, True, "p3-b1-mtp-park", args=PARK, smoke=True, note="Q3: one slot + parking 8 GiB / 4; smoke"),
    card(2, False, "p3-b2", note="two slots, no MTP, no parking (= pass 2's strata1-b2-128k-w32k): deep vs deep-think"),
    # Q9 (re-scoped): one confirmation that two slots do not pay on one card — the best two-slot setup (--batch-mtp,
    # parking) against one slot + parking on the same queue
    card(2, True, "p3-b2-bmtp", args=PARK, add=["--batch-mtp"], note="Q9: two slots, --batch-mtp, parking"),
    # user 2026-10-22: pass 2's multi-stream points ran without --batch-mtp — do three streams with it tie or beat one
    # stream with MTP? Same work (par3) on one, two and three slots, compared by tokens written / wall time
    card(3, True, "p3-b3-bmtp", args=PARK, add=["--batch-mtp"], note="Q9: three slots, --batch-mtp, parking"),
]
BASE = {"kind": "longtask", "tasks": "tasks/longtask-cat.json", "corpus": p1.CORPUS, "max_turns": 3,
        "max_tokens": 8192, "effort": ["template:low"], "request_timeout_s": 3600, "trace": True}
WORKLOADS = [
    # no plain "deep" here: pass 2 ran it on the same configurations (b1 + MTP, b2; user 2026-10-21) — the deep vs
    # deep-think comparison takes those rows (docs/bench-data/bench-pass2-part1-*)
    # = pass 2's deep (fill 0.7, 3 turns, 8K answers, min_tps 10) except think_back and the trace
    dict(BASE, id="deep-think", fill=0.7, think_back=True, min_tps=10, timeout_s=7200,
         variants=["p3-b1-mtp", "p3-b1-mtp-aux", "p3-b1-mtp-pf4k", "p3-b1-mtp-pf16k", "p3-b1-mtp-pf32k",
                   "p3-b1-mtp-c5-w4", "p3-b1-mtp-c5-w6", "p3-b1-mtp-c5-w8", "p3-b1-mtp-c5", "p3-b2"]),
    # Q6 (user 2026-10-21): a 128K slot, 80 % prefix, then 8K answers until it is full (2 full + a shorter third),
    # at 220 / 250 (the declared limit) / 300 W
    dict(BASE, id="power", fill=0.8, think_back=True, timeout_s=7200,
         variants=["p3-b1-mtp-220w", "p3-b1-mtp", "p3-b1-mtp-300w"]),
    # Q3: four conversations take turns on one slot, without and with parking; the same queue on two slots (Q9)
    dict(BASE, id="queue4", fill=0.5, think_back=True, conversations=4, timeout_s=10800,
         variants=["p3-b1-mtp", "p3-b1-mtp-park", "p3-b2-bmtp"]),
    # Q9 (user 2026-10-22): three conversations at once, the same work on one slot (queued, parking), two and three
    # slots (--batch-mtp, parking) — the verdict on written / wall time (query to completion), not per-answer speed
    dict(BASE, id="par3", fill=0.7, think_back=True, min_tps=10, conversations=3, timeout_s=10800,
         variants=["p3-b1-mtp-park", "p3-b2-bmtp", "p3-b3-bmtp"]),
    # Q4: mixed lengths arriving 60 s apart on one slot, without and with parking
    dict(BASE, id="mixed4", fills=[0.05, 0.3, 0.7, 0.3], stagger_s=60, think_back=True, conversations=4,
         timeout_s=10800, variants=["p3-b1-mtp", "p3-b1-mtp-park"]),
]


def main():
    p1.CHECK = CHECK
    p1.STALE.clear()
    vs = []
    for v, n in VARIANTS:
        v = copy.deepcopy(v)
        v["stop"] = p2.EXCLUSIVE
        vs.append((v, n))
    f = p1.write("p3-strata1", "pass 3: Strata on one card — history reuse, parking, mixed lengths", [0, 1], 2400, vs,
                 [copy.deepcopy(w) for w in WORKLOADS], on_end="stop")
    if CHECK:
        if p1.STALE:
            print(f"FAIL: {', '.join(p1.STALE)} differ from gen_phase3.py — run it and commit", file=sys.stderr)
            return 1
        print("ok: bench pass-3 plan is generated from gen_phase3.py")
        return 0
    print(f"wrote {f.relative_to(HERE.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
