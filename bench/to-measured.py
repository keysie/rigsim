#!/usr/bin/env python3
"""Turn benchw.py result rows (JSONL) into the "measured values" the simulator takes per engine.

    python3 to-measured.py results.jsonl                 # prints a JSON object for the engine's "benchmarks"
    python3 to-measured.py results.jsonl --streams 4     # the run used 4 concurrent requests -> decode_4

Per row benchw writes prompt_tokens, completion_tokens, ttft_s and total_s. Reading speed is prompt tokens
over the time to the first token; writing speed is completion tokens over the time after the first token.
With N concurrent streams the per-row writing speeds are summed within overlapping windows (decode_N total).
Medians over rows; rows that failed or were truncated are skipped.
"""
import argparse
import json
import statistics as st
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--streams", type=int, default=1, help="concurrent requests in the run (1 = alone)")
    ap.add_argument("--kv-room", type=int, default=0, help="KV room in tokens, if you read it from the engine log")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    rows = [r for r in rows if r.get("ok") and r.get("completion_tokens") and r.get("ttft_s") and r.get("total_s")]
    if not rows:
        sys.exit("no usable rows (need ok, completion_tokens, ttft_s, total_s)")
    reads = [r["prompt_tokens"] / r["ttft_s"] for r in rows if r.get("prompt_tokens") and r["ttft_s"] > 0.3]
    writes = [r["completion_tokens"] / (r["total_s"] - r["ttft_s"]) for r in rows if r["total_s"] > r["ttft_s"] + 1]
    out = {}
    if writes:
        per = st.median(writes)
        key = "decode_1" if a.streams <= 1 else f"decode_{a.streams}"
        out[key] = round(per * max(1, a.streams), 1)
    if reads:
        out["prefill"] = round(st.median(reads), 0)
    if a.kv_room:
        out["kv_room_tokens"] = a.kv_room
    print(json.dumps(out, indent=1))
    print(f"# {len(rows)} rows; writes median {st.median(writes) if writes else 0:.1f} tok/s per request, "
          f"reads median {st.median(reads) if reads else 0:.0f} tok/s", file=sys.stderr)


if __name__ == "__main__":
    main()
