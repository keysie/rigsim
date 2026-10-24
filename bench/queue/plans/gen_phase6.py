#!/usr/bin/env python3
"""Writes the benchmark pass 6 plans (docs/bench-plan.md "Pass 6") from p6-selection.toml and the pass-1 variant
definitions in gen_phase1.py (one source: a chosen configuration is benched exactly as in pass 1).

  group 1  p6-maker.toml   the chosen Strata variants on #1+#2          } in parallel
           p6-sub.toml     the chosen sub-maker variants on #3          }   (llama.cpp ones re-slotted to vllm-sub)
  group 2  p6-extra-g0/g1/g2.toml   the deferred variants, screened like pass 1 (Strata and guard stopped first)
  group 3  p6-exclusive.toml        a chosen maker3gpu-* variant (if any)

Pass-2 workloads (full): depth at 25/50/75/90 % (slots and slots + 2, two rounds), ramp, agent loops (parallel agents
= slots, growing context, direct vs Turnstone's request shape — tasks/turnstone-shape/), sustained load (30 min at
slots clients), the answer-length tasks (all seed tasks). While p6-selection.toml holds placeholders nothing is
written and `--check` passes (pass 6 not chosen yet).
"""
import copy
import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_phase1 as p1  # noqa: E402

K = p1.K
CHECK = "--check" in sys.argv
CORPUS = p1.CORPUS

DEPTH = {"id": "depth", "kind": "throughput", "fill": [0.25, 0.5, 0.75, 0.9], "streams": "slots+0,slots+2",
         "max_tokens": 256, "rounds": 2, "corpus": CORPUS}
RAMP = dict(p1.RAMP)
SUSTAINED = {"id": "sustained", "kind": "throughput", "streams": "slots+0", "prompt_tokens": 4000, "max_tokens": 512,
             "duration_s": 1800, "corpus": CORPUS}


def loops(shape):
    # max_tokens 8192 = Turnstone's answer budget (config.toml); agents = the engine's slots (a maker's fan-out)
    return {"id": "agents", "kind": "agentloop", "shape": ["direct", shape], "agents": "slots+0", "turns": 16,
            "tool_tokens": 2000, "max_tokens": 8192, "corpus": CORPUS, "effort": ["template:low"]}


def answerlen(ctx):
    return {"id": "answerlen", "kind": "tasks", "tasks": "tasks/seed.jsonl", "effort": ["template:low"], "streams": 2,
            "max_tokens": ctx - 4096}


def all_p1():
    """Every pass-1 variant by id (subB without its slot suffix → the vllm-sub slot version)."""
    out = {}
    for v, n in p1.MAKER + p1.GUARD + p1.SUB_A + p1.EXCL:
        out[v["id"]] = (v, n)
    for vs in p1.SUB_B.values():              # by the id without slot suffix; vllm-sub's own version wins
        for v, n in vs:
            base = v["id"].removesuffix("-g0").removesuffix("-g1")
            if base not in out or v["service"] == "vllm-sub":
                out[base] = (v, n)
    return out


def reslot_to_sub(v):
    """A subB variant from Strata's or the guard's slot as it runs in vllm-sub's (port 8000, LOCAL key, #3)."""
    v = copy.deepcopy(v)
    for suf in ("-g0", "-g1"):
        v["id"] = v["id"].removesuffix(suf)
    if v["service"] != "vllm-sub":
        cmd = v["command"]
        cmd[cmd.index("--port") + 1] = "8000"
        cmd[cmd.index("--api-key") + 1] = "${LOCAL_API_KEY}"
        v.update(service="vllm-sub", gpus=[2], compose={"volumes": ["${MODELS_DIR}/gguf:/models:ro"]})
        v.pop("compose_replace", None)
    return v


def clean(v):
    v = copy.deepcopy(v)
    for k in ("smoke", "only_if_failed"):
        v.pop(k, None)
    return v


# deferred variants (docs/bench-plan.md), screened with pass-1 workloads
def extras():
    g = {"g0": [p1.llama_sub(32 * K, 4, slot="strata"), p1.llama_sub(32 * K, 3, kv="f16", slot="strata")],
         "g1": [p1.llama_sub(16 * K, 8, slot="llama-guard"), p1.llama_sub(32 * K, 2, slot="llama-guard")],
         "g2": [p1.vllm_sub(32 * K, 2, kv="auto", note="16-bit KV"), p1.vllm_sub(16 * K, 6, novis=True),
                p1.vllm_sub(32 * K, 3, novis=True)]}
    g["g0"][0][0]["smoke"] = g["g1"][0][0]["smoke"] = g["g2"][0][0]["smoke"] = True
    return g


def main():
    sel = tomllib.loads((HERE / "p6-selection.toml").read_text())
    p3_files = sorted(HERE.glob("p6-*.toml"))
    if any("FILL" in x for x in sel.get("maker", []) + sel.get("sub", [])):
        stale = [f.name for f in p3_files if f.name != "p6-selection.toml"] + (
            ["p6-groups.json"] if (HERE / "p6-groups.json").exists() else [])
        if stale:
            print(f"FAIL: p6-selection.toml holds placeholders but {', '.join(stale)} exist", file=sys.stderr)
            return 1
        print("ok: pass 6 not chosen yet (p6-selection.toml)")
        return 0
    known = all_p1()
    missing = [i for i in sel["maker"] + sel["sub"] if i not in known]
    if missing:
        print(f"FAIL: not pass-1 variants: {', '.join(missing)}", file=sys.stderr)
        return 1
    maker2 = [(clean(known[i][0]), "pass 6") for i in sel["maker"] if not i.startswith("maker3gpu")]
    maker3 = [(clean(known[i][0]), "pass 6, exclusive") for i in sel["maker"] if i.startswith("maker3gpu")]
    sub = [(clean(reslot_to_sub(known[i][0])), "pass 6") for i in sel["sub"]]
    for lst in (maker2, maker3, sub):
        if lst:
            lst[0][0]["smoke"] = True
    p1.CHECK = CHECK
    p1.STALE.clear()
    write = p1.write
    files = []
    if maker2:
        files.append(write("p6-maker", "pass 6, maker (Strata #1+#2)", [0, 1], 2400, maker2,
                           [dict(DEPTH), dict(RAMP), loops("turnstone-maker"), dict(SUSTAINED), answerlen(32 * K)]))
    if sub:
        files.append(write("p6-sub", "pass 6, sub-maker (#3)", [2], 1800, sub,
                           [dict(DEPTH), dict(RAMP), loops("turnstone-sub"), dict(SUSTAINED), answerlen(32 * K)]))
    if sel.get("extras", True):
        for gname, vs in extras().items():
            files.append(write(f"p6-extra-{gname}", f"pass 6, deferred variants ({vs[0][0]['service']} slot)",
                               vs[0][0]["gpus"] if vs[0][0].get("gpus") else [2], 1800, vs,
                               [dict(p1.DEPTH), dict(p1.RAMP)], on_end="stop" if gname == "g0" else None))
    if maker3:
        files.append(write("p6-exclusive", "pass 6, exclusive (Strata on 3 cards)", [0, 1, 2], 2400, maker3,
                           [dict(DEPTH), dict(RAMP), loops("turnstone-maker"), dict(SUSTAINED)]))
    # the run order for ops/bench.yml (-e bench_run=pass3 | smoke-pass3): only the plans that exist
    names = {f.stem for f in files}
    groups = [[n for n in ("p6-maker", "p6-sub") if n in names],
              (["stop:strata,llama-guard"] + [n for n in ("p6-extra-g0", "p6-extra-g1", "p6-extra-g2") if n in names])
              if any(n.startswith("p6-extra") for n in names) else [],
              [n for n in ("p6-exclusive",) if n in names]]
    gtext = json.dumps([g for g in groups if g]) + "\n"
    gfile = HERE / "p6-groups.json"
    if CHECK:
        if not gfile.is_file() or gfile.read_text() != gtext:
            p1.STALE.append(gfile.name)
    else:
        gfile.write_text(gtext)
    if CHECK:
        if p1.STALE:
            print(f"FAIL: {', '.join(p1.STALE)} differ from gen_phase6.py — run it and commit", file=sys.stderr)
            return 1
        print("ok: bench pass-4 plans are generated from gen_phase6.py")
        return 0
    for f in files:
        print(f"wrote {f.relative_to(HERE.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
