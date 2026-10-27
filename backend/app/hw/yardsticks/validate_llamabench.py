"""llama-bench (CUDA) from the GPU database against the llama.cpp engine model (roadmap step 4). Validation only:
nothing here sets a value. Rows are skipped, with the reason, where the GPU is not in the spec table or the model's
architecture is not known.

  python3 -m app.hw.yardsticks.validate_llamabench"""
from __future__ import annotations

import statistics

from ...engine.llamacpp import cuda
from ...engine.model import from_config
from ..gpu import add_gpu
from ..machine import Machine
from ..primitives import Router
from .gpudb import MODEL_ID, load, val

MODELS = {"Llama 2 7B": "llama-2-7b", "llama 1B": "llama-3.2-1b", "llama 8B": "llama-3.1-8b", "gemma 2B": "gemma-2b"}


def rows() -> list[dict]:
    out = []
    for e in load()["entries"]:
        if e["family"] != "llm_inference":
            continue
        o, s = e["operation"], e["software"]
        meas = e["measurements"][0]
        r = dict(id=e["id"], source=e["source_id"], gpu=val(e["gpu"]["model"]), count=val(e["gpu"].get("count")) or 1,
                 model=val(o.get("model")), quant=val(o.get("quant")), test=val(o.get("test")),
                 backend=val(s.get("backend")), measured=meas["value"], unit=meas["unit"])
        why = None
        if r["backend"] != "CUDA":
            why = f"backend {r['backend']} not modelled yet"
        elif r["count"] != 1:
            why = "several GPUs not modelled yet"
        elif r["gpu"] not in MODEL_ID:
            why = "GPU not in the spec table"
        elif r["model"] not in MODELS:
            why = "model architecture not known"
        elif (r["quant"] or "").lower() != "q4_0":
            why = f"quant {r['quant']} not modelled yet"
        r["skip"] = why
        out.append(r)
    return out


def machine(model_id: str) -> Machine:
    m = Machine(); m.add(Router("host"))
    add_gpu(m, "gpu0", model_id, "host", slot_gen=4, slot_lanes=16)
    return m


def compare() -> list[dict]:
    res = []
    for r in rows():
        if r["skip"]:
            continue
        mdl = from_config(MODELS[r["model"]], "q4_0")
        sim = cuda.llama_bench(machine(MODEL_ID[r["gpu"]]), "gpu0", mdl, r["test"])
        res.append(dict(r, model_tok_s=sim["tokens_per_s"], error=sim["tokens_per_s"] / r["measured"] - 1,
                        share=sim["share"]))
    return res


def main() -> None:
    res = compare()
    print(f"{'gpu':10} {'model':11} {'test':7} {'measured':>9} {'model':>9} {'error':>7}  time shares")
    for r in sorted(res, key=lambda r: (r["test"][:2], r["gpu"], r["model"])):
        sh = " ".join(f"{k} {v:.0%}" for k, v in sorted(r["share"].items(), key=lambda kv: -kv[1]))
        print(f"{r['gpu']:10} {r['model']:11} {r['test']:7} {r['measured']:9.1f} {r['model_tok_s']:9.1f} {r['error']:+7.1%}  {sh}")
    for kind in ("pp", "tg"):
        e = [r["error"] for r in res if r["test"].startswith(kind)]
        if e:
            print(f"{kind}: n={len(e)} median {statistics.median(e):+.1%}, range {min(e):+.1%} .. {max(e):+.1%}")
    skipped = [r for r in rows() if r["skip"]]
    reasons: dict[str, int] = {}
    for r in skipped:
        reasons[r["skip"]] = reasons.get(r["skip"], 0) + 1
    print("skipped:", ", ".join(f"{v} x {k}" for k, v in reasons.items()))


if __name__ == "__main__":
    main()
