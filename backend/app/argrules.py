"""Rules between an engine's arguments, its hardware and its model.

Each rule can disable an argument (it has no effect in this setup; the UI greys it out and says why) or force
a value (the engine itself would override it; the UI shows the forced value and the model uses it).
Sources: Strata docs BATCHING.md / SECOND_GPU.md / MULTI_GPU.md (0.1.43.1, 0.2.2), vLLM engine arguments
(0.31), llama.cpp server flags (b11552).

evaluate(engine, version, args, n_gpus, gpu_specs, model) -> {arg: {"disabled": bool, "force": value|None, "reason": str}}
"""
from __future__ import annotations


def _state(states: dict, name: str, *, disabled: bool = False, force=None, reason: str = ""):
    st = states.setdefault(name, {"disabled": False, "force": None, "reason": ""})
    if disabled:
        st["disabled"] = True
    if force is not None:
        st["force"] = force
    if reason and reason not in st["reason"]:
        st["reason"] = (st["reason"] + "; " if st["reason"] else "") + reason


def _min_compute_capability(gpu_specs: list[dict]) -> float:
    """Rough CUDA compute capability from vendor and year: Ampere 8.6, Ada 8.9, Hopper 9.0, Blackwell 10+."""
    caps = []
    for g in gpu_specs:
        if g["vendor"] != "NVIDIA":
            caps.append(0.0)
            continue
        y = g["year"]
        caps.append(10.0 if y >= 2025 else 9.0 if g["id"].startswith("h100") or g["id"].startswith("h200") else 8.9 if y >= 2022 else 8.6 if y >= 2020 else 7.5 if y >= 2018 else 6.1)
    return min(caps) if caps else 0.0


def evaluate(engine: str, version: str, args: dict, n_gpus: int, gpu_specs: list[dict], model: dict | None) -> dict:
    s: dict = {}
    a = args
    model = model or {}
    if engine == "strata":
        packs = model.get("quants")
        if packs and str(a.get("quantization", "")) not in packs:
            _state(s, "quantization", force=packs[0], reason=f"this model comes as {', '.join(packs)}")
        if model.get("ssd_stream") and str(a.get("quantization")) == "ud_q4_k_xl":
            _state(s, "multi_gpu", disabled=True, force="layer-split", reason="UD-Q4_K_XL runs on one NVIDIA GPU")
        multi = n_gpus >= 2
        mode = str(a.get("multi_gpu", "layer-split"))
        if not multi:
            for name in ("multi_gpu", "batch_groups", "pipeline_windows", "trim_stage_weights"):
                _state(s, name, disabled=True, reason="needs two or more GPUs")
        if multi and mode == "peer-device":
            _state(s, "batch_groups", disabled=True, reason="a layer split only; the peer tier runs the model on the first card")
            _state(s, "pipeline_windows", disabled=True, reason="does not combine with --peer-device")
            _state(s, "trim_stage_weights", disabled=True, reason="only with an explicit --layer-split")
        if not model.get("mtp", False):
            _state(s, "mtp", disabled=True, force=False, reason="this model has no MTP head")
            _state(s, "batch_mtp", disabled=True, force=(False if version == "0.1.43.1" else "off"), reason="this model has no MTP head")
        elif not a.get("mtp", True):
            _state(s, "batch_mtp", disabled=True, force=(False if version == "0.1.43.1" else "off"), reason="needs --mtp")
        batch = int(a.get("batch", 1) or 1)
        groups = str(a.get("batch_groups", "auto"))
        if batch <= 1:
            _state(s, "batch_groups", disabled=True, reason="one slot: nothing to pipeline")
            _state(s, "batch_mtp", disabled=True, reason="one slot: the solo drafter is used")
        else:
            _state(s, "pipeline_windows", disabled=True, force="0", reason="off with --batch above 1 (solo path only)")
            if multi and mode != "peer-device" and groups not in ("auto", "1") and batch % int(groups) != 0:
                _state(s, "batch_groups", force="auto", reason=f"--batch-groups must divide --batch ({batch}); using auto")
            pipelined = multi and mode != "peer-device" and groups != "1"
            if pipelined:
                _state(s, "batch_mtp", disabled=True, force=(False if version == "0.1.43.1" else "off"),
                       reason="the pipelined path runs no slot drafts; set --batch-groups 1 for serial windows with --batch-mtp")
        pm, ps = int(a.get("parking_mib", 0) or 0), int(a.get("parking_slots", 0) or 0)
        # one full parked conversation: its K/V at the context window (int8) plus the recurrent state
        from .layout import conversation_mib
        conv_mib = conversation_mib(model, int(a.get("context", 32768) or 32768))
        if (pm > 0) != (ps > 0) and conv_mib > 0:
            if pm > 0:
                fit = max(1, int(pm // conv_mib))
                _state(s, "parking_slots", force=fit, reason=f"parking RAM without a count does nothing; {fit} full conversations fit in it")
            else:
                need = int(round(ps * conv_mib / 1024 + 0.5)) * 1024
                _state(s, "parking_mib", force=need, reason=f"{ps} parked conversations need about {need} MiB at this context")
        ctx = int(a.get("context", 32768) or 32768)
        if int(a.get("kv_resident", 0) or 0) > ctx:
            _state(s, "kv_resident", force=0, reason="above the context window: all cells resident anyway")
        if int(a.get("prefill_chunk", 8192) or 8192) > ctx:
            _state(s, "prefill_chunk", force=ctx, reason="a chunk larger than the context is read as one chunk")
        if version == "0.2.2" and not multi:
            pass
    elif engine == "vllm":
        tp = int(a.get("tensor_parallel", 1) or 1)
        pp = int(a.get("pipeline_parallel", 1) or 1)
        if n_gpus >= 1 and tp * pp != n_gpus:
            if n_gpus in (1, 2, 4, 8) and pp == 1:
                _state(s, "tensor_parallel", force=str(n_gpus), reason=f"tensor x pipeline parallel must equal the {n_gpus} assigned GPUs")
            elif n_gpus % pp == 0 and str(n_gpus // pp) in ("1", "2", "4", "8"):
                _state(s, "tensor_parallel", force=str(n_gpus // pp), reason=f"tensor x pipeline parallel must equal the {n_gpus} assigned GPUs")
            else:
                _state(s, "pipeline_parallel", force="1", reason=f"{n_gpus} GPUs cannot be split {tp} x {pp}")
        if n_gpus <= 1:
            _state(s, "pipeline_parallel", disabled=True, force="1", reason="needs two or more GPUs")
        if n_gpus > 1 and len({g["id"] for g in gpu_specs}) > 1 and pp == 1:
            _state(s, "tensor_parallel", reason="tensor parallel needs identical GPUs; mixed cards work only as pipeline stages")
        cc = _min_compute_capability(gpu_specs)
        if gpu_specs and cc < 8.9:
            if str(a.get("quantization")) == "fp8":
                _state(s, "quantization", force="awq4", reason="FP8 weights need Ada Lovelace / Hopper or newer (compute capability 8.9+)")
            if str(a.get("kv_cache_dtype")) == "fp8":
                _state(s, "kv_cache_dtype", force="fp16", reason="FP8 KV cache needs compute capability 8.9+")
            _state(s, "quantization", reason="FP8 is unavailable on these cards")
        if not model.get("mtp", False):
            _state(s, "speculative", disabled=True, force=False, reason="this model has no MTP / draft head")
        mml = int(a.get("max_model_len", 32768) or 32768)
        if int(a.get("max_num_batched_tokens", 8192) or 8192) > mml:
            _state(s, "max_num_batched_tokens", force=mml, reason="cannot exceed max_model_len")
        if model.get("max_context") and mml > int(model["max_context"]):
            _state(s, "max_model_len", force=int(model["max_context"]), reason=f"the model's context ends at {model['max_context']}")
    elif engine == "llama.cpp":
        layers = str(a.get("n_gpu_layers", "all"))
        if n_gpus == 0:
            _state(s, "n_gpu_layers", disabled=True, force="0", reason="no GPU assigned")
            for name in ("split_mode", "flash_attn"):
                _state(s, name, disabled=True, reason="no GPU assigned")
        if n_gpus < 2:
            _state(s, "split_mode", disabled=True, reason="needs two or more GPUs")
        if layers == "all":
            _state(s, "threads", disabled=True, reason="every layer is on the GPU; threads idle")
        if layers != "all" and model.get("experts", 0):
            pass
        if not model.get("experts", 0):
            _state(s, "cpu_moe", disabled=True, force=False, reason="a dense model has no experts to move")
        if str(a.get("cache_type", "f16")) != "f16" and not a.get("flash_attn", True):
            _state(s, "cache_type", force="f16", reason="a quantised KV cache needs flash attention on")
        if not model.get("mtp", False):
            _state(s, "mtp", disabled=True, force=False, reason="this model has no MTP head")
        ub, b = int(a.get("ubatch", 512) or 512), int(a.get("batch", 2048) or 2048)
        if ub > b:
            _state(s, "ubatch", force=b, reason="--ubatch cannot exceed --batch")
        if model.get("max_context") and int(a.get("ctx_size", 32768) or 32768) > int(model["max_context"]):
            _state(s, "ctx_size", force=int(model["max_context"]), reason=f"the model's context ends at {model['max_context']}")
    return s


def apply(states: dict, args: dict) -> list[str]:
    """Writes forced values into args; returns one note per change."""
    notes = []
    for name, st in states.items():
        if st["force"] is not None and args.get(name) != st["force"] and name in args:
            notes.append(f"{name}: {args[name]!r} → {st['force']!r} ({st['reason']})")
            args[name] = st["force"]
        elif st["force"] is not None and name not in args:
            args[name] = st["force"]
    return notes
