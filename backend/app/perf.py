"""Derive an engine's budgets (prompt reading rate, writing rate per number of parallel
requests, memory fit) from the machine, the model and the engine arguments.

The model is deliberately simple and bandwidth-centred:
- writing a token reads the weights it touches once per window; the time is bytes / effective
  bandwidth of wherever those bytes live (VRAM, host RAM through the CPU, or PCIe);
- reading a prompt is compute-bound on the GPU, plus the streaming of experts that are not
  resident;
- parallel rows share the dense weights but, for MoE models, hardly share experts.
Constants were calibrated on one measured box (3 x RTX 3090, Qwen3.8-Flash-Next IQ3_S on
Strata, Qwen3.8-27B on vLLM/llama.cpp); see tests/test_perf.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog.gpus import compute_efficiency, kernel_efficiency, layer_overhead_factor, power_factors
from .catalog.models import ENGINES, MODEL_BY_ID, QUANTS
from . import argrules, layout as layout_mod

GPU_BW_EFF = 0.75          # share of spec bandwidth a decode kernel achieves (27B AWQ on a 3090: 39 tok/s; 8B Q4_0: 167)
GPU_FLOPS_EFF = 0.50       # share of FP16 tensor peak a prefill achieves (dense models; llama.cpp pp512 8B: ~0.68 on 3090/4090)
STRATA_FLOPS_EFF = 0.41    # Strata's MoE prompt path (box at a 250 W cap, 4090 / 5090 reports uncapped)
HIT_SKEW = 4.0             # routing is skewed: hit = 1 - (1 - held)^HIT_SKEW
CPU_EXPERT_GBS_PER_CORE = 3.0   # fallback when a CPU has no ISA data
CPU_EXPERT_RAM_SHARE = 0.60     # expert compute uses at most this share of the memory pool bandwidth
PCIE_SMALL_COPY_EFF = 0.5       # expert-miss copies are small; they reach about half the link rate
LAYER_OVERHEAD_MS = 0.02   # kernel launches per layer per step, dense engines
STRATA_LAYER_MS = 0.13     # per layer per window: launches, DeltaNet and sparse-attention kernels (calibrated)
HANDOFF_MS = 0.6           # layer-split hand-off per stage per window
MTP_TOKENS_PER_WINDOW = 2.8
BATCH_MTP_TOKENS_PER_ROW = 1.66
OVERHEAD_GB = 1.2          # CUDA context, buffers, per card


@dataclass
class EnginePerf:
    engine_id: str
    kind: str                      # strata | batch | hosted | reranker
    slots: int = 1
    parking: int = 0
    prefill_rate: float = 0.0      # tokens/s, shared
    prefill_quad: float = 0.0      # s per (new token x context token): attention over the context
    decode_total: dict[int, float] = field(default_factory=dict)   # n parallel -> total tok/s
    speed_per_request: dict[int, float] = field(default_factory=dict)
    overhead_s: float = 0.5
    first_token_s: float = 0.0
    max_parallel: int = 1
    cache_convs: int = 1
    window: int = 32768
    memory: dict = field(default_factory=dict)
    layout: dict | None = None        # static memory/link layout (Strata): layout.strata_layout
    arg_states: dict = field(default_factory=dict)   # from argrules: disabled / forced arguments with reasons
    watts_busy: float = 0.0        # W drawn by this engine's GPUs (at their caps) and CPU share while it works
    watts_idle: float = 0.0        # W those parts draw when this engine is idle
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    limits: dict = field(default_factory=dict)   # resource -> share of a window's time
    resources: dict = field(default_factory=dict)  # gpus, cpu, cores
    hosted_prices: dict = field(default_factory=dict)

    def public(self) -> dict:
        d = self.__dict__.copy()
        d["decode_total"] = {str(k): round(v, 1) for k, v in self.decode_total.items()}
        d["speed_per_request"] = {str(k): round(v, 1) for k, v in self.speed_per_request.items()}
        d["prefill_rate"] = round(self.prefill_rate, 0)
        return d


def _bpw(args, default):
    return QUANTS.get(str(args.get("quantization", default)), QUANTS[default])


def _gpus(spec, machine_info):
    out = []
    for gid in spec.gpus:
        d = machine_info["_devices"].get(gid)
        if d is None or d["kind"] != "gpu":
            raise ValueError(f"{spec.id}: '{gid}' is not a GPU of this machine")
        out.append(d)
    return out


def _cpu(spec, machine_info):
    c = machine_info["cpus"].get(spec.cpu)
    if c is None:
        raise ValueError(f"{spec.id}: CPU {spec.cpu + 1} is not enabled")
    return c


def _nvlinked(gpus, machine_info):
    ids = [g["id"] for g in gpus]
    if len(ids) < 2:
        return False
    pairs = {frozenset(p) for p in machine_info["nvlinks"]}
    return all(frozenset((a, b)) in pairs for a, b in zip(ids, ids[1:]))


def _finite(v, default):
    """A usable number: None, NaN, inf, text and booleans fall back to the default."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if isinstance(v, bool) or f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def _clamp_args(p, args, schema):
    """Keep every numeric argument inside its schema range (the UI's min/max are not enforced on typing).
    Values that are not numbers fall back to the default; changed values are reported in the notes."""
    for a in schema:
        if a["type"] not in ("int", "float") or a["name"] not in args:
            continue
        raw = args[a["name"]]
        v = _finite(raw, a["default"])
        lo, hi = a.get("min", 0), a.get("max", 10_000_000)
        c = min(max(v, lo), hi)
        if a["type"] == "int":
            c = int(round(c))
        if c != raw:
            p.notes.append(f"{a['name']}: {raw!r} is outside {lo}–{hi} or not a number; using {c}")
        args[a["name"]] = c


import contextvars

_SPEC: contextvars.ContextVar = contextvars.ContextVar("spec")


def derive(spec, machine_info) -> EnginePerf:
    _SPEC.set(spec)
    eng = ENGINES.get(spec.engine)
    p = EnginePerf(engine_id=spec.id, kind="batch")
    if eng is None:
        p.errors.append(f"unknown engine '{spec.engine}'")
        return p
    if spec.version not in eng["versions"]:
        p.errors.append(f"{eng['name']} has no version '{spec.version}'")
        return p
    model = MODEL_BY_ID.get(spec.model_id)
    if model is None:
        p.errors.append(f"unknown model '{spec.model_id}'")
        return p
    if spec.model_id not in eng["models"]:
        p.errors.append(f"{eng['name']} cannot serve {model['name']}")
        return p
    schema = eng["versions"][spec.version]["args"]
    args = {a["name"]: a["default"] for a in schema}
    args.update(spec.args or {})
    _clamp_args(p, args, schema)
    _gpu_specs = []
    for gid in spec.gpus:
        d = machine_info["_devices"].get(gid) if machine_info else None
        if d and d["kind"] == "gpu":
            _gpu_specs.append(d["spec"])
    p.arg_states = argrules.evaluate(spec.engine, spec.version, args, len(spec.gpus), _gpu_specs, model)
    p.notes += argrules.apply(p.arg_states, args)
    args["_version"] = spec.version
    args["_cores"] = max(1, min(4096, int(_finite(spec.cpu_cores, 1))))
    try:
        if spec.engine == "hosted-api":
            return _apply_benchmarks(_hosted(p, args, model), spec)
        gpus = _gpus(spec, machine_info)
        cpu = _cpu(spec, machine_info)
        if model.get("reranker"):
            r = _reranker(p, args, model, gpus, cpu)
        elif spec.engine == "strata":
            r = _strata(p, args, model, gpus, cpu, machine_info)
        elif spec.engine == "vllm":
            r = _vllm(p, args, model, gpus, cpu, machine_info)
        else:
            r = _llama(p, args, model, gpus, cpu, machine_info)
        r = _apply_benchmarks(r, spec)
        _watts(r, gpus, cpu, machine_info)
        return r
    except ValueError as e:
        p.errors.append(str(e))
        return p
    except Exception as e:  # noqa: BLE001 - one broken engine must not blank the others
        p.errors.append(f"internal error while deriving this engine ({type(e).__name__}: {e}); please report it with the settings used")
        return p


def _watts(p: EnginePerf, gpus, cpu, info) -> None:
    """Power while this engine works: its GPUs at their caps (SoCs: their TDP share), plus the CPU's
    TDP scaled by the cores it uses; idle: the same parts at idle draw."""
    busy = idle = 0.0
    for g in gpus:
        cap = g.get("power_w") or g["spec"]["tdp_w"]
        busy += cap
        idle += g.get("idle_w", 0.07 * g["spec"]["tdp_w"])
    cores = int(p.resources.get("cores", 0) or 0)
    busy += cpu.get("tdp_w", 0) * min(1.0, cores / max(1, cpu["cores"])) * 0.8
    p.watts_busy = round(busy, 1)
    p.watts_idle = round(idle, 1)


def _apply_benchmarks(p: EnginePerf, spec) -> EnginePerf:
    """Measured values the user entered replace the derived ones. A measured writing rate for
    one count of parallel requests rescales the whole table; measured counts stay as given."""
    b = {}
    for k, v in (spec.benchmarks or {}).items():
        f = _finite(v, 0.0)
        if f > 0:
            b[k] = f
    if not b or p.errors:
        return p
    measured = {int(k.split("_")[1]): v for k, v in b.items() if k.startswith("decode_") and k.split("_")[1].isdigit()}
    if measured and p.decode_total:
        n0 = min(measured)
        if n0 in p.decode_total and p.decode_total[n0] > 0:
            f = measured[n0] / p.decode_total[n0]
            p.decode_total = {n: v * f for n, v in p.decode_total.items()}
        for n, v in measured.items():
            if n <= max(p.decode_total) or n <= p.slots:
                p.decode_total[n] = v
        p.speed_per_request = {n: v / n for n, v in p.decode_total.items()}
        p.notes.append("writing rates: measured " + ", ".join(f"{n} → {v:.0f} tok/s" for n, v in sorted(measured.items())))
    if "prefill" in b:
        p.prefill_rate = b["prefill"]
        p.notes.append(f"prompt reading: measured {b['prefill']:.0f} tok/s")
    if "kv_room_tokens" in b and p.kind == "batch":
        p.memory["kv_room_tokens"] = int(b["kv_room_tokens"])
        p.notes.append("KV room: measured")
    p.memory["measured"] = True
    return p


def _hosted(p, args, model):
    p.kind = "hosted"
    p.max_parallel = int(args["max_parallel"])
    p.slots = p.max_parallel
    p.prefill_rate = float(args["prefill_tok_s"])
    p.first_token_s = args["first_token_ms"] / 1000
    s = float(args["tok_s"])
    p.speed_per_request = {n: s for n in (1, 2, 4, 8, 16)}
    p.decode_total = {n: s * n for n in (1, 2, 4, 8, 16)}
    p.cache_convs = 10_000
    p.window = model["max_context"]
    p.hosted_prices = dict(price_in=args["price_in"], price_cached=args["price_cached"], price_out=args["price_out"])
    p.notes.append("hosted: speeds and prices as entered, caching assumed across turns of one conversation")
    return p


def _reranker(p, args, model, gpus, cpu):
    p.kind = "reranker"
    tokens_per_pair = 250                        # query + tool description
    cores = max(1, min(cpu["cores"], int(args["_cores"])))
    if gpus:
        flops = sum(_gpu_flops(g) for g in gpus) * 0.15
    else:
        isa_flops = {"sse": 8, "avx": 16, "avx2": 32, "avx512": 64, "amx": 64}.get(cpu.get("isa", "avx2"), 32)
        flops = cores * isa_flops * cpu.get("all_core_ghz", 3.5) * 1e9 * 0.6
        p.notes.append("reranker on the CPU: slow and limited to short pairs")
    pairs_s = flops / (2 * model["params_b"] * 1e9 * tokens_per_pair)
    p.prefill_rate = pairs_s * tokens_per_pair
    p.slots = int(args.get("parallel", 1))
    p.decode_total = {1: pairs_s}
    p.speed_per_request = {1: pairs_s}
    p.resources = dict(gpus=[g["id"] for g in gpus], cpu=cpu["name"], cores=0 if gpus else cores)
    p.notes.append(f"~{pairs_s:.1f} pairs/s; a tool search with 40 candidates takes ~{40 / pairs_s:.1f} s")
    return p


def _memory_report(gpus, per_card_gb):
    rep = {}
    for g, used in zip(gpus, per_card_gb):
        rep[g["id"]] = dict(vram_gb=g["spec"]["vram_gb"], used_gb=round(used, 1), free_gb=round(g["spec"]["vram_gb"] - used, 1))
    return rep


def _strata(p, args, model, gpus, cpu, info):
    p.kind = "strata"
    if not gpus:
        raise ValueError("Strata needs at least one GPU")
    if not model["experts"]:
        raise ValueError("Strata serves MoE models with expert offload only")
    if str(args.get("multi_gpu", "layer-split")) == "peer-device" and len(gpus) > 1:
        # the peer tier: the first card runs every layer, the other cards hold experts behind a PCIe hop. Dense
        # work and KV on card 1; the extra cards add cache (hit rate) but their experts arrive at link speed.
        peer_cache_gb = sum(g["spec"]["vram_gb"] * 0.9 for g in gpus[1:])
        p.notes.append(f"peer-device: {len(gpus) - 1} card(s) as expert tier, {peer_cache_gb:.0f} GB behind the PCIe link")
        gpus = gpus[:1]
        args["_peer_cache_gb"] = peer_cache_gb
    bpw = _bpw(args, "iq3_s")
    slots = int(args["batch"])
    ctx = int(args["context"])
    dense_gb = model["params_b"] * (1 - model["expert_share"]) * bpw / 8
    experts_gb = model["params_b"] * model["expert_share"] * bpw / 8
    active_expert_gb = max(0.0, model["active_b"] - model["params_b"] * (1 - model["expert_share"])) * bpw / 8
    active_dense_gb = dense_gb
    kv_resident = int(args.get("kv_resident", 0) or 0)
    kv_cells = ctx if kv_resident == 0 else min(ctx, kv_resident)
    kv_gb_per_slot = kv_cells * model["kv_kb_per_token"] / 2 / 1e6 + model["recurrent_state_mb"] / 1000   # int8 KV
    n_cards = len(gpus)
    trim = bool(args.get("trim_stage_weights", True))
    per_card_dense = dense_gb / n_cards if trim and n_cards > 1 else dense_gb
    cache_per_card = []
    used = []
    for g in gpus:
        u = per_card_dense + OVERHEAD_GB + kv_gb_per_slot * (slots + 1) / (n_cards if n_cards > 1 else 1) * (1 if n_cards > 1 else 1)
        if model["vision"]:
            u += 0.9
        cache = g["spec"]["vram_gb"] - u
        cache_per_card.append(max(0.0, cache))
        used.append(u + max(0.0, cache))
    cache_total = sum(cache_per_card)
    if min(cache_per_card) <= 0:
        p.errors.append("the dense weights, KV and slots do not fit; no room for an expert cache")
    ec = int(args.get("expert_cache_mib", 0) or 0)
    if ec > 0:
        cache_total = min(cache_total, ec / 1024 * len(gpus))
    cache_total += float(args.get("_peer_cache_gb", 0.0)) * 0.5     # peer experts: counted at half, they stream over PCIe
    held = min(1.0, cache_total / experts_gb)
    hit = 1 - (1 - held) ** HIT_SKEW
    # bandwidths
    gpu_bw = [_gpu_bw(g) for g in gpus]
    ram_bw = cpu["ram_bw_gbs"]
    cores = max(1, min(cpu["cores"], int(args["_cores"])))
    pw = int(args.get("pool_workers", 0) or 0)
    cpu_stream = _cpu_stream(cpu, min(cores, pw) if pw else cores)
    if str(args.get("aux_cpus", "auto")) == "off":
        cpu_stream *= 0.85          # 0.2.2: without the auxiliary CPU helpers the expert path loses its overlap
    pcie_bw = min(g["link_bw_gbs"] for g in gpus) * (0.7 if cpu.get("os") == "windows" else 1.0)
    pcie_miss_bw = pcie_bw * PCIE_SMALL_COPY_EFF        # expert misses are many small copies
    miss_bw = max(cpu_stream, pcie_miss_bw) if args.get("pcie_frac", "auto") == "auto" else (
        pcie_miss_bw if str(args["pcie_frac"]) == "1" else cpu_stream)
    mtp = bool(args.get("mtp", True)) and model["mtp"]
    groups = str(args.get("batch_groups", "auto"))
    vendor_nv = all(g["spec"]["vendor"] == "NVIDIA" for g in gpus)
    if groups == "auto":
        if n_cards == 1:
            ngroups = 1
        elif n_cards == 2 and vendor_nv and str(args["_version"]).startswith("0.2"):
            ngroups = 1
        else:
            ngroups = max([k for k in range(1, n_cards + 1) if slots % k == 0] or [1])
    else:
        ngroups = max(1, min(int(groups), slots))
    pipelined = ngroups > 1 and n_cards > 1
    bm = args.get("batch_mtp", False)
    batch_mtp = (bm is True or str(bm) == "on" or (str(bm) == "auto" and n_cards == 1)) and not pipelined and mtp

    # KV streaming at a typical conversation depth (16K): what a slot re-reads over the link per window. The layout
    # shows the full-context worst case separately.
    typical_depth = min(ctx, 16384)
    kv_streamed_gb = max(0.0, (typical_depth - kv_cells) * model["kv_kb_per_token"] / 2 / 1e6)
    solo_rows_hint = 4 if (bool(args.get("mtp", True)) and model["mtp"]) else 1
    if kv_cells < ctx:
        p.notes.append(f"KV streaming: beyond {kv_cells // 1024}K tokens a slot's K/V lives in host RAM and crosses the link every window")

    def window_time(rows: int, drafts: bool) -> float:
        """Seconds for one verify window over all cards with `rows` rows."""
        t = 0.0
        for bw in gpu_bw:
            t += (active_dense_gb / n_cards) / bw
            t += (active_expert_gb / n_cards) * rows * hit / bw
        t += active_expert_gb * rows * (1 - hit) / miss_bw
        if kv_streamed_gb > 0:
            # KV streaming: every window re-reads each slot's non-resident K/V over the link; this is what makes a
            # 128K context fit in VRAM and also what caps its speed
            t += kv_streamed_gb * max(1, rows // solo_rows_hint) / pcie_bw
        t += model["layers"] * STRATA_LAYER_MS / 1000 * (1 + 0.15 * (rows - 1))
        t += HANDOFF_MS / 1000 * (n_cards - 1)
        if drafts:
            t += 0.0015            # drafter on the last card
        return t

    solo_rows = 4 if mtp else 1
    d = {}
    d[1] = (MTP_TOKENS_PER_WINDOW if mtp else 1.0) / window_time(solo_rows, mtp)
    for n in range(2, slots + 1):
        if pipelined:
            rows = max(1, n // ngroups)
            t = window_time(rows, False)
            # groups overlap across cards: throughput of one card-time per group instead of the sum
            per_group = t / n_cards * 1.6   # stage imbalance, hand-offs and small groups (measured ~60 % overhead)
            d[n] = n / (per_group * ngroups) if ngroups <= n_cards else n / t
            d[n] = min(d[n], n / (t / n_cards))
        elif batch_mtp:
            d[n] = n * BATCH_MTP_TOKENS_PER_ROW / window_time(2 * n, True)
        else:
            d[n] = n / window_time(n, False)
    p.decode_total = d
    p.speed_per_request = {n: v / n for n, v in d.items()}
    p.slots = slots
    p.parking = int(args.get("parking_slots", 0) or 0)
    if p.parking and int(args.get("parking_mib", 0) or 0) == 0:
        p.notes.append("parking slots set without a RAM budget: parking stays off")
        p.parking = 0
    chunk = int(args.get("prefill_chunk", 8192))
    per_card_tps = [_gpu_flops(g) * STRATA_FLOPS_EFF / (2 * model["active_b"] * 1e9) for g in gpus]
    # stages overlap chunks only partly (measured 1 : 1.5 : 1.8 on 1 / 2 / 3 cards)
    compute_tps = min(per_card_tps) * n_cards / (1 + 0.33 * (n_cards - 1))
    stream_s = experts_gb * (1 - held) / pcie_bw          # non-resident experts stream once per chunk
    p.prefill_rate = chunk / (chunk / compute_tps + stream_s)
    p.prefill_quad = _attention_quad(model, sum(_gpu_flops(g) for g in gpus) * STRATA_FLOPS_EFF)
    p.overhead_s = 0.7
    p.window = ctx
    p.cache_convs = 1 + p.parking + (slots if slots > 1 else 0)
    p.memory = _memory_report(gpus, used)
    try:
        p.layout = layout_mod.strata_layout(spec=_SPEC.get(), args=args, model=model, gpus=gpus, cpu=cpu, info=info, dense_gb=dense_gb,
                                            experts_gb=experts_gb, kv_gb_per_slot=kv_gb_per_slot, kv_cells=kv_cells, ctx=ctx, slots=slots,
                                            trim=trim, cache_per_card=cache_per_card, held=held, hit=hit, decode_total=p.decode_total,
                                            prefill_rate=p.prefill_rate, pcie_bw=pcie_bw)
    except Exception as e:  # noqa: BLE001 - the layout is an aid; never break the derivation for it
        p.notes.append(f"layout not available: {type(e).__name__}: {e}")
    p.memory["expert_cache_gb"] = round(cache_total, 1)
    p.memory["experts_total_gb"] = round(experts_gb, 1)
    p.memory["held"] = round(held, 2)
    p.memory["hit"] = round(hit, 2)
    p.memory["host_ram_gb"] = round(experts_gb * 1.05 + int(args.get("parking_mib", 0) or 0) / 1024, 1)
    if p.memory["host_ram_gb"] > cpu["ram_gb"]:
        p.errors.append(f"needs ~{p.memory['host_ram_gb']} GB host RAM for the experts, CPU {cpu['name']} pool has {cpu['ram_gb']} GB")
    t1 = window_time(solo_rows, mtp)
    gpu_share = sum((active_dense_gb / n_cards + active_expert_gb / n_cards * solo_rows * hit) / bw for bw in gpu_bw) / t1
    miss_share = (active_expert_gb * solo_rows * (1 - hit) / miss_bw) / t1
    p.limits = {"gpu_memory_bandwidth": round(gpu_share, 2),
                ("pcie" if miss_bw == pcie_miss_bw and pcie_miss_bw > cpu_stream else "cpu_ram"): round(miss_share, 2),
                "kernel_launch": round(model["layers"] * STRATA_LAYER_MS / 1000 / t1, 2)}
    if pipelined:
        p.notes.append(f"{ngroups} pipeline groups: no drafts in batch windows, expert tier does not adapt")
    if batch_mtp:
        p.notes.append("batch-mtp: one draft per slot")
    if slots > 1 and held < 0.5:
        p.notes.append("fewer than half the experts fit beside the slots: batching will cost throughput")
    if str(args.get("pipeline_windows", "0")) != "0" and (slots > 1 or n_cards == 1):
        p.notes.append("pipeline-windows is off: needs a split and no batch slots")
    p.resources = dict(gpus=[g["id"] for g in gpus], cpu=cpu["name"], cores=cores)
    return p


def _attention_quad(model, flops):
    """Seconds per new token per context token for the attention products (QK^T and PV),
    4 * layers * hidden FLOPs, at the engine's effective FLOP rate."""
    layers = model.get("attn_layers", model["layers"]) or 1
    hidden = (model["params_b"] * 1e9 / (12 * max(1, model["layers"]))) ** 0.5
    return 4 * layers * hidden / max(1.0, flops)


def _cpu_stream(cpu, cores):
    """GB of expert weights per second the CPU side can compute: per-core rate from the vector
    ISA and clock (E-cores count 0.4), capped by a share of the memory pool bandwidth."""
    per_core = cpu.get("expert_gbs_per_core") or CPU_EXPERT_GBS_PER_CORE
    e = cpu.get("e_cores", 0)
    p_cores = max(0, cpu["cores"] - e)
    use_p = min(cores, p_cores)
    use_e = max(0, min(cores - use_p, e))
    eff = use_p + 0.4 * use_e
    return min(cpu["ram_bw_gbs"] * CPU_EXPERT_RAM_SHARE, eff * per_core)


def _flops_eff(model):
    """small dense models reach a larger share of the tensor peak (measured: 3B ~5,000 tok/s on a 3060)"""
    return min(0.78, GPU_FLOPS_EFF * (1 + 8 / max(1.0, model["active_b"])) ** 0.5)


def _gpu_bw(g, bpw: float = 4.5):
    return g["spec"]["mem_bw_gbs"] * GPU_BW_EFF * kernel_efficiency(g["spec"], bpw) * g.get("power_bw_factor", 1.0) * power_factors(g["spec"], g.get("power_w"))[0]


def _gpu_flops(g):
    return g["spec"]["fp16_tflops"] * 1e12 * compute_efficiency(g["spec"]) * g.get("power_compute_factor", 1.0) * power_factors(g["spec"], g.get("power_w"))[1]


def _layer_ms(gpus):
    return LAYER_OVERHEAD_MS * (max(layer_overhead_factor(g["spec"]) for g in gpus) if gpus else 1.0)


def _dense_bytes_gb(model, bpw):
    return model["params_b"] * bpw / 8


def _vllm(p, args, model, gpus, cpu, info):
    p.kind = "batch"
    tp = int(args["tensor_parallel"])
    pp = int(args.get("pipeline_parallel", 1) or 1)
    if len(gpus) != tp * pp:
        raise ValueError(f"tensor x pipeline parallel = {tp} x {pp} needs exactly {tp * pp} GPUs, {len(gpus)} assigned")
    if tp > 1 and not all(g["spec"]["id"] == gpus[0]["spec"]["id"] for g in gpus):
        raise ValueError("tensor parallel needs identical GPUs")
    bpw = _bpw(args, "awq4")
    weights_gb = _dense_bytes_gb(model, bpw)
    ctx = int(args["max_model_len"])
    kv_bytes = model["kv_kb_per_token"] * 1000 * (0.5 if args.get("kv_cache_dtype") == "fp8" else 1.0)
    util = float(args["gpu_memory_utilization"])
    p.memory["scheduling_policy"] = str(args.get("scheduling_policy", "fcfs"))
    vram = sum(g["spec"]["vram_gb"] for g in gpus) * util
    free_gb = vram - weights_gb - (OVERHEAD_GB + 1.3) * tp - (0.9 if model["vision"] else 0)   # CUDA graphs, activations
    if free_gb <= 0:
        p.errors.append(f"{model['name']} ({weights_gb:.1f} GB at {args['quantization']}) does not fit in "
                        f"{vram:.1f} GB usable VRAM")
        return p
    kv_tokens = free_gb * 1e9 / kv_bytes
    conc_by_kv = max(1, int(kv_tokens / (ctx * 0.6)))          # requests rarely sit at full context
    conc = min(int(args["max_num_seqs"]), conc_by_kv)
    nvl = _nvlinked(gpus, info)
    bw = sum(_gpu_bw(g, bpw) for g in gpus) * (1.0 if tp == 1 else (0.92 if nvl else 0.78))
    if pp > 1:
        bw /= pp                     # pipeline parallel: one stage works on a given token at a time; stages overlap only across requests
        p.notes.append(f"pipeline parallel {pp}: a single request sees one stage's bandwidth; throughput recovers with {pp}+ requests in flight")
    flops = sum(_gpu_flops(g) for g in gpus) * _flops_eff(model) * (0.8 if bpw < 8 else 1.0)   # dequant costs prefill (box at 250 W: 27B AWQ 1,050 tok/s)
    active_gb = model["active_b"] * bpw / 8 if model["experts"] else weights_gb
    expert_gb = max(0.0, model["active_b"] - model["params_b"] * (1 - model["expert_share"])) * bpw / 8
    spec_mult = 1.5 if args.get("speculative") and model["mtp"] else 1.0
    d = {}
    for n in (1, 2, 3, 4, 6, 8, 12, 16, 32):
        if n > conc:
            break
        kv_read = n * min(ctx, 8192) * kv_bytes / 1e9       # typical filled context, not the window
        shared = active_gb - expert_gb
        t = (shared + expert_gb * n + kv_read) / bw + model["layers"] * _layer_ms(gpus) / 1000
        t_compute = 2 * model["active_b"] * 1e9 * n / flops
        t = max(t, t_compute)
        d[n] = n * spec_mult / t
    p.decode_total = d
    p.speed_per_request = {n: v / n for n, v in d.items()}
    p.slots = conc
    p.max_parallel = conc
    chunk = int(args.get("max_num_batched_tokens", 8192) or 8192)
    p.prefill_rate = flops / (2 * model["active_b"] * 1e9) * (0.9 if tp > 1 and not nvl else 1.0) * min(1.0, 0.7 + chunk / 8192 * 0.3)
    p.prefill_quad = _attention_quad(model, flops)
    p.overhead_s = 0.3
    p.window = ctx
    p.cache_convs = max(conc, int(kv_tokens // 8192)) if args.get("enable_prefix_caching", True) else 0
    per_card = [(weights_gb / tp + OVERHEAD_GB + (free_gb / tp)) for _ in gpus]
    p.memory = _memory_report(gpus, per_card)
    p.memory.update(weights_gb=round(weights_gb, 1), kv_room_tokens=int(kv_tokens), concurrent_by_kv=conc_by_kv)
    if conc_by_kv < int(args["max_num_seqs"]):
        p.notes.append(f"KV room for ~{int(kv_tokens / 1000)}K tokens: about {conc_by_kv} requests at {ctx} context, "
                       f"not max_num_seqs={args['max_num_seqs']}")
    if tp > 1:
        p.notes.append("tensor parallel over " + ("NVLink" if nvl else "PCIe (all-reduce costs ~20 % bandwidth)"))
    t1 = active_gb / bw
    p.limits = {"gpu_memory_bandwidth": round(min(1.0, t1 / (1 / d[1])), 2), "kv_capacity": round(1 - conc / max(conc, int(args['max_num_seqs'])), 2)}
    p.resources = dict(gpus=[g["id"] for g in gpus], cpu=cpu["name"], cores=2)
    return p


def _llama(p, args, model, gpus, cpu, info):
    p.kind = "batch"
    bpw = _bpw(args, "iq3_s")
    weights_gb = _dense_bytes_gb(model, bpw)
    ctx = int(args["ctx_size"])
    slots = int(args["parallel"])
    kv_scale = {"f16": 1.0, "q8_0": 0.5, "q4_0": 0.28}[str(args.get("cache_type", "f16"))]
    kv_gb = ctx * model["kv_kb_per_token"] / 1e6 * kv_scale          # ctx_size is per slot here
    share = {"all": 1.0, "75%": 0.75, "50%": 0.5, "25%": 0.25, "0": 0.0}[str(args.get("n_gpu_layers", "all"))]
    vram = sum(g["spec"]["vram_gb"] for g in gpus) if gpus else 0.0
    if bool(args.get("cpu_moe", False)) and model["experts"] and gpus:
        share_bytes = model["params_b"] * (1 - model["expert_share"]) * bpw / 8
    else:
        share_bytes = weights_gb * share
    need = share_bytes + kv_gb * slots + (0.8 * len(gpus) if gpus else 0)
    if gpus and need > vram:
        p.errors.append(f"needs {need:.1f} GB VRAM ({share_bytes:.1f} weights + {kv_gb * slots:.1f} KV), "
                        f"{vram:.0f} GB available: fewer slots, smaller context or fewer GPU layers")
        return p
    if share > 0 and not gpus:
        raise ValueError("GPU layers set but no GPU assigned")
    gpu_bw = sum(_gpu_bw(g, bpw) for g in gpus) if gpus else 0.0
    cores = max(1, min(cpu["cores"], int(args.get("threads") or 0) or int(args["_cores"])))
    cpu_bw = max(_cpu_stream(cpu, cores), cpu["ram_bw_gbs"] * 0.35)   # dense layers stream better than experts
    active_gb = model["active_b"] * bpw / 8
    expert_gb = max(0.0, model["active_b"] - model["params_b"] * (1 - model["expert_share"])) * bpw / 8
    mtp = bool(args.get("mtp", True)) and model["mtp"]
    d = {}
    cpu_moe = bool(args.get("cpu_moe", False)) and expert_gb > 0 and gpus
    if cpu_moe:
        # --n-cpu-moe: every expert on the CPU, attention and dense parts on the GPU (the usual way to run a
        # 120B MoE on a 24 GB card). VRAM then needs only the dense part; host RAM holds the experts.
        dense_total_gb = model["params_b"] * (1 - model["expert_share"]) * bpw / 8
        p.memory["host_ram_gb"] = round((weights_gb - dense_total_gb) * 1.05, 1)
        p.notes.append(f"experts on the CPU: {weights_gb - dense_total_gb:.0f} GB in host RAM, {dense_total_gb:.0f} GB dense on the GPU")
    for n in range(1, slots + 1):
        shared = active_gb - expert_gb
        if cpu_moe:
            gpu_part = shared / gpu_bw
            cpu_part = expert_gb * n / cpu_bw
        else:
            gpu_part = (shared + expert_gb * n) * share / gpu_bw if share > 0 else 0.0
            cpu_part = (shared + expert_gb * n) * (1 - share) / cpu_bw if share < 1 else 0.0
        kv_read = n * min(ctx, 8192) * model["kv_kb_per_token"] * kv_scale / 1e6 / (gpu_bw or cpu_bw)
        t = gpu_part + cpu_part + kv_read + model["layers"] * _layer_ms(gpus) / 1000 * (1 + 0.1 * (n - 1))
        t *= 1 + 0.35 * (n - 1)          # llama.cpp slots batch less efficiently than vLLM (measured 2-3 slots)
        mult = (1.45 if n <= 3 else 1.2) if mtp else 1.0
        d[n] = n * mult / t
    p.decode_total = d
    p.speed_per_request = {n: v / n for n, v in d.items()}
    p.slots = slots
    p.max_parallel = slots
    flops = (sum(_gpu_flops(g) for g in gpus) * _flops_eff(model) * 0.9) if gpus else cores * 60e9
    p.prefill_rate = flops / (2 * model["active_b"] * 1e9) * (share if share > 0 else 1.0) * min(1.0, min(int(args.get("ubatch", 512)), int(args.get("batch", 2048))) / 2048 + 0.6)
    p.prefill_quad = _attention_quad(model, flops) if gpus else 0.0
    p.overhead_s = 0.3
    p.window = ctx
    p.cache_convs = slots
    per_card = [need / len(gpus) for _ in gpus]
    p.memory = _memory_report(gpus, per_card)
    p.memory.update(weights_gb=round(weights_gb, 1), kv_per_slot_gb=round(kv_gb, 1))
    if share < 1:
        p.notes.append(f"{int((1 - share) * 100)} % of the layers run on the CPU at ~{cpu_bw:.0f} GB/s")
    if len(gpus) > 1:
        p.notes.append("layer split over cards: one card works at a time")
    p.limits = {"gpu_memory_bandwidth": round(1.0 if share == 1 else share, 2), "cpu_ram": round(1 - share, 2)}
    p.resources = dict(gpus=[g["id"] for g in gpus], cpu=cpu["name"], cores=cores if share < 1 else 1)
    return p
