"""Static memory layout of a Strata engine, box by box, with the worst-case load on every link.

Exact boxes come from the model and the arguments (dense weights, KV per slot, expert pack); estimated boxes
(CUDA context, scratch, display, OS reserve, pinned buffers) carry defaults from `FIXED_DEFAULTS` that the
user can override per engine (spec.fixed) — they change with driver, OS and engine version, so nobody is
asked to submit them. The goal is to make overshoot obvious and to say what to shrink first.
"""
from __future__ import annotations

from .catalog.cpus import OS_FACTOR  # noqa: F401  (kept for parity with machine.py's OS handling)

# name: (default, unit, explanation) — estimates unless noted
FIXED_DEFAULTS = {
    "cuda_context_mib": (420, "MiB per card", "CUDA context, allocator pools, cuBLAS workspace; grows with the number of kernels loaded"),
    "min_free_vram_mib": (512, "MiB per card", "safety margin Strata keeps free (--min-free-vram-mib); fragmentation lives here"),
    "scratch_base_mib": (256, "MiB per card", "activations and workspace at one row and a small chunk"),
    "scratch_per_row_mib": (96, "MiB per slot", "verify windows, MTP draft ring and logits per parallel slot"),
    "scratch_per_8k_chunk_mib": (600, "MiB per card", "prompt activations per 8,192-token prefill chunk (scales with prefill_chunk)"),
    "vision_mib": (900, "MiB on the first card", "the vision encoder (mmproj) when the model has one"),
    "display_linux_mib": (450, "MiB", "a desktop compositor on this card under Linux"),
    "display_windows_mib": (1300, "MiB", "WDDM, DWM and the desktop under Windows"),
    "pinned_chunk_mib": (256, "MiB per card, host RAM", "pinned staging buffers for expert and KV streaming (chunk-mb)"),
    "host_os_reserve_gb": (4.0, "GB per pool", "kernel, Docker, logging, the agent harness if local"),
    "expert_pack_overhead": (1.05, "factor", "mmap/page-table and alignment overhead on the expert pack in host RAM"),
    "parking_min_free_mib": (1024, "MiB", "parking stops before the pool is this full (conversation-cache-min-free-mib)"),
    "link_efficiency": (0.8, "factor", "share of the PCIe link's nominal rate that streaming reaches"),
    "hit_rate": (None, "0–1 or empty", "override the guessed expert-cache hit rate"),
}


def fixed_value(fixed: dict | None, key: str):
    v = (fixed or {}).get(key)
    if v is None or v == "":
        return FIXED_DEFAULTS[key][0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return FIXED_DEFAULTS[key][0]


def strata_layout(*, spec, args, model, gpus, cpu, info, dense_gb, experts_gb, kv_gb_per_slot, kv_cells, ctx,
                  slots, trim, cache_per_card, held, hit, decode_total, prefill_rate, pcie_bw, miss_via_pcie=False,
                  tps_card_bound=None, cpu_stream=0.0) -> dict:
    fx = spec.fixed if hasattr(spec, "fixed") else {}
    f = lambda k: fixed_value(fx, k)  # noqa: E731
    n = len(gpus)
    os_name = info["cpus"][0].get("os", "linux") if info.get("cpus") else "linux"
    hit_used = hit
    if f("hit_rate") is not None and 0 < f("hit_rate") <= 1:
        hit_used = float(f("hit_rate"))
    chunk = int(args.get("prefill_chunk", 8192) or 8192)
    peer = str(args.get("multi_gpu", "layer-split")) == "peer-device" and n > 1
    kv_total_per_slot_gb = ctx * model["kv_kb_per_token"] / 2 / 1e6 + model["recurrent_state_mb"] / 1000
    kv_streamed_per_slot_gb = max(0.0, kv_total_per_slot_gb - kv_gb_per_slot)
    cards = []
    for i, g in enumerate(gpus):
        boxes = []
        vram = float(g["spec"]["vram_gb"])
        dense = dense_gb / n if (trim and n > 1 and not peer) else dense_gb
        if peer and i > 0:
            dense = 0.0
        if dense:
            boxes.append(dict(kind="dense", label="dense weights" + (" (this stage)" if trim and n > 1 and not peer else ""), gb=dense, exact=True))
        if i == 0 and model.get("vision"):
            boxes.append(dict(kind="dense", label="vision encoder", gb=f("vision_mib") / 1024, exact=False))
        if not (peer and i > 0):
            share = 1 / n if n > 1 else 1.0
            per_slot = kv_gb_per_slot * share
            for s_ in range(slots):
                boxes.append(dict(kind="kv", label=f"slot {s_ + 1} KV" + (" (resident part)" if kv_streamed_per_slot_gb > 0 else ""), gb=per_slot, exact=True))
            boxes.append(dict(kind="kv", label="prefill slot KV", gb=per_slot, exact=True))
            boxes.append(dict(kind="scratch", label="scratch: windows, drafts, prompt chunk",
                              gb=(f("scratch_base_mib") + f("scratch_per_row_mib") * slots + f("scratch_per_8k_chunk_mib") * chunk / 8192) / 1024, exact=False))
        boxes.append(dict(kind="overhead", label="CUDA context", gb=f("cuda_context_mib") / 1024, exact=False))
        dev = info["_devices"].get(g["id"], {})
        if dev.get("display"):
            boxes.append(dict(kind="overhead", label="display / desktop", gb=(f("display_windows_mib") if os_name == "windows" else f("display_linux_mib")) / 1024, exact=False))
        foreign = float(dev.get("_foreign_gb", 0.0))
        if foreign:
            boxes.append(dict(kind="foreign", label="other engines on this card", gb=foreign, exact=True))
        boxes.append(dict(kind="reserve", label="kept free (min-free-vram)", gb=f("min_free_vram_mib") / 1024, exact=False))
        fixed_sum = sum(b["gb"] for b in boxes)
        ec = int(args.get("expert_cache_mib", 0) or 0)
        cache = max(0.0, vram - fixed_sum)
        if ec > 0:
            cache = min(cache, ec / 1024)
        boxes.append(dict(kind="experts", label="expert cache" + (" (explicit)" if ec > 0 else " (fills the rest)"), gb=cache, exact=ec > 0))
        total = sum(b["gb"] for b in boxes)
        cards.append(dict(id=g["id"], name=g["spec"]["name"], capacity_gb=vram, boxes=boxes, total_gb=total,
                          overshoot_gb=max(0.0, total - vram), cache_gb=cache))
    # ---- host RAM pools
    pools = []
    cpu_idx = spec.cpu if hasattr(spec, "cpu") else 0
    cpu_info = info["cpus"].get(cpu_idx) or info["cpus"].get(str(cpu_idx)) or cpu
    ram = float(cpu_info["ram_gb"])
    boxes = [dict(kind="overhead", label="OS, Docker, the rest", gb=f("host_os_reserve_gb"), exact=False),
             dict(kind="experts", label="expert pack (mmap)" + (" — RAM budget; the rest on the SSD" if model.get("ssd_stream") and experts_gb > max(0.0, ram - 24) else ""),
                  gb=(min(experts_gb, max(0.0, ram - 24)) if model.get("ssd_stream") else experts_gb) * f("expert_pack_overhead"), exact=True),
             dict(kind="overhead", label="pinned streaming buffers", gb=f("pinned_chunk_mib") * n / 1024, exact=False)]
    if kv_streamed_per_slot_gb > 0:
        for s_ in range(slots):
            boxes.append(dict(kind="kv", label=f"slot {s_ + 1} streamed KV", gb=kv_streamed_per_slot_gb, exact=True))
    pm = int(args.get("parking_mib", 0) or 0)
    ps = int(args.get("parking_slots", 0) or 0)
    parking = None
    if pm > 0 and ps > 0:
        per_conv = kv_total_per_slot_gb                      # a full conversation at the context window
        by_size = int(pm / 1024 // per_conv) if per_conv > 0 else 0
        fits = min(ps, by_size)
        parking = dict(gb=pm / 1024, slots=ps, per_full_conversation_gb=per_conv, fits_full=fits,
                       limited_by="parking_slots" if ps <= by_size else "parking_mib")
        boxes.append(dict(kind="parking", label=f"parking: fits at most {fits} full conversations", gb=pm / 1024, exact=True, slots=fits))
    boxes.append(dict(kind="reserve", label="kept free (parking min-free)", gb=f("parking_min_free_mib") / 1024, exact=False))
    total = sum(b["gb"] for b in boxes)
    pools.append(dict(cpu=cpu_idx, name=cpu_info["name"], capacity_gb=ram, boxes=boxes, total_gb=total,
                      overshoot_gb=max(0.0, total - ram), parking=parking, ram_bw_gbs=cpu_info["ram_bw_gbs"]))
    # ---- links, worst case: every slot writing at the engine's fastest total rate, plus a prefill chunk in flight
    links = []
    eff = f("link_efficiency")
    active_expert_gb = max(0.0, model["active_b"] - model["params_b"] * (1 - model["expert_share"])) * (experts_gb / max(1e-9, model["params_b"] * model["expert_share"]))
    tps = max(decode_total.values()) if decode_total else 0.0
    rows = slots
    # worst case = the demand if the cards ran at their own pace (every expert resident): the expert path — PCIe
    # copies or the CPU's compute — must then carry the misses; the ratio says how far it falls short
    demand_tps = tps_card_bound or tps
    miss_gbs = active_expert_gb * (1 - hit_used) * demand_tps               # expert bytes missed per second at card pace
    # where those bytes go: in PCIe-copy mode they cross the link to the card; in CPU mode the CPU computes the
    # missed experts in place and only activations cross (a few MB per window, negligible)
    miss_link_gbs = miss_gbs if miss_via_pcie else 0.0
    # streamed KV: each window re-reads the slot's non-resident K/V; the engine's rate already includes that cost,
    # so this part shows how much of the link it eats rather than a hypothetical overshoot
    windows_per_s = tps / (2.8 if model.get("mtp") and args.get("mtp", True) else 1.0)
    kv_gbs = kv_streamed_per_slot_gb * windows_per_s if kv_streamed_per_slot_gb > 0 else 0.0
    prefill_gbs = experts_gb * (1 - held) * prefill_rate / chunk if chunk else 0.0
    for i, g in enumerate(gpus):
        dev = info["_devices"].get(g["id"], {})
        cap = float(dev.get("link_bw_gbs", pcie_bw)) * eff
        share = 1.0 if n == 1 else (1.0 if (peer and i == 0) else 1 / n)
        parts = {("expert misses copied to the card (decode)" if miss_via_pcie else "expert misses (computed on the CPU, nothing crosses)"): miss_link_gbs * share,
                 "streamed KV (decode)": kv_gbs * share, "expert streaming (prefill)": prefill_gbs * share}
        worst_decode = miss_link_gbs * share + parts["streamed KV (decode)"]
        worst = max(worst_decode, parts["expert streaming (prefill)"])
        links.append(dict(a=g["id"], b=f"cpu-{cpu_idx}", kind=dev.get("link_type", "pcie"), capacity_gbs=cap, nominal_gbs=float(dev.get("link_bw_gbs", pcie_bw)),
                          worst_gbs=worst, ratio=(worst / cap if cap else 0.0), parts=parts,
                          verdict=("oversubscribed" if cap and worst > cap else "fits")))
    # host memory: expert reads for the misses (read once by the CPU, or read + DMA'd when copied), streamed KV, prefill streaming
    if not miss_via_pcie and cpu_stream > 0:
        links.append(dict(a="cpu expert path", b=f"cpu-{cpu_idx}", kind="cpu compute", capacity_gbs=float(cpu_stream), nominal_gbs=float(cpu_stream),
                          worst_gbs=miss_gbs, ratio=(miss_gbs / cpu_stream), parts={"expert misses computed on the CPU, at card pace": miss_gbs},
                          verdict=("oversubscribed" if miss_gbs > cpu_stream else "fits")))
    ram_load = (miss_gbs * (2.0 if miss_via_pcie else 1.0)) + kv_gbs * 2.0 + max(0.0, prefill_gbs) * 2.0
    links.append(dict(a=f"cpu-{cpu_idx}", b="ram", kind="memory", capacity_gbs=float(cpu_info["ram_bw_gbs"]), nominal_gbs=float(cpu_info["ram_bw_gbs"]),
                      worst_gbs=ram_load, ratio=(ram_load / cpu_info["ram_bw_gbs"] if cpu_info["ram_bw_gbs"] else 0.0),
                      parts={"expert reads for misses": miss_gbs * (2.0 if miss_via_pcie else 1.0), "streamed KV": kv_gbs * 2.0, "prefill expert streaming": prefill_gbs * 2.0},
                      verdict=("oversubscribed" if ram_load > cpu_info["ram_bw_gbs"] else "fits")))
    headline_kv = None
    if kv_streamed_per_slot_gb > 0:
        # worst case: every slot at the full context window; the link then bounds the window rate
        per_window_gb = kv_streamed_per_slot_gb * slots
        link_cap = min(l["capacity_gbs"] for l in links[:-1]) if len(links) > 1 else links[0]["capacity_gbs"]
        max_windows = link_cap / per_window_gb if per_window_gb else float("inf")
        tok_per_window = 2.8 if model.get("mtp") and args.get("mtp", True) else 1.0
        all_slots = f", {per_window_gb:.1f} GB with all {slots} slots" if slots > 1 else ""
        headline_kv = (f"worst case at the full {ctx // 1024}K context: {kv_streamed_per_slot_gb:.1f} GB of streamed K/V per slot and window{all_slots} "
                       f"— the link allows at most {max_windows * tok_per_window:.0f} tok/s total then")
    worst_card = max(cards, key=lambda c: c["overshoot_gb"]) if cards else None
    worst_link = max(links, key=lambda l: l["ratio"]) if links else None
    headline = [headline_kv] if headline_kv else []
    for c in cards:
        if c["overshoot_gb"] > 0:
            headline.append(f"{c['id']}: over by {c['overshoot_gb']:.1f} GB — shrink the expert cache, slots or context, or turn trim_stage_weights on")
    for p_ in pools:
        if p_["overshoot_gb"] > 0:
            headline.append(f"host RAM: over by {p_['overshoot_gb']:.1f} GB — less parking, fewer streamed slots, or more RAM")
    if worst_link and worst_link["ratio"] > 1:
        what, amount = max(worst_link["parts"].items(), key=lambda kv: kv[1]) if worst_link["parts"] else ("", 0.0)
        over = f"{worst_link['a']}↔{worst_link['b']} {worst_link['kind'].upper()} oversubscribed by {(worst_link['ratio'] - 1) * 100:.0f} %"
        if "KV" in what:
            headline.append(f"worst case at the full {ctx // 1024}K context: {over} by streamed K/V "
                            f"({worst_link['worst_gbs']:.1f} of {worst_link['capacity_gbs']:.1f} GB/s usable)")
        else:
            headline.append(f"worst case: {over} ({worst_link['worst_gbs']:.1f} of {worst_link['capacity_gbs']:.1f} GB/s usable at the cards' own pace, "
                            f"{demand_tps:.0f} tok/s) — the engine actually runs at {tps:.0f} tok/s because of it")
    return dict(cards=cards, pools=pools, links=links, hit=hit_used, hit_guessed=hit, held=held, headline=headline,
                fixed_defaults={k: dict(value=v[0], unit=v[1], help=v[2]) for k, v in FIXED_DEFAULTS.items()},
                fixed=dict(fx or {}), worst_card=(worst_card["id"] if worst_card and worst_card["overshoot_gb"] > 0 else None))
