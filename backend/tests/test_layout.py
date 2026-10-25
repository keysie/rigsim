"""Static memory layout for Strata (app/layout.py)."""
from app import api

BOX = {"os": "linux", "cpus": [{"cpu_id": "i9-10900x", "enabled": True, "ram_gb": 128, "ram_mts": 3200, "ram_channels": 4}],
       "devices": [{"id": "gpu-1", "kind": "gpu", "model_id": "rtx-3090", "link_gen": 3, "link_width": 16, "power_w": 250, "display": True},
                   {"id": "gpu-2", "kind": "gpu", "model_id": "rtx-3090", "link_gen": 3, "link_width": 16, "power_w": 250}], "nvlinks": []}


def strata(args, fixed=None, extra=()):
    e = [{"id": "s", "engine": "strata", "version": "0.1.43.1", "model_id": "qwen3.8-flash-next", "gpus": ["gpu-1", "gpu-2"], "cpu_cores": 9, "args": args, "fixed": fixed or {}}, *extra]
    return api.engines_derive({"machine": BOX, "engines": e})["engines"]["s"]


def test_boxes_fill_the_card_exactly_when_the_cache_is_automatic():
    L = strata({"batch": 4})["layout"]
    for c in L["cards"]:
        assert abs(c["total_gb"] - c["capacity_gb"]) < 1e-6 and c["overshoot_gb"] == 0
        kinds = {b["kind"] for b in c["boxes"]}
        assert {"dense", "kv", "scratch", "overhead", "reserve", "experts"} <= kinds
    assert any(b["label"] == "display / desktop" for b in L["cards"][0]["boxes"])          # the display flag on gpu-1
    assert not any(b["label"] == "display / desktop" for b in L["cards"][1]["boxes"])
    assert not L["headline"]


def test_a_second_engine_on_the_card_is_refused():
    guard = {"id": "g", "engine": "llama.cpp", "version": "b11552", "model_id": "ministral-3-3b", "gpus": ["gpu-2"], "cpu_cores": 1,
             "args": {"quantization": "q8_0", "parallel": 1, "ctx_size": 32768}}
    r = strata({"batch": 4}, extra=[guard])
    assert any("one engine per card" in e for e in r["errors"])


def test_parking_counts_full_conversations_and_names_the_limit():
    L = strata({"batch": 4, "parking_mib": 8192, "parking_slots": 8})["layout"]
    pk = L["pools"][0]["parking"]
    assert pk["fits_full"] == 8 and pk["limited_by"] == "parking_slots"
    L = strata({"batch": 4, "parking_mib": 2048, "parking_slots": 32})["layout"]
    pk = L["pools"][0]["parking"]
    assert pk["fits_full"] < 32 and pk["limited_by"] == "parking_mib"
    assert any("fits at most" in b["label"] for b in L["pools"][0]["boxes"])


def test_kv_streaming_moves_kv_into_the_pool_and_states_the_link_bound():
    L = strata({"batch": 4, "kv_resident": 8192, "context": 131072})["layout"]
    assert sum(1 for b in L["pools"][0]["boxes"] if "streamed KV" in b["label"]) == 4
    assert any("at the full 128K context every window streams" in h for h in L["headline"])


def test_overshoot_is_reported_loudly():
    # a tiny host pool cannot hold the expert pack
    small = dict(BOX, cpus=[dict(BOX["cpus"][0], ram_gb=32)])
    r = api.engines_derive({"machine": small, "engines": [{"id": "s", "engine": "strata", "version": "0.1.43.1", "model_id": "qwen3.8-flash-next",
                                                            "gpus": ["gpu-1", "gpu-2"], "cpu_cores": 9, "args": {"batch": 4}}]})["engines"]["s"]
    L = r["layout"]
    assert L["pools"][0]["overshoot_gb"] > 20 and any("host RAM: over by" in h for h in L["headline"])


def test_worst_case_link_load_responds_to_the_hit_rate():
    low = strata({"batch": 8, "expert_cache_mib": 2048}, fixed={"hit_rate": 0.3})["layout"]
    high = strata({"batch": 8, "expert_cache_mib": 2048}, fixed={"hit_rate": 0.95})["layout"]
    worst = lambda L: max(l["ratio"] for l in L["links"])  # noqa: E731
    assert worst(low) > worst(high) * 2
    assert any("oversubscribed" in h and "actually runs at" in h for h in low["headline"])
    # the expert path that carries the misses is named: CPU compute in the default mode, the PCIe link in copy mode
    assert any(l["a"] == "cpu expert path" for l in low["links"])
    copy = strata({"batch": 8, "expert_cache_mib": 2048, "pcie_frac": "1"}, fixed={"hit_rate": 0.3})["layout"]
    assert not any(l["a"] == "cpu expert path" for l in copy["links"]) and copy["links"][0]["ratio"] > 1


def test_fixed_overrides_change_the_boxes():
    base = strata({"batch": 4})["layout"]["cards"][0]
    big = strata({"batch": 4}, fixed={"cuda_context_mib": 2048})["layout"]["cards"][0]
    ctx_base = next(b["gb"] for b in base["boxes"] if b["label"] == "CUDA context")
    ctx_big = next(b["gb"] for b in big["boxes"] if b["label"] == "CUDA context")
    assert abs(ctx_big - 2.0) < 0.01 and ctx_base < 0.5
    assert big["cache_gb"] < base["cache_gb"]


def test_kv_bound_does_not_fall_with_more_slots():
    """Each window streams every slot's K/V but also yields a token per slot; the two cards stream their own halves."""
    import re
    bound = {}
    for b in (2, 4, 8):
        L = strata({"batch": b, "kv_resident": 8192, "context": 131072})["layout"]
        h = next(x for x in L["headline"] if "every window streams" in x)
        bound[b] = int(re.search(r"at most (\d+) tok/s", h).group(1))
    assert bound[2] == bound[4] == bound[8], bound
    assert 10 <= bound[8] <= 30, bound          # 2 links x 10 GB/s usable / 1.9 GB per slot and window


def test_parked_conversation_size_does_not_depend_on_the_slots():
    sizes = {b: strata({"batch": b, "parking_mib": 32000, "parking_slots": 13, "context": 128000})["layout"]["pools"][0]["parking"]
             for b in (1, 4, 8)}
    assert len({round(p["per_full_conversation_gb"], 3) for p in sizes.values()}) == 1
    assert len({p["fits_full"] for p in sizes.values()}) == 1


def test_an_explicit_cache_bigger_than_the_room_overshoots():
    r = strata({"batch": 4, "expert_cache_mib": 30000})
    assert r["layout"]["cards"][0]["overshoot_gb"] > 4 and any("expert_cache_mib asks for" in h for h in r["layout"]["headline"])
    assert any("expert_cache_mib asks for" in e for e in r["errors"])


def test_overshoot_advice_without_a_cache_does_not_say_shrink_the_cache():
    r = strata({"batch": 8, "context": 262144})
    h = next(x for x in r["layout"]["headline"] if "over by" in x)
    assert "before any expert cache" in h and "shrink the expert cache" not in h


def test_a_hand_set_hit_rate_drives_the_speeds_too():
    base = strata({"batch": 1})
    low = strata({"batch": 1}, fixed={"hit_rate": 0.4})
    assert low["memory"]["hit"] == 0.4 and low["decode_total"]["1"] < base["decode_total"]["1"] * 0.7
    assert low["layout"]["hit"] == 0.4 and low["layout"]["hit_guessed"] > 0.9


def test_peer_device_shows_the_peer_card():
    L = strata({"batch": 1, "multi_gpu": "peer-device"})["layout"]
    assert [c["id"] for c in L["cards"]] == ["gpu-1", "gpu-2"]
    assert any(b["label"] == "peer expert tier" for b in L["cards"][1]["boxes"])


def test_unified_memory_is_one_pool():
    from app.catalog.prebuilts import PREBUILTS
    m = PREBUILTS[0]["machine"]
    e = {"id": "s", "engine": "strata", "version": "0.1.43.1", "model_id": "qwen3.8-flash-next", "gpus": ["gpu-1"], "cpu_cores": 8, "args": {"batch": 1}}
    L = api.engines_derive({"machine": m, "engines": [e]})["engines"]["s"]["layout"]
    assert L["cards"] == [] and len(L["pools"]) == 1 and L["pools"][0]["name"].startswith("unified memory")
    assert sum(1 for b in L["pools"][0]["boxes"] if b["kind"] == "experts") == 1     # the experts counted once
    assert L["pools"][0]["total_gb"] <= L["pools"][0]["capacity_gb"] + 0.01
