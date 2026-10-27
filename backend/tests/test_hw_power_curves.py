"""SM-clock curves per load class: calibration data and held-out validation (medium trust)."""
from app.hw import Machine, Router
from app.hw.curves import calibration_curves, load
from app.hw.gpu import add_gpu, set_load_class

# Held-out points outside tolerance, each with its reason.
TOLERANCE = 0.10
KNOWN_MISSES = {
    ("gpu1 (Gigabyte; pass-3 card, inferred)", "compute"): "same card, same limit, both 'compute': one card reading "
        "the whole model (pass 3, 1252 MHz) vs half the layers in a split (pass 6, 1845 MHz). Two load classes do "
        "not capture how much power a load draws per MHz",
    ("gpu2 (ASUS Tuf)", "compute"): "same, and a different card than the curve's",
    ("gpu1 (Gigabyte; pass-3 card, inferred)", "memory"): "same effect for writing (1500 vs 1845 MHz)",
    ("gpu2 (ASUS Tuf)", "memory"): "same effect for writing, other card (1500 vs 1695 MHz)",
}


def test_compute_runs_below_memory_at_every_limit():
    c = calibration_curves()["rtx-3090"]
    for w in (220, 250, 300):
        assert c[("sm_clock", "compute")].value(w) < c[("sm_clock", "memory")].value(w)


def test_clock_rises_with_the_limit():
    for cls in ("compute", "memory"):
        c = calibration_curves()["rtx-3090"][("sm_clock", cls)]
        assert c.value(220) < c.value(250) < c.value(300)


def test_4090_full_limit_point_reproduces_the_paper():
    """Inferred point: at 450 W the compute clock makes the spec per-SM rates give the measured throughput."""
    m = Machine(); m.add(Router("cpu"))
    add_gpu(m, "g", "rtx-4090", "cpu", slot_gen=4, slot_lanes=16, curves=calibration_curves()["rtx-4090"])
    set_load_class(m, "g", "compute")
    fp16 = m.nodes["g.sm"].rate("fp16_tensor")
    assert abs(fp16 / 357.6e12 - 1) < 0.01


def held_out():
    m = Machine(); m.add(Router("cpu"))
    curves = calibration_curves()["rtx-3090"]
    rows = []
    for v in load()["validation"]:
        for p in v["points"]:
            gid = f"g{len(rows)}"
            add_gpu(m, gid, v["model"], "cpu", slot_gen=3, slot_lanes=16, power_limit_w=p["limit_w"], curves=curves)
            set_load_class(m, gid, p["load_class"])
            rows.append(dict(p, model_mhz=m.nodes[f"{gid}.sm"].clock_hz / 1e6))
    for r in rows:
        r["error"] = r["model_mhz"] / r["sm_mhz"] - 1
    return rows


def test_held_out_points_within_tolerance():
    bad = {(r["card"], r["load_class"]): round(100 * r["error"], 1) for r in held_out()
           if abs(r["error"]) > TOLERANCE and (r["card"], r["load_class"]) not in KNOWN_MISSES}
    assert not bad, bad


def test_known_misses_are_still_misses():
    by = {(r["card"], r["load_class"]): r for r in held_out()}
    assert all(abs(by[k]["error"]) > TOLERANCE for k in KNOWN_MISSES)
