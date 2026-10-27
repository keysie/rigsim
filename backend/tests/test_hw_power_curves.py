"""Clock data (data/gpu-power-curves.json): what the measurements show, and the curve as a user override. The
clock model itself is the spec-sheet law (power.spec_clock_mhz); its validation is in test_hw_globals.py."""
from app.hw import Machine, Router
from app.hw.curves import calibration_curves
from app.hw.gpu import add_gpu, set_load_class


def test_memory_heavy_runs_faster_than_compute_heavy_at_every_limit():
    c = calibration_curves()["rtx-3090"]
    for w in (220, 250, 300):
        assert c[("sm_clock", "compute")].value(w) < c[("sm_clock", "memory")].value(w)


def test_clock_rises_with_the_limit():
    for cls in ("compute", "memory"):
        c = calibration_curves()["rtx-3090"][("sm_clock", cls)]
        assert c.value(220) < c.value(250) < c.value(300)


def test_a_measured_curve_overrides_the_law():
    """The 4090 point as a user curve: the spec per-SM rates at that clock give the paper's FP16 throughput."""
    m = Machine(); m.add(Router("cpu"))
    add_gpu(m, "g", "rtx-4090", "cpu", slot_gen=4, slot_lanes=16, curves=calibration_curves()["rtx-4090"])
    st = set_load_class(m, "g", "compute")
    assert not st["sm_clock_basis"].startswith("spec-sheet law")
    assert abs(m.nodes["g.sm"].rate("fp16_tensor") / 357.6e12 - 1) < 0.01
