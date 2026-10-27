"""Yardstick 1 (VSC, Intel MLC): fitted values reproduce what they were fitted to; the rest is validation."""
import pytest

from app.hw.yardsticks import vsc_mlc as y


@pytest.mark.parametrize("name", list(y.NODES))
def test_fitted_parameters_reproduce_their_measurements(name):
    n, meas = y.NODES[name], y.NODES[name]["measured"]
    o = y.observe(y.build(n, **y.fit(name)))
    for k in ("lat_local_ns", "lat_remote_ns", "bw_local_gbs", "bw_remote_gbs"):   # remote bw: the input itself
        assert abs(o[k] / y.mean(meas[k]) - 1) < 1e-6


@pytest.mark.parametrize("name", list(y.NODES))
def test_validation_system_bandwidth_from_one_socket(name):
    """Held out: both sockets reading their own RAM at once (MLC 'ALL Reads')."""
    n, meas = y.NODES[name], y.NODES[name]["measured"]
    o = y.observe(y.build(n, **y.fit(name)))
    assert abs(o["bw_system_gbs"] / meas["bw_system_gbs"] - 1) < 0.01


@pytest.mark.parametrize("name", list(y.NODES))
def test_fitted_parameters_are_physical(name):
    p = y.fit(name)
    assert 30e-9 < p["fixed_s"] < 120e-9 and 20e-9 < p["socket_latency_s"] < 120e-9
    assert 0.5 < p["ram_eff"] <= 1.0


def test_one_fixed_latency_term_per_platform_is_too_coarse():
    """Finding, not a target: Skylake's fixed term applied to Cascade Lake (same platform) overshoots its local
    latency by ~9 ns (~11 %). Kept as a test so a model change that alters it is noticed."""
    sky = y.fit("skylake")
    n = y.NODES["cascadelake"]
    o = y.observe(y.build(n, **{**y.fit("cascadelake"), "fixed_s": sky["fixed_s"]}))
    miss = o["lat_local_ns"] / y.mean(n["measured"]["lat_local_ns"]) - 1
    assert 0.08 < miss < 0.14


# --- Yardstick 2: i9-12900K at ten DDR5 speeds (Phoronix / OpenBenchmarking) -----------------------------------
from app.hw.yardsticks import phoronix_ddr5 as p5


@pytest.mark.parametrize("kernel", list(p5.STREAM))
def test_constant_efficiency_holds_from_5000_to_6000(kernel):
    """Fitted at 6000 MT/s, 5000-5600 are predicted within 1.5 % (measurement SE < 0.1 % there)."""
    err = dict(zip(p5.SPEEDS, p5.errors(kernel, p5.fit_at(kernel, 6000))))
    assert all(abs(err[s]) < 0.015 for s in (5000, 5400, 5600))


@pytest.mark.parametrize("kernel", list(p5.STREAM))
def test_finding_constant_efficiency_overshoots_below_5000(kernel):
    """Finding, not a target: below 5000 MT/s the measured efficiency drops (0.76 -> 0.65 for Triad), with an extra
    dip at 4200/4600; a constant efficiency overshoots by up to ~10-17 %. Timings per speed are not published."""
    err = dict(zip(p5.SPEEDS, p5.errors(kernel, p5.fit_at(kernel, 6000))))
    assert 0.08 < err[3000] < 0.18 and err[4200] > 0.08 and err[4600] > 0.08 and abs(err[3800]) < 0.05


def test_both_core_groups_take_part():
    m = p5.build(6000, 0.8)
    res = m.solve_reads([p5.Read("p", "pcores", "ram"), p5.Read("e", "ecores", "ram")])
    assert abs(res["p"]["rate_bps"] - res["e"]["rate_bps"]) < 1                  # 8 + 8 cores, equal weight per core


# --- Yardstick 3 (candidate): Core Ultra 7 265K, five memory configurations (igor'sLAB, MLC 3.12) ----------------
from app.hw.yardsticks import igorslab_arrowlake as il


def test_rate_change_with_absolute_timings_kept_is_predicted():
    """6000 CL28 vs 7800 CL36 (same absolute primary timings): constant efficiency holds within 1 %."""
    assert abs(il.predicted_error("xmp7800", "xmp6000")) < 0.01


def test_uncore_clock_moves_latency_not_bandwidth():
    """Same rate and timings, only the uncore clock: reads +0.4 %, idle latency -11.6 ns. Consistent with the model
    (the controller/fabric term is in latency only); it also means that term depends on the uncore clock."""
    a, b = il.CONFIGS["xmp7800"], il.CONFIGS["xmp7800_uncore"]
    assert abs(b[1] / a[1] - 1) < 0.01 and a[3] - b[3] > 10


def test_finding_timings_do_enter_bandwidth():
    """Finding, not a target: at the same rate, tighter secondary timings give +3.5 % reads (+10.6 % 1:1); the
    JEDEC 5600 CL46 profile reads ~10 % below the XMP efficiency. The model has no path for this yet."""
    assert -0.05 < il.predicted_error("xmp7800", "manual7800_uncore") < -0.025
    assert 0.08 < il.predicted_error("xmp7800", "jedec5600") < 0.14
