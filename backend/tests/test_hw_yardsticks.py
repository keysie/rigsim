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
