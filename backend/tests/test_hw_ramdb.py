"""RAM database as validation points (medium to low trust), within one source's sequence."""
import pytest

from app.hw.yardsticks import ramdb
from app.hw.yardsticks.validate_ramdb import run

TOLERANCE = {"medium": 0.05, "low": 0.10}
# Held-out points outside tolerance, each with its reason. Adding one needs a reason here.
KNOWN_MISSES = {
    "ocg-05": "tREFI raised: refresh is not modelled (proposal §10, avenue)",
    "ocg-06": "after the tREFI step (tWR only, reads unchanged)",
    "ocg-07": "fully tuned timings incl. tREFI",
}


def test_every_config_field_states_its_basis():
    db = ramdb.load()
    for e in db["entries"]:
        for part in ("system", "memory", "timings", "clocks", "tool"):
            for k, f in (e.get(part) or {}).items():
                assert isinstance(f, dict) and f.get("basis") in ("stated", "inferred", "not_stated"), (e["id"], part, k)


def test_no_read_exceeds_the_theoretical_peak():
    for p in ramdb.points():
        assert p["read_gbs"] <= p["peak_gbs"] * 1.001, p["id"]


def test_trust_is_medium_or_low_only():
    assert {p["trust"] for p in ramdb.points()} <= {"medium", "low"}


@pytest.fixture(scope="module")
def rows():
    return run()


def test_held_out_points_within_their_trust_tolerance(rows):
    held = [r for r in rows if r["held_out"]]
    assert len(held) >= 20
    bad = {r["id"]: round(100 * r["error"], 1) for r in held
           if abs(r["error"]) > TOLERANCE[r["trust"]] and r["id"] not in KNOWN_MISSES}
    assert not bad, bad


def test_known_misses_are_still_misses(rows):
    """If a model change fixes one, it should be moved out of KNOWN_MISSES deliberately."""
    by = {r["id"]: r for r in rows}
    for i in KNOWN_MISSES:
        assert abs(by[i]["error"]) > TOLERANCE[by[i]["trust"]], i


def test_chiplet_link_carries_across_transfer_rate_and_fabric_clock(rows):
    """ocinside 7800X3D: one link efficiency fitted at 5200 / FCLK 1733 predicts 6200-6400 at FCLK 2000-2167."""
    by = {r["id"]: r for r in rows}
    assert all(abs(by[i]["error"]) < 0.01 for i in ("ocg-02", "ocg-03", "ocg-04"))


def test_finding_link_efficiency_differs_between_sources(rows):
    """Same CPU (9800X3D) and FCLK 2000: 0.877 (ocinside) vs 0.994 (TechPowerUp). Absolute values do not compare
    across sources; that is why validation stays within one source."""
    eff = {r["source"]: r["param"] for r in rows if r["cpu"] == "Ryzen 7 9800X3D" and r["fitted_param"] == "link"}
    assert eff["tpu-9800x3d-scaling"] - eff["ocinside-crucial-6400"] > 0.08
