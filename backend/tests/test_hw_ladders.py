"""Yardstick 4: core ladders (TU Dresden, ICPE 2022), medium trust (values from the paper's figures)."""
from app.hw.yardsticks import tud_ladders as t

TOLERANCE = 0.05
# Held-out points outside tolerance, each with its reason.
KNOWN_MISSES = {
    ("Xeon Gold 6248 (1 SNC)", "avx512", 4): "sharp knee: max-min fills linearly up to the cap; real saturation is "
                                            "gradual (latency under load, proposal §10)",
    ("EPYC 7702 (1 NPS4 node)", "avx", 2): "same, just below the chiplet link's cap",
    ("EPYC 7702 (1 NPS4 node)", "sse", 2): "same",
}


def test_held_out_points_within_tolerance():
    bad = {(x["cpu"], x["isa"], x["cores"]): round(100 * x["error"], 1) for x in t.report()
           if not x["fitted"] and abs(x["error"]) > TOLERANCE
           and (x["cpu"], x["isa"], x["cores"]) not in KNOWN_MISSES}
    assert not bad, bad


def test_known_misses_are_still_misses():
    by = {(x["cpu"], x["isa"], x["cores"]): x for x in t.report()}
    assert all(abs(by[k]["error"]) > TOLERANCE for k in KNOWN_MISSES)


def test_caps_carry_across_instruction_width():
    """Caps fitted on AVX; on SSE only the per-core value is refitted. The plateaus must still match."""
    by = {(x["cpu"], x["isa"], x["cores"]): x for x in t.report()}
    for k in (("Xeon Gold 6248 (1 SNC)", "sse", 8), ("Xeon Gold 6248 (1 SNC)", "sse", 10),
              ("EPYC 7702 (1 NPS4 node)", "sse", 4), ("EPYC 7702 (1 NPS4 node)", "sse", 16)):
        assert abs(by[k]["error"]) < 0.03, k


def test_second_chiplet_raises_the_cap_to_the_channels():
    r = t.fit_rome()
    one, two = t.rome_read(8, r["core_gbs"], r["eff_link"], r["eff_dram"]), t.rome_read(16, r["core_gbs"], r["eff_link"], r["eff_dram"])
    assert two > one * 1.1 and abs(two - 0.836 * 51.2) < 0.5         # the 2 channels (DDR4-3200) bind with 2 CCDs


def test_fitted_values_are_physical():
    c, r = t.fit_clx(), t.fit_rome()
    assert 0.75 < c["eff_dram"] < 0.95 and 0.75 < r["eff_dram"] < 0.95 and 0.6 < r["eff_link"] < 1.0
