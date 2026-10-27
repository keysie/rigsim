"""The global constants (spec-sheet first, 2026-10-27): reproducible from their data, and every yardstick against
the model with no per-system fitting. A miss outside tolerance must be named here with its suspected cause in the
model — and a named miss must still miss (when a model change fixes one, this test says so)."""
import re

from app.hw import constants as K
from app.hw.yardsticks.fit_globals import derive
from app.hw.yardsticks.validate_globals import TOLERANCE, run

# (source prefix, point regex) -> suspected cause in the model
KNOWN_MISSES = {
    ("TUD ladders", r"(Cascade Lake|Rome) \w+ [1-4] cores"):
        "per-core limit not identified: queue sizes (outstanding misses per core) not yet sourced from the "
        "optimisation manuals (Little's law, arch.py)",
    ("TUD ladders", r"Rome \w+ 8 cores"):
        "one chiplet-link efficiency for every Zen generation: Rome's link sustains 0.78 of 32 B x FCLK, Zen 4/5 0.88-0.99",
    ("VSC MLC", r"cascadelake lat_local_ns"):
        "Cascade Lake idles 8.5 ns faster than Skylake and Ice Lake; not explained by the DRAM timings",
    ("igor'sLAB MLC", r"265K (xmp7800|xmp7800_uncore|manual7800_uncore)"):
        "DDR5: one read efficiency for DDR4 and DDR5 (DDR5 has two independent 32-bit sub-channels per DIMM)",
    ("RAM db hwluxx", r"hwl-0[24]"):
        "DDR5 read efficiency, as igor'sLAB",
    ("RAM db binarytides", r"bt-01"):
        "Zen 3 APU modelled as monolithic; its cores reach memory through the same 32-byte data-fabric link as the "
        "chiplet parts",
    ("RAM db binarytides", r"bt-02"):
        "CPU not named in the source: a generic package",
    ("RAM db whirlpool", r"wp-01"):
        "FCLK not stated in the source: the chiplet link cannot be built, so the channels bind",
    ("Phoronix STREAM", r"copy (3000|4200|4600)"):
        "STREAM copy also writes; write traffic is not modelled (and the DDR5 effect above)",
    ("box-3090-pci0-pass3", r"compute 220 W"):
        "same card, same limit: one card reading the whole model ran at 1252 MHz, one stage of a split at ~1.8 GHz "
        "(pass 6); how much of the card a load uses is not modelled",
    ("box-3090-pass6-220w", r"(compute|Gigabyte.*memory) 220 W"):
        "as above, the other side of the same gap",
}


def named(r):
    return [k for k in KNOWN_MISSES if r["source"].startswith(k[0]) and re.search(k[1], r["point"])]


def test_constants_are_what_their_data_gives():
    for name, value in derive().items():
        assert abs(getattr(K, name) / value - 1) < 0.002, (name, getattr(K, name), value)


def test_every_yardstick_within_tolerance_or_a_named_miss():
    bad = [(r["source"], r["point"], round(100 * r["error"], 1)) for r in run()
           if abs(r["error"]) > TOLERANCE[r["trust"]] and not named(r)]
    assert not bad, bad


def test_named_misses_still_miss():
    hits = {k: [] for k in KNOWN_MISSES}
    for r in run():
        for k in named(r):
            hits[k].append(abs(r["error"]) > TOLERANCE[r["trust"]] * 0.9)
    for k, outside in hits.items():
        assert outside and any(outside), f"{k} no longer misses — the model changed: update KNOWN_MISSES"


def test_plateaus_and_full_systems_hold():
    """Where the documented structure binds (all channels busy, or both Rome chiplets), within 3 %."""
    rows = [r for r in run() if re.search(r"(Cascade Lake \w+ (8|10) cores|Rome \w+ 16 cores)", r["point"])]
    assert len(rows) == 6 and all(abs(r["error"]) < 0.03 for r in rows)
