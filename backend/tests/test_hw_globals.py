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
    ("igor'sLAB MLC", r"265K (xmp7800_uncore|manual7800_uncore)"):
        "Intel DDR5 above what JEDEC refresh allows even with perfect bank-group scheduling: the systems most likely "
        "run a longer tREFI (XMP / board defaults) — not stated in the source",
    ("RAM db hwluxx", r"hwl-04"):
        "Intel DDR5 refresh, as igor'sLAB",
    ("RAM db binarytides", r"bt-01"):
        "Zen 3 APU modelled as monolithic; its cores reach memory through the same 32-byte data-fabric link as the "
        "chiplet parts",
    ("RAM db binarytides", r"bt-02"):
        "CPU not named in the source: a generic package",
    ("RAM db whirlpool", r"wp-01"):
        "FCLK not stated in the source: the chiplet link cannot be built, so the channels bind",
    ("Phoronix STREAM", r"copy (3000|3200|3600|4200|4600)"):
        "below DDR5's JEDEC speed range the measured efficiency falls with speed, the JEDEC mechanism (bank-group "
        "gaps, refresh) predicts it rising; STREAM copy also writes (write traffic not modelled)",
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


def test_refresh_experiment_supports_all_bank_refresh():
    """ocinside OC guide, Ryzen 7 7800X3D DDR5-6200: only tREFI changes (10126 -> 62000 clocks, 3.27 -> 20.0 us)
    and the read bandwidth rises 61.5 -> 67.5 GB/s. All-bank refresh explains it with a tRFC near 340 ns (32 Gb dies;
    JEDEC lists the value as TBD); same-bank refresh (one bank per group blocked) could not give more than ~2 %."""
    from app.hw.yardsticks import ramdb
    pts = {p["id"]: p for p in ramdb.points()}
    a, b = pts["ocg-04"], pts["ocg-05"]
    ratio = b["read_gbs"] / a["read_gbs"]
    lo, hi = 0.0, 1000.0                               # tRFC (ns) that gives the measured ratio with all-bank refresh
    for _ in range(60):
        mid = (lo + hi) / 2
        r = (1 - mid / b["trefi_ns"]) / (1 - mid / a["trefi_ns"])
        lo, hi = (mid, hi) if r < ratio else (lo, mid)
    assert 300 < lo < 380
    same_bank = (1 - 0.25 * 190 / b["trefi_ns"]) / (1 - 0.25 * 190 / a["trefi_ns"])   # generous tRFCsb
    assert same_bank < 1.02 < ratio


def test_plateaus_and_full_systems_hold():
    """Where the documented structure binds (all channels busy, or both Rome chiplets), within 3 %."""
    rows = [r for r in run() if re.search(r"(Cascade Lake \w+ (8|10) cores|Rome \w+ 16 cores)", r["point"])]
    assert len(rows) == 6 and all(abs(r["error"]) < 0.03 for r in rows)
