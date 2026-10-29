"""CPU core microarchitectures as budgets (docs/overhaul/cpu-compute.md): how many instructions a core issues per
clock and how many of each class its execution ports take per clock — 256-bit (AVX2) operations, which is what the
engines here run. From the vendors' optimisation manuals and Agner Fog's instruction tables:

  Skylake family (6th-10th gen Core: Skylake, Kaby Lake, Coffee Lake, Comet Lake) — Intel 64 and IA-32 Architectures
    Optimization Reference Manual, Skylake microarchitecture: 4-wide, vector ALU on ports 0/1/5, shuffles on port 5
    only, vector integer multiply (pmaddubsw, pmaddwd, psign) on 0/1, FMA on 0/1, two loads and one store per clock.
  Golden Cove / Raptor Cove (12th / 13th gen P-cores) — Intel optimization manual, Golden Cove: 6-wide, three vector
    ALUs, two shuffle ports (1 and 5), integer multiply and FMA on 0/1, three loads, two stores.
  Gracemont (E-cores) — Intel optimization manual, Gracemont: 5-wide allocation; 128-bit vector units, so a 256-bit
    operation is two: per 256-bit operation two ALU, one shuffle, one multiply, one FMA, one load per clock.
  Zen 2 / Zen 3 / Zen 4 — AMD Software Optimization Guides for family 17h / 19h: 6-wide dispatch (Zen 2: 5... taken
    as the guide states per family), four 256-bit FP/vector pipes: ALU on all four, shuffles on two, integer multiply
    on two, FMA on two; three loads (two 256-bit), two stores.

Energy per instruction is documented nowhere: a global constant per microarchitecture, set from measurements
(calibrate.py, the stick's cpu-gemm one core vs all cores). None until calibrated: the power budget then does not
bind, which overestimates many cores together — stated in every prediction.
"""
from __future__ import annotations

U = {
    "skylake": dict(issue=4, rates=dict(vec_alu=3, shuffle=1, vec_mul=2, fma=2, load=2, store=1)),
    "cypresscove": dict(issue=5, rates=dict(vec_alu=3, shuffle=2, vec_mul=2, fma=2, load=2, store=2)),
    "goldencove": dict(issue=6, rates=dict(vec_alu=3, shuffle=2, vec_mul=2, fma=2, load=3, store=2)),
    "gracemont": dict(issue=5, rates=dict(vec_alu=2, shuffle=1, vec_mul=1, fma=1, load=1, store=1)),
    "zen": dict(issue=6, rates=dict(vec_alu=2, shuffle=1, vec_mul=1, fma=1, load=1, store=0.5)),
    "zen2": dict(issue=6, rates=dict(vec_alu=4, shuffle=2, vec_mul=2, fma=2, load=2, store=1)),
    "zen3": dict(issue=6, rates=dict(vec_alu=4, shuffle=2, vec_mul=2, fma=2, load=2, store=2)),
    "zen4": dict(issue=6, rates=dict(vec_alu=4, shuffle=2, vec_mul=2, fma=2, load=2, store=2)),
}
ALIASES = {"kabylake": "skylake", "coffeelake": "skylake", "cometlake": "skylake", "raptorcove": "goldencove",
           "zenplus": "zen"}
ENERGY_J: dict[str, dict[str, float]] = {}      # per microarchitecture, from calibrate.py; empty until calibrated


def lookup(core: str | None) -> dict | None:
    if not core:
        return None
    k = ALIASES.get(core, core)
    u = U.get(k)
    return dict(u, uarch=k, energy=ENERGY_J.get(k, {})) if u else None
