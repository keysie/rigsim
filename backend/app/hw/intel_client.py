"""Intel Core desktop processors, 6th to 13th generation (Skylake-S .. Raptor Lake-S): what the block model needs,
per generation, from Intel's documentation (proposal §5, spec-sheet first; nothing fitted).

All of them are one monolithic die: cores and the memory controller on one ring, two DDR channels (DDR4: one
64-bit bus each; DDR5 on 12th / 13th gen: two 32-bit sub-channels each, arch.SUBCHANNELS). The ring's capacity and
the cores' miss queues are not documented per generation here: non-binding, as for every other monolithic CPU in
cpus.py. What differs between SKUs of one generation — how many cores of which type — is read from the machine
itself (the run's inventory), not from a SKU table: that is the device's own fact, and it covers every SKU.

Sources:
  - memory support per generation and tier: Intel, "Supported Memory Type for Intel Core Desktop Processors"
    (support article 000094585): 6th DDR4-2133, 7th DDR4-2400, 8th i5/i7 DDR4-2666 (i3 2400), 9th i5/i7/i9 2666
    (i3 2400), 10th i7/i9 DDR4-2933 (i3/i5 2666), 11th i5/i7/i9 DDR4-3200, 12th DDR5-4800 / DDR4-3200, 13th
    DDR5-5600 for i5-13600 and above, DDR5-4800 below / DDR4-3200; two channels throughout;
  - core types: Intel product briefs — hybrid P + E cores from 12th gen (Golden Cove + Gracemont; 13th: Raptor
    Cove + Gracemont); before, one core type (Skylake cores through 10th gen, Cypress Cove on 11th).
"""
from __future__ import annotations

import re

# generation -> architecture facts
GENERATIONS = {
    6: dict(codename="skylake-s", core_types={"core": "skylake"}, mem=dict(ddr4=2133)),
    7: dict(codename="kabylake-s", core_types={"core": "kabylake"}, mem=dict(ddr4=2400)),
    8: dict(codename="coffeelake-s", core_types={"core": "coffeelake"}, mem=dict(ddr4=2666), mem_i3=dict(ddr4=2400)),
    9: dict(codename="coffeelake-s-refresh", core_types={"core": "coffeelake"}, mem=dict(ddr4=2666), mem_i3=dict(ddr4=2400)),
    10: dict(codename="cometlake-s", core_types={"core": "cometlake"}, mem=dict(ddr4=2933), mem_i3=dict(ddr4=2666),
             mem_i5=dict(ddr4=2666)),
    11: dict(codename="rocketlake-s", core_types={"core": "cypresscove"}, mem=dict(ddr4=3200), mem_i3=dict(ddr4=2666)),
    12: dict(codename="alderlake-s", core_types={"p": "goldencove", "e": "gracemont"}, mem=dict(ddr5=4800, ddr4=3200)),
    13: dict(codename="raptorlake-s", core_types={"p": "raptorcove", "e": "gracemont"}, mem=dict(ddr5=5600, ddr4=3200),
             mem_below_13600=dict(ddr5=4800, ddr4=3200)),
}
CHANNELS = 2
SOURCE = ("Intel support article 000094585 (memory per generation and tier, two channels); Intel product briefs "
          "(core types); core counts from the machine's own inventory")
MODEL = re.compile(r"\bi([3579])-(\d{4,5})([A-Z]*)\b")


def generation(cpu: str) -> tuple[int, str, int] | None:
    """(generation, tier i3/i5/i7/i9, model number) from a CPU name such as
    '13th Gen Intel(R) Core(TM) i7-13700K' or 'Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz'. Desktop numbering:
    4 digits = 6th-9th gen (first digit), 5 digits = 10th-13th (first two)."""
    m = MODEL.search(cpu or "")
    if not m or "Intel" not in cpu:
        return None
    num = m.group(2)
    gen = int(num[:2]) if len(num) == 5 else int(num[0])
    return (gen, f"i{m.group(1)}", int(num)) if gen in GENERATIONS else None


def memory_support(gen: int, tier: str, number: int) -> dict:
    g = GENERATIONS[gen]
    if gen == 13 and number < 13600:
        return g["mem_below_13600"]
    return g.get(f"mem_{tier}", g["mem"])


def spec(cpu: str, cores: dict[str, int]) -> dict | None:
    """A cpus.py-style entry for this CPU, with the core counts the machine reports ({'p': 8, 'e': 8} on hybrid
    generations, {'core': n} before)."""
    g = generation(cpu)
    if not g:
        return None
    gen, tier, number = g
    info = GENERATIONS[gen]
    types = info["core_types"]
    if set(types) == {"core"}:
        cores = {"core": sum(cores.values())}
    else:
        cores = {t: cores.get(t, 0) for t in types if cores.get(t, 0)}
    return dict(arch=info["codename"], topology="monolithic", cores=cores, channels=CHANNELS, generation=gen,
                core_uarch={t: types[t] for t in cores}, memory_support=memory_support(gen, tier, number),
                source=SOURCE)
