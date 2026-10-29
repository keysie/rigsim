"""Intel Memory Latency Checker (MLC) as an engine: what its tests do, as steps of activities on the machine's graph
(hw/activity.py). It knows nothing about the hardware beyond the graph's names for core groups and memory.

  idle latency      one core, a chain of dependent 64-byte reads across a buffer far larger than the caches
                    (MLC's pointer chase): each read needs the previous one's data, so each is its own step, one
                    after another; what a read costs — the memory's latency included — is the graph's.
                    Reported: time per access.
  max bandwidth     every core streams at once (MLC's default buffer per thread, all cores): one step with every
                    core group reading — and for the mixes writing — the memory concurrently, cut into steps of
                    what the cores have in flight at once (their miss queues). Reported: bytes moved by the program
                    per second (reads + writes), as MLC reports them. The mixes are MLC's read:write ratios (3:1,
                    2:1, 1:1; 'stream-triad like' = two reads, one write). How much a step's direction change costs
                    is the memory's (storage turnaround); write-allocate (RFO) reads are not in the graph yet.
"""
from __future__ import annotations

from ...hw.activity import Step, Transfer, solve_step
from ...hw.machine import Machine

LINE = 64
CHASE = 2000                                  # accesses simulated in the chain (enough for a stable mean)
MIX = {"all_reads": (1, 0), "3to1_rw": (3, 1), "2to1_rw": (2, 1), "1to1_rw": (1, 1), "stream_triad": (2, 1)}


def idle_latency_s(m: Machine, core_group: str, memory: str = "ram") -> float:
    """Seconds per access of a dependent read chain from one core: CHASE reads, each a step after the previous."""
    cache: dict = {}
    total = sum(solve_step(m, Step(f"read {i}", [Transfer("read", memory, core_group, LINE, units=1)]), cache).seconds
                for i in range(CHASE))
    return total / CHASE


def in_flight_lines(m: Machine, core_groups: list[str]) -> int | None:
    """Cache lines all cores can have in flight at once (each unit's miss queue, a property of the cores in the
    graph); None if any is not identified."""
    qs = [(m.nodes[g].count, m.nodes[g].outstanding) for g in core_groups]
    return None if any(q is None for _, q in qs) else sum(n * q for n, q in qs)


def bandwidth_bps(m: Machine, core_groups: list[str], traffic: str = "all_reads", memory: str = "ram",
                  total_bytes: float = 4e9) -> tuple[float, str]:
    """Program bytes per second with every core streaming, and how the steps were cut. One step = what all cores
    have in flight at once (their miss queues), reads and writes in MLC's ratio, each core's share by its queue;
    steps follow one another. Where a queue is not identified: the whole stream as one step (the solver may then
    batch all of it — the optimistic case, stated). How a stream's steps overlap in a real core is open: deferred
    until more machines are measured."""
    r, w = MIX[traffic]
    lines = in_flight_lines(m, core_groups)
    per_step = lines * LINE if lines else total_bytes
    weight = {g: (m.nodes[g].count * m.nodes[g].outstanding if lines else m.nodes[g].count) for g in core_groups}
    tot = sum(weight.values())
    acts = []
    for g in core_groups:
        b = per_step * weight[g] / tot
        acts.append(Transfer(f"{g}.read", memory, g, b * r / (r + w)))
        if w:
            acts.append(Transfer(f"{g}.write", g, memory, b * w / (r + w)))
    res = solve_step(m, Step(f"bandwidth {traffic}", acts))
    how = (f"steps of {lines} lines in flight ({per_step / 1024:.0f} KiB)" if lines
           else "one step for the whole stream (miss queues not identified)")
    return per_step / res.seconds, how


NOT_IN_GRAPH = "write-allocate (RFO) reads not modelled"
