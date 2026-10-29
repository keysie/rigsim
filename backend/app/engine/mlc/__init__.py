"""Intel Memory Latency Checker (MLC) as an engine: what its tests do, as steps of activities on the machine's graph
(hw/activity.py). It knows nothing about the hardware beyond the graph's names for core groups and memory.

  idle latency      one core, a chain of dependent 64-byte reads across a buffer far larger than the caches
                    (MLC's pointer chase): each read needs the previous one's data, so each is its own step, one
                    after another; what a read costs — the memory's latency included — is the graph's.
                    Reported: time per access.
  max bandwidth     every core streams at once (MLC's default buffer per thread, all cores): one step with every
                    core group reading — and for the mixes writing — the memory concurrently. Reported: bytes moved
                    by the program per second (reads + writes), as MLC reports them.
                    The mixes are MLC's read:write ratios (3:1, 2:1, 1:1; 'stream-triad like' = two reads, one
                    write). Writes are transfers core -> memory over the same channels. What the graph does not
                    contain yet: the bus turnaround between reads and writes and any write-allocate (RFO) read
                    traffic — the predictions say so.
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


def bandwidth_bps(m: Machine, core_groups: list[str], traffic: str = "all_reads", memory: str = "ram",
                  bytes_per_group: float = 4e9) -> float:
    """Program bytes per second with every core group reading (and writing) concurrently."""
    r, w = MIX[traffic]
    acts = []
    for g in core_groups:
        acts.append(Transfer(f"{g}.read", memory, g, bytes_per_group * r / (r + w)))
        if w:
            acts.append(Transfer(f"{g}.write", g, memory, bytes_per_group * w / (r + w)))
    res = solve_step(m, Step(f"bandwidth {traffic}", acts))
    return bytes_per_group * len(core_groups) / res.seconds


NOT_IN_GRAPH = "bus turnaround between reads and writes and write-allocate (RFO) reads not modelled"
