"""The four hardware primitives (proposal §5). Units: bytes, seconds, bytes per second.

A value of None means "not identified": the primitive has the property, but no measurement fixes it. A None
bandwidth is treated as non-binding, and every result that crosses it says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Storage:
    """Any storage — a cache, DRAM, VRAM, an SSD, an HDD — with one set of parameters, each per direction where
    devices differ. None = not identified: the solver treats it as non-binding / zero and results say so.

      bandwidth_bps / access_s   shorthand for both directions (read_* / write_* override them)
      read_bps, write_bps        streaming limits per direction
      read_access_s, write_access_s   latency of one access per direction (DRAM: CL / CWL plus the fixed path)
      seek_s                     positioning before an access: 0 for RAM, small for SSDs, milliseconds for HDDs —
                                 paid once by a sequential transfer, per request by a random one; it occupies the
                                 device
      turnaround_s               switching between reading and writing; it occupies the device

    The solver batches what one step asks of a storage as well as it can (all reads, then all writes: one switch);
    it never batches across steps."""
    id: str
    capacity_b: float
    bandwidth_bps: float | None
    access_s: float | None
    read_bps: float | None = None
    write_bps: float | None = None
    read_access_s: float | None = None
    write_access_s: float | None = None
    seek_s: float | None = None
    turnaround_s: float | None = None

    def bps(self, direction: str = "read") -> float | None:
        v = self.read_bps if direction == "read" else self.write_bps
        return self.bandwidth_bps if v is None else v

    def access(self, direction: str = "read") -> float | None:
        v = self.read_access_s if direction == "read" else self.write_access_s
        return self.access_s if v is None else v

    def effective_bps(self, access_size_b: float) -> float | None:
        """Rate for accesses of a given size: size / (access time + size / bandwidth)."""
        if self.bps() is None:
            return None
        return access_size_b / ((self.access() or 0.0) + access_size_b / self.bps())


@dataclass
class Cache(Storage):
    """A storage in front of another (`backing`: a storage or an interleaved set's name) — a CPU's L3, a GPU's L2, an
    SSD's DRAM or SLC cache. Its own speeds and latencies are a Storage's; what it adds is its policy:

      write_allocate   a write to a line it does not hold first reads that line from the backing storage
      write_back       written lines go to the backing storage when evicted (else: written through at once)

    Whether an access hits is not computed: the engine states it (Transfer.hits) as an assumption about its software,
    like a kernel's efficiency. Flows pass through a cache on their way to the backing storage."""
    backing: str | None = None
    write_allocate: bool | None = None
    write_back: bool | None = None


@dataclass
class Compute:
    """Clock and per-class rates as multiples of it; `count` identical units (a core group is one object)."""
    id: str
    clock_hz: float
    ops_per_clock: dict[str, float] = field(default_factory=dict)   # per unit, per operation class
    count: int = 1
    outstanding: int | None = None    # memory requests one unit can have in flight (its miss queue); None = not identified
    # CPU cores as budgets (docs/overhaul/cpu-compute.md): instructions issued per clock per unit (None: no issue
    # limit), and the energy one instruction of a class draws from the power budget it belongs to
    issue_per_clock: float | None = None
    energy_j: dict[str, float] = field(default_factory=dict)
    budget: str | None = None         # the PowerBudget it draws from (the package)

    def rate(self, op_class: str, units: int | None = None) -> float:
        n = self.count if units is None else min(units, self.count)
        return self.clock_hz * self.ops_per_clock.get(op_class, 0.0) * n


@dataclass
class Link:
    """Point to point. Simplex links share one capacity for both directions; duplex links have one per direction.
    `per_unit` links exist once per unit of a Compute group (a core's own link to its router)."""
    id: str
    a: str
    b: str
    bandwidth_bps: float | None
    latency_s: float = 0.0
    duplex: bool = True
    per_unit: bool = False


@dataclass
class Router:
    """Joins links. Forwarding capacity None = not identified (non-binding). A router whose forwarding needs CPU
    cycles (host bounce buffers, software copies) names that compute and the cycles per byte; those cycles compete
    with everything else that compute does."""
    id: str
    forward_bps: float | None = None
    latency_s: float = 0.0
    cpu: str | None = None
    cycles_per_byte: float = 0.0


@dataclass
class PowerBudget:
    """A power budget compute draws energy from — a CPU package's limit (RAPL PL1 / PL2, else the documented TDP).
    Not part of any path: compute nodes name it (Compute.budget)."""
    id: str
    watts: float | None
    basis: str = ""

