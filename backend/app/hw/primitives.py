"""The four hardware primitives (proposal §5). Units: bytes, seconds, bytes per second.

A value of None means "not identified": the primitive has the property, but no measurement fixes it. A None
bandwidth is treated as non-binding, and every result that crosses it says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Storage:
    """Capacity, access time, bandwidth. Nothing more (access patterns are the engine's to state).
    access_s None = not identified (latency results that need it say so)."""
    id: str
    capacity_b: float
    bandwidth_bps: float | None
    access_s: float | None

    def effective_bps(self, access_size_b: float) -> float | None:
        """Rate for accesses of a given size: size / (access time + size / bandwidth)."""
        if self.bandwidth_bps is None:
            return None
        return access_size_b / ((self.access_s or 0.0) + access_size_b / self.bandwidth_bps)


@dataclass
class Compute:
    """Clock and per-class rates as multiples of it; `count` identical units (a core group is one object)."""
    id: str
    clock_hz: float
    ops_per_clock: dict[str, float] = field(default_factory=dict)   # per unit, per operation class
    count: int = 1

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
