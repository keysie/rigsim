"""GPU clock against the power limit (proposal §5, spec-sheet first).

Default: the spec-sheet law. At the rated power (TDP) a compute-heavy load sustains SUSTAINED_CLOCK_FACTOR x the
spec boost clock; below it the clock falls as (limit / TDP)^CLOCK_POWER_EXPONENT; a memory-heavy load runs
MEMORY_LOAD_CLOCK_GAIN higher at the same limit. Three global constants for every card; nothing per card.

Override: a measured curve (the user's CSV in the UI) per primitive and load class — the absolute value (SM clock in
MHz, VRAM bandwidth in GB/s) against the power limit in watts, linear between points, flat beyond the ends.
Load classes: "compute" / "memory"; a state picks its class from its operation mix (loadclass.py), one pass, no
feedback between clock and load."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

from . import constants as K

LOAD_CLASSES = ("compute", "memory")


@dataclass
class PowerCurve:
    points: list[tuple[float, float]] = field(default_factory=list)      # (power limit W, value)
    basis: str = "measured"                                               # measured | inferred
    source: str = ""

    def value(self, limit_w: float) -> float:
        pts = sorted(self.points)
        if limit_w <= pts[0][0]:
            return pts[0][1]
        for (w0, v0), (w1, v1) in zip(pts, pts[1:]):
            if w0 <= limit_w <= w1:
                return v0 + (v1 - v0) * (limit_w - w0) / (w1 - w0) if w1 > w0 else v1
        return pts[-1][1]

    def covers(self, limit_w: float) -> bool:
        ws = [w for w, _ in self.points]
        return min(ws) <= limit_w <= max(ws)

    @classmethod
    def from_csv(cls, text: str, source: str = "user CSV") -> "PowerCurve":
        """CSV with columns power_w,value (absolute: MHz for a clock, GB/s for a bandwidth)."""
        rows = [(float(r["power_w"]), float(r["value"])) for r in csv.DictReader(io.StringIO(text.strip()))]
        if not rows:
            raise ValueError("a power curve needs at least one point")
        return cls(points=rows, basis="measured", source=source)


def spec_clock_mhz(spec: dict, limit_w: float, load_class: str) -> float:
    """The spec-sheet law: SM clock for a load class at a power limit, from boost clock and TDP only."""
    f = K.SUSTAINED_CLOCK_FACTOR * spec["boost_mhz"] * (limit_w / spec["tdp_w"]) ** K.CLOCK_POWER_EXPONENT
    return f * (K.MEMORY_LOAD_CLOCK_GAIN if load_class == "memory" else 1.0)
