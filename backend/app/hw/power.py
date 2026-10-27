"""Power curves (proposal §5): per primitive in a power-limited device, its scalable parameter vs the device's
power limit, as a share of the value at the full limit. Points are (limit / TDP, factor); linear in between, flat
beyond the ends. Curves without data are flat and marked unmeasured, never guessed."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field


@dataclass
class PowerCurve:
    points: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 1.0), (1.0, 1.0)])
    measured: bool = False
    source: str = "unmeasured: flat"

    def factor(self, limit_ratio: float) -> float:
        pts = sorted(self.points)
        if limit_ratio <= pts[0][0]:
            return pts[0][1]
        for (r0, f0), (r1, f1) in zip(pts, pts[1:]):
            if r0 <= limit_ratio <= r1:
                return f0 + (f1 - f0) * (limit_ratio - r0) / (r1 - r0) if r1 > r0 else f1
        return pts[-1][1]

    @classmethod
    def from_csv(cls, text: str, tdp_w: float, source: str = "user CSV") -> "PowerCurve":
        """CSV with columns power_w,value (a clock in MHz or a bandwidth); normalised to the value at the highest
        power listed, power to TDP."""
        rows = [(float(r["power_w"]), float(r["value"])) for r in csv.DictReader(io.StringIO(text.strip()))]
        if len(rows) < 2:
            raise ValueError("a power curve needs at least two points")
        top = max(rows)[1]
        return cls(points=[(w / tdp_w, v / top) for w, v in rows], measured=True, source=source)
