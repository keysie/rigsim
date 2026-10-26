"""Max-min fair share by progressive filling (weighted). Pure Python, small inputs."""
from __future__ import annotations


def max_min_fair_share(capacity: dict[str, float], flows: dict[str, tuple[float, list[str], float | None]]):
    """capacity: resource -> bytes/s. flows: id -> (weight, resources used, own cap or None).
    Each active flow grows at weight x lambda until a resource it uses is full or it reaches its own cap.
    Returns (rates, bottleneck) with bottleneck[flow] = the resource (or "own cap") that froze it."""
    rate = {f: 0.0 for f in flows}
    left = dict(capacity)
    active = {f for f, (w, res, cap) in flows.items() if w > 0}
    bottleneck: dict[str, str | None] = {f: None for f in flows}
    while active:
        best, why = float("inf"), None
        for r, c in left.items():
            w = sum(flows[f][0] for f in active if r in flows[f][1])
            if w > 0 and c / w < best:
                best, why = c / w, r
        for f in active:
            w, _, cap = flows[f]
            if cap is not None and (cap - rate[f]) / w < best:
                best, why = (cap - rate[f]) / w, ("own cap", f)
        if best == float("inf"):            # nothing binds: unidentified resources only
            for f in active:
                rate[f] = float("inf"); bottleneck[f] = "unbounded"
            break
        for f in active:
            rate[f] += flows[f][0] * best
            for r in flows[f][1]:
                left[r] -= flows[f][0] * best
        frozen = set()
        for f in active:
            w, res, cap = flows[f]
            if cap is not None and rate[f] >= cap * (1 - 1e-12):
                frozen.add(f); bottleneck[f] = "own cap"
            for r in res:
                if left[r] <= capacity[r] * 1e-12:
                    frozen.add(f); bottleneck[f] = bottleneck[f] or r
        if not frozen:                       # numerical guard
            frozen = set(active)
        active -= frozen
    return rate, bottleneck
