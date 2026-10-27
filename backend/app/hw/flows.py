"""Max-min fair share by progressive filling (weighted, with usage coefficients). Pure Python, small inputs."""
from __future__ import annotations


def max_min_fair_share(capacity: dict[str, float],
                       flows: dict[str, tuple[float, list[str] | dict[str, float], float | None]]):
    """capacity: resource -> bytes/s.
    flows: id -> (weight, resources, own cap or None). `resources` is a list (each used fully) or a dict
    resource -> share of the flow's bytes it carries (an interleaved stream puts 1/n on each of n channels).
    Each active flow grows at weight x lambda until a resource it uses is full or it reaches its own cap.
    Returns (rates, bottleneck)."""
    use = {f: (dict.fromkeys(r, 1.0) if isinstance(r, list) else dict(r)) for f, (_, r, _) in flows.items()}
    rate = {f: 0.0 for f in flows}
    left = dict(capacity)
    active = {f for f, (w, _, _) in flows.items() if w > 0}
    bottleneck: dict[str, str | None] = {f: None for f in flows}
    while active:
        best = float("inf")
        for r, c in left.items():
            w = sum(flows[f][0] * use[f].get(r, 0.0) for f in active)
            if w > 0 and c / w < best:
                best = c / w
        for f in active:
            w, _, cap = flows[f]
            if cap is not None and (cap - rate[f]) / w < best:
                best = (cap - rate[f]) / w
        if best == float("inf"):
            for f in active:
                rate[f] = float("inf"); bottleneck[f] = "unbounded"
            break
        for f in active:
            rate[f] += flows[f][0] * best
            for r, k in use[f].items():
                left[r] -= flows[f][0] * best * k
        frozen = set()
        for f in active:
            cap = flows[f][2]
            if cap is not None and rate[f] >= cap * (1 - 1e-12):
                frozen.add(f); bottleneck[f] = "own cap"
            for r in use[f]:
                if left[r] <= capacity[r] * 1e-12:
                    frozen.add(f); bottleneck[f] = bottleneck[f] or r
        if not frozen:
            frozen = set(active)
        active -= frozen
    return rate, bottleneck
