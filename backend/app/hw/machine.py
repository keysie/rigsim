"""The machine as a graph of primitives; paths, idle latency and concurrent reads under max-min fair share."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .flows import max_min_fair_share
from .primitives import Compute, Link, Router, Storage


@dataclass
class Read:
    """`units` of a compute group read from a storage concurrently (e.g. 18 cores streaming from RAM)."""
    id: str
    compute: str
    storage: str
    units: int | None = None


class Machine:
    def __init__(self):
        self.nodes: dict[str, Storage | Compute | Router] = {}
        self.links: dict[str, Link] = {}
        self._adj: dict[str, list[Link]] = {}

    def add(self, obj):
        self.nodes[obj.id] = obj
        self._adj.setdefault(obj.id, [])
        return obj

    def connect(self, link: Link) -> Link:
        for end in (link.a, link.b):
            if end not in self.nodes:
                raise KeyError(f"link {link.id}: unknown node {end}")
        self.links[link.id] = link
        self._adj[link.a].append(link)
        self._adj[link.b].append(link)
        return link

    # --- paths -------------------------------------------------------------------------------------------------
    def path(self, src: str, dst: str) -> list[tuple[Link, str]]:
        """Fewest hops from src to dst; only routers may be passed through. Returns [(link, node reached), ...]."""
        prev: dict[str, tuple[Link, str] | None] = {src: None}
        q = deque([src])
        while q:
            n = q.popleft()
            if n == dst:
                break
            if n != src and not isinstance(self.nodes[n], Router):
                continue
            for l in self._adj[n]:
                m = l.b if l.a == n else l.a
                if m not in prev:
                    prev[m] = (l, n)
                    q.append(m)
        if dst not in prev:
            raise ValueError(f"no path {src} -> {dst}")
        hops, n = [], dst
        while prev[n] is not None:
            l, p = prev[n]
            hops.append((l, n))
            n = p
        return list(reversed(hops))

    def idle_latency_s(self, compute: str, storage: str) -> float:
        """One access with nothing else running: link and router latencies on the path plus the storage's access time."""
        t = self.nodes[storage].access_s
        for l, reached in self.path(compute, storage):
            t += l.latency_s
            node = self.nodes[reached]
            if isinstance(node, Router):
                t += node.latency_s
        return t

    # --- concurrent reads ----------------------------------------------------------------------------------------
    def solve_reads(self, reads: list[Read]) -> dict[str, dict]:
        """Rates of concurrent reads under max-min fair share. Each unit of a group is weighted equally."""
        capacity: dict[str, float] = {}
        flows: dict[str, tuple[float, list[str], float | None]] = {}
        unidentified: dict[str, list[str]] = {}
        for rd in reads:
            comp = self.nodes[rd.compute]
            units = comp.count if rd.units is None else min(rd.units, comp.count)
            used, own_cap, unknown = [], None, []
            st = self.nodes[rd.storage]
            hops = self.path(rd.storage, rd.compute)          # data flows storage -> compute
            src = rd.storage
            for l, reached in hops:
                direction = "ab" if (l.a == src) else "ba"
                if l.per_unit:
                    if l.bandwidth_bps is None:
                        unknown.append(l.id)
                    else:
                        cap = l.bandwidth_bps * units
                        own_cap = cap if own_cap is None else min(own_cap, cap)
                else:
                    key = f"link:{l.id}:{direction if l.duplex else 'shared'}"
                    if l.bandwidth_bps is None:
                        unknown.append(l.id)
                    else:
                        capacity[key] = l.bandwidth_bps; used.append(key)
                node = self.nodes[reached]
                if isinstance(node, Router):
                    if node.forward_bps is None:
                        unknown.append(node.id)
                    else:
                        capacity[f"router:{node.id}"] = node.forward_bps; used.append(f"router:{node.id}")
                src = reached
            if st.bandwidth_bps is None:
                unknown.append(st.id)
            else:
                capacity[f"storage:{st.id}"] = st.bandwidth_bps; used.append(f"storage:{st.id}")
            flows[rd.id] = (float(units), used, own_cap)
            unidentified[rd.id] = sorted(set(unknown))
        rates, bottleneck = max_min_fair_share(capacity, flows)
        return {f: {"rate_bps": rates[f], "bottleneck": bottleneck[f], "unidentified_on_path": unidentified[f]}
                for f in flows}
