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
        self.interleaved: dict[str, list[str]] = {}

    def interleave(self, name: str, storages: list[str]) -> None:
        """A memory controller spreads addresses evenly over these storages (e.g. one per channel): a stream read
        from `name` puts an equal share of its bytes on each member and its path."""
        for s in storages:
            if not isinstance(self.nodes.get(s), Storage):
                raise KeyError(f"interleave {name}: {s} is not a storage")
        self.interleaved[name] = list(storages)

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

    def idle_latency_s(self, compute: str, storage: str, direction: str = "read", seek: bool = True) -> float:
        """One access with nothing else running: link and router latencies on the path plus the storage's positioning
        (seek) and access time for that direction. For an interleaved set: the mean over its members."""
        if storage in self.interleaved:
            members = self.interleaved[storage]
            return sum(self.idle_latency_s(compute, s, direction, seek) for s in members) / len(members)
        st = self.nodes[storage]
        t = st.access(direction)
        if t is None:
            raise ValueError(f"access time of {storage} not identified")
        t += (st.seek_s or 0.0) if seek else 0.0
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
            members = self.interleaved.get(rd.storage, [rd.storage])
            share = 1.0 / len(members)
            used: dict[str, float] = {}
            own_cap, unknown = None, []
            for member in members:
                st = self.nodes[member]
                src = member
                for l, reached in self.path(member, rd.compute):       # data flows storage -> compute
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
                            capacity[key] = l.bandwidth_bps; used[key] = used.get(key, 0.0) + share
                    node = self.nodes[reached]
                    if isinstance(node, Router):
                        if node.forward_bps is None:
                            unknown.append(node.id)
                        else:
                            key = f"router:{node.id}"
                            capacity[key] = node.forward_bps; used[key] = used.get(key, 0.0) + share
                    src = reached
                if st.bps("read") is None:
                    unknown.append(st.id)
                else:
                    key = f"storage:{st.id}"
                    capacity[key] = st.bps("read"); used[key] = used.get(key, 0.0) + share
            flows[rd.id] = (float(units), used, own_cap)
            unidentified[rd.id] = sorted(set(unknown))
        rates, bottleneck = max_min_fair_share(capacity, flows)
        return {f: {"rate_bps": rates[f], "bottleneck": bottleneck[f], "unidentified_on_path": unidentified[f]}
                for f in flows}


@dataclass
class Transfer:
    """Bytes moving from a storage to a storage or a compute, concurrently with other transfers. `units` (for a
    compute destination) is how many of its units receive, each over its own per-unit link."""
    id: str
    src: str
    dst: str
    units: int | None = None


def _resources_along(m: "Machine", src: str, dst: str, share: float, units: int, capacity: dict, used: dict,
                     unknown: list) -> float | None:
    """Adds the resources a transfer src -> dst uses (links per direction, routers, router CPU cycles, source and
    destination storage bandwidth). Returns the cap from per-unit links, if any."""
    own_cap = None
    here = src
    for l, reached in m.path(src, dst):
        direction = "ab" if (l.a == here) else "ba"
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
                capacity[key] = l.bandwidth_bps; used[key] = used.get(key, 0.0) + share
        node = m.nodes[reached]
        if isinstance(node, Router) and reached != dst:
            if node.forward_bps is None:
                unknown.append(node.id)
            else:
                key = f"router:{node.id}"
                capacity[key] = node.forward_bps; used[key] = used.get(key, 0.0) + share
            if node.cpu:
                cpu = m.nodes[node.cpu]
                key = f"cycles:{cpu.id}"
                capacity[key] = cpu.clock_hz * cpu.count; used[key] = used.get(key, 0.0) + share * node.cycles_per_byte
        here = reached
    # a storage's time is its capacity: a read uses 1/read_bps of it per byte, a write 1/write_bps — so reads and
    # writes through one storage share it at their own speeds
    for end, direction in ((src, "read"), (dst, "write")):
        st = m.nodes[end]
        if isinstance(st, Storage):
            bw = st.bps(direction)
            if bw is None:
                unknown.append(st.id)
            elif bw > 0:
                key = f"storage:{st.id}"
                capacity[key] = 1.0; used[key] = used.get(key, 0.0) + share / bw
    return own_cap


def solve_transfers(m: "Machine", transfers: list[Transfer]) -> dict[str, dict]:
    """Rates of concurrent transfers under max-min fair share. Interleaved sets spread over their members."""
    capacity: dict[str, float] = {}
    flows: dict[str, tuple[float, dict, float | None]] = {}
    unidentified: dict[str, list[str]] = {}
    for t in transfers:
        units = 1
        for end in (t.dst, t.src):                    # a compute at either end: its units (a read or a write)
            node = m.nodes.get(end)
            if isinstance(node, Compute):
                units = node.count if t.units is None else min(t.units, node.count)
                break
        srcs = m.interleaved.get(t.src, [t.src])
        dsts = m.interleaved.get(t.dst, [t.dst])
        used: dict[str, float] = {}
        unknown: list[str] = []
        own_cap = None
        for a in srcs:
            for b in dsts:
                cap = _resources_along(m, a, b, 1.0 / (len(srcs) * len(dsts)), units, capacity, used, unknown)
                if cap is not None:
                    own_cap = cap if own_cap is None else min(own_cap, cap)
        flows[t.id] = (float(units), used, own_cap)
        unidentified[t.id] = sorted(set(unknown))
    rates, bottleneck = max_min_fair_share(capacity, flows)
    return {f: {"rate_bps": rates[f], "bottleneck": bottleneck[f], "unidentified_on_path": unidentified[f]}
            for f in flows}
