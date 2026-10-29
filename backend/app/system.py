"""The machine of a rigsim-benchmark run as ONE graph of the base blocks (proposal §5): every prediction about the
run is a question to this graph, and the graph can be drawn so its owner can say "yes, that is my machine".

  CPU package      arch.add_cpu: core groups, their links, the package router ('pkg', or 'iod' on chiplet parts),
                   the memory channels (DDR4 channel / DDR5 sub-channel each a link + storage), interleaved as 'ram'
  PCIe             every hop of a card's path from inventory/pcie.json as a router: root port, switch ports, a
                   chipset ('chipset', when the root port is a chipset / PCH port; its uplink capacity is not
                   identified), each link at the width and speed both ends support
  dedicated GPUs   gpu.add_gpu at the router of their own port
  integrated GPU   gpu.add_igpu on the package router, reading the same 'ram' as the cores — the bandwidth is shared

    python3 -m app.system <run folder>              the machine as a Mermaid diagram
    python3 -m app.system <run folder> --json       nodes and edges as JSON
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import inventory as inv
from .hw import amd_client, arch, intel_client, intel_igpu
from .hw.cpus import lookup as cpu_lookup
from .hw.gpu import add_gpu, add_igpu
from .hw.links import pcie_bps
from .hw.machine import Machine
from .hw.primitives import Compute, Link, Router, Storage

GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
CHIPSET = re.compile(r"(?i)\bPCH\b|chipset|promontory|\b[ABXZHQW]\d{3}\b.*(root port|express)")


@dataclass
class System:
    m: Machine
    io: str | None = None                        # the CPU package's router
    groups: list[str] = field(default_factory=list)
    cpu: dict = field(default_factory=dict)      # what the CPU and memory were built from
    cpu_why: str | None = None                   # why there is no CPU / memory in the graph
    gpus: dict = field(default_factory=dict)     # gid -> {pci, name, model, kind}
    by_pci: dict = field(default_factory=dict)   # pci -> gid
    cuda: dict = field(default_factory=dict)     # nvidia-smi / CUDA index -> gid
    labels: dict = field(default_factory=dict)   # node id -> what the owner calls it
    notes: list = field(default_factory=list)


def _gen(speed: str | None) -> int | None:
    return GEN.get(str(speed or "").split(" ")[0])


def _negotiated(a: dict, b: dict) -> tuple[int | None, int | None]:
    la, lb = a.get("link") or {}, b.get("link") or {}
    g = [x for x in (_gen(la.get("max_speed")), _gen(lb.get("max_speed"))) if x]
    w = [int(x) for x in (la.get("max_width"), lb.get("max_width")) if str(x or "").isdigit()]
    return (min(g) if g else None), (min(w) if w else None)


def _add_cpu(s: System, run: Path) -> None:
    name = inv.cpu_name(run)
    topo = inv.lscpu_topology(run)
    spec = (cpu_lookup(name) or intel_client.spec(name, inv.core_counts(run))
            or amd_client.spec(name, topo["cores"], topo["l3_slices"]))
    ds = inv.dimms(run)
    why = None
    if not spec:
        why = f"CPU not in rigsim's catalogue: {name or 'unknown'}"
    elif not ds:
        why = "no DIMM data in the inventory (stick before 0.11.0: dmidecode read nothing)"
    mts = min((d["mts"] for d in ds if d["mts"]), default=None)
    t = (ds[0]["type"] if ds else "").lower()
    if not why and (t not in ("ddr4", "ddr5") or not mts):
        why = f"memory not identified (type {ds[0]['type']!r}, speed {mts})"
    fclk = None
    if not why and spec.get("topology") == "chiplet":
        fam = amd_client.family(name)
        fclk = amd_client.fclk_mhz(fam, mts) if fam else None
        if not fclk:
            why = f"fabric clock rule not documented for {name}"
    if why:
        s.cpu_why = why
        s.io = "host"
        s.m.add(Router("host"))
        s.labels["host"] = f"CPU ({name or 'unknown'}) — not modelled: {why}"
        s.notes.append(why)
        return

    def channel(d: dict) -> str | None:
        for txt in (d["bank"], d["locator"]):
            mm = re.search(r"(?i)channel\s*([a-d])\b", txt) or re.search(r"(?i)(?:^|[-_ ])([a-d])\d$", txt)
            if mm:
                return mm.group(1).upper()
        return None
    chans = sorted({c for c in map(channel, ds) if c})
    channels = min(spec["channels"], len(chans) if chans else len(ds))
    spd = inv.spd_timing(run)
    cl = round(spd["taa_ns"] * mts / 2000) if spd.get("taa_ns") else None
    l3 = inv.l3_bytes(run)
    s.io, s.groups = arch.add_cpu(s.m, spec, mts, channels, fclk_mhz=fclk, cl=cl, mem_type=t,
                                  density_gbit=spd.get("density_gbit"), l3_bytes=l3)
    for c in [n for n, x in s.m.nodes.items() if type(x).__name__ == "Cache"]:
        s.labels[c] = f"L3 {s.m.nodes[c].capacity_b / 2**20:.0f} MiB (write-back, write-allocate)" + (
            f" on {c.split('.')[0]}" if "." in c else "")
    s.cpu = dict(name=name, cpu=spec.get("arch"),
                 cores=spec.get("cores") or {"ccds": spec.get("ccds"), **spec.get("cores_per_ccd", {})},
                 fclk_mhz=f"{fclk:.0f} (AMD's default rule, not read from the machine)" if fclk else None,
                 mem=f"{t.upper()}-{mts}", channels=channels, cl=cl, density_gbit=spd.get("density_gbit"),
                 channels_from="locators" if chans else "DIMM count")
    s.labels[s.io] = f"{name} — package" if s.io == "pkg" else f"{name} — I/O die"
    uarch = spec.get("core_uarch") or {}
    for g in s.groups:
        n, ct = s.m.nodes[g].count, g.split(".")[-1]
        kind = {"p": "P-cores", "e": "E-cores", "core": "cores"}.get(ct, f"{ct} cores")
        u = uarch.get(ct) if isinstance(uarch, dict) else uarch
        s.labels[g] = f"{n} {kind}" + (f" ({u})" if u and u != ct else "") + (f" on {g.split('.')[0]}" if "." in g else "")
    for d in range(spec.get("ccds") or 0):
        s.labels[f"ccd{d}"] = f"core die {d}"
    letters = chans or [chr(65 + i) for i in range(channels)]
    for nid in s.m.interleaved.get("ram", []):
        c = int(re.match(r"ch(\d+)", nid).group(1))
        sub = f" sub-channel {nid.split('sub')[1]}" if "sub" in nid else ""
        s.labels[nid] = f"{t.upper()}-{mts} channel {letters[c] if c < len(letters) else c}{sub}"
    s.labels["ram"] = f"system memory {t.upper()}-{mts}, {channels} channels"


def _port_router(s: System, run: Path, devs: dict, pci: str) -> tuple[str, int, int]:
    """The router of a card's own port, building its path from the CPU package; and the slot's gen and width."""
    path = (devs.get(pci) or {}).get("path") or []
    if len(path) < 2:
        s.notes.append(f"{pci}: no PCIe path in the inventory (pcie.json, format 1.2+): attached to the CPU as x16 gen 4")
        return s.io, 4, 16
    parent = s.io
    root = devs.get(path[0], {})
    if CHIPSET.search(root.get("name", "")):
        if "chipset" not in s.m.nodes:
            s.m.add(Router("chipset"))
            s.m.connect(Link("chipset.uplink", "chipset", s.io, None))
            s.labels["chipset"] = "chipset (uplink to the CPU: capacity not identified)"
        parent = "chipset"
    for i, hop in enumerate(path[:-1]):
        rid = f"pcie:{hop}"
        d = devs.get(hop, {})
        if rid not in s.m.nodes:
            s.m.add(Router(rid))
            if i == 0:
                s.m.connect(Link(f"{rid}.up", rid, parent, None))
            else:
                g, w = _negotiated(d, devs.get(path[i - 1], {}))
                s.m.connect(Link(f"{rid}.up", rid, f"pcie:{path[i - 1]}", pcie_bps(g, w) if g and w else None))
            lk = d.get("link") or {}
            s.labels[rid] = (f"{d.get('port_type') or 'PCIe port'} {hop[5:]}"
                             + (f" (max x{lk.get('max_width')} {lk.get('max_speed')})" if lk.get("max_width") else ""))
    port = devs.get(path[-2], {})
    g, w = _negotiated(devs.get(pci, {}), port)
    return f"pcie:{path[-2]}", g or 4, w or 16


def build_system(run: Path) -> System:
    s = System(Machine())
    _add_cpu(s, run)
    devs = inv.pcie_devices(run)
    limits = inv.power_limits(run)
    for g in inv.nvidia_gpus(run):
        mid = inv.catalogue_id(g["name"], g["mib"])
        gid = f"gpu{g['index']}"
        s.cuda[g["index"]] = gid
        if not mid:
            s.gpus[gid] = dict(pci=g["pci"], name=g["name"], model=None, kind="dgpu")
            s.notes.append(f"{g['name']} ({g['pci']}): not in rigsim's GPU catalogue")
            continue
        router, gen, lanes = _port_router(s, run, devs, g["pci"])
        add_gpu(s.m, gid, mid, router, slot_gen=gen, slot_lanes=lanes, power_limit_w=limits.get(g["pci"]))
        s.gpus[gid] = dict(pci=g["pci"], name=g["name"], model=mid, kind="dgpu")
        s.by_pci[g["pci"]] = gid
        s.labels[f"{gid}.sm"] = f"{g['name']} — {s.m.nodes[gid + '.sm'].count} SMs"
        s.labels[f"{gid}.vram"] = f"{g['name']} — VRAM"
        s.labels[f"{gid}.copy"] = f"{g['name']} — copy engines / PCIe x{lanes} gen {gen}"
        s.labels[f"{gid}.xbar"] = f"{g['name']} — crossbar"
        s.labels[f"{gid}.l2"] = f"{g['name']} — L2 {s.m.nodes[gid + '.l2'].capacity_b / 2**20:.0f} MB"
    intel = next((n for n in inv.gpu_names(run) if "Intel" in n and intel_igpu.product(n)), "")
    if intel:
        pci = "0000:00:02.0"
        clock = inv.igpu_clock_mhz(run, pci)
        if s.cpu_why:
            s.notes.append(f"{intel}: not modelled ({s.cpu_why})")
        elif not clock:
            s.notes.append(f"{intel}: no graphics clock recorded (samples-drm.jsonl)")
        else:
            t, mts = s.cpu["mem"].split("-")
            key = intel_igpu.register(intel, clock, t, float(mts), s.cpu["channels"], s.cpu.get("density_gbit"))
            add_igpu(s.m, "igpu", key, s.io, memory="ram")
            s.gpus["igpu"] = dict(pci=pci, name=intel, model=key, kind="igpu")
            s.by_pci[pci] = "igpu"
            s.labels["igpu.sm"] = f"{intel_igpu.product(intel)} — {s.m.nodes['igpu.sm'].count} EUs at {clock:.0f} MHz"
            s.labels["igpu.gti"] = "graphics interface to the ring"
    return s


# --- drawing ---------------------------------------------------------------------------------------------------------
def describe(s: System) -> dict:
    """Nodes and edges for a diagram; the members of an interleaved set (a card's VRAM slices) are drawn as their set,
    their links summed — except the system memory's channels, which an owner recognises."""
    m = s.m
    member_of = {mem: name for name, mems in m.interleaved.items() if name != "ram" for mem in mems}
    nodes, edges = {}, {}
    for nid, n in m.nodes.items():
        tid = member_of.get(nid, nid)
        kind = "compute" if isinstance(n, Compute) else "storage" if isinstance(n, Storage) else "router"
        nodes.setdefault(tid, dict(id=tid, kind=kind, label=s.labels.get(tid, tid)))
    if "ram" in m.interleaved:
        nodes["ram"] = dict(id="ram", kind="memory", label=s.labels.get("ram", "system memory"))
        for mem in m.interleaved["ram"]:
            edges[(mem, "ram")] = dict(a=mem, b="ram", gbs=None, label="interleaved")
    for l in m.links.values():
        a, b = member_of.get(l.a, l.a), member_of.get(l.b, l.b)
        e = edges.setdefault((a, b), dict(a=a, b=b, gbs=0.0, label="", per_unit=l.per_unit))
        if l.bandwidth_bps is None:
            e["gbs"] = None
        elif e["gbs"] is not None:
            e["gbs"] += l.bandwidth_bps / 1e9
    for e in edges.values():
        if e["label"]:
            continue
        if e["gbs"] is None:
            e["label"] = "not identified"
        else:
            e["label"] = f"{e['gbs']:.0f} GB/s" + (" per core" if e.get("per_unit") else "")
    return dict(nodes=list(nodes.values()), edges=list(edges.values()), notes=s.notes)


def mermaid(s: System) -> str:
    d = describe(s)
    shape = {"compute": ("[", "]"), "storage": ("[(", ")]"), "memory": ("[(", ")]"), "router": ("([", "])")}
    safe = lambda x: re.sub(r"[^A-Za-z0-9_]", "_", x)
    lines = ["flowchart TB"]
    for n in d["nodes"]:
        o, c = shape[n["kind"]]
        lines.append(f'  {safe(n["id"])}{o}"{n["label"]}"{c}')
    for e in d["edges"]:
        lines.append(f'  {safe(e["a"])} ---|"{e["label"]}"| {safe(e["b"])}')
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    s = build_system(a.run)
    print(json.dumps(describe(s), indent=1) if a.json else mermaid(s), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
