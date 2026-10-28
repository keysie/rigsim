"""The RAM benchmark database (data/ram-bench-data.json, schema v1) as validation points, medium to low trust.

Trust rule: low if the entry is anecdotal, reader- or forum-submitted, rounded, partial, or misses primaries;
otherwise medium. Nothing is high trust: almost all entries are single runs without a published spread, mostly
AIDA64 on Windows, and absolute values differ between sources for the same hardware (up to 12 %). Compare within one
source's sequence only.
"""
from __future__ import annotations

import re

import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "ram-bench-data.json"
LOW_FLAGS = {"anecdotal", "reader_submitted", "rounded_values", "partial", "primaries_missing", "primaries_incomplete"}


def load() -> dict:
    return json.loads(DATA.read_text())


def val(field):
    return field.get("value") if isinstance(field, dict) else field


def trust(entry: dict, sources: dict) -> str:
    src = sources.get(entry["source_id"], {})
    if set(entry.get("quality") or []) & LOW_FLAGS or "forum" in (src.get("site", "") + src.get("title", "")).lower():
        return "low"
    return "medium"


def read_gbs(entry: dict) -> float | None:
    """All-cores read bandwidth in GB/s (AIDA64 'Read' or MLC 'ALL Reads'); a stated range uses its midpoint."""
    for m in entry["measurements"]:
        if m["metric"] in ("read_bw", "all_reads_bw"):
            scale = 1000 if m["unit"] == "MB/s" else 1
            if m["value"] is not None:
                return m["value"] / scale
            sp = m.get("spread") or {}
            if "min" in sp and "max" in sp:
                return (sp["min"] + sp["max"]) / 2 / scale
    return None


def peak_gbs(entry: dict) -> float | None:
    mts, ch = val(entry["memory"].get("mt_s")), val(entry["memory"].get("channels"))
    return ch * mts * 8 / 1000 if mts and ch else None


def points() -> list[dict]:
    db = load()
    sources = {s["id"]: s for s in db["sources"]}
    out = []
    for e in db["entries"]:
        rd = read_gbs(e)
        if rd is None or peak_gbs(e) is None:
            continue
        seq = e.get("sequence") or {}
        sysid = (val(e["system"].get("cpu")), val(e["system"].get("board")), val(e["system"].get("os")),
                 val(e["memory"].get("kit_part")))
        # one system = one sequence group; without a stated group, entries match only if CPU, board, OS and kit do
        group = seq.get("group") or (None if all(x is None for x in sysid[:2]) else repr(sysid))
        out.append(dict(id=e["id"], source=e["source_id"], group=group or e["id"],
                        step=seq.get("step"), change=seq.get("changed_from_previous"),
                        cpu=val(e["system"].get("cpu")), mts=val(e["memory"]["mt_s"]),
                        channels=val(e["memory"]["channels"]), fclk=val((e.get("clocks") or {}).get("fclk_mhz")),
                        mem_type=val(e["memory"].get("type")), density_gbit=density(e),
                        trefi_ns=_trefi_ns(e),
                        tool=val(e["tool"].get("name")), read_gbs=rd, peak_gbs=peak_gbs(e), trust=trust(e, sources)))
    return out


def density(e: dict) -> int | None:
    """Die density in Gbit: stated ("32 Gbit"), or from the stated die (SK Hynix M- and A-die: 16 Gb, INFERRED),
    or from a stated tRFCsb of 130 ns (the JEDEC 16 Gb value, INFERRED)."""
    ic = str(val(e["memory"].get("ic")) or "")
    m = re.search(r"(\d+)\s*Gbit", ic)
    if m:
        return int(m.group(1))
    if re.search(r"Hynix [MA]-die", ic):
        return 16
    t = e.get("timings") or {}
    sb, mts = val(t.get("tRFCsb")), val(e["memory"].get("mt_s"))
    if sb and mts and abs(sb * 2000 / mts - 130) < 3:
        return 16
    return None


def _trefi_ns(e: dict) -> float | None:
    t = e.get("timings") or {}
    clk, mts = val(t.get("tREFI")), val(e["memory"].get("mt_s"))
    return clk * 2000 / mts if clk and mts else None
