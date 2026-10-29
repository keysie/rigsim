"""Energy per instruction per CPU microarchitecture, from measurements (docs/overhaul/cpu-compute.md): the only
calibrated constants of the CPU budget model.

    python3 -m app.hw.calibrate_cpu_energy <run folder> ...

Method: the stick's cpu-gemm, OpenBLAS sgemm 4096 x 4096 on all logical CPUs, is taken as bound by the package's power
budget (the specified sustained power: intel_client / amd_client.package_power_w). Its instruction mix is taken as one
256-bit FMA plus half a load per 16 FLOPs (an AVX2 sgemm micro-kernel's FMAs and its broadcast / load stream), and all
classes cost the same energy per instruction. Then energy per instruction = power / instructions per second. Per
microarchitecture the median over runs; hybrid parts (P- and E-cores together) give one value for both core types.
The result is pasted into cpu_uarch.ENERGY_J with the runs it came from.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from .. import inventory as inv
from . import amd_client, cpu_uarch, intel_client

FLOPS_PER_FMA = 16                 # 256-bit fp32 FMA: 8 lanes x 2
INSTR_PER_FMA = 1.5                # the FMA plus half a load


def all_core_sgemm(run: Path) -> float | None:
    f = run / "q" / "cpu-gemm" / "results.jsonl"
    vals = []
    for line in (f.read_text().splitlines() if f.exists() else []):
        r = json.loads(line)
        if r.get("rc") == 0 and r["params"].get("prec") == "fp32" and r["params"].get("n") == 4096 \
                and r["params"].get("threads") == 0:
            vals += [x["ops_per_s"] for x in r.get("results", []) if x.get("ops_per_s")]
    return statistics.median(vals) if vals else None


def uarchs(cpu: str, run: Path) -> list[str]:
    s = intel_client.spec(cpu, inv.core_counts(run))
    if s:
        return sorted({cpu_uarch.lookup(u)["uarch"] for u in s["core_uarch"].values() if cpu_uarch.lookup(u)})
    fam = amd_client.family(cpu)
    u = cpu_uarch.lookup(amd_client.FAMILIES[fam]["core"]) if fam else None
    return [u["uarch"]] if u else []


def calibrate(runs: list[Path]) -> dict[str, dict]:
    per: dict[str, list[tuple[float, str]]] = {}
    for run in runs:
        cpu = inv.cpu_name(run)
        watts, _ = (intel_client.package_power_w(cpu) if intel_client.generation(cpu) else amd_client.package_power_w(cpu))
        flops = all_core_sgemm(run)
        if not watts or not flops:
            continue
        e = watts / (flops / FLOPS_PER_FMA * INSTR_PER_FMA)
        for u in uarchs(cpu, run):
            per.setdefault(u, []).append((e, run.name))
    return {u: dict(energy_j=statistics.median(e for e, _ in v), runs=[r for _, r in v],
                    spread=(min(e for e, _ in v), max(e for e, _ in v))) for u, v in per.items()}


if __name__ == "__main__":
    for u, d in sorted(calibrate([Path(p) for p in sys.argv[1:]]).items()):
        print(f"{u:12} {d['energy_j'] * 1e9:.3f} nJ/instr  (min {d['spread'][0] * 1e9:.3f}, max {d['spread'][1] * 1e9:.3f}; "
              f"{len(d['runs'])} runs: {', '.join(r[-4:] for r in d['runs'])})")
