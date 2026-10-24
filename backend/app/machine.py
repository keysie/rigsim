"""Validate a machine and derive its link and memory bandwidths."""
from __future__ import annotations

from .catalog.cpus import CPU_BY_ID, E_CORE_WEIGHT, ISA_GBS_PER_GHZ, OS_FACTOR, RAM_SPEEDS
from .catalog.cpus import SOC_RAM_BW
from .catalog.gpus import EXTERNAL_LINKS, GPU_BY_ID, PCIE_LANE_GBS, SOC_GPU_FOR, SSD_BY_ID, power_factors, power_range
from .schemas import Machine

RAM_EFFICIENCY = 0.72      # measured stream share of the theoretical channel bandwidth


def ram_bandwidth_gbs(channels: int, mts: int) -> float:
    return channels * mts * 8 / 1000 * RAM_EFFICIENCY


def link_bandwidth_gbs(gen: int, width: int) -> float:
    return PCIE_LANE_GBS.get(gen, 0.0) * width


def device_spec(dev):
    return GPU_BY_ID[dev.model_id] if dev.kind == "gpu" else SSD_BY_ID[dev.model_id]


def validate(m: Machine) -> dict:
    errors, warnings = [], []
    cpus = []
    for i, slot in enumerate(m.cpus):
        spec = CPU_BY_ID.get(slot.cpu_id)
        if spec is None:
            errors.append(f"CPU {i + 1}: unknown part '{slot.cpu_id}'")
            cpus.append(None)
            continue
        cpus.append(spec)
    if len(m.cpus) == 2 and m.cpus[1].enabled:
        a, b = cpus
        if a and b:
            if a["id"] != b["id"]:
                errors.append("Both CPUs must be the same part")
            elif a["sockets"] < 2:
                errors.append(f"{a['name']} does not support a second socket")
    active = [i for i, s in enumerate(m.cpus) if s.enabled and cpus[i]]
    if not active:
        errors.append("At least one CPU must be enabled")

    per_cpu = {}
    for i in active:
        spec, slot = cpus[i], m.cpus[i]
        if slot.ram_channels > spec["mem_channels"]:
            errors.append(f"CPU {i + 1}: {slot.ram_channels} memory channels, {spec['name']} has {spec['mem_channels']}")
        if slot.ram_channels < spec["mem_channels"]:
            warnings.append(f"CPU {i + 1}: only {slot.ram_channels} of {spec['mem_channels']} channels populated")
        if slot.ram_mts > spec["max_mem_mts"]:
            warnings.append(f"CPU {i + 1}: {slot.ram_mts} MT/s is above the rated {spec['max_mem_mts']} (overclock)")
        if slot.ram_mts not in RAM_SPEEDS.get(spec["mem_type"], []):
            warnings.append(f"CPU {i + 1}: {slot.ram_mts} MT/s is not a standard {spec['mem_type']} speed")
        eff_cores = (spec["cores"] - spec["e_cores"]) + E_CORE_WEIGHT * spec["e_cores"]
        per_core = ISA_GBS_PER_GHZ[spec["isa"]] * spec["all_core_ghz"] * OS_FACTOR[m.os]
        per_cpu[i] = dict(
            isa=spec["isa"], e_cores=spec["e_cores"], all_core_ghz=spec["all_core_ghz"], eff_cores=round(eff_cores, 1),
            expert_gbs_per_core=round(per_core, 2), os=m.os,
            name=spec["name"], cores=spec["cores"], threads=spec["threads"], lanes_total=spec["pcie_lanes"], tdp_w=spec["tdp_w"],
            lanes_used=0, pcie_gen=spec["pcie_gen"], ram_gb=slot.ram_gb, ram_channels=slot.ram_channels,
            ram_mts=slot.ram_mts,
            ram_bw_gbs=(float(SOC_RAM_BW[spec["id"]]) if spec["id"] in SOC_RAM_BW
                        else round(ram_bandwidth_gbs(slot.ram_channels, slot.ram_mts), 1)),
            soc=spec["id"] if spec["id"] in SOC_RAM_BW else None,
            mem_type=spec["mem_type"], devices=[])

    devices = {}
    ids = set()
    for d in m.devices:
        if d.id in ids:
            errors.append(f"Duplicate device id '{d.id}'")
        ids.add(d.id)
        table = GPU_BY_ID if d.kind == "gpu" else SSD_BY_ID
        spec = table.get(d.model_id)
        if spec is None:
            errors.append(f"{d.id}: unknown {d.kind} '{d.model_id}'")
            continue
        if d.cpu not in per_cpu:
            errors.append(f"{d.id}: attached to CPU {d.cpu + 1}, which is not enabled")
            continue
        cpu = per_cpu[d.cpu]
        soc_of = SOC_GPU_FOR.get(d.model_id) if d.kind == "gpu" else None
        link_type = "soc" if soc_of else d.link_type
        if soc_of is not None:
            if cpu.get("soc") != soc_of:
                errors.append(f"{d.id}: {spec['name']} is the integrated GPU of '{soc_of}' and cannot be attached to {cpu['name']}")
            gen, width, bw = 0, 0, cpu["ram_bw_gbs"]           # unified memory: the link IS the pool
        elif d.link_type != "pcie":
            gen, width = 0, 0
            bw = EXTERNAL_LINKS[d.link_type]
            if d.kind == "ssd":
                bw = min(bw, spec["seq_read_gbs"])
            integrated = cpu.get("soc") is not None or CPU_BY_ID[m.cpus[d.cpu].cpu_id]["year"] >= 2022
            where = "integrated port, no CPU lanes" if integrated else "discrete controller or add-in card, 4 CPU lanes"
            if not integrated:
                cpu["lanes_used"] += 4
            warnings.append(f"{d.id}: external {d.link_type.upper()} link ({where}), ~{EXTERNAL_LINKS[d.link_type]:g} GB/s per direction — "
                            "fine for models resident in VRAM, slow for expert streaming, tensor parallel or frequent weight loads")
        elif d.kind == "ssd" and spec["pcie_lanes"] == 0:
            gen, width, bw = 0, 0, spec["seq_read_gbs"]
        else:
            gen = min(d.link_gen, cpu["pcie_gen"], spec["pcie_gen"])
            width = min(d.link_width, spec["pcie_lanes"])
            if d.link_gen > cpu["pcie_gen"]:
                warnings.append(f"{d.id}: link gen {d.link_gen} capped at the CPU's gen {cpu['pcie_gen']}")
            if d.link_gen > spec["pcie_gen"]:
                warnings.append(f"{d.id}: link gen {d.link_gen} capped at the device's gen {spec['pcie_gen']}")
            if d.link_width > spec["pcie_lanes"]:
                warnings.append(f"{d.id}: width x{d.link_width} capped at the device's x{spec['pcie_lanes']}")
            cpu["lanes_used"] += width
            bw = link_bandwidth_gbs(gen, width)
            if d.kind == "ssd":
                bw = min(bw, spec["seq_read_gbs"])
        if d.kind == "ssd" and d.capacity_tb is not None and d.capacity_tb not in spec["capacities_tb"]:
            warnings.append(f"{d.id}: {d.capacity_tb} TB is not a listed capacity for {spec['name']}")
        if d.kind == "gpu" and spec["form"] not in ("pcie", "soc"):
            warnings.append(f"{d.id}: {spec['name']} is an {spec['form'].upper()} module, modelled as a x16 link")
        power_w, pf = None, (1.0, 1.0)
        if d.kind == "gpu":
            rng = power_range(spec)
            if rng is not None:
                power_w = float(d.power_w) if d.power_w else float(rng[1])
                if power_w < rng[0] or power_w > rng[1]:
                    errors.append(f"{d.id}: power cap {power_w:g} W outside {spec['name']}'s range {rng[0]}–{rng[1]} W")
                    power_w = min(max(power_w, rng[0]), rng[1])
                pf = power_factors(spec, power_w)
            elif d.power_w:
                warnings.append(f"{d.id}: {spec['name']} has no settable power cap; the value is ignored")
        idle_w = round((spec["tdp_w"] if d.kind == "gpu" else 5) * (0.08 if spec.get("form") == "soc" else 0.07), 1)
        devices[d.id] = dict(id=d.id, kind=d.kind, model_id=d.model_id, name=spec["name"], cpu=d.cpu, link_gen=gen,
                             link_width=width, link_bw_gbs=round(bw, 2), link_type=link_type, spec=spec, capacity_tb=d.capacity_tb,
                             power_w=power_w, power_range=power_range(spec) if d.kind == "gpu" else None,
                             power_bw_factor=round(pf[0], 3), power_compute_factor=round(pf[1], 3), idle_w=idle_w)
        cpu["devices"].append(d.id)

    for i, cpu in per_cpu.items():
        if cpu["lanes_used"] > cpu["lanes_total"]:
            errors.append(f"CPU {i + 1}: {cpu['lanes_used']} PCIe lanes requested, {cpu['name']} has {cpu['lanes_total']}")

    gpu_order = [d.id for d in m.devices if d.kind == "gpu"]
    for a, b in m.nvlinks:
        if a not in devices or b not in devices or devices[a]["kind"] != "gpu" or devices[b]["kind"] != "gpu":
            errors.append(f"NVLink {a}–{b}: both ends must be GPUs")
            continue
        ia, ib = gpu_order.index(a), gpu_order.index(b)
        if abs(ia - ib) != 1:
            errors.append(f"NVLink {a}–{b}: only adjacent GPUs can be bridged")
        sa, sb = devices[a]["spec"], devices[b]["spec"]
        if sa["id"] != sb["id"]:
            errors.append(f"NVLink {a}–{b}: both GPUs must be the same model")
        elif sa["nvlink_gbs"] == 0:
            errors.append(f"NVLink {a}–{b}: {sa['name']} has no NVLink")
    linked = {x for pair in m.nvlinks for x in pair}
    for dev_id in linked:
        if sum(1 for p in m.nvlinks if dev_id in p) > 1:
            errors.append(f"{dev_id}: a GPU can carry only one bridge")

    return dict(ok=not errors, errors=errors, warnings=warnings, cpus=per_cpu,
                devices={k: {kk: vv for kk, vv in v.items() if kk != "spec"} for k, v in devices.items()},
                _devices=devices, nvlinks=[tuple(p) for p in m.nvlinks])
