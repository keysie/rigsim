import { useEffect, useMemo, useState } from "react";
import { api, useStore, type Catalog, type Device, type Machine } from "../store";
import { Bar, Field, LinkTicks, Notices, Select } from "../components/ui";

type DevInfo = { link_gen: number; link_width: number; link_bw_gbs: number; link_type?: string; power_w?: number | null;
  power_range?: [number, number] | null; power_bw_factor?: number; power_compute_factor?: number };
type Validation = {
  ok: boolean; errors: string[]; warnings: string[];
  cpus: Record<string, { name: string; cores: number; lanes_total: number; lanes_used: number; pcie_gen: number;
    ram_bw_gbs: number; mem_type: string; isa: string; eff_cores: number; expert_gbs_per_core: number }>;
  devices: Record<string, { link_gen: number; link_width: number; link_bw_gbs: number }>;
};

const nextId = (kind: string, m: Machine) => {
  const used = new Set(m.devices.map((d) => d.id));
  for (let i = 1; ; i++) { const id = `${kind}-${i}`; if (!used.has(id)) return id; }
};

export default function MachinePage() {
  const catalog = useStore((s) => s.catalog)!;
  const machine = useStore((s) => s.machine);
  const setMachine = useStore((s) => s.setMachine);
  const [v, setV] = useState<Validation | null>(null);

  useEffect(() => {
    const t = setTimeout(() => api<Validation>("/api/machine/validate", machine).then(setV).catch(() => setV(null)), 150);
    return () => clearTimeout(t);
  }, [machine]);

  const cpuSpec = (i: number) => catalog.cpus.find((c) => c.id === machine.cpus[i].cpu_id)!;
  const twoCpuOk = cpuSpec(0).sockets >= 2;
  const activeCpus = machine.cpus.map((c, i) => ({ ...c, i })).filter((c) => c.enabled);

  const loadPrebuilt = useStore((s) => s.loadPrebuilt);
  const patch = (fn: (m: Machine) => void) => { const m = structuredClone(machine); fn(m); setMachine(m); };
  const setDevice = (id: string, fn: (d: Device) => void) => patch((m) => { fn(m.devices.find((d) => d.id === id)!); });
  const removeDevice = (id: string) => patch((m) => {
    m.devices = m.devices.filter((d) => d.id !== id);
    m.nvlinks = m.nvlinks.filter((p) => !p.includes(id));
  });
  const addDevice = (kind: "gpu" | "ssd") => patch((m) => {
    const cpu = activeCpus[0]?.i ?? 0;
    const gen = cpuSpec(cpu).pcie_gen;
    m.devices.push(kind === "gpu"
      ? { id: nextId("gpu", m), kind, model_id: "rtx-3090", cpu, link_gen: gen, link_width: 16 }
      : { id: nextId("ssd", m), kind, model_id: "nvme-gen4", cpu, link_gen: gen, link_width: 4, capacity_tb: 2 });
  });

  const gpus = machine.devices.filter((d) => d.kind === "gpu");
  const ssds = machine.devices.filter((d) => d.kind === "ssd");
  const gpuSpec = (d: Device) => catalog.gpus.find((g) => g.id === d.model_id)!;
  const ssdSpec = (d: Device) => catalog.ssds.find((g) => g.id === d.model_id)!;
  const bridged = (a: string, b: string) => machine.nvlinks.some((p) => p.includes(a) && p.includes(b));
  const canBridge = (a: Device, b: Device) => a.model_id === b.model_id && gpuSpec(a).nvlink_gbs > 0
    && !machine.nvlinks.some((p) => (p.includes(a.id) || p.includes(b.id)) && !(p.includes(a.id) && p.includes(b.id)));
  const toggleBridge = (a: string, b: string) => patch((m) => {
    m.nvlinks = bridged(a, b) ? m.nvlinks.filter((p) => !(p.includes(a) && p.includes(b))) : [...m.nvlinks, [a, b]];
  });

  const years = useMemo(() => [...catalog.cpus].sort((a, b) => a.vendor.localeCompare(b.vendor) || a.year - b.year), [catalog]);
  const gpuOptions = useMemo(() => [...catalog.gpus]
    .filter((g) => { const soc = catalog.soc_gpu_for?.[g.id]; return !soc || machine.cpus.some((c) => c.enabled && c.cpu_id === soc); })   // an integrated GPU belongs to its SoC only
    .sort((a, b) => a.vendor.localeCompare(b.vendor) || a.year - b.year)
    .map((g) => ({ value: g.id, label: `${g.name} · ${g.vram_gb} GB · ${g.year}${g.proposed ? " · community, unverified" : ""}` })), [catalog, machine.cpus]);

  return (
    <div>
      <h1>Machine</h1>
      <p className="mt-1 text-muted">Every part changes the lane budget and the bandwidths below. Errors block the simulation; warnings do not.</p>
      <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
        <span className="text-muted">Prebuilt system</span>
        <Select value="" ariaLabel="prebuilt system" onChange={(v) => { const pb = catalog.prebuilts.find((p) => p.id === v); if (pb) loadPrebuilt(pb.machine); }}
                options={[{ value: "", label: "pick one to fill the page…" }, ...catalog.prebuilts.map((p) => ({ value: p.id, label: p.name }))]} />
        <span className="text-muted">Operating system</span>
        <Select value={machine.os ?? "linux"} options={[{ value: "linux", label: "Linux" }, { value: "windows", label: "Windows (WDDM: slower CPU and PCIe expert traffic)" }]}
                onChange={(v) => patch((m) => { m.os = v as "linux" | "windows"; })} />
      </div>

      <section className="mt-6 rounded-lg border border-line bg-panel p-5 board">
        <div className="grid gap-6 md:grid-cols-2">
          {machine.cpus.map((slot, i) => {
            const spec = catalog.cpus.find((c) => c.id === slot.cpu_id)!;
            const info = v?.cpus[String(i)];
            const speeds = catalog.ram_speeds[spec.mem_type] ?? [];
            const disabled = !slot.enabled;
            return (
              <div key={i} className={`min-w-0 rounded-md border p-4 ${disabled ? "border-dashed border-line opacity-60" : "border-mask/40"}`}>
                <div className="flex items-center justify-between">
                  <h2>CPU {i + 1}</h2>
                  {i === 1 && (
                    <label className="flex items-center gap-2 text-sm">
                      <input type="checkbox" checked={slot.enabled} disabled={!twoCpuOk}
                             onChange={(e) => patch((m) => { m.cpus[1].enabled = e.target.checked; m.cpus[1].cpu_id = m.cpus[0].cpu_id;
                               if (!e.target.checked) m.devices.forEach((d) => { if (d.cpu === 1) d.cpu = 0; }); })} />
                      second socket{!twoCpuOk && <span className="text-xs text-muted">(not for {spec.name})</span>}
                    </label>
                  )}
                </div>
                <div className="mt-3 grid min-w-0 gap-3 sm:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
                  <Field label="Processor">
                    <Select className="w-full min-w-0" ariaLabel={`processor CPU ${i + 1}`} disabled={disabled || i === 1} value={slot.cpu_id}
                            options={years.map((c) => ({ value: c.id, label: `${c.name} · ${c.cores}c · ${c.year}${c.proposed ? " · community, unverified" : ""}` }))}
                            onChange={(id) => patch((m) => switchCpu(m, id, catalog))} />
                  </Field>
                  <div className="text-xs text-muted self-end leading-5">
                    {spec.cores} cores{spec.e_cores ? ` (${spec.cores - spec.e_cores}P + ${spec.e_cores}E)` : ""} / {spec.threads} threads · {spec.isa.toUpperCase()} · {spec.all_core_ghz} GHz<br />
                    PCIe {spec.pcie_gen}.0, {spec.pcie_lanes} lanes<br />{spec.mem_channels}-ch {spec.mem_type}
                  </div>
                </div>
                <div className="mt-4 flex items-end gap-4">
                  <Dimms channels={slot.ram_channels} max={spec.mem_channels} />
                  <div className="grid flex-1 grid-cols-3 gap-2">
                    <Field label="RAM">
                      <Select disabled={disabled} value={slot.ram_gb} options={(catalog.soc_ram_gb[slot.cpu_id] ?? catalog.ram_sizes_gb).map((g) => ({ value: g, label: `${g} GB` }))}
                              onChange={(g) => patch((m) => { m.cpus[i].ram_gb = g; })} />
                    </Field>
                    <Field label="Speed">
                      <Select disabled={disabled} value={slot.ram_mts} options={speeds.map((s) => ({ value: s, label: `${s} MT/s` }))}
                              onChange={(s) => patch((m) => { m.cpus[i].ram_mts = s; })} />
                    </Field>
                    <Field label="Channels">
                      <Select disabled={disabled} value={slot.ram_channels}
                              options={Array.from({ length: spec.mem_channels }, (_, k) => ({ value: k + 1, label: String(k + 1) }))}
                              onChange={(c) => patch((m) => { m.cpus[i].ram_channels = c; })} />
                    </Field>
                  </div>
                </div>
                {info && slot.enabled && (
                  <div className="mt-4 grid gap-2">
                    <Bar value={info.lanes_used} max={info.lanes_total} label={`${info.lanes_used} of ${info.lanes_total} PCIe lanes`} />
                    <div className="text-xs text-muted">Memory pool {slot.ram_gb} GB at about {info.ram_bw_gbs} GB/s · expert compute about {info.expert_gbs_per_core} GB/s per core</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="mt-6 grid gap-6 md:grid-cols-[2fr_3fr]">
          <div>
            <div className="flex items-center justify-between">
              <h2>Storage</h2>
              <button className="btn" disabled={machine.devices.filter((d) => d.kind === "ssd").length >= 8} title="at most 8 drives" onClick={() => addDevice("ssd")}>Add SSD</button>
            </div>
            <div className="mt-3 flex flex-col gap-2">
              {ssds.length === 0 && <p className="text-sm text-muted">No SSD yet. Models load from here; add one to measure load times.</p>}
              {ssds.map((d) => {
                const spec = ssdSpec(d); const dv = v?.devices[d.id];
                return (
                  <div key={d.id} className="rounded border border-line bg-paper p-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-medium text-sm w-12">{d.id}</span>
                      <Select value={d.model_id} options={catalog.ssds.map((s) => ({ value: s.id, label: s.name }))}
                              onChange={(id) => setDevice(d.id, (x) => { const s = catalog.ssds.find((y) => y.id === id)!; x.model_id = id;
                                x.link_gen = Math.min(x.link_gen, s.pcie_gen || x.link_gen); x.link_width = s.pcie_lanes || 0;
                                x.capacity_tb = s.capacities_tb.includes(x.capacity_tb ?? 0) ? x.capacity_tb : s.capacities_tb[0]; })} />
                      <Select value={d.capacity_tb ?? spec.capacities_tb[0]} options={spec.capacities_tb.map((c) => ({ value: c, label: `${c} TB` }))}
                              onChange={(c) => setDevice(d.id, (x) => { x.capacity_tb = c; })} />
                      <button className="btn btn-quiet ml-auto" onClick={() => removeDevice(d.id)} aria-label={`Remove ${d.id}`}>Remove</button>
                    </div>
                    <LinkRow d={d} maxGen={spec.pcie_gen} maxWidth={spec.pcie_lanes} dv={dv} activeCpus={activeCpus.map((c) => c.i)} setDevice={setDevice} />
                  </div>
                );
              })}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between">
              <h2>GPUs</h2>
              <button className="btn" disabled={machine.devices.filter((d) => d.kind === "gpu").length >= 8} title="at most 8 GPUs" onClick={() => addDevice("gpu")}>Add GPU</button>
            </div>
            <div className="mt-3 flex flex-col gap-2">
              {gpus.length === 0 && <p className="text-sm text-muted">No GPU yet. Engines need at least one unless they run on the CPU.</p>}
              {gpus.map((d, k) => {
                const spec = gpuSpec(d); const dv = v?.devices[d.id]; const prev = gpus[k - 1];
                return (
                  <div key={d.id}>
                    {prev && (
                      <div className="flex items-center gap-2 pl-14 py-1 text-xs">
                        {canBridge(prev, d) || bridged(prev.id, d.id) ? (
                          <button className={`btn ${bridged(prev.id, d.id) ? "btn-primary" : ""}`} onClick={() => toggleBridge(prev.id, d.id)}>
                            {bridged(prev.id, d.id) ? `NVLink bridge · ${spec.nvlink_gbs} GB/s` : "Add NVLink bridge"}
                          </button>
                        ) : <span className="text-muted">{spec.nvlink_gbs ? "NVLink needs two identical GPUs next to each other" : "no NVLink on this model"}</span>}
                      </div>
                    )}
                    <div className="rounded border border-line bg-paper p-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium text-sm w-12">{d.id}</span>
                        <Select value={d.model_id} options={gpuOptions}
                                onChange={(id) => patch((m) => { const x = m.devices.find((y) => y.id === d.id)!; const g = catalog.gpus.find((y) => y.id === id)!;
                                  x.model_id = id; x.power_w = null; x.link_gen = Math.min(x.link_gen, g.pcie_gen); x.link_width = Math.min(x.link_width, g.pcie_lanes);
                                  m.nvlinks = m.nvlinks.filter((p) => !p.includes(d.id)); })} />
                        <span className="text-xs text-muted">{spec.vram_gb} GB · {spec.mem_bw_gbs} GB/s · {spec.fp16_tflops} TFLOPS</span>
                        <button className="btn btn-quiet ml-auto" onClick={() => removeDevice(d.id)} aria-label={`Remove ${d.id}`}>Remove</button>
                      </div>
                      <LinkRow d={d} maxGen={spec.pcie_gen} maxWidth={spec.pcie_lanes} dv={dv} activeCpus={activeCpus.map((c) => c.i)} setDevice={setDevice} />
                      <PowerRow d={d} dv={dv as DevInfo | undefined} setDevice={setDevice} />
                      <label className="mt-2 flex items-center gap-2 text-xs">
                        <input type="checkbox" aria-label={`display on ${d.id}`} checked={Boolean(d.display)} onChange={(ev) => setDevice(d.id, (x) => { x.display = ev.target.checked; })} />
                        a screen / desktop is attached to this card <span className="text-muted">(reserves VRAM: ~0.45 GB Linux, ~1.3 GB Windows)</span>
                      </label>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        {v && <Notices errors={v.errors} warnings={v.warnings} />}
      </section>
      <p className="mt-4 text-xs text-muted">Chipset lanes, M.2 slots behind the chipset and physical slot spacing are not modelled: every device is on CPU lanes.</p>
    </div>
  );
}

function Dimms({ channels, max }: { channels: number; max: number }) {
  return (
    <svg width={max * 10 + 2} height="44" aria-label={`${channels} of ${max} memory channels populated`}>
      {Array.from({ length: max }).map((_, i) => (
        <rect key={i} x={i * 10 + 1} y="2" width="6" height="40" rx="1.5" fill={i < channels ? "#1F6B5B" : "#E3E8E6"} stroke={i < channels ? "none" : "#C9D1CE"} />
      ))}
    </svg>
  );
}

function PowerRow({ d, dv, setDevice }: { d: Device; dv?: DevInfo; setDevice: (id: string, fn: (d: Device) => void) => void }) {
  const rng = dv?.power_range;
  if (!rng) return null;
  const val = d.power_w ?? rng[1];
  return (
    <div className="mt-2 flex flex-wrap items-center gap-3 text-xs">
      <label className="flex items-center gap-2">power cap
        <input type="range" min={rng[0]} max={rng[1]} step={5} value={val} aria-label={`power cap ${d.id}`} className="w-32"
               onChange={(e) => setDevice(d.id, (x) => { x.power_w = Number(e.target.value) === rng[1] ? null : Number(e.target.value); })} />
        <span className="w-14">{val} W</span>
      </label>
      {d.power_w ? (
        <span className="text-muted">{Math.round((1 - (dv?.power_compute_factor ?? 1)) * 100)} % slower prompts, {Math.round((1 - (dv?.power_bw_factor ?? 1)) * 100)} % slower writing</span>
      ) : <span className="text-muted">default limit ({rng[0]}–{rng[1]} W settable)</span>}
    </div>
  );
}

const LINK_TYPES = [{ value: "pcie", label: "PCIe" }, { value: "tb3", label: "Thunderbolt 3 (eGPU)" }, { value: "tb4", label: "TB4 / USB4" },
  { value: "usb4v2", label: "USB4 v2" }, { value: "tb5", label: "Thunderbolt 5" }];

/** Switch the CPU of a machine and keep everything around it consistent: memory settings the new part
 *  supports, the second socket only where it exists, and integrated GPUs that belong to the CPU. */
export function switchCpu(m: Machine, id: string, catalog: Catalog) {
  const c = catalog.cpus.find((x) => x.id === id);
  if (!c) return;
  const iGpuOf = catalog.soc_gpu_for ?? {};
  const socSizes = catalog.soc_ram_gb?.[id];
  m.cpus.forEach((s) => {
    s.cpu_id = id;
    s.ram_channels = c.mem_channels;      // a new CPU comes fully populated; reduce it by hand if the board does not
    s.ram_mts = (catalog.ram_speeds[c.mem_type] ?? []).includes(s.ram_mts) ? s.ram_mts : c.max_mem_mts;
    if (socSizes && !socSizes.includes(s.ram_gb)) s.ram_gb = socSizes.reduce((a, b) => (Math.abs(b - s.ram_gb) < Math.abs(a - s.ram_gb) ? b : a));
  });
  if (c.sockets < 2 && m.cpus[1]) { m.cpus[1].enabled = false; m.devices.forEach((d) => { d.cpu = 0; }); }
  // integrated GPUs: drop those of other SoCs, add this SoC's own
  m.devices = m.devices.filter((d) => !(d.kind === "gpu" && iGpuOf[d.model_id] && iGpuOf[d.model_id] !== id));
  const own = Object.entries(iGpuOf).find(([, cpu]) => cpu === id)?.[0];
  if (own && !m.devices.some((d) => d.model_id === own)) {
    let n = 1; while (m.devices.some((d) => d.id === `gpu-${n}`)) n++;
    m.devices.unshift({ id: `gpu-${n}`, kind: "gpu", model_id: own, cpu: 0, link_gen: 4, link_width: 16, link_type: "pcie" });
  }
  m.nvlinks = m.nvlinks.filter(([a, b]) => m.devices.some((d) => d.id === a) && m.devices.some((d) => d.id === b));
  m.devices.forEach((d) => { if ((d.link_type ?? "pcie") === "pcie") d.link_gen = Math.min(d.link_gen, Math.max(1, c.pcie_gen)); });
}

function LinkRow({ d, maxGen, maxWidth, dv, activeCpus, setDevice }: {
  d: Device; maxGen: number; maxWidth: number; dv?: { link_gen: number; link_width: number; link_bw_gbs: number; link_type?: string };
  activeCpus: number[]; setDevice: (id: string, fn: (d: Device) => void) => void;
}) {
  if (dv?.link_type === "soc") return <div className="mt-2 text-xs text-muted">unified memory · {dv.link_bw_gbs} GB/s shared with the CPU</div>;
  if (!maxWidth) return <div className="mt-2 text-xs text-muted">SATA, not on PCIe</div>;
  const widths = [1, 2, 4, 8, 16].filter((w) => w <= maxWidth);
  const external = (d.link_type ?? "pcie") !== "pcie";
  return (
    <div className="mt-2 flex flex-wrap items-center gap-3">
      <label className="flex items-center gap-1 text-xs">Link
        <Select ariaLabel={`link type ${d.id}`} value={d.link_type ?? "pcie"} options={LINK_TYPES}
                onChange={(t) => setDevice(d.id, (x) => { x.link_type = t as string; })} />
      </label>
      {external ? (
        <span className="text-xs text-muted">external · {dv ? `${dv.link_bw_gbs} GB/s per direction` : ""}</span>
      ) : (<>
      <label className="flex items-center gap-1 text-xs">Gen
        <Select value={d.link_gen} options={[1, 2, 3, 4, 5, 6].filter((g) => g <= Math.max(maxGen, 1)).map((g) => ({ value: g, label: String(g) }))}
                onChange={(g) => setDevice(d.id, (x) => { x.link_gen = g; })} />
      </label>
      <label className="flex items-center gap-1 text-xs">Width
        <Select value={d.link_width} options={widths.map((w) => ({ value: w, label: `x${w}` }))}
                onChange={(w) => setDevice(d.id, (x) => { x.link_width = w; })} />
      </label>
      </>)}
      {activeCpus.length > 1 && (
        <label className="flex items-center gap-1 text-xs">on CPU
          <Select value={d.cpu} options={activeCpus.map((c) => ({ value: c, label: String(c + 1) }))}
                  onChange={(c) => setDevice(d.id, (x) => { x.cpu = c; })} />
        </label>
      )}
      {dv && !external && <LinkTicks gen={dv.link_gen} width={dv.link_width} bw={dv.link_bw_gbs} />}
    </div>
  );
}
