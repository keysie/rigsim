import { useEffect, useState } from "react";
import { api, useStore, type ArgDef, type EngineSpec } from "../store";
import { Field, Select } from "../components/ui";
import ExportMeasurement from "../components/ExportMeasurement";

type Cluster = { key: { gpus: string[]; engine: string; model: string; quant: string; streams: number }; n: number; trust: number;
  results: Record<string, { median: number; n: number; spread: number }>; cpus: string[]; os: string[]; near: string | null };

type Derived = Record<string, {
  kind: string; slots: number; prefill_rate: number; decode_total: Record<string, number>;
  speed_per_request: Record<string, number>; memory: Record<string, unknown>; notes: string[]; errors: string[];
  limits: Record<string, number>;
}>;

export default function EnginesPage() {
  const catalog = useStore((s) => s.catalog)!;
  const machine = useStore((s) => s.machine);
  const engines = useStore((s) => s.engines);
  const setEngines = useStore((s) => s.setEngines);
  const [derived, setDerived] = useState<Derived>({});
  const [community, setCommunity] = useState<Record<string, Cluster[]>>({});
  const [exporting, setExporting] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      api<{ engines: Derived }>("/api/engines/derive", { machine, engines }).then((r) => setDerived(r.engines)).catch(() => setDerived({}));
      Promise.all(engines.map((e) => api<{ clusters: Cluster[] }>("/api/community/match", { machine, engine: e })
        .then((r) => [e.id, r.clusters] as const).catch(() => [e.id, []] as const)))
        .then((pairs) => setCommunity(Object.fromEntries(pairs)));
    }, 200);
    return () => clearTimeout(t);
  }, [machine, engines]);

  const useCluster = (id: string, c: Cluster) => patch(id, (x) => {
    const b = { ...(x.benchmarks ?? {}) };
    for (const [k, v] of Object.entries(c.results)) if (k !== "hit") b[k] = v.median;
    x.benchmarks = b;
  });

  const gpus = machine.devices.filter((d) => d.kind === "gpu");
  const ssds = machine.devices.filter((d) => d.kind === "ssd");
  const activeCpus = machine.cpus.map((c, i) => ({ ...c, i })).filter((c) => c.enabled);
  const cpuSpec = (i: number) => catalog.cpus.find((c) => c.id === machine.cpus[i].cpu_id)!;
  const patch = (id: string, fn: (e: EngineSpec) => void) => {
    const next = structuredClone(engines); fn(next.find((e) => e.id === id)!); setEngines(next);
  };
  const add = () => {
    const used = new Set(engines.map((e) => e.id));
    let n = engines.length + 1; while (used.has(`engine-${n}`)) n++;
    setEngines([...engines, { id: `engine-${n}`, name: `Engine ${n}`, engine: "llama.cpp", version: "b11552",
      model_id: "qwen3.8-27b", args: {}, gpus: [], cpu: activeCpus[0]?.i ?? 0, cpu_cores: 4, storage: ssds[0]?.id ?? null }]);
  };
  const remove = (id: string) => setEngines(engines.filter((e) => e.id !== id));

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1>Engines</h1>
        <button className="btn btn-primary" onClick={add}>Add engine</button>
      </div>
      <p className="mt-1 text-muted">One model per engine. The version decides which arguments exist; the hardware you assign decides the speeds shown on the right.</p>
      <div className="mt-6 flex flex-col gap-4">
        {engines.map((e) => {
          const def = catalog.engines[e.engine];
          const versions = Object.keys(def.versions);
          const args: ArgDef[] = def.versions[e.version]?.args ?? [];
          const models = catalog.models.filter((m) => def.models.includes(m.id));
          const d = derived[e.id];
          const hosted = def.kind === "hosted";
          const usedGpus = new Map<string, string>();
          engines.forEach((o) => { if (o.id !== e.id) o.gpus.forEach((g) => usedGpus.set(g, o.name || o.id)); });
          return (
            <section key={e.id} className="rounded-lg border border-line bg-panel">
              <div className="grid gap-5 p-5 lg:grid-cols-[3fr_2fr]">
                <div>
                  <div className="flex flex-wrap items-end gap-3">
                    <Field label="Name"><input type="text" value={e.name} onChange={(ev) => patch(e.id, (x) => { x.name = ev.target.value; })} /></Field>
                    <Field label="Engine">
                      <Select value={e.engine} options={Object.entries(catalog.engines).map(([k, v]) => ({ value: k, label: v.name }))}
                              onChange={(k) => patch(e.id, (x) => { const nd = catalog.engines[k]; x.engine = k; x.version = Object.keys(nd.versions)[0];
                                x.args = {}; if (!nd.models.includes(x.model_id)) x.model_id = nd.models[0]; if (nd.kind === "hosted") x.gpus = []; })} />
                    </Field>
                    <Field label="Version">
                      <Select value={e.version} options={versions.map((v) => ({ value: v, label: v }))}
                              onChange={(v) => patch(e.id, (x) => { x.version = v; x.args = {}; })} />
                    </Field>
                    <Field label="Model">
                      <Select value={e.model_id} options={models.map((m) => ({ value: m.id, label: m.name }))}
                              onChange={(m) => patch(e.id, (x) => { x.model_id = m; })} />
                    </Field>
                    <button className="btn btn-quiet ml-auto" onClick={() => remove(e.id)}>Remove</button>
                  </div>

                  {!hosted && (
                    <div className="mt-4 flex flex-wrap items-end gap-3">
                      <div>
                        <label>GPUs</label>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {gpus.length === 0 && <span className="text-xs text-muted">none on the machine</span>}
                          {gpus.map((g) => {
                            const on = e.gpus.includes(g.id); const other = usedGpus.get(g.id);
                            const spec = catalog.gpus.find((x) => x.id === g.model_id)!;
                            return (
                              <button key={g.id} className={`chip border ${on ? "bg-mask text-white border-mask" : "border-line bg-paper"}`}
                                      title={`${spec.name}${other ? ` · also used by ${other}` : ""}`}
                                      onClick={() => patch(e.id, (x) => { x.gpus = on ? x.gpus.filter((y) => y !== g.id) : [...x.gpus, g.id]; })}>
                                {g.id}{other ? "*" : ""}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                      {activeCpus.length > 1 && (
                        <Field label="CPU">
                          <Select value={e.cpu} options={activeCpus.map((c) => ({ value: c.i, label: `CPU ${c.i + 1}` }))}
                                  onChange={(c) => patch(e.id, (x) => { x.cpu = c; })} />
                        </Field>
                      )}
                      <Field label="CPU cores" hint={`${cpuSpec(e.cpu).cores} on CPU ${e.cpu + 1}`}>
                        <input type="number" min={0} max={cpuSpec(e.cpu).cores} value={e.cpu_cores} className="w-20"
                               onChange={(ev) => patch(e.id, (x) => { x.cpu_cores = Number(ev.target.value); })} />
                      </Field>
                      <Field label="Model storage">
                        <Select value={e.storage ?? ""} options={[{ value: "", label: "none" }, ...ssds.map((s) => ({ value: s.id, label: s.id }))]}
                                onChange={(s) => patch(e.id, (x) => { x.storage = s || null; })} />
                      </Field>
                    </div>
                  )}

                  <div className="mt-4 grid gap-x-4 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
                    {args.map((a) => <ArgField key={a.name} a={a} value={e.args[a.name]} onChange={(v) => patch(e.id, (x) => { x.args = { ...x.args, [a.name]: v }; })} />)}
                  </div>
                  {(community[e.id] ?? []).length > 0 && !hosted && (
                    <div className="mt-4 rounded border border-mask/40 bg-mask-soft/40 p-3 text-sm">
                      <div className="font-medium">{(community[e.id] ?? [])[0]?.near ? "Nearest community measurements (not this exact configuration)" : "Community measurements for this configuration"}</div>
                      <ul className="mt-1 flex flex-col gap-1">
                        {(community[e.id] ?? []).map((c, i) => (
                          <li key={i} className="flex flex-wrap items-center gap-x-3">
                            <span>{c.key.streams} stream{c.key.streams > 1 ? "s" : ""}: {Object.entries(c.results).filter(([k]) => k !== "hit")
                              .map(([k, v]) => `${k.replace("decode_", "writes with ").replace("prefill", "reads").replace("kv_room_tokens", "KV room")} ${v.median}${v.spread ? ` ±${(v.spread * 50).toFixed(0)} %` : ""}`).join(", ")}</span>
                            <span className="text-xs text-muted">{c.near ? `${c.near}; ` : ""}n = {c.n}, trust {c.trust}, CPU {c.cpus.join(" / ")}, {c.os.join(" / ")}</span>
                            {!c.near && <button className="btn" onClick={() => useCluster(e.id, c)}>Use</button>}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <details className="mt-4" open={Boolean(e.benchmarks && Object.values(e.benchmarks).some((v) => v))}>
                    <summary className="cursor-pointer text-sm text-muted">Measured on real hardware (overrides the derived speeds for this configuration)</summary>
                    <div className="mt-2 flex flex-wrap gap-3">
                      {[["decode_1", "writes alone, tok/s"], [`decode_${d?.slots && d.slots > 1 ? d.slots : 2}`, `writes with ${d?.slots && d.slots > 1 ? d.slots : 2} at once, tok/s total`],
                        ["prefill", "reads prompts, tok/s"], ["kv_room_tokens", "KV room, tokens"]].map(([k, label]) => (
                        <Field key={k} label={label}>
                          <input type="number" min={0} className="w-32" aria-label={`measured ${k}`} value={e.benchmarks?.[k] ?? ""}
                                 placeholder="derived" onChange={(ev) => patch(e.id, (x) => {
                                   const b = { ...(x.benchmarks ?? {}) }; const v = ev.target.value === "" ? null : Number(ev.target.value);
                                   if (v === null) delete b[k]; else b[k] = v; x.benchmarks = b; })} />
                        </Field>
                      ))}
                    </div>
                    {!hosted && (
                      <div className="mt-3 flex items-center gap-3">
                        <button className="btn" disabled={!e.benchmarks || !Object.values(e.benchmarks).some((v) => v)}
                                onClick={() => setExporting(e.id)}>Export measurement</button>
                        <span className="text-xs text-muted">bundles this machine, this engine and your values into a record for the repository</span>
                      </div>
                    )}
                    {exporting === e.id && <ExportMeasurement engine={e} machine={machine} onClose={() => setExporting(null)} />}
                  </details>
                </div>

                <aside className="rounded border border-line bg-paper p-4 text-sm">
                  {!d ? <span className="text-muted">Calculating…</span> : d.errors.length ? (
                    <ul className="text-red">{d.errors.map((x, i) => <li key={i}>{x}</li>)}</ul>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                        <span className="text-muted">reads prompts</span><span>{Math.round(d.prefill_rate).toLocaleString()} tok/s</span>
                        <span className="text-muted">writes, alone</span><span>{d.decode_total["1"]?.toFixed(0)} tok/s</span>
                        {Object.entries(d.decode_total).filter(([k]) => k !== "1").slice(0, 4).map(([k, v]) => (
                          <><span key={`l${k}`} className="text-muted">writes, {k} at once</span><span key={`v${k}`}>{v.toFixed(0)} tok/s total · {(v / Number(k)).toFixed(0)} each</span></>
                        ))}
                        {typeof d.memory.hit === "number" && <><span className="text-muted">expert cache</span><span>{(d.memory.held as number * 100).toFixed(0)} % held, {(d.memory.hit as number * 100).toFixed(0)} % hits</span></>}
                        {typeof d.memory.kv_room_tokens === "number" && <><span className="text-muted">KV room</span><span>{Math.round((d.memory.kv_room_tokens as number) / 1000)}K tokens</span></>}
                        {typeof d.memory.host_ram_gb === "number" && <><span className="text-muted">host RAM</span><span>{String(d.memory.host_ram_gb)} GB</span></>}
                      </div>
                      {Object.entries(d.memory).filter(([, v]) => typeof v === "object" && v && "vram_gb" in (v as object)).map(([k, v]) => {
                        const m = v as { vram_gb: number; used_gb: number };
                        return <div key={k} className="mt-2 text-xs text-muted">{k}: {m.used_gb} of {m.vram_gb} GB VRAM</div>;
                      })}
                      {d.memory.measured ? <div className="mt-2 text-xs text-mask-deep">uses your measured values</div> : null}
                      {d.notes.map((n, i) => <div key={i} className="mt-2 text-xs text-amber">{n}</div>)}
                    </>
                  )}
                </aside>
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}

function ArgField({ a, value, onChange }: { a: ArgDef; value: unknown; onChange: (v: unknown) => void }) {
  const v = value ?? a.default;
  if (a.type === "bool") return (
    <label className="flex items-center gap-2 text-sm text-ink" title={a.help}>
      <input type="checkbox" aria-label={a.name} checked={Boolean(v)} onChange={(e) => onChange(e.target.checked)} />{a.name}
    </label>
  );
  if (a.type === "choice") return (
    <Field label={a.name} hint={a.help}>
      <Select ariaLabel={a.name} value={String(v)} options={(a.options ?? []).map((o) => ({ value: o, label: o }))} onChange={onChange} />
    </Field>
  );
  return (
    <Field label={a.name} hint={a.help}>
      <input type="number" aria-label={a.name} value={Number(v)} min={a.min} max={a.max} step={a.type === "float" ? 0.05 : 1}
             onChange={(e) => onChange(a.type === "float" ? parseFloat(e.target.value) : parseInt(e.target.value, 10))} />
    </Field>
  );
}
