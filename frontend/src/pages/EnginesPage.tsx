import { useEffect, useState, type ReactNode } from "react";
import { api, useStore, type ArgDef, type EngineSpec } from "../store";
import { Field, NumInput, Select } from "../components/ui";
import ExportMeasurement from "../components/ExportMeasurement";
import StrataLayout, { type Layout } from "../components/StrataLayout";

type Cluster = { key: { gpus: string[]; engine: string; model: string; quant: string; streams: number }; n: number; trust: number;
  results: Record<string, { median: number; n: number; spread: number }>; cpus: string[]; os: string[]; near: string | null };

type Derived = Record<string, {
  kind: string; slots: number; prefill_rate: number; decode_total: Record<string, number>;
  speed_per_request: Record<string, number>; memory: Record<string, unknown>; notes: string[]; errors: string[];
  limits: Record<string, number>;
  arg_states?: Record<string, { disabled: boolean; force: unknown; reason: string }>;
  layout?: Layout | null;
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

  const wide = useWide();
  const side = (e: EngineSpec) => wide && e.engine === "strata";

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
    const next = structuredClone(engines); const x = next.find((e) => e.id === id)!; fn(x);
    // tensor parallel follows the number of assigned GPUs whenever that is a valid choice: a mismatch is always an error
    const tp = catalog.engines[x.engine]?.versions[x.version]?.args.find((a) => a.name === "tensor_parallel");
    if (tp?.options && x.gpus.length >= 1 && tp.options.includes(String(x.gpus.length)) && String(x.args.tensor_parallel ?? tp.default) !== String(x.gpus.length)) {
      x.args = { ...x.args, tensor_parallel: String(x.gpus.length) };
    }
    // a model that comes in specific packs: the pack field is limited to them and lands on the first one
    const packs = catalog.models.find((m) => m.id === x.model_id)?.quants;
    if (packs?.length && !packs.includes(String(x.args.quantization ?? ""))) x.args = { ...x.args, quantization: packs[0] };
    setEngines(next);
  };
  const add = () => {
    const used = new Set(engines.map((e) => e.id));
    let n = engines.length + 1; while (used.has(`engine-${n}`)) n++;
    const gpuList = machine.devices.filter((d) => d.kind === "gpu");
    const free = gpuList.find((d) => !engines.some((e) => e.gpus.includes(d.id))) ?? gpuList[0];
    setEngines([...engines, { id: `engine-${n}`, name: `Engine ${n}`, engine: "llama.cpp", version: "b11552",
      model_id: "qwen3.8-27b", args: {}, gpus: free ? [free.id] : [], cpu: activeCpus[0]?.i ?? 0, cpu_cores: 4, storage: ssds[0]?.id ?? null }]);
  };
  const roles = useStore((s) => s.roles);
  const ROLE_NAMES: Record<string, string> = { coordinator: "coordinator", maker: "maker", sub: "sub-makers", guard: "guard", judge: "judge", reranker: "reranker" };
  const rolesUsing = (id: string) => Object.entries(roles).filter(([, v]) => v === id).map(([k]) => ROLE_NAMES[k] ?? k);
  const [removed, setRemoved] = useState<{ engine: EngineSpec; index: number } | null>(null);
  const remove = (id: string) => {
    const index = engines.findIndex((e) => e.id === id);
    if (index < 0) return;
    setRemoved({ engine: engines[index], index });
    setEngines(engines.filter((e) => e.id !== id));
  };
  const undo = () => {
    if (!removed) return;
    const list = engines.slice();
    list.splice(Math.min(removed.index, list.length), 0, removed.engine);
    setEngines(list);
    setRemoved(null);
  };

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1>Engines</h1>
        <button className="btn btn-primary" onClick={add}>Add engine</button>
      </div>
      {removed && (
        <div className="mt-4 flex items-center justify-between rounded border border-line bg-panel px-3 py-2 text-sm" role="status">
          <span>Removed “{removed.engine.name || removed.engine.id}”.{rolesUsing(removed.engine.id).length ? ` It was assigned to: ${rolesUsing(removed.engine.id).join(", ")} — those roles are now empty.` : ""}</span>
          <button className="btn" onClick={undo}>Undo</button>
        </div>
      )}
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
          // cores on this engine's CPU not taken by other local engines: no oversubscription, no sharing
          const coresLeft = (me: EngineSpec) => cpuSpec(me.cpu).cores - engines.filter((o) => o.id !== me.id && o.cpu === me.cpu && o.engine !== "hosted-api").reduce((a, o) => a + (o.cpu_cores || 0), 0);
          return (
            <section key={e.id} className="rounded-lg border border-line bg-panel">
              <div className={`grid gap-5 p-5 ${side(e) ? "grid-cols-[minmax(0,1fr)_minmax(0,1fr)]" : "lg:grid-cols-[3fr_2fr]"}`}>
                <div>
                  <div className="flex flex-wrap items-start gap-x-3 gap-y-2">
                    <Field label="Name"><input type="text" className="w-36" value={e.name} onChange={(ev) => patch(e.id, (x) => { x.name = ev.target.value; })} /></Field>
                    <Field label="Engine">
                      <Select value={e.engine} options={Object.entries(catalog.engines).map(([k, v]) => ({ value: k, label: v.name }))}
                              onChange={(k) => patch(e.id, (x) => { const nd = catalog.engines[k]; x.engine = k; x.version = Object.keys(nd.versions)[0];
                                x.args = {}; if (!nd.models.includes(x.model_id)) x.model_id = nd.models[0]; if (nd.kind === "hosted") x.gpus = []; })} />
                    </Field>
                    <Field label="Version">
                      <Select value={e.version} options={versions.map((v) => ({ value: v, label: v }))}
                              onChange={(v) => patch(e.id, (x) => { x.version = v; x.args = {}; })} />
                    </Field>
                    <div className="min-w-[14rem] flex-1">
                      <Field label="Model">
                        <Select className="w-full min-w-0 max-w-full truncate" value={e.model_id} options={models.map((m) => ({ value: m.id, label: m.name }))}
                                onChange={(m) => patch(e.id, (x) => { x.model_id = m; })} />
                      </Field>
                    </div>
                  </div>
                  <div className="mt-1 flex items-center gap-3">
                    <div className="min-w-0 flex-1 truncate text-xs text-muted" title={catalog.models.find((m) => m.id === e.model_id)?.note}>{catalog.models.find((m) => m.id === e.model_id)?.note}</div>
                    <button className="shrink-0 rounded border border-red/40 px-2 py-0.5 text-xs text-red hover:bg-red-soft" onClick={() => remove(e.id)}>Remove engine</button>
                  </div>

                  {!hosted && (
                    <div className="mt-3 flex flex-wrap items-start gap-x-4 gap-y-2">
                      <div>
                        <label>GPUs</label>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {gpus.length === 0 && <span className="text-xs text-muted">none on the machine</span>}
                          {gpus.map((g) => {
                            const on = e.gpus.includes(g.id); const other = usedGpus.get(g.id);
                            const spec = catalog.gpus.find((x) => x.id === g.model_id)!;
                            return (
                              <button key={g.id} className={`chip border ${on ? "bg-mask text-white border-mask" : other ? "border-line bg-paper opacity-40 cursor-not-allowed line-through" : "border-line bg-paper"}`}
                                      disabled={Boolean(other) && !on} aria-disabled={Boolean(other) && !on}
                                      title={`${spec.name}${other ? ` · used by ${other}: one engine per card (sharing is out of scope)` : ""}`}
                                      onClick={() => patch(e.id, (x) => { x.gpus = on ? x.gpus.filter((y) => y !== g.id) : [...x.gpus, g.id]; })}>
                                {g.id}
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
                      <Field label={`CPU cores · ${coresLeft(e)} of ${cpuSpec(e.cpu).cores} free`}>
                        <NumInput min={1} max={Math.max(1, coresLeft(e))} value={e.cpu_cores} className="w-20"
                                  onChange={(n) => patch(e.id, (x) => { x.cpu_cores = n ?? 1; })} />
                      </Field>
                      <Field label="Model storage">
                        <Select value={e.storage ?? ""} options={[{ value: "", label: "none" }, ...ssds.map((s) => ({ value: s.id, label: s.id }))]}
                                onChange={(s) => patch(e.id, (x) => { x.storage = s || null; })} />
                      </Field>
                    </div>
                  )}

                  <div className="mt-3 grid gap-x-4 gap-y-2.5 sm:grid-cols-2 lg:grid-cols-3">
                    {args.map((a) => <ArgField key={a.name} dense={side(e)} a={packArg(a, catalog.models.find((m) => m.id === e.model_id)?.quants)} value={e.args[a.name]} state={d?.arg_states?.[a.name]}
                                               onChange={(v) => patch(e.id, (x) => { x.args = { ...x.args, [a.name]: v }; })} />)}
                  </div>
                  {e.engine === "strata" && d?.layout && !side(e) && (
                    <StrataLayout layout={d.layout} hitGuessed={d.layout.hit_guessed} fixed={e.fixed ?? {}}
                                  onFixed={(f) => patch(e.id, (x) => { x.fixed = f; })} />
                  )}
                  {(community[e.id] ?? []).length > 0 && !hosted && (
                    <details open={!side(e)} className="mt-3 rounded border border-mask/40 bg-mask-soft/40 p-3 text-sm">
                      <summary className="cursor-pointer font-medium">{(community[e.id] ?? [])[0]?.near ? "Nearest community measurements (not this exact configuration)" : "Community measurements for this configuration"}</summary>
                      <ul className="mt-1 flex flex-col gap-1">
                        {(community[e.id] ?? []).map((c, i) => (
                          <li key={i} className="flex flex-wrap items-center gap-x-3">
                            <span>{c.key.streams} stream{c.key.streams > 1 ? "s" : ""}: {Object.entries(c.results).filter(([k]) => k !== "hit")
                              .map(([k, v]) => `${measuredLabel(k)} ${v.median}${v.spread ? ` ±${(v.spread * 50).toFixed(0)} %` : ""}`).join(", ")}</span>
                            <span className="text-xs text-muted">{c.near ? `${c.near}; ` : ""}n = {c.n}, trust {c.trust}, CPU {c.cpus.join(" / ")}, {c.os.join(" / ")}</span>
                            {!c.near && <button className="btn" onClick={() => useCluster(e.id, c)}>Use</button>}
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                  <details className="mt-4" open={Boolean(e.benchmarks && Object.values(e.benchmarks).some((v) => v))}>
                    <summary className="cursor-pointer text-sm text-muted">Measured on real hardware (overrides the derived speeds for this configuration)</summary>
                    <div className="mt-2 flex flex-wrap gap-3">
                      {[["decode_1", "writes alone, tok/s"], [`decode_${d?.slots && d.slots > 1 ? d.slots : 2}`, `writes with ${d?.slots && d.slots > 1 ? d.slots : 2} at once, tok/s total`],
                        ["prefill", "reads prompts, tok/s"], ["kv_room_tokens", "KV room, tokens"]].map(([k, label]) => (
                        <Field key={k} label={label}>
                          <NumInput min={0} int={false} step="any" allowEmpty className="w-32" aria-label={`measured ${k}`} value={e.benchmarks?.[k] ?? null}
                                    placeholder="derived" onChange={(v) => patch(e.id, (x) => {
                                      const b = { ...(x.benchmarks ?? {}) };
                                      if (v === null || v <= 0) delete b[k]; else b[k] = v; x.benchmarks = b; })} />
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

                <div className={side(e) ? "sticky top-4 flex h-[calc(100vh-2rem)] flex-col gap-3 self-start" : undefined}>
                <aside className={`rounded border border-line bg-paper text-sm ${side(e) ? "shrink-0 p-3" : "p-4"}`}>
                  {!d ? <span className="text-muted">Calculating…</span> : d.errors.length ? (
                    <ul className="text-red">{d.errors.map((x, i) => <li key={i}>{x}</li>)}</ul>
                  ) : (
                    <>
                      <div className={side(e) ? "grid grid-cols-[auto_1fr_auto_1fr] gap-x-3 gap-y-0.5" : "grid grid-cols-2 gap-x-3 gap-y-1"}>
                        <span className="text-muted">reads prompts</span><span>{Math.round(d.prefill_rate).toLocaleString()} tok/s</span>
                        <span className="text-muted">writes, alone</span><span>{d.decode_total["1"]?.toFixed(0)} tok/s</span>
                        {typeof d.memory.decode_1_at_32k === "number" && <><span className="text-muted">writes, 32K deep</span><span>{(d.memory.decode_1_at_32k as number).toFixed(0)} tok/s</span></>}
                        {typeof d.memory.followup_2k_s === "number" && <><span className="text-muted">2K follow-up</span><span>~{(d.memory.followup_2k_s as number).toFixed(1)} s to read</span></>}
                        {typeof d.memory.full_prompt_s === "number" && <><span className="text-muted">full {Math.round((d.memory.full_prompt_tokens as number) / 1024)}K prompt</span><span>~{(d.memory.full_prompt_s as number) < 120 ? `${(d.memory.full_prompt_s as number).toFixed(0)} s` : `${((d.memory.full_prompt_s as number) / 60).toFixed(1)} min`} to read</span></>}
                        {speedRows(d.decode_total, Number(e.args.batch ?? e.args.parallel ?? e.args.max_num_seqs ?? 1)).map(([k, v, configured]) => (
                          <><span key={`l${k}`} className={configured ? "font-medium" : "text-muted"} title="while all of them write; prompt reads pause them (see the note below)">writes, {k} at once{configured ? " (set)" : ""}</span><span key={`v${k}`} className={configured ? "font-medium" : undefined}>{v.toFixed(0)} tok/s total · {(v / Number(k)).toFixed(0)} each</span></>
                        ))}
                        {typeof d.memory.hit === "number" && <><span className="text-muted">expert cache</span><span>{(d.memory.held as number * 100).toFixed(0)} % held, {(d.memory.hit as number * 100).toFixed(0)} % hits</span></>}
                        {typeof d.memory.kv_room_tokens === "number" && <><span className="text-muted">KV room</span><span>{Math.round((d.memory.kv_room_tokens as number) / 1000)}K tokens</span></>}
                        {d.layout?.pools?.[0] ? (
                          <><span className="text-muted">{d.layout.cards.length ? "host RAM" : "unified memory"}</span><span>{d.layout.pools[0].total_gb.toFixed(1)} of {d.layout.pools[0].capacity_gb.toFixed(0)} GB</span></>
                        ) : typeof d.memory.host_ram_gb === "number" && <><span className="text-muted">host RAM</span><span>{String(d.memory.host_ram_gb)} GB</span></>}
                      </div>
                      {d.layout ? (
                        <div className="mt-2 text-xs text-muted">{d.layout.cards.map((c) => `${c.id}: expert cache ${c.cache_gb.toFixed(1)} of ${c.capacity_gb.toFixed(0)} GB`).join(" · ")}</div>
                      ) : Object.entries(d.memory).filter(([, v]) => typeof v === "object" && v && "vram_gb" in (v as object)).map(([k, v]) => {
                        const m = v as { vram_gb: number; used_gb: number };
                        return <div key={k} className="mt-2 text-xs text-muted">{k}: {m.used_gb} of {m.vram_gb} GB VRAM</div>;
                      })}
                      {d.memory.measured ? <div className="mt-2 text-xs text-mask-deep">uses your measured values</div> : null}
                      {d.notes.map((n, i) => <div key={i} className="mt-2 text-xs text-amber">{n}</div>)}
                    </>
                  )}
                </aside>
                {side(e) && d?.layout && (
                  <StrataLayout compact layout={d.layout} hitGuessed={d.layout.hit_guessed} fixed={e.fixed ?? {}}
                                onFixed={(f) => patch(e.id, (x) => { x.fixed = f; })} />
                )}
                </div>
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}

/** decode_1 → "writes alone", decode_4 → "writes with 4 at once (total)", prefill → "reads", kv_room_tokens → "KV room". */
export function measuredLabel(k: string): string {
  const m = k.match(/^decode_(\d+)$/);
  if (m) return m[1] === "1" ? "writes alone" : `writes with ${m[1]} at once (total)`;
  return { prefill: "reads", kv_room_tokens: "KV room", hit: "cache hit" }[k] ?? k;
}

/** Wide enough to show a Strata card's settings and its memory graphic side by side. */
function useWide(query = "(min-width: 1500px)") {
  const [wide, setWide] = useState(() => typeof window !== "undefined" && !!window.matchMedia?.(query).matches);
  useEffect(() => {
    const mq = window.matchMedia?.(query);
    if (!mq) return;
    const on = () => setWide(mq.matches);
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, [query]);
  return wide;
}

/** The pack (quantization) field offers only the packs the chosen model comes in. */
function packArg(a: ArgDef, packs?: string[]): ArgDef {
  if (a.name !== "quantization" || !packs?.length || !a.options) return a;
  const options = a.options.filter((o) => packs.includes(o));
  return { ...a, options: options.length ? options : a.options, default: packs[0], help: packs.length === 1 ? "the only pack this model comes in" : a.help };
}

/** "writes, N at once" rows: 2, the middle and the configured count — never hide the one that is set. */
function speedRows(table: Record<string, number>, set: number): [string, number, boolean][] {
  const keys = Object.keys(table).map(Number).filter((n) => n > 1).sort((a, b) => a - b);
  if (!keys.length) return [];
  const want = new Set([keys[0], keys[Math.floor((keys.length - 1) / 2)], Math.min(set, keys[keys.length - 1])].filter((n) => n > 1));
  return keys.filter((n) => want.has(n)).map((n) => [String(n), table[String(n)], n === set]);
}

type ArgState = { disabled: boolean; force: unknown; reason: string };

/** One argument. `state` comes from the rules the engine itself applies: a greyed-out field has no effect in this
 *  setup; a forced value is what the engine would use regardless of what is set. The reason is shown under it. */
function ArgField({ a, value, state, onChange, dense = false }: { a: ArgDef; value: unknown; state?: ArgState; onChange: (v: unknown) => void; dense?: boolean }) {
  const forced = state?.force !== null && state?.force !== undefined;
  const v = value ?? a.default;          // what is set; a forced value is shown in the hint ("→ x"), the field stays editable
  const off = Boolean(state?.disabled);
  const hint = state?.reason ? <span className={off || forced ? "text-amber" : undefined}>{state.reason}{forced ? ` → ${String(state!.force)}` : ""}</span>
    : <span className={dense ? "line-clamp-2" : undefined} title={a.help}>{a.help}</span>;
  const wrap = (child: ReactNode) => (
    <Field label={a.name} hint={hint}>
      <div className={off ? "opacity-50" : undefined}>{child}</div>
    </Field>
  );
  if (a.type === "bool") return wrap(
    <input type="checkbox" className="h-4 w-4" aria-label={a.name} checked={Boolean(v)} disabled={off} onChange={(e) => onChange(e.target.checked)} />
  );
  if (a.type === "choice") return wrap(
    <Select ariaLabel={a.name} disabled={off} value={String(v)} options={(a.options ?? []).map((o) => ({ value: o, label: o }))} onChange={onChange} />
  );
  return wrap(
    <NumInput aria-label={a.name} value={Number(v)} min={a.min} max={a.max} step={a.type === "float" ? 0.05 : 1} int={a.type !== "float"}
              disabled={off} onChange={(n) => onChange(n)} />
  );
}
