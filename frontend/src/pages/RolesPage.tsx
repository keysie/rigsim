import { useEffect, useState } from "react";
import { api, useStore, WORKLOAD_HELP, type Range, type RoleName, type Workload } from "../store";
import { Bar, Field, Notices, Select } from "../components/ui";

const ROLES: { id: RoleName; title: string; blurb: string }[] = [
  { id: "coordinator", title: "Coordinator", blurb: "plans each task and reviews the result; usually hosted" },
  { id: "maker", title: "Maker", blurb: "does the work: turns, tool calls, delegation; also takes vision" },
  { id: "sub", title: "Sub-makers", blurb: "task agents the maker delegates to (none = same engine as the maker)" },
  { id: "guard", title: "Output guard", blurb: "checks every tool result before the model sees it (none = skipped)" },
  { id: "judge", title: "Judge", blurb: "approves tool calls that need approval (none = skipped)" },
  { id: "reranker", title: "Reranker", blurb: "tool search (none = skipped)" },
];

type Result = {
  errors: string[]; warnings: string[];
  results: { concurrency: number; tasks_per_h: { p10: number; p50: number; p90: number }; task_p90_min: number; task_med_min: number;
    engines: Record<string, { util: number; wait_p90: number; calls: number; cost_usd?: number; skipped: number; reread_tokens: number; prefill_tokens: number }>;
    kinds: Record<string, { calls: number; per_min: number; prompt_p50: number; prompt_p90: number; wait_p90: number; dur_p90: number }>;
    gpus: Record<string, number>; bottlenecks: string[] }[];
};

export default function RolesPage() {
  const s = useStore();
  const engineIds = new Set(s.engines.map((e) => e.id));
  const stale = Object.entries(s.roles).filter(([, v]) => v && !engineIds.has(v));
  useEffect(() => {
    if (stale.length) s.setRoles(Object.fromEntries(Object.entries(s.roles).map(([k, v]) => [k, v && engineIds.has(v) ? v : null])) as typeof s.roles);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stale.length]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const res = s.results as Result | null;
  const names = Object.fromEntries(s.engines.map((e) => [e.id, e.name || e.id]));

  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api<Result>("/api/simulate", { machine: s.machine, engines: s.engines, roles: s.roles, workload: s.workload,
        concurrency: s.concurrency, scenarios: s.scenarios, seed: 1 });
      s.setResults(r);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  return (
    <div>
      <h1>Roles &amp; simulate</h1>
      <p className="mt-1 text-muted">Assign engines to the roles of the agent system, describe the load, and run. The run plays a Turnstone-shaped
        workload through the engines: fork-join delegation, approvals, guards and compaction.</p>

      <section className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {ROLES.map((r) => (
          <div key={r.id} className="rounded border border-line bg-panel p-3">
            <div className="font-medium">{r.title}</div>
            <div className="text-xs text-muted mb-2">{r.blurb}</div>
            <Select value={engineIds.has(s.roles[r.id] ?? "") ? s.roles[r.id] ?? "" : ""} options={[{ value: "", label: r.id === "maker" ? "choose an engine" : "none" },
              ...s.engines.map((e) => ({ value: e.id, label: e.name || e.id }))]}
                    onChange={(v) => s.setRoles({ ...s.roles, [r.id]: v || null })} />
          </div>
        ))}
      </section>

      <details className="mt-6 rounded border border-line bg-panel p-4">
        <summary className="cursor-pointer font-medium">Workload: what the agents do (ranges, sampled per scenario)</summary>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {(Object.keys(WORKLOAD_HELP) as (keyof Workload)[]).map((k) => {
            const [lo, hi] = s.workload[k] as Range;
            const set = (i: 0 | 1, v: number) => { const r: Range = [lo, hi]; r[i] = v; s.setWorkload({ ...s.workload, [k]: r }); };
            return (
              <div key={k} className="text-sm">
                <div>{WORKLOAD_HELP[k]}</div>
                <div className="mt-1 flex items-center gap-2">
                  <input type="number" step="any" value={lo} className="w-24" onChange={(e) => set(0, Number(e.target.value))} />
                  <span className="text-muted">to</span>
                  <input type="number" step="any" value={hi} className="w-24" onChange={(e) => set(1, Number(e.target.value))} />
                </div>
              </div>
            );
          })}
          <Field label="context window (maker, sub)"><input type="number" value={s.workload.window} onChange={(e) => s.setWorkload({ ...s.workload, window: Number(e.target.value) })} /></Field>
          <Field label="maker system prompt tokens"><input type="number" value={s.workload.maker_sys} onChange={(e) => s.setWorkload({ ...s.workload, maker_sys: Number(e.target.value) })} /></Field>
          <Field label="sub-agent system prompt tokens"><input type="number" value={s.workload.sub_sys} onChange={(e) => s.setWorkload({ ...s.workload, sub_sys: Number(e.target.value) })} /></Field>
          <Field label="tasks per workstream"><input type="number" value={s.workload.tasks_per_ws} min={1} max={10} onChange={(e) => s.setWorkload({ ...s.workload, tasks_per_ws: Number(e.target.value) })} /></Field>
          <label className="flex items-center gap-2 text-sm text-ink self-end">
            <input type="checkbox" checked={s.workload.replay_thinking} onChange={(e) => s.setWorkload({ ...s.workload, replay_thinking: e.target.checked })} />
            the orchestrator sends the model's thinking back (history reuse works)
          </label>
        </div>
      </details>

      <div className="mt-6 flex flex-wrap items-end gap-4">
        <Field label="Concurrent workstreams" hint="comma-separated; each level is simulated">
          <input type="text" value={s.concurrency.join(", ")} className="w-32"
                 onChange={(e) => s.setConcurrency(e.target.value.split(",").map((x) => parseInt(x, 10)).filter((x) => x > 0 && x <= 32))} />
        </Field>
        <Field label="Scenarios" hint="random draws from the ranges">
          <input type="number" value={s.scenarios} min={1} max={60} className="w-24" onChange={(e) => s.setScenarios(Number(e.target.value))} />
        </Field>
        <button className="btn btn-primary" onClick={run} disabled={busy || !s.roles.maker}>{busy ? "Simulating…" : "Simulate"}</button>
        {!s.roles.maker && <span className="text-sm text-amber">No engine is assigned to the maker role — pick one above (add engines on the Engines page).</span>}
        {!s.roles.maker && <span className="text-sm text-amber">Choose a maker engine first.</span>}
      </div>
      {err && <Notices errors={[err]} warnings={[]} />}

      {res && (
        <section className="mt-8">
          <Notices errors={res.errors} warnings={res.warnings} />
          {res.results.length > 0 && (
            <>
              <h2 className="mt-4">Tasks per hour</h2>
              <div className="mt-2 grid gap-2">
                {res.results.map((r) => {
                  const max = Math.max(...res.results.map((x) => x.tasks_per_h.p90));
                  return (
                    <div key={r.concurrency} className="grid grid-cols-[120px_1fr_200px] items-center gap-3 text-sm">
                      <span>{r.concurrency} workstream{r.concurrency > 1 ? "s" : ""}</span>
                      <div className="relative h-5 rounded bg-line/50">
                        <div className="absolute inset-y-0 bg-mask-soft" style={{ left: `${(r.tasks_per_h.p10 / max) * 100}%`, width: `${((r.tasks_per_h.p90 - r.tasks_per_h.p10) / max) * 100}%` }} />
                        <div className="absolute inset-y-0 w-0.5 bg-mask" style={{ left: `${(r.tasks_per_h.p50 / max) * 100}%` }} />
                      </div>
                      <span className="text-muted">{r.tasks_per_h.p50.toFixed(1)} typical · {r.tasks_per_h.p10.toFixed(1)}–{r.tasks_per_h.p90.toFixed(1)} · task p90 {r.task_p90_min.toFixed(0)} min</span>
                    </div>
                  );
                })}
              </div>
              <p className="mt-1 text-xs text-muted">Band = 10th to 90th percentile across scenarios; line = median.</p>

              {res.results.map((r) => (
                <div key={r.concurrency} className="mt-8 rounded-lg border border-line bg-panel p-5">
                  <h2>{r.concurrency} concurrent workstream{r.concurrency > 1 ? "s" : ""}</h2>
                  <div className="mt-3 grid gap-6 lg:grid-cols-2">
                    <div>
                      <div className="text-sm font-medium">Where the time goes</div>
                      <ol className="mt-2 flex flex-col gap-2 text-sm list-decimal pl-5">
                        {r.bottlenecks.length === 0 && <li className="text-muted">Nothing saturates at this load.</li>}
                        {r.bottlenecks.map((b, i) => <li key={i}>{b}</li>)}
                      </ol>
                    </div>
                    <div>
                      <div className="text-sm font-medium">Engines</div>
                      <div className="mt-2 grid gap-2">
                        {Object.entries(r.engines).map(([id, e]) => (
                          <Bar key={id} value={e.util} max={1} danger={e.util > 0.85}
                               label={`${names[id] ?? id}: busy ${(e.util * 100).toFixed(0)} %, waits ${e.wait_p90.toFixed(0)} s p90`
                                 + (e.prefill_tokens > 0 ? `, re-reads ${(100 * e.reread_tokens / e.prefill_tokens).toFixed(0)} %` : "")
                                 + (e.cost_usd ? `, $${e.cost_usd.toFixed(2)}` : "")} />
                        ))}
                      </div>
                      <div className="mt-4 text-sm font-medium">GPUs</div>
                      <div className="mt-2 grid gap-2">
                        {Object.entries(r.gpus).map(([id, u]) => <Bar key={id} value={u} max={1} danger={u > 0.9} label={`${id}: ${(u * 100).toFixed(0)} %`} />)}
                      </div>
                    </div>
                  </div>
                  <div className="mt-5 overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="text-muted text-left"><tr><th className="py-1 pr-4 font-normal">calls</th><th className="pr-4 font-normal">per minute</th><th className="pr-4 font-normal">prompt p50 / p90</th><th className="pr-4 font-normal">queue p90</th><th className="font-normal">done p90</th></tr></thead>
                      <tbody>
                        {Object.entries(r.kinds).map(([k, v]) => (
                          <tr key={k} className="border-t border-line"><td className="py-1 pr-4">{k}</td><td className="pr-4">{v.per_min.toFixed(1)}</td>
                            <td className="pr-4">{(v.prompt_p50 / 1000).toFixed(1)}K / {(v.prompt_p90 / 1000).toFixed(1)}K</td><td className="pr-4">{v.wait_p90.toFixed(1)} s</td><td>{v.dur_p90.toFixed(1)} s</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </>
          )}
        </section>
      )}
    </div>
  );
}
