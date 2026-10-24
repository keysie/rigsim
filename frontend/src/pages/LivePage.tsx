import { useEffect, useMemo, useRef, useState } from "react";
import { api, useStore } from "../store";
import { Bar, Field, Notices, Select } from "../components/ui";

type Ev = { t: number; ev: "submit" | "admit" | "done"; engine: string; kind: string; conv: string; context?: number;
  out?: number; prefill?: number; wait?: number; reread?: boolean; tag?: string };
type Trace = { errors: string[]; concurrency: number; makespan_s: number; events: Ev[];
  tasks: { ws: number; task: number; start: number; end: number }[];
  resources: Record<string, { gpus: string[]; cpu: string; cores: number; limits: Record<string, number>; slots: number; kind: string; hosted: boolean }>;
  gpus: string[]; cpus: Record<string, { name: string; cores: number; ram_bw_gbs: number }> };

const SPEEDS = [10, 30, 60, 120, 300, 1000];

export default function LivePage() {
  const s = useStore();
  const names = Object.fromEntries(s.engines.map((e) => [e.id, e.name || e.id]));
  const [conc, setConc] = useState(2);
  const [scenario, setScenario] = useState(0);
  const [speed, setSpeed] = useState(60);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [clock, setClock] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [filter, setFilter] = useState("");
  const raf = useRef<number | null>(null);
  const last = useRef<number>(0);

  const run = async () => {
    setBusy(true); setErr(null); setTrace(null); setClock(0);
    try {
      const t = await api<Trace>("/api/trace", { request: { machine: s.machine, engines: s.engines, roles: s.roles, workload: s.workload,
        concurrency: s.concurrency, scenarios: s.scenarios, seed: 1 }, concurrency: conc, scenario });
      setTrace(t);
      if (!t.errors.length) setPlaying(true);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  useEffect(() => {
    if (!playing || !trace) return;
    last.current = performance.now();
    const step = (now: number) => {
      const dt = (now - last.current) / 1000 * speed;
      last.current = now;
      setClock((c) => {
        const n = c + dt;
        if (n >= trace.makespan_s) { setPlaying(false); return trace.makespan_s; }
        return n;
      });
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [playing, speed, trace]);

  // state at the clock: active jobs per engine, log lines so far
  const view = useMemo(() => {
    if (!trace) return null;
    const active: Record<string, { n: number; kinds: Record<string, number> }> = {};
    const queued: Record<string, number> = {};
    Object.keys(trace.resources).forEach((e) => { active[e] = { n: 0, kinds: {} }; queued[e] = 0; });
    let i = 0;
    const lines: Ev[] = [];
    for (; i < trace.events.length && trace.events[i].t <= clock; i++) {
      const ev = trace.events[i];
      if (ev.ev === "submit") queued[ev.engine] = (queued[ev.engine] ?? 0) + 1;
      if (ev.ev === "admit") { queued[ev.engine] = Math.max(0, (queued[ev.engine] ?? 0) - 1); active[ev.engine].n++; active[ev.engine].kinds[ev.kind] = (active[ev.engine].kinds[ev.kind] ?? 0) + 1; }
      if (ev.ev === "done") { active[ev.engine].n = Math.max(0, active[ev.engine].n - 1); active[ev.engine].kinds[ev.kind] = Math.max(0, (active[ev.engine].kinds[ev.kind] ?? 1) - 1); }
      lines.push(ev);
    }
    const gpu: Record<string, number> = {}; trace.gpus.forEach((g) => { gpu[g] = 0; });
    const cpuCores: Record<string, number> = {}; const ram: Record<string, number> = {}; const pcie: Record<string, number> = {};
    Object.entries(trace.resources).forEach(([e, r]) => {
      const on = active[e].n > 0 ? 1 : 0;
      r.gpus.forEach((g) => { gpu[g] = Math.min(1, (gpu[g] ?? 0) + on * (r.limits.gpu_memory_bandwidth ?? 0.8)); });
      if (r.cpu) {
        const share = r.limits.cpu_ram ?? (r.gpus.length ? 0 : 1);
        cpuCores[r.cpu] = (cpuCores[r.cpu] ?? 0) + on * r.cores * share;
        ram[r.cpu] = Math.min(1, (ram[r.cpu] ?? 0) + on * share);
      }
      r.gpus.forEach((g) => { pcie[g] = Math.min(1, (pcie[g] ?? 0) + on * (r.limits.pcie ?? 0)); });
    });
    const tasksDone = trace.tasks.filter((t) => t.end <= clock).length;
    const tasksRunning = trace.tasks.filter((t) => t.start <= clock && t.end > clock).length;
    return { active, queued, lines, gpu, cpuCores, ram, pcie, tasksDone, tasksRunning };
  }, [trace, clock]);

  const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  const describe = (ev: Ev) => {
    const who = `${names[ev.engine] ?? ev.engine} · ${ev.kind}`;
    if (ev.ev === "submit") return `${who}: request ${ev.conv} (${(ev.context! / 1000).toFixed(1)}K prompt, ${ev.out} out) waits for a slot`;
    if (ev.ev === "admit") return `${who}: ${ev.conv} starts, reads ${(ev.prefill! / 1000).toFixed(1)}K tokens${ev.reread ? " (history re-read)" : ""}`;
    return `${who}: ${ev.conv} done after waiting ${ev.wait} s`;
  };

  return (
    <div>
      <h1>Live run</h1>
      <p className="mt-1 text-muted">One scenario, replayed at a chosen speed: every request as it is queued, admitted and finished, and the load it puts on each part.</p>
      <div className="mt-5 flex flex-wrap items-end gap-4">
        <Field label="Concurrent workstreams">
          <input type="number" min={1} max={16} value={conc} className="w-24" onChange={(e) => setConc(Number(e.target.value))} />
        </Field>
        <Field label="Scenario" hint="index into the sampled scenarios">
          <input type="number" min={0} max={s.scenarios - 1} value={scenario} className="w-24" onChange={(e) => setScenario(Number(e.target.value))} />
        </Field>
        <Field label="Replay speed">
          <Select value={speed} options={SPEEDS.map((x) => ({ value: x, label: `${x}×` }))} onChange={setSpeed} />
        </Field>
        <button className="btn btn-primary" onClick={run} disabled={busy || !s.roles.maker}>{busy ? "Simulating…" : "Run and watch"}</button>
        {trace && !trace.errors.length && (
          <>
            <button className="btn" onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : clock >= trace.makespan_s ? "Replay" : "Resume"}</button>
            <input type="range" min={0} max={trace.makespan_s} value={clock} step={1} className="w-48" aria-label="position"
                   onChange={(e) => { setClock(Number(e.target.value)); if (clock >= trace.makespan_s) setPlaying(false); }}
                   onMouseDown={() => setPlaying(false)} />
          </>
        )}
      </div>
      {err && <Notices errors={[err]} warnings={[]} />}
      {trace?.errors.length ? <Notices errors={trace.errors} warnings={[]} /> : null}

      {trace && view && (
        <div className="mt-6 grid gap-6 lg:grid-cols-[2fr_3fr]">
          <section className="rounded-lg border border-line bg-panel p-4">
            <div className="flex items-baseline justify-between">
              <h2>{fmt(clock)} <span className="text-sm font-normal text-muted">of {fmt(trace.makespan_s)}</span></h2>
              <span className="text-sm text-muted">{view.tasksDone} of {trace.tasks.length} tasks done, {view.tasksRunning} running</span>
            </div>
            <div className="mt-3 text-sm font-medium">Engines</div>
            <div className="mt-2 grid gap-2">
              {Object.entries(trace.resources).map(([e, r]) => {
                const a = view.active[e];
                const kinds = Object.entries(a.kinds).filter(([, n]) => n > 0).map(([k, n]) => `${n} ${k}`).join(", ");
                return <Bar key={e} value={a.n} max={r.slots} danger={view.queued[e] > 0}
                            label={`${names[e] ?? e}: ${a.n}/${r.slots}${kinds ? ` (${kinds})` : ""}${view.queued[e] ? `, ${view.queued[e]} waiting` : ""}`} />;
              })}
            </div>
            <div className="mt-4 text-sm font-medium">GPUs: memory bandwidth</div>
            <div className="mt-2 grid gap-2">
              {trace.gpus.map((g) => <Bar key={g} value={view.gpu[g]} max={1} danger={view.gpu[g] > 0.9} label={`${g}: ${(view.gpu[g] * 100).toFixed(0)} %`} />)}
            </div>
            <div className="mt-4 text-sm font-medium">PCIe links (expert / weight traffic)</div>
            <div className="mt-2 grid gap-2">
              {trace.gpus.map((g) => <Bar key={g} value={view.pcie[g] ?? 0} max={1} label={`${g}: ${((view.pcie[g] ?? 0) * 100).toFixed(0)} %`} />)}
            </div>
            <div className="mt-4 text-sm font-medium">CPUs: cores and memory bandwidth</div>
            <div className="mt-2 grid gap-2">
              {Object.values(trace.cpus).map((c) => (
                <div key={c.name} className="grid gap-1">
                  <Bar value={view.cpuCores[c.name] ?? 0} max={c.cores} label={`${c.name}: ${(view.cpuCores[c.name] ?? 0).toFixed(0)} of ${c.cores} cores`} />
                  <Bar value={view.ram[c.name] ?? 0} max={1} label={`memory ${((view.ram[c.name] ?? 0) * 100).toFixed(0)} % of ${c.ram_bw_gbs} GB/s`} />
                </div>
              ))}
            </div>
          </section>
          <section className="rounded-lg border border-line bg-panel p-4">
            <div className="flex items-center justify-between">
              <h2>Log</h2>
              <Select value={filter} options={[{ value: "", label: "all engines" }, ...Object.keys(trace.resources).map((e) => ({ value: e, label: names[e] ?? e }))]} onChange={setFilter} />
            </div>
            <ol className="mt-2 h-[520px] overflow-y-auto text-sm" aria-live="polite">
              {view.lines.filter((l) => !filter || l.engine === filter).slice(-200).map((ev, i) => (
                <li key={i} className={`border-t border-line/60 py-1 ${ev.ev === "submit" ? "text-muted" : ev.ev === "done" ? "" : "text-mask-deep"}`}>
                  <span className="inline-block w-12 text-muted">{fmt(ev.t)}</span>{describe(ev)}
                </li>
              ))}
            </ol>
            <p className="mt-1 text-xs text-muted">Last 200 lines up to the current time.</p>
          </section>
        </div>
      )}
    </div>
  );
}
