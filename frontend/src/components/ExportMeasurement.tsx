import { useState } from "react";
import { api, type EngineSpec, type Machine } from "../store";
import { Field, Notices } from "./ui";

type Check = { ok: boolean; errors: string[]; warnings: string[]; id?: string };

const today = () => new Date().toISOString().slice(0, 10);

export default function ExportMeasurement({ engine, machine, onClose }: { engine: EngineSpec; machine: Machine; onClose: () => void }) {
  const streamsGuess = Object.keys(engine.benchmarks ?? {}).map((k) => k.match(/^decode_(\d+)$/)?.[1]).filter(Boolean).map(Number).sort((a, b) => b - a)[0] ?? 1;
  const [m, setM] = useState({ tool: "bench/benchw.py tasks", engine_build: "", runs: 3, streams: streamsGuess, context: 4096,
    power_limit_w: 0, driver: "", date: today(), submitter: "", notes: "" });
  const [check, setCheck] = useState<Check | null>(null);
  const [busy, setBusy] = useState(false);

  const record = () => {
    const results: Record<string, number> = {};
    for (const [k, v] of Object.entries(engine.benchmarks ?? {})) if (v) results[k] = v;
    const method: Record<string, unknown> = { ...m };
    for (const k of Object.keys(method)) if (method[k] === "" || method[k] === 0) delete method[k];
    const { benchmarks: _b, name: _n, ...eng } = engine;
    return { format: "rigsim-measurement/1", machine, engine: eng, method, results };
  };

  const run = async (download: boolean) => {
    setBusy(true);
    try {
      const rec = record();
      const r = await api<Check>("/api/measurements/validate", rec);
      setCheck(r);
      if (download && r.ok && r.id) {
        const blob = new Blob([JSON.stringify(rec, null, 1)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a"); a.href = url; a.download = `${r.id}.json`;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    } catch (e) { setCheck({ ok: false, errors: [String(e)], warnings: [] }); } finally { setBusy(false); }
  };

  const set = (k: keyof typeof m, v: string | number) => setM({ ...m, [k]: v });

  return (
    <div className="mt-3 rounded border border-line bg-paper p-4 text-sm" role="dialog" aria-label="Export measurement">
      <div className="font-medium">How was it measured?</div>
      <p className="text-xs text-muted">Alias only, no names, addresses or links. The record carries this machine and engine configuration with your values.</p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="tool"><input type="text" aria-label="method tool" value={m.tool} onChange={(e) => set("tool", e.target.value)} /></Field>
        <Field label="engine build"><input type="text" aria-label="method engine_build" value={m.engine_build} onChange={(e) => set("engine_build", e.target.value)} /></Field>
        <Field label="runs"><input type="number" aria-label="method runs" min={1} value={m.runs} onChange={(e) => set("runs", Number(e.target.value))} /></Field>
        <Field label="streams (concurrent requests)"><input type="number" aria-label="method streams" min={1} value={m.streams} onChange={(e) => set("streams", Number(e.target.value))} /></Field>
        <Field label="context tokens"><input type="number" aria-label="method context" min={0} value={m.context} onChange={(e) => set("context", Number(e.target.value))} /></Field>
        <Field label="GPU power limit W"><input type="number" aria-label="method power_limit_w" min={0} value={m.power_limit_w} onChange={(e) => set("power_limit_w", Number(e.target.value))} /></Field>
        <Field label="driver"><input type="text" aria-label="method driver" value={m.driver} onChange={(e) => set("driver", e.target.value)} /></Field>
        <Field label="date"><input type="text" aria-label="method date" value={m.date} onChange={(e) => set("date", e.target.value)} /></Field>
        <Field label="submitter alias"><input type="text" aria-label="method submitter" value={m.submitter} onChange={(e) => set("submitter", e.target.value)} /></Field>
        <Field label="notes"><input type="text" aria-label="method notes" value={m.notes} onChange={(e) => set("notes", e.target.value)} /></Field>
      </div>
      <div className="mt-3 flex gap-2">
        <button className="btn" disabled={busy} onClick={() => run(false)}>Check</button>
        <button className="btn btn-primary" disabled={busy} onClick={() => run(true)}>Check and download</button>
        <button className="btn btn-quiet" onClick={onClose}>Close</button>
      </div>
      {check && (
        <>
          <Notices errors={check.errors} warnings={check.warnings} />
          {check.ok && <div className="mt-2 text-xs text-mask-deep">Record id {check.id}. Add the file to the repository's <code>measurements/</code> folder in a pull request.</div>}
        </>
      )}
    </div>
  );
}
