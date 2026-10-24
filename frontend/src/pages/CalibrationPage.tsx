import { useEffect, useState } from "react";
import { api } from "../store";

type Point = { group: "CAL" | "VAL"; label: string; model: number; measured: number; error: number; source: string };
type Source = { name: string; url: string | null; note: string };
type Claim = { source: string; url: string; claim: string; claimed: number; model: number; verdict: string };
type Data = { sources: Record<string, Source>; points: Point[]; claims: Claim[] };
type Cluster = { key: { gpus: string[]; engine: string; model: string; quant: string; streams: number }; n: number; trust: number;
  results: Record<string, { median: number; n: number; spread: number }>; cpus: string[]; os: string[];
  model_decode_1: number | null; model_prefill: number | null; model_error_decode_1: number | null };

const fmt = (x: number) => (Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(1));
const errClass = (e: number) => (Math.abs(e) <= 0.15 ? "text-mask-deep" : Math.abs(e) <= 0.3 ? "text-ink" : "text-amber");

export default function CalibrationPage() {
  const [d, setD] = useState<Data | null>(null);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api<Data>("/api/calibration").then(setD).catch((e) => setErr(String(e)));
    api<{ clusters: Cluster[] }>("/api/community").then((r) => setClusters(r.clusters)).catch(() => setClusters([]));
  }, []);
  if (err) return <p className="text-red">Could not load the calibration data: {err}</p>;
  if (!d) return <p className="text-muted">Running the model against the measured points…</p>;

  const bySource = Object.keys(d.sources).map((k) => ({ key: k, ...d.sources[k], points: d.points.filter((p) => p.source === k) }))
    .filter((s) => s.points.length);
  const stats = (g: "CAL" | "VAL") => {
    const errs = d.points.filter((p) => p.group === g).map((p) => Math.abs(p.error)).sort((a, b) => a - b);
    return { n: errs.length, median: errs[Math.floor(errs.length / 2)] ?? 0, within: errs.filter((e) => e <= 0.3).length, worst: errs[errs.length - 1] ?? 0 };
  };
  const cal = stats("CAL"), val = stats("VAL");

  return (
    <div>
      <h1>Calibration data</h1>
      <p className="mt-1 text-muted">Every measured point the model was fitted to or checked against, with where it comes from. Points marked
        <span className="chip bg-mask-soft text-mask-deep mx-1">CAL</span> were used to pick the model's constants;
        <span className="chip bg-paper border border-line mx-1">VAL</span> were held out. Numbers are recomputed from the current model on each load.</p>

      <div className="mt-5 grid gap-3 sm:grid-cols-2 max-w-2xl">
        {[["Calibration", cal], ["Validation (held out)", val]].map(([t, s]) => {
          const st = s as ReturnType<typeof stats>;
          return (
            <div key={t as string} className="rounded border border-line bg-panel p-4">
              <div className="text-sm text-muted">{t as string}</div>
              <div className="mt-1 text-2xl font-semibold">{st.within} <span className="text-base font-normal text-muted">of {st.n} within ±30 %</span></div>
              <div className="text-sm text-muted">median error {(st.median * 100).toFixed(0)} %, worst {(st.worst * 100).toFixed(0)} %</div>
            </div>
          );
        })}
      </div>

      <h2 className="mt-8">Trusted data points, by source</h2>
      <div className="mt-3 flex flex-col gap-5">
        {bySource.map((s) => (
          <section key={s.key} className="rounded-lg border border-line bg-panel p-4">
            <div className="font-medium">{s.url ? <a className="underline decoration-line hover:decoration-mask" href={s.url} target="_blank" rel="noreferrer">{s.name}</a> : s.name}</div>
            <div className="text-sm text-muted">{s.note}</div>
            <table className="mt-3 w-full text-sm">
              <thead className="text-left text-muted"><tr><th className="py-1 font-normal">point</th><th className="font-normal text-right pr-3">measured</th><th className="font-normal text-right pr-3">model</th><th className="font-normal text-right">error</th><th className="font-normal pl-3">use</th></tr></thead>
              <tbody>
                {s.points.map((p) => (
                  <tr key={p.label} className="border-t border-line/60">
                    <td className="py-1 pr-3">{p.label.replace(/^box: /, "")}</td>
                    <td className="text-right pr-3">{fmt(p.measured)}</td>
                    <td className="text-right pr-3">{fmt(p.model)}</td>
                    <td className={`text-right ${errClass(p.error)}`}>{p.error > 0 ? "+" : ""}{(p.error * 100).toFixed(0)} %</td>
                    <td className="pl-3"><span className={`chip ${p.group === "CAL" ? "bg-mask-soft text-mask-deep" : "bg-paper border border-line"}`}>{p.group}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        ))}
      </div>

      <h2 className="mt-10">Community measurements</h2>
      <p className="mt-1 text-muted">Records from <code>measurements/</code> in the repository, clustered by GPUs, engine, model, quant and streams. They feed the
        engine cards as one-click measured values; they do not change the model's constants. Error = model vs the cluster median for one stream.</p>
      {clusters.length === 0 ? <p className="mt-2 text-sm text-muted">No records yet.</p> : (
        <table className="mt-3 w-full text-sm">
          <thead className="text-left text-muted"><tr><th className="py-1 font-normal">configuration</th><th className="font-normal">streams</th><th className="font-normal">n</th><th className="font-normal">results (median)</th><th className="font-normal">trust</th><th className="font-normal text-right">model error</th></tr></thead>
          <tbody>
            {clusters.map((c, i) => (
              <tr key={i} className="border-t border-line/60 align-top">
                <td className="py-1 pr-3">{c.key.engine} · {c.key.model} · {c.key.quant} · {c.key.gpus.join(" + ") || "CPU"}<div className="text-xs text-muted">{c.cpus.join(" / ")} · {c.os.join(" / ")}</div></td>
                <td className="pr-3">{c.key.streams}</td><td className="pr-3">{c.n}</td>
                <td className="pr-3">{Object.entries(c.results).map(([k, v]) => `${k} ${v.median}${v.spread ? ` ±${(v.spread * 50).toFixed(0)} %` : ""}`).join(", ")}</td>
                <td className="pr-3">{c.trust}</td>
                <td className={`text-right ${c.model_error_decode_1 === null ? "text-muted" : errClass(c.model_error_decode_1)}`}>{c.model_error_decode_1 === null ? "–" : `${c.model_error_decode_1 > 0 ? "+" : ""}${(c.model_error_decode_1 * 100).toFixed(0)} %`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2 className="mt-10">Claims against this simulator</h2>
      <p className="mt-1 text-muted">Numbers from pages that publish no build, settings or methodology, run through the model on a typical 2-channel desktop.
        The model itself sits about 20 % below the trusted llama.cpp table on small dense models, so differences inside ±25 % say nothing.</p>
      <div className="mt-3 flex flex-col gap-2">
        {d.claims.map((c, i) => (
          <div key={i} className="rounded border border-line bg-panel p-3 text-sm">
            <div className="flex flex-wrap items-baseline gap-x-3">
              <span className="font-medium">{c.claim}</span>
              <a className="text-muted underline decoration-line hover:decoration-mask" href={c.url} target="_blank" rel="noreferrer">{c.source}</a>
              <span className="ml-auto text-muted">claimed {fmt(c.claimed)} · model {isNaN(c.model) ? "does not fit" : fmt(c.model)} tok/s</span>
            </div>
            <div className={`mt-1 ${/^(too low|doubtful|not with)/.test(c.verdict) ? "text-amber" : "text-muted"}`}>{c.verdict}</div>
          </div>
        ))}
      </div>
      <p className="mt-4 text-sm text-muted">Pattern across the aggregator pages: the RTX 4090 column looks measured, the other columns look scaled from it, and the RTX 3090 is
        consistently undersold. Trust the trend across quants on one card; distrust the cross-card ratios.</p>
    </div>
  );
}
