import { useEffect, useRef, useState } from "react";
import { api, exportConfig, onBootPhase, parseConfig, runsInBrowser, sanitizeConfig, useStore, type Catalog } from "./store";
import ErrorBoundary from "./components/ErrorBoundary";
import MachinePage from "./pages/MachinePage";
import EnginesPage from "./pages/EnginesPage";
import RolesPage from "./pages/RolesPage";
import LivePage from "./pages/LivePage";
import CalibrationPage from "./pages/CalibrationPage";
import MeasurePage from "./pages/MeasurePage";

const PAGES = [
  { id: "machine", title: "Machine", blurb: "CPUs, memory, PCIe links, SSDs and GPUs" },
  { id: "engines", title: "Engines", blurb: "the programs that run the models; add and remove them here" },
  { id: "roles", title: "Roles & simulate", blurb: "the agent system, its load, and where it stalls" },
  { id: "live", title: "Live run", blurb: "watch one run: log and load per part" },
  { id: "calibration", title: "Calibration data", blurb: "measured points, sources, and claims checked" },
  { id: "measure", title: "How to measure", blurb: "taking the numbers the model wants" },
] as const;
type PageId = (typeof PAGES)[number]["id"];

const pageFromHash = (): PageId => {
  const h = location.hash.replace("#", "").toLowerCase();
  return (PAGES.some((p) => p.id === h) ? h : "machine") as PageId;
};

export default function App() {
  const [page, setPage] = useState<PageId>(pageFromHash);
  const catalog = useStore((s) => s.catalog);
  const setCatalog = useStore((s) => s.setCatalog);
  const reset = useStore((s) => s.reset);
  const loadConfig = useStore((s) => s.loadConfig);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const repairNote = (notes: string[]) => `Repaired settings: ${notes.slice(0, 3).join("; ")}${notes.length > 3 ? ` (+${notes.length - 3} more)` : ""}`;
  const [boot, setBoot] = useState("starting");
  useEffect(() => onBootPhase(setBoot), []);
  const fileInput = useRef<HTMLInputElement>(null);

  const save = () => {
    const st = useStore.getState();
    const blob = new Blob([JSON.stringify(exportConfig(st), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ai-host-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const load = async (file: File | undefined) => {
    if (!file) return;
    const r = parseConfig(await file.text());
    if (!r.ok) { setNotice(`Could not load ${file.name}: ${r.error}`); return; }
    if (!catalog) { setNotice("The simulator is still starting; try again in a moment."); return; }
    const fixed = sanitizeConfig(r.config, catalog);
    loadConfig({ ...r.config, machine: fixed.machine, engines: fixed.engines, roles: fixed.roles });
    setNotice(`Loaded ${file.name}${r.config.saved ? ` (saved ${r.config.saved.slice(0, 16).replace("T", " ")})` : ""}.${fixed.notes.length ? " " + repairNote(fixed.notes) : ""}`);
    if (fileInput.current) fileInput.current.value = "";
  };

  useEffect(() => {
    api<Catalog>("/api/catalog").then((cat) => {
      // repair saved settings against this version's catalog BEFORE any page is drawn: a page that meets an
      // unknown part while rendering would otherwise fail on its first frame
      const st = useStore.getState();
      const r = sanitizeConfig({ machine: st.machine, engines: st.engines, roles: st.roles }, cat);
      if (r.notes.length) { useStore.setState({ machine: r.machine, engines: r.engines, roles: r.roles, results: null }); setNotice(repairNote(r.notes)); }
      setCatalog(cat);
    }).catch((e) => setError(String(e)));
  }, [setCatalog]);
  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="min-h-screen md:grid md:grid-cols-[230px_1fr]">
      <aside className="min-w-0 border-b md:border-b-0 md:border-r border-line bg-panel px-5 py-6 md:min-h-screen">
        <div className="text-base font-semibold">rigsim</div>
        <p className="mt-1 text-xs text-muted leading-snug">Compose a machine, place engines on it, run an agent
          workload and see what stalls.</p>
        <nav className="mt-6 flex md:flex-col gap-1 overflow-x-auto md:overflow-visible">
          {PAGES.map((p, i) => (
            <a key={p.id} href={`#${p.id}`}
               className={`block rounded px-3 py-2 ${page === p.id ? "bg-mask-soft text-mask-deep" : "hover:bg-paper"}`}>
              <div className="text-sm font-medium">{i + 1}. {p.title}</div>
              <div className="hidden md:block text-xs text-muted">{p.blurb}</div>
            </a>
          ))}
        </nav>
        <div className="mt-6 rounded border border-amber/40 bg-amber-soft px-3 py-2 text-xs text-amber">
          A vibe-coded side project. Calibrated on one machine and a handful of public reports; typical error ±25 %, worse off that path. Do not trust it with a purchase.
        </div>
        <div className="mt-6 flex flex-col gap-2 text-xs text-muted">
          <div className="flex gap-2">
            <button className="btn" onClick={save} title="Download machine, engines, roles and workload as a JSON file">Save file</button>
            <button className="btn" onClick={() => fileInput.current?.click()} title="Load a saved JSON file">Load file</button>
            <input ref={fileInput} type="file" accept="application/json,.json" className="hidden" aria-label="config file"
                   onChange={(e) => load(e.target.files?.[0])} />
          </div>
          <div className="hidden md:block">Everything is kept in this browser. <button className="underline" onClick={() => { if (confirm("Start over with the example machine?")) reset(); }}>Start over</button></div>
        </div>
      </aside>
      <main className="px-5 py-6 md:px-10 max-w-6xl">
        {error && <div className="mb-4 rounded bg-red-soft px-3 py-2 text-sm text-red">The catalog did not load: {error}</div>}
        {notice && (
          <div className={`mb-4 flex items-center justify-between rounded px-3 py-2 text-sm ${notice.startsWith("Could not") ? "bg-red-soft text-red" : "bg-mask-soft text-mask-deep"}`} role="status">
            <span>{notice}</span><button className="underline" onClick={() => setNotice(null)}>dismiss</button>
          </div>
        )}
        {!catalog ? <p className="text-muted">{runsInBrowser ? `Starting the simulator in your browser — ${boot}. Nothing is sent anywhere.` : "Loading the parts catalog…"}</p> : (
          <ErrorBoundary key={page} onReset={reset}>
            {page === "machine" ? <MachinePage /> : page === "engines" ? <EnginesPage /> : page === "live" ? <LivePage /> : page === "calibration" ? <CalibrationPage /> : page === "measure" ? <MeasurePage /> : <RolesPage />}
          </ErrorBoundary>
        )}
      </main>
    </div>
  );
}
