import { create } from "zustand";
import { persist } from "zustand/middleware";

// ---------------------------------------------------------------- catalog types
export type Cpu = { id: string; name: string; vendor: string; year: number; cores: number; threads: number;
  mem_channels: number; max_mem_mts: number; mem_type: string; pcie_gen: number; pcie_lanes: number; sockets: number;
  tdp_w: number; avx512: boolean; isa: string; e_cores: number; all_core_ghz: number; proposed?: boolean };
export type Gpu = { id: string; name: string; vendor: string; year: number; vram_gb: number; mem_bw_gbs: number;
  fp16_tflops: number; pcie_gen: number; pcie_lanes: number; nvlink_gbs: number; slots: number; tdp_w: number; form: string; proposed?: boolean };
export type Ssd = { id: string; name: string; pcie_gen: number; pcie_lanes: number; seq_read_gbs: number;
  rand_read_mbs: number; capacities_tb: number[] };
export type Model = { id: string; name: string; family: string; params_b: number; active_b: number; experts: number;
  mtp: boolean; vision: boolean; max_context: number; hosted?: boolean; reranker?: boolean };
export type ArgDef = { name: string; type: "int" | "float" | "bool" | "choice"; default: unknown; min?: number;
  max?: number; options?: string[]; help: string };
export type EngineDef = { name: string; kind: string; models: string[]; versions: Record<string, { args: ArgDef[] }> };
export type Prebuilt = { id: string; name: string; note: string; machine: Machine };
export type Catalog = { cpus: Cpu[]; gpus: Gpu[]; ssds: Ssd[]; ram_speeds: Record<string, number[]>;
  prebuilts: Prebuilt[]; soc_ram_gb: Record<string, number[]>;
  ram_sizes_gb: number[]; pcie_lane_gbs: Record<string, number>; models: Model[]; quants: Record<string, number>;
  engines: Record<string, EngineDef> };

// ---------------------------------------------------------------- config types
export type CpuSlot = { cpu_id: string; enabled: boolean; ram_gb: number; ram_mts: number; ram_channels: number };
export type Device = { id: string; kind: "gpu" | "ssd"; model_id: string; cpu: number; link_gen: number; link_type?: string;
  link_width: number; capacity_tb?: number | null };
export type Machine = { os: "linux" | "windows"; cpus: CpuSlot[]; devices: Device[]; nvlinks: [string, string][] };
export type EngineSpec = { id: string; name: string; engine: string; version: string; model_id: string;
  args: Record<string, unknown>; gpus: string[]; cpu: number; cpu_cores: number; storage: string | null;
  benchmarks?: Record<string, number | null> };
export type RoleName = "coordinator" | "maker" | "sub" | "guard" | "judge" | "reranker";
export type Roles = Record<RoleName, string | null>;
export type Range = [number, number];
export type Workload = { maker_turns: Range; answer_median: Range; tool_calls_mean: Range; p_delegate: Range;
  fanout: Range; tool_result_median: Range; tool_exec_median_s: Range; p_approval: Range; sub_turns: Range;
  sub_answer_median: Range; sub_tool_calls_mean: Range; sub_final_tokens: Range; task_tokens: Range;
  maker_sys: number; sub_sys: number; window: number; compact_at: number; visible_frac: number;
  replay_thinking: boolean; tasks_per_ws: number };

export const DEFAULT_WORKLOAD: Workload = {
  maker_turns: [6, 30], answer_median: [200, 700], tool_calls_mean: [0.8, 2.0], p_delegate: [0, 0.4], fanout: [1, 4],
  tool_result_median: [500, 3000], tool_exec_median_s: [0.5, 5], p_approval: [0, 0.5], sub_turns: [3, 12],
  sub_answer_median: [200, 500], sub_tool_calls_mean: [0.8, 1.6], sub_final_tokens: [300, 1500], task_tokens: [200, 1500],
  maker_sys: 11000, sub_sys: 5500, window: 32768, compact_at: 0.8, visible_frac: 0.1, replay_thinking: false, tasks_per_ws: 3,
};

export const WORKLOAD_HELP: Record<string, string> = {
  maker_turns: "model turns per task", answer_median: "answer tokens per turn (incl. thinking)",
  tool_calls_mean: "tool calls per turn", p_delegate: "share of turns that delegate to sub-agents",
  fanout: "sub-agents per delegating turn", tool_result_median: "tokens per tool result",
  tool_exec_median_s: "seconds a tool runs", p_approval: "share of tool calls needing approval (judge)",
  sub_turns: "turns per sub-agent", sub_answer_median: "sub-agent answer tokens", sub_tool_calls_mean: "sub-agent tool calls per turn",
  sub_final_tokens: "tokens a sub-agent returns", task_tokens: "task prompt tokens",
};

// the measured box as the starting point
export const DEFAULT_MACHINE: Machine = {
  os: "linux",
  cpus: [{ cpu_id: "i9-10900x", enabled: true, ram_gb: 128, ram_mts: 3200, ram_channels: 4 },
         { cpu_id: "i9-10900x", enabled: false, ram_gb: 128, ram_mts: 3200, ram_channels: 4 }],
  devices: [
    { id: "ssd-1", kind: "ssd", model_id: "nvme-gen3", cpu: 0, link_gen: 3, link_width: 4, capacity_tb: 2 },
    { id: "gpu-1", kind: "gpu", model_id: "rtx-3090", cpu: 0, link_gen: 3, link_width: 16 },
    { id: "gpu-2", kind: "gpu", model_id: "rtx-3090", cpu: 0, link_gen: 3, link_width: 16 },
    { id: "gpu-3", kind: "gpu", model_id: "rtx-3090", cpu: 0, link_gen: 3, link_width: 8 },
    { id: "gpu-4", kind: "gpu", model_id: "rtx-3060-12g", cpu: 0, link_gen: 3, link_width: 4 },
  ],
  nvlinks: [],
};

export const DEFAULT_ENGINES: EngineSpec[] = [
  { id: "maker", name: "Strata maker", engine: "strata", version: "0.1.43.1", model_id: "qwen3.8-flash-next",
    args: { batch: 4 }, gpus: ["gpu-1", "gpu-2"], cpu: 0, cpu_cores: 9, storage: "ssd-1" },
  { id: "sub", name: "vLLM 27B", engine: "vllm", version: "0.31.0", model_id: "qwen3.8-27b",
    args: { tensor_parallel: "1", max_num_seqs: 4 }, gpus: ["gpu-3"], cpu: 0, cpu_cores: 2, storage: "ssd-1" },
  { id: "guard", name: "Guard", engine: "llama.cpp", version: "b11552", model_id: "ministral-3-3b",
    args: { parallel: 1, quantization: "q8_0", ctx_size: 8192 }, gpus: ["gpu-4"], cpu: 0, cpu_cores: 1, storage: "ssd-1" },
  { id: "judge", name: "Judge", engine: "llama.cpp", version: "b11552", model_id: "granite-4.2-8b",
    args: { parallel: 1, quantization: "q4_k_m", ctx_size: 32768 }, gpus: ["gpu-4"], cpu: 0, cpu_cores: 1, storage: "ssd-1" },
  { id: "rerank", name: "Reranker", engine: "llama.cpp", version: "b11552", model_id: "bge-reranker-v2-m3",
    args: { n_gpu_layers: "0", parallel: 2 }, gpus: [], cpu: 0, cpu_cores: 4, storage: "ssd-1" },
  { id: "coord", name: "Coordinator (hosted)", engine: "hosted-api", version: "provider", model_id: "deepseek-v4.1-flash",
    args: {}, gpus: [], cpu: 0, cpu_cores: 0, storage: null },
];

export const DEFAULT_ROLES: Roles = { coordinator: "coord", maker: "maker", sub: "sub", guard: "guard", judge: "judge", reranker: "rerank" };

// ---------------------------------------------------------------- api
const API_BASE = (import.meta.env?.VITE_API_BASE as string | undefined) ?? "";

// Without VITE_API_BASE the simulator runs inside the browser (Pyodide in a worker); with it, calls go to the dev server.
let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
export const runsInBrowser = !API_BASE;

export let bootPhase = "starting";
const bootListeners = new Set<(s: string) => void>();
export function onBootPhase(fn: (s: string) => void) { bootListeners.add(fn); return () => { bootListeners.delete(fn); }; }
function setBoot(s: string) { bootPhase = s; bootListeners.forEach((f) => f(s)); }

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./py.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<{ id: number; result?: unknown; error?: string; boot?: string }>) => {
      if (e.data.boot) { setBoot(e.data.boot); return; }
      const p = pending.get(e.data.id);
      if (!p) return;
      pending.delete(e.data.id);
      if (e.data.error !== undefined) p.reject(new Error(e.data.error)); else p.resolve(e.data.result);
    };
    worker.onerror = (e) => { pending.forEach((p) => p.reject(new Error(e.message))); pending.clear(); };
  }
  return worker;
}

export async function api<T>(path: string, body?: unknown): Promise<T> {
  if (!API_BASE) {
    const id = ++seq;
    const indexURL = new URL(import.meta.env.BASE_URL, document.baseURI).href;
    return new Promise<T>((resolve, reject) => {
      pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      getWorker().postMessage({ id, path, body, indexURL });
    });
  }
  const r = await fetch(API_BASE + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

// ---------------------------------------------------------------- config files
export type ConfigFile = { app: "rigsim"; version: 1; saved: string; machine: Machine; engines: EngineSpec[];
  roles: Roles; workload: Workload; concurrency: number[]; scenarios: number };

export function exportConfig(s: Pick<State, "machine" | "engines" | "roles" | "workload" | "concurrency" | "scenarios">): ConfigFile {
  return { app: "rigsim", version: 1, saved: new Date().toISOString(), machine: s.machine, engines: s.engines,
    roles: s.roles, workload: s.workload, concurrency: s.concurrency, scenarios: s.scenarios };
}

/** Checks the shape of a loaded file; returns the config or an error message. */
export function parseConfig(text: string): { ok: true; config: ConfigFile } | { ok: false; error: string } {
  let j: unknown;
  try { j = JSON.parse(text); } catch { return { ok: false, error: "not a JSON file" }; }
  const c = j as Partial<ConfigFile>;
  if (c?.app !== "rigsim" && c?.app !== "ai-host-simulator") return { ok: false, error: "not a rigsim config (missing app marker)" };   // older files carry the pre-rename marker
  if (c.version !== 1) return { ok: false, error: `unsupported config version ${String(c.version)}` };
  if (!c.machine || !Array.isArray(c.machine.cpus) || !Array.isArray(c.machine.devices)) return { ok: false, error: "machine section is missing or malformed" };
  if (!Array.isArray(c.engines)) return { ok: false, error: "engines section is missing" };
  if (!c.roles || typeof c.roles !== "object") return { ok: false, error: "roles section is missing" };
  return { ok: true, config: {
    app: "rigsim", version: 1, saved: c.saved ?? "", machine: { ...c.machine, os: c.machine.os ?? "linux", nvlinks: c.machine.nvlinks ?? [] } as Machine, engines: c.engines,
    roles: { ...DEFAULT_ROLES, ...c.roles }, workload: { ...DEFAULT_WORKLOAD, ...(c.workload ?? {}) },
    concurrency: Array.isArray(c.concurrency) && c.concurrency.length ? c.concurrency : [1, 2, 4],
    scenarios: typeof c.scenarios === "number" ? c.scenarios : 12 } };
}

// ---------------------------------------------------------------- store
type State = {
  catalog: Catalog | null;
  machine: Machine;
  engines: EngineSpec[];
  roles: Roles;
  workload: Workload;
  concurrency: number[];
  scenarios: number;
  results: unknown | null;
  setCatalog: (c: Catalog) => void;
  setMachine: (m: Machine) => void;
  setEngines: (e: EngineSpec[]) => void;
  setRoles: (r: Roles) => void;
  setWorkload: (w: Workload) => void;
  setConcurrency: (c: number[]) => void;
  setScenarios: (n: number) => void;
  setResults: (r: unknown | null) => void;
  reset: () => void;
  loadConfig: (c: ConfigFile) => void;
  loadPrebuilt: (m: Machine) => void;
};

export const useStore = create<State>()(
  persist(
    (set) => ({
      catalog: null,
      machine: DEFAULT_MACHINE,
      engines: DEFAULT_ENGINES,
      roles: DEFAULT_ROLES,
      workload: DEFAULT_WORKLOAD,
      concurrency: [1, 2, 4],
      scenarios: 12,
      results: null,
      setCatalog: (catalog) => set({ catalog }),
      setMachine: (machine) => set({ machine, results: null }),
      setEngines: (engines) => set({ engines, results: null }),
      setRoles: (roles) => set({ roles, results: null }),
      setWorkload: (workload) => set({ workload, results: null }),
      setConcurrency: (concurrency) => set({ concurrency }),
      setScenarios: (scenarios) => set({ scenarios }),
      setResults: (results) => set({ results }),
      reset: () => set({ machine: DEFAULT_MACHINE, engines: DEFAULT_ENGINES, roles: DEFAULT_ROLES,
        workload: DEFAULT_WORKLOAD, concurrency: [1, 2, 4], scenarios: 12, results: null }),
      loadConfig: (c) => set({ machine: c.machine, engines: c.engines, roles: c.roles, workload: c.workload,
        concurrency: c.concurrency, scenarios: c.scenarios, results: null }),
      loadPrebuilt: (m) => set((s) => {
        const ids = new Set(m.devices.map((d) => d.id));
        return { machine: structuredClone(m), results: null,
          engines: s.engines.map((e) => ({ ...e, gpus: e.gpus.filter((g) => ids.has(g)),
            storage: e.storage && ids.has(e.storage) ? e.storage : null })) };
      }),
    }),
    { name: "rigsim", partialize: (s) => ({ machine: s.machine, engines: s.engines, roles: s.roles,
      workload: s.workload, concurrency: s.concurrency, scenarios: s.scenarios }) },
  ),
);
