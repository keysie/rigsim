/// <reference lib="webworker" />
// Runs the Python simulator inside the browser. Messages: {id, path, body} -> {id, result | error}.
type Msg = { id: number; path: string; body?: unknown; indexURL?: string };
let ready: Promise<(path: string, body?: string) => string> | null = null;

const tell = (boot: string) => (self as unknown as Worker).postMessage({ id: -1, boot });

async function boot(indexURL: string) {
  tell("downloading the Python runtime (~10 MB, cached after the first visit)");
  const mod = await import(/* @vite-ignore */ indexURL + "pyodide/pyodide.mjs");
  const py = await mod.loadPyodide({ indexURL: indexURL + "pyodide/" });
  tell("loading the simulator");
  const manifest = await (await fetch(indexURL + "py/manifest.json")).json() as { files: string[] };
  py.FS.mkdirTree("/sim");
  for (const f of manifest.files) {
    const text = await (await fetch(indexURL + "py/" + f)).text();
    const dir = "/sim/" + f.split("/").slice(0, -1).join("/");
    py.FS.mkdirTree(dir);
    py.FS.writeFile("/sim/" + f, text);
  }
  tell("first start of the model");
  py.runPython("import sys; sys.path.insert(0, '/sim')");
  const call = py.pyimport("app.api").call as (path: string, body?: string) => string;
  try {
    const recs = await (await fetch(indexURL + "measurements/records.json")).text();
    call("/api/measurements/load", recs);
  } catch { /* no records shipped: the app works without them */ }
  return call;
}

self.onmessage = async (e: MessageEvent<Msg>) => {
  const { id, path, body, indexURL } = e.data;
  try {
    if (!ready) ready = boot(indexURL ?? "./");
    const call = await ready;
    const out = call(path, body === undefined ? undefined : JSON.stringify(body));
    const parsed = JSON.parse(out);
    if (parsed && typeof parsed === "object" && "error" in parsed && Object.keys(parsed).length === 1) {
      (self as unknown as Worker).postMessage({ id, error: parsed.error });
    } else {
      (self as unknown as Worker).postMessage({ id, result: parsed });
    }
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, error: String(err) });
  }
};
