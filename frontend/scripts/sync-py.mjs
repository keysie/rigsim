// Copies the Python simulator and the Pyodide runtime into public/ so the built site is self-contained.
import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
const hash = (buf) => createHash("sha256").update(buf).digest("hex").slice(0, 12);
import { join, relative } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const src = join(root, "..", "backend", "app");
const dstPy = join(root, "public", "py", "app");
const dstPyo = join(root, "public", "pyodide");

rmSync(join(root, "public", "py"), { recursive: true, force: true });
mkdirSync(dstPy, { recursive: true });
const files = [];
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) { if (name !== "__pycache__") walk(p); continue; }
    if (name.endsWith(".py")) { const rel = relative(src, p); cpSync(p, join(dstPy, rel)); files.push({ path: "app/" + rel.split("\\").join("/"), v: hash(readFileSync(p)) }); }
  }
};
walk(src);
files.sort((a, b) => a.path.localeCompare(b.path));
const buildId = hash(JSON.stringify(files));
writeFileSync(join(root, "public", "py", "manifest.json"), JSON.stringify({ build: buildId, files }, null, 1));
writeFileSync(join(root, "src", "build-id.ts"), `// written by scripts/sync-py.mjs: hash over the Python files this build ships\nexport const BUILD_ID = "${buildId}";\n`);

rmSync(dstPyo, { recursive: true, force: true });
mkdirSync(dstPyo, { recursive: true });
const pyo = join(root, "node_modules", "pyodide");
for (const f of ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]) cpSync(join(pyo, f), join(dstPyo, f));
// community measurement records -> one file the site loads at start
const measDir = join(root, "..", "measurements");
const records = [];
for (const name of readdirSync(measDir)) {
  if (name.endsWith(".json") && name !== "schema.json") records.push(JSON.parse(readFileSync(join(measDir, name), "utf8")));
}
mkdirSync(join(root, "public", "measurements"), { recursive: true });
const recText = JSON.stringify({ records });
writeFileSync(join(root, "public", "measurements", "records.json"), recText);
writeFileSync(join(root, "public", "measurements", "version.txt"), hash(recText));
console.log(`synced ${files.length} python files, ${records.length} measurement records and the pyodide runtime`);
