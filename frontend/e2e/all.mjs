// Runs the walkthrough and the robustness checks against the same site, in order.
import { spawnSync } from "node:child_process";
const url = process.argv[2] ?? "http://127.0.0.1:8090/";
for (const f of ["walkthrough.mjs", "robustness.mjs", "cancel.mjs", "layout.mjs"]) {
  const r = spawnSync("node", [new URL(f, import.meta.url).pathname, url], { stdio: "inherit" });
  if (r.status !== 0) process.exit(r.status ?? 1);
}
