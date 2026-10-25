// Every defect found in exploratory browser testing, replayed as a check (see the PR that introduced this file).
//   CHROMIUM=/path/to/chromium node e2e/robustness.mjs http://127.0.0.1:8090/
import { chromium } from "playwright-core";
import fs from "node:fs";

const BASE = process.argv[2] ?? "http://127.0.0.1:8090/";
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new"] });
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await ctx.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
let step = 0;
const ok = (cond, what) => { step++; console.log(`${cond ? "ok  " : "FAIL"} ${String(step).padStart(2)} ${what}`); if (!cond) { console.log("page errors:", errors); process.exit(1); } };
const stored = () => page.evaluate(() => JSON.parse(localStorage.getItem("rigsim") ?? "{}").state);
const boot = async (hash = "machine") => { await page.goto(BASE + "#" + hash); await page.waitForSelector("nav", { timeout: 120000 }); await page.waitForTimeout(1500); };
const fresh = async (hash = "machine") => {
  await page.goto(BASE); await page.waitForSelector("nav", { timeout: 120000 });
  page.once("dialog", (d) => d.accept());
  await page.getByRole("button", { name: "Start over" }).click(); await page.waitForTimeout(800);   // writes the defaults to storage
  await page.goto(BASE + "#" + hash); await page.reload();            // a hash-only navigation would keep the old state in memory
  await page.waitForSelector("nav", { timeout: 120000 }); await page.waitForTimeout(2500);
};
const card = (name) => page.locator("section", { has: page.locator(`input[value="${name}"]`) });
const panels = async () => (await page.locator("section aside").allInnerTexts()).filter((t) => /tok\/s/.test(t)).length;

// ---- routing
await boot("bogus");
ok((await page.locator("h1").first().innerText()) === "Machine", "an unknown address shows the Machine page");

// ---- a saved state that names parts this version does not know must not brick the app
await fresh();
await page.evaluate(() => { const s = JSON.parse(localStorage.getItem("rigsim")); s.state.machine.cpus[0].cpu_id = "ghost-cpu";
  s.state.machine.devices.push({ id: "gpu-9", kind: "gpu", model_id: "ghost-gpu", cpu: 0, link_gen: 3, link_width: 16 }); localStorage.setItem("rigsim", JSON.stringify(s)); });
await page.reload(); await page.waitForSelector("text=Operating system", { timeout: 120000 });
ok((await page.getByText(/Repaired settings/).count()) === 1, "stored state with unknown parts is repaired and the user is told");
let st = await stored();
ok(st.machine.cpus[0].cpu_id !== "ghost-cpu" && !st.machine.devices.some((d) => d.model_id === "ghost-gpu"), "the unknown CPU and GPU are gone from the stored state");

// ---- loading a hostile file
fs.writeFileSync("/tmp/rigsim-hostile.json", JSON.stringify({ app: "rigsim", version: 1, machine: { cpus: [{ cpu_id: "nope", enabled: true, ram_gb: 1, ram_mts: 1, ram_channels: 99 }],
  devices: [{ id: "gpu-1", kind: "gpu", model_id: "nonexistent", cpu: 0, link_gen: 9, link_width: 99 }] },
  engines: [{ id: "x", name: "X", engine: "strata", version: "9", model_id: "zzz", args: {}, gpus: ["gpu-9"], cpu: 0, cpu_cores: 1, storage: null }], roles: { maker: "x" }, workload: { maker_turns: [9, 1] }, concurrency: [], scenarios: 0 }));
await page.getByLabel("config file").setInputFiles("/tmp/rigsim-hostile.json"); await page.waitForTimeout(2500);
ok((await page.getByText("Operating system").count()) === 1, "loading a file full of unknown parts does not blank the page");
ok((await page.getByText(/Loaded rigsim-hostile.json.*Repaired settings/).count()) === 1, "...and says what was repaired");

// ---- machine page
await fresh();
const gsel = page.locator("select").filter({ has: page.locator("option", { hasText: "GeForce RTX 3090" }) }).first();
const gopts = await gsel.locator("option").allInnerTexts();
ok(!gopts.some((t) => /Apple|GB10|Strix|8060S/.test(t)), "the GPU list offers no integrated GPUs of other SoCs");
await page.getByLabel("processor CPU 1").selectOption("epyc-9654"); await page.waitForTimeout(1500);
ok((await stored()).machine.cpus[0].ram_channels === 12, "switching to a 12-channel CPU populates all 12 channels");
await page.getByLabel("processor CPU 1").selectOption("i9-10900x"); await page.waitForTimeout(1000);
for (let i = 0; i < 10; i++) { const b = page.getByRole("button", { name: "Add GPU" }); if (await b.isDisabled()) break; await b.click(); await page.waitForTimeout(120); }
st = await stored();
ok(st.machine.devices.filter((d) => d.kind === "gpu").length === 8 && await page.getByRole("button", { name: "Add GPU" }).isDisabled(), "Add GPU stops at 8");

// ---- engines page
await fresh("engines");
await page.waitForSelector("text=writes, alone", { timeout: 60000 }); await page.waitForTimeout(1000);
const n0 = await panels();
const vllm = card("vLLM 27B");
await vllm.getByLabel("max_num_seqs").fill("0"); await page.waitForTimeout(1500);
ok((await panels()) === n0, `an invalid number no longer blanks the other panels (${await panels()} of ${n0})`);
await page.getByRole("heading", { name: "Engines" }).click(); await page.waitForTimeout(500);
ok((await vllm.getByLabel("max_num_seqs").inputValue()) === "1", "...and the field settles on its smallest valid value");
const strata = card("Strata maker");
const t0 = Date.now();
await strata.getByRole("spinbutton", { name: "batch", exact: true }).fill("99999999"); await page.waitForTimeout(1500);
ok(Date.now() - t0 < 8000 && (await panels()) === n0, "typing 99999999 into Strata's batch does not hang the page");
ok((await stored()).engines[0].args.batch === 8, "...it is limited to the schema's 8");
await strata.getByText(/Measured on real hardware/).click();
await strata.getByLabel("measured decode_1").fill("-5"); await page.waitForTimeout(1200);
ok(!/-\d+ tok\/s/.test(await strata.locator("aside").innerText()), "a negative measured speed is not shown as one");
await strata.getByLabel("measured decode_1").fill("");
await strata.getByRole("combobox").nth(0).selectOption("vllm"); await page.waitForTimeout(1500);
const aside = await strata.locator("aside").innerText();
ok(/tok\/s/.test(aside) && !/tensor_parallel/.test(aside), "switching Strata -> vLLM follows the GPU count for tensor parallel");
await page.getByRole("button", { name: "Add engine" }).click(); await page.waitForTimeout(1500);
const last = page.locator("section").last();
ok(!/no GPU assigned/.test(await last.innerText()), "a new engine starts on a GPU instead of with an error");
await card("vLLM 27B").locator('input[type="text"]').first().fill("Guard"); await page.waitForTimeout(500);
await page.goto(BASE + "#roles"); await page.waitForSelector("text=Concurrent workstreams", { timeout: 60000 }); await page.waitForTimeout(800);
const ropts = await page.locator("section select").first().locator("option").allInnerTexts();
ok(new Set(ropts).size === ropts.length, "engines with the same name stay distinguishable in the role lists");

// ---- argument rules: what the engine would ignore is greyed out and says why
await fresh("engines");
await page.waitForSelector("text=writes, alone", { timeout: 60000 }); await page.waitForTimeout(1000);
const st2 = card("Strata maker");
ok(await st2.getByLabel("pipeline_windows").isDisabled(), "Strata with 4 slots: pipeline_windows is greyed out");
ok((await st2.getByText(/off with --batch above 1/).count()) >= 1, "...and the reason is shown");
await st2.getByRole("spinbutton", { name: "batch", exact: true }).fill("1"); await page.waitForTimeout(1500);
ok(!(await st2.getByLabel("pipeline_windows").isDisabled()), "with one slot it becomes available again");
ok(await st2.getByLabel("batch_groups").isDisabled(), "...while batch_groups is greyed out (nothing to pipeline)");
const vl = card("vLLM 27B");
await vl.getByLabel("quantization").selectOption("fp8"); await page.waitForTimeout(1500);
ok((await vl.getByText(/FP8 weights need Ada Lovelace/).count()) >= 1 && (await vl.getByLabel("quantization").inputValue()) === "awq4", "FP8 on a 3090 is forced back to AWQ with the reason");
ok(await vl.getByLabel("speculative").isDisabled() === false || (await vl.getByText(/no MTP/).count()) === 0, "speculative decoding is offered for a model with an MTP head");
await vl.getByRole("combobox").nth(2).selectOption("qwen3-32b"); await page.waitForTimeout(1500);
ok(await vl.getByLabel("speculative").isDisabled(), "...and greyed out for one without");

// ---- roles page
await fresh("roles");
const conc = page.getByLabel(/Concurrent workstreams/);
await conc.fill(""); await conc.pressSequentially("1,2,4");
ok((await conc.inputValue()) === "1,2,4" && (await page.getByText("using 1, 2, 4").count()) === 1, "concurrency can be typed character by character (the comma survives)");
await conc.fill("abc, 64, 3");
ok((await page.getByText(/ignored: abc, 64/).count()) === 1 && (await page.getByText(/using 3/).count()) === 1, "ignored concurrency entries are named");
await conc.fill("1");
await page.getByLabel(/Scenarios/).fill("2");
await page.getByText(/Workload: what the agents do/).click();
await page.getByLabel("model turns per task, from").fill("30"); await page.getByLabel("model turns per task, to").fill("6");
await page.getByRole("button", { name: "Simulate" }).click();
await page.waitForSelector("text=Throughput, power, utilisation", { timeout: 120000 });
ok((await page.locator(".text-red").count()) === 0 && (await page.getByText(/maker_turns.*swapped/).count()) >= 1, "reversed workload ranges are repaired and reported, not crashed");

// ---- live page
await fresh("live");
await page.getByLabel(/Concurrent workstreams/).fill("2"); await page.getByLabel(/^Scenario/).fill("0");
await page.locator("select").first().selectOption("1000");
await page.getByRole("button", { name: "Run and watch" }).click();
await page.waitForSelector("text=tasks done", { timeout: 120000 });
await page.getByRole("button", { name: /Pause/ }).click();
const slider = page.getByLabel("position");
await slider.focus(); await page.keyboard.press("End"); await page.waitForTimeout(600);
const done = (await page.getByText(/tasks done/).innerText()).match(/(\d+) of (\d+)/);
ok(done && done[1] === done[2], `the slider reaches the very end (${done?.[0]} tasks done)`);
const secs = async () => { const m = (await page.locator("h2").first().innerText()).match(/(\d+):(\d+) of (\d+):(\d+)/); return m ? [Number(m[1]) * 60 + Number(m[2]), Number(m[3]) * 60 + Number(m[4])] : [0, 0]; };
await page.getByRole("button", { name: /Replay|Resume/ }).click(); await page.waitForTimeout(700);
const [now, total] = await secs();
ok(now < total * 0.9, `Replay at the end starts over (clock ${now}s of ${total}s)`);

// ---- phone width
const ph = await ctx.newPage(); await ph.setViewportSize({ width: 390, height: 800 });
for (const h of ["machine", "engines", "roles", "live", "calibration", "measure"]) {
  await ph.goto(BASE + "#" + h); await ph.waitForSelector("nav", { timeout: 120000 }); await ph.waitForTimeout(h === "calibration" ? 6000 : 2500);
  const d = await ph.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
  ok(d.sw <= d.cw + 4, `phone width: #${h} does not scroll sideways (${d.sw} in ${d.cw})`);
}
ok(errors.length === 0, `no uncaught page errors (${errors.length})`);
await browser.close();
console.log(`all ${step} robustness checks passed`);
