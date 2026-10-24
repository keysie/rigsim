// End-to-end walkthrough in a real browser against a built site.
//   cd frontend && npm run build && npx serve dist -l 8090 &
//   CHROMIUM=/path/to/chromium node e2e/walkthrough.mjs http://127.0.0.1:8090/
// Needs playwright-core. Exits non-zero on the first failed check; prints every step.
import { chromium } from "playwright-core";

const BASE = process.argv[2] ?? "http://127.0.0.1:8090/";
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new"],
});
const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true })).newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
let step = 0;
const ok = (cond, what) => {
  step++;
  console.log(`${cond ? "ok  " : "FAIL"} ${String(step).padStart(2)} ${what}`);
  if (!cond) { console.log("page errors:", errors); process.exit(1); }
};
const stored = () => page.evaluate(() => JSON.parse(localStorage.getItem("rigsim") ?? "{}").state);
const go = async (hash) => { await page.goto(BASE + "#" + hash); await page.waitForTimeout(800); };
const cpuSelect = () => page.getByLabel("processor CPU 1");
const pick = async (locator, value) => { await locator.selectOption(value); await page.waitForTimeout(1500); };

// 1. first load
await page.goto(BASE + "#machine");
await page.waitForSelector("text=Operating system", { timeout: 120000 });
ok(await page.getByText("CPU 2", { exact: true }).count() === 1, "first load: both CPU panels shown");
ok((await page.getByText(/48 of 48 PCIe lanes/).count()) === 1, "first load: the example box uses 48 of 48 lanes");

// 2. CPU switch on a normal machine
await pick(cpuSelect(), "ryzen9-9950x");
ok((await stored()).machine.cpus[0].cpu_id === "ryzen9-9950x", "switch to a desktop CPU sticks");
ok(await page.getByLabel(/second socket/).isDisabled(), "second socket disabled for a one-socket CPU");
await pick(cpuSelect(), "epyc-9654");
ok(!(await page.getByLabel(/second socket/).isDisabled()), "second socket available on EPYC");
await page.getByLabel(/second socket/).check(); await page.waitForTimeout(1500);
ok((await stored()).machine.cpus[1].enabled === true, "second socket switched on");
ok((await page.getByText(/of 128 PCIe lanes/).count()) === 2, "two lane budgets shown");

// 3. prebuilt, then switch away and back (the regressions)
await pick(page.getByLabel("prebuilt system"), "dgx-spark");
let st = await stored();
ok(st.machine.cpus.length === 2 && st.machine.cpus[0].cpu_id === "gb10", "Spark prebuilt keeps two CPU slots");
ok(await page.getByText("CPU 2", { exact: true }).count() === 1, "CPU 2 panel still there after a prebuilt");
ok((await page.getByText(/unified memory · 273 GB\/s/).count()) === 1, "Spark shows its iGPU on unified memory");
await pick(cpuSelect(), "ryzen9-7950x");
st = await stored();
ok(st.machine.cpus[0].cpu_id === "ryzen9-7950x", "switch from the Spark to a desktop CPU works");
ok(!st.machine.devices.some((d) => d.model_id === "gb10-gpu"), "the Spark's iGPU left with its CPU");
ok((await page.getByText(/integrated GPU of/).count()) === 0, "no stale iGPU error");
ok(st.engines.every((e) => e.gpus.every((g) => st.machine.devices.some((d) => d.id === g))), "engine GPU assignments pruned");
await pick(cpuSelect(), "epyc-9654");
ok(!(await page.getByLabel(/second socket/).isDisabled()), "second socket available again after a prebuilt");
await pick(cpuSelect(), "apple-m4-max");
st = await stored();
ok(st.machine.devices.some((d) => d.model_id === "m4-max-gpu"), "switching to an Apple SoC adds its iGPU");
ok([36, 48, 64, 128].includes(st.machine.cpus[0].ram_gb), "RAM snapped to an M4 Max size");
await pick(page.getByLabel("prebuilt system"), "gaming-4090");
ok((await stored()).machine.cpus[0].cpu_id === "ryzen9-9950x", "a desktop prebuilt replaces the SoC");

// 4. devices
const gpusBefore = (await stored()).machine.devices.filter((d) => d.kind === "gpu").length;
await page.getByRole("button", { name: "Add GPU" }).click(); await page.waitForTimeout(1200);
ok((await stored()).machine.devices.filter((d) => d.kind === "gpu").length === gpusBefore + 1, "Add GPU");
await page.getByLabel("link type gpu-2").selectOption("tb4"); await page.waitForTimeout(1200);
ok((await page.getByText(/external · 3 GB\/s per direction/).count()) === 1, "a GPU on Thunderbolt");
await page.getByRole("button", { name: "Remove gpu-2" }).click(); await page.waitForTimeout(1200);
ok((await stored()).machine.devices.filter((d) => d.kind === "gpu").length === gpusBefore, "Remove GPU");

// 5. back to the example and through every page
await page.evaluate(() => localStorage.clear()); await page.reload();
await page.waitForSelector("text=Operating system", { timeout: 120000 });
await go("engines");
await page.waitForSelector("text=writes, alone", { timeout: 60000 });
ok((await page.getByRole("button", { name: /^Remove( engine)?$/ }).count()) >= 5, "engines page lists the example engines");
await go("roles");
await page.getByLabel(/Scenarios/).fill("3");
await page.getByRole("button", { name: "Simulate" }).click();
await page.waitForSelector("text=Tasks per hour", { timeout: 180000 });
ok((await page.getByText(/typical ·/).count()) >= 1, "simulation returns results");
await go("live");
await page.locator("select").first().selectOption("1000");
await page.getByRole("button", { name: "Run and watch" }).click();
await page.waitForSelector("text=tasks done", { timeout: 180000 });
ok((await page.getByText(/starts, reads/).count()) > 0, "live run shows a log");
await go("calibration");
await page.waitForSelector("text=Trusted data points, by source", { timeout: 60000 });
ok((await page.getByText("CAL", { exact: true }).count()) > 5, "calibration page lists points");
await go("measure");
ok((await page.getByText(/llama-bench/).count()) > 0, "measuring page renders");

// 6. save and load
await go("machine");
const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Save file" }).click()]);
const path = await dl.path();
ok(Boolean(path), "Save file downloads a configuration");
await pick(cpuSelect(), "ryzen9-7950x");
await page.getByLabel("config file").setInputFiles(path); await page.waitForTimeout(1500);
ok((await stored()).machine.cpus[0].cpu_id === "i9-10900x", "Load file restores the saved machine");

ok(errors.length === 0, `no uncaught page errors (${errors.length})`);
await browser.close();
console.log(`all ${step} checks passed`);
