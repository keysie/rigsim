// The Strata memory/link layout: appears on Strata cards, shows overshoot loudly, reacts to arguments, overrides and the display flag.
import { chromium } from "playwright-core";
const BASE = process.argv[2] ?? "http://127.0.0.1:8090/";
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new"] });
const page = await (await browser.newContext({ viewport: { width: 1400, height: 1000 } })).newPage();
const errors = []; page.on("pageerror", (e) => errors.push(e.message));
let step = 0; const ok = (c, w) => { step++; console.log(`${c ? "ok  " : "FAIL"} ${step} ${w}`); if (!c) { console.log(errors); process.exit(1); } };
await page.goto(BASE + "#engines"); await page.waitForSelector("text=writes, alone", { timeout: 120000 }); await page.waitForTimeout(2000);
const strata = page.locator("section", { has: page.locator('input[value="Strata maker"]') });
const lay = strata.getByTestId("strata-layout");
ok((await lay.count()) === 1 && (await page.getByTestId("strata-layout").count()) === 1, "the layout appears on the Strata card only");
ok((await lay.getByText(/everything fits/).count()) === 1, "the example box fits");
ok((await lay.getByText(/expert cache \(fills the rest\)/).count()) === 2, "both cards show the expert cache filling the rest");
ok((await lay.getByText(/host RAM · Core i9-10900X/).count()) === 1 && (await lay.getByText(/expert pack \(mmap\)/).count()) === 1, "the host pool shows the expert pack");
await strata.getByLabel("parking_mib", { exact: true }).fill("8192"); await strata.getByLabel("parking_slots", { exact: true }).fill("8"); await page.waitForTimeout(1500);
ok((await lay.getByText(/fits at most 8 full conversations/).count()) >= 1, "parking shows how many full conversations fit");
await strata.getByRole("spinbutton", { name: "batch", exact: true }).fill("8");
await strata.getByLabel("expert_cache_mib", { exact: true }).fill("2048"); await page.waitForTimeout(1800);
ok((await lay.getByText(/oversubscribed by \d+ %/).count()) >= 1, "a starved cache makes the PCIe worst case visibly oversubscribed");
await lay.getByText(/Fixed allocations and assumptions/).click(); await page.waitForTimeout(300);
await lay.getByLabel("fixed hit_rate").fill("0.97"); await page.waitForTimeout(1800);
ok((await lay.getByText(/\(set by hand\)/).count()) === 1 && (await lay.getByText(/oversubscribed by \d+ %/).count()) === 0, "a hand-set hit rate is used and relieves the link");
await lay.getByLabel("fixed hit_rate").fill("");
await strata.getByLabel("expert_cache_mib", { exact: true }).fill("0"); await strata.getByRole("spinbutton", { name: "batch", exact: true }).fill("4");
await strata.getByLabel("kv_resident", { exact: true }).fill("8192"); await strata.getByLabel("context", { exact: true }).fill("131072"); await page.waitForTimeout(1800);
ok((await lay.locator('[title*="streamed KV"]').count()) >= 4, "KV streaming shows the streamed slots in the host pool");
ok((await lay.getByText(/worst case at the full 128K context/).count()) === 1, "...and states the link-bound worst case");
await page.goto(BASE + "#machine"); await page.waitForSelector("text=/PCIe lanes/", { timeout: 60000 });
await page.getByLabel("display on gpu-1").check(); await page.waitForTimeout(800);
await page.goto(BASE + "#engines"); await page.waitForSelector("text=writes, alone", { timeout: 60000 }); await page.waitForTimeout(1500);
ok((await page.getByTestId("strata-layout").locator('[title*="display / desktop"]').count()) === 1, "the display checkbox adds a display box on that card");
// wide windows: settings and graphic side by side, both on one screen; narrower windows: graphic below
for (const [w, h, side] of [[1920, 1080, true], [1600, 900, true], [1440, 900, false]]) {
  const pg = await browser.newPage({ viewport: { width: w, height: h } });
  await pg.goto(BASE + "#engines"); await pg.waitForSelector("text=writes, alone", { timeout: 120000 }); await pg.waitForTimeout(2000);
  await pg.locator("section").first().evaluate((el) => el.scrollIntoView({ block: "start" })); await pg.waitForTimeout(300);
  const m = await pg.evaluate(() => {
    const sec = document.querySelector("section"); const lay = sec.querySelector("[data-testid=strata-layout]");
    const grid = [...sec.querySelectorAll("div.grid")].filter((g) => g.querySelector("[aria-label=batch]")).pop();
    const r = lay.getBoundingClientRect(), s = sec.getBoundingClientRect();
    return { right: r.left > s.left + s.width / 3, settings: grid.getBoundingClientRect().bottom, graphic: r.bottom, vh: innerHeight };
  });
  if (side) ok(m.right && m.settings <= m.vh && m.graphic <= m.vh, `${w}x${h}: settings and graphic side by side, both in view (${Math.round(m.settings)} / ${Math.round(m.graphic)} of ${m.vh} px)`);
  else ok(!m.right, `${w}x${h}: the graphic sits below the settings`);
  await pg.close();
}
ok(errors.length === 0, "no uncaught page errors");
await browser.close(); console.log(`all ${step} layout checks passed`);
