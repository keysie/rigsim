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
await lay.getByText(/Fixed allocations and assumptions/).click(); await page.waitForTimeout(300);
await lay.getByLabel("fixed hit_rate").fill("0.2"); await page.waitForTimeout(1800);          // a starved cache, hit rate set low
ok((await lay.getByText(/oversubscribed by \d+ %/).count()) >= 1, "a starved cache makes the PCIe worst case visibly oversubscribed");
await lay.getByLabel("fixed hit_rate").fill("0.97"); await page.waitForTimeout(1800);
ok((await lay.getByText(/\(set by hand\)/).count()) === 1 && (await lay.getByText(/oversubscribed by \d+ %/).count()) === 0, "a hand-set hit rate is used and relieves the link");
await lay.getByLabel("fixed hit_rate").fill("");
await strata.getByLabel("expert_cache_mib", { exact: true }).fill("0"); await strata.getByRole("spinbutton", { name: "batch", exact: true }).fill("4");
await strata.getByLabel("kv_resident", { exact: true }).fill("8192"); await strata.getByLabel("context", { exact: true }).fill("131072"); await page.waitForTimeout(1800);
ok((await lay.locator('[title*="streamed KV"]').count()) >= 4, "KV streaming shows the streamed slots in the host pool");
ok((await lay.getByText(/at the full 128K context every window streams [0-9.]+ GB of K\/V per slot/).count()) === 1, "...and states the link-bound worst case");
await page.goto(BASE + "#machine"); await page.waitForSelector("text=/PCIe lanes/", { timeout: 60000 });
await page.getByLabel("display on gpu-1").check(); await page.waitForTimeout(800);
await page.goto(BASE + "#engines"); await page.waitForSelector("text=writes, alone", { timeout: 60000 }); await page.waitForTimeout(1500);
ok((await page.getByTestId("strata-layout").locator('[title*="display / desktop"]').count()) === 1, "the display checkbox adds a display box on that card");
// plausibility: what one setting must not do to another, and what the speeds must show
await page.goto(BASE + "#engines"); await page.reload(); await page.waitForSelector("text=writes, alone", { timeout: 120000 }); await page.waitForTimeout(1500);
{
  const st = page.locator("section", { has: page.locator('input[value="Strata maker"]') });
  const L = st.getByTestId("strata-layout");
  const put = async (name, v) => { const el = name === "batch" ? st.getByRole("spinbutton", { name: "batch", exact: true }) : st.getByLabel(name, { exact: true }); await el.fill(String(v)); await page.waitForTimeout(1000); };
  await put("parking_mib", 32000); await put("parking_slots", 0); await put("context", 128000);
  const seen = new Set();
  for (const b of [1, 4, 8]) { await put("batch", b); seen.add(await L.getByText(/parking fits|no parking/).first().getAttribute("title")); }
  ok(seen.size === 1, `the parked conversations do not change with the slot count (${[...seen][0]})`);
  ok(await st.getByLabel("parking_slots", { exact: true }).isEditable(), "a field with a forced value stays editable");
  ok((await st.locator("aside").first().getByText(/writes, 8 at once \(set\)/).count()) === 1, "the speeds show the configured slot count");
  ok(/host RAM\s+[0-9.]+ of 128 GB/.test(await st.locator("aside").first().innerText()), "host RAM in the speeds is the graphic's pool");
  ok(/full 125K prompt/.test(await st.locator("aside").first().innerText()), "the speeds say how long a full-context prompt takes to read");
  const before = await st.locator("aside").first().innerText();
  await L.getByText(/Fixed allocations/).click(); await L.getByLabel("fixed hit_rate").fill("0.4"); await page.waitForTimeout(1500);
  ok(/40 % hits/.test(await st.locator("aside").first().innerText()) && before !== await st.locator("aside").first().innerText(), "a hand-set hit rate changes the speeds, not only the graphic");
  await L.getByLabel("fixed hit_rate").fill(""); await page.waitForTimeout(800);
  await put("batch", 1); await st.getByLabel("multi_gpu").selectOption("peer-device"); await page.waitForTimeout(1500);
  ok((await L.getByText(/peer expert tier/).count()) >= 1, "peer-device shows the peer card as an expert tier");
  await st.getByLabel("multi_gpu").selectOption("layer-split"); await page.waitForTimeout(800);
}

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
  if (side) {
    ok(m.right && m.settings <= m.vh && m.graphic <= m.vh, `${w}x${h}: settings and graphic side by side, both in view (${Math.round(m.settings)} / ${Math.round(m.graphic)} of ${m.vh} px)`);
    const g = await pg.evaluate(() => {
      const sec = document.querySelector("section"); const lay = sec.querySelector("[data-testid=strata-layout]");
      const bars = [...lay.querySelectorAll("div.relative.mt-1")].map((e) => [Math.round(e.getBoundingClientRect().top), Math.round(e.getBoundingClientRect().bottom)]);
      const stats = [...sec.querySelectorAll("aside")].find((a) => /reads prompts/.test(a.textContent));
      return { bars, statsAbove: stats.getBoundingClientRect().bottom <= lay.getBoundingClientRect().top + 1 };
    });
    ok(g.bars.length >= 3 && g.bars.every((b) => b[0] === g.bars[0][0] && b[1] === g.bars[0][1]), `${w}x${h}: VRAM and RAM stacks share top and bottom (${JSON.stringify(g.bars[0])})`);
    ok(g.statsAbove, `${w}x${h}: the speeds sit above the graphic`);
    await pg.getByTestId("overhead-block").first().hover(); await pg.waitForTimeout(300);
    const tip = pg.getByRole("tooltip").first();
    const tb = await tip.boundingBox();
    ok(tb && tb.x >= 0 && tb.x + tb.width <= w && tb.y >= 0 && tb.y + tb.height <= h, `${w}x${h}: the overhead breakdown stays inside the window`);
    ok((await tip.isVisible()) && (await tip.innerText()).includes("CUDA context") && (await tip.innerText()).includes("kept free"), `${w}x${h}: hovering the overhead shows its parts, readable`);
  }
  else ok(!m.right, `${w}x${h}: the graphic sits below the settings`);
  await pg.close();
}
ok(errors.length === 0, "no uncaught page errors");
await browser.close(); console.log(`all ${step} layout checks passed`);
