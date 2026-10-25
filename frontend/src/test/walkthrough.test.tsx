/**
 * A scripted user session: rebuild the real box through the UI, then change things and
 * read the rendered results. Prints what the user would see (run with --reporter=verbose).
 */
import { render, screen, waitFor, within, fireEvent, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it } from "vitest";
import App from "../App";
import { useStore } from "../store";

const log: string[] = [];
const say = (s: string) => { log.push(s); console.log(s); };

async function openPage(hash: string) {
  cleanup();
  location.hash = hash;
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog/)).toBeNull(), { timeout: 10000 });
}

async function simulate(label: string) {
  await openPage("#roles");
  fireEvent.change(screen.getByLabelText(/Scenarios/), { target: { value: "10" } });
  fireEvent.change(screen.getByLabelText(/Concurrent workstreams/), { target: { value: "1, 2, 4" } });
  await userEvent.click(screen.getByRole("button", { name: "Simulate" }));
  await screen.findByText(/Tasks per hour/, {}, { timeout: 120000 });
  const rows = screen.getAllByText(/typical ·/).map((el) => el.textContent!);
  const maker = screen.getAllByText(/Strata maker: busy/).map((el) => el.textContent!);
  const first = screen.getAllByText(/Where the time goes/)[2].parentElement!.querySelectorAll("li");
  say(`\n== ${label}`);
  rows.forEach((r, i) => say(`   ${[1, 2, 4][i]} ws: ${r}`));
  say(`   bottleneck @4: ${first[0]?.textContent}`);
  say(`   maker @4: ${maker[2]}`);
  const reread = parseFloat(maker[2].match(/re-reads (\d+)/)?.[1] ?? "0");
  return Object.assign(rows.map((r) => parseFloat(r)), { reread });
}

it("rebuilds the AI server through the UI and checks that changes matter", async () => {
  localStorage.clear();
  useStore.getState().reset();

  // --- 1. Machine: the real box ----------------------------------------------------
  await openPage("#machine");
  await screen.findByText(/48 of 48 PCIe lanes/, {}, { timeout: 10000 });
  const m = useStore.getState().machine;
  say(`Machine page: ${m.cpus[0].cpu_id}, ${m.cpus[0].ram_gb} GB ${m.cpus[0].ram_mts} MT/s ${m.cpus[0].ram_channels}-ch; `
    + `${m.devices.filter((d) => d.kind === "gpu").length} GPUs, ${m.devices.filter((d) => d.kind === "ssd").length} SSD; `
    + `lanes line: "${screen.getByText(/of 48 PCIe lanes/).textContent}"`);
  say(`   links: ${screen.getAllByText(/Gen3 x/).map((e) => e.textContent).join(" | ")}`);

  // --- 2. Engines: today's production placement -------------------------------------
  await openPage("#engines");
  await screen.findAllByText(/writes, alone/, {}, { timeout: 10000 });
  const cards = Array.from(document.querySelectorAll("section"));
  for (const c of cards) {
    const name = (c.querySelector("input[type=text]") as HTMLInputElement)?.value;
    const aside = c.querySelector("aside")?.textContent?.replace(/\s+/g, " ").slice(0, 160);
    if (name) say(`   ${name}: ${aside}`);
  }
  const base = await simulate("Baseline: Strata on 2 x 3090 (4 slots, no parking), 27B vLLM on the third, guard+judge on the 3060, reranker on CPU");

  // --- 3. Change A: parking on the maker ----------------------------------------------
  await openPage("#engines");
  await screen.findAllByText(/writes, alone/, {}, { timeout: 10000 });
  let strata = screen.getByDisplayValue("Strata maker").closest("section")!;
  fireEvent.change(within(strata).getByLabelText("parking_mib"), { target: { value: "8192" } });
  fireEvent.change(within(strata).getByLabelText("parking_slots"), { target: { value: "8" } });
  await waitFor(() => expect(useStore.getState().engines[0].args.parking_slots).toBe(8));
  const parked = await simulate("A: parking 8 GiB / 8 conversations on the maker");

  // --- 4. Change B: the guard with a bigger model on the 3060 -------------------------------
  await openPage("#engines");
  await screen.findAllByText(/writes, alone/, {}, { timeout: 10000 });
  const guard = screen.getByDisplayValue("Guard").closest("section")!;
  await userEvent.selectOptions(within(guard).getAllByRole("combobox")[2], "qwen3.8-27b");
  await within(guard).findByText(/does not fit|needs/, {}, { timeout: 10000 }).catch(() => undefined);
  say(`   guard -> 27B on the 3060: ${guard.querySelector("aside")?.textContent?.replace(/\s+/g, " ").slice(0, 140)}`);
  await userEvent.selectOptions(within(guard).getAllByRole("combobox")[2], "ministral-3-3b");

  // --- 5. Change C: layout A3 — Strata on one card, 27B TP2 on two, judge = 27B ---------
  await openPage("#engines");
  await screen.findAllByText(/writes, alone/, {}, { timeout: 10000 });
  strata = screen.getByDisplayValue("Strata maker").closest("section")!;
  const sub = screen.getByDisplayValue("vLLM 27B").closest("section")!;
  await userEvent.click(within(sub).getByRole("button", { name: /^gpu-3/ }));      // vLLM frees gpu-3 first: one engine per card
  await userEvent.click(within(strata).getByRole("button", { name: /^gpu-1/ }));   // Strata off its pair...
  await userEvent.click(within(strata).getByRole("button", { name: /^gpu-2/ }));
  await userEvent.click(within(strata).getByRole("button", { name: /^gpu-3/ }));   // ...onto gpu-3
  fireEvent.change(within(strata).getByLabelText("batch"), { target: { value: "1" } });
  await userEvent.click(within(sub).getByRole("button", { name: /^gpu-1/ }));      // vLLM takes the freed pair
  await userEvent.click(within(sub).getByRole("button", { name: /^gpu-2/ }));
  await userEvent.selectOptions(within(sub).getByLabelText("tensor_parallel"), "2");
  await waitFor(() => expect(useStore.getState().engines[1].gpus).toEqual(["gpu-1", "gpu-2"]));
  await new Promise((r) => setTimeout(r, 600));
  say(`   Strata on the Blower (x8): ${strata.querySelector("aside")?.textContent?.replace(/\s+/g, " ").slice(0, 150)}`);
  say(`   27B TP2 over PCIe: ${sub.querySelector("aside")?.textContent?.replace(/\s+/g, " ").slice(0, 150)}`);
  await openPage("#roles");
  await userEvent.selectOptions(screen.getAllByRole("combobox")[4], "sub");         // judge = 27B
  const a3 = await simulate("C: Strata on one card + 27B tensor-parallel on two, the 27B also judges");

  // --- 6. Change D: NVLink between the two 3090s under the TP2 engine -----------------
  await openPage("#machine");
  await screen.findByText(/48 of 48 PCIe lanes/, {}, { timeout: 10000 });
  await userEvent.click((await screen.findAllByRole("button", { name: /Add NVLink bridge/ }))[0]);
  await screen.findByText(/NVLink bridge · 112 GB\/s/);
  const nvl = await simulate("D: + NVLink bridge on gpu-1/gpu-2");

  // --- 7. Change E: fewer approvals (SSH tools split by risk) -------------------------
  await openPage("#roles");
  const wl = useStore.getState().workload;
  useStore.getState().setWorkload({ ...wl, p_approval: [0, 0.1] });
  const fewAppr = await simulate("E: approval share 0-10 % instead of 0-50 %");

  // --- 8. Change F: swap the three 3090s for 4090s (same lanes) ------------------------
  await openPage("#machine");
  await screen.findByText(/48 of 48 PCIe lanes/, {}, { timeout: 10000 });
  const mm = structuredClone(useStore.getState().machine);
  mm.devices.forEach((d) => { if (d.model_id === "rtx-3090") d.model_id = "rtx-4090"; });
  mm.nvlinks = [];
  useStore.getState().setMachine(mm);
  await screen.findByText(/no NVLink on this model/, {}, { timeout: 10000 });
  const r4090 = await simulate("F: RTX 4090s instead of 3090s (no NVLink)");

  say("\nSummary (tasks/h at 1 / 2 / 4 workstreams):");
  say(`   baseline ${base.map((x) => x.toFixed(1)).join(" / ")}`);
  say(`   A parking ${parked.map((x) => x.toFixed(1)).join(" / ")}`);
  say(`   C A3 layout ${a3.map((x) => x.toFixed(1)).join(" / ")}`);
  say(`   D +NVLink ${nvl.map((x) => x.toFixed(1)).join(" / ")}`);
  say(`   E fewer approvals ${fewAppr.map((x) => x.toFixed(1)).join(" / ")}`);
  say(`   F 4090s ${r4090.map((x) => x.toFixed(1)).join(" / ")}`);
  expect(parked.reread).toBeLessThan(base.reread);     // parking keeps history: fewer re-reads
  expect(fewAppr[1]).toBeGreaterThan(a3[1]);           // fewer approvals help
  expect(r4090[0]).toBeGreaterThan(fewAppr[0]);        // faster cards help
}, 600000);
