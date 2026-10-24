import { render, screen, waitFor, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it } from "vitest";
import App from "../App";
import { useStore } from "../store";

beforeEach(() => { cleanup(); localStorage.clear(); useStore.getState().reset(); });

it("a prebuilt fills the machine and engine assignments are pruned", async () => {
  location.hash = "#machine";
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog/)).toBeNull(), { timeout: 10000 });
  await userEvent.selectOptions(screen.getByLabelText("prebuilt system"), "dgx-spark");
  await screen.findByText(/unified memory · 273 GB\/s shared with the CPU/, {}, { timeout: 10000 });
  expect(useStore.getState().machine.cpus[0].cpu_id).toBe("gb10");
  expect(useStore.getState().engines.every((e) => e.gpus.every((g) => g === "gpu-1"))).toBe(true);
}, 20000);

it("a GPU can hang off Thunderbolt and stops costing lanes", async () => {
  location.hash = "#machine";
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog/)).toBeNull(), { timeout: 10000 });
  await screen.findByText(/48 of 48 PCIe lanes/);
  await userEvent.selectOptions(screen.getByLabelText("link type gpu-4"), "tb4");
  await screen.findByText(/external · 3 GB\/s per direction/, {}, { timeout: 10000 });
  // the i9-10900X (2019) has no integrated port: the freed x4 slot becomes a x4 controller card
  await screen.findByText(/48 of 48 PCIe lanes/, {}, { timeout: 10000 });
  expect(screen.getByText(/add-in card, 4 CPU lanes/)).toBeInTheDocument();
}, 20000);
