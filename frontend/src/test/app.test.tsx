/**
 * User-level checks against the real backend (VITE_API_BASE points at a running server).
 * They click through the three pages the way a person would.
 */
import { render, screen, waitFor, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import App from "../App";
import { useStore } from "../store";

beforeEach(() => {
  localStorage.clear();
  useStore.getState().reset();
  location.hash = "#machine";
});

const open = async () => {
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog/)).toBeNull(), { timeout: 10000 });
};

describe("machine page", () => {
  it("shows the example box with its lane budget and bandwidths", async () => {
    await open();
    await screen.findByText(/48 of 48 PCIe lanes/, {}, { timeout: 10000 });
    expect(screen.getByText(/Memory pool 128 GB at about 73.7 GB\/s/)).toBeInTheDocument();
    expect(screen.getAllByText(/Gen3 x16 · 15.76 GB\/s/).length).toBe(2);
  });

  it("refuses a second socket on a desktop CPU and allows it on an EPYC", async () => {
    await open();
    const box = screen.getByLabelText(/second socket/) as HTMLInputElement;
    expect(box).toBeDisabled();
    const cpuSelect = screen.getAllByRole("combobox")[2];   // [0] prebuilt, [1] operating system
    await userEvent.selectOptions(cpuSelect, "epyc-9654");
    await waitFor(() => expect(screen.getByLabelText(/second socket/)).not.toBeDisabled());
    await userEvent.click(screen.getByLabelText(/second socket/));
    expect(useStore.getState().machine.cpus[1].enabled).toBe(true);
    await screen.findAllByText(/on CPU/);
  }, 20000);

  it("adds a GPU, caps a too-wide link, and bridges two 3090s", async () => {
    await open();
    await userEvent.click(screen.getByRole("button", { name: "Add GPU" }));
    expect(useStore.getState().machine.devices.filter((d) => d.kind === "gpu").length).toBe(5);
    await screen.findByText(/64 PCIe lanes requested/, {}, { timeout: 10000 });
    // the first two 3090s can be bridged
    const bridge = (await screen.findAllByRole("button", { name: /Add NVLink bridge/ }))[0];
    await userEvent.click(bridge);
    expect(useStore.getState().machine.nvlinks).toEqual([["gpu-1", "gpu-2"]]);
    await screen.findByText(/NVLink bridge · 112 GB\/s/);
    // remove the GPU again
    await userEvent.click(screen.getByRole("button", { name: "Remove gpu-5" }));
    expect(useStore.getState().machine.devices.filter((d) => d.kind === "gpu").length).toBe(4);
  });
});

describe("engines page", () => {
  it("lists the example engines with derived speeds and reacts to the version", async () => {
    location.hash = "#engines";
    await open();
    await screen.findAllByText(/writes, alone/, {}, { timeout: 10000 });
    const strataCard = screen.getByDisplayValue("Strata maker").closest("section")!;
    await within(strataCard).findAllByText(/1[0-4]\d tok\/s/);
    expect(within(strataCard).getByText(/% held/)).toBeInTheDocument();
    const version = within(strataCard).getAllByRole("combobox")[1];
    await userEvent.selectOptions(version, "0.2.2");
    await within(strataCard).findByText("aux_cpus");       // only 0.2.2 has it
  }, 20000);

  it("explains when a model does not fit", async () => {
    location.hash = "#engines";
    await open();
    await screen.findAllByText(/writes, alone/, {}, { timeout: 10000 });
    const subCard = screen.getByDisplayValue("vLLM 27B").closest("section")!;
    const model = within(subCard).getAllByRole("combobox")[2];
    await userEvent.selectOptions(model, "llama-3.3-70b");
    await within(subCard).findByText(/does not fit/, {}, { timeout: 10000 });
  }, 20000);
});

describe("roles page", () => {
  it("runs a simulation and names a bottleneck", async () => {
    location.hash = "#roles";
    await open();
    fireEvent.change(screen.getByLabelText(/Scenarios/), { target: { value: "3" } });
    fireEvent.change(screen.getByLabelText(/Concurrent workstreams/), { target: { value: "1, 2" } });
    await userEvent.click(screen.getByRole("button", { name: "Simulate" }));
    await screen.findByText(/Tasks per hour/, {}, { timeout: 60000 });
    expect(screen.getByText(/2 concurrent workstreams/)).toBeInTheDocument();
    expect(screen.getAllByText(/Where the time goes/).length).toBe(2);
    expect(screen.getAllByRole("listitem").length).toBeGreaterThan(0);
  }, 70000);

  it("asks for a maker before it simulates", async () => {
    location.hash = "#roles";
    await open();
    const makerSelect = screen.getAllByRole("combobox")[1];
    await userEvent.selectOptions(makerSelect, "");
    expect(screen.getByRole("button", { name: "Simulate" })).toBeDisabled();
    expect(screen.getByText(/Choose a maker engine first/)).toBeInTheDocument();
  });
});
