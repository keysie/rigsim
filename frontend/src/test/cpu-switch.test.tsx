import { render, screen, waitFor, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it } from "vitest";
import App from "../App";
import { useStore } from "../store";

beforeEach(() => { cleanup(); localStorage.clear(); useStore.getState().reset(); });

const open = async () => {
  location.hash = "#machine";
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog|Starting the simulator/)).toBeNull(), { timeout: 10000 });
};

it("after a SoC prebuilt the CPU can be switched and the second socket comes back", async () => {
  await open();
  await userEvent.selectOptions(screen.getByLabelText("prebuilt system"), "dgx-spark");
  await waitFor(() => expect(useStore.getState().machine.cpus[0].cpu_id).toBe("gb10"));
  expect(useStore.getState().machine.cpus.length).toBe(2);
  expect(screen.getByText("CPU 2", { exact: true })).toBeInTheDocument();
  await userEvent.selectOptions(screen.getByLabelText("processor CPU 1"), "ryzen9-7950x");   // a one-socket CPU: used to throw
  await waitFor(() => expect(useStore.getState().machine.cpus[0].cpu_id).toBe("ryzen9-7950x"));
  expect(useStore.getState().machine.devices.some((d) => d.model_id === "gb10-gpu")).toBe(false);
  await userEvent.selectOptions(screen.getByLabelText("processor CPU 1"), "epyc-9654");
  await waitFor(() => expect(screen.getByLabelText(/second socket/)).not.toBeDisabled());
}, 30000);

it("removing a GPU drops it from the engines that used it", async () => {
  await open();
  await screen.findByText(/48 of 48 PCIe lanes/, {}, { timeout: 10000 });
  await userEvent.click(screen.getByRole("button", { name: "Remove gpu-3" }));
  await waitFor(() => expect(useStore.getState().engines.some((e) => e.gpus.includes("gpu-3"))).toBe(false));
}, 30000);
