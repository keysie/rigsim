import { render, screen, waitFor, fireEvent, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import App from "../App";
import { exportConfig, parseConfig, useStore } from "../store";

beforeEach(() => { cleanup(); localStorage.clear(); useStore.getState().reset(); });

const open = async () => {
  location.hash = "#machine";
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog/)).toBeNull(), { timeout: 10000 });
};

it("saves the current configuration as a JSON download", async () => {
  await open();
  const urls: string[] = [];
  URL.createObjectURL = vi.fn((b: Blob) => { urls.push("blob:x"); (globalThis as unknown as { lastBlob: Blob }).lastBlob = b; return "blob:x"; });
  URL.revokeObjectURL = vi.fn();
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  await userEvent.click(screen.getByRole("button", { name: "Save file" }));
  expect(click).toHaveBeenCalled();
  const text = await (globalThis as unknown as { lastBlob: Blob }).lastBlob.text();
  const parsed = parseConfig(text);
  expect(parsed.ok && parsed.config.machine.devices.length).toBe(5);
});

it("loads a saved file back and rejects a foreign one", async () => {
  await open();
  const st = useStore.getState();
  const cfg = exportConfig({ ...st, machine: { ...st.machine, devices: st.machine.devices.slice(0, 3) }, scenarios: 7 });
  const input = screen.getByLabelText("config file") as HTMLInputElement;
  const file = new File([JSON.stringify(cfg)], "box.json", { type: "application/json" });
  Object.defineProperty(file, "text", { value: () => Promise.resolve(JSON.stringify(cfg)) });
  fireEvent.change(input, { target: { files: [file] } });
  await screen.findByText(/Loaded box.json/);
  expect(useStore.getState().machine.devices.length).toBe(3);
  expect(useStore.getState().scenarios).toBe(7);
  const bad = new File(["{\"hello\":1}"], "other.json", { type: "application/json" });
  Object.defineProperty(bad, "text", { value: () => Promise.resolve("{\"hello\":1}") });
  fireEvent.change(input, { target: { files: [bad] } });
  await screen.findByText(/Could not load other.json: not a rigsim config/);
  expect(useStore.getState().machine.devices.length).toBe(3);
});
