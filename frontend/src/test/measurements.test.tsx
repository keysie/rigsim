import { render, screen, waitFor, within, fireEvent, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import App from "../App";
import { useStore } from "../store";

beforeEach(() => { cleanup(); localStorage.clear(); useStore.getState().reset(); });

const open = async (hash: string) => {
  location.hash = hash;
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog|Starting the simulator/)).toBeNull(), { timeout: 10000 });
};

it("offers community measurements that match the engine and applies them", async () => {
  await open("#engines");
  await screen.findAllByText(/Community measurements for this configuration/, {}, { timeout: 15000 });
  const sub = screen.getByDisplayValue("vLLM 27B").closest("section")!;
  expect(within(sub).getByText(/writes alone 39/)).toBeInTheDocument();
  await userEvent.click(within(sub).getByRole("button", { name: "Use" }));
  await waitFor(() => expect(useStore.getState().engines[1].benchmarks?.decode_1).toBe(39));
  await within(sub).findByText(/uses your measured values/, {}, { timeout: 10000 });
}, 30000);

it("exports a validated measurement record", async () => {
  await open("#engines");
  await screen.findAllByText(/Community measurements for this configuration/, {}, { timeout: 15000 });
  const sub = screen.getByDisplayValue("vLLM 27B").closest("section")!;
  fireEvent.change(within(sub).getByLabelText("measured decode_1"), { target: { value: "38" } });
  await userEvent.click(within(sub).getByRole("button", { name: "Export measurement" }));
  fireEvent.change(screen.getByLabelText("method submitter"), { target: { value: "tester" } });
  fireEvent.change(screen.getByLabelText("method driver"), { target: { value: "615.71" } });
  fireEvent.change(screen.getByLabelText("method power_limit_w"), { target: { value: "250" } });
  URL.createObjectURL = vi.fn(() => "blob:x"); URL.revokeObjectURL = vi.fn();
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  await userEvent.click(screen.getByRole("button", { name: "Check and download" }));
  await screen.findByText(/Record id vllm-qwen3.8-27b-awq4-rtx-3090-s1-/, {}, { timeout: 10000 });
  expect(click).toHaveBeenCalled();
  // an impossible number is refused
  fireEvent.change(within(sub).getByLabelText("measured decode_1"), { target: { value: "400" } });
  await userEvent.click(screen.getByRole("button", { name: "Check" }));
  await screen.findByText(/exceeds the memory-bus bound/, {}, { timeout: 10000 });
}, 30000);
