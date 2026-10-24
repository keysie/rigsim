import { render, screen, waitFor, within, fireEvent, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it } from "vitest";
import App from "../App";
import { useStore } from "../store";

beforeEach(() => { cleanup(); localStorage.clear(); useStore.getState().reset(); });

const open = async (hash: string) => {
  location.hash = hash;
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog/)).toBeNull(), { timeout: 10000 });
};

it("entering measured values changes the derived speeds", async () => {
  await open("#engines");
  await screen.findAllByText(/writes, alone/, {}, { timeout: 10000 });
  const sub = screen.getByDisplayValue("vLLM 27B").closest("section")!;
  await within(sub).findByText(/(3[89]|4[01]) tok\/s/);
  fireEvent.change(within(sub).getByLabelText("measured decode_1"), { target: { value: "33" } });
  fireEvent.change(within(sub).getByLabelText("measured prefill"), { target: { value: "1050" } });
  await within(sub).findAllByText(/33 tok\/s/, {}, { timeout: 10000 });
  expect(within(sub).getByText(/uses your measured values/)).toBeInTheDocument();
  expect(useStore.getState().engines[1].benchmarks).toEqual({ decode_1: 33, prefill: 1050 });
}, 20000);

it("runs one scenario live with a log and gauges", async () => {
  await open("#live");
  fireEvent.change(screen.getByLabelText(/Concurrent workstreams/), { target: { value: "2" } });
  await userEvent.selectOptions(screen.getByRole("combobox"), "1000");
  await userEvent.click(screen.getByRole("button", { name: "Run and watch" }));
  await screen.findByText(/tasks done/, {}, { timeout: 60000 });
  await userEvent.click(screen.getByRole("button", { name: /Pause/ }));
  const slider = screen.getByLabelText("position") as HTMLInputElement;
  fireEvent.change(slider, { target: { value: slider.max } });          // jump to the end
  await screen.findByText(/6 of 6 tasks done/, {}, { timeout: 10000 });
  expect(screen.getAllByText(/starts, reads/).length).toBeGreaterThan(5);
  expect(screen.getAllByText(/gpu-3: /).length).toBe(2);
  expect(screen.getByText(/of 10 cores/)).toBeInTheDocument();
  // filter the log to one engine
  await userEvent.selectOptions(screen.getAllByRole("combobox")[1], "guard");
  await waitFor(() => expect(screen.queryAllByText(/Strata maker · maker/).length).toBe(0));
}, 90000);
