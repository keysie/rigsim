import { render, screen, waitFor, within, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it } from "vitest";
import App from "../App";
import { useStore } from "../store";

beforeEach(() => { cleanup(); localStorage.clear(); useStore.getState().reset(); });

const open = async (hash: string) => {
  location.hash = hash;
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog|Starting the simulator/)).toBeNull(), { timeout: 10000 });
};

it("removing an engine names the emptied roles and can be undone", async () => {
  await open("#engines");
  await screen.findAllByText(/writes, alone/, {}, { timeout: 15000 });
  const sub = screen.getByDisplayValue("vLLM 27B").closest("section")!;
  await userEvent.click(within(sub).getByRole("button", { name: "Remove engine" }));
  await screen.findByText(/Removed “vLLM 27B”\. It was assigned to: sub-makers — those roles are now empty\./);
  expect(useStore.getState().engines.some((e) => e.name === "vLLM 27B")).toBe(false);
  await userEvent.click(screen.getByRole("button", { name: "Undo" }));
  expect(useStore.getState().engines.some((e) => e.name === "vLLM 27B")).toBe(true);
  expect(screen.queryByText(/Removed “vLLM 27B”/)).toBeNull();
}, 30000);

it("a stale role is cleared and the locked Simulate button is explained", async () => {
  useStore.setState({ engines: useStore.getState().engines.filter((e) => e.id !== "maker") });
  await open("#roles");
  await screen.findByText(/No engine is assigned to the maker role/, {}, { timeout: 15000 });
  expect(useStore.getState().roles.maker).toBeNull();
  expect(screen.getByRole("button", { name: "Simulate" })).toBeDisabled();
}, 30000);
