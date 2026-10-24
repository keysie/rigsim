import { render, screen, waitFor, cleanup } from "@testing-library/react";
import { beforeEach, expect, it } from "vitest";
import App from "../App";
import { useStore } from "../store";

beforeEach(() => { cleanup(); localStorage.clear(); useStore.getState().reset(); });

it("calibration page lists sources, points and claims", async () => {
  location.hash = "#calibration";
  render(<App />);
  await screen.findByText(/Trusted data points, by source/, {}, { timeout: 15000 });
  await screen.findByText("<keysie>", {}, { timeout: 15000 });
  expect(screen.getAllByText("CAL").length).toBeGreaterThan(5);
  expect(screen.getAllByText("VAL").length).toBeGreaterThan(5);
  expect(screen.getByText(/Claims against this simulator/)).toBeInTheDocument();
  expect(screen.getAllByRole("link", { name: /mustafa.net/ }).length).toBeGreaterThan(2);
  expect(screen.getByText(/not with Q2_K/)).toBeInTheDocument();
}, 20000);

it("measuring page has snippets and script links", async () => {
  location.hash = "#measure";
  render(<App />);
  await waitFor(() => expect(screen.queryByText(/Loading the parts catalog/)).toBeNull(), { timeout: 10000 });
  expect(screen.getByText(/llama-bench -m model.gguf/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "membw-calibrate.py" })).toHaveAttribute("href", expect.stringContaining("rigsim/blob/main/bench/"));
  expect(screen.getByRole("link", { name: "benchw.py" })).toBeInTheDocument();
});
