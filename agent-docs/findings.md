# Findings — rigsim (curated)

Current facts only. Stale or superseded items are corrected or removed here and recorded in the journal.
Tag: *measured* / *docs* (read from a repo document or source) / *inferred* / *decision*. Last curated: 2026-10-31.

## Accuracy of the established model (docs: README, Calibration page)

- Hand-fitted constants (about a dozen) on twelve measured points; 41 further points held out (README text). Result as
  stated: 28 of 28 calibration and 50 of 54 held-out points within ±30 %, median error 7 % and 12 %, worst miss 36 %.
  Re-derive from `calibration.py` before quoting; the README names twelve fitted points but reports 28 calibration points — check the difference.
- Calibrated mainly on **one machine** (i9-10900X, 4-ch DDR4-3200, 3 × RTX 3090 + RTX 3060). Outside its path: off by
  a factor possible.
- Strata on the reference box (docs: README): prompt reading works per chunk with a per-chunk floor; while a prompt is
  read the slots barely write (~2 % with two pipeline groups, ~15 % serial/one card); writing depends on power cap
  (flat above ~250 W on a 3090), depth, and layer split (22–24 of 48 layers on first card beat auto 28 by ~15 %);
  several slots saturate early.

## Overhaul decisions (docs: engine.md, cpu-compute.md; user decisions)

- 2026-10-26: overhaul proposal consolidated on branch `overhaul-of-engine`; Strata 0.1.43.1 is the first engine
  version (only one with enough benchmark data).
- 2026-10-27: spec-sheet first for every device; global constants in `hw/constants.py` derived by
  `yardsticks/fit_globals.py`, validated by `validate_globals.py`; GPU clock per state from two load classes; GPU code
  built (no data yet, curves flat, marked unmeasured).
- 2026-10-27: rejected/set aside as too complex: clock from power per operation; per-core link as requests in flight;
  refresh share; timings; uncore clock (see open-points, "Set aside").
- 2026-10-29: CPU compute = cycle budgets per core + power budget per package (cpu-compute.md). Energy per op is one
  global constant per microarchitecture set from the stick's cpu-gemm (one core vs all cores). Commit 83da70e: the package
  clock follows its power budget with the f² law.
- 2026-10-29: performance work deferred until predictions hold across a spread of machines.
- 2026-10-31: `bench/hw` removed from rigsim; the runner and toolkits live only in rigsim-benchmark (PR #25).

## Hardware yardsticks (docs: yardsticks.md)

- Memory reads reach 80–86 % of theoretical peak on Xeon nodes; 1:1 read/write mixes 71–78 %. The model has no
  turnaround cost: it predicts mix = all reads, measured 11–17 % lower.
- One core cannot saturate memory (≈15–25 % of socket bandwidth); couples cores to bandwidth via Little's law.
- Socket link is a separate narrow link (34–56 GB/s). Latency rises under load; the model uses a fixed time.
- Free parameters per CPU family: RAM efficiency, fixed fabric latency, socket-link latency/efficiency, per-core
  bandwidth, router capacity. Per node the MLC output gives 5 usable results → 4 fitted, 1 validation.
- Rejected until explained: one 18-DIMM-on-12-channels Xeon 8280 result (44 % of peak).

## Simulation cost (docs: performance-notes.md, measured 2026-10-29, i7-13700K / RTX 3080 run)

- Machine graph build 140 ms; one engine step ~20 µs; one graph solve ~0.3 ms (0.29–0.8 ms with concurrent transfers).
  llama-bench tg128 on Llama 2 7B: 1.15 s, 61,824 steps. Multi-GPU tg128 naively ~30 s per value.

## Interface with rigsim-benchmark (docs)

- Predictions are made per release tag; `calibration: true` marks a value whose measurement went into a constant.
  Currently **no** prediction is marked calibration (predict.py docstring).
- `predict.py` supports: GPUs by nvidia-smi name and memory (catalogue `hw/gpus.py`), power limits, PCIe from
  `inventory/pcie.json`, CPU memory values (MLC idle latency, all-core read bandwidth) for Intel Core 6th–13th gen and
  AMD Ryzen 1000–7000; everything else returned as `unsupported` with the reason.
- Tags: v0.1.0, v0.2.0, v0.3.0 exist.
