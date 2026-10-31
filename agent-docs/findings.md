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

## CPU compute: first data from the new stick questions (stick 0.13.1, two runs of 2026-10-31; measured, analysed 2026-10-31)

- **i7-13700K (board limits PL1 = PL2 = 253 W, from `powercap.json`):** 120 s all-thread sgemm is flat — 1030 GFLOP/s,
  ~4555 MHz mean clock, no change over time; temperature 77 → 88 °C. No budget transition is visible, as expected
  when PL1 = PL2.
- **i7-7600U (laptop, PL1 17 W, tau 28 s, PL2 25 W; 2 cores / 4 threads):** 194 GFLOP/s at 3427 MHz falls to ~178 within
  20 s (temperature 78 → 85 °C), plateaus ~20–42 s, then steps down to ~169 GFLOP/s at 2936 MHz from ~45 s and stays flat
  (temperature drops 3 °C at the step). Two plateaus and a step, not a smooth curve. Hypothesis (*inferred*): first part
  heat/boost decay, step = PL2 → PL1 change. The machine was warm at the start (73 °C when the idle period began).
- **Ladder (OpenBLAS static thread partition):** scales linearly while every thread has its own core (13700K 1–8
  threads: ~150 GFLOP/s per thread; 7600U 1–2 threads: ~100), then *drops* when threads share cores or E-cores join
  (13700K 8 → 12 threads: 1201 → 651; 7600U 2 → 3 threads: 202 → 150). Throughput ≈ threads × slowest thread's rate
  (*inferred*). On hybrid or SMT parts the ladder beyond the physical cores measures the straggler, not the clock or
  power budget.
- **Consequence for the energy calibration (inferred, to be checked):** `calibrate_cpu_energy.py` takes the all-core
  sgemm shortfall against the model's at-max rate as a clock share (13700K run: 1164 / 2074 GFLOP/s = 0.56) and sets
  energy per instruction from the *specified* power (125 W). But this board runs at 253 W and the sampled mean clock
  (4555 MHz of roughly 5.0 GHz all-core max, *inferred* from spec) shows ~0.9 of maximum — the budget did not bind and the
  shortfall is probably the straggler effect. That would inflate the hybrid constant (6.00 nJ); explains part of the
  1163 vs 882 GFLOP/s spread of the same CPU. The 7600U fits the clock-bound arithmetic exactly (2 cores × 32 FLOP/clk ×
  2.94 GHz = 188 GFLOP/s measured 189) — HT adds nothing.
- Same CPU, different matrix size: 24-thread sgemm n = 4096 gives 1161, n = 2048 gives 1030 GFLOP/s.
- Gaps in the data: `package_w` is −1 in both runs (RAPL energy counter unreadable inside the container), so power over
  time is not measured; the rigsim model cannot build the 7600U system (`sgemm_at_max` returns None).
