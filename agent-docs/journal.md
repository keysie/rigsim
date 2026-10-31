# Journal — rigsim

Dated log of decisions, findings and events, newest at the bottom. Append only; never rewrite old entries (correct
by adding a new entry). Rarely read: it explains how the current state came to be. Current facts live in
`findings.md`. History before 2026-10-31 is reconstructed from the repository (107 commits, 2026-10-24 to
2026-10-31) and its documents, not from the original sessions.

- **2026-10-23** — Predictions for bench pass 5 written before the pass ran (`docs/predictions/2026-10-23-pass5.md`), to
  be scored afterwards. (docs)
- **2026-10-24 .. 2026-10-31** — 107 commits, PRs up to #25. Calibration against the reference box's bench passes 1–6
  (raw requests and engine traces), then the engine overhaul. (git log)
- **2026-10-26** — Overhaul proposal consolidated on branch `overhaul-of-engine`; yardsticks identifiability noted. (docs)
- **2026-10-27** — Decided (user): spec-sheet first for all hardware; GPU internals from vendor architecture data; GPU
  clock per state from load classes; roadmap of seven steps; GPU code built (step 1 done). Set aside: clock from power
  per operation and other detailed avenues. (docs: engine.md)
- **2026-10-29** — Decided (user): CPU compute as cycle and power budgets (cpu-compute.md); performance optimisation
  deferred (performance-notes.md). Commits: CPU cores as budgets, specified package power with calibrated energy per
  instruction, package clock with the f² law, llama.cpp v0.7.0 on the CPU from its code. (docs, git log)
- **2026-10-31** — `bench/hw` removed from rigsim (PR #25); the runner and toolkits live only in rigsim-benchmark.
  (git log, docs)
- **2026-10-31** — agent-docs created in both repositories (briefing, instructions, open-points, findings, journal) on
  branch `docs/agent-docs`. Open points derived by the agent from reading are tagged [agent] and need the user's
  confirmation.
- **2026-10-31** — User confirmed (decision): the reasons for the two-repo split in the briefing are correct; changes to
  agent-docs never need clearing with the user; the pre-journal history here is inferred from the repos (an older chat
  exists but kept no journal) — from now on the journal is kept. Folder moved from `docs/agent-docs` to `agent-docs/` at the
  repository root.
- **2026-10-31** — First look at the two newest runs (stick 0.13.1: cpu-gemm-ladder and cpu-power-time) against the CPU
  budget model: 13700K flat over 120 s; 7600U shows two plateaus and a step; the OpenBLAS ladder shows a straggler effect
  beyond the physical cores. The energy calibration's premise (all-core sgemm is power-bound) looks wrong for the 13700K
  at 253 W. Recorded in findings and open-points; no code changed. (measured, analysis by agent)
