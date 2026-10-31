# Briefing — rigsim (entry point for every new agent)

Hard limit: **16k tokens**. If this file grows beyond ~12k, move detail to `findings.md` or `docs/`.
Last revised: 2026-10-31. Facts here are *read from the repository* unless marked otherwise.

## 0. Now: next steps and recently solved

Keep this section current (instruction 3). Last updated 2026-10-31.

**User priority (2026-10-31): the measurement stick first** (work happens in rigsim-benchmark; see its agent-docs). The
CPU-model rework below waits for trustworthy stick data.

**Current sprint:** `sprint/stick-data` (both repositories; the work is in rigsim-benchmark, here only notes).

**Next steps, in order**
1. Stick fixes (rigsim-benchmark PRs #19, #22, merged) verified on a booted stick (needs the user): new runs with
   working Blender, ladder and power data.
2. **Rework the CPU energy calibration** (open-points 1): board limits from `powercap.json` instead of specified power;
   no clock share from an all-core shortfall when the sampled clock is near maximum; time behaviour from the 7600U curve.
   Needs the user's decision on the method before any code.
3. Make the model able to build the 7600U system (`sgemm_at_max` returns None); README and `bench/README.md` cleanup.

**Recently solved** (merged, newest first)
- 2026-10-31 PR #27: first analysis of the 7600U and 13700K runs recorded in agent-docs.
- 2026-10-31 PR #26: agent-docs added, outside `docs/`.
- 2026-10-31 PR #25: `bench/hw` removed from rigsim (lives only in rigsim-benchmark).
- 2026-10-29: CPU compute as cycle and power budgets (core budgets, specified package power with calibrated energy per
  instruction, package clock with the f² law, llama.cpp v0.7.0 CPU engine).
- 2026-10-30/31: CI hardening (runners pinned to Ubuntu 24.04, expressive workflow names, commit hook lets merge commits through).

## 1. What this project is

**rigsim** simulates an LLM inference host before you buy it: CPUs, memory, PCIe links, SSDs, GPUs, inference engines
(Strata, vLLM, llama.cpp, hosted API) and an agent workload (Turnstone-shaped: coordinator, maker, sub-makers, guard,
judge, reranker). It runs entirely in the browser (Python on Pyodide in a web worker); the site is static, nothing is
sent anywhere. It is a **vibe-coded side project**: one person (keysie) directs, an AI assistant writes code, model and
docs in chat sessions. Not a product, not research. Accuracy claim: ±25 % on held-out points near the calibration
path; may be off by a factor elsewhere. Never overstate this.

## 2. The two repositories and how they interact

| | **rigsim** (this repo) | **rigsim-benchmark** |
|---|---|---|
| Is | the simulator: model, catalogs, UI | a bootable USB stick that measures real machines unattended, plus the open data it produces |
| Output | static site (GitHub Pages) | `results/runs/` (CC BY 4.0), the stick image build, a data page, `predictions/` |
| Licence | Apache 2.0 | code Apache 2.0, data CC BY 4.0 |
| Changes | fast, branches, model keeps evolving | runs are **immutable** once published; format only ever grows (`results/FORMAT.md`) |

**Data flow.**
1. The stick (benchmark repo) produces a run folder per boot of a machine (inventory, low-level probes, llama-bench,
   Blender, …). Contributors submit runs by GitHub issue or PR; CI validates; the data table updates itself.
2. `tools/measurements.py` (benchmark repo) extracts stable, named values (`llama-bench/<backend>/<dev>/<model>/<test>`,
   `gpu-vram/…`, `cpu-mem-ladder/…`, `mlc/…`).
3. A **daily job in the benchmark repo** checks out rigsim's newest tag `vX.Y.Z`, runs
   `python3 -m app.predict --run <run> --measurements <m.json> --out <o.json>` from `backend/` (this repo's
   `backend/app/predict.py`) and stores `predictions/<tag>/<run>.json`. The data page shows measured value, rigsim's
   prediction and deviation. Old tags' predictions are never recomputed.
4. rigsim never imports benchmark code; benchmark never imports rigsim code beyond that CLI at a tag. **The tag is the
   only interface**, plus the shared key names and the `predict.py` output format (contract: benchmark repo
   `predictions/README.md`).
5. Data → model: runs are used to *validate* (and later calibrate global constants of) the hardware model. Nothing from
   those runs is in rigsim's constants today (`predict.py` docstring). The roadmap (`docs/overhaul/engine.md` §11,
   step 2) names the stick as the source of own low-level measurements.

**Why separate** (stated in the repos: different licences and immutability of the data; independent use — the data
"stands on its own"; the runner and toolkits live only in the benchmark repo since 2026-10-31. *Inferred, confirmed by the user 2026-10-31:* keeps validation data out of the model's repo so it cannot silently become fitting data; different release
cadence; contributors can submit runs without touching simulator code).

Do not confuse: `rigsim/measurements/` = user-submitted **engine** measurements (tokens/s of one engine, via the app's
export and the intake issue form); `rigsim-benchmark/results/runs/` = **hardware** runs from the stick.

## 3. These documents

| File | Content | Read when |
|---|---|---|
| `briefing.md` | this file | always, first |
| `instructions.md` | permanent working rules (commits, branches, propose-first, secrets) | always, second |
| `open-points.md` | what is open, roughly by priority | before proposing work |
| `findings.md` | curated facts about the model, data and decisions; stale items removed | before touching model or docs claims |
| `journal.md` | dated log of decisions and findings; rarely read | to learn how a state came to be |

How to work: read briefing + instructions; read open-points; read findings for the area at hand; propose; after
clearance branch, build, test, update these files in the same branch, open a PR. At the end of every answer re-check
the files (instruction 3). Journal entries are appended and never rewritten; findings are rewritten to stay current.

## 4. Repository layout

```
backend/app/catalog/   CPUs, GPUs, SSDs, models, engines with per-version argument schemas
backend/app/hw/        NEW hardware layer (overhaul): primitives, machine graph, gpu.py, cpus.py, dram.py, links.py,
                       power.py, rates.py, execute.py, constants.py (global constants), yardsticks/ (validation data)
backend/app/engine/    NEW engine layer: llamacpp/ (cpu, cuda, vulkan), likwid/, mlc/, model.py, gguf.py, quant.py
backend/app/perf.py, sim/, machine.py, calibration.py   the established model, simulation and Calibration page
backend/app/predict.py the CLI the benchmark repo calls (see §2)
backend/app/api.py     function-level API for the browser; main.py wraps it for a dev server
frontend/src/          React + TypeScript + Tailwind; py.worker.ts runs Python in a worker
bench/                 standalone measurement scripts for engines + the unattended queue (reference only)
measurements/          user-submitted engine measurements (schema.json), intake-validated
docs/overhaul/         engine.md (the proposal, agreed + open), yardsticks.md, cpu-compute.md, performance-notes.md
docs/predictions/      predictions written before a bench pass, to be scored
agent-docs/            this folder (repository root, deliberately outside docs/)
.github/workflows/     ci.yml (tests on PRs), commits.yml, pages.yml (build+deploy on main), intake.yml (issue forms)
```

Two models coexist right now: the **established** one (perf.py/sim/, fitted constants, what the site shows) and the
**overhaul** (hw/ + engine/, spec-sheet-first, in progress, used by `predict.py`). Know which one you touch.

## 5. The overhaul in one page (`docs/overhaul/engine.md`, consolidated 2026-10-26)

- **Principles:** mechanism over fit; calibrate at the bottom (microbenchmarks), validate at the top (end-to-end);
  every constant names something measurable; modes explicit; few honestly identified parameters.
- **Hardware model:** four primitives — storage, compute, link, router; a part is a bundle of them; the machine is a graph;
  concurrent flows share resources max-min fairly; a DRAM channel is simplex. CPU: core groups, a link per core to the
  package router. Spec-sheet first for all devices; measurements only set a few global constants.
- **CPU compute (decided 2026-10-29):** capacities, not clocks — per core cycles (issue budget + per-class port limits),
  per package power (RAPL limit or TDP) with one energy-per-op constant per microarchitecture. Open: time (PL2/tau).
- **GPU:** SMs, memory controllers, copy engines, PCIe/NVLink as primitives; clock from the spec-sheet law,
  clock ~ power^(1/3) below the limit; two load classes (compute/memory); measured curves only as user overrides.
- **Engines state demands** (allocations, work per step, reachable states); only the hardware model answers durations.
  Rates per (runtime version, operation class) from reference libraries; engine versions hold one dimensionless factor
  per class, inherited until measured.
- **Roadmap (user, 2026-10-27):** 1 GPU code (done) → 2 calibrate/validate on low-level data (published done; own via the
  stick) → 3 engine framework → 4 model one or two open-source benchmark tools as engines → 5 use their public result
  sets for validation → 6 tune the hardware model → 7 continue Strata.
- **First engine slice:** llama.cpp CUDA one GPU dense; CPU engine from llama.cpp v0.7.0 code exists (commit 9cbe18d).

## 6. Commands

```
cd backend && pip install fastapi "uvicorn[standard]" pytest httpx && python -m pytest -q   # backend tests
cd frontend && npm ci && npm run build                                                      # build site
cd frontend && VITE_API_BASE=http://127.0.0.1:8080 npm test                                 # frontend (vitest), needs dev server
cd frontend && npm run build && (cd dist && python3 -m http.server 8090 &) \
  && npx playwright-core install chromium && npm run e2e                                    # browser checks
uvicorn app.main:app --port 8080                                                            # dev server (from backend/)
```
The README's test counts (20 backend) are probably stale — there are ~25 backend test files; count before quoting.

## 7. Conventions and gotchas

- Conventional Commits, enforced by CI and `.githooks/commit-msg` (merge commits are allowed).
- CI runners are pinned to Ubuntu 24.04. Pages deploys on every push to `main` after tests pass.
- `frontend/package.json` version is `0.0.0`; the rigsim *release* is a git tag `vX.Y.Z` (v0.1.0–v0.3.0 exist) and
  tagging is what triggers new predictions in the benchmark repo.
- Contributor-facing pages (README, app text) must state the vibe-coded provenance and the accuracy limits.
- Known documentation rot: see `open-points.md` ("README out of date").
- The user's wider context (not in this repo): the author runs a private local-AI server project; its reference machine
  is the main calibration source. Do not put private details of that project into this repo.

## 8. Session start checklist

1. Ask the user for a fine-grained GitHub token (only `keysie/rigsim` and `keysie/rigsim-benchmark`, Contents +
   Pull requests read/write, short expiry). Treat it as a secret (instruction 9).
2. Clone, `git config core.hooksPath .githooks`, `git log --oneline -15`, check open PRs and branches.
3. Read this folder; run the tests; then propose.
