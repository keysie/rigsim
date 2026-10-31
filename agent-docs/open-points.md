# Open points — rigsim

Roughly by priority (top = most important). Status tags: **[user]** = stated by the user or in a repo doc,
**[agent]** = derived by an agent from reading, not yet confirmed. Closed items move to the journal, not here.
Last revised: 2026-10-31.

## High

1. **Time in the CPU budget model** [user: `docs/overhaul/cpu-compute.md`]. Budgets are static; real CPUs run PL2 for up
   to tau (28–56 s), then PL1, and change with temperature. One budget per workload duration is chosen and stated
   for now.
2. **Engine framework (roadmap step 3)** [user: engine.md §11]: engine interface, roles and binding, step graphs,
   state tables, the profile — framework only, no engines. Then step 4: take apart one or two popular open-source
   benchmark tools and model them as engines.
3. **Which microbenchmarks define each primitive** [user: engine.md §9]: core-type compute per operation class,
   single-core bandwidth, GPU clock vs power limit under compute- and memory-heavy load, host-device copies,
   reference GEMM per precision. The stick's questions in rigsim-benchmark are the intended source.
4. **Kernel factors are unmeasured (1.0)** [user: engine.md §9]; they must come from per-kernel measurements
   (`test-backend-ops` on the stick), never from llama-bench, which validates.

## Medium

5. **Open from the hardware validation** [user: engine.md §5]: queue sizes for Little's law (optimisation manuals);
   DDR4 vs DDR5 read efficiency (sub-channels); Zen APUs' data-fabric link; drives are not modelled yet.
6. **Strata at small expert caches** [user: engine.md §9]: prompt reading was PCIe-bound (239 tok/s at every worker
   count, pass 3); the CPU expert path in writing is limited per core (compute vs per-core bandwidth not separated).
7. **Pass 5 predictions: score them** [agent]: `docs/predictions/2026-10-23-pass5.md` was written to be scored against
   the pass; no scoring is recorded in the repo. Ask the user whether pass 5 results exist.
8. **Strata engine continuation (roadmap step 7)** and the known Strata misses: 16K/32K-chunk prompts ~30 % too fast,
   Blower (x8) writes ~23 % faster than modelled, 2 × TITAN RTX +25–32 % [user: README].
9. **Make the data useful for tuning** (roadmap steps 5–6) [user]: public result sets of the modelled tools for
   validation; tune the hardware model afterwards.
10. **Simulation cost** [user: performance-notes.md, 2026-10-29]: accept slow runs (~7 s for 17 predictions, naive
    multi-GPU tg128 ~30 s per value) until predictions hold up across several benchmarked machines; then cache rates
    by step structure and interpolate tokens by depth.

## Low

11. **README is out of date and has damaged text** [agent]: it describes the pre-overhaul fitted model as the model;
    contains broken fragments ("…worst held-out miss 36 % … see below).py` keeps the model inside those bounds",
    "What this does **not** cover:What this does **not** cover"); test counts look stale. Propose a rewrite once the
    user decides how the established and overhauled models are presented.
12. **`bench/README.md` still documents a `hw/` layer** [agent] that was removed on 2026-10-31 (PR #25); the runner now
    lives only in rigsim-benchmark. Fix the table and link.
13. **Two models coexist** [agent]: perf.py/sim (site) and hw/+engine/ (overhaul, used by `predict.py`). When and how the
    site switches is not recorded in the repo. Ask the user.
14. **Not covered by the model, by design** [user: README, engine.md §8]: sharing a card or cores between engines, CPU
    caches, NUMA inside a socket, thermals beyond the power limit, hyperthreads, answer quality.
15. **Set aside, kept as avenues** [user: engine.md §10]: clock from power per operation, per-core link as requests in
    flight, refresh share, timings in bandwidth, uncore clock, power-state latency, latency under load.
