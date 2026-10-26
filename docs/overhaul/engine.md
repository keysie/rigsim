# Overhaul of the engine model

Working document for the branch `overhaul-of-engine`. Points we want to establish before anything is built.
Status: **draft, under discussion**. Nothing here is decided until marked so.

## Goal

The simulator models what the engines and the system actually do, not curves fitted to the data we have.
Speeds, waits and limits should emerge from hardware resources, engine operations and scheduler rules.

## Principles

1. **Mechanism over fit.** Every quantity is counted (bytes, layers, tokens), read from a spec sheet, or measured
   as a primitive. No constant exists only to make an end-to-end number come out right.
2. **Calibrate at the bottom, validate at the top.** Remaining efficiency factors are per hardware class and come
   from microbenchmarks (one kernel, one copy, one expert pass). End-to-end benchmarks validate only.
3. **Every constant names something measurable:** what it is, its unit, how to measure it, where the value came from.
4. **Modes are explicit.** The operating mode (cards, slots, peer-device or layer split, pipeline groups, version)
   decides which rules apply, and it is shown to the user.
5. **Few parameters, honestly identified.** If the data cannot tell two parameters apart, the model has one.

## Layers to establish

1. **Hardware resources as resources:** per card compute from clock under the power limit, memory bandwidth, PCIe
   per direction; CPU cores for experts; RAM bandwidth. Operations consume them and contend for them.
2. **Engine execution as operations:** per layer, for a window or a prompt chunk: attention or DeltaNet, router,
   hit experts on the GPU, missed experts on the CPU or copied, MTP head, sampling. Overlap and serialisation
   follow from the mode.
3. **Scheduler semantics per engine:** Strata admission, chunk interleaving with the decode share, piped groups
   reading in one run, slot choice, parking. vLLM continuous batching, token budget, chunked prefill, preemption.
   llama.cpp slots.
4. **Routing and caching:** expert selection from a popularity distribution and the engine's cache policy; hit rate
   comes out of the model.
5. **Calibration and validation:** which primitives to measure on which hardware, which end-to-end runs validate
   which layer.

## Decided so far

- **Clean split between engines and hardware.** An engine never knows what card it runs on; it states demands
  (compute of a kind on the device holding a stage, bytes moved between memory pools, memory to allocate). The
  hardware model answers durations and memory fit for everything running at the same moment.
- **Modular hardware as a graph.** Cards, CPU sockets, RAM pools and SSDs are nodes with resources; PCIe slots, the
  chipset uplink, NVLink, socket links and external links are edges. Transfers follow a path; every edge on it is a
  shared resource. New hardware means new nodes and edges, no special cases in engine code.
- **Hybrid time model.** The fluid event simulation stays at the system level; between events, rates are constant.
  Those rates come from the engine walking its operations through the hardware model, per pipeline stage and per
  operation class (attention, DeltaNet, routing, hit experts, missed experts, MTP head, sampling).
- **All states computed up front.** When an engine is configured, every state it can reach is computed once:
  busy slots, reading or not, depth bucket, prompt chunk size. Each engine on its own; shared paths between engines
  (RAM bandwidth, chipset uplink, socket links) are resolved at run time with a fair-share step.
- **Engine profile.** A button on the engine ("Profile engine") computes the table and shows it as a summary of
  how this engine does by itself on this hardware:
  - pure writing: total output by busy slots, at a few depths;
  - pure reading: time to read a prompt by size, from a short follow-up to the full window;
  - mixed: what happens to the writers while a prompt is read, and for how long (shown per mode: piped groups stop
    writing, one group writes slowly between chunks), as a curve over the read/write ratio or as named mixes;
  - limits: per state, which resource sets the pace.
- **Engine models are per version.** Each modelled version brings its argument schema, the rules between its
  arguments, its operations and its scheduler rules. Versions without benchmark data are marked unvalidated.
- **Strata 0.1.43.1 first.** It is the only engine and version with enough benchmark data. vLLM, llama.cpp and
  other Strata versions follow when data exists.
- **Three stages for the user.** Machine, then engine with its profile, then (optional) workload. Most users stop
  at the engine profile: pure reading and pure writing for that engine on that hardware. Workload modelling, and
  Turnstone with it, is a later, optional layer.

## Open questions

- Granularity: per layer, per stage, or per operation class?
- Time model: keep the fluid event simulation and feed it operation costs, or simulate windows and chunks as events?
- What stays out of scope (already: sharing a card or cores between engines).
