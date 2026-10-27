# Overhaul of the engine model — proposal

Consolidated 2026-10-26 on branch `overhaul-of-engine`. Everything below is agreed unless listed under "Open".
Yardsticks for the hardware layer: `docs/overhaul/yardsticks.md`.

## 1. Goal

The simulator models what the engines and the system actually do, not curves fitted to the data we have. Speeds,
waits and limits emerge from hardware resources, engine operations and scheduler rules.

## 2. Principles

1. **Mechanism over fit.** Every quantity is counted (bytes, layers, tokens), read from a spec sheet, or measured as
   a primitive. No constant exists only to make an end-to-end number come out right.
2. **Calibrate at the bottom, validate at the top.** Primitives come from microbenchmarks; end-to-end benchmarks
   validate only.
3. **Every constant names something measurable:** what it is, its unit, how to measure it, where the value came from.
4. **Modes are explicit.** The engine's operating mode (cards, slots, peer-device or layer split, pipeline groups,
   version) decides which rules apply, and it is shown to the user.
5. **Few parameters, honestly identified.** If the data cannot tell two parameters apart, the model has one.

## 3. Architecture

- **Three stages for the user:** machine, then engine with its profile, then (optional) workload. Most users stop at
  the engine profile. Workload modelling, and Turnstone with it, is a later, optional layer.
- **Clean split between engines and hardware.** The engine states demands; the hardware model alone answers
  durations, memory fit and contention (§5, §6).
- **Hybrid time model.** The fluid event simulation stays at the system level; between events, rates are constant.
  The rates come from the engine walking its operations through the hardware model, per pipeline stage and per
  operation class (attention, DeltaNet, routing, hit experts, missed experts, MTP head, sampling).
- **All states computed up front.** When an engine is configured, every state it can reach is computed once: busy
  slots, reading or not, depth bucket, prompt chunk size. Each engine on its own; shared paths between engines (RAM
  bandwidth, chipset uplink, socket links) are resolved at run time by a fair-share step.
- **Engine profile.** A button on the engine ("Profile engine") computes the table and summarises how the engine
  does by itself on this hardware: pure writing (total output by busy slots, at a few depths); pure reading (time to
  read a prompt by size, from a short follow-up to the full window); mixed (what happens to the writers while a
  prompt is read, and for how long, per mode; as a curve over the read/write ratio or as named mixes); limits (per
  state, which resource sets the pace).

## 4. Engines and versions

- **Engine models are per version:** each brings its argument schema, the rules between its arguments, its
  operations and its scheduler rules. Versions without benchmark data are marked unvalidated.
- **Strata 0.1.43.1 first**, the only engine and version with enough benchmark data. vLLM, llama.cpp and other
  Strata versions follow when data exists.
- The mode tree for Strata (one card / several; batch 1 / several slots; peer-device / layer split; one pipeline
  group / several) decides how reading and writing meet: one request at a time, read in chunks with the slots
  writing between them, or read in one run with all writing stopped (several pipeline groups only).

## 5. Hardware model

Four primitives; a part (card, CPU package, board) is a bundle of them; the machine is the graph they form.

- **Storage:** capacity, access time, bandwidth. Access patterns are not simulated: the engine states the access
  size it assumes for an operation; the hardware turns that into an effective rate
  (size / (access time + size / bandwidth)). Access time is fixed (no latency-vs-load curve).
- **Compute:** clock as a function of power; operation rates per class are multiples of the clock (units x ops per
  clock per class).
- **Link:** point to point, simplex or duplex, latency, bandwidth. No multi-master buses; sharing appears because
  several flows cross the same link. A DRAM channel is simplex (DDR's data lines serve both directions).
- **Router:** joins links; forwarding capacity; optionally linked to a compute object with a cost per byte (copies
  that need CPU cycles compete with the engine's CPU work).
- **Transfers** follow a path through the graph; every link, router and storage on it is a shared resource.
  Concurrent operations share resources by max-min fair share.
- **CPU package:** any number of core groups (type and count, e.g. P- and E-cores), each group one object with a
  count. Each core has its own link to the package router; the router links to the RAM channels. A core type has two
  primitives: compute rate per operation class, and the bandwidth one core sustains (its link). An operation takes
  the larger of the two times; which one binds emerges. Engine workers map onto cores. Hyperthreads (later, if
  ever): two computes sharing one link.
- **Power** belongs to the device (card, CPU package). Each primitive in a power-limited device has its own curve:
  its scalable parameter vs the device's power limit (a 3090: GPU clock falls quickly and non-linearly, VRAM clock
  not at all). Curves for parts with data; users can upload their own as CSV.
- **In the UI, each storage in the terms users know:** RAM as transfer rate, channels and timings (translated to
  bandwidth and access time, with one fixed controller/fabric term per CPU family); SSDs as bandwidth and access time.

## 6. Engine interface

- The engine sees roles (stage device, host memory, expert compute), never parts; a binding layer maps its arguments
  onto hardware objects.
- It may read descriptors (architecture family, capabilities such as INT8 / FP8 tensor cores, memory sizes) to pick
  its kernel path and coefficients. It never sees rates and never computes durations.
- It states allocations (role storage, bytes, purpose), work per step (operations with class, amount, access size
  and order: sequence, parallel, pipelined across stages) and its reachable states.
- The hardware returns per step: duration, busy share per resource, the bottleneck, power drawn.

## 7. Rates and efficiency factors

- **Hardware answers per (runtime type and version, operation class)** the achievable rate, defined by the vendor's
  reference library (cuBLAS / cuBLASLt matrix products, a plain copy for bandwidth), measured with microbenchmarks.
- **The engine version holds one dimensionless factor per operation class:** its kernels' speed relative to that
  reference. The reference makes factor and rate separable.
- A new engine version inherits the previous version's factors ("inherited") until a benchmark confirms or replaces
  them; it declares the runtime versions it needs.
- Unmeasured hardware: inherit the per-clock rate of a measured card of the same architecture, scaled by clock and
  units; with no data for the architecture, spec sheet with one global default efficiency, marked "unmeasured".
- Per-architecture factors only where benchmarks show the need.

## 8. Out of scope (for now)

Sharing a card or cores between engines; CPU caches, NUMA within a socket, instruction-level CPU detail; RAM
read/write turnaround (inference is almost all reads); thermals beyond the power limit; hyperthreads.

## 9. Open

- RAM bandwidth vs transfer rate: constant efficiency holds at 5000–6000 MT/s but overshoots below (yardstick 2,
  up to +16.5 %). Cause unknown (timings, controller mode or software); looking for data with published timings
  before any model change.
- Which microbenchmarks define each primitive (core-type compute per operation class, single-core bandwidth, GPU
  clock vs power limit under compute-heavy and memory-heavy load, host-device copies, reference GEMM per precision).
- Strata at small expert caches: prompt reading was PCIe-bound (239 tok/s at every worker count, pass 3); the CPU
  expert path in writing is limited per core (compute or per-core bandwidth, not yet separated).
