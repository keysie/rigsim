# Overhaul of the engine model

Working document for the branch `overhaul-of-engine`. Points we want to establish before anything is built.
Status: sections marked *agreed* are decided (2026-10-26); everything else is under discussion.

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

## Decided so far (agreed)

- **Clean split between engines and hardware.** An engine never knows how fast its hardware is; it may read
  descriptors (see "Rates and efficiency factors") and states demands (compute of a kind on the device holding a
  stage, bytes moved between memory pools, memory to allocate). The hardware model alone answers durations and
  memory fit for everything running at the same moment.
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

## Hardware interface (agreed)

Four primitives; a part (card, CPU, board) is a bundle of them, the machine is the graph they form.

- **Storage:** capacity, access time, bandwidth. Nothing more. Access patterns are not simulated: the engine model
  states, from the engine's code, the access size it assumes for an operation; the hardware turns that into an
  effective rate (size / (access time + size / bandwidth)).
- **Compute:** clock as a function of power; operation rates per class are multiples of the clock (units x ops per
  clock per class). No microarchitecture beyond that.
- **Link:** point to point, simplex or duplex, latency, bandwidth. No multi-master buses. Sharing appears because
  several flows cross the same link (e.g. everything behind the chipset crosses its uplink).
- **Router:** joins links; forwarding capacity; optionally linked to a compute object with a cost per byte (copies
  that need CPU cycles, e.g. host bounce buffers) — those cycles compete with the engine's own CPU work.
- **In the UI, each storage in the terms users know:** RAM as transfer rate (MT/s), channels and timings
  (translated to bandwidth and access time in the background, with one fixed controller/fabric term per CPU
  family); SSDs directly as bandwidth and access time.
- **Power:** the power limit belongs to the device (card, CPU package). Each primitive in a power-limited device has
  its own curve: its scalable parameter vs the device's power limit (e.g. a 3090: GPU clock drops quickly and
  non-linearly with the limit, the VRAM clock not at all). CPUs the same (core clock vs package power). Curves for
  parts we have data for; users can upload their own as CSV.

## Engine interface (agreed)

- The engine sees roles (stage device, host memory, expert compute), never parts; a binding layer maps its arguments
  onto hardware objects.
- It states allocations (role storage, bytes, purpose), work per step (operations with their class, amount, access
  size and order: sequence, parallel, pipelined across stages) and its reachable states.
- The hardware returns, per step: duration, busy share per resource, the bottleneck, power drawn. Concurrent
  operations share resources by max-min fair share.

## Rates and efficiency factors (agreed)

- **Hardware answers per (runtime type and version, operation class):** the achievable rate, defined by the vendor's
  reference library (cuBLAS / cuBLASLt matrix products, a plain copy for bandwidth). Measured with microbenchmarks.
- **The engine version holds one dimensionless factor per operation class:** its kernels' speed relative to that
  reference. Kernels are not a limited set; they change with engine versions, so their quality belongs there.
- **The reference makes the two separable:** end-to-end runs of one engine only show the product of factor and rate.
- **A new engine version inherits the previous version's factors**, marked "inherited", until a benchmark confirms
  or replaces them. It declares the runtime versions it needs.
- **Unmeasured hardware:** inherit the per-clock rate from a measured card of the same architecture, scaled by clock
  and unit count; with no data for the architecture, fall back to the spec sheet with one global default
  efficiency, and mark the result "unmeasured".
- **Architecture-specific factors are allowed but not pre-emptive:** one factor per engine version and operation class
  by default; an engine version stores per-architecture factors where benchmarks show the need.
- **What the engine may know about the hardware:** read-only descriptors (architecture family, capabilities such as
  INT8 / FP8 tensor cores, memory sizes). It uses them to pick its kernel path and coefficients. It never sees rates
  or computes durations itself; the hardware model alone does that.

## Hardware yardsticks (for developing the hardware model)

Published measurements with a fully stated configuration, used as microbenchmarks for the hardware layer only:
memory bandwidth (STREAM-type), memory latency, PCIe / host-device copies, GEMM throughput per runtime. Each entry
records source, configuration (CPU, DIMMs per channel, speed, threads; card, driver, runtime) and what it tests.
Collected in `docs/overhaul/yardsticks.md` (first set: Intel MLC on Skylake, Cascade Lake and Ice Lake Xeon nodes). The reference box's own microbenchmarks rank first where available.

## Open questions

- What stays out of scope (already: sharing a card or cores between engines; CPU caches, NUMA within a socket,
  instruction-level CPU detail; thermals beyond the power limit).
- Which microbenchmarks to run on the reference box (memory latency/bandwidth, host-device copies per slot, cuBLAS
  GEMM per precision, GPU clock vs power limit under compute-heavy and memory-heavy load).
