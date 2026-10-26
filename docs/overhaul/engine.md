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

## Open questions

- Granularity: per layer, per stage, or per operation class?
- Time model: keep the fluid event simulation and feed it operation costs, or simulate windows and chunks as events?
- Which engines first: Strata only, or vLLM and llama.cpp in the same pass?
- What stays out of scope (already: sharing a card or cores between engines).
