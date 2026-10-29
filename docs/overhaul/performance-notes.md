# Simulation cost — notes for later

Decision (2026-10-29): stay the course, accept longer simulation times. Profiling and optimisation start once the
predictions hold up across a spread of benchmarked machines — not before.

## Measured (0.11.0 run: i7-13700K, RTX 3080, UHD 770; branch one-machine-graph)

| What | Time | Steps | Graph solves |
|---|---|---|---|
| Build the machine graph | 140 ms | – | (24 nodes, 23 links) |
| llama-bench pp512, Llama 2 7B | 12 ms | 483 | 1 |
| llama-bench tg128, Llama 2 7B | 1.15 s | 61 824 | 128 |
| All 17 predictions of the run | ~7 s | | |

One engine step costs ~20 µs (Python); a graph solve ~0.3 ms, cached per forward pass while one transfer runs alone.
Steps with truly concurrent activities are solved again whenever one finishes: 0.29 ms (2 transfers) .. 0.8 ms (9).
Naively, a multi-GPU tg128 would be ~60 000 steps x ~0.5 ms ≈ 30 s per predicted value.

## Possible improvements (none changes the model)

1. **Cache rates by step structure.** A step's rates depend on which activities run on which parts, not on their
   amounts: key the solved rates (and each progressive-filling stage) by that structure. A multi-GPU layer step
   repeats with the same structure for every layer of every token.
2. **Fewer simulated tokens in tgN.** Tokens differ only in KV-cache depth, and their time is linear in it within a
   kernel path: simulate a few depths and interpolate exactly instead of every token.
3. Afterwards, if still needed: profile the step loop (Python overhead dominates single-device runs).
