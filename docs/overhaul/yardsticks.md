# Hardware yardsticks

Published measurements with a stated configuration, for developing and testing the hardware layer only (not
engines). Rules: the configuration must be stated by the source; anything inferred is marked *inferred*; ratios
are computed here against the theoretical peak (channels x MT/s x 8 bytes).

## Memory: Intel MLC on dual-socket Xeon nodes (VSC Leuven, HPC centre documentation)

Same tool and version (MLC v3.10, prefetchers not modified, random access for latency), three generations.
Sources: docs.vscentrum.be, "Memory bandwidth and latency measurements" (Genius, wICE) and the hardware pages.

| | Skylake | Cascade Lake | Ice Lake |
|---|---|---|---|
| CPUs | 2 x Xeon Gold 6140 (18 cores) | 2 x Xeon Gold 6240 (18 cores) | 2 x Xeon Platinum 8360Y (36 cores) |
| RAM (stated) | 192 GiB | 192 GiB | 256 GiB |
| Channels x speed (*inferred*: 1 DIMM per channel, CPU max speed) | 12 x DDR4-2666 | 12 x DDR4-2933 | 16 x DDR4-3200 |
| Theoretical peak | 256.0 GB/s | 281.6 GB/s | 409.6 GB/s |
| Idle latency, local / remote socket | 90.2 / 144.3 ns | 80.8 / 140.4 ns | 89.2 / 142.7 ns |
| All reads (system) | 211.2 GB/s (82 %) | 225.6 GB/s (80 %) | 351.8 GB/s (86 %) |
| 3:1 / 2:1 / 1:1 reads:writes | 193.4 / 192.6 / 185.6 | 208.3 / 206.5 / 200.0 | 313.1 / 297.2 / 291.8 |
| Stream-triad-like | 172.8 GB/s (67 %) | 187.5 GB/s (67 %) | 320.2 GB/s (78 %) |
| One socket, local, reads | 104.4–106.0 GB/s | 112.8–113.1 GB/s | 175.7–175.9 GB/s |
| Across the socket link, reads | 34.3 GB/s | 34.3 GB/s | 55.7 GB/s |
| Loaded latency at saturation | ~166 ns | ~156 ns | ~264 ns |

To confirm before use: the DIMM speed and population (ask VSC or find the node spec sheet).

## Memory: single-core limit (qualitative)

Karl Rupp (karlrupp.net), STREAM on three Xeon generations: one core reaches only ~15–25 % of the socket's STREAM
bandwidth; a few cores saturate the channels. Needed as a yardstick for how many CPU workers it takes to saturate
RAM (Strata's expert path).

## Rejected or on hold

- OpenBenchmarking 2106273-FO: 2 x Xeon Platinum 8280, 18 x 64 GB DDR4-2933 (uneven: 18 DIMMs on 12 channels).
  All reads only 123.3 GB/s (44 % of peak), idle latency 102.4 ns. Anomalous for the configuration; not used until
  explained.
- GPU matrix-product rates and host-device copies for the RTX 3090: no reputable source with a full configuration
  found yet (forum measurements only). Candidates to read: Sun et al., "Dissecting Tensor Cores via
  Microbenchmarks" (IEEE TPDS 2023, PNNL); NVIDIA's GA102 architecture whitepaper for per-clock peaks. The
  reference box itself is the better source (cuBLAS matmul bench, nvbandwidth).

## What these already tell the hardware model

1. Achievable bandwidth is 80–86 % of peak for reads, less with writes mixed in (1:1 at 71–78 %). A DRAM channel
   behaves like a link shared by both directions with a turnaround cost, not like a duplex link.
2. Latency rises under load (81 → 156 ns on Cascade Lake). A constant access time is right when idle, too short
   near saturation.
3. One core cannot saturate memory: its bandwidth is limited by the requests it keeps in flight (Little's law:
   bytes in flight / latency). This couples compute (cores) to storage bandwidth.
4. The socket link is a separate, much narrower link (34–56 GB/s): a yardstick for the link primitive.

## Identifiability with the agreed model (2026-10-26)

Free parameters per CPU family: RAM efficiency, fixed controller/fabric latency, socket-link latency, socket-link
efficiency, per-core bandwidth, router forwarding capacity. Per node the MLC output gives 5 usable results (local
and remote idle latency, local and system read bandwidth, cross-socket bandwidth): 4 parameters fitted, one
validation (system = 2 x local). Not usable by design: read/write mixes (no turnaround cost in the model: it
predicts mix = all reads, measured 11–17 % lower), loaded latency (fixed access time), cache-to-cache latencies.

- Per-core bandwidth and router capacity are not identifiable here: every bandwidth test runs all cores.
- RAM efficiency is confounded with the inferred DIMM speed until the speed is confirmed.
- Sharing parameters across Skylake and Cascade Lake (same platform) predicts the same local latency; measured
  90.2 vs 80.8 ns. A fixed latency term per CPU family is too coarse, or something else differs.
- Total: 8 parameters against 15 usable of 47 published results.

## What a good yardstick source needs

1. The full configuration stated or auto-captured: CPU, DIMM speed, ranks and population, board, OS.
2. Primitive measurements, not composite scores: latency, bandwidth by core count, single-core bandwidth.
3. A core ladder (bandwidth at 1, 2, 4, ... cores): the only way to identify the per-core link.

Not suitable: composite scores (cpubenchmark.net / PassMark CPU Mark, Geekbench totals). They mix integer,
floating-point, compression and other work with undisclosed weights, average user submissions with unknown RAM,
power limits and cooling, and do not map onto any single primitive.
Candidates: OpenBenchmarking.org (Phoronix Test Suite captures the system configuration automatically; individual
tests such as STREAM, Intel MLC, tinymembench), Chips and Cheese (latency and bandwidth by core count, per core type).

## Iteration 1 results (branch `overhault-1st-try`, `backend/app/hw/yardsticks/vsc_mlc.py`)

Each memory channel is its own simplex link and storage; a socket's RAM is the interleaved set of its channels
(equal shares per channel). The socket-link capacity is a user input; unknown for these nodes, so the measured
cross-socket bandwidth stands in for it (and is therefore not a validation).

Fitted through the model (one measured value each, mean of the two sockets):

| | fixed latency term | socket-link latency | RAM efficiency |
|---|---|---|---|
| Skylake | 75.4 ns | 54.1 ns | 0.822 |
| Cascade Lake | 66.8 ns | 59.8 ns | 0.802 |
| Ice Lake | 75.3 ns | 53.7 ns | 0.858 |

Held out: system read bandwidth (both sockets at once) −0.4 % / +0.1 % / −0.0 %.

Observations (no model change):
- One fixed latency term per platform is too coarse (Skylake's term on Cascade Lake: +11 % local latency).
- Statistics: the source publishes one MLC run per node type; the only spread available is between the two sockets
  (≤ 1.6 %). The validation is consistency, not a statistical test.

## Yardstick 2: one Linux system at ten DDR5 speeds (iteration 2)

Source: Phoronix, "DDR5-6000 Memory Performance On Linux" (2022), result file openbenchmarking.org/result/
2202236-PTS-DDR5600092. Core i9-12900K (8 P + 8 E), ASUS ROG STRIX Z690-E, 2 x 16 GB DDR5, Ubuntu 22.04, STREAM at
3000–6000 MT/s, N = 5 runs each (standard error 0.0–0.7 %). Timings per speed not published; no latency test.
Code: `backend/app/hw/yardsticks/phoronix_ddr5.py`.

STREAM writes as well as reads, and counts bytes differently from the traffic on the RAM; that ratio is the same at
every speed, so the efficiency is fitted at one speed and the model's claim is tested: bandwidth = channels x
transfer rate x 8 bytes x a constant efficiency.

| MT/s | 3000 | 3200 | 3600 | 3800 | 4200 | 4600 | 5000 | 5400 | 5600 | 6000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Triad / peak | 0.652 | 0.668 | 0.711 | 0.729 | 0.682 | 0.688 | 0.767 | 0.763 | 0.761 | 0.760 |
| Model error (fitted at 6000) | +16.5 % | +13.7 % | +6.9 % | +4.3 % | +11.4 % | +10.4 % | −0.9 % | −0.4 % | 0.0 % | fit |
| Copy: model error | +9.5 % | +8.2 % | +3.2 % | +0.8 % | +9.4 % | +9.0 % | −1.0 % | −0.6 % | 0.0 % | fit |

- 5000–6000 MT/s: the model holds within ±1 %.
- Below 5000 the efficiency is not constant: it falls with speed, with an extra dip at 4200/4600. The deviations are
  far outside the measurement's spread. Unknown timings per speed are the obvious suspect, but the model has no
  timing beyond CAS in its access time, and access time does not enter bandwidth.

**Status (2026-10-27):** closed as a documented miss; model unchanged. RAMspeed Triad on the same runs stays within
±3 % of a constant efficiency, so the drop is STREAM-specific. Timing effects are listed as a possible avenue of
improvement in the proposal (§10). Earlier note: Next: look for scaling data with published
timings (option 3) to decide whether timings should affect bandwidth (option 2). Treat every explanation as
suspicious until data separates it: timings, BIOS / memory-controller mode, or software (compiler, OS, STREAM build).

## Yardstick 3 (candidate): what timings and uncore do (tangent of option 3)

Source: igor'sLAB, "Arrow Lake DDR5 in detail: memory clock, uncore, timings" (2026), Core Ultra 7 265K, Intel
MLC 3.12, one system, five configurations changing one variable at a time. Single runs, no spread published; DIMM
count inferred from the stated theoretical bandwidth. Code: `backend/app/hw/yardsticks/igorslab_arrowlake.py`.

| Configuration | All reads / peak | 1:1 R/W | Idle latency | Model error (efficiency from XMP 7800) |
|---|---|---|---|---|
| JEDEC 5600 CL46 | 0.848 | 61.15 GB/s | 106.7 ns | +11.3 % |
| XMP 6000 CL28 | 0.938 | 75.28 GB/s | 87.2 ns | +0.6 % |
| XMP 7800 CL36 | 0.944 | 92.75 GB/s | 88.6 ns | fit |
| 7800 + uncore OC | 0.947 | 92.59 GB/s | 77.0 ns | −0.4 % |
| 7800 + uncore OC + tight secondary timings | 0.980 | 102.42 GB/s | 78.1 ns | −3.8 % |

What it separates (each a single observation, not yet statistics):
1. **Rate with absolute timings kept** (6000 CL28 → 7800 CL36): constant efficiency holds (+0.6 %). Supports the model.
2. **Uncore clock alone:** bandwidth unchanged (+0.4 %), idle latency −11.6 ns. Supports the model's split (the
   controller/fabric term is latency only) and shows that term depends on the uncore clock, a configuration value.
3. **Secondary timings alone** (tRAS, tRFC, tREFI x 2.2, tWR, tWTR, tRTP, tRRD): reads +3.5 %, 1:1 +10.6 %. Timings do
   enter bandwidth, more for mixed traffic (write-to-read turnaround) than for reads. Refresh alone accounts for
   ~1.4 % of the read gain (tRFC / tREFI from 2.0 % to 0.7 % of the time).
4. **JEDEC profile** (loose timings, 1.1 V class): ~10 % below the XMP efficiency — the size of yardstick 2's drop.
   Plausible link to yardstick 2 (auto/JEDEC-like timings at its lower speeds), but voltages and controller mode
   also differ between JEDEC and XMP, and yardstick 2's timings are unknown. Not established.

Status: suggestive, not valid data yet (one system, single runs, no spread). Needed before a model change: the same
kind of one-variable-at-a-time study on other systems, with repeats.

## RAM benchmark database as validation points (2026-10-27)

`backend/app/hw/yardsticks/data/ram-bench-data.json` (schema v1, compiled in a separate research chat: 10 sources,
39 entries, every field marked stated / inferred / not stated). Used for validation only, never for fitting, at
**medium or low trust**: low if anecdotal, reader- or forum-submitted, rounded, partial or missing primaries; medium
otherwise. Nothing is high trust (1 of 39 entries has a spread, 2 run on Linux; values for the same hardware differ
between sources by up to 12 %), so comparisons stay within one source's sequence. Tolerances: medium ±5 %, low ±10 %.
Harness: `validate_ramdb.py`; tests: `tests/test_hw_ramdb.py`.

CPU topologies come from spec sheets and architecture documentation (`backend/app/hw/cpus.py`): AMD Zen 4/5 desktop
parts get one link per chiplet to the I/O die (32 B read per fabric clock); Arrow Lake and APUs are monolithic for
now (tile link not documented). Per group, one parameter is fitted on the first point: the chiplet-link efficiency
where that link binds, else the RAM efficiency. The AMD RAM efficiency is inherited within the architecture from the
single-DIMM point (tpu-04: 0.959).

| Group | Held-out points | Result |
|---|---|---|
| ocinside 7800X3D (5200–6400 MT/s, FCLK 1733–2167) | 6 | steps 2–4 within ±0.5 %; tREFI and later steps −8.5 % (refresh not modelled; known misses) |
| ocinside 9800X3D (5600–6400) | 2 | −1.4 %, −1.6 % |
| TechPowerUp 9800X3D (4800–8000, read flat) | 6 | 0.0 % (the chiplet link binds throughout) |
| Hardwareluxx 285K (5600–7200) | 3 | +0.9 %, +1.5 %, −4.7 % |
| igor'sLAB 265K (D2D / NGU clocks only) | 3 | −2.3 … −3.8 % (no tile link modelled) |
| FPS Review DDR4 (2667–3933) | 2 | +0.6 %, +1.9 % |

Finding: the chiplet-link efficiency for the same CPU and fabric clock differs between sources (0.877 ocinside vs
0.994 TechPowerUp): tool version or settings, not hardware.
