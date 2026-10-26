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
