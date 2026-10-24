# Benchmarking tools

The scripts that produced the measured points on the simulator's Calibration page. Standard-library Python,
no installation. Two layers:

| Layer | What it is | Needs |
|---|---|---|
| **standalone** (this folder) | measure one engine behind an OpenAI-compatible URL, measure memory bandwidth, inspect a host, turn results into the simulator's "measured values" | Python 3.11+, the engine running |
| **queue** (`queue/`) | run a whole plan of variants × workloads unattended with Docker Compose restarts, resume, reports, charts | Docker Compose and a site-specific wrapper — kept as reference for how the calibration passes were run; adapt before use |

## Measure one engine

```bash
export LOCAL_API_KEY=...                                # if the engine wants a key
# writing speed alone: a few generation tasks, one at a time
python3 benchw.py tasks --url http://127.0.0.1:8080/v1 --model <name> --item solo --tasks tasks/seed.jsonl \
  --out solo.jsonl --streams 1 --repeats 3
python3 to-measured.py solo.jsonl                       # -> {"decode_1": ..., "prefill": ...}
# with 4 requests at once
python3 benchw.py tasks --url ... --item par4 --tasks tasks/seed.jsonl --out par4.jsonl --streams 4
python3 to-measured.py par4.jsonl --streams 4           # -> {"decode_4": ...}
```

Paste the JSON into the engine's *Measured on real hardware* block in the simulator (or into a saved config file
under `engines[].benchmarks`). Other workloads: `multiturn` (history reuse), `agentloop` (a Turnstone-shaped
tool loop), `longtask`, `guard`, `judge`, `rerank` — see `python3 benchw.py -h` and the docstring at the top.

`benchreport.py results.jsonl` summarises a run: pass rate, truncations, reasoning tokens, think/total time,
tok/s, context high-water.

## Memory bandwidth of a CPU pool

```bash
python3 membw-calibrate.py            # STREAM-style read and copy over the whole pool, cached per hardware fingerprint
gcc -O3 -march=native -fopenmp membw_probe.c -o membw_probe && ./membw_probe 4096
```

## Inspect a host

`boxstats.py` samples GPU power, clocks, PCIe link state and CPU load while a run goes on; `inspect-engines.py`
lists running engines, their arguments and model files (read-only).

## Contributing a measurement

Open an issue or pull request with: the `to-measured.py` output, the engine command line, model and quant,
context length, number of streams, GPU power limit, driver, CPU, RAM channels and speed, OS, and three runs'
medians. Points with that information go on the Calibration page; the model constants are only refitted against
the CAL set, everything else stays held out.
