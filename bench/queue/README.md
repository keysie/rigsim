# Bench queue — unattended benchmarking

Runs a whole benchmark plan (models × engine settings × effort × sampling × workloads) one item after another,
without anyone starting the next run. Resumable, restores the production config at the end, tells you when it
is done or stuck. Stdlib Python + the existing `lib.sh`; nothing new to install.

## Pieces

| File | Role |
|---|---|
| `benchq.py` | queue: plan → items, state, restarts, retries, watchdog, notifications (host) |
| `bench-svc.sh` | the only part that touches Docker: recreate a service with an override, wait for `/models`, restore, run a tool in the hub's network |
| `benchw.py` | workloads against one engine (inside the network namespace): `tasks`, `multiturn` |
| `../scripts/bench.py` | workload `throughput` (unchanged, same as gate 07) |
| `benchreport.py` | `report.md`: pass rate, truncations, reasoning tokens, think/total time, tok/s, context high-water |
| `plans/*.toml` | what to run; `tasks/*.jsonl` what to ask and how to check the answer |

## A plan

```toml
[[variant]]                       # one engine configuration; the engine restarts once per variant
id = "sub-swift-awq"
service = "vllm-sub"              # strata | vllm-sub | vllm-maker-b | llama-judge | llama-guard
model = "ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AWQ"   # vLLM: first positional; llama.cpp: -m
args_set = { "--max-num-seqs" = "3" }              # replace or add a flag (--f=x and --f x both handled)
args_add = ["--enable-prefix-caching"]
args_remove = ["--kv-cache-dtype"]
env = { SOME_VAR = "1" }
# Strata: strata_args_set / strata_args_add / strata_args_remove patch strata-maker.json (mounted as a copy)

[[workload]]
id = "effort"
kind = "tasks"                    # tasks | multiturn | throughput
tasks = "tasks/seed.jsonl"
variants = ["sub-*"]              # globs
effort = ["medium", "template:low", "prefix:low"]   # list-valued knobs multiply into items
sampling = ["engine", "card"]     # profiles from [sampling.<name>]; "engine" = send nothing
replay_reasoning = [false, true]  # multiturn only
```

Effort modes: `medium` adds nothing; `template:<lvl>` sends `chat_template_kwargs {<effort.key>: lvl}`;
`prefix:<lvl>` puts `[effort.prefix] <lvl>` (the template's own sentence, verbatim) in front of the first user
message — how the maker would set a sub-maker's effort through the brief. Placeholders (`FILL…`) are rejected.

The engine command is taken from `docker compose config --no-interpolate`, so override files keep `${VAR}`
references and never contain keys.

## Running it

This queue ran on the reference machine from a configuration-management play that stopped the agent harness,
ran the stages as a transient systemd unit, fetched the results and restored every engine afterwards. That
wrapper is site-specific and not included; the pieces below are, and run anywhere:

By hand:

```bash
cd <your checkout>
benchq.py plan plans/stage1-sub-maker.toml [--smoke]   # check the queue, changes nothing
run-stages.sh [--smoke] [--fresh] plans/*.toml         # what the play starts as ai-bench
journalctl -u ai-bench -f                                                       # follow
benchq.py status plans/stage1-sub-maker.toml            # progress, failures
systemctl stop ai-bench                                                         # clean stop: baseline restored, resumable
benchq.py run … --retry-failed | --only 'sub-swift*'               # resume; subsets
benchq.py report …                                                  # tables (also written after each run)
```

Effort modes: `default` = nothing sent (the template's own default — **xhigh** on Qwen3.8, not medium);
`template:<lvl>` = `chat_template_kwargs` (what Turnstone sends, low in production); `prefix:<lvl>` = the template's
sentence in front of the first user message. A bare `medium` is rejected as ambiguous. Sampling: the plans send
nothing (`[sampling.engine]`), like Turnstone — the engines are pinned from `infra/models`.
The runner shares the network namespace of the reverse proxy (on every engine's edge; `VIA=<service>` for another hub).

Behaviour:
- Items run grouped by variant; the service restarts cold per variant (`recreate = "always"`).
- A variant that does not come up (crash loop, timeout) → its items are `skipped`, the queue goes on.
- A failed item is retried later (`max_attempts`); an interrupted item starts over, its partial rows are ignored.
- Watchdog per item: timeout, GPU core temperature above `max_gpu_temp_c` (kill, cool down, retry), free disk.
- Notifications: `BENCH_NOTIFY_URL` = any push-monitor URL; up on progress, down on failure. Set the
  monitor's heartbeat interval above the longest item, so a hung run also alarms. `status.json` is there for the
  dashboard collector.
- One run per plan at a time (lock file). Other services keep running; drain Turnstone traffic first, it skews
  the numbers.

State: `infra/scripts/logs/bench/<plan>/` (git-ignored): `state.json`, `results.jsonl`, `throughput.csv`,
`events.log`, `items.log`, `overrides/`, `report.md`.

## Tests

`tests/run-tests.sh` → 52 cases (fake service script, mock reasoning server): plan validation, smoke mode, run-stages,
override contents, effort modes, resume, reset, variant failure, hot GPU, SIGTERM, multiturn replay, Strata patch.
