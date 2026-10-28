# Hardware bench — low-level measurements for the hardware model

Measurements aimed at specific questions in the hardware layer (`backend/app/hw/`, proposal
`docs/overhaul/engine.md`), not end-to-end engine runs: a multi-card Strata run has too much runtime and engine
between it and the hardware to identify a primitive. Each question here isolates one: a clock against a power
limit, a library's matrix-product rate, a VRAM or PCIe bandwidth, a core ladder, a latency.

The methodology follows the AI-SERVER benchmark queue (unattended from the Ansible VM, vault once, plans validated
first, a resumable queue on the box as a transient systemd unit, smoke runs first and short, ALWAYS restore, the
run removes what it created, raw rows plus host-side samples, archives masked and checked clean) with three
differences: **questions instead of numbered passes**, **only containers from common images** (no images built:
one session container per run, tools installed and compiled inside it, items run with `docker exec`, the container
discarded at the end), and **low-level probes** (cuBLAS, streaming kernels, pinned-memory copies, likwid-bench, a
pointer chase, fio).

**A run takes the box over.** Every container is stopped (except `keep_running`), and the run refuses to start if
any process is still on a GPU. Engines and agents on the box are unavailable while it runs.

## Questions

| Question | Asks | Feeds |
|---|---|---|
| `gpu-clock-power` | SM / VRAM clock vs power limit, compute- vs memory-heavy, at busy shares 1.0 … 0.25 | power curves per load class; tests the pass-6 miss (busy share) |
| `gpu-gemm` | cuBLAS share of SMs x measured clock x per-SM rate, per precision and shape | `rates.MEASUREMENTS` (the kernel factors' reference) |
| `gpu-vram` | VRAM read / write / copy bandwidth vs power limit | VRAM storage bandwidth and its curve |
| `pcie-copy` | host <-> card per slot, alone and all at once (`@together`), both directions, card -> card | link capacities, shared links, routers |
| `cpu-mem-ladder` | read bandwidth by cores, instruction width and core clock | per-core link, RAM efficiency; compute vs bandwidth per core |
| `cpu-mem-latency` | idle RAM latency by working-set size and core clock | RAM access time (CAS + fixed term) |
| `nvme-read` | sequential bandwidth and random access time of the models disk | SSD storage primitive |

Each file in `questions/` states its question, what it feeds, its method, the grid and a smoke subset (at most four
items). Power limits are watts or `default` / `max` per card; points outside a card's range (or a CPU's frequency
range) are skipped with the reason.

## Setup (once, on the Ansible VM)

1. `cp bench/hw/config.example.yml bench/hw/config.local.yml` (git-ignored) and set: the AI-SERVER Ansible
   directory (its inventory is used as is), the vault file, the target alias, `gpu_names` (PCI bus id -> card
   name; only names reach the results), and how the target reaches common images (below).
2. Vault password for the session, kept in RAM: `read -rs VP && printf '%s' "$VP" > /dev/shm/vp && chmod 600
   /dev/shm/vp && unset VP; export ANSIBLE_VAULT_PASSWORD_FILE=/dev/shm/vp` (then `rm /dev/shm/vp` when done).

**Common images on a box without internet access.** The GPU questions use `nvidia/cuda:<version>-devel-ubuntu24.04`
(compiler and cuBLAS included: the setup needs no network beyond the pull). The CPU questions use `debian:trixie`
and install likwid and fio with apt during setup. A box that reaches only an internal GitLab needs a pull-through
proxy for images (`image_mirror`, e.g. GitLab's dependency proxy) and, for the CPU toolkit, an HTTP proxy for the
setup step (`setup_proxy`). The run fails early with a message naming the setting if a pull does not work. The
session container is cut off from the network after setup, before the first item.

## Toolkits

`toolkits/<name>/` holds the sources and `setup.sh`, copied into the session container and run there once:

| Toolkit | Image | Setup | Probes |
|---|---|---|---|
| `cuda` | `nvidia/cuda:12.8.1-devel-ubuntu24.04` | `nvcc` builds `probe` (sm_80/86/89/90 + PTX) | `probe gemm`, `stream`, `load`, `copy` |
| `cpu` | `debian:trixie` | apt: likwid, fio, gcc; builds `chase` | `cpuprobe ladder`, `latency`, `fio` |

Each item runs as `docker exec` in the session; GPU questions select the item's card with
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` (`@together` = every card visible at once). A hung
probe is killed by restarting the session (its setup is kept). The image's digest is recorded in `session.json`.

## Running

```bash
bench/hw/run.sh list
bench/hw/run.sh gpu-gemm --check          # validates the plan and the inventory, changes nothing
bench/hw/run.sh gpu-gemm --smoke          # one short item per mechanism
bench/hw/run.sh gpu-clock-power --smoke-first   # smoke, then the real run if the smoke was clean
bench/hw/run.sh gpu-clock-power --fresh   # forget an earlier run of this question (default: resume)
```

On the box, by hand: `touch /var/lib/rigsim-hwbench/<question>/PAUSE` pauses between items (resume: remove it and
start again — done items are kept); `systemctl stop rigsim-hwbench` stops cleanly (restores first).

## What a run does to the box — and undoes

- **Before the first item** (`baseline.json`, written once per question directory, so a resumed run keeps the
  original state): GPU power limits, CPU frequency limits and governors, Docker containers / images / volumes,
  the running containers.
- **The box is taken over:** every running container is stopped except `keep_running` (each recorded before the
  next), and the run stops if any process is still on a GPU.
- **One session container** from the question's common image (labelled, `no-new-privileges`), toolkit copied in
  and set up, then disconnected from every network.
- **Per item:** the item's card at the item's limit, every other card at its baseline; the CPU clock pinned only for
  the item; the probe via `docker exec`; nvidia-smi samples every 100 ms on the host; temperature watchdog (cool
  down, retry), timeout, free-disk check.
- **At the end, on SIGTERM and on any error:** the session container is discarded, the box restored from
  `baseline.json` and verified; the play restores again (idempotent), removes what the run created (labelled
  containers, images pulled for the run, new dangling volumes, the test-file directory), fetches everything, and
  **fails if the box is not as found**.

## Results

Fetched to `results_dir/<question>[-smoke]-<time>/` on the Ansible VM (private). `runner/archive.py` writes the
masked copy `…-public/` and `…-public.tar.gz` (IPs, host names incl. the registry, home paths, GPU UUIDs, MACs and
PCI ids replaced; card names from `gpu_names`) and refuses to pack it if anything is left. `convert/to_rigsim.py
<masked dir>` turns a run into rigsim data (trust: high — own controlled runs with repeats; every point keeps its
spread). Raw rows first: summaries are derived, never the other way round.

## Tests

`python3 -m pytest -q bench/hw/tests`: plans valid and smokes short; a full run against a simulated host (one
session: set up, cut off, items on the right card, discarded; power limits per item; containers stopped and
started; the host as found); `@together`; setup failure, pull failure and leftover GPU processes stop the run and
restore; resume; baseline kept across an interruption; skips outside a card's range; retries; CPU clock pinned and
restored; cleanup removes only what the run created; masking; conversion; the play parses (and runs in `--check`
against no host). **Not tested here:** the probes and setup scripts themselves (no GPU, no Docker in this
environment) — the first smoke run of each question is their test.
