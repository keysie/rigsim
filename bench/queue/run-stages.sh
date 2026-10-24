#!/usr/bin/env bash
# Run benchmark plans in groups (ops/bench.yml starts this as the transient unit ai-bench, 2026-10-20).
#   run-stages.sh [--smoke] [--fresh] [--membw | --membw-reuse] PLAN.toml [...] [--then [--stop SVC,SVC] PLAN.toml ...] ...
# --membw (pass 2, 2026-10-21): first stop every engine named by the first group's --stop and run membw-calibrate.py
# (the box's real RAM-bandwidth peak → $OUT/membw-peak.json, exported as BENCH_MEMBW_PEAK for the ladders); a
# successful calibration is kept in $OUT/membw-cache (git-ignored). --membw-reuse (user 2026-10-22): take that cached
# calibration if the hardware fingerprint is unchanged (ops/bench.yml checks first), else measure as with --membw.
# Plans within a group run in parallel (they must use different GPUs: e.g. maker queue on #1+#2, sub-maker queue on
# #3); groups run one after another (an exclusive block — Strata on 3 cards, vLLM over #1+#2 — comes after `--then`).
# `--stop SVC,SVC` in a group: those services are stopped once before the group starts (2026-10-21: llama.cpp grid in
# Strata's and the guard's slots — no parallel plan may race the other for a GPU still held by production).
# Every group runs even if an earlier one had failures (benchq restores its services itself, also on SIGTERM).
# Summary: $BENCH_LOGS (default infra/scripts/logs/bench)/last-run.txt, one line per plan: plan, rc, counts.
set -u
B="$(cd "$(dirname "$0")" && pwd)"; OUT="${BENCH_LOGS:-$B/../scripts/logs/bench}"; mkdir -p "$OUT"
flags=(); membw=0
while [[ "${1:-}" == --smoke || "${1:-}" == --fresh || "${1:-}" == --membw || "${1:-}" == --membw-reuse || "${1:-}" == --membw-reuse-only || "${1:-}" == --skip ]]; do
  if [[ "$1" == --membw ]]; then membw=1
  elif [[ "$1" == --membw-reuse ]]; then membw=2
  elif [[ "$1" == --membw-reuse-only ]]; then membw=3     # smoke runs (user 2026-10-23: keep smokes short): never measure
  elif [[ "$1" == --skip ]]; then flags+=(--skip "$2"); shift      # bench_skip: item globs left out (benchq run)
  else flags+=("$1"); fi; shift
done
# pause between items (benchq.py pause|resume, ops/bench-control.yml); a new run starts unpaused
export BENCH_PAUSE_FILE="${BENCH_PAUSE_FILE:-$OUT/PAUSE}"; rm -f "$BENCH_PAUSE_FILE"
: >"$OUT/last-run.txt"
# box stats for the whole run (boxstats.py; benchreport averages them per item)
: >"$OUT/boxstats.csv"
python3 "$B/boxstats.py" --out "$OUT/boxstats.csv" --every "${BENCH_BOXSTATS_EVERY:-5}" & sampler=$!
export BENCH_BOXSTATS="$OUT/boxstats.csv"
children=()
trap 'echo "stopped by signal" >>"$OUT/last-run.txt"; kill -TERM "${children[@]}" 2>/dev/null; kill -TERM "$sampler" 2>/dev/null; wait; exit 143' TERM INT
worst=0
SVC="${BENCH_SVC:-$B/bench-svc.sh}"
stops=""
run_group() {
  local plans=("$@") pids=() i rc counts s
  ((${#plans[@]})) || return 0
  echo "=== $(date -Is) group: ${plans[*]##*/} ${flags[*]}${stops:+ (stopped first: $stops)}"
  for s in ${stops//,/ }; do "$SVC" stop "$s"; done
  stops=""
  for p in "${plans[@]}"; do
    python3 "$B/benchq.py" run "$p" "${flags[@]}" > >(sed -u "s|^|[$(basename "$p" .toml)] |") 2>&1 &
    pids+=($!)
  done
  children=("${pids[@]}")
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    counts=$(python3 "$B/benchq.py" status "${plans[$i]}" $(printf '%s\n' "${flags[@]}" | grep -x -- --smoke) 2>/dev/null | head -1)
    echo "$(basename "${plans[$i]}") rc=$rc $counts" | tee -a "$OUT/last-run.txt"
    ((rc > worst)) && worst=$rc
  done
  children=()
}
export BENCH_MEMBW_PEAK="$OUT/membw-peak.json"
rm -f "$OUT/membw-peak.json" "$OUT/membw-calibrate.csv" "$OUT/membw-hardware.json"   # never a previous run's files
MEMBW_CACHE="${BENCH_MEMBW_CACHE:-$OUT/membw-cache}"
if ((membw == 2 || membw == 3)); then
  # a calibration of the same hardware is reused (re-checked here: the hardware may have changed since the play looked)
  python3 "$B/membw-calibrate.py" --reuse "$MEMBW_CACHE" --out "$OUT" 2>&1 | tee -a "$OUT/last-run.txt"
  if ((PIPESTATUS[0] == 0)); then membw=0
  elif ((membw == 3)); then membw=0; echo "smoke: no calibration (none to reuse) — runs without the RAM peak" | tee -a "$OUT/last-run.txt"
  else membw=1; fi                                            # the reason is in last-run.txt; measure instead
fi
if ((membw)); then
  # the engines of the first group's --stop go first, so the calibration runs on an idle box
  first_stop="$(awk 'BEGIN{for(i=1;i<ARGC;i++) if(ARGV[i]=="--stop"){print ARGV[i+1]; exit}}' "$@")"
  SVC="${BENCH_SVC:-$B/bench-svc.sh}"
  for s in ${first_stop//,/ }; do "$SVC" stop "$s"; done
  echo "=== $(date -Is) memory bandwidth calibration (stopped first: ${first_stop:-none})"
  python3 "$B/membw-calibrate.py" --out "$OUT" --cache "$MEMBW_CACHE" 2>&1 | tee -a "$OUT/last-run.txt"
  if ((PIPESTATUS[0] != 0)); then
    echo "membw calibration FAILED — ladders run without the bandwidth limit" | tee -a "$OUT/last-run.txt"; worst=1
  fi
fi
group=(); want_stop=0
for a in "$@"; do
  if ((want_stop)); then stops="$a"; want_stop=0
  elif [[ "$a" == --then ]]; then run_group "${group[@]}"; group=()
  elif [[ "$a" == --stop ]]; then want_stop=1
  else group+=("$a"); fi
done
run_group "${group[@]}"
kill -TERM "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
echo "=== $(date -Is) all groups done (worst rc $worst)"
exit "$worst"
