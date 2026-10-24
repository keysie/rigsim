#!/usr/bin/env bash
# Service control for benchq.py — the only part that touches Docker. Reuses lib.sh (dc, py_net conventions).
#   bench-svc.sh base SVC                    resolved service definition as JSON, WITHOUT interpolation (no secrets)
#   bench-svc.sh up SVC OVERRIDE|- TIMEOUT   recreate SVC (+ override file), wait for /models; 1 on crash loop/timeout
#   bench-svc.sh restore SVC                 recreate SVC from the plain compose files (baseline)
#   bench-svc.sh net STATEDIR -- TOOL ARGS   run a stdlib tool in the hub's network namespace (/b bench, /s scripts,
#                                            /out STATEDIR); hub = $VIA (default caddy, like py_net)
#   bench-svc.sh stop SVC                    stop a service another variant needs the GPU of (restore brings it back)
#   bench-svc.sh gpu                         one sample per GPU: index, temperature, power, memory used
#   bench-svc.sh logs SVC [OUT]              the engine's log lines from under load: spec acceptance, preemption, errors;
#                                            OUT: the raw last 400 lines too (IPs masked), for the board's patterns, and
#                                            OUT.full.gz the whole log since the variant's start (pass 3)
#   bench-svc.sh power GPU WATTS | restore   a GPU's power limit for one variant; restore = the declared limits again
#                                            (gpu-power-limit.service, infra/host)
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../scripts/lib.sh"; load_env
BENCH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cmd="${1:?subcommand}"; shift

ready() { # ready SVC TIMEOUT -> 0 when /models answers; 1 on crash loop, exit or timeout
  local svc="$1" to="$2" t0=$SECONDS id rc url kv
  url="$(svc_url "$svc")"; kv="$(svc_keyvar "$svc")"
  while ((SECONDS - t0 < to)); do
    py_net probe.py models --url "$url" --key-env "$kv" >/dev/null 2>&1 && { echo "ready after $((SECONDS - t0))s"; return 0; }
    id="$(dc ps -q "$svc" 2>/dev/null | head -1)"
    [[ -z "$id" || "$(docker inspect -f '{{.State.Status}}' "$id" 2>/dev/null)" != running ]] && { echo "not running"; return 1; }
    rc="$(docker inspect -f '{{.RestartCount}}' "$id" 2>/dev/null)"; ((${rc:-0} > 0)) && { echo "crash loop"; return 1; }
    sleep 10
  done
  echo "timeout after ${to}s"; return 1
}

case "$cmd" in
  base)
    dc config --no-interpolate --format json | jq -c --arg s "$1" '.services[$s]';;
  up)
    svc="$1"; ov="$2"; to="${3:-1800}"
    extra=(); [[ "$ov" != - ]] && extra=(-f "$ov")
    dc "${extra[@]}" up -d --no-deps --force-recreate "$svc" || exit 1
    ready "$svc" "$to"; r=$?
    # what the engine made of the settings (bench-plan: start-log summary per variant) — vLLM capacity, llama.cpp
    # context/slots/VRAM, Strata expert slots, VRAM free, resident KV, layer split, MTP
    dc logs --no-color --tail 400 "$svc" 2>&1 | grep -iE "KV cache|maximum concurrency|coverage|batch_slots|n_ctx|n_slots|slot|expert|VRAM|resident|layer split|mtp|spec|draft|out of memory|error" | tail -40
    exit $r;;
  stop)
    dc stop "$1";;
  restore)
    dc up -d --no-deps --force-recreate "$1" && ready "$1" 1800;;
  net)
    out="$1"; shift; [[ "${1:-}" == -- ]] && shift
    hub="$(dc ps -q "${VIA:-caddy}" 2>/dev/null | head -1)"
    [[ -n "$hub" ]] || { echo "bench-svc: ${VIA:-caddy} is not running" >&2; exit 2; }
    tool="$1"; shift
    exec docker run --rm --network "container:$hub" -e LOCAL_API_KEY -e STRATA_API_KEY \
      -v "$BENCH:/b:ro" -v "$INFRA/scripts:/s:ro" -v "$INFRA/../docs:/d:ro" -v "$out:/out" "$PY_IMAGE" python "$tool" "$@";;
  logs)
    # filtered summary to stdout; with OUT also the raw tail under load (IPs masked) — the exact line formats the
    # board's log patterns (collector log_metrics) are written against
    dc logs --no-color --tail 4000 "$1" 2>&1 | grep -iE "SpecDecoding|acceptance|accepted|draft|preempt|swap|out of memory|error|exceed" | tail -40
    if [[ -n "${2:-}" ]]; then
      dc logs --no-color --timestamps --tail 400 "$1" 2>&1 | sed -E 's/([0-9]{1,3}\.){3}[0-9]{1,3}/<ip>/g' >"$2"
      # the whole log of this container (recreated per variant = this variant only): slot choice, interleaving
      dc logs --no-color --timestamps "$1" 2>&1 | sed -E 's/([0-9]{1,3}\.){3}[0-9]{1,3}/<ip>/g' | gzip -c >"$2.full.gz"
    fi
    true;;
  power)
    if [[ "$1" == restore ]]; then systemctl restart gpu-power-limit.service
    else nvidia-smi -i "$1" -pl "$2"; fi;;
  gpu)
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader,nounits;;
  *) echo "unknown subcommand $cmd" >&2; exit 2;;
esac
