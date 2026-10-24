#!/usr/bin/env bash
# Self-test of the bench tooling: fake service script (no Docker), mock reasoning server, real queue logic.
# Usage: bench/queue/tests/run-tests.sh      (needs bash, python3 >= 3.11)
set -u
B="$(cd "$(dirname "$0")/.." && pwd)"; T="$(mktemp -d)"; trap 'kill $MOCK 2>/dev/null; rm -rf "$T"' EXIT
python3 "$B/tests/mock_reason.py" 18989 & MOCK=$!
python3 "$B/tests/mock_openai.py" 18988 2>/dev/null & MOCK2=$!   # optional (throughput), copied from scripts/tests
sleep 1
export BENCH_SVC="$B/tests/fake-svc.sh" BENCH_LOCAL=1 BENCH_URL_OVERRIDE="http://127.0.0.1:18989/v1" \
       LOCAL_API_KEY=testkey STRATA_API_KEY=testkey FAKE_SVC_LOG="$T/svc.log"
n=0; bad=0
t() { # t "name" expected_rc "grep-pattern" -- command...
  local name="$1" want="$2" pat="$3"; shift 4; n=$((n+1))
  local out rc; out="$("$@" 2>&1)"; rc=$?
  if [[ "$rc" == "$want" ]] && grep -qE "$pat" <<<"$out"; then printf "  ok   %s\n" "$name"
  else printf "  FAIL %s (rc=%s want %s, pattern '%s')\n%s\n" "$name" "$rc" "$want" "$pat" "$(tail -8 <<<"$out")"; bad=$((bad+1)); fi
}
plan() { # plan NAME EXTRA_TOML -> path
  cat >"$T/$1.toml" <<EOF
name = "$1"
state_dir = "$T/state-$1"
[defaults]
repeats = 2
ready_timeout_s = 5
[guard]
max_gpu_temp_c = 80
cooldown_s = 0
min_free_gb = 0
[effort]
key = "reasoning_effort"
[effort.prefix]
low = "THINK LESS."
[sampling.card]
temperature = 1.0
top_k = 20
[[variant]]
id = "sub-a"
service = "vllm-sub"
[[variant]]
id = "sub-b"
service = "vllm-sub"
model = "org/Other-Model-AWQ"
args_set = { "--max-num-seqs" = "3" }
args_add = ["--enable-prefix-caching"]
[[workload]]
id = "effort"
kind = "tasks"
tasks = "tasks/seed.jsonl"
effort = ["default", "template:low", "prefix:low"]
sampling = ["card"]
streams = 2
max_tokens = 4000
$2
EOF
  echo "$T/$1.toml"
}
plan0() { # plan0 NAME EXTRA_TOML -> path: header only, the variants and workloads all from EXTRA
  printf 'name = "%s"\nstate_dir = "%s"\n[defaults]\nrepeats = 1\nready_timeout_s = 5\n[guard]\nmax_gpu_temp_c = 80\ncooldown_s = 0\nmin_free_gb = 0\n[effort]\nkey = "reasoning_effort"\n%s\n' \
    "$1" "$T/state-$1" "$2" >"$T/$1.toml"
  echo "$T/$1.toml"
}
Q="$B/benchq.py"
P1="$(plan basic '')"
echo "plan:"
t "inspect-engines unit"   0 "OK"                                -- python3 -m unittest discover -s "$B/tests" -p "test_inspect_engines.py"
t "plan expands"            0 "^6 items, 2 variant restarts"      -- python3 "$Q" plan "$P1"
t "plan item ids"           0 "sub-b/effort/effort=prefix:low/sampling=card" -- python3 "$Q" plan "$P1"
P2="$(plan bad '[[workload]]
id = "x"
kind = "tasks"
tasks = "tasks/nope.jsonl"
effort = ["prefix:high", "turbo"]
sampling = ["fast"]
variants = ["zzz*"]')"
t "plan errors"             2 "tasks file .*not found"            -- python3 "$Q" plan "$P2"
t "plan errors: prefix"     2 "prefix:high needs"                 -- python3 "$Q" plan "$P2"
t "plan errors: sampling"   2 "sampling profile 'fast'"           -- python3 "$Q" plan "$P2"
t "plan errors: no match"   2 "matches no variant"                -- python3 "$Q" plan "$P2"
P2m="$(plan badm '[[workload]]
id = "x"
kind = "tasks"
tasks = "tasks/seed.jsonl"
effort = ["medium"]')"
t "plan errors: medium"     2 "effort 'medium' is ambiguous"      -- python3 "$Q" plan "$P2m"
t "smoke plan: first knobs" 0 "^2 items, 2 variant restarts"      -- python3 "$Q" plan "$P1" --smoke
t "missing model skipped"   0 "not in the HF cache"               -- env -u BENCH_LOCAL MODELS_DIR="$T/nomodels" python3 -c \
  'import sys; sys.path.insert(0, sys.argv[1]); import benchq; print(benchq.model_missing({"service": "vllm-sub", "model": "org/m"}))' "$B"
# ---- 2026-10-20: stop/restore of other services, full command/image/compose overrides, depth (fill), slots streams,
# GPU-scoped temperature guard, smoke picks the flagged variants
P5="$(plan0 excl '[[variant]]
id = "s3"
service = "strata"
stop = ["vllm-sub"]
gpus = [0, 1, 2]
smoke = true
strata_config_set = { gpu = [0, 1, 2], layer_split = "auto" }
compose = { deploy = { resources = { reservations = { devices = [{ driver = "nvidia", device_ids = ["0", "1", "2"], capabilities = ["gpu"] }] } } } }
compose_replace = ["deploy"]
ctx = 65536
slots = 2
[[variant]]
id = "llsub"
service = "vllm-sub"
image = "ghcr.io/ggml-org/llama.cpp:${LLAMACPP_CUDA_TAG}"
command = ["-m", "/models/sub/x.gguf", "--port", "8000", "-c", "65536", "-np", "2"]
compose = { volumes = ["${MODELS_DIR}/gguf:/models:ro"] }
ctx = 32768
slots = 2
[[variant]]
id = "tp2"
service = "vllm-sub"
stop = ["strata", "llama-guard"]
args_set = { "--max-model-len" = "65536" }
ctx = 65536
slots = 1
[[workload]]
id = "depth"
kind = "throughput"
fill = [0.25, 0.9]
streams = "slots+2"
max_tokens = 64')"
t "plan: fill expands"      0 "^6 items, 3 variant restarts"      -- python3 "$Q" plan "$P5"
t "smoke: flagged only"     0 "^1 items, 1 variant restarts"      -- python3 "$Q" plan "$P5" --smoke
: >"$T/svc.log"
t "run: stop/restore"       0 "finished: \{'done': 6\}"           -- python3 "$Q" run "$P5"
t "order: stop before up"   0 "stop vllm-sub.up strata"           -- bash -c "tr '\n' '.' <'$T/svc.log' | grep -o 'stop vllm-sub.up strata'"
t "order: own svc, then back" 0 "restore strata.restore vllm-sub" -- bash -c "tr '\n' '.' <'$T/svc.log' | grep -o 'restore strata.restore vllm-sub'"
t "order: guard before strata" 0 "restore llama-guard.restore strata" -- bash -c "tr '\n' '.' <'$T/svc.log' | grep -o 'restore llama-guard.restore strata' | tail -1"
t "override: !override deploy" 0 '"deploy": !override'            -- cat "$T/state-excl/overrides/s3.yaml"
t "override: image + command"  0 "8000.*llama.cpp"              -- bash -c "tr -d '\n ' <'$T/state-excl/overrides/llsub.yaml'"
t "override: volumes appended" 0 "gguf:/models:ro"                -- cat "$T/state-excl/overrides/llsub.yaml"
t "depth: 3 streams at 90%"    0 "streams 3 .*prompt 58"          -- python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import benchq
v = {"slots": 1, "ctx": 65536}; w = {"streams": "slots+2", "max_tokens": 64}
print("streams", benchq.streams_for(w, v), "prompt", int(0.9 * 65536) - 64 - 512)
print("ramp", benchq.streams_for({"streams": "ramp+1"}, {"slots": 2}))' "$B"
t "streams list"            0 "^2,4 1,2,4$"                       -- python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import benchq
print(benchq.streams_for({"streams": "slots+0,slots+2"}, {"slots": 2}), benchq.streams_for({"streams": "ramp+0,slots+2"}, {"slots": 2}))' "$B"
t "streams ramp"            0 "ramp 1,2,3"                        -- python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import benchq; print("ramp", benchq.streams_for({"streams": "ramp+1"}, {"slots": 2}))' "$B"
t "gpu guard scoped"        0 "^ok$"                              -- env FAKE_TEMP=95 python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import benchq
p = {"guard": {"max_gpu_temp_c": 83}}
assert benchq.gpu_hot(p, {"gpus": [0]}).startswith("GPU 0"); assert benchq.gpu_hot(p, {"gpus": [1]}) == ""; print("ok")' "$B"
P7="$(plan0 onend '[[variant]]
id = "slot0"
service = "strata"
image = "ghcr.io/ggml-org/llama.cpp:${LLAMACPP_CUDA_TAG}"
command = ["-m", "/models/sub/x.gguf", "--port", "8080"]
ctx = 16384
slots = 1
[[workload]]
id = "ramp"
kind = "throughput"
streams = "1"
prompt_tokens = 256
max_tokens = 16')"
sed -i 's/^\[defaults\]$/[defaults]\non_end = "stop"/' "$P7"
: >"$T/svc.log"
t "on_end stop: slot left empty" 0 "^stop strata$"                -- bash -c "BENCH_URL_OVERRIDE=http://127.0.0.1:18988/v1 python3 '$Q' run '$P7' >/dev/null; grep -E '^(stop|restore) strata' '$T/svc.log'"
t "on_end: no restore"      1 "^$"                               -- grep -E "^restore strata" "$T/svc.log"
: >"$T/svc.log"
t "run-stages --stop"       0 "stop strata.stop llama-guard.up strata" -- env BENCH_LOGS="$T/rs4" bash -c '"$0/run-stages.sh" --smoke --stop strata,llama-guard "$1" >/dev/null; tr "\n" "." <"$2" | grep -o "stop strata.stop llama-guard.up strata"' "$B" "$P7" "$T/svc.log"
P8="$(plan0 loop2 "[sampling.engine]
[[variant]]
id = \"m\"
service = \"vllm-sub\"
ctx = 32768
slots = 2
[[workload]]
id = \"agents\"
kind = \"agentloop\"
shape = [\"direct\", \"turnstone-sub\"]
agents = \"slots+0\"
turns = 4
tool_tokens = 300
max_tokens = 200
corpus = \"$B\"")"
t "agentloop: 2 shapes"     0 "finished: \{'done': 2\}"           -- python3 "$Q" run "$P8"
t "agentloop: rows"         0 "^16$"                              -- bash -c "wc -l <'$T/state-loop2/results.jsonl' | tr -d ' '"
t "agentloop: tools sent"   0 "n_tool_calls\": 1"                 -- grep -m1 'shape=turnstone-sub' "$T/state-loop2/results.jsonl"
t "agentloop: report"       0 "Agent loops"                       -- python3 "$Q" report "$P8"
t "agentloop: smoke"        0 "^1 items|^2 items"                 -- python3 "$Q" plan "$P8" --smoke
P10="$(plan0 long1 "[sampling.engine]
[[variant]]
id = \"m\"
service = \"vllm-sub\"
ctx = 2048
slots = 2
[[workload]]
id = \"long\"
kind = \"longtask\"
tasks = \"tasks/longtask-cat.json\"
max_turns = 30
max_tokens = 300
until_fill = 0.3")"
t "longtask: runs"          0 "finished: \{'done': 1\}"           -- python3 "$Q" run "$P10"
t "longtask: 2 conversations, stops at until_fill" 0 "^2 ok$"     -- python3 -c '
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(len({r["conv"] for r in rows}), "ok" if max(r["turn"] for r in rows) < 29 else "no")' "$T/state-long1/results.jsonl"
P9="$(plan0 lad "[sampling.engine]
[ladder.l]
stop_membw_pct = 90
stop_tps_below = 1e12
consecutive = 2
[[variant]]
id = \"base\"
service = \"strata\"
ctx = 32768
slots = 1
ladder = \"l\"
[[variant]]
id = \"c80\"
service = \"strata\"
ctx = 32768
slots = 1
ladder = \"l\"
expert_cache_frac = 0.8
ladder_base = \"base\"
[[variant]]
id = \"c50\"
service = \"strata\"
ctx = 32768
slots = 1
ladder = \"l\"
expert_cache_frac = 0.5
ladder_base = \"base\"
[[variant]]
id = \"last\"
service = \"strata\"
ctx = 32768
slots = 1
expert_cache_from = \"l\"
[[workload]]
id = \"deep\"
kind = \"longtask\"
tasks = \"tasks/longtask-cat.json\"
corpus = \"$B\"
fill = 0.05
max_turns = 2
max_tokens = 2000
ladder_eval = true")"
t "ladder: stops after 2 slow" 0 "'done': 2, 'skipped': 2|'skipped': 2, 'done': 2" -- bash -c "BENCH_STRATA_CONFIG='$B/../strata/strata-maker.json' python3 '$Q' run '$P9' >/dev/null; python3 '$Q' status '$P9'"
t "ladder: reason recorded"  0 "ladder l stopped: writing speed below"  -- python3 "$Q" status "$P9"
t "ladder: cache from base"  0 '"--expert-cache", *"4000"'          -- bash -c "tr -d '\\n' <'$T/state-lad/overrides/c80.strata.json' | grep -o '\"--expert-cache\", *\"4000\"'"
t "longtask: turns written"  0 "^4$"                              -- bash -c "grep -c '\"conv\"' '$T/state-lad/results.jsonl'"
t "longtask: report"         0 "Long tasks"                       -- python3 "$Q" report "$P9"
echo "run:"
t "run completes"           0 "finished: \{'done': 6\}"           -- python3 "$Q" run "$P1"
t "smoke run, own state"   0 "finished: \{'done': 2\}"           -- python3 "$Q" run "$P1" --smoke
t "smoke: 1 task x 1 repeat" 0 "^2$"                              -- bash -c 'wc -l < "$0/results.jsonl"' "$T/state-basic-smoke"
t "run-stages: all stages"  0 "basic.toml rc=0 basic-smoke: \{'done': 2\}" -- env BENCH_LOGS="$T/rs" bash -c '"$0/run-stages.sh" --smoke "$1" >/dev/null; cat "$2/last-run.txt"' "$B" "$P1" "$T/rs"
t "run-stages: groups"      0 "basic.toml rc=0.*excl.toml rc=0"  -- env BENCH_LOGS="$T/rs2" bash -c '"$0/run-stages.sh" --smoke "$1" --then "$2" >/dev/null; tr "\n" " " <"$3/last-run.txt"' "$B" "$P1" "$P5" "$T/rs2"
t "run-stages: parallel"    0 "rc=0.*rc=0"                         -- env BENCH_LOGS="$T/rs3" bash -c '"$0/run-stages.sh" --smoke "$1" "$2" >/dev/null; tr "\n" " " <"$3/last-run.txt"' "$B" "$P1" "$P5" "$T/rs3"
S="$T/state-basic"
t "results rows"            0 "^120$"                              -- bash -c "wc -l <'$S/results.jsonl' | tr -d ' '"
t "low via template"        0 "\"reasoning_tokens\": 20"          -- grep -m1 "effort=template:low" "$S/results.jsonl"
t "low via prefix"          0 "\"reasoning_tokens\": 25"          -- grep -m1 "effort=prefix:low" "$S/results.jsonl"
t "default untouched"       0 "\"reasoning_tokens\": 60"          -- grep -m1 "effort=default" "$S/results.jsonl"
t "override: model + args"  0 "Other-Model-AWQ.*--max-num-seqs=3.*--enable-prefix-caching" -- bash -c "tr -d '\n ' <'$S/overrides/sub-b.yaml'"
t "override: no secrets"    1 "^$"                               -- grep -r testkey "$S/overrides"
t "baseline restored"       0 "restore vllm-sub"                  -- cat "$T/svc.log"
t "report tables"           0 "think tok p50"                     -- python3 "$Q" report "$P1"
t "report passes counted"   0 "sub-a/effort/effort=default/sampling=card · low \| 10 \| 0 \| 6/10" -- cat "$S/report.md"
t "resume: nothing left"    0 "0 items pending of 6"              -- python3 "$Q" run "$P1"
t "reset"                   0 "1 items reset"                     -- python3 "$Q" reset "$P1" "sub-a/*template*"
echo "failures:"
P3="$(plan broken '[[variant]]
id = "sub-broken"
service = "vllm-sub"
args_add = ["BROKEN"]
[[variant]]
id = "sub-fallback"
service = "vllm-sub"
only_if_failed = "sub-broken"
[[variant]]
id = "sub-fb2"
service = "vllm-sub"
only_if_failed = "sub-a"')"
t "variant down -> reason"  0 "skipped.*sub-broken.*did not start: .*needs --spec" -- bash -c "python3 '$Q' run '$P3' >/dev/null; python3 '$Q' status '$P3'"
t "fallback runs, other not" 0 "'done': 9, 'skipped': 6|'skipped': 6, 'done': 9" -- python3 "$Q" status "$P3"
t "fallback not needed"     0 "sub-fb2.*not needed: sub-a started" -- python3 "$Q" status "$P3"
t "logs after the workloads" 0 "SpecDecoding"                     -- bash -c "grep -A1 'after the workloads' '$T/state-broken/up-sub-a.log'"
t "raw log sample kept"     0 "raw sample"                        -- cat "$T/state-broken/log-sub-a.txt"
P4="$(plan hot '')"
t "hot gpu -> failed"       1 "FAILED .*hot: GPU 0 at 95"         -- env FAKE_TEMP=95 python3 "$Q" run "$P4" --only "sub-a/effort/effort=default*"
t "retry-failed"            0 "finished: \{'done': 1"             -- python3 "$Q" run "$P4" --retry-failed --only "sub-a/effort/effort=default*"
P5="$(plan loop '[[workload]]
id = "loop"
kind = "multiturn"
tasks = "tasks/seed.jsonl"
variants = ["sub-a"]
replay_reasoning = [false, true]')"
t "multiturn runs"          0 "finished"                          -- python3 "$Q" run "$P5" --only "*loop*"
t "replay grows context"    0 "replay_reasoning=True"             -- bash -c "python3 '$Q' report '$P5' | grep -A3 'context high-water'"
python3 - "$T/state-loop/results.jsonl" >"$T/hw" <<'PY'
import json, sys
hw = {}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if "turn" in r: hw[r["item"]] = max(hw.get(r["item"], 0), r["prompt_tokens"])
print("replay larger" if hw["sub-a/loop/effort=default/sampling=engine/replay_reasoning=True"] >
      hw["sub-a/loop/effort=default/sampling=engine/replay_reasoning=False"] else "replay NOT larger")
PY
t "replay > no replay"      0 "replay larger"                     -- cat "$T/hw"
P6="$(plan strata '[[variant]]
id = "maker-b2"
service = "strata"
strata_args_set = { "--batch" = "2" }
[[workload]]
id = "q"
kind = "tasks"
tasks = "tasks/seed.jsonl"
variants = ["maker-b2"]
only_level = "low"')"
t "strata override"         0 "\"--batch\",\"2\""                 -- bash -c "python3 '$Q' run '$P6' --only 'maker-b2/*' >/dev/null; tr -d '\n ' <'$T/state-strata/overrides/maker-b2.strata.json'"
t "strata mount"            0 "/data/config/strata-maker.json:ro" -- cat "$T/state-strata/overrides/maker-b2.yaml"
echo "pause:"
P11="$(plan pz '')"; export BENCH_PAUSE_FILE="$T/PAUSE"
python3 "$Q" pause >/dev/null
python3 "$Q" run "$P11" --only "sub-a/effort/effort=default*" >"$T/pz.out" 2>&1 & R=$!; sleep 6
t "pause: nothing starts"   1 "^0$"                               -- grep -c "^.\{20\}done " "$T/state-pz/events.log"
t "pause: status"           0 "PAUSED before sub-a"               -- cat "$T/state-pz/status.json"
t "pause: progress line"    0 "state-pz: 0 done of 6.*PAUSED"     -- python3 "$B/benchprogress.py" "$T" --since 1 "$T/state-pz"
python3 "$Q" resume >/dev/null; wait $R
t "resume: continues"       0 "resumed after 0 min"               -- cat "$T/pz.out"
t "resume: item done"       0 "finished: \{'done': 1"            -- cat "$T/pz.out"
t "progress: counts"        0 "state-pz: 1 done of 6 \| last:"   -- python3 "$B/benchprogress.py" "$T" --since 1 "$T/state-pz"
unset BENCH_PAUSE_FILE
echo "pass 3:"
P13="$(plan0 p3t "[sampling.engine]
[[variant]]
id = \"one\"
service = \"strata\"
ctx = 8192
slots = 1
gpus = [0]
gpu_power_w = 300
[[workload]]
id = \"nothink\"
kind = \"longtask\"
tasks = \"tasks/longtask-cat.json\"
corpus = \"$B\"
fill = 0.02
max_turns = 3
max_tokens = 300
trace = true
[[workload]]
id = \"think\"
kind = \"longtask\"
tasks = \"tasks/longtask-cat.json\"
corpus = \"$B\"
fill = 0.02
max_turns = 3
max_tokens = 300
think_back = true
trace = true
[[workload]]
id = \"mixed\"
kind = \"longtask\"
tasks = \"tasks/longtask-cat.json\"
corpus = \"$B\"
fills = [0.02, 0.05]
stagger_s = 1
conversations = 2
max_turns = 2
max_tokens = 300
think_back = true")"
t "p3: runs"                0 "finished: \{'done': 3\}"           -- bash -c "BENCH_STRATA_CONFIG='$B/../strata/strata-maker.json' python3 '$Q' run '$P13'"
t "p3: rows have times"     0 "^ok$"                              -- python3 -c '
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print("ok" if all(r["t_start"] <= (r["t_first"] or r["t_end"]) <= r["t_end"] for r in rows) else "no")' "$T/state-p3t/results.jsonl"
t "p3: thinking back reuses the answer" 0 "^ok$"                 -- python3 -c '
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
def gain(w):     # reused tokens of turn k beyond the whole prompt of turn k-1 (same conversation)
    rs = sorted((r for r in rows if w in r["item"]), key=lambda r: r["turn"])
    return [b["cached_tokens"] - a["prompt_tokens"] for a, b in zip(rs, rs[1:])]
think, nothink = gain("/think/"), gain("/nothink/")
# without the thinking the reuse stops at the previous prompt (+ the turn header); with it, the answer is reused too
print("ok" if max(abs(x) for x in nothink) <= 6 and min(think) > 10 else f"no think {think} nothink {nothink}")' "$T/state-p3t/results.jsonl"
t "p3: mixed fills, staggered" 0 "^ok$"                          -- python3 -c '
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if "/mixed/" in l]
f = {r["conv"]: r["fill"] for r in rows}; s = {r["conv"]: min(x["t_start"] for x in rows if x["conv"] == r["conv"]) for r in rows}
print("ok" if f == {0: 0.02, 1: 0.05} and s[1] - s[0] >= 0.9 else f"no {f} {s}")' "$T/state-p3t/results.jsonl"
t "p3: engine record matched" 0 "engine_decode_tps"              -- head -1 "$T/state-p3t/results.jsonl"
t "p3: trace written"       0 "\"slots\": \[\{\"slot\": 0"         -- head -1 "$T/state-p3t/trace.jsonl"
t "p3: power set, restored" 0 "power 0 300.*power restore"        -- bash -c "tr '\\n' ' ' <'$T/svc.log'"
t "p3: report reuse"        0 "History reuse on follow-up.*Slot trace" -- bash -c "python3 '$Q' report '$P13' | tr '\\n' ' '"
t "p3: plan valid"          0 "21 items"                          -- python3 "$Q" plan "$B/plans/p3-strata1.toml"
echo "pass 4:"
P14="$(plan0 p4t "[sampling.engine]
[ladder.w]
stop_gain_below_pct = 3
consecutive = 2
[[variant]]
id = \"w1\"
service = \"strata\"
ctx = 32768
slots = 1
ladder = \"w\"
strata_args_set = {\"--pool-workers\" = \"1\"}
[[variant]]
id = \"w2\"
service = \"strata\"
ctx = 32768
slots = 1
ladder = \"w\"
strata_args_set = {\"--pool-workers\" = \"2\"}
[[variant]]
id = \"w3\"
service = \"strata\"
ctx = 32768
slots = 1
ladder = \"w\"
strata_args_set = {\"--pool-workers\" = \"3\"}
[[variant]]
id = \"w4\"
service = \"strata\"
ctx = 32768
slots = 1
ladder = \"w\"
strata_args_set = {\"--pool-workers\" = \"4\"}
[[variant]]
id = \"gd\"
service = \"llama-guard\"
[[variant]]
id = \"jd\"
service = \"llama-judge\"
[[variant]]
id = \"rr\"
service = \"llama-rerank\"
[[workload]]
id = \"deep\"
kind = \"longtask\"
tasks = \"tasks/longtask-cat.json\"
corpus = \"$B\"
fill = 0.02
max_turns = 1
max_tokens = 300
ladder_eval = true
variants = [\"w*\"]
[[workload]]
id = \"g\"
kind = \"guard\"
variants = [\"gd\"]
corpus = \"$B\"
sizes = [256, 1024]
conc = [1, 2]
repeats = 1
[[workload]]
id = \"j\"
kind = \"judge\"
variants = [\"jd\"]
corpus = \"$B\"
history = [512]
tool_turns = [0, 2]
conc = [1]
repeats = 1
read_tokens = 200
max_tokens = 2048
chat_kwargs = {enable_thinking = false}
[[workload]]
id = \"r\"
kind = \"rerank\"
variants = [\"rr\"]
corpus = \"$B\"
shapes = [\"mem:50:40:30\", \"web:30:100:20\"]
conc = [1, 4]
repeats = 1
[[workload]]
id = \"rs\"
kind = \"rerank\"
variants = [\"rr\"]
corpus = \"$B\"
shapes = [\"sweep:64:40:40\"]
conc = [2]
duration_s = 2")"
t "p4: runs, ladder stops on no gain" 0 "'done': 7, 'skipped': 1|'skipped': 1, 'done': 7" -- bash -c "BENCH_STRATA_CONFIG='$B/../strata/strata-maker.json' python3 '$Q' run '$P14' >/dev/null; python3 '$Q' status '$P14'"
t "p4: ladder reason"       0 "ladder w stopped: writing gained less than 3 %"  -- python3 "$Q" status "$P14"
t "p4: pool workers set"    0 '"--pool-workers", *"3"'                         -- bash -c "tr -d '\\n' <'$T/state-p4t/overrides/w3.strata.json' | grep -o '\"--pool-workers\", *\"3\"'"
t "p4: rows per kind"       0 "^guard 6 6 judge 2 2 rerank 10 [0-9]+$"         -- python3 -c '
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
c = collections.Counter(r.get("kind") for r in rows if r.get("kind"))
okj = collections.Counter(r.get("kind") for r in rows if r.get("kind") and r.get("ok") and r.get("json_ok", True))
rr = [r for r in rows if r.get("kind") == "rerank"]
print("guard", c["guard"], okj["guard"], "judge", c["judge"], okj["judge"], "rerank", sum(1 for r in rr if not r["closed_loop_s"]), sum(1 for r in rr if r["closed_loop_s"]))' "$T/state-p4t/results.jsonl"
t "p4: judge turns"         0 "^\[1, 3\]$"                                  -- python3 -c '
import json, sys
print(sorted(r["turns_done"] for r in map(json.loads, open(sys.argv[1])) if r.get("kind") == "judge"))' "$T/state-p4t/results.jsonl"
t "p4: report envelopes"    0 "Small models.*guard.*judge.*rerank"          -- bash -c "python3 '$Q' report '$P14' | tr '\\n' ' '"
t "p4: plans valid"         0 "20 items.*38 items"                          -- bash -c "(python3 '$Q' plan '$B/plans/p4-cpu.toml' | tail -1; python3 '$Q' plan '$B/plans/p4-small.toml' | tail -1) | tr '\\n' ' '"
t "p4: no engine fails the item" 0 "^1$"                                      -- bash -c "python3 '$B/benchw.py' guard --url http://127.0.0.1:9/v1 --model m --item x --out '$T/x.jsonl' --sizes 10 --timeout 2 >/dev/null 2>&1; echo \$?" 
echo "membw cache:"
MC="$T/mcache"; HW="$T/hw.json"; mkdir -p "$MC"
printf '{"board":"PRIME X299-A","bios":"4401","cpu":"i9-10900X","logical_cpus":20,"physical_cores":10,"ram_gib":125,"dimms":[{"locator":"A1","size":"32 GB","speed":"3200 MT/s","part":"X"}]}' >"$HW"
cp "$HW" "$MC/hardware.json"; printf 'kernel,threads\ncopy,10\n' >"$MC/membw-calibrate.csv"
printf '{"peak_total_gbs": 70.4, "peak_read_gbs": 64.2, "measured_at": 1792600000}' >"$MC/membw-peak.json"
t "membw: same hw -> reuse" 0 "^reuse: calibration of .*70.4 GB/s.*hardware unchanged" -- env MEMBW_HW_JSON="$HW" python3 "$B/membw-calibrate.py" --check "$MC"
t "membw: no cache -> measure" 3 "^measure: no previous calibration"  -- env MEMBW_HW_JSON="$HW" python3 "$B/membw-calibrate.py" --check "$T/nocache"
sed 's/"4401"/"4501"/' "$HW" >"$T/hw2.json"
t "membw: bios changed -> measure" 3 "measure: hardware changed.*bios '4401' -> '4501'" -- env MEMBW_HW_JSON="$T/hw2.json" python3 "$B/membw-calibrate.py" --check "$MC"
sed 's/3200 MT\/s/2666 MT\/s/' "$HW" >"$T/hw3.json"
t "membw: dimm speed changed"  3 "hardware changed.*dimms"          -- env MEMBW_HW_JSON="$T/hw3.json" python3 "$B/membw-calibrate.py" --check "$MC"
t "membw: reuse copies"     0 "reused_at"                          -- bash -c "MEMBW_HW_JSON='$HW' python3 '$B/membw-calibrate.py' --reuse '$MC' --out '$T/mout' >/dev/null && cat '$T/mout/membw-peak.json' && test -s '$T/mout/membw-hardware.json'"
t "run-stages: --membw-reuse" 0 "reused: calibration"              -- bash -c "MEMBW_HW_JSON='$HW' BENCH_LOGS='$T/rsl' BENCH_MEMBW_CACHE='$MC' bash '$B/run-stages.sh' --membw-reuse --stop strata >/dev/null 2>&1; cat '$T/rsl/last-run.txt'; test -s '$T/rsl/membw-peak.json'"
t "run-stages: changed hw measures" 0 "measure: hardware changed.*membw calibration FAILED" -- bash -c "MEMBW_HW_JSON='$T/hw2.json' AI_MEMBW_FILE=/nonexistent BENCH_LOGS='$T/rsl2' BENCH_MEMBW_CACHE='$MC' timeout 60 bash '$B/run-stages.sh' --membw-reuse --stop strata >/dev/null 2>&1; tr '\\n' ' ' <'$T/rsl2/last-run.txt'"
echo "skip:"
P12="$(plan skp '')"
t "skip: marked by user"    0 "'skipped': 3"                      -- bash -c "python3 '$Q' run '$P12' --skip 'sub-b/*' --only 'sub-a/effort/effort=default*' >/dev/null; python3 '$Q' status '$P12'"
t "skip: reason"            0 "skipped by user \\(bench_skip sub-b/\\*\\)" -- python3 "$Q" status "$P12"
echo "interrupt:"
P7="$(plan sig '')"; sed -i 's/^repeats = 2$/repeats = 40/' "$P7"   # long enough to be interrupted
python3 "$Q" run "$P7" --only "sub-a/effort/effort=default*" >"$T/sig.out" 2>&1 & R=$!; sleep 1.5; kill -TERM $R; wait $R; echo "rc=$?" >>"$T/sig.out"
t "sigterm stops cleanly"   0 "rc=130"                            -- cat "$T/sig.out"
t "sigterm restores"        0 "stopped: \{'pending': 6\}"       -- cat "$T/sig.out"
t "resume after stop"       0 "finished: \{'done': 1, 'pending': 5\}" -- python3 "$Q" run "$P7" --only "sub-a/effort/effort=default*"
t "partial rows dropped"    0 "sub-a/effort/effort=default/sampling=card · low \| 200 \|" -- python3 "$Q" report "$P7"
echo; ((bad)) && { echo "✗ $bad of $n failed"; exit 1; } || echo "✓ $n of $n passed"
