#!/usr/bin/env bash
# Stand-in for bench-svc.sh in tests: no Docker. Logs calls to $FAKE_SVC_LOG.
cmd="$1"; shift
echo "$cmd $*" >>"${FAKE_SVC_LOG:-/dev/null}"
case "$cmd" in
  base) case "$1" in
          vllm-sub) echo '{"command":["${SUB_MODEL}","--served-model-name=qwen3.8-27b","--max-num-seqs=4","--kv-cache-dtype=fp8"]}';;
          *) echo '{"command":["-m","/models/${JUDGE_GGUF}","-c","32768","-np","2"]}';;
        esac;;
  up) [[ "$2" != - ]] && grep -q BROKEN "$2" && { echo "crash loop"; echo "engine-1  | strata generate: the pack needs --spec T (T >= 2)"; exit 1; }; echo "ready after 0s"
      if [[ "$1" == strata ]]; then echo "strata-1  | [strata] filling the GPU's expert cache (5000 experts, 9.0 GiB of VRAM) ..."; fi;;
  logs) echo "engine-1  | SpecDecoding metrics: Mean acceptance length: 2.10"; [[ -n "${2:-}" ]] && echo "raw sample" >"$2";;
  restore) echo restored;;
  net) shift; [[ "${1:-}" == -- ]] && shift; exec python3 "$@";;
  gpu) echo "0, ${FAKE_TEMP:-50}, 200, 1000"; echo "1, 55, 210, 1000";;
esac
