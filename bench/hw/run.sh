#!/usr/bin/env bash
# rigsim hardware bench, from the Ansible VM.
#   bench/hw/run.sh QUESTION [--smoke | --smoke-first] [--fresh] [--check]   QUESTION = a file stem in questions/
#   bench/hw/run.sh images [PROBE ...]                                          build + push probe images, write the lock
#   bench/hw/run.sh list
# Reads bench/hw/config.local.yml (copy of config.example.yml). Vault password: ANSIBLE_VAULT_PASSWORD_FILE.
set -euo pipefail
H="$(cd "$(dirname "$0")" && pwd)"
CFG="${RIGSIM_HWBENCH_CONFIG:-$H/config.local.yml}"
[[ "${1:-}" == list ]] && { for q in "$H"/questions/*.toml; do basename "$q" .toml; done; exit 0; }
[[ -r "$CFG" ]] || { echo "no $CFG — copy config.example.yml and adjust" >&2; exit 2; }
get() { python3 -c 'import os,sys,yaml; v=yaml.safe_load(open(sys.argv[1])).get(sys.argv[2]); print(os.path.expanduser(v) if isinstance(v,str) else (v if v is not None else ""))' "$CFG" "$1"; }
AI="$(get ai_server_ansible)"; VAULT="$(get vault_file)"
[[ -d "$AI/inventory" ]] || { echo "ai_server_ansible: no inventory under $AI" >&2; exit 2; }
inv=(-i "$AI/inventory/hosts.yml" -i "$AI/inventory/private")
extra=(-e "@$CFG" -e "hw_root=$H")
[[ -n "$VAULT" ]] && extra+=(-e "@$VAULT")
export ANSIBLE_CONFIG="$H/ansible/ansible.cfg"
if [[ "${1:-}" == images ]]; then
  shift; probes="$(printf '%s,' "${@:-cuda nvbandwidth cpu}")"
  exec ansible-playbook "${inv[@]}" "$H/ansible/images.yml" "${extra[@]}" -e "probes=${probes%,}"
fi
q="${1:?question (see: run.sh list)}"; shift
[[ -r "$H/questions/$q.toml" ]] || { echo "no question $q" >&2; exit 2; }
smoke=false; smoke_first=false; fresh=false; check=()
for a in "$@"; do case "$a" in
  --smoke) smoke=true;; --smoke-first) smoke_first=true;; --fresh) fresh=true;; --check) check=(--check);;
  *) echo "unknown option $a" >&2; exit 2;; esac; done
play() { ansible-playbook "${inv[@]}" "$H/ansible/hwbench.yml" "${extra[@]}" -e "question=$q" -e "smoke=$1" -e "fresh=$fresh" "${check[@]}"; }
if $smoke_first; then
  play true || { echo "smoke failed — the real run does not start" >&2; exit 1; }
  play false
else
  play "$smoke"
fi
