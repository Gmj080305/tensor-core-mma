#!/usr/bin/env bash
# Self-test of the GPU characterisation pipeline without a GPU:
# generate probes, produce a fake "GPU" dump from a planted, non-default model
# configuration, and check that `tcmma fit` recovers exactly that configuration.
# This proves the probe families can tell the hypotheses apart.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
CLI=model/build/tcmma
[ -x "$CLI" ] || { echo "build the model first (make model)"; exit 1; }

python3 cuda/probes/gen_probes.py --out "$TMP" --random-tiles 20 >/dev/null
status=0
check() {  # fmt, planted settings...
  local fmt="$1"; shift
  "$CLI" eval --in "$TMP/$fmt.bin" --out "$TMP/$fmt.fake.out" "$@" >/dev/null
  local best
  best="$("$CLI" fit --in "$TMP/$fmt.bin" --gpu "$TMP/$fmt.fake.out" --top 1 | awk '/^ +1 /{print}')"
  local ok=1
  [[ "$best" =~ ^\ +1\ +0\  ]] || ok=0
  for kv in "$@"; do [[ "$best" == *"$kv"* ]] || ok=0; done
  if [ $ok -eq 1 ]; then echo "PASS $fmt planted [$*]"; else echo "FAIL $fmt planted [$*] -> $best"; status=1; fi
}
check fp16 block_k=8 align_frac=24
check fp16 block_k=4 align_frac=25 round=rne
check fp16 align_ref=actual trunc=floor
check bf16 zero_in_emax=1 align_frac=26
check bf16 overflow_to_inf=0 block_k=16
exit $status
