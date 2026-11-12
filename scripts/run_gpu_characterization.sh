#!/usr/bin/env bash
# End-to-end GPU characterisation (run on the machine with the RTX 3050):
#   1. build the model and the CUDA programs
#   2. generate probe tiles
#   3. run them through the tensor cores with WMMA and with mma.sync
#   4. predict them with the model's default (hypothesis) configuration
#   5. compare family by family, then search the parameter grid (tcmma fit)
# Everything lands in results/. Usage: bash scripts/run_gpu_characterization.sh [ARCH]
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="${1:-sm_86}"
mkdir -p results

cmake -S model -B model/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build model/build -j >/dev/null
make -C cuda ARCH="$ARCH" >/dev/null
python3 cuda/probes/gen_probes.py --out probes --random-tiles 200

{
  echo "# GPU characterisation run, $(date -u +%Y-%m-%dT%H:%MZ)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
} > results/summary.txt

overall=0
for fmt in fp16 bf16; do
  for api in wmma mma; do
    cuda/build/tc_probe --in "probes/$fmt.bin" --out "results/$fmt.$api.gpu.out" --api "$api" \
      | tee -a results/summary.txt
  done
  model/build/tcmma eval --in "probes/$fmt.bin" --out "results/$fmt.model.out" | tee -a results/summary.txt
  if cmp -s "results/$fmt.wmma.gpu.out" "results/$fmt.mma.gpu.out"; then
    echo "$fmt: WMMA and mma.sync results are bit-identical" | tee -a results/summary.txt
  else
    echo "$fmt: WMMA and mma.sync results DIFFER (see compare output)" | tee -a results/summary.txt
  fi
  echo "== $fmt: GPU (wmma) vs model default hypothesis" | tee -a results/summary.txt
  if ! python3 scripts/compare_dumps.py --probes "probes/$fmt.bin" --gpu "results/$fmt.wmma.gpu.out" \
       --model "results/$fmt.model.out" --families "probes/$fmt.families.txt" | tee -a results/summary.txt; then
    overall=1
  fi
  echo "== $fmt: parameter fit" | tee -a results/summary.txt
  model/build/tcmma fit --in "probes/$fmt.bin" --gpu "results/$fmt.wmma.gpu.out" \
    --families "probes/$fmt.families.txt" --top 8 | tee -a results/summary.txt
done

if [ "$overall" -eq 0 ]; then
  echo "RESULT: the default hypothesis matches the GPU bit-exactly for FP16 and BF16." | tee -a results/summary.txt
else
  echo "RESULT: mismatches found; read results/summary.txt (fit ranking + failing families)." | tee -a results/summary.txt
fi
