# GPU characterisation on the RTX 3050 (GA107, sm_86)

Goal: measure how the RTX 3050's tensor cores round, and fix the model
parameters (doc 01) to what the hardware does. The RTL is then signed off
against those parameters.

**Status:** not yet run. The CUDA programs compile for sm_86 with clang 18
against the CUDA 12.6 headers, and `ptxas` accepts the `mma.sync` PTX. They
have not been executed: no GPU was available while building the repository.

## Setup

* Linux, or Windows with WSL2 (install the NVIDIA Windows driver, then the CUDA toolkit for WSL inside Ubuntu).
* CUDA toolkit 12.x (`nvcc`), CMake, a C++17 compiler, Python 3.
* Check with `nvidia-smi`; the RTX 3050 Laptop GPU should report compute capability 8.6.

## One command

```bash
scripts/run_gpu_characterization.sh sm_86
```

The script runs these steps:

1. Builds the model and `cuda/build/tc_probe`.
2. Generates probes (`probes/fp16.bin`, `probes/bf16.bin`, about 825 tiles each, about 211,000 checked output elements per format).
3. Runs every tile through WMMA and through `mma.sync`.
4. Evaluates the model with the default hypothesis.
5. Compares the GPU and model results by family, then runs `tcmma fit`.

Results go to `results/` and a summary to `results/summary.txt`.

## What each probe family asks

| Family | Question | How it decides |
|---|---|---|
| `align_bits` | How many fractional bits survive alignment (F)? | `1 - 1 + 2^-n` returns `2^-n` iff n <= F |
| `block_size` | How many products per fused block? | `1` and `-1` at k = p, p+1, then `2^-30` at k = j: non-zero iff j is in a different block |
| `chain_order` | Does c join the first block? | c = `2^-n` with `1 - 1` in block 0 vs block 1 |
| `trunc_mode` | Are dropped bits truncated per term toward zero or toward minus infinity? | `1 - 2^-n` with n > F: 1.0 vs 1 - 2^-24 |
| `final_round` | RZ or RNE to FP32? | remainders below, at and above half an ulp, both signs |
| `zero_in_emax` | Does a zero product still raise the alignment exponent? | 0 x max-finite next to `1 - 1 + 2^-n` |
| `align_ref_product_msb` | Is alignment based on e_a+e_b or on the true leading one? | 1.5 x 1.5 = 2.25 has its leading one above e_a+e_b |
| `align_ref_subnormal` | Does a subnormal input lower the reference exponent? | subnormal x 2^14 next to a small term |
| `subnormal_in`, `subnormal_out` | Are subnormals supported or flushed? | exact tiny products, FP32 subnormal results |
| `overflow` | Does `>= 2^128` give inf or the largest finite value? | BF16 sums near FP32 max |
| `specials` | NaN, inf, -0 rules | table in doc 01 |
| `random_*` | Everything at once | full random tiles |

`make selftest` shows that these families separate every parameter value in
the fit grid. If the 3050 matches the default hypothesis, the summary ends with:

```
RESULT: the default hypothesis matches the GPU bit-exactly for FP16 and BF16.
```

## If it does not match

1. Read the `tcmma fit` ranking. If a grid point matches exactly, it is the answer: record it in doc 01, and if it is a model-only option, implement it in the RTL (doc 02 lists where).
2. If nothing matches, the per-family table shows which question failed. `scripts/compare_dumps.py --show 20` prints the failing elements with the positions of their non-zero products.
3. Check the first published characterisations before changing the model: Khattak & Mikaitis (2025) for the RTX 3060, MMA-Sim (Xie et al., 2025) for the A100.
4. Also check whether WMMA and `mma.sync` agree (the summary says so). They should both compile to `HMMA.16816` on sm_86.

## Layout self-check

`tc_probe` first runs random tiles against a double-precision reference with
a loose tolerance (1e-4 relative). A wrong fragment layout in the hand-written
`mma.sync` path gives errors of order 1 and stops the run. Run only this check
with `cuda/build/tc_probe --selfcheck-only --api mma`.

## GEMM benchmark

`make bench` (or `cuda/build/gemm_bench 1024 2048 4096`) times naive and
tiled FP32 kernels, two WMMA kernels and cuBLAS. It prints ms, TFLOP/s and
maximum relative error against `cublasSgemm`. On GeForce Ampere, tensor-core
throughput with FP32 accumulation is half the FP16-accumulate rate; compare
against that number. To profile:

```bash
ncu --kernel-name wmma_tiled --launch-count 1 --set full cuda/build/gemm_bench 2048
```
