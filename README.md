# Tensor-Core MMA Unit with GPU-Matched Numerics

A mixed-precision tensor-core matrix-multiply-accumulate block in
SystemVerilog. It is built to reproduce, bit for bit, the arithmetic of an
NVIDIA Ampere GPU (RTX 3050, GA107) rather than IEEE 754. The repository
contains:

- a bit-accurate C++ golden model
- the RTL, with Verilator and gate-level verification
- CUDA programs that measure the real GPU, plus a GEMM benchmark
- synthesis and place-and-route flows

```
D[4x2] = A[4x8] * B[8x2] + C[4x2]     FP16 or BF16 inputs, FP32 accumulate
64 multipliers, 8-product fused dot-add lanes, 1-6 pipeline stages
```

## Why it is interesting

Tensor cores do not round like a CPU:

- all products are aligned to the largest exponent
- bits shifted out are dropped (no sticky bit)
- the sum is normalised once and truncated

As a result, a tensor-core dot product differs from an IEEE FP32 FMA chain in
about 57% of random FP16 cases in our tests. It can even be **non-monotonic**:

```
$ model/build/tcmma demo
  8 products of 2^-24, c = 0x3FFFFFFF (1.99999988) -> d = 0x40000001 (2.00000024)
  8 products of 2^-24, c = 0x40000000 (2)          -> d = 0x40000000 (2)   (larger c, smaller d)
```

Published measurements (Khattak & Mikaitis 2025; MMA-Sim 2025) describe
Ampere as summing **8** products per fused block, so the default lane here is
8-term rather than the Volta-style 4-term lane. Both are supported.

## Status

| Part | State |
|---|---|
| Numerics spec, C++ golden model, unit tests | done |
| RTL (`tc_fda` lane, `tc_mma_core`), 1-6 stages, FP16/BF16 | done: 0 mismatches vs model over 2M+ vectors, 12 configurations |
| Gate-level simulation of the Yosys netlist | done: 0 mismatches (caught and fixed a synthesis/sim mismatch) |
| GPU probe generator + hypothesis fitting tool | done: self-test recovers planted configurations |
| Yosys area / logic-depth sweep | done: [syn/yosys/results.md](syn/yosys/results.md) |
| CUDA probe runner and GEMM benchmark | compiled for sm_86, **not yet run on the GPU** |
| OpenROAD (ASAP7 / Sky130) | flow configured, **not yet run** |

The default numeric parameters are a hypothesis taken from the literature
until `scripts/run_gpu_characterization.sh` confirms them on the RTX 3050.
See [docs/04](docs/04_gpu_characterization.md).

## Quick start

```bash
sudo apt-get install -y verilator yosys cmake g++ python3   # Ubuntu 24.04
make test       # model unit tests
make sim        # Verilator regression (all configs) + gate-level sim
make selftest   # probe + fit pipeline self-test
make synth      # Yosys sweep -> syn/yosys/results.md

# on the machine with the GPU (CUDA toolkit installed):
make gpu        # characterise the tensor cores -> results/summary.txt
make bench      # GEMM: naive / tiled / WMMA / cuBLAS
```

## Layout

```
docs/       01 numerics spec · 02 microarchitecture · 03 verification · 04 GPU characterisation
            05 synthesis & PD · 06 plan, status, resume entry · references
model/      C++17 golden model (libtcmma), `tcmma` CLI (eval / fit / one / demo), unit tests
rtl/        tc_fda (lane), tc_mma_core (block), tc_csa_tree, tc_lzc, tc_mul, unpackers, tc_pipe
tb/         Verilator testbenches (lane, block, gate-level) and regression Makefile
cuda/       tc_probe (WMMA + mma.sync probe runner), gen_probes.py, gemm_bench
syn/        yosys/ generic sweep · orfs/ OpenROAD-flow-scripts configs (asap7, sky130hd)
scripts/    GPU characterisation driver, dump comparison, fit self-test
```

## Key results so far

| Result | Value | Source |
|---|---|---|
| RTL vs model, all configurations | 0 mismatches | `make sim` |
| BF16 support area cost, shared 11x11 multiplier | +4.5% | Yosys sweep |
| BF16 support area cost, separate 8x8 multiplier | +20.4% | Yosys sweep |
| Logic depth, 1 -> 5 pipeline stages | 183 -> 64 levels (+11% area) | Yosys sweep |
| Alignment window 24 -> 28 bits | +5.3% area | Yosys sweep |

## License

MIT. See [LICENSE](LICENSE).
