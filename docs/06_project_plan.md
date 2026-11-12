# Project plan, status and resume entry

## Status

| Part | State | Evidence |
|---|---|---|
| Numerics spec + C++ golden model | done | `make test`: 53 checks |
| RTL: FDA lane + MMA block, 1-6 stages, FP16/BF16 | done | `make sim`: 0 mismatches across 12 configurations + gate-level |
| Gate-level simulation of the Yosys netlist | done | found and fixed a signed-arithmetic synthesis mismatch |
| Probe generator + fit tool | done | `make selftest`: recovers 5 planted configurations |
| Yosys area/depth sweep | done | `syn/yosys/results.md` |
| CUDA probe runner (WMMA + `mma.sync`) | written, compiled for sm_86 | **run on the RTX 3050** |
| CUDA GEMM benchmark (naive, tiled, 2x WMMA, cuBLAS) | written, compiled for sm_86 | **run on the RTX 3050** |
| OpenROAD ASAP7/Sky130 flow | configs + sweep script | **run with ORFS** |

## Remaining steps (about 3 weeks at 12 h/week)

1. **Week 1: GPU.** Install CUDA on the laptop (WSL2 or Linux), then run
   `scripts/run_gpu_characterization.sh`. If the default hypothesis matches,
   record it. If not, follow doc 04 "If it does not match". Then run
   `make bench` and one Nsight Compute profile of `wmma_tiled`.
2. **Week 2: physical design.** Run the ORFS sweep for stages 3-6 in ASAP7,
   then the BF16 on/off and F=24/28 comparisons at the best depth. Fill in
   `syn/orfs/results_asap7.md`.
3. **Week 3: write-up.** Update the README numbers. Optional stretch: a
   SymbiYosys equivalence check RTL vs netlist, or implementing one
   model-only variant (e.g. RNE) in RTL.

## Learning checklist (interview topics this project covers)

- [ ] Explain why the tensor core is not IEEE 754: one alignment, truncation without sticky bits, single normalisation.
- [ ] Derive by hand why c = 2-2^-23 gives a larger result than c = 2 (non-monotonicity).
- [ ] Explain 8-term vs 4-term blocks and why a 4x4x4 Volta-style unit cannot match Ampere bit-for-bit.
- [ ] Walk through the CSA tree: why 3:2 compressors, and why only one carry-propagate adder.
- [ ] Explain the gate-level bug: how signed casts can be interpreted differently by two tools, and how offset-binary compares avoid it.
- [ ] Explain WMMA vs `mma.sync` and the m16n8k16 fragment layout.
- [ ] Explain why GeForce FP16 tensor throughput halves with FP32 accumulation.

## Resume entry

Replace each `[...]` with the measured number. Bullets 1, 3 and 5 can
already use the numbers from this repository.

**Tensor-Core MMA Unit with GPU-Matched Numerics** | SystemVerilog, C++, CUDA, Verilator, Yosys, OpenROAD | [Month 2027]

* Designed a pipelined (1-6 stage) mixed-precision tensor-core MMA block in SystemVerilog: 64 FP16/BF16 multipliers with FP32 accumulation, organised as 8-term fused dot-product lanes. Each lane uses a 3:2 carry-save tree, a single normalisation and truncating (RZ) rounding, as NVIDIA Ampere does.
* Reproduced published Ampere tensor-core numerics on an RTX 3050 (GA107) using [N] targeted CUDA WMMA/`mma.sync` probe tiles: [truncation during alignment, 8-product blocks, no intermediate normalisation]. Built a parameter-fitting tool that ranks 432 numeric hypotheses against the GPU dumps.
* Built a configurable bit-accurate C++ golden model and Verilator testbenches: 0 mismatches over 2M+ constrained-random vectors across 12 RTL configurations and 16-deep chained reductions. Gate-level simulation of the Yosys netlist caught and fixed a signed-arithmetic synthesis/simulation mismatch.
* Wrote CUDA GEMM kernels (naive, shared-memory tiled, WMMA, tiled WMMA) and benchmarked them against cuBLAS: [X] TFLOP/s for tiled WMMA vs [Y] for cuBLAS FP16->FP32 at [n]^3. Profiled with Nsight Compute.
* Quantified design trade-offs: BF16 support costs +4.5% area with a shared 11x11 multiplier vs +20% with a separate one, and pipelining from 1 to 5 stages cuts logic depth 2.9x for +11% area (Yosys). [ASAP7 OpenROAD: X MHz in Y um^2.]
