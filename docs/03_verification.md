# Verification

The RTL is checked against the C++ golden model, which is in turn checked
against an independent exact computation. The model's parameters are
checked against the GPU (doc 04).

```
GPU tensor cores --(probes, tcmma fit)--> C++ model --(Verilator, bit-exact)--> RTL --(Yosys netlist)--> gate-level sim
                                              ^
                       exact-sum and hand-derived unit tests
```

## 1. Model unit tests (`model/tests/test_model.cpp`, `make test`)

| Check | What it proves |
|---|---|
| All 2^16 FP16 and BF16 patterns decode then re-encode identically | format helpers |
| `round_to_fp32` vs host `double -> float` (RNE) and stepped truncation (RZ), 200k values | final rounding, subnormals, overflow |
| FDA with a 60-bit window equals the exactly computed, RZ-rounded sum (50k vectors) | alignment + summation logic, independent reference |
| Result independent of product order inside a block (20k vectors, all classes) | exact fixed-point sum |
| Hand-derived results for every probe family (alignment bits, block size, truncation mode, final rounding, zero in E, product-MSB reference, subnormal reference) | each parameter changes exactly the bits the spec says |
| Non-monotonic example: c = 2-2^-23 gives 2+2^-22, c = 2 gives 2 | reproduces the Mikaitis (2024) effect |
| Specials: inf*0, inf-inf, NaN, -0 rules, x-x = +0; overflow; FP32 subnormal outputs | special-value table |

The run also reports how often the tensor-core result differs from an IEEE
FP32 FMA chain on random FP16 dot products (57% in the default run). That
shows the model is not IEEE in disguise.

## 2. RTL simulation (`tb/verilator`, `make sim`)

**Lane testbench (`tb_fda.cpp`).** Each operation is predicted by the model
when it is issued and compared bit-for-bit when it leaves the pipeline.
Valid has random gaps (85% issue rate), and the latency is checked against
`NUM_STAGES`. Stimulus comes from `tcmma::VecGen`, the same generator the
model tests use:

| Class | Targets |
|---|---|
| normal | typical values, small exponent spread |
| wide | full exponent range, large alignment shifts |
| cancel | mirrored products cancel exactly, small residues |
| subnormal | subnormal inputs; FP32 subnormal outputs; exactly-zero products passing a subnormal c |
| special | zeros of both signs, infinities, NaNs, the -0 rule |
| overflow | sums at or above 2^128 |
| rawbits | uniformly random bit patterns |
| boundary | small terms placed F-3 .. F+3 bits below the largest one (window edge) |

Required coverage bins: every output class (normal, subnormal, zero, inf,
NaN) and the main input class/format pairs. A hole fails the test.

**Block testbench (`tb_mma.cpp`).**
* Streaming random operations, every lane compared.
* 16-deep chained reductions with D fed back as C, compared with the model's `dot_chain`.

**Regression matrix (`make sim`).** Results from the run that produced this repository:

| Configuration | Operations | Mismatches |
|---|---|---|
| `tc_fda` K=8 F=24, NUM_STAGES = 1, 2, 3, 4, 5, 6 | 200,000 each | 0 |
| `tc_fda` K=4 (Volta lane) | 200,000 | 0 |
| `tc_fda` F=28 window | 200,000 | 0 |
| `tc_fda` split BF16 multiplier | 200,000 | 0 |
| `tc_fda` FP16-only | 200,000 | 0 |
| `tc_mma_core` 4x2x8, 3 stages | 20,000 streamed + 5,000 chained (16-deep) | 0 |
| `tc_mma_core` 4x4x4, 4 stages | 20,000 streamed + 5,000 chained (16-deep) | 0 |
| Gate-level: Yosys netlist of `tc_fda` K=8, 3 stages | 100,000 | 0 |

## 3. Gate-level simulation (`make gl`)

Yosys synthesises `tc_fda` to generic gates, and the netlist runs under the
same testbench. On the first attempt this found a real
synthesis-versus-simulation mismatch: 13.5% of results were wrong in the
netlist, while the RTL passed. Yosys 0.33 and Verilator interpreted signed
size casts such as `-EW'(14)` and mixed signed/unsigned comparisons
differently. The fix rewrote the exponent arithmetic in plain two's
complement with explicit sign-bit tests (offset-binary compares in the max
tree, biased-exponent range checks in the packer). RTL and netlist now both
match the model.

## 4. Probe and fit self-test (`make selftest`)

Without a GPU this proves the characterisation method works. The model with
a planted non-default configuration acts as a fake GPU, and `tcmma fit` must
rank exactly that configuration first with 0 mismatches. It currently passes
for five planted configurations covering every parameter. Writing this test
exposed a weak probe family (`zero_in_emax` could not see a 26-bit window in
BF16), which was then fixed.

## Known limits

* Formal equivalence (for example Yosys `equiv` or SymbiYosys) between RTL and netlist is not set up; equivalence rests on simulation.
* The model-only numeric variants are not implemented in RTL (doc 02).
* No stall/backpressure: the block assumes the consumer accepts every result.
