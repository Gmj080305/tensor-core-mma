# Numerics specification: the fused dot-add (FDA)

This document defines, bit for bit, the arithmetic implemented by the C++
golden model (`model/`) and the RTL (`rtl/`). The default parameter values are
our reading of published measurements of NVIDIA Ampere tensor cores. They are a
**hypothesis**: `cuda/probes/` exists to confirm or refute each one on a real
RTX 3050 (GA107), and `tcmma_fit` picks the parameter set that matches the GPU.

## 1. Operation

One FDA computes

```
d = c + sum_{i=0}^{K-1} a_i * b_i
```

* `a_i`, `b_i`: FP16 (binary16) or BF16 (bfloat16), selected per operation.
* `c`, `d`: FP32 (binary32).
* `K` = `block_k` products per FDA (8 for the Ampere hypothesis, 4 for a
  Volta-style unit).

A `16x16x16` (WMMA) or `m16n8k16` (`mma.sync`) tile reduces 16 products per
output element. The model computes it as a chain of FDAs, with the
intermediate result rounded to FP32 between links:

```
d = FDA( FDA(c, a[0:8], b[0:8]), a[8:16], b[8:16] )      # block_k = 8
```

That is the `(c + T1) + T2` order reported by Khattak & Mikaitis (2025) and
the "two chained fused-dot-add operations of K/2 pairs" reported by MMA-Sim
(Xie et al., 2025).

## 2. Decoding

Every finite input is turned into an integer significand and an exponent.

| Format | Fraction bits | Bias | Normal significand | Subnormal significand | Unbiased exponent |
|---|---|---|---|---|---|
| FP16 | 10 | 15 | `1.f` (11 bits) | `0.f`, exponent = -14 | -14 ... 15 |
| BF16 | 7 | 127 | `1.f` (8 bits) | `0.f`, exponent = -126 | -126 ... 127 |
| FP32 | 23 | 127 | `1.f` (24 bits) | `0.f`, exponent = -126 | -126 ... 127 |

The hardware widens a BF16 significand to 11 bits by appending 3 zero bits,
so FP16 and BF16 share one 11x11 multiplier. This does not change any value.

## 3. Terms

Each product is exact: an 11x11 significand product has 22 bits, of which 20
are fractional. Its **nominal exponent** is `e_p = e_a + e_b`. The product's
value lies in `[0, 4) * 2^e_p`, so its leading one can sit at `e_p` or
`e_p + 1`.

The accumulator `c` is a term with nominal exponent `e_c`, value in
`[0, 2) * 2^e_c`.

All terms are widened to `F = align_frac` fractional bits (default 24):

```
product term   T = p << (F - 20)     value = T * 2^(e_p - F)
c term         T = s_c << (F - 23)   value = T * 2^(e_c - F)
```

## 4. Alignment (the non-IEEE step)

1. **Reference exponent.** `E = max(e_t)` over the terms that take part:
   * `align_ref = nominal` (default): `e_t` is the nominal exponent from
     section 3, read straight from the exponent fields with no leading-zero
     count. A subnormal input does not lower it.
   * `align_ref = actual`: `e_t` is the true position of the term's leading
     one. Kept in the model only, to test the alternative.
   * `zero_in_emax = false` (default): zero terms are ignored. If `true`, a
     zero product still contributes `e_a + e_b`. That is what a datapath with no
     zero check before the max tree would do.
2. **Window.** Only bits with weight `>= 2^(E - F)` survive. Each term is
   shifted right by `E - e_t` and the bits that fall out are **discarded**:
   * `trunc = signmag` (default): truncate the magnitude, then apply the sign
     (toward zero for each term). Valpey et al. (2025) describe the hardware
     this way: "bits discarded during the shift are lost".
   * `trunc = floor`: two's-complement truncation (toward minus infinity for
     each term). Model-only alternative.
3. **No sticky bit, no rounding, no intermediate normalisation.** The aligned
   terms are added as exact signed integers. Order does not matter inside an FDA.

## 5. Normalisation and final rounding

The exact signed integer sum `S` has value `S * 2^(E - F)`. It is converted to
FP32 once:

* `round = rz` (default): truncate the magnitude to 24 significant bits (round
  toward zero). MMA-Sim and Khattak & Mikaitis both report truncation for
  FP32 output.
* `round = rne`: round to nearest even. Model-only alternative.
* Results below `2^-126` become FP32 subnormals (truncated under `rz`).
* `overflow_to_inf = true` (default): a magnitude `>= 2^128` becomes infinity,
  as MMA-Sim reports. If `false`, RZ saturates to the largest finite value, as
  IEEE 754 RZ would.

## 6. Zeros and special values

Applied identically in the model and the RTL:

| Condition | Result |
|---|---|
| any `a_i`, `b_i` or `c` is NaN | NaN (`0x7FFFFFFF`) |
| any product is `inf * 0` | NaN |
| both `+inf` and `-inf` among products and `c` | NaN |
| otherwise, any infinite product or `c` | that infinity |
| `S == 0` and every term is a zero with negative sign | `-0` |
| `S == 0` otherwise | `+0` |
| `S != 0` but the result underflows to zero | zero with the sign of `S` |

The NaN payload produced by the GPU is not specified anywhere we found.
`scripts/compare_dumps.py` treats any NaN as matching any NaN unless you pass
`--strict-nan`.

## 7. Parameters and what the GPU probes test

| Parameter | Default (hypothesis) | Alternatives | Probe family that separates them |
|---|---|---|---|
| `block_k` | 8 | 4, 16 | `block_size` |
| `align_frac` (F) | 24 | 23 ... 30 | `align_bits` |
| `align_ref` | nominal | actual | `align_ref_subnormal`, `align_ref_product_msb` |
| `zero_in_emax` | false | true | `zero_in_emax` |
| `trunc` | signmag | floor | `trunc_mode` |
| `round` | rz | rne | `final_round` |
| `overflow_to_inf` | true | false | `overflow` |
| subnormal inputs | supported | flushed | `subnormal_in` |

## 8. Where each default comes from

* 8 products per block, chain order `(c + T1) + T2`, truncation, subnormal
  support: Khattak & Mikaitis, *Generalized Methodology for Determining
  Numerical Features of Hardware Floating-Point Matrix Multipliers: Part I*,
  IEEE HPEC 2025 (RTX 3060 and A100). arXiv:2510.15884.
* 24 fractional alignment bits, exact order-independent fixed-point sum, RZ
  to FP32, overflow to infinity: Xie et al., *MMA-Sim*, arXiv:2511.10909
  (A100).
* Truncation rather than IEEE round-toward-zero: Valpey et al., *An SMT
  Formalization of Mixed-Precision Matrix Multiplication*, NFM 2025.
* Non-monotonicity of such adders: Mikaitis, IEEE TC 73(6), 2024.

Khattak & Mikaitis describe the width as "1 extra alignment bit, 25-bit
datapath" and MMA-Sim as "24 fractional bits". We believe these count the
same window differently. `align_frac` is a parameter so the GPU can settle it.
