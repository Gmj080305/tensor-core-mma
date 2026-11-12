// Bit-accurate model of a tensor-core fused dot-add (FDA) and of tile MMA
// built from chained FDAs. See docs/01_numerics_spec.md for the definition.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tcmma/formats.hpp"

namespace tcmma {

enum class AlignRef { Nominal, Actual };
enum class TruncMode { SignMag, Floor };
enum class RoundMode { RZ, RNE };

struct FdaConfig {
    int block_k = 8;                      // products per FDA (8: Ampere hypothesis)
    int align_frac = 24;                  // fractional bits kept below the reference exponent
    AlignRef align_ref = AlignRef::Nominal;
    bool zero_in_emax = false;            // zero products still feed the exponent max
    TruncMode trunc = TruncMode::SignMag; // how bits shifted out are dropped
    RoundMode round = RoundMode::RZ;      // final rounding to FP32
    bool overflow_to_inf = true;          // |result| >= 2^128 -> inf even under RZ
    bool daz = false;                     // flush subnormal 16-bit inputs to zero
    uint32_t nan_bits = 0x7FFFFFFFu;      // canonical NaN produced

    std::string describe() const;
    // Parse "key=value" (e.g. "block_k=4", "round=rne"). Returns false on error.
    bool set(const std::string& key, const std::string& value);
};

// The configuration the RTL implements (and the default hypothesis).
FdaConfig rtl_config(int block_k = 8, int align_frac = 24);

// One fused dot-add: d = c + sum_{i<k} a[i]*b[i]. `k` may be smaller than
// cfg.block_k (a partial block); it must not be larger.
uint32_t fda(const FdaConfig& cfg, InFmt fmt, const uint16_t* a, const uint16_t* b, int k,
             uint32_t c);

// Dot product of length k_total computed as chained FDAs of cfg.block_k
// products: acc = c; for each block: acc = fda(acc, block).
uint32_t dot_chain(const FdaConfig& cfg, InFmt fmt, const uint16_t* a, const uint16_t* b,
                   int k_total, uint32_t c);

// Tile MMA: D[MxN] = A[MxK] * B[KxN] + C[MxN]; all row-major.
void mma_tile(const FdaConfig& cfg, InFmt fmt, int M, int N, int K, const uint16_t* A,
              const uint16_t* B, const uint32_t* C, uint32_t* D);

// Convert sign * mag * 2^lsb_exp to FP32 bits with the given rounding.
uint32_t round_to_fp32(bool neg, unsigned __int128 mag, int lsb_exp, RoundMode rm,
                       bool overflow_to_inf);

// Reference: IEEE-754 sequential FP32 FMA chain, acc = fma(a_i, b_i, acc) for
// i = 0..k-1 (what a CUDA core loop would compute). Used for comparisons.
uint32_t ieee_fma_chain(InFmt fmt, const uint16_t* a, const uint16_t* b, int k, uint32_t c);

}  // namespace tcmma
