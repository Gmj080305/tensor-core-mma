// Constrained-random stimulus for the FDA: shared by the model unit tests and
// the Verilator testbench so both exercise the same corner classes.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "tcmma/formats.hpp"

namespace tcmma {

enum class GenClass {
    Normal,      // normals with a small exponent spread
    Wide,        // normals over the whole exponent range of the format
    Cancel,      // products that cancel, small residues survive or not
    Subnormal,   // many subnormal inputs (and FP32 subnormal accumulators)
    Special,     // zeros, infinities, NaNs mixed in
    Overflow,    // large magnitudes near the FP32 overflow threshold (BF16)
    RawBits,     // uniformly random bit patterns
    Boundary,    // terms placed exactly at / just past the alignment window edge
    kCount
};

const char* gen_class_name(GenClass g);

struct FdaVector {
    std::vector<uint16_t> a, b;
    uint32_t c = 0;
};

class VecGen {
public:
    explicit VecGen(uint64_t seed) : rng_(seed) {}
    FdaVector make(GenClass g, InFmt fmt, int k, int align_frac = 24);
    uint64_t next() { return rng_(); }

private:
    uint16_t rand_in(InFmt fmt, int exp_lo, int exp_hi, double p_sub = 0.0);
    uint16_t in_from_parts(InFmt fmt, bool s, int unbiased_exp, uint32_t frac);
    uint32_t rand_fp32(int exp_lo, int exp_hi);
    std::mt19937_64 rng_;
};

}  // namespace tcmma
