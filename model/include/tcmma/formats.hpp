// Floating-point format helpers for the tensor-core MMA golden model.
//
// Values are described exactly as (sign, integer significand, exponent):
//   value = (-1)^sign * sig * 2^(exp - frac_bits)
// For the 16-bit input formats the significand is always widened to 11 bits
// with 10 fractional bits, so FP16 and BF16 share one representation (this is
// what the RTL does to share one 11x11 multiplier).
#pragma once

#include <cstdint>
#include <string>

namespace tcmma {

enum class InFmt : uint32_t { FP16 = 0, BF16 = 1 };

const char* fmt_name(InFmt f);
bool parse_fmt(const std::string& s, InFmt& out);

struct Decoded {
    bool sign = false;
    bool is_zero = false;
    bool is_inf = false;
    bool is_nan = false;
    bool is_subnormal = false;
    uint32_t sig = 0;  // integer significand incl. hidden bit (0 for zero)
    int exp = 0;       // unbiased exponent of the hidden-bit position
};

// Fractional bits of the widened input significand (both formats).
constexpr int kInFracBits = 10;
// Fractional bits of an FP32 significand.
constexpr int kF32FracBits = 23;

// Decode a 16-bit input. `daz` flushes subnormal inputs to zero.
Decoded decode_in(uint16_t bits, InFmt f, bool daz = false);
// Decode an FP32 value (accumulator input).
Decoded decode_fp32(uint32_t bits);

// Exact conversions to double (all FP16/BF16/FP32 values are exact in double).
double in_to_double(uint16_t bits, InFmt f);
double fp32_to_double(uint32_t bits);

// Round-to-nearest-even encoders, used by generators and tests.
uint16_t double_to_in(double x, InFmt f);
uint32_t double_to_fp32(double x);

// Pretty-print helpers.
std::string hex16(uint16_t v);
std::string hex32(uint32_t v);
std::string describe_fp32(uint32_t bits);

inline bool fp32_is_nan(uint32_t b) {
    return ((b >> 23) & 0xFF) == 0xFF && (b & 0x7FFFFF) != 0;
}

}  // namespace tcmma
