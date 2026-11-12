#include "tcmma/formats.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace tcmma {

namespace {

struct FmtInfo {
    int ebits;
    int fbits;
    int bias;
};

FmtInfo info(InFmt f) {
    return f == InFmt::FP16 ? FmtInfo{5, 10, 15} : FmtInfo{8, 7, 127};
}

// Generic round-to-nearest-even encoder for a binary format with `ebits`
// exponent bits and `fbits` fraction bits. Handles subnormals, overflow to
// infinity and NaN.
uint64_t encode_rne(double x, int ebits, int fbits) {
    const int bias = (1 << (ebits - 1)) - 1;
    const uint64_t sign = std::signbit(x) ? 1 : 0;
    const uint64_t sign_bit = sign << (ebits + fbits);
    const uint64_t exp_all = (1ull << ebits) - 1;
    if (std::isnan(x)) return sign_bit | (exp_all << fbits) | (1ull << (fbits - 1));
    double a = std::fabs(x);
    if (std::isinf(a)) return sign_bit | (exp_all << fbits);
    if (a == 0.0) return sign_bit;

    int e2;
    std::frexp(a, &e2);  // a = m * 2^e2, m in [0.5,1)
    int e = e2 - 1;      // a in [2^e, 2^(e+1))
    const int emin = 1 - bias;
    int q;  // exponent of the lsb of the target grid
    if (e < emin) q = emin - fbits; else q = e - fbits;
    double scaled = std::ldexp(a, -q);  // exact (power-of-two scaling)
    double fl = std::floor(scaled);
    double rem = scaled - fl;
    uint64_t m = static_cast<uint64_t>(fl);
    if (rem > 0.5 || (rem == 0.5 && (m & 1))) m += 1;
    // m * 2^q is the rounded magnitude.
    if (e < emin) {
        // subnormal (may round up into the smallest normal: encoding carries).
        return sign_bit | m;
    }
    if (m == (1ull << (fbits + 1))) {  // carried into next binade
        m >>= 1;
        e += 1;
    }
    if (e > bias) return sign_bit | (exp_all << fbits);  // overflow -> inf
    uint64_t biased = static_cast<uint64_t>(e + bias);
    return sign_bit | (biased << fbits) | (m & ((1ull << fbits) - 1));
}

}  // namespace

const char* fmt_name(InFmt f) { return f == InFmt::FP16 ? "fp16" : "bf16"; }

bool parse_fmt(const std::string& s, InFmt& out) {
    if (s == "fp16" || s == "f16" || s == "half") { out = InFmt::FP16; return true; }
    if (s == "bf16" || s == "bfloat16") { out = InFmt::BF16; return true; }
    return false;
}

Decoded decode_in(uint16_t bits, InFmt f, bool daz) {
    const FmtInfo fi = info(f);
    Decoded d;
    d.sign = (bits >> 15) & 1;
    const uint32_t e = (bits >> fi.fbits) & ((1u << fi.ebits) - 1);
    const uint32_t fr = bits & ((1u << fi.fbits) - 1);
    const uint32_t emax = (1u << fi.ebits) - 1;
    const int widen = kInFracBits - fi.fbits;  // 0 for FP16, 3 for BF16
    if (e == emax) {
        if (fr) d.is_nan = true; else d.is_inf = true;
        return d;
    }
    if (e == 0) {
        d.exp = 1 - fi.bias;
        if (fr == 0 || daz) { d.is_zero = true; d.sig = 0; return d; }
        d.is_subnormal = true;
        d.sig = fr << widen;
        return d;
    }
    d.exp = static_cast<int>(e) - fi.bias;
    d.sig = ((1u << fi.fbits) | fr) << widen;
    return d;
}

Decoded decode_fp32(uint32_t bits) {
    Decoded d;
    d.sign = (bits >> 31) & 1;
    const uint32_t e = (bits >> 23) & 0xFF;
    const uint32_t fr = bits & 0x7FFFFF;
    if (e == 0xFF) {
        if (fr) d.is_nan = true; else d.is_inf = true;
        return d;
    }
    if (e == 0) {
        d.exp = -126;
        if (fr == 0) { d.is_zero = true; return d; }
        d.is_subnormal = true;
        d.sig = fr;
        return d;
    }
    d.exp = static_cast<int>(e) - 127;
    d.sig = (1u << 23) | fr;
    return d;
}

double in_to_double(uint16_t bits, InFmt f) {
    Decoded d = decode_in(bits, f);
    if (d.is_nan) return std::nan("");
    if (d.is_inf) return d.sign ? -INFINITY : INFINITY;
    double v = std::ldexp(static_cast<double>(d.sig), d.exp - kInFracBits);
    return d.sign ? -v : v;
}

double fp32_to_double(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return static_cast<double>(f);
}

uint16_t double_to_in(double x, InFmt f) {
    const FmtInfo fi = info(f);
    return static_cast<uint16_t>(encode_rne(x, fi.ebits, fi.fbits));
}

uint32_t double_to_fp32(double x) { return static_cast<uint32_t>(encode_rne(x, 8, 23)); }

std::string hex16(uint16_t v) {
    char b[8];
    std::snprintf(b, sizeof b, "0x%04X", v);
    return b;
}

std::string hex32(uint32_t v) {
    char b[12];
    std::snprintf(b, sizeof b, "0x%08X", v);
    return b;
}

std::string describe_fp32(uint32_t bits) {
    char b[64];
    std::snprintf(b, sizeof b, "%s (%.9g)", hex32(bits).c_str(), fp32_to_double(bits));
    return b;
}

}  // namespace tcmma
