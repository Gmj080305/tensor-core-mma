#include "tcmma/gen.hpp"

#include <algorithm>
#include <cmath>

namespace tcmma {

const char* gen_class_name(GenClass g) {
    switch (g) {
        case GenClass::Normal: return "normal";
        case GenClass::Wide: return "wide";
        case GenClass::Cancel: return "cancel";
        case GenClass::Subnormal: return "subnormal";
        case GenClass::Special: return "special";
        case GenClass::Overflow: return "overflow";
        case GenClass::RawBits: return "rawbits";
        case GenClass::Boundary: return "boundary";
        default: return "?";
    }
}

namespace {
int ebits_of(InFmt f) { return f == InFmt::FP16 ? 5 : 8; }
int fbits_of(InFmt f) { return f == InFmt::FP16 ? 10 : 7; }
int bias_of(InFmt f) { return f == InFmt::FP16 ? 15 : 127; }
}  // namespace

uint16_t VecGen::in_from_parts(InFmt fmt, bool s, int ue, uint32_t frac) {
    const int bias = bias_of(fmt), fb = fbits_of(fmt);
    int be = ue + bias;
    be = std::clamp(be, 1, (1 << ebits_of(fmt)) - 2);
    return static_cast<uint16_t>((s ? 0x8000u : 0u) | (uint32_t(be) << fb) |
                                 (frac & ((1u << fb) - 1)));
}

uint16_t VecGen::rand_in(InFmt fmt, int lo, int hi, double p_sub) {
    std::uniform_real_distribution<double> u01(0, 1);
    const bool s = rng_() & 1;
    const uint32_t frac = static_cast<uint32_t>(rng_());
    if (u01(rng_) < p_sub) {
        const uint32_t f = frac & ((1u << fbits_of(fmt)) - 1);
        return static_cast<uint16_t>((s ? 0x8000u : 0u) | (f ? f : 1u));
    }
    std::uniform_int_distribution<int> ed(lo, hi);
    return in_from_parts(fmt, s, ed(rng_), frac);
}

uint32_t VecGen::rand_fp32(int lo, int hi) {
    std::uniform_int_distribution<int> ed(lo, hi);
    const uint32_t s = static_cast<uint32_t>(rng_() & 1) << 31;
    const int be = std::clamp(ed(rng_) + 127, 1, 254);
    return s | (uint32_t(be) << 23) | (static_cast<uint32_t>(rng_()) & 0x7FFFFF);
}

FdaVector VecGen::make(GenClass g, InFmt fmt, int k, int align_frac) {
    FdaVector v;
    v.a.resize(k);
    v.b.resize(k);
    std::uniform_real_distribution<double> u01(0, 1);
    const int emin = 1 - bias_of(fmt), emax = bias_of(fmt);

    switch (g) {
        case GenClass::Normal:
            for (int i = 0; i < k; ++i) { v.a[i] = rand_in(fmt, -3, 3); v.b[i] = rand_in(fmt, -3, 3); }
            v.c = rand_fp32(-4, 6);
            break;
        case GenClass::Wide:
            for (int i = 0; i < k; ++i) { v.a[i] = rand_in(fmt, emin, emax); v.b[i] = rand_in(fmt, emin, emax); }
            v.c = (rng_() & 3) ? rand_fp32(-126, 127) : 0;
            break;
        case GenClass::Cancel: {
            for (int i = 0; i < k; ++i) { v.a[i] = rand_in(fmt, -2, 2); v.b[i] = rand_in(fmt, -2, 2); }
            // Mirror pairs so that products cancel exactly, then perturb one pair.
            for (int i = 0; i + 1 < k; i += 2) {
                v.a[i + 1] = v.a[i] ^ 0x8000;
                v.b[i + 1] = v.b[i];
            }
            if (k >= 2) {
                const int i = static_cast<int>(rng_() % k);
                v.a[i] = rand_in(fmt, -14, -6);
            }
            const double s = u01(rng_) < 0.5 ? 0.0 : std::ldexp(u01(rng_), -int(rng_() % 30));
            v.c = double_to_fp32((rng_() & 1) ? s : -s);
            break;
        }
        case GenClass::Subnormal:
            for (int i = 0; i < k; ++i) {
                if (fmt == InFmt::BF16) {
                    // products land around 2^-150..2^-120: FP32 subnormal outputs
                    v.a[i] = rand_in(fmt, emin, emin + 10, 0.4);
                    v.b[i] = rand_in(fmt, -22, 2, 0.1);
                } else {
                    v.a[i] = rand_in(fmt, emin, emin + 6, 0.5);
                    v.b[i] = rand_in(fmt, -2, 6, 0.3);
                }
            }
            v.c = (rng_() % 3) ? (static_cast<uint32_t>(rng_()) & 0x807FFFFFu)  // fp32 subnormal
                               : rand_fp32(-126, -110);
            if (rng_() % 4 == 0) {
                // All products exactly zero: a subnormal c passes straight through.
                // (Products that merely cancel would still set the alignment
                // exponent and truncate a tiny c away; see the spec, section 4.)
                for (int i = 0; i < k; ++i) v.a[i] = (rng_() & 1) ? 0x8000 : 0;
                v.c = static_cast<uint32_t>(rng_()) & 0x807FFFFFu;
            }
            break;
        case GenClass::Special: {
            const uint16_t pinf = fmt == InFmt::FP16 ? 0x7C00 : 0x7F80;
            const uint16_t qnan = fmt == InFmt::FP16 ? 0x7E00 : 0x7FC0;
            for (int i = 0; i < k; ++i) {
                v.a[i] = rand_in(fmt, -4, 4);
                v.b[i] = rand_in(fmt, -4, 4);
                const uint64_t r = rng_() % 10;
                if (r == 0) v.a[i] = 0;
                else if (r == 1) v.a[i] = 0x8000;
                else if (r == 2) v.b[i] = pinf;
                else if (r == 3) v.b[i] = pinf | 0x8000;
                else if (r == 4 && (rng_() % 4 == 0)) v.a[i] = qnan;
            }
            const uint64_t rc = rng_() % 6;
            v.c = rc == 0 ? 0x7F800000u : rc == 1 ? 0xFF800000u : rc == 2 ? 0x80000000u
                : rc == 3 ? 0u : rand_fp32(-4, 4);
            if ((rng_() % 3) == 0) {  // all-zero cases exercise the zero-sign rule
                for (int i = 0; i < k; ++i) { v.a[i] = (rng_() & 1) ? 0x8000 : 0; v.b[i] = rand_in(fmt, -2, 2); }
                v.c = (rng_() & 1) ? 0x80000000u : 0u;
            }
            break;
        }
        case GenClass::Overflow:
            for (int i = 0; i < k; ++i) {
                if (fmt == InFmt::BF16) { v.a[i] = rand_in(fmt, 60, 64); v.b[i] = rand_in(fmt, 60, 64); }
                else { v.a[i] = rand_in(fmt, 13, 15); v.b[i] = rand_in(fmt, 13, 15); }
            }
            v.c = rand_fp32(124, 127);
            break;
        case GenClass::RawBits:
            for (int i = 0; i < k; ++i) {
                v.a[i] = static_cast<uint16_t>(rng_());
                v.b[i] = static_cast<uint16_t>(rng_());
            }
            v.c = static_cast<uint32_t>(rng_());
            break;
        case GenClass::Boundary: {
            // One large term at exponent 0, the others placed around the window edge.
            for (int i = 0; i < k; ++i) {
                const int off = align_frac - 3 + static_cast<int>(rng_() % 7);  // F-3 .. F+3
                const int ea = -(off / 2), eb = -(off - off / 2);
                v.a[i] = rand_in(fmt, ea, ea);
                v.b[i] = rand_in(fmt, eb, eb);
                if (fmt == InFmt::FP16 && (ea < -14 || eb < -14)) {
                    v.a[i] = rand_in(fmt, -12, -10);
                    v.b[i] = rand_in(fmt, -14, -12);
                }
            }
            v.a[0] = rand_in(fmt, 0, 0);
            v.b[0] = rand_in(fmt, 0, 0);
            v.c = (rng_() & 1) ? double_to_fp32(-in_to_double(v.a[0], fmt) * in_to_double(v.b[0], fmt))
                               : rand_fp32(-1, 1);
            break;
        }
        default: break;
    }
    return v;
}

}  // namespace tcmma
