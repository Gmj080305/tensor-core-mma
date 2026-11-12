#include "tcmma/fda.hpp"

#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace tcmma {

namespace {

using u128 = unsigned __int128;
using i128 = __int128;

int msb_index(u128 v) {  // v != 0
    int n = 0;
    while (v >>= 1) ++n;
    return n;
}

struct Term {
    bool neg = false;
    bool zero = true;
    uint64_t t = 0;  // magnitude, value = t * 2^(e - F)
    int e = 0;       // nominal exponent
};

}  // namespace

std::string FdaConfig::describe() const {
    std::ostringstream os;
    os << "block_k=" << block_k << " align_frac=" << align_frac
       << " align_ref=" << (align_ref == AlignRef::Nominal ? "nominal" : "actual")
       << " zero_in_emax=" << (zero_in_emax ? 1 : 0)
       << " trunc=" << (trunc == TruncMode::SignMag ? "signmag" : "floor")
       << " round=" << (round == RoundMode::RZ ? "rz" : "rne")
       << " overflow_to_inf=" << (overflow_to_inf ? 1 : 0) << " daz=" << (daz ? 1 : 0);
    return os.str();
}

bool FdaConfig::set(const std::string& key, const std::string& v) {
    try {
        if (key == "block_k") block_k = std::stoi(v);
        else if (key == "align_frac") align_frac = std::stoi(v);
        else if (key == "align_ref") {
            if (v == "nominal") align_ref = AlignRef::Nominal;
            else if (v == "actual") align_ref = AlignRef::Actual;
            else return false;
        } else if (key == "zero_in_emax") zero_in_emax = std::stoi(v) != 0;
        else if (key == "trunc") {
            if (v == "signmag") trunc = TruncMode::SignMag;
            else if (v == "floor") trunc = TruncMode::Floor;
            else return false;
        } else if (key == "round") {
            if (v == "rz") round = RoundMode::RZ;
            else if (v == "rne") round = RoundMode::RNE;
            else return false;
        } else if (key == "overflow_to_inf") overflow_to_inf = std::stoi(v) != 0;
        else if (key == "daz") daz = std::stoi(v) != 0;
        else if (key == "nan_bits") nan_bits = static_cast<uint32_t>(std::stoul(v, nullptr, 0));
        else return false;
    } catch (...) {
        return false;
    }
    return block_k >= 1 && align_frac >= 23 && align_frac <= 60;
}

FdaConfig rtl_config(int block_k, int align_frac) {
    FdaConfig c;
    c.block_k = block_k;
    c.align_frac = align_frac;
    return c;
}

uint32_t round_to_fp32(bool neg, u128 mag, int lsb_exp, RoundMode rm, bool overflow_to_inf) {
    const uint32_t sbit = neg ? 0x80000000u : 0u;
    if (mag == 0) return sbit;
    const int msb = msb_index(mag);
    int e = lsb_exp + msb;  // exponent of the leading one
    if (e >= -126) {
        const int shift = msb - 23;
        u128 sig;
        bool up = false;
        if (shift <= 0) {
            sig = mag << (-shift);
        } else {
            sig = mag >> shift;
            if (rm == RoundMode::RNE) {
                const u128 rem = mag & ((u128(1) << shift) - 1);
                const u128 half = u128(1) << (shift - 1);
                up = rem > half || (rem == half && (sig & 1));
            }
        }
        if (up) {
            sig += 1;
            if (sig == (u128(1) << 24)) { sig >>= 1; e += 1; }
        }
        if (e > 127) {
            if (rm == RoundMode::RNE || overflow_to_inf) return sbit | 0x7F800000u;
            return sbit | 0x7F7FFFFFu;  // RZ saturates to max finite
        }
        return sbit | (static_cast<uint32_t>(e + 127) << 23) |
               (static_cast<uint32_t>(sig) & 0x7FFFFFu);
    }
    // Subnormal result: grid lsb is 2^-149.
    const int shift = -149 - lsb_exp;  // > 0 means drop bits
    u128 frac;
    bool up = false;
    if (shift <= 0) {
        frac = mag << (-shift);
    } else if (shift >= 128) {
        frac = 0;
        up = false;  // magnitude far below half an ulp (mag < 2^128 * 2^lsb_exp)
        if (rm == RoundMode::RNE) {
            // Only possible if mag is enormous relative to the grid; it isn't, since
            // value < 2^-126 implies mag*2^lsb_exp < 2^-126. Treat as zero.
        }
    } else {
        frac = mag >> shift;
        if (rm == RoundMode::RNE) {
            const u128 rem = mag & ((u128(1) << shift) - 1);
            const u128 half = u128(1) << (shift - 1);
            up = rem > half || (rem == half && (frac & 1));
        }
    }
    if (up) frac += 1;  // may carry into the smallest normal; the encoding handles it
    return sbit | static_cast<uint32_t>(frac);
}

uint32_t fda(const FdaConfig& cfg, InFmt fmt, const uint16_t* a, const uint16_t* b, int k,
             uint32_t c) {
    if (k < 0 || k > cfg.block_k) throw std::invalid_argument("fda: k out of range");
    const int F = cfg.align_frac;

    bool nan = false, pinf = false, ninf = false;
    std::vector<Term> terms;
    terms.reserve(k + 1);

    for (int i = 0; i < k; ++i) {
        const Decoded x = decode_in(a[i], fmt, cfg.daz);
        const Decoded y = decode_in(b[i], fmt, cfg.daz);
        const bool neg = x.sign ^ y.sign;
        if (x.is_nan || y.is_nan) { nan = true; continue; }
        if (x.is_inf || y.is_inf) {
            if (x.is_zero || y.is_zero) nan = true;
            else if (neg) ninf = true;
            else pinf = true;
            continue;
        }
        Term t;
        t.neg = neg;
        t.e = x.exp + y.exp;
        const uint64_t p = static_cast<uint64_t>(x.sig) * y.sig;  // 2*kInFracBits frac bits
        t.zero = (p == 0);
        t.t = p << (F - 2 * kInFracBits);
        terms.push_back(t);
    }
    {
        const Decoded z = decode_fp32(c);
        if (z.is_nan) nan = true;
        else if (z.is_inf) { if (z.sign) ninf = true; else pinf = true; }
        else {
            Term t;
            t.neg = z.sign;
            t.e = z.exp;
            t.zero = z.is_zero;
            t.t = static_cast<uint64_t>(z.sig) << (F - kF32FracBits);
            terms.push_back(t);
        }
    }

    if (nan || (pinf && ninf)) return cfg.nan_bits;
    if (pinf) return 0x7F800000u;
    if (ninf) return 0xFF800000u;

    // Reference exponent E: window keeps bits of weight >= 2^(E - F).
    bool have = false;
    int E = 0;
    for (const Term& t : terms) {
        int et;
        if (t.zero) {
            if (!(cfg.zero_in_emax && cfg.align_ref == AlignRef::Nominal)) continue;
            et = t.e;
        } else if (cfg.align_ref == AlignRef::Nominal) {
            et = t.e;
        } else {
            et = msb_index(t.t) + t.e - F;  // true position of the leading one
        }
        if (!have || et > E) { E = et; have = true; }
    }

    bool all_zero_negative = true;
    for (const Term& t : terms) {
        if (!(t.zero && t.neg)) { all_zero_negative = false; break; }
    }
    if (!have) return all_zero_negative ? 0x80000000u : 0u;

    i128 S = 0;
    for (const Term& t : terms) {
        if (t.zero) continue;
        const int sh = E - t.e;  // right shift onto the window grid
        u128 mag = t.t;
        i128 v;
        if (sh <= 0) {
            v = static_cast<i128>(mag << (-sh));
            if (t.neg) v = -v;
        } else if (cfg.trunc == TruncMode::SignMag || !t.neg) {
            const u128 q = sh >= 128 ? 0 : (mag >> sh);
            v = static_cast<i128>(q);
            if (t.neg) v = -v;
        } else {  // Floor for a negative term: -ceil(mag / 2^sh)
            u128 q = sh >= 128 ? 0 : (mag >> sh);
            const bool lost = sh >= 128 ? (mag != 0) : ((mag & ((u128(1) << sh) - 1)) != 0);
            if (lost) q += 1;
            v = -static_cast<i128>(q);
        }
        S += v;
    }

    if (S == 0) return all_zero_negative ? 0x80000000u : 0u;
    const bool neg = S < 0;
    const u128 mag = static_cast<u128>(neg ? -S : S);
    return round_to_fp32(neg, mag, E - F, cfg.round, cfg.overflow_to_inf);
}

uint32_t dot_chain(const FdaConfig& cfg, InFmt fmt, const uint16_t* a, const uint16_t* b,
                   int k_total, uint32_t c) {
    uint32_t acc = c;
    for (int k0 = 0; k0 < k_total; k0 += cfg.block_k) {
        const int n = (k_total - k0) < cfg.block_k ? (k_total - k0) : cfg.block_k;
        acc = fda(cfg, fmt, a + k0, b + k0, n, acc);
    }
    return acc;
}

void mma_tile(const FdaConfig& cfg, InFmt fmt, int M, int N, int K, const uint16_t* A,
              const uint16_t* B, const uint32_t* C, uint32_t* D) {
    std::vector<uint16_t> col(K);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int kk = 0; kk < K; ++kk) col[kk] = B[kk * N + j];
            D[i * N + j] = dot_chain(cfg, fmt, A + i * K, col.data(), K, C[i * N + j]);
        }
    }
}

uint32_t ieee_fma_chain(InFmt fmt, const uint16_t* a, const uint16_t* b, int k, uint32_t c) {
    float acc;
    std::memcpy(&acc, &c, 4);
    for (int i = 0; i < k; ++i) {
        // FP16/BF16 values are exact in float.
        const float x = static_cast<float>(in_to_double(a[i], fmt));
        const float y = static_cast<float>(in_to_double(b[i], fmt));
        acc = std::fmaf(x, y, acc);
    }
    uint32_t out;
    std::memcpy(&out, &acc, 4);
    return out;
}

}  // namespace tcmma
