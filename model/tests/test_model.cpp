// Unit tests for the golden model. Run: build/test_model
// Each test either checks an exact bit pattern derived by hand in
// docs/01_numerics_spec.md, or checks the model against an independent
// exact computation.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "tcmma/fda.hpp"
#include "tcmma/formats.hpp"
#include "tcmma/gen.hpp"

using namespace tcmma;

static int g_fail = 0, g_pass = 0;

#define CHECK_EQ_HEX(got, exp, what)                                                    \
    do {                                                                                \
        uint32_t g_ = (got), e_ = (exp);                                                \
        if (g_ != e_) {                                                                 \
            std::printf("FAIL %s: got %s expected %s (%s:%d)\n", what,                  \
                        describe_fp32(g_).c_str(), describe_fp32(e_).c_str(), __FILE__, \
                        __LINE__);                                                      \
            ++g_fail;                                                                   \
        } else {                                                                        \
            ++g_pass;                                                                   \
        }                                                                               \
    } while (0)

#define CHECK(cond, what)                                                   \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("FAIL %s (%s:%d)\n", what, __FILE__, __LINE__);     \
            ++g_fail;                                                       \
        } else {                                                            \
            ++g_pass;                                                       \
        }                                                                   \
    } while (0)

static uint16_t h(double x) { return double_to_in(x, InFmt::FP16); }
static uint16_t bf(double x) { return double_to_in(x, InFmt::BF16); }
static uint32_t f32(double x) { return double_to_fp32(x); }

// Pad to a full block with zero products.
static uint32_t run(const FdaConfig& cfg, InFmt fmt, std::vector<uint16_t> a,
                    std::vector<uint16_t> b, uint32_t c, int k_total = -1) {
    if (k_total < 0) k_total = cfg.block_k;
    a.resize(k_total, 0);
    b.resize(k_total, 0);
    return dot_chain(cfg, fmt, a.data(), b.data(), k_total, c);
}

static void test_format_roundtrip() {
    int bad = 0;
    for (uint32_t x = 0; x < 65536; ++x) {
        for (InFmt f : {InFmt::FP16, InFmt::BF16}) {
            const double v = in_to_double(static_cast<uint16_t>(x), f);
            if (std::isnan(v)) continue;
            if (double_to_in(v, f) != x) ++bad;
        }
    }
    CHECK(bad == 0, "fp16/bf16 decode->encode round trip for all 2^16 patterns");
}

static void test_round_to_fp32() {
    std::mt19937_64 rng(1);
    int bad_rne = 0, bad_rz = 0;
    for (int i = 0; i < 200000; ++i) {
        const uint64_t mag = rng() >> (11 + rng() % 40);  // < 2^53: exact in double
        if (!mag) continue;
        const int lsb = static_cast<int>(rng() % 400) - 250;
        const bool neg = rng() & 1;
        const double v = std::ldexp(static_cast<double>(mag), lsb) * (neg ? -1 : 1);
        const uint32_t want_rne = double_to_fp32(v);
        if (round_to_fp32(neg, mag, lsb, RoundMode::RNE, true) != want_rne) ++bad_rne;
        // RZ reference: the RNE result, stepped toward zero if it overshot.
        float fr;
        std::memcpy(&fr, &want_rne, 4);
        double r = fr;
        uint32_t want_rz = want_rne;
        if (std::fabs(r) > std::fabs(v) || std::isinf(r)) {
            float t = std::nextafterf(fr, 0.0f);
            std::memcpy(&want_rz, &t, 4);
        }
        // RZ with overflow_to_inf=false saturates like IEEE.
        if (round_to_fp32(neg, mag, lsb, RoundMode::RZ, false) != want_rz) ++bad_rz;
    }
    CHECK(bad_rne == 0, "round_to_fp32 RNE matches double->float conversion");
    CHECK(bad_rz == 0, "round_to_fp32 RZ matches truncation toward zero");
}

// Independent exact computation: when no bits fall out of the window, the FDA
// must equal the exactly rounded (RZ) sum.
static void test_exact_when_window_is_wide() {
    FdaConfig cfg;
    cfg.align_frac = 60;
    VecGen gen(7);
    int bad = 0;
    for (int n = 0; n < 50000; ++n) {
        const InFmt fmt = (n & 1) ? InFmt::BF16 : InFmt::FP16;
        FdaVector v = gen.make(GenClass::Normal, fmt, 8);
        // exact: sum of sig*2^(lsb), using the smallest lsb as the common grid
        struct T { bool neg; unsigned __int128 m; int lsb; };
        std::vector<T> ts;
        for (int i = 0; i < 8; ++i) {
            Decoded x = decode_in(v.a[i], fmt), y = decode_in(v.b[i], fmt);
            ts.push_back({bool(x.sign ^ y.sign), (unsigned __int128)x.sig * y.sig, x.exp + y.exp - 20});
        }
        Decoded z = decode_fp32(v.c);
        ts.push_back({z.sign, z.sig, z.exp - 23});
        int lo = 1 << 30;
        for (auto& t : ts) if (t.m) lo = std::min(lo, t.lsb);
        __int128 S = 0;
        for (auto& t : ts) {
            if (!t.m) continue;
            __int128 m = (__int128)(t.m << (t.lsb - lo));
            S += t.neg ? -m : m;
        }
        uint32_t want = S == 0 ? 0u
                               : round_to_fp32(S < 0, (unsigned __int128)(S < 0 ? -S : S), lo,
                                               RoundMode::RZ, true);
        if (fda(cfg, fmt, v.a.data(), v.b.data(), 8, v.c) != want) ++bad;
    }
    CHECK(bad == 0, "FDA with a 60-bit window equals the exact RZ sum (50k vectors)");
}

static void test_order_independence() {
    FdaConfig cfg;
    VecGen gen(11);
    std::mt19937_64 rng(3);
    int bad = 0;
    for (int n = 0; n < 20000; ++n) {
        const InFmt fmt = (n & 1) ? InFmt::BF16 : InFmt::FP16;
        FdaVector v = gen.make(static_cast<GenClass>(n % int(GenClass::kCount)), fmt, 8);
        const uint32_t r0 = fda(cfg, fmt, v.a.data(), v.b.data(), 8, v.c);
        std::vector<int> p{0, 1, 2, 3, 4, 5, 6, 7};
        std::shuffle(p.begin(), p.end(), rng);
        std::vector<uint16_t> a2(8), b2(8);
        for (int i = 0; i < 8; ++i) { a2[i] = v.a[p[i]]; b2[i] = v.b[p[i]]; }
        const uint32_t r1 = fda(cfg, fmt, a2.data(), b2.data(), 8, v.c);
        if (r0 != r1 && !(fp32_is_nan(r0) && fp32_is_nan(r1))) ++bad;
    }
    CHECK(bad == 0, "FDA result is independent of product order within a block");
}

// Hand-derived expectations for each probe family under the default config.
static void test_probe_expectations() {
    const InFmt F16 = InFmt::FP16;
    FdaConfig d;  // defaults: block 8, F=24, nominal, signmag, rz, ovf->inf

    // align_bits: 1 - 1 + 2^-n survives iff n <= F
    for (int n = 20; n <= 30; ++n) {
        const uint32_t got = run(d, F16, {h(1), h(1), h(std::ldexp(1, -12))},
                                 {h(1), h(-1), h(std::ldexp(1, -(n - 12)))}, 0);
        const uint32_t want = n <= 24 ? f32(std::ldexp(1, -n)) : 0u;
        char what[64];
        std::snprintf(what, sizeof what, "align_bits n=%d", n);
        CHECK_EQ_HEX(got, want, what);
    }
    // trunc_mode: 1 + (-2^-30): signmag -> 1.0, floor -> 1 - 2^-24
    {
        std::vector<uint16_t> a{h(1), h(-std::ldexp(1, -15))}, b{h(1), h(std::ldexp(1, -15))};
        CHECK_EQ_HEX(run(d, F16, a, b, 0), 0x3F800000u, "trunc_mode signmag");
        FdaConfig fl = d;
        fl.trunc = TruncMode::Floor;
        CHECK_EQ_HEX(run(fl, F16, a, b, 0), 0x3F7FFFFFu, "trunc_mode floor");
    }
    // final_round: 1 + 2^-23 + 2^-24 -> rz 1+2^-23, rne 1+2^-22
    {
        std::vector<uint16_t> a{h(1), h(std::ldexp(1, -12)), h(std::ldexp(1, -12))};
        std::vector<uint16_t> b{h(1), h(std::ldexp(1, -11)), h(std::ldexp(1, -12))};
        CHECK_EQ_HEX(run(d, F16, a, b, 0), 0x3F800001u, "final_round rz");
        FdaConfig r = d;
        r.round = RoundMode::RNE;
        CHECK_EQ_HEX(run(r, F16, a, b, 0), 0x3F800002u, "final_round rne");
    }
    // block_size: [1, -1] at k=0,1 and 2^-30 at k=j (16-long dot product)
    for (int j = 2; j < 16; ++j) {
        std::vector<uint16_t> a(16, 0), b(16, 0);
        a[0] = h(1); b[0] = h(1);
        a[1] = h(1); b[1] = h(-1);
        a[j] = h(std::ldexp(1, -15)); b[j] = h(std::ldexp(1, -15));
        const uint32_t got = run(d, F16, a, b, 0, 16);
        const uint32_t want = j < 8 ? 0u : f32(std::ldexp(1, -30));
        char what[64];
        std::snprintf(what, sizeof what, "block_size j=%d", j);
        CHECK_EQ_HEX(got, want, what);
    }
    // zero_in_emax: +0 * 65504 has nominal exponent 1, would push E from 0 to 1
    {
        std::vector<uint16_t> a{h(0), h(1), h(1), h(std::ldexp(1, -12))};
        std::vector<uint16_t> b{h(65504), h(1), h(-1), h(std::ldexp(1, -12))};
        CHECK_EQ_HEX(run(d, F16, a, b, 0), f32(std::ldexp(1, -24)), "zero_in_emax=0");
        FdaConfig z = d;
        z.zero_in_emax = true;
        CHECK_EQ_HEX(run(z, F16, a, b, 0), 0u, "zero_in_emax=1");
    }
    // align_ref_product_msb: 1.5*1.5 - 1.5*1.5 + 2^-24
    {
        std::vector<uint16_t> a{h(1.5), h(1.5), h(std::ldexp(1, -12))};
        std::vector<uint16_t> b{h(1.5), h(-1.5), h(std::ldexp(1, -12))};
        CHECK_EQ_HEX(run(d, F16, a, b, 0), f32(std::ldexp(1, -24)), "align_ref nominal");
        FdaConfig ac = d;
        ac.align_ref = AlignRef::Actual;
        CHECK_EQ_HEX(run(ac, F16, a, b, 0), 0u, "align_ref actual (2.25 has msb at 2^1)");
    }
    // align_ref_subnormal: (2^-24)*(2^14) has nominal exponent 0 but value 2^-10
    {
        std::vector<uint16_t> a{h(std::ldexp(1, -24)), h(std::ldexp(1, -15))};
        std::vector<uint16_t> b{h(16384), h(std::ldexp(1, -15))};
        CHECK_EQ_HEX(run(d, F16, a, b, 0), f32(std::ldexp(1, -10)), "align_ref subnormal nominal");
        FdaConfig ac = d;
        ac.align_ref = AlignRef::Actual;
        CHECK_EQ_HEX(run(ac, F16, a, b, 0), f32(std::ldexp(1, -10) + std::ldexp(1, -30)),
                     "align_ref subnormal actual");
    }
}

static void test_non_monotonic() {
    // Increasing c from 2-2^-23 to 2 lowers the result (Mikaitis 2024 effect).
    FdaConfig d;
    std::vector<uint16_t> a(8, h(std::ldexp(1, -12))), b(8, h(std::ldexp(1, -12)));
    const uint32_t lo = fda(d, InFmt::FP16, a.data(), b.data(), 8, f32(2.0 - std::ldexp(1, -23)));
    const uint32_t hi = fda(d, InFmt::FP16, a.data(), b.data(), 8, f32(2.0));
    CHECK_EQ_HEX(lo, 0x40000001u, "non-monotonic: c=2-2^-23 gives 2+2^-22");
    CHECK_EQ_HEX(hi, 0x40000000u, "non-monotonic: c=2 gives 2");
}

static void test_specials() {
    FdaConfig d;
    const InFmt F16 = InFmt::FP16;
    CHECK(fp32_is_nan(run(d, F16, {0x7C00}, {h(0)}, 0)), "inf * 0 -> NaN");
    CHECK(fp32_is_nan(run(d, F16, {0x7C00}, {h(1)}, 0xFF800000u)), "+inf + -inf -> NaN");
    CHECK(fp32_is_nan(run(d, F16, {0x7E00}, {h(1)}, 0)), "NaN input -> NaN");
    CHECK_EQ_HEX(run(d, F16, {0x7C00}, {h(-2)}, f32(5)), 0xFF800000u, "inf * -2 -> -inf");
    CHECK_EQ_HEX(run(d, F16, std::vector<uint16_t>(8, h(-0.0)), std::vector<uint16_t>(8, h(1)), 0x80000000u),
                 0x80000000u, "all products -0 and c -0 -> -0");
    CHECK_EQ_HEX(run(d, F16, {h(-0.0)}, {h(1)}, 0u), 0u, "-0 + +0 -> +0");
    CHECK_EQ_HEX(run(d, F16, {h(3), h(3)}, {h(5), h(-5)}, 0x80000000u), 0u, "x - x -> +0");
}

static void test_overflow_and_subnormal_out() {
    FdaConfig d;
    const InFmt B = InFmt::BF16;
    std::vector<uint16_t> a{bf(std::ldexp(1, 127)), bf(std::ldexp(1, 127))};
    std::vector<uint16_t> b{bf(1.5), bf(1.5)};
    CHECK_EQ_HEX(run(d, B, a, b, 0), 0x7F800000u, "overflow -> inf (overflow_to_inf=1)");
    FdaConfig s = d;
    s.overflow_to_inf = false;
    CHECK_EQ_HEX(run(s, B, a, b, 0), 0x7F7FFFFFu, "overflow -> max finite (overflow_to_inf=0)");
    CHECK_EQ_HEX(run(d, B, {bf(std::ldexp(1, -70))}, {bf(std::ldexp(1, -70))}, 0), 0x00000200u,
                 "2^-140 -> FP32 subnormal 0x200");
    CHECK_EQ_HEX(run(d, B, {bf(-std::ldexp(1, -100))}, {bf(std::ldexp(1, -100))}, 0), 0x80000000u,
                 "-2^-200 underflows to -0 under RZ");
}

static void report_vs_ieee() {
    FdaConfig d;
    VecGen gen(5);
    int diff = 0, n = 20000;
    for (int i = 0; i < n; ++i) {
        FdaVector v = gen.make(GenClass::Normal, InFmt::FP16, 8);
        if (fda(d, InFmt::FP16, v.a.data(), v.b.data(), 8, v.c) !=
            ieee_fma_chain(InFmt::FP16, v.a.data(), v.b.data(), 8, v.c))
            ++diff;
    }
    std::printf("info: tensor-core FDA differs from an IEEE FP32 FMA chain in %d of %d random "
                "FP16 dot products (%.1f%%)\n", diff, n, 100.0 * diff / n);
}

int main() {
    test_format_roundtrip();
    test_round_to_fp32();
    test_exact_when_window_is_wide();
    test_order_independence();
    test_probe_expectations();
    test_non_monotonic();
    test_specials();
    test_overflow_and_subnormal_out();
    report_vs_ieee();
    std::printf("%d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
