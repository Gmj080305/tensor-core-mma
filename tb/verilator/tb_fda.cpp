// Verilator testbench for one fused dot-add lane (rtl/tc_fda.sv).
// Every issued operation is predicted by the golden model and compared bit
// for bit when it leaves the pipeline. Random valid gaps exercise the
// pipeline; latency is checked against NUM_STAGES.
//
// Compile-time parameters (must match the -G values given to Verilator):
//   TB_K, TB_F, TB_STAGES        Runtime: +ops=<n> +seed=<s>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>

#include "Vtc_fda.h"
#include "tb_common.hpp"
#include "tcmma/gen.hpp"

using namespace tcmma;

#ifndef TB_K
#define TB_K 8
#endif
#ifndef TB_F
#define TB_F 24
#endif
#ifndef TB_STAGES
#define TB_STAGES 3
#endif

struct Pending {
    uint64_t issue_cycle;
    uint32_t expect;
    InFmt fmt;
    GenClass cls;
    FdaVector v;
};

static uint64_t plusarg_u64(const char* name, uint64_t dflt) {
    const char* s = Verilated::commandArgsPlusMatch(name);
    if (!s || !*s) return dflt;
    const char* eq = std::strchr(s, '=');
    return eq ? std::strtoull(eq + 1, nullptr, 0) : dflt;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const uint64_t n_ops = plusarg_u64("ops", 200000);
    const uint64_t seed = plusarg_u64("seed", 1);

    auto ctx = std::make_unique<VerilatedContext>();
    auto top = std::make_unique<Vtc_fda>(ctx.get());
    const FdaConfig cfg = rtl_config(TB_K, TB_F);
    VecGen gen(seed);
    Coverage cov;
    std::deque<Pending> q;
    uint64_t cycle = 0, issued = 0, checked = 0, errors = 0, ieee_diff = 0;

    auto tick = [&]() {
        top->clk = 0; top->eval();
        top->clk = 1; top->eval();
        ++cycle;
    };

    top->rst_n = 0;
    top->in_valid = 0;
    for (int i = 0; i < 5; ++i) tick();
    top->rst_n = 1;

    const int ncls = static_cast<int>(GenClass::kCount);
    while (checked < n_ops) {
        // ---- drive (inputs settle before the rising edge) ----
        const bool issue = issued < n_ops && (gen.next() % 100) < 85;
        top->in_valid = issue;
        if (issue) {
            Pending p;
#ifdef TB_FP16_ONLY
            p.fmt = InFmt::FP16;
            gen.next();
#else
            p.fmt = (gen.next() & 1) ? InFmt::BF16 : InFmt::FP16;
#endif
            p.cls = static_cast<GenClass>(issued % ncls);
            p.v = gen.make(p.cls, p.fmt, TB_K, TB_F);
            top->in_fmt = p.fmt == InFmt::BF16;
            for (int i = 0; i < TB_K; ++i) {
                put_bits(top->a_i, 16 * i, 16, p.v.a[i]);
                put_bits(top->b_i, 16 * i, 16, p.v.b[i]);
            }
            top->c_i = p.v.c;
            p.expect = fda(cfg, p.fmt, p.v.a.data(), p.v.b.data(), TB_K, p.v.c);
            p.issue_cycle = cycle;
            q.push_back(p);
            ++issued;
        }
        tick();

        // ---- check outputs visible after this edge ----
        if (top->out_valid) {
            if (q.empty()) {
                std::printf("ERROR: out_valid with nothing in flight at cycle %llu\n",
                            static_cast<unsigned long long>(cycle));
                return 1;
            }
            Pending p = q.front();
            q.pop_front();
            const uint32_t got = top->d_o;
            const uint64_t lat = cycle - p.issue_cycle;
            bool ok = got == p.expect;
            if (lat != TB_STAGES) {
                std::printf("ERROR: latency %llu, expected %d\n",
                            static_cast<unsigned long long>(lat), TB_STAGES);
                ok = false;
            }
            if (!ok) {
                if (errors < 10) {
                    std::printf("MISMATCH #%llu (%s, %s): got %s expected %s\n  c=%s\n",
                                static_cast<unsigned long long>(checked), fmt_name(p.fmt),
                                gen_class_name(p.cls), describe_fp32(got).c_str(),
                                describe_fp32(p.expect).c_str(), hex32(p.v.c).c_str());
                    for (int i = 0; i < TB_K; ++i)
                        std::printf("  a[%d]=%s b[%d]=%s\n", i, hex16(p.v.a[i]).c_str(), i,
                                    hex16(p.v.b[i]).c_str());
                }
                ++errors;
            }
            const std::string fm = fmt_name(p.fmt);
            cov.hit("in." + fm + "." + gen_class_name(p.cls));
            cov.hit(std::string("out.") + out_class(got));
            if (ieee_fma_chain(p.fmt, p.v.a.data(), p.v.b.data(), TB_K, p.v.c) != p.expect &&
                !(fp32_is_nan(p.expect)))
                ++ieee_diff;
            ++checked;
        }
        if (cycle > 50 * n_ops + 1000) {
            std::printf("ERROR: timeout\n");
            return 1;
        }
    }
    top->final();

    std::printf("tc_fda K=%d F=%d NUM_STAGES=%d: %llu ops checked, %llu mismatches\n", TB_K, TB_F,
                TB_STAGES, static_cast<unsigned long long>(checked),
                static_cast<unsigned long long>(errors));
    std::printf("  %llu results (%.1f%%) differ from an IEEE FP32 FMA chain: the non-IEEE "
                "behaviour is being exercised\n",
                static_cast<unsigned long long>(ieee_diff), 100.0 * ieee_diff / checked);
#ifdef TB_FP16_ONLY
    const int holes = cov.report({"out.normal", "out.subnormal", "out.zero", "out.inf", "out.nan",
                                  "in.fp16.boundary", "in.fp16.cancel", "in.fp16.subnormal"});
#else
    const int holes = cov.report({"out.normal", "out.subnormal", "out.zero", "out.inf", "out.nan",
                                  "in.fp16.boundary", "in.bf16.boundary", "in.fp16.cancel",
                                  "in.bf16.subnormal", "in.bf16.overflow"});
#endif
    const bool pass = errors == 0 && holes == 0;
    std::printf("%s\n", pass ? "TEST PASSED" : "TEST FAILED");
    return pass ? 0 : 1;
}
