// Verilator testbench for the MMA block (rtl/tc_mma_core.sv).
//  Phase 1: back-to-back random M x N x K operations; every lane is compared
//           with the golden model (checks operand wiring and the shared
//           pipeline schedule).
//  Phase 2: 16-deep reductions. A 16x16x16-style dot product is computed as
//           16/K chained issues with D fed back as C, and compared with the
//           model's dot_chain (the same chaining the GPU uses).
// Compile-time: TB_M TB_N TB_K TB_F TB_STAGES.  Runtime: +ops=<n> +chains=<n> +seed=<s>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <vector>

#include "Vtc_mma_core.h"
#include "tb_common.hpp"
#include "tcmma/gen.hpp"

using namespace tcmma;

#ifndef TB_M
#define TB_M 4
#endif
#ifndef TB_N
#define TB_N 2
#endif
#ifndef TB_K
#define TB_K 8
#endif
#ifndef TB_F
#define TB_F 24
#endif
#ifndef TB_STAGES
#define TB_STAGES 3
#endif

static uint64_t plusarg_u64(const char* name, uint64_t dflt) {
    const char* s = Verilated::commandArgsPlusMatch(name);
    if (!s || !*s) return dflt;
    const char* eq = std::strchr(s, '=');
    return eq ? std::strtoull(eq + 1, nullptr, 0) : dflt;
}

struct Op {
    InFmt fmt;
    std::vector<uint16_t> A;  // M x K row-major
    std::vector<uint16_t> B;  // K x N row-major
    std::vector<uint32_t> C;  // M x N
};

class Bench {
public:
    explicit Bench(uint64_t seed) : ctx_(new VerilatedContext), top_(new Vtc_mma_core(ctx_.get())), gen_(seed) {
        top_->rst_n = 0;
        top_->in_valid = 0;
        for (int i = 0; i < 5; ++i) tick();
        top_->rst_n = 1;
    }
    void tick() {
        top_->clk = 0; top_->eval();
        top_->clk = 1; top_->eval();
    }
    Op random_op(int k_depth) {
        Op op;
        op.fmt = (gen_.next() & 1) ? InFmt::BF16 : InFmt::FP16;
        const int ncls = static_cast<int>(GenClass::kCount);
        op.A.resize(TB_M * k_depth);
        op.B.resize(k_depth * TB_N);
        op.C.resize(TB_M * TB_N);
        // One generated vector per column n. Column n takes v_n.b, every row m
        // takes v_(m % N).a, so lanes (m, m % N) see a complete class-faithful
        // vector (cancellation, zeros, subnormals...) and the other lanes see
        // cross combinations that test the operand wiring.
        std::vector<FdaVector> vs;
        for (int n = 0; n < TB_N; ++n) {
            const GenClass cls = static_cast<GenClass>(gen_.next() % ncls);
            vs.push_back(gen_.make(cls, op.fmt, k_depth, TB_F));
            for (int k = 0; k < k_depth; ++k) op.B[k * TB_N + n] = vs[n].b[k];
        }
        for (int m = 0; m < TB_M; ++m)
            for (int k = 0; k < k_depth; ++k) op.A[m * k_depth + k] = vs[m % TB_N].a[k];
        for (int m = 0; m < TB_M; ++m)
            for (int n = 0; n < TB_N; ++n)
                op.C[m * TB_N + n] = (n == m % TB_N) ? vs[n].c
                                                     : gen_.make(GenClass::Normal, op.fmt, 1, TB_F).c;
        return op;
    }
    // Drive one K-deep slice [k0, k0+K) of op with accumulator matrix C.
    void drive(const Op& op, int k_depth, int k0, const std::vector<uint32_t>& C) {
        top_->in_valid = 1;
        top_->in_fmt = op.fmt == InFmt::BF16;
        for (int m = 0; m < TB_M; ++m)
            for (int k = 0; k < TB_K; ++k)
                put_bits(top_->a_i, (m * TB_K + k) * 16, 16, op.A[m * k_depth + k0 + k]);
        for (int k = 0; k < TB_K; ++k)
            for (int n = 0; n < TB_N; ++n)
                put_bits(top_->b_i, (k * TB_N + n) * 16, 16, op.B[(k0 + k) * TB_N + n]);
        for (int i = 0; i < TB_M * TB_N; ++i) put_bits(top_->c_i, i * 32, 32, C[i]);
    }
    std::vector<uint32_t> read_d() {
        std::vector<uint32_t> d(TB_M * TB_N);
        for (int i = 0; i < TB_M * TB_N; ++i) d[i] = get32(top_->d_o, i * 32);
        return d;
    }
    // Expected result of a k_depth reduction starting from C.
    std::vector<uint32_t> expect(const FdaConfig& cfg, const Op& op, int k_depth) {
        std::vector<uint32_t> d(TB_M * TB_N);
        mma_tile(cfg, op.fmt, TB_M, TB_N, k_depth, op.A.data(), op.B.data(), op.C.data(), d.data());
        return d;
    }
    Vtc_mma_core* top() { return top_.get(); }
    VecGen& gen() { return gen_; }

private:
    std::unique_ptr<VerilatedContext> ctx_;
    std::unique_ptr<Vtc_mma_core> top_;
    VecGen gen_;
};

static int compare(const char* phase, uint64_t idx, const std::vector<uint32_t>& got,
                   const std::vector<uint32_t>& exp, uint64_t& errors) {
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == exp[i]) continue;
        if (errors < 10)
            std::printf("MISMATCH %s op %llu lane (%zu,%zu): got %s expected %s\n", phase,
                        static_cast<unsigned long long>(idx), i / TB_N, i % TB_N,
                        describe_fp32(got[i]).c_str(), describe_fp32(exp[i]).c_str());
        ++errors;
        ++bad;
    }
    return bad;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    const uint64_t n_ops = plusarg_u64("ops", 20000);
    const uint64_t n_chains = plusarg_u64("chains", 5000);
    Bench tb(plusarg_u64("seed", 7));
    const FdaConfig cfg = rtl_config(TB_K, TB_F);
    Coverage cov;
    uint64_t errors = 0;

    // Phase 1: streaming, one op per cycle.
    std::deque<std::pair<uint64_t, std::vector<uint32_t>>> q;
    uint64_t issued = 0, done = 0;
    while (done < n_ops) {
        if (issued < n_ops) {
            Op op = tb.random_op(TB_K);
            tb.drive(op, TB_K, 0, op.C);
            q.emplace_back(issued, tb.expect(cfg, op, TB_K));
            ++issued;
        } else {
            tb.top()->in_valid = 0;
        }
        tb.tick();
        if (tb.top()->out_valid) {
            auto e = q.front();
            q.pop_front();
            auto got = tb.read_d();
            compare("stream", e.first, got, e.second, errors);
            for (uint32_t d : got) cov.hit(std::string("out.") + out_class(d));
            ++done;
        }
    }
    tb.top()->in_valid = 0;
    for (int i = 0; i < TB_STAGES + 2; ++i) tb.tick();
    if (tb.top()->out_valid) { std::printf("ERROR: spurious out_valid after drain\n"); ++errors; }

    // Phase 2: 16-deep chained reductions (D fed back as C).
    const int depth = 16;
    for (uint64_t t = 0; t < n_chains; ++t) {
        Op op = tb.random_op(depth);
        std::vector<uint32_t> acc = op.C;
        for (int k0 = 0; k0 < depth; k0 += TB_K) {
            tb.drive(op, depth, k0, acc);
            tb.tick();
            tb.top()->in_valid = 0;
            int wait = 0;
            while (!tb.top()->out_valid && wait++ < 2 * TB_STAGES + 4) tb.tick();
            if (!tb.top()->out_valid) { std::printf("ERROR: chain result never arrived\n"); return 1; }
            acc = tb.read_d();
        }
        compare("chain16", t, acc, tb.expect(cfg, op, depth), errors);
        cov.hit("chain16.done");
    }
    tb.top()->final();

    std::printf("tc_mma_core M=%d N=%d K=%d F=%d NUM_STAGES=%d: %llu streamed ops, %llu "
                "16-deep chains, %llu lane mismatches\n",
                TB_M, TB_N, TB_K, TB_F, TB_STAGES, static_cast<unsigned long long>(n_ops),
                static_cast<unsigned long long>(n_chains), static_cast<unsigned long long>(errors));
    const int holes = cov.report({"out.normal", "out.zero", "out.subnormal", "out.inf", "out.nan",
                                  "chain16.done"});
    const bool pass = errors == 0 && holes == 0;
    std::printf("%s\n", pass ? "TEST PASSED" : "TEST FAILED");
    return pass ? 0 : 1;
}
