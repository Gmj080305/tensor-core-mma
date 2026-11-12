// tcmma: command-line front end for the golden model.
//
//   tcmma eval --in probes.bin --out model.out [key=value ...] [--ieee]
//       Compute every tile in a probe file with the model (or with an IEEE FP32
//       FMA chain if --ieee) and write a result file.
//   tcmma fit  --in probes.bin --gpu gpu.out [--families f.txt] [--top N]
//       Sweep the parameter grid and rank configurations by how many output
//       elements disagree with the GPU. This is how the hypothesis in
//       docs/01_numerics_spec.md is confirmed or replaced.
//   tcmma one  --fmt fp16|bf16 --a h0,h1,.. --b h0,h1,.. --c hex [key=value ...]
//       Evaluate one dot product (chained in blocks) and print the result.
//   tcmma demo
//       Print worked examples (non-monotonic sum, difference from IEEE).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tcmma/fda.hpp"
#include "tcmma/formats.hpp"
#include "tcmma/probe_io.hpp"

using namespace tcmma;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  tcmma eval --in probes.bin --out model.out [key=value ...] [--ieee]\n"
                 "  tcmma fit  --in probes.bin --gpu gpu.out [--families f.txt] [--top N]\n"
                 "  tcmma one  --fmt fp16|bf16 --a h,h,.. --b h,h,.. --c hex [key=value ...]\n"
                 "  tcmma demo\n"
                 "config keys: block_k align_frac align_ref(nominal|actual) zero_in_emax(0|1)\n"
                 "             trunc(signmag|floor) round(rz|rne) overflow_to_inf(0|1) daz(0|1)\n");
    return 2;
}

struct Args {
    std::map<std::string, std::string> opt;
    std::vector<std::string> kv;
    bool flag(const std::string& k) const { return opt.count(k) != 0; }
    std::string get(const std::string& k, const std::string& d = "") const {
        auto it = opt.find(k);
        return it == opt.end() ? d : it->second;
    }
};

Args parse(int argc, char** argv, int start) {
    Args a;
    for (int i = start; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0) {
            std::string key = s.substr(2);
            if (key == "ieee" || key == "strict-nan") a.opt[key] = "1";
            else if (i + 1 < argc) a.opt[key] = argv[++i];
            else a.opt[key] = "";
        } else if (s.find('=') != std::string::npos) {
            a.kv.push_back(s);
        }
    }
    return a;
}

bool apply_kv(FdaConfig& cfg, const std::vector<std::string>& kv) {
    for (const auto& s : kv) {
        const auto p = s.find('=');
        if (!cfg.set(s.substr(0, p), s.substr(p + 1))) {
            std::fprintf(stderr, "bad config setting: %s\n", s.c_str());
            return false;
        }
    }
    return true;
}

bool same(uint32_t x, uint32_t y, bool strict_nan) {
    if (x == y) return true;
    return !strict_nan && fp32_is_nan(x) && fp32_is_nan(y);
}

void eval_all(const FdaConfig& cfg, const ProbeSet& ps, bool ieee, std::vector<uint32_t>& D) {
    const uint32_t M = ps.M, N = ps.N, K = ps.K, n = ps.count();
    D.assign(size_t(n) * M * N, 0);
    std::vector<uint16_t> col(K);
    for (uint32_t t = 0; t < n; ++t) {
        const uint16_t* A = &ps.A[size_t(t) * M * K];
        const uint16_t* B = &ps.B[size_t(t) * K * N];
        const uint32_t* C = &ps.C[size_t(t) * M * N];
        uint32_t* Dt = &D[size_t(t) * M * N];
        if (!ieee) {
            mma_tile(cfg, ps.fmt, M, N, K, A, B, C, Dt);
            continue;
        }
        for (uint32_t i = 0; i < M; ++i)
            for (uint32_t j = 0; j < N; ++j) {
                for (uint32_t k = 0; k < K; ++k) col[k] = B[k * N + j];
                Dt[i * N + j] = ieee_fma_chain(ps.fmt, A + i * K, col.data(), K, C[i * N + j]);
            }
    }
}

int cmd_eval(const Args& a) {
    if (!a.flag("in") || !a.flag("out")) return usage();
    FdaConfig cfg;
    if (!apply_kv(cfg, a.kv)) return 2;
    ProbeSet ps;
    std::string err;
    if (!read_probes(a.get("in"), ps, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    ResultSet rs;
    rs.M = ps.M;
    rs.N = ps.N;
    eval_all(cfg, ps, a.flag("ieee"), rs.D);
    if (!write_results(a.get("out"), rs, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::printf("evaluated %u tiles (%s) with %s\n", ps.count(), fmt_name(ps.fmt),
                a.flag("ieee") ? "IEEE FP32 FMA chain" : cfg.describe().c_str());
    return 0;
}

int cmd_fit(const Args& a) {
    if (!a.flag("in") || !a.flag("gpu")) return usage();
    ProbeSet ps;
    ResultSet gpu;
    std::string err;
    if (!read_probes(a.get("in"), ps, err) || !read_results(a.get("gpu"), gpu, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (gpu.count() != ps.count() || gpu.M != ps.M || gpu.N != ps.N) {
        std::fprintf(stderr, "probe and GPU result files do not match in shape/count\n");
        return 1;
    }
    std::vector<Family> fams;
    if (a.flag("families") && !read_families(a.get("families"), fams, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const bool strict = a.flag("strict-nan");
    const int top = std::atoi(a.get("top", "10").c_str());
    const size_t per_tile = size_t(ps.M) * ps.N;

    struct Score {
        FdaConfig cfg;
        size_t bad;
        std::vector<size_t> fam_bad;
    };
    std::vector<Score> scores;
    std::vector<uint32_t> D;
    for (int bk : {4, 8, 16})
        for (int af = 23; af <= 28; ++af)
            for (AlignRef ar : {AlignRef::Nominal, AlignRef::Actual})
                for (bool zim : {false, true})
                    for (TruncMode tm : {TruncMode::SignMag, TruncMode::Floor})
                        for (RoundMode rm : {RoundMode::RZ, RoundMode::RNE})
                            for (bool oi : {true, false}) {
                                if (ar == AlignRef::Actual && zim) continue;  // not meaningful
                                FdaConfig c;
                                c.block_k = bk; c.align_frac = af; c.align_ref = ar;
                                c.zero_in_emax = zim; c.trunc = tm; c.round = rm;
                                c.overflow_to_inf = oi;
                                eval_all(c, ps, false, D);
                                Score s{c, 0, std::vector<size_t>(fams.size(), 0)};
                                for (size_t i = 0; i < D.size(); ++i) {
                                    if (same(D[i], gpu.D[i], strict)) continue;
                                    ++s.bad;
                                    const uint32_t tile = static_cast<uint32_t>(i / per_tile);
                                    for (size_t f = 0; f < fams.size(); ++f)
                                        if (tile >= fams[f].first && tile < fams[f].first + fams[f].num)
                                            ++s.fam_bad[f];
                                }
                                scores.push_back(std::move(s));
                            }
    std::stable_sort(scores.begin(), scores.end(),
                     [](const Score& x, const Score& y) { return x.bad < y.bad; });
    const size_t total = gpu.D.size();
    std::printf("%zu configurations tried against %zu GPU output elements (%s)\n", scores.size(),
                total, fmt_name(ps.fmt));
    std::printf("rank  mismatches  configuration\n");
    for (int i = 0; i < top && i < static_cast<int>(scores.size()); ++i)
        std::printf("%4d  %10zu  %s\n", i + 1, scores[i].bad, scores[i].cfg.describe().c_str());
    if (!fams.empty()) {
        std::printf("\nper-family mismatches for the best configuration:\n");
        for (size_t f = 0; f < fams.size(); ++f)
            std::printf("  %-28s %8zu / %zu\n", fams[f].name.c_str(), scores[0].fam_bad[f],
                        size_t(fams[f].num) * per_tile);
    }
    if (scores[0].bad == 0)
        std::printf("\nRESULT: exact match. Use this configuration for the RTL sign-off.\n");
    else
        std::printf("\nRESULT: no configuration in the grid matches exactly; inspect the families "
                    "with mismatches (scripts/compare_dumps.py) and extend the model.\n");
    return 0;
}

std::vector<uint16_t> parse_list16(const std::string& s) {
    std::vector<uint16_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(static_cast<uint16_t>(std::stoul(tok, nullptr, 16)));
    return v;
}

int cmd_one(const Args& a) {
    InFmt fmt;
    if (!parse_fmt(a.get("fmt", "fp16"), fmt) || !a.flag("a") || !a.flag("b")) return usage();
    FdaConfig cfg;
    if (!apply_kv(cfg, a.kv)) return 2;
    auto va = parse_list16(a.get("a")), vb = parse_list16(a.get("b"));
    if (va.size() != vb.size()) { std::fprintf(stderr, "--a and --b lengths differ\n"); return 2; }
    const uint32_t c = static_cast<uint32_t>(std::stoul(a.get("c", "0"), nullptr, 16));
    const uint32_t d = dot_chain(cfg, fmt, va.data(), vb.data(), static_cast<int>(va.size()), c);
    const uint32_t ref = ieee_fma_chain(fmt, va.data(), vb.data(), static_cast<int>(va.size()), c);
    std::printf("config     : %s\n", cfg.describe().c_str());
    std::printf("tensor core: %s\n", describe_fp32(d).c_str());
    std::printf("IEEE chain : %s\n", describe_fp32(ref).c_str());
    return 0;
}

int cmd_demo() {
    FdaConfig cfg;
    const InFmt F16 = InFmt::FP16;
    std::vector<uint16_t> a(8, double_to_in(std::ldexp(1, -12), F16)), b = a;
    const uint32_t c_lo = double_to_fp32(2.0 - std::ldexp(1, -23)), c_hi = double_to_fp32(2.0);
    std::printf("Non-monotonic accumulation (Mikaitis 2024) under %s\n", cfg.describe().c_str());
    std::printf("  8 products of 2^-24, c = %s -> d = %s\n", describe_fp32(c_lo).c_str(),
                describe_fp32(fda(cfg, F16, a.data(), b.data(), 8, c_lo)).c_str());
    std::printf("  8 products of 2^-24, c = %s -> d = %s   (larger c, smaller d)\n",
                describe_fp32(c_hi).c_str(), describe_fp32(fda(cfg, F16, a.data(), b.data(), 8, c_hi)).c_str());
    std::printf("  IEEE FMA chain for the first case gives %s\n",
                describe_fp32(ieee_fma_chain(F16, a.data(), b.data(), 8, c_lo)).c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    const Args a = parse(argc, argv, 2);
    try {
        if (cmd == "eval") return cmd_eval(a);
        if (cmd == "fit") return cmd_fit(a);
        if (cmd == "one") return cmd_one(a);
        if (cmd == "demo") return cmd_demo();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return usage();
}
