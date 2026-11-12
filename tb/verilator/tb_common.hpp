// Helpers shared by the Verilator testbenches.
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>

#include "tcmma/fda.hpp"
#include "tcmma/formats.hpp"
#include "verilated.h"

// Verilator maps ports of 33..64 bits to QData and wider ports to VlWide.
inline void put_bits(QData& sig, int lsb, int width, uint32_t v) {
    const QData mask = ((QData(1) << width) - 1) << lsb;
    sig = (sig & ~mask) | ((QData(v) << lsb) & mask);
}
template <std::size_t W>
inline void put_bits(VlWide<W>& sig, int lsb, int width, uint32_t v) {
    for (int b = 0; b < width; ++b) {
        const int bit = lsb + b;
        EData& word = sig[bit / 32];
        const EData m = EData(1) << (bit % 32);
        if ((v >> b) & 1) word |= m; else word &= ~m;
    }
}
inline void put_bits(IData& sig, int lsb, int width, uint32_t v) {
    const uint64_t mask = ((uint64_t(1) << width) - 1) << lsb;
    sig = static_cast<IData>((sig & ~mask) | ((uint64_t(v) << lsb) & mask));
}
inline uint32_t get32(const QData& sig, int lsb) { return static_cast<uint32_t>(sig >> lsb); }
template <std::size_t W>
inline uint32_t get32(const VlWide<W>& sig, int lsb) {
    uint32_t v = 0;
    for (int b = 0; b < 32; ++b) {
        const int bit = lsb + b;
        if ((sig[bit / 32] >> (bit % 32)) & 1) v |= (1u << b);
    }
    return v;
}
inline uint32_t get32(const IData& sig, int lsb) { return lsb == 0 ? sig : 0; }

// Output-class coverage for FP32 results.
inline const char* out_class(uint32_t d) {
    const uint32_t e = (d >> 23) & 0xFF, f = d & 0x7FFFFF;
    if (e == 0xFF) return f ? "nan" : "inf";
    if (e == 0) return f ? "subnormal" : "zero";
    return "normal";
}

struct Coverage {
    std::map<std::string, uint64_t> bins;
    void hit(const std::string& b) { ++bins[b]; }
    // Print all bins; return the number of required bins that were never hit.
    int report(const std::initializer_list<std::string>& required) const {
        std::printf("coverage:\n");
        for (const auto& kv : bins) std::printf("  %-34s %10llu\n", kv.first.c_str(),
                                                static_cast<unsigned long long>(kv.second));
        int holes = 0;
        for (const auto& r : required) {
            auto it = bins.find(r);
            if (it == bins.end() || it->second == 0) {
                std::printf("  HOLE: required bin '%s' never hit\n", r.c_str());
                ++holes;
            }
        }
        return holes;
    }
};
