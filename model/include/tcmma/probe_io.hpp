// Binary file formats shared by the probe generator (Python), the CUDA probe
// runner and the model CLI. All integers are little-endian.
//
// Probe file (*.bin):
//   char[4] "TCMP", u32 version=1, u32 fmt (0=fp16, 1=bf16), u32 count,
//   u32 M, u32 N, u32 K,
//   then `count` cases: A[M*K] u16 (row-major), B[K*N] u16 (row-major),
//                       C[M*N] u32 (row-major)
// Result file (*.out):
//   char[4] "TCMD", u32 version=1, u32 count, u32 M, u32 N,
//   then `count` tiles: D[M*N] u32 (row-major)
// Family file (*.families.txt): one line per family: "<name> <first_case> <num_cases>"
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tcmma/formats.hpp"

namespace tcmma {

struct ProbeSet {
    InFmt fmt = InFmt::FP16;
    uint32_t M = 16, N = 16, K = 16;
    std::vector<uint16_t> A;  // count * M * K
    std::vector<uint16_t> B;  // count * K * N
    std::vector<uint32_t> C;  // count * M * N
    uint32_t count() const { return (M && K) ? static_cast<uint32_t>(A.size() / (M * K)) : 0; }
};

struct ResultSet {
    uint32_t M = 16, N = 16;
    std::vector<uint32_t> D;  // count * M * N
    uint32_t count() const { return (M && N) ? static_cast<uint32_t>(D.size() / (M * N)) : 0; }
};

struct Family {
    std::string name;
    uint32_t first = 0;
    uint32_t num = 0;
};

bool read_probes(const std::string& path, ProbeSet& out, std::string& err);
bool write_probes(const std::string& path, const ProbeSet& ps, std::string& err);
bool read_results(const std::string& path, ResultSet& out, std::string& err);
bool write_results(const std::string& path, const ResultSet& rs, std::string& err);
bool read_families(const std::string& path, std::vector<Family>& out, std::string& err);

}  // namespace tcmma
