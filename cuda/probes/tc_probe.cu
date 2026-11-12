// tc_probe: run every 16x16x16 tile of a probe file through the GPU tensor
// cores and dump the FP32 results bit-exactly.
//
//   tc_probe --in probes/fp16.bin --out results/fp16.gpu.out [--api wmma|mma]
//
//   --api wmma : nvcuda::wmma m16n16k16, row-major A and B (default)
//   --api mma  : PTX mma.sync.aligned.m16n8k16 (two n-halves per tile)
//
// Before the probes it runs a layout self-check: random tiles are compared
// with a double-precision reference within a loose tolerance, which catches
// fragment-layout bugs (a wrong layout gives errors of order 1, not 1e-7).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../common/cuda_check.cuh"
#include "tcmma/formats.hpp"
#include "tcmma/probe_io.hpp"

using namespace tcmma;

constexpr int T = 16;  // tile edge

// ---------------------------------------------------------------------------
// WMMA kernel: one warp (one block) per 16x16x16 tile
// ---------------------------------------------------------------------------
template <typename E>
__global__ void wmma_tiles(const E* A, const E* B, const float* C, float* D) {
    using namespace nvcuda;
    const size_t t = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, E, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, E, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::load_matrix_sync(fa, A + t * T * T, T);
    wmma::load_matrix_sync(fb, B + t * T * T, T);
    wmma::load_matrix_sync(acc, C + t * T * T, T, wmma::mem_row_major);
    wmma::mma_sync(acc, fa, fb, acc);
    wmma::store_matrix_sync(D + t * T * T, acc, T, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 kernel. Fragment layout (PTX ISA, "Matrix Fragments for
// mma.m16n8k16 with floating point type"): groupID g = lane/4, q = lane%4.
//   A regs: {A[g][2q..2q+1]}, {A[g+8][2q..]}, {A[g][2q+8..]}, {A[g+8][2q+8..]}
//   B regs (col-major operand): {B[2q..2q+1][g]}, {B[2q+8..2q+9][g]}
//   C/D   : C[g][2q], C[g][2q+1], C[g+8][2q], C[g+8][2q+1]
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t pack2(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

template <bool BF16>
__global__ void mma_tiles(const uint16_t* A, const uint16_t* B, const float* C, float* D) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const size_t t = blockIdx.x;
    const uint16_t* a = A + t * T * T;
    const uint16_t* b = B + t * T * T;
    const float* c = C + t * T * T;
    float* d = D + t * T * T;
    const int lane = threadIdx.x;
    const int g = lane >> 2, q = lane & 3;

    uint32_t ra0 = pack2(a[g * T + 2 * q], a[g * T + 2 * q + 1]);
    uint32_t ra1 = pack2(a[(g + 8) * T + 2 * q], a[(g + 8) * T + 2 * q + 1]);
    uint32_t ra2 = pack2(a[g * T + 2 * q + 8], a[g * T + 2 * q + 9]);
    uint32_t ra3 = pack2(a[(g + 8) * T + 2 * q + 8], a[(g + 8) * T + 2 * q + 9]);

    for (int nh = 0; nh < 2; ++nh) {
        const int n = nh * 8 + g;
        const uint32_t rb0 = pack2(b[(2 * q) * T + n], b[(2 * q + 1) * T + n]);
        const uint32_t rb1 = pack2(b[(2 * q + 8) * T + n], b[(2 * q + 9) * T + n]);
        const int col = nh * 8 + 2 * q;
        float c0 = c[g * T + col], c1 = c[g * T + col + 1];
        float c2 = c[(g + 8) * T + col], c3 = c[(g + 8) * T + col + 1];
        float d0, d1, d2, d3;
        if (BF16) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3));
        } else {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1),
                  "f"(c0), "f"(c1), "f"(c2), "f"(c3));
        }
        d[g * T + col] = d0;
        d[g * T + col + 1] = d1;
        d[(g + 8) * T + col] = d2;
        d[(g + 8) * T + col + 1] = d3;
    }
#endif
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
static void run_tiles(const ProbeSet& ps, const std::string& api, std::vector<uint32_t>& out) {
    const size_t n = ps.count();
    const size_t elems = n * T * T;
    uint16_t *dA, *dB;
    float *dC, *dD;
    CUDA_CHECK(cudaMalloc(&dA, elems * 2));
    CUDA_CHECK(cudaMalloc(&dB, elems * 2));
    CUDA_CHECK(cudaMalloc(&dC, elems * 4));
    CUDA_CHECK(cudaMalloc(&dD, elems * 4));
    CUDA_CHECK(cudaMemcpy(dA, ps.A.data(), elems * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, ps.B.data(), elems * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, ps.C.data(), elems * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0xFF, elems * 4));
    const bool bf = ps.fmt == InFmt::BF16;
    if (api == "wmma") {
        if (bf)
            wmma_tiles<__nv_bfloat16><<<n, 32>>>(reinterpret_cast<__nv_bfloat16*>(dA),
                                                 reinterpret_cast<__nv_bfloat16*>(dB), dC, dD);
        else
            wmma_tiles<__half><<<n, 32>>>(reinterpret_cast<__half*>(dA),
                                          reinterpret_cast<__half*>(dB), dC, dD);
    } else {
        if (bf) mma_tiles<true><<<n, 32>>>(dA, dB, dC, dD);
        else mma_tiles<false><<<n, 32>>>(dA, dB, dC, dD);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    out.resize(elems);
    CUDA_CHECK(cudaMemcpy(out.data(), dD, elems * 4, cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
}

// Random normal tiles + double reference: detects fragment layout errors.
static bool layout_selfcheck(InFmt fmt, const std::string& api) {
    ProbeSet ps;
    ps.fmt = fmt;
    const int n = 64;
    std::mt19937_64 rng(123);
    std::normal_distribution<double> nd(0.0, 1.0);
    ps.A.resize(n * T * T); ps.B.resize(n * T * T); ps.C.resize(n * T * T);
    for (auto& x : ps.A) x = double_to_in(nd(rng), fmt);
    for (auto& x : ps.B) x = double_to_in(nd(rng), fmt);
    for (auto& x : ps.C) x = double_to_fp32(nd(rng));
    std::vector<uint32_t> D;
    run_tiles(ps, api, D);
    double worst = 0;
    for (int t = 0; t < n; ++t)
        for (int i = 0; i < T; ++i)
            for (int j = 0; j < T; ++j) {
                double ref = fp32_to_double(ps.C[t * 256 + i * T + j]), mag = std::fabs(ref);
                for (int k = 0; k < T; ++k) {
                    const double p = in_to_double(ps.A[t * 256 + i * T + k], fmt) *
                                     in_to_double(ps.B[t * 256 + k * T + j], fmt);
                    ref += p;
                    mag += std::fabs(p);
                }
                const double err = std::fabs(fp32_to_double(D[t * 256 + i * T + j]) - ref) / (mag + 1e-30);
                if (err > worst) worst = err;
            }
    const bool ok = worst < 1e-4;  // truncation error is ~1e-6; a layout bug is ~1
    std::printf("layout self-check (%s, %s): worst relative error %.3g -> %s\n", fmt_name(fmt),
                api.c_str(), worst, ok ? "OK" : "FAILED (fragment layout or API problem)");
    return ok;
}

int main(int argc, char** argv) {
    std::string in, out, api = "wmma";
    bool selfcheck_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--in" && i + 1 < argc) in = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--api" && i + 1 < argc) api = argv[++i];
        else if (a == "--selfcheck-only") selfcheck_only = true;
        else {
            std::fprintf(stderr, "usage: tc_probe --in probes.bin --out gpu.out [--api wmma|mma] "
                                 "[--selfcheck-only]\n");
            return 2;
        }
    }
    if (api != "wmma" && api != "mma") { std::fprintf(stderr, "--api must be wmma or mma\n"); return 2; }
    int maj = 0, mnr = 0;
    print_device(maj, mnr);
    if (maj < 8) {
        std::fprintf(stderr, "this project targets Ampere or newer (sm_80+); found sm_%d%d\n", maj, mnr);
        return 1;
    }
    const bool ok16 = layout_selfcheck(InFmt::FP16, api);  // run both checks, then decide
    const bool okbf = layout_selfcheck(InFmt::BF16, api);
    const bool ok = ok16 && okbf;
    if (!ok) return 1;
    if (selfcheck_only) return 0;
    if (in.empty() || out.empty()) { std::fprintf(stderr, "--in and --out are required\n"); return 2; }

    ProbeSet ps;
    std::string err;
    if (!read_probes(in, ps, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (ps.M != T || ps.N != T || ps.K != T) {
        std::fprintf(stderr, "probe tiles must be 16x16x16\n");
        return 1;
    }
    ResultSet rs;
    rs.M = T;
    rs.N = T;
    run_tiles(ps, api, rs.D);
    if (!write_results(out, rs, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::printf("ran %u tiles (%s) through %s -> %s\n", ps.count(), fmt_name(ps.fmt), api.c_str(),
                out.c_str());
    return 0;
}
