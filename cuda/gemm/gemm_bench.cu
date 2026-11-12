// gemm_bench: square GEMM C = A * B (row-major) with five implementations.
//
//   1. naive_sgemm   FP32, one thread per output, operands from global memory
//   2. tiled_sgemm   FP32, 32x32 shared-memory tiles
//   3. wmma_naive    FP16 in / FP32 accumulate, one warp per 16x16 tile, operands
//                    loaded straight from global memory
//   4. wmma_tiled    FP16 in / FP32 accumulate, 64x64 block tile staged in shared
//                    memory, 4 warps x (2x2) fragments
//   5. cuBLAS        cublasSgemm (FP32) and cublasGemmEx (FP16 in, FP32 compute)
//
// Usage: gemm_bench [sizes...]   (default 1024 2048 4096; sizes must be multiples of 64)
//
// All inputs are FP16-representable, so the FP32 and FP16 paths see identical
// values; errors are reported against cublasSgemm.
// Note for GeForce Ampere (RTX 30xx): tensor-core throughput with FP32
// accumulation is half the FP16-accumulate rate, so compare against the
// FP16-in/FP32-accumulate peak of your card, not the headline number.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../common/cuda_check.cuh"

#define CUBLAS_CHECK(call)                                                          \
    do {                                                                            \
        cublasStatus_t s_ = (call);                                                 \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                          \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s_), __FILE__,   \
                         __LINE__);                                                 \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

// ---------------------------------------------------------------- kernels --
__global__ void naive_sgemm(int n, const float* A, const float* B, float* C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float acc = 0.f;
    for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
    C[row * n + col] = acc;
}

constexpr int TS = 32;
__global__ void tiled_sgemm(int n, const float* A, const float* B, float* C) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS + 1];  // +1: avoid bank conflicts on column reads
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TS + ty, col = blockIdx.x * TS + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < n; k0 += TS) {
        As[ty][tx] = A[row * n + k0 + tx];
        Bs[ty][tx] = B[(k0 + ty) * n + col];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TS; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * n + col] = acc;
}

// One warp computes one 16x16 tile; 4 warps per block (2x2 tiles = 32x32).
__global__ void wmma_naive(int n, const __half* A, const __half* B, float* C) {
    using namespace nvcuda;
    const int warp = threadIdx.x / 32;
    const int tile_r = (blockIdx.y * 2 + warp / 2) * 16;
    const int tile_c = (blockIdx.x * 2 + warp % 2) * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k = 0; k < n; k += 16) {
        wmma::load_matrix_sync(fa, A + tile_r * n + k, n);
        wmma::load_matrix_sync(fb, B + k * n + tile_c, n);
        wmma::mma_sync(acc, fa, fb, acc);
    }
    wmma::store_matrix_sync(C + tile_r * n + tile_c, acc, n, wmma::mem_row_major);
}

// 64x64 block tile, K step 32 staged in shared memory, 4 warps each owning a
// 32x32 sub-tile (2x2 fragments). Padding keeps rows 32-byte aligned and
// staggers banks.
constexpr int BT = 64, BK = 32, APAD = 8, BPAD = 8;
__global__ void wmma_tiled(int n, const __half* A, const __half* B, float* C) {
    using namespace nvcuda;
    __shared__ __align__(32) __half As[BT][BK + APAD];
    __shared__ __align__(32) __half Bs[BK][BT + BPAD];
    const int tid = threadIdx.x, warp = tid / 32;
    const int wr = warp / 2, wc = warp % 2;
    const int row0 = blockIdx.y * BT, col0 = blockIdx.x * BT;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    for (int k0 = 0; k0 < n; k0 += BK) {
        // 128 threads load 64x32 halves of A and 32x64 halves of B (16 each).
        for (int e = tid; e < BT * BK; e += blockDim.x) {
            const int r = e / BK, c = e % BK;
            As[r][c] = A[(row0 + r) * n + k0 + c];
        }
        for (int e = tid; e < BK * BT; e += blockDim.x) {
            const int r = e / BT, c = e % BT;
            Bs[r][c] = B[(k0 + r) * n + col0 + c];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(fa[i], &As[wr * 32 + i * 16][kk], BK + APAD);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(fb[j], &Bs[kk][wc * 32 + j * 16], BT + BPAD);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(C + (row0 + wr * 32 + i * 16) * n + col0 + wc * 32 + j * 16,
                                    acc[i][j], n, wmma::mem_row_major);
}

// ------------------------------------------------------------------ host --
struct Timer {
    cudaEvent_t a, b;
    Timer() { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    template <typename F>
    double ms(F f, int iters) {
        f();  // warm-up
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) f();
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float t = 0;
        CUDA_CHECK(cudaEventElapsedTime(&t, a, b));
        CUDA_CHECK(cudaGetLastError());
        return t / iters;
    }
};

static double max_rel_err(const std::vector<float>& x, const std::vector<float>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        num = std::fmax(num, std::fabs(double(x[i]) - ref[i]));
        den = std::fmax(den, std::fabs(double(ref[i])));
    }
    return den > 0 ? num / den : num;
}

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {1024, 2048, 4096};
    int maj, mnr;
    print_device(maj, mnr);
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));

    std::printf("\n%6s  %-14s %10s %10s %12s\n", "n", "kernel", "ms", "TFLOP/s", "max rel err");
    for (int n : sizes) {
        if (n % 64) { std::fprintf(stderr, "size %d is not a multiple of 64, skipped\n", n); continue; }
        const size_t nn = size_t(n) * n;
        std::vector<float> hA(nn), hB(nn);
        std::vector<__half> hA16(nn), hB16(nn);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        for (size_t i = 0; i < nn; ++i) {
            hA16[i] = __float2half(u(rng)); hA[i] = __half2float(hA16[i]);
            hB16[i] = __float2half(u(rng)); hB[i] = __half2float(hB16[i]);
        }
        float *dA, *dB, *dC, *dRef;
        __half *dA16, *dB16;
        CUDA_CHECK(cudaMalloc(&dA, nn * 4)); CUDA_CHECK(cudaMalloc(&dB, nn * 4));
        CUDA_CHECK(cudaMalloc(&dC, nn * 4)); CUDA_CHECK(cudaMalloc(&dRef, nn * 4));
        CUDA_CHECK(cudaMalloc(&dA16, nn * 2)); CUDA_CHECK(cudaMalloc(&dB16, nn * 2));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), nn * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), nn * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dA16, hA16.data(), nn * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB16, hB16.data(), nn * 2, cudaMemcpyHostToDevice));

        const float one = 1.f, zero = 0.f;
        // Row-major C = A*B  <=>  column-major C^T = B^T * A^T.
        auto cublas_f32 = [&](float* out) {
            CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dB, n, dA, n, &zero, out, n));
        };
        auto cublas_f16 = [&](float* out) {
            CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dB16, CUDA_R_16F, n,
                                      dA16, CUDA_R_16F, n, &zero, out, CUDA_R_32F, n,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        };
        cublas_f32(dRef);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> ref(nn), got(nn);
        CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nn * 4, cudaMemcpyDeviceToHost));

        Timer tm;
        const double flops = 2.0 * n * n * n;
        auto report = [&](const char* name, double ms) {
            CUDA_CHECK(cudaMemcpy(got.data(), dC, nn * 4, cudaMemcpyDeviceToHost));
            std::printf("%6d  %-14s %10.3f %10.2f %12.2e\n", n, name, ms, flops / ms / 1e9,
                        max_rel_err(got, ref));
        };
        const int it_slow = n >= 4096 ? 2 : 5, it_fast = 20;

        dim3 b16(16, 16), g16(n / 16, n / 16);
        report("naive_sgemm", tm.ms([&] { naive_sgemm<<<g16, b16>>>(n, dA, dB, dC); }, it_slow));
        dim3 b32(TS, TS), g32(n / TS, n / TS);
        report("tiled_sgemm", tm.ms([&] { tiled_sgemm<<<g32, b32>>>(n, dA, dB, dC); }, it_slow));
        dim3 gw(n / 32, n / 32);
        report("wmma_naive", tm.ms([&] { wmma_naive<<<gw, 128>>>(n, dA16, dB16, dC); }, it_fast));
        dim3 gt(n / BT, n / BT);
        report("wmma_tiled", tm.ms([&] { wmma_tiled<<<gt, 128>>>(n, dA16, dB16, dC); }, it_fast));
        report("cublas_sgemm", tm.ms([&] { cublas_f32(dC); }, it_fast));
        report("cublas_f16f32", tm.ms([&] { cublas_f16(dC); }, it_fast));

        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef); cudaFree(dA16); cudaFree(dB16);
    }
    cublasDestroy(h);
    std::printf("\nProfile one kernel with Nsight Compute, e.g.\n"
                "  ncu --kernel-name wmma_tiled --launch-count 1 --set full ./build/gemm_bench 2048\n");
    return 0;
}
