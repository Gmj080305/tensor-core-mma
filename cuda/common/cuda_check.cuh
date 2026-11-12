#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),  \
                         __FILE__, __LINE__, cudaGetErrorString(err_));                   \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

inline void print_device(int& sm_major, int& sm_minor) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    sm_major = p.major;
    sm_minor = p.minor;
    std::printf("device: %s (sm_%d%d, %d SMs, %.1f GB)\n", p.name, p.major, p.minor,
                p.multiProcessorCount, p.totalGlobalMem / 1073741824.0);
}
