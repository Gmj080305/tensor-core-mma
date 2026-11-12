# CUDA programs. Requires the CUDA toolkit (nvcc) and an Ampere-or-newer GPU.
#   make                 # builds build/tc_probe and build/gemm_bench for ARCH
#   make ARCH=sm_89      # e.g. for an RTX 40xx
#
# RTX 3050 laptop GPU (GA107) is sm_86.
NVCC  ?= nvcc
ARCH  ?= sm_86
ROOT  := ..
NVFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -I$(ROOT)/model/include
MODEL_IO := $(ROOT)/model/src/formats.cpp $(ROOT)/model/src/probe_io.cpp

.PHONY: all clean
all: build/tc_probe build/gemm_bench

build:
	mkdir -p build

build/tc_probe: probes/tc_probe.cu common/cuda_check.cuh $(MODEL_IO) | build
	$(NVCC) $(NVFLAGS) probes/tc_probe.cu $(MODEL_IO) -o $@

build/gemm_bench: gemm/gemm_bench.cu common/cuda_check.cuh | build
	$(NVCC) $(NVFLAGS) gemm/gemm_bench.cu -lcublas -o $@

clean:
	rm -rf build
