# Top-level entry points. Run `make help`.
OPS ?= 200000

.PHONY: help model test sim gl lint synth probes selftest check cuda gpu bench clean

help:
	@echo "make model     build the C++ golden model (model/build/tcmma, test_model)"
	@echo "make test      model unit tests"
	@echo "make sim       Verilator regression: all pipeline depths and shapes + gate-level (OPS=$(OPS))"
	@echo "make gl        gate-level simulation of the Yosys netlist only"
	@echo "make lint      Verilator lint of the RTL"
	@echo "make synth     Yosys generic-synthesis sweep -> syn/yosys/results.md"
	@echo "make probes    generate GPU probe tiles -> probes/"
	@echo "make selftest  prove the probes + fit tool recover planted configurations"
	@echo "make check     test + lint + sim + selftest (what CI runs)"
	@echo "make cuda      build CUDA programs (needs nvcc; ARCH=sm_86 default)"
	@echo "make gpu       full GPU characterisation on this machine -> results/"
	@echo "make bench     GEMM benchmark (naive, tiled, WMMA, cuBLAS)"

model:
	cmake -S model -B model/build -DCMAKE_BUILD_TYPE=Release
	cmake --build model/build -j

test: model
	./model/build/test_model

lint:
	$(MAKE) -C tb/verilator lint

sim: model
	$(MAKE) -C tb/verilator regress OPS=$(OPS)

gl: model
	$(MAKE) -C tb/verilator gl OPS=$(OPS)

synth:
	python3 syn/yosys/sweep.py

probes:
	python3 cuda/probes/gen_probes.py --out probes

selftest: model
	./scripts/selftest_fit.sh

check: test lint sim selftest

cuda:
	$(MAKE) -C cuda

gpu:
	./scripts/run_gpu_characterization.sh

bench: cuda
	./cuda/build/gemm_bench

clean:
	rm -rf model/build cuda/build results
	$(MAKE) -C tb/verilator clean
