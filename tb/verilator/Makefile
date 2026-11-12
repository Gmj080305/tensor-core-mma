# Verilator regressions for the tensor-core MMA RTL.
#
#   make            # full regression matrix (all configs below)
#   make fda K=8 F=24 STAGES=3 OPS=200000
#   make mma M=4 N=2 K=8 STAGES=3
#   make lint
#   make clean
#
# Each run builds into its own obj_* directory and prints TEST PASSED/FAILED.

VERILATOR ?= verilator
ROOT      := $(abspath ../..)
RTL_DIR   := $(ROOT)/rtl
RTL       := $(addprefix $(RTL_DIR)/,$(shell cat $(RTL_DIR)/files.f))
MODEL_SRC := $(wildcard $(ROOT)/model/src/*.cpp)
JOBS      ?= 2

K      ?= 8
F      ?= 24
STAGES ?= 3
SHARED ?= 1
BF16   ?= 1
M      ?= 4
N      ?= 2
OPS    ?= 200000
CHAINS ?= 5000
SEED   ?= 1

VFLAGS := --cc --exe --build -j $(JOBS) -O3 --x-assign fast --x-initial fast \
          -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM
CFLAGS := -std=c++17 -O2 -I$(ROOT)/model/include -I$(CURDIR)

.PHONY: all fda mma gl lint regress clean

all: regress

fda:
	$(VERILATOR) $(VFLAGS) --top-module tc_fda \
	  -GK=$(K) -GF=$(F) -GNUM_STAGES=$(STAGES) -GSHARED_MUL=$(SHARED) -GENABLE_BF16=$(BF16) \
	  -CFLAGS "$(CFLAGS) -DTB_K=$(K) -DTB_F=$(F) -DTB_STAGES=$(STAGES) $(if $(filter 0,$(BF16)),-DTB_FP16_ONLY)" \
	  -Mdir obj_fda_k$(K)_f$(F)_s$(STAGES)_m$(SHARED)_b$(BF16) -o tb \
	  $(RTL) tb_fda.cpp $(MODEL_SRC)
	./obj_fda_k$(K)_f$(F)_s$(STAGES)_m$(SHARED)_b$(BF16)/tb +ops=$(OPS) +seed=$(SEED)

mma:
	$(VERILATOR) $(VFLAGS) --top-module tc_mma_core \
	  -GM=$(M) -GN=$(N) -GK=$(K) -GF=$(F) -GNUM_STAGES=$(STAGES) -GSHARED_MUL=$(SHARED) \
	  -CFLAGS "$(CFLAGS) -DTB_M=$(M) -DTB_N=$(N) -DTB_K=$(K) -DTB_F=$(F) -DTB_STAGES=$(STAGES)" \
	  -Mdir obj_mma_m$(M)n$(N)k$(K)_s$(STAGES) -o tb \
	  $(RTL) tb_mma.cpp $(MODEL_SRC)
	./obj_mma_m$(M)n$(N)k$(K)_s$(STAGES)/tb +ops=$(OPS) +chains=$(CHAINS) +seed=$(SEED)

# Gate-level simulation: synthesise tc_fda with Yosys (generic cells) and run
# the same model-checked testbench on the netlist. Catches synthesis /
# simulation mismatches (e.g. signed-arithmetic interpretation differences).
YOSYS ?= yosys
gl:
	mkdir -p obj_gl_k$(K)_s$(STAGES)
	$(YOSYS) -q -p "read_verilog -sv $(RTL); chparam -set K $(K) -set F $(F) -set NUM_STAGES $(STAGES) -set SHARED_MUL $(SHARED) tc_fda; synth -top tc_fda -flatten; write_verilog -noattr obj_gl_k$(K)_s$(STAGES)/tc_fda_gl.v"
	$(VERILATOR) --cc --exe --build -j $(JOBS) -O2 -Wno-fatal -Wno-lint -Wno-style --top-module tc_fda \
	  -CFLAGS "$(CFLAGS) -DTB_K=$(K) -DTB_F=$(F) -DTB_STAGES=$(STAGES)" \
	  -Mdir obj_gl_k$(K)_s$(STAGES) -o tb \
	  obj_gl_k$(K)_s$(STAGES)/tc_fda_gl.v tb_fda.cpp $(MODEL_SRC)
	./obj_gl_k$(K)_s$(STAGES)/tb +ops=$(OPS) +seed=$(SEED)

lint:
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME --top-module tc_mma_core $(RTL)

# Regression matrix: every pipeline depth, both lane widths, a wider window,
# the split-multiplier variant, and both MMA shapes.
regress:
	@set -e; \
	for s in 1 2 3 4 5 6; do $(MAKE) --no-print-directory fda K=8 F=24 STAGES=$$s OPS=$(OPS); done; \
	$(MAKE) --no-print-directory fda K=4 F=24 STAGES=3 OPS=$(OPS); \
	$(MAKE) --no-print-directory fda K=8 F=28 STAGES=3 OPS=$(OPS); \
	$(MAKE) --no-print-directory fda K=8 F=24 STAGES=3 SHARED=0 OPS=$(OPS); \
	$(MAKE) --no-print-directory fda K=8 F=24 STAGES=3 BF16=0 OPS=$(OPS); \
	$(MAKE) --no-print-directory mma M=4 N=2 K=8 STAGES=3 OPS=20000; \
	$(MAKE) --no-print-directory mma M=4 N=4 K=4 STAGES=4 OPS=20000; \
	if command -v $(YOSYS) >/dev/null; then $(MAKE) --no-print-directory gl K=8 F=24 STAGES=3 OPS=100000; \
	else echo "yosys not found: gate-level run skipped"; fi; \
	echo "REGRESSION PASSED"

clean:
	rm -rf obj_*
