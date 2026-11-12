#!/usr/bin/env python3
"""Technology-independent synthesis sweep with Yosys.

For each configuration of the FDA lane this script runs generic synthesis and
reports:
  * cells        generic gate count after ABC mapping
  * transistors  Yosys `stat -tech cmos` estimate (relative area proxy)
  * flops        number of flip-flop bits
  * depth        longest register-to-register path in gate levels (`ltp -noff`),
                 a proxy for the achievable clock period

These are relative numbers for comparing design choices, not PDK results. Use
syn/orfs/ (OpenROAD-flow-scripts) for timing and area in ASAP7 or Sky130.

    python3 syn/yosys/sweep.py [--quick] [--out syn/yosys/results.md]
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RTL_DIR = os.path.join(ROOT, "rtl")

CONFIGS = [
    # (label, K, F, NUM_STAGES, SHARED_MUL, ENABLE_BF16)
    ("ampere lane, 3 stages", 8, 24, 3, 1, 1),
    ("ampere lane, FP16 only (no BF16)", 8, 24, 3, 1, 0),
    ("ampere lane, split BF16 multiplier", 8, 24, 3, 0, 1),
    ("volta lane (K=4), 3 stages", 4, 24, 3, 1, 1),
    ("ampere lane, F=28 window", 8, 28, 3, 1, 1),
    ("ampere lane, 1 stage", 8, 24, 1, 1, 1),
    ("ampere lane, 2 stages", 8, 24, 2, 1, 1),
    ("ampere lane, 4 stages", 8, 24, 4, 1, 1),
    ("ampere lane, 5 stages", 8, 24, 5, 1, 1),
    ("ampere lane, 6 stages", 8, 24, 6, 1, 1),
]


def rtl_files():
    with open(os.path.join(RTL_DIR, "files.f")) as f:
        return [os.path.join(RTL_DIR, l.strip()) for l in f if l.strip()]


def run(k, f, s, shared, bf16):
    script = (
        "read_verilog -sv {files}; "
        "chparam -set K {k} -set F {f} -set NUM_STAGES {s} -set SHARED_MUL {m} -set ENABLE_BF16 {b} tc_fda; "
        "synth -top tc_fda -flatten; "
        "tee -o {stat} stat -tech cmos; "
        "tee -o {ltp} ltp -noff"
    )
    stat_p = "/tmp/tc_sweep_stat.txt"
    ltp_p = "/tmp/tc_sweep_ltp.txt"
    cmd = ["yosys", "-q", "-p", script.format(files=" ".join(rtl_files()), k=k, f=f, s=s, m=shared, b=bf16,
                                              stat=stat_p, ltp=ltp_p)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stat = open(stat_p).read()
    ltp = open(ltp_p).read()
    cells = int(re.search(r"Number of cells:\s+(\d+)", stat).group(1))
    trans = re.search(r"Estimated number of transistors:\s+(\d+)", stat)
    flops = sum(int(n) for n in re.findall(r"\$_\w*DFF\w*\s+(\d+)", stat))
    depth = int(re.search(r"length=(\d+)", ltp).group(1))
    return cells, int(trans.group(1)) if trans else -1, flops, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first three configurations only")
    ap.add_argument("--out", default=os.path.join(ROOT, "syn", "yosys", "results.md"))
    a = ap.parse_args()
    cfgs = CONFIGS[:4] if a.quick else CONFIGS
    rows = []
    for label, k, f, s, m, b in cfgs:
        t0 = time.time()
        cells, trans, flops, depth = run(k, f, s, m, b)
        rows.append((label, k, f, s, m, b, cells, trans, flops, depth))
        print("%-38s cells=%6d transistors=%7d flops=%5d depth=%3d  (%.0fs)"
              % (label, cells, trans, flops, depth, time.time() - t0), flush=True)
    with open(a.out, "w") as o:
        o.write("# Yosys generic synthesis sweep (one FDA lane)\n\n")
        o.write("Relative numbers from `synth` + `stat -tech cmos` + `ltp -noff` (Yosys, no PDK).\n")
        o.write("Depth = longest register-to-register path in generic gate levels.\n\n")
        o.write("| configuration | K | F | stages | shared mul | BF16 | cells | est. transistors | flops | depth |\n")
        o.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            o.write("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d |\n" % r)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
