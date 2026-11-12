# Synthesis and physical design

## 1. Technology-independent sweep (Yosys, measured)

`make synth` runs `syn/yosys/sweep.py`: generic synthesis of one FDA lane per
configuration. The table below is from the run that produced this
repository (Yosys 0.33). "Transistors" is Yosys's `stat -tech cmos` estimate,
a relative area proxy. "Depth" is the longest register-to-register path in
generic gate levels (`ltp -noff`), a proxy for clock period. Neither is a PDK
result.

| Configuration | K | F | Stages | Cells | Est. transistors | Flops | Depth |
|---|---|---|---|---|---|---|---|
| Ampere lane (baseline) | 8 | 24 | 3 | 14,781 | 114,842 | 420 | 91 |
| FP16 only (no BF16) | 8 | 24 | 3 | 14,210 | 109,926 | 428 | 91 |
| Split BF16 multiplier | 8 | 24 | 3 | 17,068 | 132,360 | 420 | 92 |
| Volta lane | 4 | 24 | 3 | 8,239 | 64,144 | 284 | 79 |
| Wider window | 8 | 28 | 3 | 15,601 | 120,882 | 431 | 93 |
| 1 stage | 8 | 24 | 1 | 14,463 | 109,992 | 33 | 183 |
| 2 stages | 8 | 24 | 2 | 14,740 | 113,658 | 291 | 97 |
| 4 stages | 8 | 24 | 4 | 14,859 | 117,962 | 651 | 79 |
| 5 stages | 8 | 24 | 5 | 15,102 | 122,582 | 971 | 64 |
| 6 stages | 8 | 24 | 6 | 15,168 | 123,808 | 1,045 | 64 |

What these numbers say:

* **BF16 is cheap with a shared multiplier.** Adding BF16 costs +4.5% (estimated transistors) over an FP16-only lane. A separate 8x8 BF16 multiplier would cost +20.4%.
* **An 8-product lane costs 1.79x a 4-product lane** for 2x the products. The fixed parts (accumulator path, normaliser, packer) are amortised over more products.
* **Widening the window from 24 to 28 bits costs +5.3%.** This is a measure of how much area NVIDIA's narrow truncating window saves, one of the open questions from the scoping review.
* **Pipelining.** Depth drops from 183 (1 stage) to 64 (5 stages) at +11% transistors, mostly flops. A sixth stage adds 74 flops and no depth: the unpack + multiply stage is now critical.

## 2. OpenROAD-flow-scripts (ASAP7 / Sky130)

### On GitHub Actions (no local install)

`.github/workflows/pnr.yml` runs the full RTL-to-GDS flow inside the official
`openroad/orfs` Docker image, with one runner per configuration:

* stages 3, 4, 5, 6
* FP16-only at 5 stages
* split multiplier at 5 stages

It runs automatically when `rtl/` or `syn/orfs/` changes. To start it by
hand, open Actions -> **pnr** -> Run workflow, then choose the PDK
(`asap7` or `sky130hd`) and optionally a clock period.

* The run summary page shows one combined table: worst slack, area and power per configuration.
* Each job uploads its ORFS reports, logs and the final GDS (`6_final.gds`) as an artifact. Open the GDS in KLayout to see the layout.
* Public repositories get free Linux runners. Expect roughly 15-60 minutes per configuration; the six run in parallel.

If a job fails, download its artifact. `orfs_work/logs/<pdk>/tc_fda/<variant>/`
holds the log of the step that failed (synthesis `1_*`, floorplan `2_*`, and so on).

### Locally

`syn/orfs/` contains design configs and SDC files for `asap7` and `sky130hd`.
The same scripts run locally:
`sweep.sh` generates a fixed-parameter top wrapper per configuration and runs
the full flow (synthesis, floorplan, placement, CTS, routing, finish) as a
separate `FLOW_VARIANT`. `collect.py` then gathers worst slack, area and power.

```bash
git clone --recursive https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts ~/OpenROAD-flow-scripts
# build or use the ORFS docker image, then:
ORFS=~/OpenROAD-flow-scripts bash syn/orfs/sweep.sh asap7 3 4 5 6
TC_CLK_PERIOD=800 ORFS=~/OpenROAD-flow-scripts bash syn/orfs/sweep.sh asap7 5
TC_BF16=0 ORFS=~/OpenROAD-flow-scripts bash syn/orfs/sweep.sh asap7 3     # FP16-only baseline
```

Achievable period = constraint period - worst setup slack, so report
`f_max = 1 / (period - WNS)` and the std-cell area per configuration.

ORFS changes its report formats between releases. If `collect.py` leaves a
column empty, open `reports/<pdk>/tc_fda/<variant>/6_finish.rpt` (under
`WORK_HOME`, or `flow/` locally) and adjust the regex.

Suggested experiments, in order:

1. Stages 3 to 6 at a fixed period: area and slack versus depth (the "pipeline sweep" bullet).
2. BF16 on/off and shared/split multiplier at the best depth: does the +4.5% hold after placement?
3. F = 24 vs 28: the area cost of a more precise window.
4. Multi-corner: ASAP7 ships several corners. Close timing at the slow corner and report all of them.
