# Yosys generic synthesis sweep (one FDA lane)

Relative numbers from `synth` + `stat -tech cmos` + `ltp -noff` (Yosys, no PDK).
Depth = longest register-to-register path in generic gate levels.

| configuration | K | F | stages | shared mul | BF16 | cells | est. transistors | flops | depth |
|---|---|---|---|---|---|---|---|---|---|
| ampere lane, 3 stages | 8 | 24 | 3 | 1 | 1 | 14781 | 114842 | 420 | 91 |
| ampere lane, FP16 only (no BF16) | 8 | 24 | 3 | 1 | 0 | 14210 | 109926 | 428 | 91 |
| ampere lane, split BF16 multiplier | 8 | 24 | 3 | 0 | 1 | 17068 | 132360 | 420 | 92 |
| volta lane (K=4), 3 stages | 4 | 24 | 3 | 1 | 1 | 8239 | 64144 | 284 | 79 |
| ampere lane, F=28 window | 8 | 28 | 3 | 1 | 1 | 15601 | 120882 | 431 | 93 |
| ampere lane, 1 stage | 8 | 24 | 1 | 1 | 1 | 14463 | 109992 | 33 | 183 |
| ampere lane, 2 stages | 8 | 24 | 2 | 1 | 1 | 14740 | 113658 | 291 | 97 |
| ampere lane, 4 stages | 8 | 24 | 4 | 1 | 1 | 14859 | 117962 | 651 | 79 |
| ampere lane, 5 stages | 8 | 24 | 5 | 1 | 1 | 15102 | 122582 | 971 | 64 |
| ampere lane, 6 stages | 8 | 24 | 6 | 1 | 1 | 15168 | 123808 | 1045 | 64 |
