# References

Numerics of NVIDIA matrix units
* F. A. Khattak, M. Mikaitis. *Generalized Methodology for Determining Numerical Features of Hardware Floating-Point Matrix Multipliers: Part I.* IEEE HPEC 2025. arXiv:2510.15884. Code: https://github.com/faiziktk/IEEE_HPEC2025_block_FMA_tests
* P. Xie, Y. Wang, F. Yang, M. Yang. *MMA-Sim: Bit-Accurate Reference Model of Tensor Cores and Matrix Cores.* arXiv:2511.10909, 2025.
* B. Valpey, X. Li, S. Pai, G. Gopalakrishnan. *An SMT Formalization of Mixed-Precision Matrix Multiplication: Modeling Three Generations of Tensor Cores.* NASA Formal Methods 2025. arXiv:2502.15999.
* M. Fasi, N. J. Higham, M. Mikaitis, S. Pranesh. *Numerical behavior of NVIDIA tensor cores.* PeerJ Computer Science 7:e330, 2021.
* B. Hickmann, D. Bradford. *Experimental analysis of matrix multiplication functional units.* IEEE ARITH 2019.
* P. Blanchard, N. J. Higham, F. Lopez, T. Mary, S. Pranesh. *Mixed precision block fused multiply-add: error analysis and application to GPU tensor cores.* SIAM J. Sci. Comput. 42(3), 2020.
* M. Mikaitis. *Monotonicity of Multi-Term Floating-Point Adders.* IEEE Trans. Computers 73(6), 2024.

Tensor-core architecture and programming
* M. A. Raihan, N. Goli, T. M. Aamodt. *Modeling Deep Learning Accelerator Enabled GPUs.* IEEE ISPASS 2019.
* W. Sun, A. Li, T. Geng, S. Stuijk, H. Corporaal. *Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors.* IEEE TPDS 34(1), 2023.
* NVIDIA. *Programming Tensor Cores in CUDA 9* (developer blog), 2017; *PTX ISA*, section "Matrix Fragments for mma.m16n8k16".

Floating-point dot-product hardware
* A. F. Tenca. *Multi-operand floating-point addition.* IEEE ARITH 2009.
* J. Sohn, E. E. Swartzlander. *Improved architectures for a floating-point fused dot product unit.* IEEE ARITH 2013.
* J. Sohn, E. E. Swartzlander. *A fused floating-point three-term adder.* IEEE TCAS-I 61(10), 2014.
* L. Bertaccini et al. *MiniFloat-NN and ExSdotp.* IEEE ARITH 2022.
* S. Mach et al. *FPnew: An Open-Source Multiformat Floating-Point Unit Architecture.* IEEE TVLSI 29(4), 2021.

Formats and tools
* D. Kalamkar et al. *A Study of BFLOAT16 for Deep Learning Training.* arXiv:1905.12322, 2019.
* P. Micikevicius et al. *FP8 Formats for Deep Learning.* arXiv:2209.05433, 2022.
* M. Fasi, M. Mikaitis. *CPFloat: A C library for simulating low-precision arithmetic.* ACM TOMS 49(2), 2023.

The scoping review that motivated the design choices is in the companion
document "Scoping Review: Tensor-Core MMA Units and Their Numerics".
