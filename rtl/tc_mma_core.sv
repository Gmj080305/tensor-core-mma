// Tensor-core MMA block:  D[MxN] = A[MxK] * B[KxN] + C[MxN]
// built from M*N fused dot-add lanes of K products each (M*N*K multipliers).
//
// Two shapes with 64 multipliers each are used in this project:
//   Ampere-style  M=4 N=2 K=8   (8-product lanes, matches the RTX 3050 hypothesis)
//   Volta-style   M=4 N=4 K=4   (4-product lanes, the classic 4x4x4 tensor core)
// A 16x16x16 tile is computed by issuing K-deep slices and feeding each D back
// as the next C (see tb/verilator/tb_mma.cpp and docs/02_microarchitecture.md).
//
// Operand packing (all row-major, element 0 in the least significant bits):
//   a_i[((m*K)+k)*16 +: 16] = A[m][k]
//   b_i[((k*N)+n)*16 +: 16] = B[k][n]
//   c_i[((m*N)+n)*32 +: 32] = C[m][n]       d_o likewise
module tc_mma_core #(
    parameter int M          = 4,
    parameter int N          = 2,
    parameter int K          = 8,
    parameter int F          = 24,
    parameter int NUM_STAGES = 3,
    parameter int SHARED_MUL = 1
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              in_valid,
    input  logic              in_fmt,          // 0: FP16, 1: BF16
    input  logic [M*K*16-1:0] a_i,
    input  logic [K*N*16-1:0] b_i,
    input  logic [M*N*32-1:0] c_i,
    output logic              out_valid,
    output logic [M*N*32-1:0] d_o
);
    /* verilator lint_off UNUSEDSIGNAL */  // all lanes share one schedule; lane 0 is used
    logic [M*N-1:0] lane_valid;
    /* verilator lint_on UNUSEDSIGNAL */

    for (genvar m = 0; m < M; m++) begin : g_row
        for (genvar n = 0; n < N; n++) begin : g_col
            logic [K*16-1:0] bcol;
            for (genvar k = 0; k < K; k++) begin : g_gather
                assign bcol[k*16 +: 16] = b_i[(k*N + n)*16 +: 16];
            end
            tc_fda #(
                .K(K), .F(F), .NUM_STAGES(NUM_STAGES), .SHARED_MUL(SHARED_MUL)
            ) u_lane (
                .clk       (clk),
                .rst_n     (rst_n),
                .in_valid  (in_valid),
                .in_fmt    (in_fmt),
                .a_i       (a_i[m*K*16 +: K*16]),
                .b_i       (bcol),
                .c_i       (c_i[(m*N + n)*32 +: 32]),
                .out_valid (lane_valid[m*N + n]),
                .d_o       (d_o[(m*N + n)*32 +: 32])
            );
        end
    end

    // All lanes share one pipeline schedule.
    assign out_valid = lane_valid[0];
endmodule
