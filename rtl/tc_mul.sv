// Significand multiplier. With SHARED=1 one 11x11 array serves FP16 and BF16
// (BF16 significands arrive widened with three trailing zeros). With SHARED=0
// a separate 8x8 array is used for BF16 and the results are muxed; this is the
// baseline for the "area cost of BF16" comparison in docs/05_pd_flow.md.
module tc_mul #(
    parameter int SHARED = 1
) (
    /* verilator lint_off UNUSEDSIGNAL */  // unused when SHARED=1
    input  logic        fmt,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [10:0] a,
    input  logic [10:0] b,
    output logic [21:0] p           // 20 fractional bits
);
    if (SHARED != 0) begin : g_shared
        assign p = a * b;
    end else begin : g_split
        logic [21:0] p16;
        logic [15:0] pb;
        assign p16 = a * b;
        assign pb  = a[10:3] * b[10:3];
        assign p   = fmt ? {pb, 6'b000000} : p16;
    end
endmodule
